"""RingID key/keybook artifact schema, persistence and compatibility gates.

This module is deliberately *only* schema/persistence/validation. Every RingID
algorithm (ring masks, keybook construction, lossless imprinting, spatial
shift, injection, per-channel distances, identification, inversion) lives in
``utils/wm/ringid_provider.py`` and must not be duplicated here or in a runner.

Official reference for the algorithm the artifact describes:
    https://github.com/showlab/RingID
    commit 45631a59aecd7d63ccdb640aaaf3e616fdb89fb9
    (``utils.py``, ``verify.py``, ``identify.py``, ``inverse_stable_diffusion.py``)

Canonical hashing/JSON/git-provenance helpers are *reused* from
``gm_bundle`` (Issue #1) rather than reimplemented, so that a RAVEN artifact
hash means exactly the same thing for every method. Only RingID-specific
schema, key persistence and compatibility rules are defined here.

Layout::

    <rid_bundle>/
    ├── manifest.json
    ├── selected_pattern.pt   # official complex pattern, shape (1, 4, 64, 64)
    ├── watermark_mask.pt     # official bool mask, shape (2, 64, 64)
    └── threshold.json        # only after calibration or explicit import
"""

from __future__ import annotations

import json
import math
import typing
from pathlib import Path

import torch

# Reused canonical implementation (experiment-integrity skill §3/§6: one
# authoritative hashing/serialization path shared by every RAVEN artifact).
from .gm_bundle import (  # noqa: F401  (re-exported on purpose)
    _canonicalize,
    canonical_json,
    canonical_sha256,
    cohort_sha256,
    git_provenance,
    read_jsonl,
    append_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_tensor,
    sha256_text,
    utc_now,
    write_jsonl,
)


OFFICIAL_RINGID_REPO = "https://github.com/showlab/RingID"
OFFICIAL_RINGID_COMMIT = "45631a59aecd7d63ccdb640aaaf3e616fdb89fb9"

RID_BUNDLE_SCHEMA = "rid_bundle_v1"
RID_THRESHOLD_SCHEMA = "rid_threshold_v1"

PATTERN_FILENAME = "selected_pattern.pt"
MASK_FILENAME = "watermark_mask.pt"
MANIFEST_FILENAME = "manifest.json"
THRESHOLD_FILENAME = "threshold.json"

#: Reporting labels. ``official_code_exact`` parity and the paper-described
#: scaled shift are never conflated, and a cohort ROC result is never labelled
#: the same as a fixed-threshold deployment decision.
REPORT_LABELS = (
    "official_paper_verification",
    "official_paper_identification",
    "official_profile_raw_scores",
    "calibrated_deployment_verification",
    "deployment_verification_extension",
    "user_supplied_threshold",
    "paper_described_shift_ablation",
    "legacy_or_ablation_mode",
)

#: Manifest fields that fully determine *which tensor* a key id refers to.
#: Any difference means the same key index denotes a different watermark, so
#: reusing the bundle would silently change the watermark identity.
KEY_IDENTITY_FIELDS = (
    "latent_shape",
    "radius",
    "radius_cutoff",
    "ring_width",
    "rounder_ring",
    "anchor_x_offset",
    "anchor_y_offset",
    "heterogeneous_channels",
    "ring_channels",
    "quantization_levels",
    "ring_value_range",
    "quantization_values",
    "assigned_keys",
    "candidate_count",
    "candidate_order_sha256",
    "fix_gt",
    "spatial_shift",
    "spatial_shift_factor",
    "spatial_shift_factor_semantics",
    "rng_algorithm",
    "rng_seed",
    "rng_device",
    "rng_dtype",
)

#: Additionally required before a bundle may be *reused for detection*: a key
#: scored through a different model/inversion configuration is not comparable
#: with the cohort the bundle was created for.
DETECTOR_COMPAT_FIELDS = (
    "model_id",
    "model_revision",
    "torch_dtype",
    "scheduler",
    "resolution",
    "inversion_prompt_sha256",
    "inversion_guidance_scale",
    "inversion_steps",
    "vae_sample",
    "vae_scaling_factor",
    "channel_min",
    "score_definition",
)

REQUIRED_BUNDLE_COMPAT_FIELDS = KEY_IDENTITY_FIELDS + DETECTOR_COMPAT_FIELDS

#: What a threshold artifact is bound to.
THRESHOLD_BINDING_FIELDS = REQUIRED_BUNDLE_COMPAT_FIELDS + (
    "bundle_config_sha256",
    "selected_key_index",
    "selected_pattern_sha256",
    "mask_sha256",
    "profile_name",
)

REQUIRED_THRESHOLD_FIELDS = (
    "schema",
    "threshold",
    "score_definition",
    "score_direction",
    "comparison_operator",
    "threshold_source",
    "report_label",
)


class RidBundleError(RuntimeError):
    """Raised whenever a RingID artifact fails a closed gate."""


def _validate_pattern(pattern: torch.Tensor) -> None:
    if not isinstance(pattern, torch.Tensor):
        raise RidBundleError("RingID pattern must be a torch.Tensor")
    if not pattern.is_complex():
        raise RidBundleError("RingID pattern must be an official-compatible complex tensor")
    if len(pattern.shape) != 4 or pattern.shape[0] != 1 or pattern.shape[-1] != pattern.shape[-2]:
        raise RidBundleError(
            f"RingID pattern must have shape (1, C, S, S), got {tuple(pattern.shape)}"
        )


def _validate_mask(mask: torch.Tensor) -> None:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise RidBundleError("RingID watermark mask must be a bool torch.Tensor")
    if len(mask.shape) != 3:
        raise RidBundleError(f"RingID mask must have shape (C, S, S), got {tuple(mask.shape)}")


def save_pattern(path: typing.Union[str, Path], pattern: torch.Tensor) -> None:
    pattern = pattern.detach().cpu()
    _validate_pattern(pattern)
    torch.save(pattern, str(path))


def load_pattern(path: typing.Union[str, Path]) -> torch.Tensor:
    pattern = torch.load(str(path), map_location="cpu", weights_only=False)
    _validate_pattern(pattern)
    return pattern


def save_mask(path: typing.Union[str, Path], mask: torch.Tensor) -> None:
    mask = mask.detach().cpu()
    _validate_mask(mask)
    torch.save(mask, str(path))


def load_mask(path: typing.Union[str, Path]) -> torch.Tensor:
    mask = torch.load(str(path), map_location="cpu", weights_only=False)
    _validate_mask(mask)
    return mask


class RidBundle:
    """A validated on-disk RingID key artifact.

    The bundle stores the *selected* key tensor and the mask verbatim, plus the
    complete, auditable recipe (RNG algorithm/seed/device/dtype, candidate
    ordering, quantization profile, shift semantics) needed to regenerate the
    full candidate keybook for multi-key identification. Regeneration is always
    checked against ``keybook_sha256`` / ``candidate_order_sha256`` before it is
    used, so a key id means the same tensor across processes and machines.
    """

    def __init__(self, directory: typing.Union[str, Path]):
        self.dir = Path(directory)
        self.pattern_path = self.dir / PATTERN_FILENAME
        self.mask_path = self.dir / MASK_FILENAME
        self.manifest_path = self.dir / MANIFEST_FILENAME
        self.threshold_path = self.dir / THRESHOLD_FILENAME
        self.manifest: typing.Dict[str, typing.Any] = {}
        self.pattern: typing.Optional[torch.Tensor] = None
        self.mask: typing.Optional[torch.Tensor] = None
        self.read_only = True

    # -- inspection ---------------------------------------------------------

    def exists(self) -> bool:
        return any(p.exists() for p in (self.manifest_path, self.pattern_path, self.mask_path))

    def complete(self) -> bool:
        return all(p.exists() for p in (self.manifest_path, self.pattern_path, self.mask_path))

    # -- creation -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        directory: typing.Union[str, Path],
        pattern: torch.Tensor,
        mask: torch.Tensor,
        config: typing.Mapping[str, typing.Any],
    ) -> "RidBundle":
        """Create a new bundle. Never overwrites an existing artifact."""
        bundle = cls(directory)
        if bundle.exists():
            raise RidBundleError(
                f"refusing to overwrite existing RingID bundle artifacts in {bundle.dir}; "
                "point --rid_bundle_dir at a new directory or reuse the existing bundle"
            )
        bundle.dir.mkdir(parents=True, exist_ok=True)
        save_pattern(bundle.pattern_path, pattern)
        save_mask(bundle.mask_path, mask)

        manifest = dict(config)
        manifest.update(
            {
                "schema": RID_BUNDLE_SCHEMA,
                "method": "RID",
                "created_utc": utc_now(),
                "official_reference_repo": OFFICIAL_RINGID_REPO,
                "official_reference_commit": OFFICIAL_RINGID_COMMIT,
                "pattern_file_sha256": sha256_file(bundle.pattern_path),
                "mask_file_sha256": sha256_file(bundle.mask_path),
                "selected_pattern_sha256": sha256_tensor(pattern),
                "mask_sha256": sha256_tensor(mask),
            }
        )
        manifest.update(git_provenance())
        manifest["bundle_config_sha256"] = cls.config_sha256(manifest)
        bundle.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

        bundle.manifest = manifest
        bundle.pattern = load_pattern(bundle.pattern_path)
        bundle.mask = load_mask(bundle.mask_path)
        bundle._assert_roundtrip()
        return bundle

    @staticmethod
    def config_sha256(manifest: typing.Mapping[str, typing.Any]) -> str:
        """Hash of the manifest excluding volatile/derived fields."""
        volatile = {
            "bundle_config_sha256",
            "created_utc",
            "git_branch",
            "git_commit",
            "git_dirty",
        }
        return canonical_sha256({k: v for k, v in manifest.items() if k not in volatile})

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, directory: typing.Union[str, Path], read_only: bool = True) -> "RidBundle":
        bundle = cls(directory)
        missing = [
            path.name
            for path in (bundle.manifest_path, bundle.pattern_path, bundle.mask_path)
            if not path.exists()
        ]
        if missing:
            raise RidBundleError(
                f"incomplete RingID bundle {bundle.dir}: missing {', '.join(missing)}"
            )

        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != RID_BUNDLE_SCHEMA:
            raise RidBundleError(
                f"unsupported RingID bundle schema {manifest.get('schema')!r}, "
                f"expected {RID_BUNDLE_SCHEMA!r}"
            )
        if manifest.get("official_reference_commit") != OFFICIAL_RINGID_COMMIT:
            raise RidBundleError(
                "RingID bundle was created against official commit "
                f"{manifest.get('official_reference_commit')!r}, this build is frozen to "
                f"{OFFICIAL_RINGID_COMMIT!r}"
            )

        recomputed = cls.config_sha256(manifest)
        if recomputed != manifest.get("bundle_config_sha256"):
            raise RidBundleError(
                "RingID bundle manifest hash mismatch after reload "
                f"({recomputed} != {manifest.get('bundle_config_sha256')}); manifest was edited "
                "or corrupted"
            )

        for name, path in (
            ("pattern_file_sha256", bundle.pattern_path),
            ("mask_file_sha256", bundle.mask_path),
        ):
            expected, actual = manifest.get(name), sha256_file(path)
            if expected != actual:
                raise RidBundleError(
                    f"{path.name} hash mismatch: manifest {expected}, file {actual}"
                )

        bundle.manifest = manifest
        bundle.pattern = load_pattern(bundle.pattern_path)
        bundle.mask = load_mask(bundle.mask_path)
        bundle._assert_roundtrip()
        bundle.read_only = read_only
        return bundle

    def _assert_roundtrip(self) -> None:
        for field, tensor in (
            ("selected_pattern_sha256", self.pattern),
            ("mask_sha256", self.mask),
        ):
            actual = sha256_tensor(tensor)
            if self.manifest.get(field) != actual:
                raise RidBundleError(
                    f"RingID bundle {field} mismatch: manifest {self.manifest.get(field)}, "
                    f"artifact {actual}"
                )

    # -- compatibility ------------------------------------------------------

    def assert_compatible(
        self,
        config: typing.Mapping[str, typing.Any],
        required_fields: typing.Sequence[str] = (),
    ) -> None:
        """Fail closed when the current configuration disagrees with the bundle."""
        missing, mismatched = [], {}

        for field in required_fields:
            if field not in config:
                missing.append(f"{field} (not produced by this run)")
            elif field not in self.manifest:
                missing.append(f"{field} (absent from manifest)")

        for field, value in config.items():
            if field not in self.manifest:
                continue
            if self.manifest[field] != _canonicalize(value):
                mismatched[field] = (self.manifest[field], _canonicalize(value))

        if missing or mismatched:
            details = []
            if missing:
                details.append("missing required field(s): " + ", ".join(sorted(missing)))
            if mismatched:
                details.append(
                    "mismatched: "
                    + "; ".join(
                        f"{k}: bundle={v[0]!r} run={v[1]!r}" for k, v in sorted(mismatched.items())
                    )
                )
            raise RidBundleError(
                f"RingID bundle {self.dir} is incompatible with this run ({' | '.join(details)})"
            )

    def public_manifest(self) -> typing.Dict[str, typing.Any]:
        return dict(self.manifest)

    # -- threshold ----------------------------------------------------------

    def has_threshold(self) -> bool:
        return self.threshold_path.exists()

    def load_threshold(self) -> typing.Dict[str, typing.Any]:
        if not self.threshold_path.exists():
            raise RidBundleError(f"no threshold artifact in {self.dir}")
        artifact = json.loads(self.threshold_path.read_text(encoding="utf-8"))
        validate_threshold_artifact(artifact)
        return artifact

    def save_threshold(
        self, artifact: typing.Mapping[str, typing.Any], overwrite: bool = False
    ) -> Path:
        validate_threshold_artifact(artifact)
        if self.threshold_path.exists() and not overwrite:
            raise RidBundleError(
                f"{self.threshold_path} already exists; pass --overwrite_threshold to replace it"
            )
        self.threshold_path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
        return self.threshold_path

    def artifact_mtimes(self) -> typing.Dict[str, typing.Any]:
        """Snapshot used to prove the bundle was not touched by a read-only run."""
        snapshot = {}
        for path in sorted(self.dir.iterdir()):
            if path.is_file():
                stat = path.stat()
                snapshot[path.name] = (stat.st_mtime_ns, stat.st_size, sha256_file(path))
        return snapshot


# ---------------------------------------------------------------------------
# Threshold artifacts
# ---------------------------------------------------------------------------

def validate_threshold_artifact(artifact: typing.Mapping[str, typing.Any]) -> None:
    if artifact.get("schema") != RID_THRESHOLD_SCHEMA:
        raise RidBundleError(
            f"unsupported RingID threshold schema {artifact.get('schema')!r}, "
            f"expected {RID_THRESHOLD_SCHEMA!r}"
        )
    for field in REQUIRED_THRESHOLD_FIELDS:
        if field not in artifact:
            raise RidBundleError(f"RingID threshold artifact is missing required field {field!r}")
    if artifact["score_direction"] != "higher_is_watermarked":
        raise RidBundleError(
            f"unsupported RingID score direction {artifact['score_direction']!r}; the canonical "
            "score is -channel_min_l1, which is higher_is_watermarked"
        )
    if artifact["comparison_operator"] != ">=":
        raise RidBundleError(
            "RingID ROC thresholds compare with '>=' on the canonical score, got "
            f"{artifact['comparison_operator']!r}"
        )
    if artifact["report_label"] not in REPORT_LABELS:
        raise RidBundleError(f"unknown RingID report label {artifact['report_label']!r}")
    threshold = artifact["threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise RidBundleError("RingID threshold must be numeric")
    if math.isnan(float(threshold)) or math.isinf(float(threshold)):
        raise RidBundleError("RingID threshold must be finite")


def assert_threshold_compatible(
    artifact: typing.Mapping[str, typing.Any],
    binding: typing.Mapping[str, typing.Any],
) -> None:
    """Reject a threshold artifact produced under a different configuration."""
    validate_threshold_artifact(artifact)
    recorded = artifact.get("binding")
    if not isinstance(recorded, dict):
        raise RidBundleError(
            "RingID threshold artifact carries no 'binding' block; it is not auditable"
        )
    mismatched = {}
    for field in THRESHOLD_BINDING_FIELDS:
        if field not in binding:
            continue
        want = _canonicalize(binding[field])
        if field not in recorded:
            mismatched[field] = (None, want)
        elif recorded[field] != want:
            mismatched[field] = (recorded[field], want)
    if mismatched:
        detail = "; ".join(
            f"{k}: threshold={v[0]!r} run={v[1]!r}" for k, v in sorted(mismatched.items())
        )
        raise RidBundleError(
            f"RingID threshold artifact is incompatible with this configuration ({detail})"
        )


def build_threshold_artifact(
    threshold: float,
    binding: typing.Mapping[str, typing.Any],
    score_definition: str,
    threshold_source: str,
    report_label: str,
    target_fpr: typing.Optional[float] = None,
    empirical_fpr: typing.Optional[float] = None,
    tpr_at_target_fpr: typing.Optional[float] = None,
    roc_auc: typing.Optional[float] = None,
    positive_count: typing.Optional[int] = None,
    negative_count: typing.Optional[int] = None,
    positive_cohort_sha256: typing.Optional[str] = None,
    negative_cohort_sha256: typing.Optional[str] = None,
    extra: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> typing.Dict[str, typing.Any]:
    artifact = {
        "schema": RID_THRESHOLD_SCHEMA,
        "created_utc": utc_now(),
        "official_reference_repo": OFFICIAL_RINGID_REPO,
        "official_reference_commit": OFFICIAL_RINGID_COMMIT,
        "threshold": float(threshold),
        "score_definition": score_definition,
        "score_direction": "higher_is_watermarked",
        "comparison_operator": ">=",
        "threshold_source": threshold_source,
        "report_label": report_label,
        "target_fpr": target_fpr,
        "empirical_fpr": empirical_fpr,
        "tpr_at_target_fpr": tpr_at_target_fpr,
        "roc_auc": roc_auc,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_cohort_sha256": positive_cohort_sha256,
        "negative_cohort_sha256": negative_cohort_sha256,
        "binding": {f: binding[f] for f in THRESHOLD_BINDING_FIELDS if f in binding},
    }
    artifact.update(git_provenance())
    if extra:
        artifact.update(dict(extra))
    validate_threshold_artifact(artifact)
    return artifact
