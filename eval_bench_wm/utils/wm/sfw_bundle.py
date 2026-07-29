"""SFWMark (HSQR / HSTR) watermark-artifact bundle: schema, hashing and IO.

This module is deliberately **only** schema validation, canonical hashing and
save/load. Every SFWMark algorithm — QR construction, the RFFT sign injection,
the inverse RFFT, the center-region FFT and the complex L1 detector — lives in
``utils/wm/hsqr_provider.py`` (HSQR) and must not be duplicated here or in any
runner. The generic canonicalization/hashing primitives are imported from
``utils/wm/artifact_core.py``, which is shared with the GaussMarker bundle.

Official reference for the HSQR artifact semantics:
    https://github.com/thomas11809/SFWMark
    commit 78666128b44614a0cc471993649e3132d5dddfcb
    (``src/generate.py`` / ``src/detect.py`` / ``src/utils.py``)

Official ``generate.py`` persists ``pattern_list-2048.pt`` (the full keybook)
and ``identify_gt_indices_<N>.npy`` (the per-image key mapping). This bundle is
a superset: it additionally persists the *selected* pattern together with the
configuration it is only valid under, so a fresh process can verify one key
without regenerating the 2048-pattern keybook.

Layout::

    <hsqr_bundle>/
    ├── manifest.json          # schema, profile, geometry, key identity, hashes
    ├── selected_pattern.pt    # bool tensor (c_wm, 42, 42) — the selected key
    ├── keybook.pt             # optional bool tensor (2048, c_wm, 42, 42)
    ├── key_mapping.json       # optional per-sample key index mapping
    └── threshold.json         # only after calibration or explicit import

The bundle is intentionally method-tagged (``method``: ``"HSQR"``) so the same
schema/loader can carry the HSTR artifacts of Issue #4 without a second module.
"""

from __future__ import annotations

import json
import math
import typing
from pathlib import Path

import torch

from . import artifact_core
from .artifact_core import (  # noqa: F401 - re-exported so runners need one import
    git_provenance,
    optional_file_sha256,
    read_jsonl,
    sha256_array,
    sha256_bytes,
    sha256_file,
    sha256_tensor,
    sha256_text,
    utc_now,
)


# ---------------------------------------------------------------------------
# Frozen official reference
# ---------------------------------------------------------------------------

OFFICIAL_SFWMARK_REPO = "https://github.com/thomas11809/SFWMark"
OFFICIAL_SFWMARK_COMMIT = "78666128b44614a0cc471993649e3132d5dddfcb"

SFW_BUNDLE_SCHEMA = "sfw_bundle_v1"
SFW_THRESHOLD_SCHEMA = "sfw_threshold_v1"

SUPPORTED_METHODS = ("HSQR", "HSTR")

MANIFEST_FILENAME = "manifest.json"
SELECTED_PATTERN_FILENAME = "selected_pattern.pt"
KEYBOOK_FILENAME = "keybook.pt"
KEY_MAPPING_FILENAME = "key_mapping.json"
THRESHOLD_FILENAME = "threshold.json"

#: Reporting labels. Mirrors the GaussMarker vocabulary (Issue #1 §9) so a
#: cross-method report never has to guess what a label means.
REPORT_LABELS = (
    "official_paper_evaluation",
    "official_profile_raw_scores",
    "calibrated_deployment_verification",
    "deployment_verification_extension",
    "user_supplied_threshold",
    "legacy_threshold",
    "legacy_or_ablation_mode",
)

#: Manifest fields required in *both* the manifest and the current run before an
#: existing bundle may be reused. Covers the watermark identity, the QR profile,
#: the latent geometry and the detector/inversion configuration: a bundle reused
#: under a different one of these produces scores that are not comparable with
#: the ones it was created for.
REQUIRED_BUNDLE_COMPAT_FIELDS = (
    "method",
    "profile_name",
    "model_id",
    "model_revision",
    "scheduler_type",
    "torch_dtype",
    "resolution",
    "latent_shape",
    "center_slice",
    "watermark_channels",
    "qr_version",
    "box_size",
    "border",
    "error_correction",
    "delta",
    "wm_capacity",
    "base_key_seed",
    "selected_key_index",
    "selected_key_seed",
    "payload_text",
    "inversion_prompt_sha256",
    "inversion_guidance_scale",
    "inversion_steps",
    "vae_sample",
    "vae_scaling_factor",
)

HSTR_BUNDLE_COMPAT_FIELDS = (
    "method",
    "profile_name",
    "model_id",
    "model_revision",
    "scheduler_type",
    "torch_dtype",
    "resolution",
    "latent_shape",
    "center_slice",
    "radius",
    "radius_cutoff",
    "watermark_channels",
    "heterogeneous_channels",
    "wm_capacity",
    "base_key_seed",
    "selected_key_index",
    "selected_key_seed",
    "rng_algorithm",
    "rng_device",
    "hermitian_enforcement_version",
)

#: Manifest fields a threshold artifact is bound to. A threshold whose recorded
#: value for any of these differs from the current run is rejected.
THRESHOLD_BINDING_FIELDS = (
    "bundle_config_sha256",
    "method",
    "profile_name",
    "selected_pattern_sha256",
    "selected_key_index",
    "selected_key_seed",
    "payload_text",
    "base_key_seed",
    "model_id",
    "model_revision",
    "scheduler_type",
    "torch_dtype",
    "resolution",
    "latent_shape",
    "center_slice",
    "watermark_channels",
    "qr_version",
    "box_size",
    "border",
    "error_correction",
    "delta",
    "inversion_prompt_sha256",
    "inversion_guidance_scale",
    "inversion_steps",
    "vae_sample",
    "vae_scaling_factor",
)


class SfwBundleError(RuntimeError):
    """Raised whenever an SFWMark bundle artifact fails a closed gate."""


# ---------------------------------------------------------------------------
# Canonical serialization (SFW-flavoured wrappers over artifact_core)
# ---------------------------------------------------------------------------

def canonicalize(value: typing.Any) -> typing.Any:
    return artifact_core.canonicalize(value, SfwBundleError)


def canonical_json(payload: typing.Mapping[str, typing.Any]) -> str:
    return artifact_core.canonical_json(payload, SfwBundleError)


def canonical_sha256(payload: typing.Mapping[str, typing.Any]) -> str:
    return artifact_core.canonical_sha256(payload, SfwBundleError)


def cohort_sha256(image_paths: typing.Sequence[typing.Union[str, Path]]) -> str:
    return artifact_core.cohort_sha256(image_paths, SfwBundleError)


def write_jsonl(path, rows) -> None:
    artifact_core.write_jsonl(path, rows, SfwBundleError)


def append_jsonl(path, row) -> None:
    artifact_core.append_jsonl(path, row, SfwBundleError)


# ---------------------------------------------------------------------------
# Pattern tensors
# ---------------------------------------------------------------------------

def validate_pattern(pattern: torch.Tensor, channels: int) -> None:
    """Validate a selected SFWMark pattern tensor.

    HSQR persists a boolean ``(c_wm, qr, qr)`` QR pattern. HSTR persists the
    selected Fourier target as a complex ``(1, 4, 64, 64)`` tensor, so the shared
    bundle validates by tensor semantics instead of coercing HSTR into HSQR's
    QR schema.
    """
    if not isinstance(pattern, torch.Tensor):
        raise SfwBundleError("SFW pattern must be a torch.Tensor")
    if pattern.dtype == torch.bool:
        if pattern.ndim != 3:
            raise SfwBundleError(
                f"HSQR pattern must have shape (c_wm, qr, qr), got {tuple(pattern.shape)}"
            )
        if pattern.shape[0] != channels:
            raise SfwBundleError(
                f"HSQR pattern has {pattern.shape[0]} channel(s), configuration declares {channels}"
            )
        if pattern.shape[-1] != pattern.shape[-2]:
            raise SfwBundleError(f"HSQR pattern must be square, got {tuple(pattern.shape)}")
        return
    if not torch.is_complex(pattern):
        raise SfwBundleError(
            f"HSTR pattern must keep official complex Fourier semantics, got dtype {pattern.dtype}"
        )
    if pattern.ndim != 4:
        raise SfwBundleError(
            f"HSTR pattern must have shape (batch, channels, h, w), got {tuple(pattern.shape)}"
        )
    if pattern.shape[1] < channels:
        raise SfwBundleError(
            f"HSTR pattern has {pattern.shape[1]} channel(s), configuration declares {channels}"
        )
    if pattern.shape[-1] != pattern.shape[-2]:
        raise SfwBundleError(f"HSTR pattern must be square, got {tuple(pattern.shape)}")


def validate_keybook(keybook: torch.Tensor, capacity: int, channels: int) -> None:
    if not isinstance(keybook, torch.Tensor):
        raise SfwBundleError("HSQR keybook must be a torch.Tensor")
    if keybook.dtype == torch.bool:
        if keybook.ndim != 4 or keybook.shape[0] != capacity:
            raise SfwBundleError(
                f"HSQR keybook must have shape ({capacity}, c_wm, qr, qr), got {tuple(keybook.shape)}"
            )
    elif torch.is_complex(keybook):
        if keybook.ndim != 5 or keybook.shape[0] != capacity:
            raise SfwBundleError(
                f"HSTR keybook must have shape ({capacity}, batch, channels, h, w), got {tuple(keybook.shape)}"
            )
    else:
        raise SfwBundleError(f"SFW keybook has unsupported dtype {keybook.dtype}")
    validate_pattern(keybook[0], channels)


def save_pattern(path: typing.Union[str, Path], pattern: torch.Tensor) -> None:
    torch.save(pattern.detach().cpu().contiguous(), str(path))


def load_pattern(path: typing.Union[str, Path]) -> torch.Tensor:
    pattern = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(pattern, torch.Tensor):
        raise SfwBundleError(f"{path} does not contain a torch.Tensor")
    return pattern


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

class SfwBundle:
    """A validated on-disk SFWMark watermark bundle."""

    def __init__(self, directory: typing.Union[str, Path]):
        self.dir = Path(directory)
        self.manifest_path = self.dir / MANIFEST_FILENAME
        self.pattern_path = self.dir / SELECTED_PATTERN_FILENAME
        self.keybook_path = self.dir / KEYBOOK_FILENAME
        self.key_mapping_path = self.dir / KEY_MAPPING_FILENAME
        self.threshold_path = self.dir / THRESHOLD_FILENAME
        self.manifest: typing.Dict[str, typing.Any] = {}
        self.pattern: typing.Optional[torch.Tensor] = None
        self.keybook: typing.Optional[torch.Tensor] = None
        self.key_mapping: typing.Optional[typing.List[int]] = None
        self.read_only = True

    # -- inspection ---------------------------------------------------------

    def exists(self) -> bool:
        return self.manifest_path.exists() or self.pattern_path.exists()

    def complete(self) -> bool:
        return self.manifest_path.exists() and self.pattern_path.exists()

    # -- creation -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        directory: typing.Union[str, Path],
        pattern: torch.Tensor,
        config: typing.Mapping[str, typing.Any],
        keybook: typing.Optional[torch.Tensor] = None,
        key_mapping: typing.Optional[typing.Sequence[int]] = None,
    ) -> "SfwBundle":
        """Create a new bundle. Never overwrites an existing artifact."""
        bundle = cls(directory)
        if bundle.exists():
            raise SfwBundleError(
                f"refusing to overwrite existing SFW bundle artifacts in {bundle.dir}; "
                "point --hsqr_bundle_dir at a new directory or reuse the existing bundle"
            )

        method = config.get("method")
        if method not in SUPPORTED_METHODS:
            raise SfwBundleError(f"unsupported SFW method {method!r}, expected one of {SUPPORTED_METHODS}")
        channels = len(config["watermark_channels"])
        validate_pattern(pattern, channels)

        bundle.dir.mkdir(parents=True, exist_ok=True)
        save_pattern(bundle.pattern_path, pattern)

        manifest = dict(config)
        manifest.update(
            {
                "schema": SFW_BUNDLE_SCHEMA,
                "created_utc": utc_now(),
                "official_reference_repo": OFFICIAL_SFWMARK_REPO,
                "official_reference_commit": OFFICIAL_SFWMARK_COMMIT,
                "selected_pattern_file": SELECTED_PATTERN_FILENAME,
                "selected_pattern_sha256": sha256_tensor(pattern),
                "selected_pattern_file_sha256": sha256_file(bundle.pattern_path),
                "keybook_file": None,
                "keybook_file_sha256": None,
                "keybook_sha256": None,
                "key_mapping_file": None,
                "key_mapping_file_sha256": None,
                "key_mapping_sha256": None,
            }
        )

        if keybook is not None:
            validate_keybook(keybook, int(config["wm_capacity"]), channels)
            save_pattern(bundle.keybook_path, keybook)
            manifest.update(
                {
                    "keybook_file": KEYBOOK_FILENAME,
                    "keybook_file_sha256": sha256_file(bundle.keybook_path),
                    "keybook_sha256": sha256_tensor(keybook),
                }
            )

        if key_mapping is not None:
            mapping = [int(index) for index in key_mapping]
            capacity = int(config["wm_capacity"])
            if any(not 0 <= index < capacity for index in mapping):
                raise SfwBundleError(f"key mapping contains an index outside [0, {capacity})")
            payload = {"schema": SFW_BUNDLE_SCHEMA, "key_mapping": mapping}
            bundle.key_mapping_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            manifest.update(
                {
                    "key_mapping_file": KEY_MAPPING_FILENAME,
                    "key_mapping_file_sha256": sha256_file(bundle.key_mapping_path),
                    "key_mapping_sha256": canonical_sha256(payload),
                    "key_mapping_count": len(mapping),
                }
            )

        manifest.update(git_provenance())
        manifest["bundle_config_sha256"] = cls.config_sha256(manifest)
        bundle.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

        # Reload from disk so the created bundle passes exactly the same gates a
        # fresh process would apply (canonical round trip, §6 of the skill).
        return cls.load(bundle.dir)

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
    def load(cls, directory: typing.Union[str, Path], read_only: bool = True) -> "SfwBundle":
        bundle = cls(directory)
        missing = [
            path.name for path in (bundle.manifest_path, bundle.pattern_path) if not path.exists()
        ]
        if missing:
            raise SfwBundleError(f"incomplete SFW bundle {bundle.dir}: missing {', '.join(missing)}")

        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SFW_BUNDLE_SCHEMA:
            raise SfwBundleError(
                f"unsupported SFW bundle schema {manifest.get('schema')!r}, expected {SFW_BUNDLE_SCHEMA!r}"
            )
        if manifest.get("method") not in SUPPORTED_METHODS:
            raise SfwBundleError(f"unsupported SFW method {manifest.get('method')!r}")

        # Canonical round trip: the manifest must hash identically after reload.
        recomputed = cls.config_sha256(manifest)
        if recomputed != manifest.get("bundle_config_sha256"):
            raise SfwBundleError(
                "SFW bundle manifest hash mismatch after reload "
                f"({recomputed} != {manifest.get('bundle_config_sha256')}); manifest was edited or corrupted"
            )

        actual_file_sha = sha256_file(bundle.pattern_path)
        if manifest.get("selected_pattern_file_sha256") != actual_file_sha:
            raise SfwBundleError(
                f"{bundle.pattern_path.name} hash mismatch: manifest "
                f"{manifest.get('selected_pattern_file_sha256')}, file {actual_file_sha}"
            )

        pattern = load_pattern(bundle.pattern_path)
        validate_pattern(pattern, len(manifest["watermark_channels"]))
        pattern_sha = sha256_tensor(pattern)
        if manifest.get("selected_pattern_sha256") != pattern_sha:
            raise SfwBundleError(
                f"SFW bundle selected_pattern_sha256 mismatch: manifest "
                f"{manifest.get('selected_pattern_sha256')}, artifact {pattern_sha}"
            )

        keybook = None
        if manifest.get("keybook_file"):
            if not bundle.keybook_path.exists():
                raise SfwBundleError(
                    f"manifest declares {manifest['keybook_file']} but {bundle.keybook_path} is missing"
                )
            if manifest.get("keybook_file_sha256") != sha256_file(bundle.keybook_path):
                raise SfwBundleError(f"{bundle.keybook_path.name} file hash mismatch")
            keybook = load_pattern(bundle.keybook_path)
            validate_keybook(
                keybook, int(manifest["wm_capacity"]), len(manifest["watermark_channels"])
            )
            if manifest.get("keybook_sha256") != sha256_tensor(keybook):
                raise SfwBundleError("SFW bundle keybook_sha256 mismatch")
            selected = int(manifest["selected_key_index"])
            if sha256_tensor(keybook[selected]) != pattern_sha:
                raise SfwBundleError(
                    f"keybook entry {selected} does not equal the persisted selected pattern"
                )
        elif bundle.keybook_path.exists():
            raise SfwBundleError(
                f"{bundle.keybook_path} exists but the manifest does not declare it; "
                "the bundle contains an unaccounted-for artifact"
            )

        key_mapping = None
        if manifest.get("key_mapping_file"):
            if not bundle.key_mapping_path.exists():
                raise SfwBundleError(
                    f"manifest declares {manifest['key_mapping_file']} but the file is missing"
                )
            if manifest.get("key_mapping_file_sha256") != sha256_file(bundle.key_mapping_path):
                raise SfwBundleError(f"{bundle.key_mapping_path.name} file hash mismatch")
            payload = json.loads(bundle.key_mapping_path.read_text(encoding="utf-8"))
            if canonical_sha256(payload) != manifest.get("key_mapping_sha256"):
                raise SfwBundleError("SFW bundle key_mapping_sha256 mismatch")
            key_mapping = [int(index) for index in payload["key_mapping"]]
        elif bundle.key_mapping_path.exists():
            raise SfwBundleError(
                f"{bundle.key_mapping_path} exists but the manifest does not declare it"
            )

        bundle.manifest = manifest
        bundle.pattern = pattern
        bundle.keybook = keybook
        bundle.key_mapping = key_mapping
        bundle.read_only = read_only
        return bundle

    # -- compatibility ------------------------------------------------------

    def assert_compatible(
        self,
        config: typing.Mapping[str, typing.Any],
        required_fields: typing.Sequence[str] = REQUIRED_BUNDLE_COMPAT_FIELDS,
    ) -> None:
        """Fail closed when the current configuration disagrees with the bundle.

        Every field in ``required_fields`` must be present in *both* the manifest
        and ``config`` and must be equal. A required field missing from the
        manifest is a rejection, never a silent continue: an old or hand-edited
        manifest cannot be used to relax the gate.
        """
        missing, mismatched = [], {}

        for field in required_fields:
            if field not in config:
                missing.append(f"{field} (not produced by this run)")
            elif field not in self.manifest:
                missing.append(f"{field} (absent from manifest)")

        for field, value in config.items():
            if field not in self.manifest:
                continue
            if self.manifest[field] != canonicalize(value):
                mismatched[field] = (self.manifest[field], canonicalize(value))

        if missing or mismatched:
            details = []
            if missing:
                details.append("missing required field(s): " + ", ".join(sorted(missing)))
            if mismatched:
                details.append(
                    "mismatched: "
                    + "; ".join(f"{k}: bundle={v[0]!r} run={v[1]!r}" for k, v in sorted(mismatched.items()))
                )
            raise SfwBundleError(
                f"SFW bundle {self.dir} is incompatible with this run ({' | '.join(details)})"
            )

    def public_manifest(self) -> typing.Dict[str, typing.Any]:
        return dict(self.manifest)

    # -- threshold ----------------------------------------------------------

    def has_threshold(self) -> bool:
        return self.threshold_path.exists()

    def load_threshold(self) -> typing.Dict[str, typing.Any]:
        if not self.threshold_path.exists():
            raise SfwBundleError(f"no threshold artifact in {self.dir}")
        artifact = json.loads(self.threshold_path.read_text(encoding="utf-8"))
        validate_threshold_artifact(artifact)
        return artifact

    def save_threshold(self, artifact: typing.Mapping[str, typing.Any], overwrite: bool = False) -> Path:
        validate_threshold_artifact(artifact)
        if self.threshold_path.exists() and not overwrite:
            raise SfwBundleError(
                f"{self.threshold_path} already exists; pass --overwrite_threshold to replace it"
            )
        self.threshold_path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
        return self.threshold_path

    def artifact_mtimes(self) -> typing.Dict[str, typing.Any]:
        """Snapshot used to prove the bundle is untouched by a verification run."""
        snapshot = {}
        for path in sorted(self.dir.iterdir()):
            if path.is_file():
                stat = path.stat()
                snapshot[path.name] = (stat.st_mtime_ns, stat.st_size, sha256_file(path))
        return snapshot


# ---------------------------------------------------------------------------
# Threshold artifacts
# ---------------------------------------------------------------------------

REQUIRED_THRESHOLD_FIELDS = (
    "schema",
    "method",
    "threshold",
    "score_definition",
    "score_direction",
    "comparison_operator",
    "threshold_source",
    "report_label",
)


def validate_threshold_artifact(artifact: typing.Mapping[str, typing.Any]) -> None:
    if artifact.get("schema") != SFW_THRESHOLD_SCHEMA:
        raise SfwBundleError(
            f"unsupported SFW threshold schema {artifact.get('schema')!r}, expected {SFW_THRESHOLD_SCHEMA!r}"
        )
    for field in REQUIRED_THRESHOLD_FIELDS:
        if field not in artifact:
            raise SfwBundleError(f"SFW threshold artifact is missing required field {field!r}")
    if artifact["method"] not in SUPPORTED_METHODS:
        raise SfwBundleError(f"unsupported SFW method {artifact['method']!r} in threshold artifact")
    # The official HSQR canonical score is ``-L1 distance``: higher is more
    # watermarked, so the decision is ``score >= threshold``. A threshold stored
    # with the raw positive distance semantics would silently invert the test.
    if artifact["comparison_operator"] != ">=":
        raise SfwBundleError(
            "official-compatible SFWMark score thresholds compare with '>=', got "
            f"{artifact['comparison_operator']!r}"
        )
    if artifact["score_direction"] != "higher_is_watermarked":
        raise SfwBundleError(f"unsupported SFW score direction {artifact['score_direction']!r}")
    if artifact["report_label"] not in REPORT_LABELS:
        raise SfwBundleError(f"unknown SFW report label {artifact['report_label']!r}")
    threshold = artifact["threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise SfwBundleError("SFW threshold must be numeric")
    if math.isnan(float(threshold)) or math.isinf(float(threshold)):
        raise SfwBundleError("SFW threshold must be finite")


def assert_threshold_compatible(
    artifact: typing.Mapping[str, typing.Any],
    binding: typing.Mapping[str, typing.Any],
) -> None:
    """Reject a threshold artifact produced under a different configuration."""
    validate_threshold_artifact(artifact)
    recorded = artifact.get("binding")
    if not isinstance(recorded, dict):
        raise SfwBundleError("SFW threshold artifact carries no 'binding' block; it is not auditable")
    mismatched = {}
    for field in THRESHOLD_BINDING_FIELDS:
        if field not in binding:
            continue
        want = canonicalize(binding[field])
        if field not in recorded:
            mismatched[field] = (None, want)
        elif recorded[field] != want:
            mismatched[field] = (recorded[field], want)
    if mismatched:
        detail = "; ".join(f"{k}: threshold={v[0]!r} run={v[1]!r}" for k, v in sorted(mismatched.items()))
        raise SfwBundleError(
            f"SFW threshold artifact is incompatible with this configuration ({detail})"
        )


def build_threshold_artifact(
    threshold: float,
    binding: typing.Mapping[str, typing.Any],
    score_definition: str,
    threshold_source: str,
    report_label: str,
    method: str = "HSQR",
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
        "schema": SFW_THRESHOLD_SCHEMA,
        "method": method,
        "created_utc": utc_now(),
        "official_reference_repo": OFFICIAL_SFWMARK_REPO,
        "official_reference_commit": OFFICIAL_SFWMARK_COMMIT,
        "threshold": float(threshold),
        # Equivalent raw-distance operating point, so a reader can never compare a
        # positive distance against a negative score threshold by accident.
        "distance_threshold": -float(threshold),
        "score_definition": score_definition,
        "score_direction": "higher_is_watermarked",
        "comparison_operator": ">=",
        "distance_comparison_operator": "<=",
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
        "binding": {field: binding[field] for field in THRESHOLD_BINDING_FIELDS if field in binding},
    }
    artifact.update(git_provenance())
    if extra:
        artifact.update(dict(extra))
    validate_threshold_artifact(artifact)
    return artifact
