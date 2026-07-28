"""GaussMarker bundle schema, canonical hashing and artifact validation.

This module is deliberately *only* schema/hashing/IO. Every GaussMarker
algorithm (state creation, latent sampling, ring injection, inversion, bit
recovery, GNR restoration, ring feature, classifier probability, threshold
decision) lives in ``utils/wm/gm_provider.py`` and must not be duplicated here
or in any runner.

Official reference for the on-disk ``w1.pth`` / ``w2.pth`` representation:
    https://github.com/SunnierLee/GaussMarker
    commit 4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155
    (``gaussmarker_gen.py`` / ``gaussmarker_det.py`` / ``watermark.py``)

``w1.pth`` stores exactly the official dict::

    {"w": torch.Tensor, "m": numpy.ndarray, "key": bytes, "nonce": bytes}

``w2.pth`` stores the official complex tensor of shape ``(1, 4, 64, 64)``.
Both files are therefore directly loadable by the official scripts.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import subprocess
import typing
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Frozen official reference
# ---------------------------------------------------------------------------

OFFICIAL_GAUSSMARKER_REPO = "https://github.com/SunnierLee/GaussMarker"
OFFICIAL_GAUSSMARKER_COMMIT = "4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155"

GM_BUNDLE_SCHEMA = "gm_bundle_v1"
GM_THRESHOLD_SCHEMA = "gm_threshold_v1"

W1_FILENAME = "w1.pth"
W2_FILENAME = "w2.pth"
MANIFEST_FILENAME = "manifest.json"
THRESHOLD_FILENAME = "threshold.json"

#: Reporting labels required by Issue #1 (official-code/paper-parity comment §9).
REPORT_LABELS = (
    "official_paper_evaluation",
    "official_profile_raw_scores",
    "calibrated_deployment_verification",
    "deployment_verification_extension",
    "user_supplied_threshold",
    "legacy_or_ablation_mode",
)

#: Manifest fields that must be present and equal before an existing bundle may
#: be reused. Covers the watermark identity configuration *and* the detector
#: configuration (model, inversion, VAE, GNR, classifier), because a bundle
#: reused under a different detector configuration produces results that are not
#: comparable with the ones the bundle was created for.
REQUIRED_BUNDLE_COMPAT_FIELDS = (
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
    "gnr_sha256",
    "classifier_sha256",
    "classifier_type",
    "model_nf",
    "channel_copy",
    "w_copy",
    "h_copy",
    "w_seed",
    "w_channel",
    "w_pattern",
    "w_mask_shape",
    "w_radius",
    "w_measurement",
    "w_injection",
    "latent_shape",
)

#: Manifest fields a threshold artifact is bound to. A threshold whose recorded
#: value for any of these differs from the current run is rejected.
THRESHOLD_BINDING_FIELDS = (
    "bundle_config_sha256",
    "w1_file_sha256",
    "w2_file_sha256",
    "watermark_sha256",
    "m_sha256",
    "w2_tensor_sha256",
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
    "gnr_sha256",
    "classifier_sha256",
    "classifier_type",
    "model_nf",
    "channel_copy",
    "w_copy",
    "h_copy",
    "w_seed",
    "w_channel",
    "w_pattern",
    "w_mask_shape",
    "w_radius",
    "w_measurement",
    "w_injection",
)


class GmBundleError(RuntimeError):
    """Raised whenever a GaussMarker bundle artifact fails a closed gate."""


# ---------------------------------------------------------------------------
# Canonical serialization and hashing
# ---------------------------------------------------------------------------

def _canonicalize(value: typing.Any) -> typing.Any:
    """Normalize a value into a deterministic JSON-serializable form.

    Rejects NaN/Inf, normalizes paths to POSIX strings, sorts mapping keys and
    keeps int/bool distinct from float. The very same function is used before
    writing metadata, after reloading it, during resume validation and during
    threshold-compatibility checks so that
    ``hash before serialization == hash after serialization and reload``.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise GmBundleError(f"non-finite float rejected by canonical hashing: {value!r}")
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _canonicalize(float(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        raise GmBundleError("raw bytes must never enter canonical metadata (possible secret leak)")
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, torch.Size):
        return [int(item) for item in value]
    raise GmBundleError(f"unsupported type in canonical metadata: {type(value)!r}")


def canonical_json(payload: typing.Mapping[str, typing.Any]) -> str:
    """Deterministic JSON text for hashing and on-disk manifests."""
    return json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(payload: typing.Mapping[str, typing.Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: typing.Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash a numpy array together with its dtype and shape."""
    array = np.ascontiguousarray(array)
    header = f"numpy|{array.dtype.str}|{tuple(int(d) for d in array.shape)}|".encode("utf-8")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash a torch tensor together with its dtype and shape (device agnostic)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_complex():
        real = sha256_tensor(tensor.real.contiguous())
        imag = sha256_tensor(tensor.imag.contiguous())
        return hashlib.sha256(f"complex|{real}|{imag}".encode("utf-8")).hexdigest()
    header = f"torch|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def sha256_image_file(path: typing.Union[str, Path]) -> str:
    return sha256_file(path)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_provenance(root: typing.Optional[Path] = None) -> typing.Dict[str, typing.Any]:
    root = Path(root) if root is not None else repo_root()

    def _run(*args: str) -> typing.Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return out.stdout.strip()

    status = _run("status", "--porcelain")
    return {
        "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _run("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def optional_file_sha256(path: typing.Optional[typing.Union[str, Path]]) -> typing.Optional[str]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return sha256_file(path)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def _validate_official_w1(state: typing.Mapping[str, typing.Any]) -> None:
    for key in ("w", "m", "key", "nonce"):
        if key not in state:
            raise GmBundleError(f"w1 is not official-compatible: missing field {key!r}")
    if not isinstance(state["w"], torch.Tensor):
        raise GmBundleError("w1['w'] must be a torch.Tensor (official representation)")
    if not isinstance(state["m"], np.ndarray):
        raise GmBundleError("w1['m'] must be a flat numpy.ndarray (official representation)")
    if state["m"].ndim != 1 or state["m"].size != 4 * 64 * 64:
        raise GmBundleError(
            "w1['m'] must be the flattened 16384-bit encrypted message, got shape "
            f"{tuple(state['m'].shape)}"
        )
    if not isinstance(state["key"], (bytes, bytearray)) or len(state["key"]) != 32:
        raise GmBundleError("w1['key'] must be exactly 32 bytes")
    if not isinstance(state["nonce"], (bytes, bytearray)) or len(state["nonce"]) != 12:
        raise GmBundleError("w1['nonce'] must be exactly 12 bytes")


def _validate_official_w2(patch: torch.Tensor) -> None:
    if not isinstance(patch, torch.Tensor):
        raise GmBundleError("w2 must be a torch.Tensor")
    if not patch.is_complex():
        raise GmBundleError("w2 must be an official-compatible complex tensor")
    if tuple(patch.shape) != (1, 4, 64, 64):
        raise GmBundleError(f"w2 must have shape (1, 4, 64, 64), got {tuple(patch.shape)}")


def save_official_w1(path: typing.Union[str, Path], state: typing.Mapping[str, typing.Any]) -> None:
    """Write ``w1.pth`` in the exact official on-disk representation."""
    payload = {
        "w": state["w"].detach().cpu(),
        "m": np.ascontiguousarray(state["m"]),
        "key": bytes(state["key"]),
        "nonce": bytes(state["nonce"]),
    }
    _validate_official_w1(payload)
    torch.save(payload, str(path))


def load_official_w1(path: typing.Union[str, Path]) -> typing.Dict[str, typing.Any]:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise GmBundleError(f"{path} does not contain an official w1 dict")
    _validate_official_w1(state)
    return {
        "w": state["w"],
        "m": np.ascontiguousarray(state["m"]),
        "key": bytes(state["key"]),
        "nonce": bytes(state["nonce"]),
    }


def save_official_w2(path: typing.Union[str, Path], patch: torch.Tensor) -> None:
    patch = patch.detach().cpu()
    _validate_official_w2(patch)
    torch.save(patch, str(path))


def load_official_w2(path: typing.Union[str, Path]) -> torch.Tensor:
    patch = torch.load(str(path), map_location="cpu", weights_only=False)
    _validate_official_w2(patch)
    return patch


def w1_artifact_hashes(state: typing.Mapping[str, typing.Any]) -> typing.Dict[str, str]:
    """Secret-free hashes of the w1 artifact.

    The raw key and nonce never leave the bundle file; only their SHA256 is
    recorded so a threshold or a result row can be bound to the exact secret
    without disclosing it.
    """
    return {
        "watermark_sha256": sha256_tensor(state["w"]),
        "m_sha256": sha256_array(np.ascontiguousarray(state["m"])),
        "key_sha256": sha256_bytes(bytes(state["key"])),
        "nonce_sha256": sha256_bytes(bytes(state["nonce"])),
    }


class GmBundle:
    """A validated on-disk GaussMarker bundle.

    Layout::

        <gm_bundle>/
        ├── manifest.json
        ├── w1.pth
        ├── w2.pth
        └── threshold.json   # only after calibration or explicit import
    """

    def __init__(self, directory: typing.Union[str, Path]):
        self.dir = Path(directory)
        self.w1_path = self.dir / W1_FILENAME
        self.w2_path = self.dir / W2_FILENAME
        self.manifest_path = self.dir / MANIFEST_FILENAME
        self.threshold_path = self.dir / THRESHOLD_FILENAME
        self.manifest: typing.Dict[str, typing.Any] = {}
        self.w1: typing.Dict[str, typing.Any] = {}
        self.w2: typing.Optional[torch.Tensor] = None
        self.read_only = True

    # -- inspection ---------------------------------------------------------

    def exists(self) -> bool:
        return self.manifest_path.exists() or self.w1_path.exists() or self.w2_path.exists()

    def complete(self) -> bool:
        return self.manifest_path.exists() and self.w1_path.exists() and self.w2_path.exists()

    # -- creation -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        directory: typing.Union[str, Path],
        w1_state: typing.Mapping[str, typing.Any],
        w2_patch: torch.Tensor,
        config: typing.Mapping[str, typing.Any],
    ) -> "GmBundle":
        """Create a new bundle. Never overwrites an existing artifact."""
        bundle = cls(directory)
        if bundle.exists():
            raise GmBundleError(
                f"refusing to overwrite existing GM bundle artifacts in {bundle.dir}; "
                "point --gm_bundle_dir at a new directory or reuse the existing bundle"
            )
        bundle.dir.mkdir(parents=True, exist_ok=True)
        save_official_w1(bundle.w1_path, w1_state)
        save_official_w2(bundle.w2_path, w2_patch)

        manifest = dict(config)
        manifest.update(
            {
                "schema": GM_BUNDLE_SCHEMA,
                "created_utc": utc_now(),
                "official_reference_repo": OFFICIAL_GAUSSMARKER_REPO,
                "official_reference_commit": OFFICIAL_GAUSSMARKER_COMMIT,
                "w1_file_sha256": sha256_file(bundle.w1_path),
                "w2_file_sha256": sha256_file(bundle.w2_path),
                "w2_tensor_sha256": sha256_tensor(w2_patch),
            }
        )
        manifest.update(w1_artifact_hashes(w1_state))
        manifest.update(git_provenance())
        manifest["bundle_config_sha256"] = cls.config_sha256(manifest)
        bundle.manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

        bundle.manifest = manifest
        bundle.w1 = load_official_w1(bundle.w1_path)
        bundle.w2 = load_official_w2(bundle.w2_path)
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
    def load(cls, directory: typing.Union[str, Path], read_only: bool = True) -> "GmBundle":
        bundle = cls(directory)
        missing = [
            path.name
            for path in (bundle.manifest_path, bundle.w1_path, bundle.w2_path)
            if not path.exists()
        ]
        if missing:
            raise GmBundleError(f"incomplete GM bundle {bundle.dir}: missing {', '.join(missing)}")

        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != GM_BUNDLE_SCHEMA:
            raise GmBundleError(
                f"unsupported GM bundle schema {manifest.get('schema')!r}, expected {GM_BUNDLE_SCHEMA!r}"
            )

        # Canonical round-trip: the manifest must hash identically after reload.
        recomputed_config = cls.config_sha256(manifest)
        if recomputed_config != manifest.get("bundle_config_sha256"):
            raise GmBundleError(
                "GM bundle manifest hash mismatch after reload "
                f"({recomputed_config} != {manifest.get('bundle_config_sha256')}); manifest was edited or corrupted"
            )

        for name, path in (("w1_file_sha256", bundle.w1_path), ("w2_file_sha256", bundle.w2_path)):
            expected = manifest.get(name)
            actual = sha256_file(path)
            if expected != actual:
                raise GmBundleError(f"{path.name} hash mismatch: manifest {expected}, file {actual}")

        bundle.manifest = manifest
        bundle.w1 = load_official_w1(bundle.w1_path)
        bundle.w2 = load_official_w2(bundle.w2_path)
        bundle._assert_roundtrip()
        bundle.read_only = read_only
        return bundle

    def _assert_roundtrip(self) -> None:
        hashes = w1_artifact_hashes(self.w1)
        for field, value in hashes.items():
            if self.manifest.get(field) != value:
                raise GmBundleError(
                    f"GM bundle {field} mismatch: manifest {self.manifest.get(field)}, artifact {value}"
                )
        w2_hash = sha256_tensor(self.w2)
        if self.manifest.get("w2_tensor_sha256") != w2_hash:
            raise GmBundleError(
                f"GM bundle w2_tensor_sha256 mismatch: manifest {self.manifest.get('w2_tensor_sha256')}, artifact {w2_hash}"
            )

    # -- compatibility ------------------------------------------------------

    def assert_compatible(
        self,
        config: typing.Mapping[str, typing.Any],
        required_fields: typing.Sequence[str] = (),
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
            if self.manifest[field] != _canonicalize(value):
                mismatched[field] = (self.manifest[field], _canonicalize(value))

        if missing or mismatched:
            details = []
            if missing:
                details.append("missing required field(s): " + ", ".join(sorted(missing)))
            if mismatched:
                details.append(
                    "mismatched: "
                    + "; ".join(f"{k}: bundle={v[0]!r} run={v[1]!r}" for k, v in sorted(mismatched.items()))
                )
            raise GmBundleError(f"GM bundle {self.dir} is incompatible with this run ({' | '.join(details)})")

    def public_manifest(self) -> typing.Dict[str, typing.Any]:
        """Manifest copy safe for logs/CSV (it never contained secrets)."""
        return dict(self.manifest)

    # -- threshold ----------------------------------------------------------

    def has_threshold(self) -> bool:
        return self.threshold_path.exists()

    def load_threshold(self) -> typing.Dict[str, typing.Any]:
        if not self.threshold_path.exists():
            raise GmBundleError(f"no threshold artifact in {self.dir}")
        artifact = json.loads(self.threshold_path.read_text(encoding="utf-8"))
        validate_threshold_artifact(artifact)
        return artifact

    def save_threshold(self, artifact: typing.Mapping[str, typing.Any], overwrite: bool = False) -> Path:
        validate_threshold_artifact(artifact)
        if self.threshold_path.exists() and not overwrite:
            raise GmBundleError(
                f"{self.threshold_path} already exists; pass --overwrite_threshold to replace it"
            )
        self.threshold_path.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
        return self.threshold_path

    def artifact_mtimes(self) -> typing.Dict[str, float]:
        """Snapshot used by tests/verification to prove the bundle is untouched."""
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
    "threshold",
    "score_definition",
    "score_direction",
    "comparison_operator",
    "threshold_source",
    "report_label",
)


def validate_threshold_artifact(artifact: typing.Mapping[str, typing.Any]) -> None:
    if artifact.get("schema") != GM_THRESHOLD_SCHEMA:
        raise GmBundleError(
            f"unsupported GM threshold schema {artifact.get('schema')!r}, expected {GM_THRESHOLD_SCHEMA!r}"
        )
    for field in REQUIRED_THRESHOLD_FIELDS:
        if field not in artifact:
            raise GmBundleError(f"GM threshold artifact is missing required field {field!r}")
    if artifact["comparison_operator"] != ">=":
        raise GmBundleError(
            "official-compatible GaussMarker ROC thresholds compare with '>=', got "
            f"{artifact['comparison_operator']!r}"
        )
    if artifact["score_direction"] != "higher_is_watermarked":
        raise GmBundleError(
            f"unsupported GM score direction {artifact['score_direction']!r}"
        )
    if artifact["report_label"] not in REPORT_LABELS:
        raise GmBundleError(f"unknown GM report label {artifact['report_label']!r}")
    threshold = artifact["threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise GmBundleError("GM threshold must be numeric")
    if math.isnan(float(threshold)) or math.isinf(float(threshold)):
        raise GmBundleError("GM threshold must be finite")


def assert_threshold_compatible(
    artifact: typing.Mapping[str, typing.Any],
    binding: typing.Mapping[str, typing.Any],
) -> None:
    """Reject a threshold artifact produced under a different configuration."""
    validate_threshold_artifact(artifact)
    recorded = artifact.get("binding")
    if not isinstance(recorded, dict):
        raise GmBundleError("GM threshold artifact carries no 'binding' block; it is not auditable")
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
        raise GmBundleError(
            f"GM threshold artifact is incompatible with this configuration ({detail})"
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
        "schema": GM_THRESHOLD_SCHEMA,
        "created_utc": utc_now(),
        "official_reference_repo": OFFICIAL_GAUSSMARKER_REPO,
        "official_reference_commit": OFFICIAL_GAUSSMARKER_COMMIT,
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
        "binding": {field: binding[field] for field in THRESHOLD_BINDING_FIELDS if field in binding},
    }
    artifact.update(git_provenance())
    if extra:
        artifact.update(dict(extra))
    validate_threshold_artifact(artifact)
    return artifact


def cohort_sha256(image_paths: typing.Sequence[typing.Union[str, Path]]) -> str:
    """Order-independent-per-name hash of a cohort of image files."""
    entries = []
    for path in image_paths:
        path = Path(path)
        entries.append({"name": path.name, "sha256": sha256_file(path)})
    entries.sort(key=lambda item: (item["name"], item["sha256"]))
    return canonical_sha256({"cohort": entries})


def write_jsonl(path: typing.Union[str, Path], rows: typing.Iterable[typing.Mapping[str, typing.Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def append_jsonl(path: typing.Union[str, Path], row: typing.Mapping[str, typing.Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: typing.Union[str, Path]) -> typing.List[typing.Dict[str, typing.Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
