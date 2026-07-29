"""Hash-bound HSTR watermark artifacts.

This module owns only serialization, hashing and compatibility checks. HSTR
mask, pattern, injection and detector math stay in ``hstr_provider.py``.
"""

from __future__ import annotations

import json
import subprocess
import typing
from pathlib import Path

import torch

from utils.canonical import canonical_json_dumps, canonical_json_sha256, sha256_path, tensor_sha256


OFFICIAL_SFWMARK_REPO = "https://github.com/thomas11809/SFWMark"
OFFICIAL_SFWMARK_COMMIT = "78666128b44614a0cc471993649e3132d5dddfcb"
HSTR_BUNDLE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
SELECTED_PATTERN_FILENAME = "selected_pattern.pt"
KEYBOOK_FILENAME = "pattern_list-2048.pt"
THRESHOLD_FILENAME = "threshold.json"


class HstrBundleError(RuntimeError):
    pass


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: typing.Union[str, Path]) -> str:
    return sha256_path(Path(path))


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: typing.Mapping[str, typing.Any]) -> str:
    return canonical_json_dumps(payload)


def canonical_sha256(payload: typing.Mapping[str, typing.Any]) -> str:
    return canonical_json_sha256(payload)


def git_provenance() -> dict[str, typing.Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        "creation_git_branch": run("branch", "--show-current"),
        "creation_git_commit": run("rev-parse", "HEAD"),
        "creation_git_status_short": run("status", "--short"),
    }


def save_tensor(path: typing.Union[str, Path], tensor: torch.Tensor) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().contiguous(), Path(path))


def load_tensor(path: typing.Union[str, Path]) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu")


def write_json(path: typing.Union[str, Path], payload: typing.Mapping[str, typing.Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(canonical_json(payload) + "\n", encoding="utf-8")


def read_json(path: typing.Union[str, Path]) -> dict[str, typing.Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: typing.Union[str, Path], rows: typing.Iterable[typing.Mapping[str, typing.Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def append_jsonl(path: typing.Union[str, Path], row: typing.Mapping[str, typing.Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")


def build_provider_config(
    *,
    profile_name: str,
    model_id: str | None,
    model_revision: str | None,
    scheduler_type: str | None,
    resolution: int,
    latent_shape: typing.Sequence[int],
    center_slice: typing.Sequence[int],
    radius: int,
    radius_cutoff: int,
    watermark_channels: typing.Sequence[int],
    heterogeneous_channels: typing.Sequence[int],
    wm_capacity: int,
    base_key_seed: int,
    selected_key_index: int,
    selected_key_seed: int,
    rng_algorithm: str,
    rng_device: str,
    runtime_dtype: str,
) -> dict[str, typing.Any]:
    return {
        "schema_version": HSTR_BUNDLE_SCHEMA_VERSION,
        "method": "HSTR",
        "official_reference_repo": OFFICIAL_SFWMARK_REPO,
        "official_reference_commit": OFFICIAL_SFWMARK_COMMIT,
        "profile_name": profile_name,
        "model_id": model_id,
        "model_revision": model_revision,
        "scheduler_type": scheduler_type,
        "resolution": int(resolution),
        "latent_shape": [int(x) for x in latent_shape],
        "center_slice": [int(x) for x in center_slice],
        "radius": int(radius),
        "radius_cutoff": int(radius_cutoff),
        "watermark_channels": [int(x) for x in watermark_channels],
        "heterogeneous_channels": [int(x) for x in heterogeneous_channels],
        "wm_capacity": int(wm_capacity),
        "base_key_seed": int(base_key_seed),
        "selected_key_index": int(selected_key_index),
        "selected_key_seed": int(selected_key_seed),
        "rng_algorithm": rng_algorithm,
        "rng_device": rng_device,
        "runtime_dtype": runtime_dtype,
        "hermitian_enforcement_version": "sfwmark_7866612_utils_enforce_hermitian_symmetry",
        "injection_configuration": {
            "center": True,
            "cut_real": False,
            "score_mode": "center_channel_min_complex_l1",
        },
    }


def build_threshold_artifact(
    *,
    threshold: float,
    binding: typing.Mapping[str, typing.Any],
    target_fpr: float,
    empirical_fpr: float,
    empirical_tpr: float,
    roc_auc: float,
    positive_count: int,
    negative_count: int,
    threshold_source: str,
) -> dict[str, typing.Any]:
    artifact = {
        "schema_version": HSTR_BUNDLE_SCHEMA_VERSION,
        "method": "HSTR",
        "score_definition": "hstr_score=-min(channel_0_l1,channel_3_l1)",
        "score_direction": "higher_is_watermarked",
        "threshold": float(threshold),
        "threshold_type": "empirical_clean_1pct_fpr",
        "threshold_source": threshold_source,
        "comparison_operator": ">=",
        "target_fpr": float(target_fpr),
        "empirical_fpr": float(empirical_fpr),
        "empirical_tpr": float(empirical_tpr),
        "roc_auc": float(roc_auc),
        "positive_count": int(positive_count),
        "negative_count": int(negative_count),
        "binding": dict(binding),
        "created_utc": utc_now(),
    }
    validate_threshold_artifact(artifact)
    return artifact


def validate_threshold_artifact(artifact: typing.Mapping[str, typing.Any]) -> None:
    if artifact.get("schema_version") != HSTR_BUNDLE_SCHEMA_VERSION:
        raise HstrBundleError("unsupported HSTR threshold schema")
    if artifact.get("method") != "HSTR":
        raise HstrBundleError("threshold artifact is not for HSTR")
    if artifact.get("comparison_operator") != ">=":
        raise HstrBundleError("HSTR score thresholds must use '>='")
    threshold = artifact.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise HstrBundleError("HSTR threshold must be numeric")
    binding = artifact.get("binding")
    if not isinstance(binding, dict):
        raise HstrBundleError("HSTR threshold artifact carries no auditable binding")


class HstrBundle:
    def __init__(self, directory: typing.Union[str, Path]):
        self.dir = Path(directory)
        self.manifest_path = self.dir / MANIFEST_FILENAME
        self.selected_pattern_path = self.dir / SELECTED_PATTERN_FILENAME
        self.keybook_path = self.dir / KEYBOOK_FILENAME
        self.threshold_path = self.dir / THRESHOLD_FILENAME
        self.manifest: dict[str, typing.Any] | None = None
        self.selected_pattern: torch.Tensor | None = None

    def complete(self) -> bool:
        return self.manifest_path.is_file() and self.selected_pattern_path.is_file()

    def artifact_mtimes(self) -> dict[str, int | None]:
        files = (
            MANIFEST_FILENAME,
            SELECTED_PATTERN_FILENAME,
            KEYBOOK_FILENAME,
            THRESHOLD_FILENAME,
        )
        result = {}
        for filename in files:
            path = self.dir / filename
            result[filename] = path.stat().st_mtime_ns if path.exists() else None
        return result

    @classmethod
    def load(cls, directory: typing.Union[str, Path]) -> "HstrBundle":
        bundle = cls(directory)
        if not bundle.complete():
            raise HstrBundleError(f"incomplete HSTR bundle: {bundle.dir}")
        manifest = read_json(bundle.manifest_path)
        if manifest.get("schema_version") != HSTR_BUNDLE_SCHEMA_VERSION or manifest.get("method") != "HSTR":
            raise HstrBundleError(f"{bundle.manifest_path} is not an HSTR bundle manifest")
        pattern = load_tensor(bundle.selected_pattern_path)
        if tensor_sha256(pattern) != manifest.get("selected_pattern_sha256"):
            raise HstrBundleError("HSTR selected pattern SHA mismatch")
        if sha256_file(bundle.selected_pattern_path) != manifest.get("selected_pattern_file_sha256"):
            raise HstrBundleError("HSTR selected pattern file SHA mismatch")
        if manifest.get("full_keybook_file"):
            if not bundle.keybook_path.is_file():
                raise HstrBundleError("HSTR manifest declares a keybook but the file is missing")
            if sha256_file(bundle.keybook_path) != manifest.get("full_keybook_file_sha256"):
                raise HstrBundleError("HSTR keybook file SHA mismatch")
        bundle.manifest = manifest
        bundle.selected_pattern = pattern
        return bundle

    @classmethod
    def create(
        cls,
        directory: typing.Union[str, Path],
        *,
        provider_config: typing.Mapping[str, typing.Any],
        selected_pattern: torch.Tensor,
        full_keybook: torch.Tensor | None = None,
        overwrite: bool = False,
    ) -> "HstrBundle":
        bundle = cls(directory)
        if bundle.manifest_path.exists() and not overwrite:
            raise HstrBundleError(f"HSTR bundle already exists: {bundle.manifest_path}")
        bundle.dir.mkdir(parents=True, exist_ok=True)
        save_tensor(bundle.selected_pattern_path, selected_pattern)
        manifest = dict(provider_config)
        manifest.update(git_provenance())
        manifest["created_utc"] = utc_now()
        manifest["provider_config_sha256"] = canonical_sha256(provider_config)
        manifest["selected_pattern_file"] = SELECTED_PATTERN_FILENAME
        manifest["selected_pattern_sha256"] = tensor_sha256(selected_pattern)
        manifest["selected_pattern_file_sha256"] = sha256_file(bundle.selected_pattern_path)
        if full_keybook is not None:
            save_tensor(bundle.keybook_path, full_keybook)
            manifest["full_keybook_file"] = KEYBOOK_FILENAME
            manifest["full_keybook_sha256"] = tensor_sha256(full_keybook)
            manifest["full_keybook_file_sha256"] = sha256_file(bundle.keybook_path)
        else:
            manifest["full_keybook_file"] = None
            manifest["full_keybook_sha256"] = None
            manifest["full_keybook_file_sha256"] = None
        write_json(bundle.manifest_path, manifest)
        return cls.load(directory)

    def assert_compatible(self, provider_config: typing.Mapping[str, typing.Any]) -> None:
        if self.manifest is None:
            raise HstrBundleError("bundle is not loaded")
        expected = canonical_sha256(provider_config)
        actual = self.manifest.get("provider_config_sha256")
        if actual != expected:
            raise HstrBundleError(
                f"HSTR bundle is incompatible with this provider config: bundle={actual!r} current={expected!r}"
            )

    def load_threshold(
        self,
        binding: typing.Mapping[str, typing.Any],
        *,
        explicit_threshold: float | None = None,
    ) -> dict[str, typing.Any]:
        if explicit_threshold is not None:
            return {
                "threshold_available": True,
                "threshold": float(explicit_threshold),
                "threshold_source": "user_supplied",
                "score_direction": "higher_is_watermarked",
                "comparison_operator": ">=",
                "target_fpr": None,
                "empirical_fpr": None,
                "empirical_tpr": None,
                "roc_auc": None,
            }
        if not self.threshold_path.is_file():
            return {
                "threshold_available": False,
                "threshold": None,
                "threshold_source": None,
                "score_direction": "higher_is_watermarked",
                "comparison_operator": ">=",
            }
        artifact = read_json(self.threshold_path)
        validate_threshold_artifact(artifact)
        expected = {
            key: binding.get(key)
            for key in (
                "provider_config_sha256",
                "selected_pattern_sha256",
                "selected_key_index",
                "selected_key_seed",
                "profile_name",
                "model_id",
                "model_revision",
                "scheduler_type",
                "latent_shape",
            )
        }
        observed = {key: artifact["binding"].get(key) for key in expected}
        if observed != expected:
            raise HstrBundleError("HSTR threshold artifact is incompatible with this bundle/provider")
        return {
            "threshold_available": True,
            "threshold": float(artifact["threshold"]),
            "threshold_source": artifact.get("threshold_source"),
            "score_direction": artifact.get("score_direction"),
            "comparison_operator": artifact.get("comparison_operator"),
            "target_fpr": artifact.get("target_fpr"),
            "empirical_fpr": artifact.get("empirical_fpr"),
            "empirical_tpr": artifact.get("empirical_tpr"),
            "roc_auc": artifact.get("roc_auc"),
        }

    def save_threshold(self, artifact: typing.Mapping[str, typing.Any], *, overwrite: bool = False) -> Path:
        validate_threshold_artifact(artifact)
        if self.threshold_path.exists() and not overwrite:
            raise HstrBundleError(f"HSTR threshold already exists: {self.threshold_path}")
        write_json(self.threshold_path, artifact)
        return self.threshold_path

    def binding_config(self) -> dict[str, typing.Any]:
        if self.manifest is None:
            raise HstrBundleError("bundle is not loaded")
        keys = (
            "provider_config_sha256",
            "selected_pattern_sha256",
            "selected_pattern_file_sha256",
            "selected_key_index",
            "selected_key_seed",
            "profile_name",
            "model_id",
            "model_revision",
            "scheduler_type",
            "latent_shape",
        )
        return {key: self.manifest.get(key) for key in keys}
