"""Runner-level helpers shared by every watermark verification/generation runner.

Extracted from ``utils/wm/gm_runtime.py`` (GaussMarker, Issue #1) so that the
SFWMark runners (HSQR, Issue #5; HSTR, Issue #4) reuse the very same GPU
preflight, deterministic directory walk, resume gates and ROC bookkeeping
instead of growing a second copy. ``gm_runtime`` now delegates here.

This module contains **no** watermark algorithm.
"""

from __future__ import annotations

import json
import platform
import typing
from pathlib import Path

import numpy as np
import torch


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def gpu_preflight(device: torch.device) -> typing.Dict[str, typing.Any]:
    """Fail closed on the Docker/NVML/CUDA failures described in AGENTS.md."""
    info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "platform": platform.platform(),
    }
    if device.type != "cuda":
        return info
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("GPU preflight failed: CUDA is not available inside this container")
    probe = torch.ones(8, device=device)
    if float((probe * 2).sum().item()) != 16.0:
        raise RuntimeError("GPU preflight failed: basic CUDA kernel execution is wrong")
    info["device_name"] = torch.cuda.get_device_name(device)
    return info


def enumerate_images(path: typing.Union[str, Path]) -> typing.List[Path]:
    """One image or a deterministically sorted directory of images."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"suspect path does not exist: {path}")
    images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images, key=lambda p: p.name)


def assert_unique_inputs(paths: typing.Sequence[Path], role: str = "input") -> None:
    """Reject the same resolved file appearing twice in one cohort."""
    seen: typing.Dict[Path, Path] = {}
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            raise RuntimeError(
                f"duplicate {role} path {resolved}: every image must be scored exactly once"
            )
        seen[resolved] = path


def assert_run_manifest_compatible(
    manifest_path: typing.Union[str, Path],
    run_config_sha256: str,
    method: str = "run",
) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Validate an existing run manifest *before* anything in the run is mutated.

    Returns the existing manifest verbatim when it is compatible (so no field,
    ``created_utc`` included, is rewritten), or ``None`` when no manifest exists
    yet and the caller may create one. An incompatible manifest raises and the
    output directory is left untouched.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = manifest.get("run_config_sha256")
    if existing != run_config_sha256:
        raise RuntimeError(
            f"{method} run manifest {manifest_path} was written by a different configuration "
            f"(existing run_config_sha256 {existing!r}, current {run_config_sha256!r}); "
            "nothing was modified. Use a fresh --out_dir."
        )
    return manifest


def assert_resumable(
    name: str,
    existing: typing.Mapping[str, typing.Any],
    expected: typing.Mapping[str, typing.Any],
    method: str = "sample",
) -> None:
    """Fail closed unless an existing sample was produced by exactly this run.

    File existence alone is never sufficient (experiment-integrity skill §8).
    """
    for field, value in expected.items():
        if existing.get(field) != value:
            raise RuntimeError(
                f"{method} sample {name} cannot be resumed: {field} differs "
                f"(existing {existing.get(field)!r}, current {value!r}); use a fresh --out_dir"
            )


def official_roc(
    positive_scores: typing.Sequence[float],
    negative_scores: typing.Sequence[float],
    target_fpr: float,
    score_definition: str,
    error_cls: type = RuntimeError,
) -> typing.Dict[str, typing.Any]:
    """ROC bookkeeping shared by the official cohort-evaluation protocols.

    ``sklearn.metrics.roc_curve`` on the pooled positive/negative scores, then
    the operating point at the last index whose FPR is strictly below the target
    FPR. Scores are always "higher is watermarked" and the decision operator is
    ``>=``.
    """
    try:
        from sklearn import metrics
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError("cohort evaluation requires scikit-learn") from exc

    if not positive_scores or not negative_scores:
        raise error_cls("ROC evaluation needs a non-empty positive and negative cohort")

    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    preds = list(positive_scores) + list(negative_scores)
    if not np.isfinite(preds).all():
        raise error_cls("ROC evaluation received a non-finite score")

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = float(metrics.auc(fpr, tpr))
    acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
    below = np.where(fpr < target_fpr)[0]
    if below.size == 0:
        raise error_cls(
            f"no ROC operating point with FPR < {target_fpr}; cohort is too small or too noisy"
        )
    index = int(below[-1])
    threshold = float(thresholds[index])
    decisions_neg = [score >= threshold for score in negative_scores]
    decisions_pos = [score >= threshold for score in positive_scores]
    return {
        "roc_auc": auc,
        "best_accuracy": acc,
        "target_fpr": float(target_fpr),
        "threshold": threshold,
        "tpr_at_target_fpr": float(tpr[index]),
        "roc_fpr_at_threshold": float(fpr[index]),
        "empirical_fpr": float(np.mean(decisions_neg)),
        "empirical_tpr": float(np.mean(decisions_pos)),
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
        "comparison_operator": ">=",
        "score_direction": "higher_is_watermarked",
        "score_definition": score_definition,
    }
