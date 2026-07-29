"""Thin IO/orchestration helpers shared by the HSTR runners.

This module deliberately contains no HSTR watermark math. Pattern generation,
injection, inversion and detector scoring live in ``hstr_provider.py``.
"""

from __future__ import annotations

import csv
import gc
import json
import platform
import typing
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import hstr_bundle
from .hstr_bundle import HstrBundleError
from .hstr_provider import (
    HSTR_SCORE_DEFINITION,
    HSTR_SCORE_DIRECTION,
    HSTRProvider,
    OFFICIAL_GUIDANCE_SCALE,
    OFFICIAL_HSTR_PROFILE,
    OFFICIAL_MODEL_ID,
    OFFICIAL_RESOLUTION,
    OFFICIAL_SCHEDULER,
    OFFICIAL_STEPS,
)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
CSV_COLUMNS = [
    "image_index",
    "image_path",
    "image_sha256",
    "cohort_role",
    "status",
    "error",
    "report_label",
    "hstr_channel_0_l1",
    "hstr_channel_3_l1",
    "hstr_channel_min_l1",
    "score",
    "score_definition",
    "score_direction",
    "threshold",
    "threshold_source",
    "comparison_operator",
    "detection_success",
    "selected_key_index",
    "selected_key_seed",
    "selected_pattern_sha256",
    "bundle_sha256",
    "inversion_steps",
    "inversion_prompt_sha256",
    "inversion_guidance_scale",
    "recovered_latent_sha256",
]

PAIR_REQUIRED_FIELDS = ("sample_id", "prompt_sha256", "sample_seed")
PAIR_DISTORTION_FIELDS = ("distortion_config_sha256", "distortion_seed")
RESUME_FIELDS = (
    "sample_seed",
    "prompt_sha256",
    "run_config_sha256",
    "hstr_bundle_config_sha256",
)


def gpu_preflight(device: torch.device) -> dict[str, typing.Any]:
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


def enumerate_images(path: typing.Union[str, Path]) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"image path does not exist: {path}")
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: p.name,
    )


def write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, typing.Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pipe_provider(args, device: torch.device):
    from utils.pipe import pipe_utils

    return pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
        revision=(getattr(args, "model_revision", None) or None),
        torch_dtype=torch.float32,
    )


def build_provider(args, latent_shape, device: torch.device, create_bundle: bool = False) -> HSTRProvider:
    kwargs = dict(vars(args))
    kwargs.pop("latent_shape", None)
    kwargs.pop("device", None)
    kwargs.pop("hstr_create_bundle", None)
    return HSTRProvider(
        latent_shape=latent_shape,
        device=device,
        hstr_create_bundle=create_bundle,
        **kwargs,
    )


def run_provenance(args, provider: HSTRProvider, pipe_provider) -> dict[str, typing.Any]:
    provenance = {
        "official_reference_repo": hstr_bundle.OFFICIAL_SFWMARK_REPO,
        "official_reference_commit": hstr_bundle.OFFICIAL_SFWMARK_COMMIT,
        "hstr_profile": provider.profile,
        "hstr_profile_is_official": bool(provider.profile == OFFICIAL_HSTR_PROFILE),
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None),
        "scheduler": args.scheduler_target,
        "torch_dtype": "float32",
        "resolution": int(args.resolution),
        "num_inference_steps": int(args.num_inference_steps_target),
        "guidance_scale": float(args.guidance_scale_target),
        "created_utc": hstr_bundle.utc_now(),
        "hstr_state_source": provider.state_source,
    }
    provenance.update(hstr_bundle.git_provenance())
    if provider.bundle is not None:
        provenance["hstr_bundle_dir"] = provider.bundle.dir.as_posix()
        provenance["hstr_bundle_config_sha256"] = provider.bundle.manifest.get("provider_config_sha256")
        provenance["hstr_bundle_sha256"] = hstr_bundle.sha256_file(provider.bundle.manifest_path)
        provenance["hstr_selected_pattern_sha256"] = provider.bundle.manifest.get("selected_pattern_sha256")
        provenance["hstr_selected_pattern_file_sha256"] = provider.bundle.manifest.get("selected_pattern_file_sha256")
    return provenance


def assert_run_manifest_compatible(manifest_path: typing.Union[str, Path], run_config_sha256: str):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = manifest.get("run_config_sha256")
    if existing != run_config_sha256:
        raise RuntimeError(
            f"HSTR run manifest {manifest_path} was written by a different configuration "
            f"(existing run_config_sha256 {existing!r}, current {run_config_sha256!r}); use a fresh --out_dir"
        )
    return manifest


def assert_resumable(name: str, existing: typing.Mapping[str, typing.Any], expected: typing.Mapping[str, typing.Any]) -> None:
    for field in RESUME_FIELDS:
        if existing.get(field) != expected.get(field):
            raise RuntimeError(
                f"HSTR sample {name} cannot be resumed: {field} differs "
                f"(existing {existing.get(field)!r}, current {expected.get(field)!r}); use a fresh --out_dir"
            )


def load_pair_metadata(image_path: Path):
    candidates = (
        image_path.parent.parent.parent / "sample_metadata" / f"{image_path.stem}.json",
        image_path.parent.parent / "sample_metadata" / f"{image_path.stem}.json",
        image_path.parent / "sample_metadata" / f"{image_path.stem}.json",
        image_path.parent / f"{image_path.stem}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _declared_value(explicit: typing.Mapping, role: str, field: str):
    for key in (f"{role}_{field}", field):
        if explicit.get(key) is not None:
            return explicit[key]
    return None


def _pair_side(role: str, path: Path, explicit: typing.Mapping, fields: typing.Sequence[str]):
    meta = load_pair_metadata(path) or {}
    resolved = {}
    for field in fields:
        declared = _declared_value(explicit, role, field)
        observed = meta.get(field)
        if declared is not None and observed is not None and declared != observed:
            return None, f"pair manifest and sample metadata for {role} image {path.name} disagree on {field}"
        resolved[field] = declared if declared is not None else observed
    resolved["_has_metadata"] = bool(meta)
    resolved["protocol"] = _declared_value(explicit, role, "protocol") or meta.get("protocol")
    return resolved, None


def _pair_entry(positive: Path, negative: Path, explicit: typing.Optional[typing.Mapping] = None):
    explicit = dict(explicit or {})
    fields = tuple(PAIR_REQUIRED_FIELDS) + tuple(PAIR_DISTORTION_FIELDS)
    sides = {}
    for role, path in (("positive", positive), ("negative", negative)):
        side, reason = _pair_side(role, path, explicit, fields)
        if side is None:
            return None, reason
        sides[role] = side
    pos, neg = sides["positive"], sides["negative"]
    for field in PAIR_REQUIRED_FIELDS:
        if pos[field] is None or neg[field] is None:
            return None, f"pair {positive.name}/{negative.name} lacks {field!r} provenance"
    for field in ("sample_id", "prompt_sha256"):
        if pos[field] != neg[field]:
            return None, f"pair {positive.name}/{negative.name} disagrees on {field}"
    if not explicit and pos["sample_seed"] != neg["sample_seed"]:
        return None, f"pair {positive.name}/{negative.name} disagrees on sample_seed without pair_manifest"
    for field in PAIR_DISTORTION_FIELDS:
        values = (pos[field], neg[field])
        if any(value is not None for value in values) and values[0] != values[1]:
            return None, f"pair {positive.name}/{negative.name} disagrees on {field}"
    return {
        "positive": positive.name,
        "negative": negative.name,
        "sample_id": pos["sample_id"],
        "prompt_sha256": pos["prompt_sha256"],
        "positive_sample_seed": pos["sample_seed"],
        "negative_sample_seed": neg["sample_seed"],
        "distortion_config_sha256": pos["distortion_config_sha256"],
        "distortion_seed": pos["distortion_seed"],
        "protocol": pos["protocol"] or neg["protocol"],
        "pairing_source": "pair_manifest" if explicit else "sample_metadata",
    }, None


def resolve_pairing(positives: typing.Sequence[Path], negatives: typing.Sequence[Path], pair_manifest=None) -> dict[str, typing.Any]:
    result = {"paired": False, "reason": None, "pairs": [], "pairing_sha256": None, "protocol": None, "pair_manifest": None if pair_manifest is None else Path(pair_manifest).as_posix()}
    if len(positives) != len(negatives):
        result["reason"] = f"cohort sizes differ ({len(positives)} positive vs {len(negatives)} negative)"
        return result
    if pair_manifest is not None:
        manifest = json.loads(Path(pair_manifest).read_text(encoding="utf-8"))
        declared = manifest.get("pairs")
        if not isinstance(declared, list) or len(declared) != len(positives):
            result["reason"] = "pair manifest does not declare one entry per cohort image"
            return result
        by_pos, by_neg = {p.name: p for p in positives}, {p.name: p for p in negatives}
        candidates = []
        used_pos, used_neg = set(), set()
        for item in declared:
            pos_name, neg_name = Path(str(item.get("positive", ""))).name, Path(str(item.get("negative", ""))).name
            if pos_name not in by_pos or neg_name not in by_neg or pos_name in used_pos or neg_name in used_neg:
                result["reason"] = "pair manifest references missing or repeated images"
                return result
            used_pos.add(pos_name); used_neg.add(neg_name)
            candidates.append((by_pos[pos_name], by_neg[neg_name], item))
        result["protocol"] = manifest.get("protocol")
    else:
        pos_by_stem, neg_by_stem = {p.stem: p for p in positives}, {p.stem: p for p in negatives}
        if set(pos_by_stem) != set(neg_by_stem):
            result["reason"] = "cohort file stems do not correspond one-to-one; supply --pair_manifest"
            return result
        candidates = [(pos_by_stem[stem], neg_by_stem[stem], None) for stem in sorted(pos_by_stem)]
    pairs = []
    for positive, negative, explicit in candidates:
        entry, reason = _pair_entry(positive, negative, explicit)
        if entry is None:
            result["reason"] = reason
            return result
        pairs.append(entry)
    result.update({"paired": True, "pairs": pairs, "pairing_sha256": hstr_bundle.canonical_sha256({"pairs": pairs}), "protocol": result["protocol"] or (pairs[0].get("protocol") if pairs else None)})
    return result


def score_image(provider: HSTRProvider, pipe_provider, image_path: typing.Union[str, Path], threshold_info, image_index: int | None = None) -> dict[str, typing.Any]:
    image_path = Path(image_path)
    row = {
        "image_index": image_index,
        "image_path": image_path.as_posix(),
        "image_sha256": None,
        "status": "error",
        "error": None,
        "hstr_channel_0_l1": None,
        "hstr_channel_3_l1": None,
        "hstr_channel_min_l1": None,
        "score": None,
        "score_definition": HSTR_SCORE_DEFINITION,
        "score_direction": threshold_info.get("score_direction", HSTR_SCORE_DIRECTION),
        "threshold": threshold_info.get("threshold"),
        "threshold_source": threshold_info.get("threshold_source"),
        "comparison_operator": threshold_info.get("comparison_operator", ">="),
        "detection_success": None,
        "selected_key_index": provider.key_index,
        "selected_key_seed": provider.selected_key_seed,
        "selected_pattern_sha256": provider.selected_pattern_sha256,
        "bundle_sha256": hstr_bundle.sha256_file(provider.bundle.manifest_path) if provider.bundle is not None else None,
        "inversion_steps": int(provider.inversion_steps),
        "inversion_prompt_sha256": hstr_bundle.sha256_text(provider.inversion_prompt),
        "inversion_guidance_scale": float(provider.inversion_guidance),
        "recovered_latent_sha256": None,
    }
    try:
        row["image_sha256"] = hstr_bundle.sha256_file(image_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inversion = provider.invert_pil_image(image, pipe_provider, image_sha256=row["image_sha256"])
        detection = provider.detect_from_latent(inversion["zT_torch"])
        row.update({
            "hstr_channel_0_l1": detection["hstr_channel_0_l1"],
            "hstr_channel_3_l1": detection["hstr_channel_3_l1"],
            "hstr_channel_min_l1": detection["hstr_channel_min_l1"],
            "score": detection["score"],
            "recovered_latent_sha256": inversion["recovered_latent_sha256"],
        })
        row["detection_success"] = provider.decide(row["score"], threshold_info.get("threshold"))
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["detection_success"] = None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def official_roc(positive_scores: typing.Sequence[float], negative_scores: typing.Sequence[float], target_fpr: float) -> dict[str, typing.Any]:
    try:
        from sklearn import metrics
    except ImportError as exc:  # pragma: no cover
        raise ImportError("HSTR cohort evaluation requires scikit-learn") from exc
    if not positive_scores or not negative_scores:
        raise HstrBundleError("HSTR ROC evaluation needs a non-empty positive and negative cohort")
    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    preds = list(positive_scores) + list(negative_scores)
    if not np.isfinite(preds).all():
        raise HstrBundleError("HSTR ROC evaluation received a non-finite score")
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    below = np.where(fpr < target_fpr)[0]
    if below.size == 0:
        raise HstrBundleError(f"no ROC operating point with FPR < {target_fpr}")
    index = int(below[-1])
    threshold = float(thresholds[index])
    return {
        "roc_auc": float(metrics.auc(fpr, tpr)),
        "target_fpr": float(target_fpr),
        "threshold": threshold,
        "tpr_at_target_fpr": float(tpr[index]),
        "roc_fpr_at_threshold": float(fpr[index]),
        "empirical_fpr": float(np.mean([score >= threshold for score in negative_scores])),
        "empirical_tpr": float(np.mean([score >= threshold for score in positive_scores])),
        "positive_count": len(positive_scores),
        "negative_count": len(negative_scores),
        "comparison_operator": ">=",
        "score_direction": HSTR_SCORE_DIRECTION,
        "score_definition": HSTR_SCORE_DEFINITION,
    }


def require_official_generation_profile(args) -> None:
    expected = {
        "hstr_profile": OFFICIAL_HSTR_PROFILE,
        "modelid_target": OFFICIAL_MODEL_ID,
        "scheduler_target": OFFICIAL_SCHEDULER,
        "resolution": OFFICIAL_RESOLUTION,
        "num_inference_steps_target": OFFICIAL_STEPS,
        "guidance_scale_target": OFFICIAL_GUIDANCE_SCALE,
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise HstrBundleError(f"official HSTR generation requires {field}={value!r}, got {getattr(args, field)!r}")
