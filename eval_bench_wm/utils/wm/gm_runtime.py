"""Thin IO/orchestration helpers shared by the GaussMarker runners.

This module deliberately contains **no** GaussMarker algorithm. Every numeric
step is delegated to :class:`utils.wm.gm_provider.GmProvider`, which is the
single source of truth. Only image enumeration, pipeline construction, per-image
error containment, ROC bookkeeping and provenance assembly live here so that
``run_watermark.py`` and ``run_verify_watermark.py`` do not each grow their own
copy.
"""

from __future__ import annotations

import json
import platform
import typing
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import gm_bundle
from .gm_bundle import GmBundleError
from .gm_provider import GM_SCORE_DEFINITION, TORCH_DTYPES, GmProvider


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


def build_pipe_provider(args, device: torch.device):
    from utils.pipe import pipe_utils

    return pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.modelid_target,
        resolution=args.resolution,
        device=device,
        eager_loading=True,
        schedulers_name=args.scheduler_target,
        disable_tqdm=True,
        # An explicitly empty --model_revision means "the repository default".
        revision=(getattr(args, "model_revision", None) or None),
        torch_dtype=TORCH_DTYPES[args.gm_torch_dtype],
    )


def build_provider(args, latent_shape, device: torch.device, create_bundle: bool = False) -> GmProvider:
    kwargs = dict(vars(args))
    kwargs.pop("latent_shape", None)
    kwargs.pop("device", None)
    return GmProvider(
        latent_shape=latent_shape,
        device=device,
        gm_create_bundle=create_bundle,
        gm_allow_in_memory_state=False if getattr(args, "gm_bundle_dir", None) else True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Paired cohort provenance (official paper protocol)
# ---------------------------------------------------------------------------

#: Per-image fields required before a positive/negative pair may be treated as a
#: genuine pair of the official paper protocol.
PAIR_REQUIRED_FIELDS = ("sample_id", "prompt_sha256", "sample_seed")

#: Distortion/attack fields. If either side of a pair declares any of them, both
#: sides must declare all of them and they must be identical.
PAIR_DISTORTION_FIELDS = ("distortion_config_sha256", "distortion_seed")


def load_pair_metadata(image_path: Path) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Locate the per-sample metadata sidecar for a cohort image.

    Looked up as ``<dir>/../sample_metadata/<stem>.json`` (the layout written by
    ``run_watermark.py``) and then ``<dir>/<stem>.json``.
    """
    image_path = Path(image_path)
    candidates = (
        image_path.parent.parent / "sample_metadata" / f"{image_path.stem}.json",
        image_path.parent / "sample_metadata" / f"{image_path.stem}.json",
        image_path.parent / f"{image_path.stem}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def _pair_entry(positive: Path, negative: Path, explicit: typing.Optional[typing.Mapping] = None):
    """Validate one positive/negative pair; returns (entry, reason)."""
    pos_meta = load_pair_metadata(positive)
    neg_meta = load_pair_metadata(negative)
    explicit = dict(explicit or {})

    if pos_meta is None or neg_meta is None:
        side = "positive" if pos_meta is None else "negative"
        return None, f"{side} image {(positive if pos_meta is None else negative).name} has no sample metadata"

    for field in PAIR_REQUIRED_FIELDS:
        for role, meta, path in (("positive", pos_meta, positive), ("negative", neg_meta, negative)):
            if meta.get(field) is None:
                return None, f"{role} metadata for {path.name} is missing {field!r}"

    for field in ("sample_id", "prompt_sha256"):
        if pos_meta[field] != neg_meta[field]:
            return None, (
                f"pair {positive.name}/{negative.name} disagrees on {field} "
                f"({pos_meta[field]!r} vs {neg_meta[field]!r})"
            )

    # "generation seed OR explicit pairing provenance": an explicit pair manifest
    # entry may declare the pairing, otherwise the generation seeds must match.
    if not explicit and pos_meta["sample_seed"] != neg_meta["sample_seed"]:
        return None, (
            f"pair {positive.name}/{negative.name} disagrees on sample_seed "
            f"({pos_meta['sample_seed']!r} vs {neg_meta['sample_seed']!r}) and no explicit "
            "pairing provenance was supplied"
        )

    declared = [f for f in PAIR_DISTORTION_FIELDS if pos_meta.get(f) is not None or neg_meta.get(f) is not None]
    if declared:
        for field in PAIR_DISTORTION_FIELDS:
            if pos_meta.get(field) is None or neg_meta.get(field) is None:
                return None, (
                    f"pair {positive.name}/{negative.name} declares distortion but is missing {field!r} "
                    "on one side"
                )
            if pos_meta[field] != neg_meta[field]:
                return None, (
                    f"pair {positive.name}/{negative.name} was distorted with a different {field} "
                    f"({pos_meta[field]!r} vs {neg_meta[field]!r})"
                )

    entry = {
        "positive": positive.name,
        "negative": negative.name,
        "sample_id": pos_meta["sample_id"],
        "prompt_sha256": pos_meta["prompt_sha256"],
        "positive_sample_seed": pos_meta["sample_seed"],
        "negative_sample_seed": neg_meta["sample_seed"],
        "distortion_config_sha256": pos_meta.get("distortion_config_sha256"),
        "distortion_seed": pos_meta.get("distortion_seed"),
        "protocol": pos_meta.get("protocol") or neg_meta.get("protocol"),
        "pairing_source": "pair_manifest" if explicit else "sample_metadata",
    }
    return entry, None


def resolve_pairing(
    positives: typing.Sequence[Path],
    negatives: typing.Sequence[Path],
    pair_manifest: typing.Optional[typing.Union[str, Path]] = None,
) -> typing.Dict[str, typing.Any]:
    """Establish one-to-one pair identity between the two cohorts.

    Counting the two cohorts is *not* pairing. A cohort pair only qualifies for
    ``official_paper_evaluation`` when every positive is bound to exactly one
    negative through explicit pairing provenance (a pair manifest) or matching
    per-sample metadata (sample id, prompt hash, generation seed, and identical
    distortion parameters/seed when a distortion was applied).
    """
    result = {
        "paired": False,
        "reason": None,
        "pairs": [],
        "pairing_sha256": None,
        "protocol": None,
        "pair_manifest": None if pair_manifest is None else Path(pair_manifest).as_posix(),
    }

    if len(positives) != len(negatives):
        result["reason"] = (
            f"cohort sizes differ ({len(positives)} positive vs {len(negatives)} negative)"
        )
        return result

    if pair_manifest is not None:
        manifest_path = Path(pair_manifest)
        if not manifest_path.is_file():
            result["reason"] = f"pair manifest {manifest_path} does not exist"
            return result
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = manifest.get("pairs")
        if not isinstance(declared, list) or len(declared) != len(positives):
            result["reason"] = "pair manifest does not declare one entry per cohort image"
            return result
        by_name_pos = {p.name: p for p in positives}
        by_name_neg = {p.name: p for p in negatives}
        candidates = []
        for item in declared:
            pos_name = Path(str(item.get("positive", ""))).name
            neg_name = Path(str(item.get("negative", ""))).name
            if pos_name not in by_name_pos or neg_name not in by_name_neg:
                result["reason"] = f"pair manifest references unknown image(s) {pos_name!r}/{neg_name!r}"
                return result
            candidates.append((by_name_pos[pos_name], by_name_neg[neg_name], item))
        result["protocol"] = manifest.get("protocol")
    else:
        pos_by_stem = {p.stem: p for p in positives}
        neg_by_stem = {p.stem: p for p in negatives}
        if set(pos_by_stem) != set(neg_by_stem):
            result["reason"] = (
                "cohort file stems do not correspond one-to-one; supply --pair_manifest to declare "
                "the pairing explicitly"
            )
            return result
        candidates = [(pos_by_stem[stem], neg_by_stem[stem], None) for stem in sorted(pos_by_stem)]

    pairs = []
    for positive, negative, explicit in candidates:
        entry, reason = _pair_entry(positive, negative, explicit)
        if entry is None:
            result["reason"] = reason
            return result
        pairs.append(entry)

    sample_ids = [entry["sample_id"] for entry in pairs]
    if len(set(sample_ids)) != len(sample_ids):
        result["reason"] = "the same sample_id appears in more than one pair"
        return result

    protocols = {entry["protocol"] for entry in pairs if entry["protocol"]}
    if len(protocols) > 1:
        result["reason"] = f"pairs declare more than one protocol: {sorted(protocols)}"
        return result

    result.update(
        {
            "paired": True,
            "pairs": pairs,
            "pairing_sha256": gm_bundle.canonical_sha256({"pairs": pairs}),
            "protocol": result["protocol"] or (sorted(protocols)[0] if protocols else None),
        }
    )
    return result


def cohort_report_label(mode: str, profile_is_official: bool, paired: bool) -> str:
    """Label for a cohort run. Only a verifiably paired official run qualifies."""
    if mode == "paper_eval":
        return "official_paper_evaluation" if (profile_is_official and paired) else "legacy_or_ablation_mode"
    return "calibrated_deployment_verification"


def assert_run_manifest_compatible(
    manifest_path: typing.Union[str, Path],
    run_config_sha256: str,
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
            f"GM run manifest {manifest_path} was written by a different configuration "
            f"(existing run_config_sha256 {existing!r}, current {run_config_sha256!r}); "
            "nothing was modified. Use a fresh --out_dir."
        )
    return manifest


RESUME_FIELDS = (
    "sample_seed",
    "prompt_sha256",
    "run_config_sha256",
    "gm_bundle_config_sha256",
)


def assert_resumable(
    name: str,
    existing: typing.Mapping[str, typing.Any],
    expected: typing.Mapping[str, typing.Any],
) -> None:
    """Fail closed unless an existing sample was produced by exactly this run.

    File existence alone is never sufficient (experiment-integrity skill §8).
    """
    for field in RESUME_FIELDS:
        if field not in expected:
            continue
        if existing.get(field) != expected[field]:
            raise RuntimeError(
                f"GM sample {name} cannot be resumed: {field} differs "
                f"(existing {existing.get(field)!r}, current {expected[field]!r}); use a fresh --out_dir"
            )


def run_provenance(args, provider: GmProvider, pipe_provider) -> typing.Dict[str, typing.Any]:
    provenance = {
        "official_reference_repo": gm_bundle.OFFICIAL_GAUSSMARKER_REPO,
        "official_reference_commit": gm_bundle.OFFICIAL_GAUSSMARKER_COMMIT,
        "gm_profile": provider.profile,
        # Effective officialness: the CLI profile AND the bundle's own provenance.
        "gm_profile_is_official": bool(provider.profile_is_official),
        "gm_cli_profile_is_official": bool(
            provider.profile == "official_sd21" and not provider.profile_overrides
        ),
        "gm_bundle_profile_is_official": provider.bundle_profile_is_official,
        "gm_profile_overrides": dict(provider.profile_overrides),
        "gm_bundle_profile_overrides": dict(provider.bundle_profile_overrides),
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None),
        "scheduler": args.scheduler_target,
        "torch_dtype": args.gm_torch_dtype,
        "resolution": int(args.resolution),
        "num_inference_steps": int(args.num_inference_steps_target),
        "guidance_scale": float(args.guidance_scale_target),
        "created_utc": gm_bundle.utc_now(),
        "gm_state_source": provider.state_source,
    }
    provenance.update(gm_bundle.git_provenance())
    if provider.bundle is not None:
        provenance["gm_bundle_dir"] = provider.bundle.dir.as_posix()
        for field in ("bundle_config_sha256", "w1_file_sha256", "w2_file_sha256",
                      "watermark_sha256", "m_sha256", "w2_tensor_sha256"):
            provenance[f"gm_{field}" if not field.startswith("gm_") else field] = provider.bundle.manifest.get(field)
    return provenance


# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

def score_image(
    provider: GmProvider,
    pipe_provider,
    image_path: typing.Union[str, Path],
    threshold_info: typing.Mapping[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    """Invert and score one suspect image.

    A failure is recorded as ``status="error"`` — never as a negative detection.
    """
    image_path = Path(image_path)
    row: typing.Dict[str, typing.Any] = {
        "image_path": image_path.as_posix(),
        "image_sha256": None,
        "status": "error",
        "error": None,
        "inversion_seed": None,
        "inversion_steps": int(provider.inversion_steps),
        "inversion_prompt_sha256": gm_bundle.sha256_text(provider.inversion_prompt),
        "inversion_guidance_scale": float(provider.inversion_guidance),
        "vae_sample": bool(provider.vae_sample),
        "vae_scaling_factor": float(provider.vae_scaling_factor),
        "recovered_latent_sha256": None,
        "raw_bit_accuracy": None,
        "restored_bit_accuracy": None,
        "raw_ring_l1": None,
        "ring_classifier_feature": None,
        "classifier_probability": None,
        "score": None,
        "score_definition": GM_SCORE_DEFINITION,
        "threshold": threshold_info.get("threshold"),
        "threshold_source": threshold_info.get("threshold_source"),
        "score_direction": threshold_info.get("score_direction"),
        "comparison_operator": threshold_info.get("comparison_operator"),
        "detection_success": None,
    }
    try:
        row["image_sha256"] = gm_bundle.sha256_file(image_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inversion = provider.invert_pil_image(
            image,
            pipe_provider_target=pipe_provider,
            image_sha256=row["image_sha256"],
        )
        detection = provider.detect_from_latent(inversion["zT_torch"])
        row.update(
            {
                "inversion_seed": inversion["inversion_seed"],
                "inversion_steps": inversion["inversion_steps"],
                "recovered_latent_sha256": inversion["recovered_latent_sha256"],
                "raw_bit_accuracy": detection["raw_bit_accuracy"],
                "restored_bit_accuracy": detection["restored_bit_accuracy"],
                "raw_ring_l1": detection["raw_ring_l1"],
                "ring_classifier_feature": detection["ring_classifier_feature"],
                "classifier_probability": detection["classifier_probability"],
                "gnr_used": detection["gnr_used"],
                "classifier_used": detection["classifier_used"],
            }
        )
        score = provider.ensemble_score(detection)
        row["score"] = score
        row["detection_success"] = provider.decide(score, threshold_info.get("threshold"))
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - per-image containment is required
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "error"
        row["detection_success"] = None
    return row


# ---------------------------------------------------------------------------
# ROC / threshold (official Evaluator semantics)
# ---------------------------------------------------------------------------

def official_roc(
    positive_scores: typing.Sequence[float],
    negative_scores: typing.Sequence[float],
    target_fpr: float,
) -> typing.Dict[str, typing.Any]:
    """Official ``Evaluator.eval_ensemble`` ROC bookkeeping.

    Mirrors ``gaussmarker_det.py``: ``sklearn.metrics.roc_curve`` on the pooled
    positive/negative ensemble probabilities, then the operating point at the
    last index whose FPR is strictly below the target FPR.
    """
    try:
        from sklearn import metrics
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise ImportError("GM cohort evaluation requires scikit-learn") from exc

    if not positive_scores or not negative_scores:
        raise GmBundleError("GM ROC evaluation needs a non-empty positive and negative cohort")

    labels = [1] * len(positive_scores) + [0] * len(negative_scores)
    preds = list(positive_scores) + list(negative_scores)
    if not np.isfinite(preds).all():
        raise GmBundleError("GM ROC evaluation received a non-finite score")

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    auc = float(metrics.auc(fpr, tpr))
    acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
    below = np.where(fpr < target_fpr)[0]
    if below.size == 0:
        raise GmBundleError(
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
        "score_definition": GM_SCORE_DEFINITION,
    }
