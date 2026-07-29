"""Thin IO/orchestration helpers shared by the SFWMark (HSQR) runners.

This module deliberately contains **no** SFWMark algorithm. Every numeric step
is delegated to :class:`utils.wm.hsqr_provider.HSQRProvider`, which is the single
source of truth; the inversion front-end lives in ``utils/wm/sfw_inversion.py``
and the artifact schema in ``utils/wm/sfw_bundle.py``. Generic runner plumbing
(GPU preflight, deterministic directory walk, resume gates, ROC bookkeeping) is
imported from ``utils/wm/runner_common.py``, which is shared with GaussMarker.

Only pipeline construction, per-image error containment and provenance assembly
live here so that ``run_watermark.py`` and ``run_verify_watermark.py`` do not
each grow their own copy.
"""

from __future__ import annotations

import csv
import gc
import json
import typing
from pathlib import Path

import torch
from PIL import Image

from . import runner_common, sfw_bundle, sfw_inversion
from .hsqr_provider import (
    HSQR_COMPARISON_OPERATOR,
    HSQR_SCORE_DEFINITION,
    HSQR_SCORE_DIRECTION,
    TORCH_DTYPES,
    HSQRProvider,
)
from .hstr_provider import (
    HSTR_SCORE_DEFINITION,
    HSTR_SCORE_DIRECTION,
    HSTRProvider,
    OFFICIAL_GUIDANCE_SCALE as HSTR_OFFICIAL_GUIDANCE_SCALE,
    OFFICIAL_HSTR_PROFILE,
    OFFICIAL_MODEL_ID as HSTR_OFFICIAL_MODEL_ID,
    OFFICIAL_RESOLUTION as HSTR_OFFICIAL_RESOLUTION,
    OFFICIAL_SCHEDULER as HSTR_OFFICIAL_SCHEDULER,
    OFFICIAL_STEPS as HSTR_OFFICIAL_STEPS,
)
from .runner_common import (  # noqa: F401 - re-exported so runners need one import
    assert_resumable,
    assert_run_manifest_compatible,
    assert_unique_inputs,
    enumerate_images,
    gpu_preflight,
)
from .sfw_bundle import SfwBundle, SfwBundleError


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
        torch_dtype=(
            torch.float32 if getattr(args, "wm_type", None) == "HSTR"
            else TORCH_DTYPES[args.hsqr_torch_dtype]
        ),
    )


def assert_pipeline_matches_bundle(args, bundle: SfwBundle) -> None:
    """Reject a run that would load a different pipeline than the bundle declares.

    The provider is rebuilt from the manifest, so the *pipeline* is the one thing
    the manifest cannot enforce by itself. This runs before any model is loaded,
    so a mistake costs nothing.
    """
    pipeline_fields = {
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None) or None,
        "scheduler_type": args.scheduler_target,
        "resolution": int(args.resolution),
        "torch_dtype": args.hsqr_torch_dtype,
    }
    mismatched = {
        field: (bundle.manifest.get(field), value)
        for field, value in pipeline_fields.items()
        if bundle.manifest.get(field) != value
    }
    if mismatched:
        detail = "; ".join(f"{k}: bundle={v[0]!r} run={v[1]!r}" for k, v in sorted(mismatched.items()))
        raise SfwBundleError(
            f"the pipeline this run would load does not match the HSQR bundle ({detail})"
        )


def build_provider(args, latent_shape, device: torch.device, create_bundle: bool = False):
    """Construct the authoritative SFWMark provider from CLI arguments."""
    kwargs = dict(vars(args))
    kwargs.pop("latent_shape", None)
    kwargs.pop("device", None)
    if getattr(args, "wm_type", None) == "HSTR":
        kwargs.pop("hstr_create_bundle", None)
        return HSTRProvider(
            latent_shape=latent_shape,
            device=device,
            hstr_create_bundle=create_bundle,
            **kwargs,
        )
    return HSQRProvider(latent_shape=latent_shape, device=device, **kwargs)


def load_provider_from_bundle(args, latent_shape, device: torch.device) -> HSQRProvider:
    """Rebuild a provider from a persisted bundle (verification side).

    Verification must never create watermark state: the persisted pattern is
    used verbatim and every configuration field is compared against the manifest
    before a single image is scored.
    """
    bundle = SfwBundle.load(Path(args.hsqr_bundle_dir))
    assert_pipeline_matches_bundle(args, bundle)
    provider = HSQRProvider.from_bundle(
        bundle,
        latent_shape=latent_shape,
        device=device,
        hsqr_inversion_prompt=args.hsqr_inversion_prompt,
        hsqr_target_fpr=args.hsqr_target_fpr,
        hsqr_threshold=args.hsqr_threshold,
        hsqr_allow_legacy_threshold=args.hsqr_allow_legacy_threshold,
    )
    if provider.pattern_source != "bundle":
        raise SfwBundleError(
            f"verification must use the persisted pattern, got source {provider.pattern_source!r}"
        )
    # The bundle is authoritative, so an explicitly requested key identity that
    # disagrees with it must fail rather than be silently ignored.
    if int(args.hsqr_key_index) != provider.selected_key_index:
        raise SfwBundleError(
            f"--hsqr_key_index {args.hsqr_key_index} disagrees with the bundle's key index "
            f"{provider.selected_key_index}; the bundle defines the watermark identity"
        )
    return provider


def run_provenance(args, provider, pipe_provider) -> typing.Dict[str, typing.Any]:
    method = provider.get_wm_type() if hasattr(provider, "get_wm_type") else "HSQR"
    provenance = {
        "method": method,
        "official_reference_repo": sfw_bundle.OFFICIAL_SFWMARK_REPO,
        "official_reference_commit": sfw_bundle.OFFICIAL_SFWMARK_COMMIT,
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None),
        "scheduler": args.scheduler_target,
        "torch_dtype": "float32" if method == "HSTR" else args.hsqr_torch_dtype,
        "resolution": int(args.resolution),
        "num_inference_steps": int(args.num_inference_steps_target),
        "guidance_scale": float(args.guidance_scale_target),
        "inversion_impl_version": sfw_inversion.SFW_INVERSION_IMPL_VERSION,
        "inversion_parity_status": sfw_inversion.SFW_INVERSION_PARITY_STATUS,
        "inversion_weights_parity": sfw_inversion.SFW_INVERSION_WEIGHTS_PARITY,
        "created_utc": sfw_bundle.utc_now(),
    }
    if method == "HSTR":
        provenance.update({
            "hstr_profile": provider.profile,
            "hstr_profile_is_official": bool(provider.profile == OFFICIAL_HSTR_PROFILE),
            "hstr_state_source": provider.state_source,
            "selected_key_index": provider.key_index,
            "selected_key_seed": provider.selected_key_seed,
            "selected_pattern_sha256": provider.selected_pattern_sha256,
        })
    else:
        provenance.update({
            "hsqr_profile": provider.profile,
            "hsqr_profile_is_official": bool(provider.profile_is_official),
            "hsqr_profile_overrides": dict(provider.profile_overrides),
        })
        provenance.update(provider.key_identity())
    provenance.update(sfw_bundle.git_provenance())
    if provider.bundle is not None:
        key = method.lower()
        provenance[f"{key}_bundle_dir"] = provider.bundle.dir.as_posix()
        provenance[f"{key}_bundle_config_sha256"] = provider.bundle.manifest.get("bundle_config_sha256")
        provenance[f"{key}_bundle_sha256"] = sfw_bundle.sha256_file(provider.bundle.manifest_path)
    return provenance


# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

def score_image(
    provider: HSQRProvider,
    pipe_provider,
    image_path: typing.Union[str, Path],
    threshold_info: typing.Mapping[str, typing.Any],
) -> typing.Dict[str, typing.Any]:
    """Invert and score one suspect image.

    A failure is recorded as ``status="error"`` — never as a negative detection.
    Both the raw positive distance and the canonical negative-distance score are
    always emitted.
    """
    image_path = Path(image_path)
    identity = provider.key_identity()
    row: typing.Dict[str, typing.Any] = {
        "image_path": image_path.as_posix(),
        "image_sha256": None,
        "status": "error",
        "error": None,
        "hsqr_l1_distance": None,
        "hsqr_score": None,
        "score_definition": HSQR_SCORE_DEFINITION,
        "score_direction": HSQR_SCORE_DIRECTION,
        "comparison_operator": HSQR_COMPARISON_OPERATOR,
        "threshold": threshold_info.get("threshold"),
        "distance_threshold": threshold_info.get("distance_threshold"),
        "threshold_source": threshold_info.get("threshold_source"),
        "detection_success": None,
        "inversion_steps": int(provider.inversion_steps),
        "inversion_prompt_sha256": sfw_bundle.sha256_text(provider.inversion_prompt),
        "inversion_guidance_scale": float(provider.inversion_guidance),
        "inversion_impl_version": sfw_inversion.SFW_INVERSION_IMPL_VERSION,
        "inversion_parity_status": sfw_inversion.SFW_INVERSION_PARITY_STATUS,
        "inversion_weights_parity": sfw_inversion.SFW_INVERSION_WEIGHTS_PARITY,
        "vae_sample": bool(provider.vae_sample),
        "recovered_latent_sha256": None,
    }
    row.update(identity)
    try:
        row["image_sha256"] = sfw_bundle.sha256_file(image_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inversion = provider.invert_pil_image(image, pipe_provider_target=pipe_provider)
        detections = provider.detect_from_latent(inversion["zT_torch"])
        if len(detections) != 1:
            raise SfwBundleError(
                f"one suspect image must produce exactly one detector record, got {len(detections)}"
            )
        detection = detections[0]
        row.update(detection)
        row["inversion_steps"] = inversion["inversion_steps"]
        row["recovered_latent_sha256"] = inversion["recovered_latent_sha256"]
        row["detection_success"] = provider.decide(
            detection["hsqr_score"], threshold_info.get("threshold")
        )
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - per-image containment is required
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "error"
        row["detection_success"] = None
    return row


def official_roc(positive_scores, negative_scores, target_fpr) -> typing.Dict[str, typing.Any]:
    """Official ``detect.py`` ROC bookkeeping on ``score = -L1 distance``."""
    return runner_common.official_roc(
        positive_scores,
        negative_scores,
        target_fpr,
        score_definition=HSQR_SCORE_DEFINITION,
        error_cls=SfwBundleError,
    )


def cohort_report_label(mode: str, profile_is_official: bool) -> str:
    """Label for a cohort run. Only an official-profile paper run qualifies."""
    if mode == "paper_eval":
        return "official_paper_evaluation" if profile_is_official else "legacy_or_ablation_mode"
    return "calibrated_deployment_verification"


# ---------------------------------------------------------------------------
# HSTR runner support
# ---------------------------------------------------------------------------

HSTR_CSV_COLUMNS = [
    "image_index", "image_path", "image_sha256", "cohort_role", "status", "error",
    "report_label", "hstr_channel_0_l1", "hstr_channel_3_l1", "hstr_channel_min_l1",
    "score", "score_definition", "score_direction", "threshold", "threshold_source",
    "comparison_operator", "detection_success", "selected_key_index", "selected_key_seed",
    "selected_pattern_sha256", "bundle_sha256", "inversion_steps", "inversion_prompt_sha256",
    "inversion_guidance_scale", "inversion_parity_status", "inversion_weights_parity",
    "recovered_latent_sha256",
]
PAIR_REQUIRED_FIELDS = ("sample_id", "prompt_sha256", "sample_seed")
PAIR_DISTORTION_FIELDS = ("distortion_config_sha256", "distortion_seed")


def write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, typing.Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = HSTR_CSV_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def resolve_pairing(positives: typing.Sequence[Path], negatives: typing.Sequence[Path], pair_manifest=None) -> typing.Dict[str, typing.Any]:
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
    result.update({"paired": True, "pairs": pairs, "pairing_sha256": sfw_bundle.canonical_sha256({"pairs": pairs}), "protocol": result["protocol"] or (pairs[0].get("protocol") if pairs else None)})
    return result


def hstr_score_image(provider: HSTRProvider, pipe_provider, image_path: typing.Union[str, Path], threshold_info, image_index: int | None = None) -> typing.Dict[str, typing.Any]:
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
        "bundle_sha256": sfw_bundle.sha256_file(provider.bundle.manifest_path) if provider.bundle is not None else None,
        "inversion_steps": int(provider.inversion_steps),
        "inversion_prompt_sha256": sfw_bundle.sha256_text(provider.inversion_prompt),
        "inversion_guidance_scale": float(provider.inversion_guidance),
        "inversion_parity_status": sfw_inversion.SFW_INVERSION_PARITY_STATUS,
        "inversion_weights_parity": sfw_inversion.SFW_INVERSION_WEIGHTS_PARITY,
        "recovered_latent_sha256": None,
    }
    try:
        row["image_sha256"] = sfw_bundle.sha256_file(image_path)
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
            "inversion_parity_status": inversion.get("inversion_parity_status"),
            "inversion_weights_parity": inversion.get("inversion_weights_parity"),
        })
        row["detection_success"] = provider.decide(row["score"], threshold_info.get("threshold"))
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - per-image containment is required
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["detection_success"] = None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def hstr_official_roc(positive_scores: typing.Sequence[float], negative_scores: typing.Sequence[float], target_fpr: float) -> typing.Dict[str, typing.Any]:
    return runner_common.official_roc(
        positive_scores,
        negative_scores,
        target_fpr,
        score_definition=HSTR_SCORE_DEFINITION,
        error_cls=SfwBundleError,
    )


def require_official_generation_profile(args) -> None:
    expected = {
        "hstr_profile": OFFICIAL_HSTR_PROFILE,
        "modelid_target": HSTR_OFFICIAL_MODEL_ID,
        "scheduler_target": HSTR_OFFICIAL_SCHEDULER,
        "resolution": HSTR_OFFICIAL_RESOLUTION,
        "num_inference_steps_target": HSTR_OFFICIAL_STEPS,
        "guidance_scale_target": HSTR_OFFICIAL_GUIDANCE_SCALE,
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise SfwBundleError(f"official HSTR generation requires {field}={value!r}, got {getattr(args, field)!r}")
