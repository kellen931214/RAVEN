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
        torch_dtype=TORCH_DTYPES[args.hsqr_torch_dtype],
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


def build_provider(args, latent_shape, device: torch.device) -> HSQRProvider:
    """Construct a provider from the CLI arguments (generation side)."""
    kwargs = dict(vars(args))
    kwargs.pop("latent_shape", None)
    kwargs.pop("device", None)
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


def run_provenance(args, provider: HSQRProvider, pipe_provider) -> typing.Dict[str, typing.Any]:
    provenance = {
        "method": "HSQR",
        "official_reference_repo": sfw_bundle.OFFICIAL_SFWMARK_REPO,
        "official_reference_commit": sfw_bundle.OFFICIAL_SFWMARK_COMMIT,
        "hsqr_profile": provider.profile,
        "hsqr_profile_is_official": bool(provider.profile_is_official),
        "hsqr_profile_overrides": dict(provider.profile_overrides),
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None),
        "scheduler": args.scheduler_target,
        "torch_dtype": args.hsqr_torch_dtype,
        "resolution": int(args.resolution),
        "num_inference_steps": int(args.num_inference_steps_target),
        "guidance_scale": float(args.guidance_scale_target),
        "inversion_impl_version": sfw_inversion.SFW_INVERSION_IMPL_VERSION,
        "inversion_parity_status": sfw_inversion.SFW_INVERSION_PARITY_STATUS,
        "inversion_weights_parity": sfw_inversion.SFW_INVERSION_WEIGHTS_PARITY,
        "created_utc": sfw_bundle.utc_now(),
    }
    provenance.update(provider.key_identity())
    provenance.update(sfw_bundle.git_provenance())
    if provider.bundle is not None:
        provenance["hsqr_bundle_dir"] = provider.bundle.dir.as_posix()
        provenance["hsqr_bundle_config_sha256"] = provider.bundle.manifest.get("bundle_config_sha256")
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
