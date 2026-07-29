"""Thin IO/orchestration helpers shared by the RingID runners.

This module contains **no** RingID algorithm. Every numeric step is delegated to
:class:`utils.wm.ringid_provider.RingIDProvider`, which is the single source of
truth. Generic bookkeeping that is not RingID-specific (GPU preflight, image
enumeration, cohort pairing, ROC at a target FPR, run-manifest/resume gates) is
reused from :mod:`utils.wm.gm_runtime` rather than reimplemented, so RAVEN keeps
one authoritative implementation of each.
"""

from __future__ import annotations

import platform
import typing
from pathlib import Path

import torch
from PIL import Image

from . import gm_runtime, rid_bundle
from .rid_bundle import RidBundle, RidBundleError
from .ringid_provider import (
    RID_SCORE_DEFINITION,
    TORCH_DTYPES,
    RingIDProvider,
)

# Reused generic helpers (see module docstring).
enumerate_images = gm_runtime.enumerate_images
gpu_preflight = gm_runtime.gpu_preflight
resolve_pairing = gm_runtime.resolve_pairing
assert_run_manifest_compatible = gm_runtime.assert_run_manifest_compatible
official_roc = gm_runtime.official_roc

#: Fields that must match before an existing RingID sample may be resumed.
RESUME_FIELDS = (
    "sample_seed",
    "prompt_sha256",
    "run_config_sha256",
    "rid_bundle_config_sha256",
    "selected_pattern_sha256",
)


def assert_resumable(name: str, existing: typing.Mapping[str, typing.Any],
                     expected: typing.Mapping[str, typing.Any]) -> None:
    """RingID resume gate (shared comparison, RingID identity fields)."""
    gm_runtime.assert_resumable(name, existing, expected, fields=RESUME_FIELDS)


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
        torch_dtype=TORCH_DTYPES[args.rid_torch_dtype],
    )


def build_provider(args, latent_shape, device: torch.device,
                   create_bundle: bool = False) -> RingIDProvider:
    kwargs = dict(vars(args))
    kwargs.pop("latent_shape", None)
    kwargs.pop("device", None)
    return RingIDProvider(
        latent_shape=latent_shape,
        device=device,
        rid_create_bundle=create_bundle,
        **kwargs,
    )


def run_provenance(args, provider: RingIDProvider, pipe_provider=None
                   ) -> typing.Dict[str, typing.Any]:
    provenance = {
        "method": "RID",
        "official_reference_repo": rid_bundle.OFFICIAL_RINGID_REPO,
        "official_reference_commit": rid_bundle.OFFICIAL_RINGID_COMMIT,
        "profile_name": provider.profile,
        "rid_profile_is_official": bool(provider.profile_is_official),
        "rid_profile_overrides": dict(provider.profile_overrides),
        "spatial_shift_factor_semantics": provider.shift_semantics,
        "spatial_shift_factor": provider.time_shift_factor,
        "model_id": args.modelid_target,
        "model_revision": getattr(args, "model_revision", None),
        "scheduler": args.scheduler_target,
        "torch_dtype": args.rid_torch_dtype,
        "resolution": int(args.resolution),
        "num_inference_steps": int(args.num_inference_steps_target),
        "guidance_scale": float(args.guidance_scale_target),
        "inversion_steps": int(provider.inversion_steps),
        "inversion_prompt_sha256": rid_bundle.sha256_text(provider.inversion_prompt),
        "inversion_guidance_scale": float(provider.inversion_guidance),
        "vae_sample": bool(provider.vae_sample),
        "vae_scaling_factor": float(provider.vae_scaling_factor),
        "channel_min": int(provider.channel_min),
        "score_definition": RID_SCORE_DEFINITION,
        "score_direction": "higher_is_watermarked",
        "selected_key_index": int(provider.key_index),
        "selected_key_id": provider.selected_key_id,
        "selected_pattern_sha256": rid_bundle.sha256_tensor(provider.gt_patch),
        "mask_sha256": rid_bundle.sha256_tensor(provider.watermarking_mask),
        "candidate_count": provider.candidate_count,
        "candidate_order_sha256": provider.candidate_order_sha256(),
        "rng_seed": provider.key_seed,
        "rng_device": provider.key_rng_device,
        "rng_dtype": provider.key_rng_dtype_name,
        "rid_state_source": provider.state_source,
        "created_utc": rid_bundle.utc_now(),
        "platform": platform.platform(),
    }
    provenance.update(rid_bundle.git_provenance())
    if provider.bundle is not None:
        provenance["rid_bundle_dir"] = provider.bundle.dir.as_posix()
        provenance["rid_bundle_config_sha256"] = provider.bundle.manifest.get(
            "bundle_config_sha256"
        )
    return provenance


def inversion_config_sha256(provider: RingIDProvider) -> str:
    """Hash of everything that determines the recovered latent."""
    return rid_bundle.canonical_sha256({
        "model_id": provider.model_id,
        "model_revision": provider.model_revision,
        "scheduler": provider.scheduler,
        "torch_dtype": provider.torch_dtype_name,
        "resolution": provider.resolution,
        "inversion_prompt_sha256": rid_bundle.sha256_text(provider.inversion_prompt),
        "inversion_guidance_scale": provider.inversion_guidance,
        "inversion_steps": provider.inversion_steps,
        "vae_sample": provider.vae_sample,
        "vae_scaling_factor": provider.vae_scaling_factor,
        "update_rule": "official_ringid_forward_diffusion_manual_ddim",
    })


#: Per-image output columns required by Issue #3. Fields that do not apply to
#: the selected mode stay ``None``; they are never filled with a fabricated
#: default.
RESULT_FIELDS = (
    "image_index",
    "image_path",
    "image_sha256",
    "status",
    "error",
    "profile_name",
    "keybook_sha256",
    "candidate_count",
    "selected_reference_key_id",
    "rid_channel_0_l1",
    "rid_channel_3_l1",
    "rid_channel_min_l1",
    "rid_score",
    "score_definition",
    "score_direction",
    "threshold",
    "threshold_source",
    "comparison_operator",
    "detection_success",
    "predicted_key_index",
    "predicted_key_id",
    "best_distance",
    "second_best_distance",
    "identification_margin",
    "top_k_candidates",
    "true_key_index",
    "identification_correct",
    "recovered_latent_sha256",
    "inversion_config_sha256",
    "model_id",
    "model_revision",
    "official_reference_commit",
    "git_branch",
    "git_commit",
)


def _empty_row(image_index: int, image_path: Path,
               provider: RingIDProvider) -> typing.Dict[str, typing.Any]:
    return {
        "image_index": int(image_index),
        "image_path": Path(image_path).as_posix(),
        "image_sha256": None,
        "status": "error",
        "error": None,
        "profile_name": provider.profile,
        "keybook_sha256": None,
        "candidate_count": None,
        "selected_reference_key_id": provider.selected_key_id,
        "rid_channel_0_l1": None,
        "rid_channel_3_l1": None,
        "rid_channel_min_l1": None,
        "rid_score": None,
        "score_definition": RID_SCORE_DEFINITION,
        "score_direction": "higher_is_watermarked",
        "threshold": None,
        "threshold_source": None,
        "comparison_operator": ">=",
        "detection_success": None,
        "predicted_key_index": None,
        "predicted_key_id": None,
        "best_distance": None,
        "second_best_distance": None,
        "identification_margin": None,
        "top_k_candidates": None,
        "true_key_index": None,
        "identification_correct": None,
        "recovered_latent_sha256": None,
        "inversion_config_sha256": inversion_config_sha256(provider),
    }


def score_image(provider: RingIDProvider, pipe_provider, image_path: typing.Union[str, Path],
                image_index: int, threshold_info: typing.Mapping[str, typing.Any],
                identify: bool = False,
                candidate_indices: typing.Optional[typing.Sequence[int]] = None,
                true_key_index: typing.Optional[int] = None
                ) -> typing.Dict[str, typing.Any]:
    """Invert and score exactly one suspect image.

    A per-image failure is recorded as ``status="error"`` — never as a negative
    detection and never merged into another image's score.
    """
    image_path = Path(image_path)
    row = _empty_row(image_index, image_path, provider)
    row.update({
        "threshold": threshold_info.get("threshold"),
        "threshold_source": threshold_info.get("threshold_source"),
        "comparison_operator": threshold_info.get("comparison_operator", ">="),
    })
    try:
        row["image_sha256"] = rid_bundle.sha256_file(image_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inversion = provider.invert_pil_image(
            image, pipe_provider_target=pipe_provider, image_sha256=row["image_sha256"]
        )
        latent = inversion["zT_torch"]
        row["recovered_latent_sha256"] = inversion["recovered_latent_sha256"]

        distances = provider.channel_distances(latent)[0]
        row.update({
            "rid_channel_0_l1": distances["rid_channel_0_l1"],
            "rid_channel_3_l1": distances["rid_channel_3_l1"],
            "rid_channel_min_l1": distances["rid_channel_min_l1"],
            "rid_score": distances["rid_score"],
        })
        row["detection_success"] = provider.decide(
            distances["rid_score"], threshold_info.get("threshold")
        )

        if identify:
            identification = provider.identify_key(
                latent, true_key_index=true_key_index, candidate_indices=candidate_indices
            )[0]
            row.update({
                "keybook_sha256": identification["keybook_sha256"],
                "candidate_count": identification["candidate_count"],
                "predicted_key_index": identification["predicted_key_index"],
                "predicted_key_id": identification["predicted_key_id"],
                "best_distance": identification["best_distance"],
                "second_best_distance": identification["second_best_distance"],
                "identification_margin": identification["identification_margin"],
                "top_k_candidates": [
                    {"key_index": idx, "key_id": f"rid-key-{idx:06d}", "distance": dist}
                    for idx, dist in zip(identification["top_k_key_indices"],
                                         identification["top_k_distances"])
                ],
                "true_key_index": identification["true_key_index"],
                "identification_correct": identification["identification_correct"],
            })
        row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - per-image containment is required
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "error"
        row["detection_success"] = None
    return row


def assert_bundle_untouched(bundle_dir: typing.Union[str, Path],
                            before: typing.Mapping[str, typing.Any],
                            allowed: typing.Sequence[str] = ()) -> None:
    """A read-only run must not modify the key artifact."""
    after = RidBundle.load(bundle_dir).artifact_mtimes()
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    unexpected = changed - set(allowed)
    if unexpected:
        raise RidBundleError(
            f"the RingID bundle was modified by a run that must not touch it: {sorted(unexpected)}"
        )
