"""Normalized experiment configuration for the unified main/eval pipeline.

Every algorithm parameter that affects reproducibility is declared here with
an explicit default so the recorded config is self-contained and a config
mismatch during resume fails fast with the list of drifted fields.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .eval_protocol import canonical_json_hash

# --------------------------------------------------------------------------- #
# Diffusion mode mapping
# --------------------------------------------------------------------------- #
# Three modes are supported.  DDPM inversion + DDIM scheduler is rejected
# because the DDPM forward-noise path is not compatible with DDIM denoising.
#
#   ddim       DDIM inversion, DDIM scheduler
#   ddpm       DDPM inversion, DDPM scheduler
#   ddim-ddpm  DDIM inversion, DDPM scheduler  (paper hybrid)
# --------------------------------------------------------------------------- #
DIFFUSION_MODE_MAP: dict[str, dict[str, str]] = {
    "ddim": {
        "inversion_mode": "ddim",
        "scheduler_mode": "ddim",
    },
    "ddpm": {
        "inversion_mode": "ddpm",
        "scheduler_mode": "ddpm",
    },
    "ddim-ddpm": {
        "inversion_mode": "ddim",
        "scheduler_mode": "ddpm",
    },
}

FORBIDDEN_PAIR = ("ddpm", "ddim")  # inversion=ddpm, scheduler=ddim
VALID_DIFFUSION_MODES = frozenset(DIFFUSION_MODE_MAP)


# --------------------------------------------------------------------------- #
# Default attack configuration
# --------------------------------------------------------------------------- #
DEFAULT_ATTACK_CONFIG: dict[str, Any] = {
    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    "steps": 50,
    "strength": 0.15,
    "guidance_scale": 2.5,
    "shift_space": "image_pixels",
    "warp_mode": "raven_paper_nfpa_gap_fill",
    "latent_sampling_mode": "nearest",
    "padding_mode": "reflection",
    "view_guided_attention": True,
    "color_transfer": True,
    "color_transfer_mode": "paper_exact_two_stage_aligned",
    "prompt": "",
    "negative_prompt": "",
    "shift_mode": "random",
    "shift_magnitude_min": 24,
    "shift_magnitude_max": 32,
    "shift_x": None,
    "shift_y": None,
    "inversion_mode": "ddim",
    "scheduler_mode": "ddim",
    "base_seed": 42,
    "save_input_copy": True,
    "debug": False,
}

IMMUTABLE_FIELDS = frozenset({
    "model_id",
    "model_revision",
    "steps",
    "strength",
    "guidance_scale",
    "shift_space",
    "warp_mode",
    "latent_sampling_mode",
    "padding_mode",
    "view_guided_attention",
    "color_transfer",
    "color_transfer_mode",
})


# --------------------------------------------------------------------------- #
# Config normalization
# --------------------------------------------------------------------------- #
def resolve_diffusion_mode(diffusion_mode: str) -> dict[str, str]:
    """Map a named diffusion mode to its (inversion_mode, scheduler_mode) pair."""
    if diffusion_mode not in DIFFUSION_MODE_MAP:
        raise ValueError(
            f"Unsupported diffusion_mode={diffusion_mode!r}. "
            f"Valid modes: {sorted(DIFFUSION_MODE_MAP)}"
        )
    return dict(DIFFUSION_MODE_MAP[diffusion_mode])


def validate_diffusion_pair(inversion_mode: str, scheduler_mode: str) -> None:
    """Reject the forbidden DDPM-inversion + DDIM-scheduler combination."""
    pair = (inversion_mode, scheduler_mode)
    if pair == FORBIDDEN_PAIR:
        raise ValueError(
            "DDPM inversion + DDIM scheduler is not supported. "
            "Valid pairs: (ddim, ddim), (ddpm, ddpm), (ddim, ddpm)."
        )
    valid_pairs = {
        (v["inversion_mode"], v["scheduler_mode"])
        for v in DIFFUSION_MODE_MAP.values()
    }
    if pair not in valid_pairs:
        raise ValueError(
            f"Unsupported inversion/scheduler pair: {pair}. "
            f"Valid pairs: {sorted(valid_pairs)}"
        )


def normalize_config(
    *,
    diffusion_mode: str = "ddim",
    method: str = "TR",
    dataset: str = "unspecified",
    metadata_path: str = "",
    output_dir: str = "",
    roles: list[str] | None = None,
    limit: int | None = None,
    gpu: int = 0,
    overwrite: bool = False,
    resume: bool = False,
    shift_mode: str = "random",
    shift_x: float | None = None,
    shift_y: float | None = None,
    shift_magnitude_min: int = 24,
    shift_magnitude_max: int = 32,
    base_seed: int = 42,
    steps: int = 50,
    strength: float = 0.15,
    guidance_scale: float = 2.5,
    shift_space: str = "image_pixels",
    warp_mode: str = "raven_paper_nfpa_gap_fill",
    latent_sampling_mode: str = "nearest",
    padding_mode: str = "reflection",
    view_guided_attention: bool = True,
    color_transfer: bool = True,
    color_transfer_mode: str = "paper_exact_two_stage_aligned",
    prompt: str = "",
    negative_prompt: str = "",
    debug: bool = False,
    save_input_copy: bool = True,
    model_id: str = "RedbeardNZ/stable-diffusion-2-1-base",
    model_revision: str = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    dtype: str = "float16",
    **kwargs: Any,
) -> dict[str, Any]:
    """Produce a self-contained normalized config for a run.

    Every algorithm parameter is resolved explicitly so the recorded config
    fully determines reproducibility.  Unknown kwargs are rejected.
    """
    if kwargs:
        raise ValueError(f"Unknown config keys: {sorted(kwargs)}")

    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    diffusion = resolve_diffusion_mode(diffusion_mode)
    inversion_mode = diffusion["inversion_mode"]
    scheduler_mode = diffusion["scheduler_mode"]
    validate_diffusion_pair(inversion_mode, scheduler_mode)

    if shift_mode not in ("none", "fixed", "random"):
        raise ValueError(
            f"shift_mode must be 'none', 'fixed', or 'random', got {shift_mode!r}"
        )
    if shift_mode == "fixed" and (shift_x is None or shift_y is None):
        raise ValueError("shift_mode='fixed' requires --shift-x and --shift-y")
    if shift_mode == "none":
        shift_x, shift_y = 0.0, 0.0

    if roles is None:
        roles = ["watermarked"]

    config: dict[str, Any] = {
        "method": method.upper(),
        "dataset": dataset,
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "roles": list(roles),
        "limit": limit,
        "gpu": gpu,
        "overwrite": overwrite,
        "resume": resume,
        "diffusion_mode": diffusion_mode,
        "inversion_mode": inversion_mode,
        "scheduler_mode": scheduler_mode,
        "shift_mode": shift_mode,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "shift_magnitude_min": int(shift_magnitude_min),
        "shift_magnitude_max": int(shift_magnitude_max),
        "base_seed": int(base_seed),
        "steps": int(steps),
        "strength": float(strength),
        "guidance_scale": float(guidance_scale),
        "shift_space": shift_space,
        "warp_mode": warp_mode,
        "latent_sampling_mode": latent_sampling_mode,
        "padding_mode": padding_mode,
        "view_guided_attention": bool(view_guided_attention),
        "color_transfer": bool(color_transfer),
        "color_transfer_mode": color_transfer_mode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "debug": bool(debug),
        "save_input_copy": bool(save_input_copy),
        "model_id": model_id,
        "model_revision": model_revision,
        "dtype": dtype,
    }
    config["config_hash"] = canonical_json_hash(
        {key: config[key] for key in sorted(config) if key != "config_hash"}
    )
    return config


def config_for_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the subset of config keys passed to ``RavenPipeline.run()``."""
    return {
        "steps": config["steps"],
        "strength": config["strength"],
        "guidance_scale": config["guidance_scale"],
        "shift_space": config["shift_space"],
        "warp_mode": config["warp_mode"],
        "latent_sampling_mode": config["latent_sampling_mode"],
        "padding_mode": config["padding_mode"],
        "shift_x": config["shift_x"],
        "shift_y": config["shift_y"],
        "view_guided_attention": config["view_guided_attention"],
        "color_transfer": config["color_transfer"],
        "seed": None,  # filled per-sample
        "prompt": config["prompt"],
        "negative_prompt": config["negative_prompt"],
        "debug": config["debug"],
        "inversion_mode": config["inversion_mode"],
        "save_input_copy": config["save_input_copy"],
    }


def check_config_match(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Return list of fields whose values differ between stored and current config.

    An empty list means the configs match and resume is safe.
    """
    mismatches = []
    for key in sorted(set(stored) | set(current)):
        if key == "config_hash":
            continue
        stored_val = stored.get(key)
        current_val = current.get(key)
        if stored_val != current_val:
            mismatches.append(key)
    return mismatches
