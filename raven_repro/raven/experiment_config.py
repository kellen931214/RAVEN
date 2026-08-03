"""Normalized experiment configuration for the unified main/eval pipeline.

Every algorithm parameter that affects reproducibility is declared here with
an explicit default.  The config is split into *algorithm* fields (must match
for resume) and *execution* fields (can differ across runs).
"""

from __future__ import annotations

from typing import Any, Mapping

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
# Algorithm fields — must match for resume
# --------------------------------------------------------------------------- #
# Fields whose values determine attack output.  If any of these differ between
# the stored config.json and the current run, resume is refused.
ALGORITHM_FIELDS = frozenset({
    "model_id",
    "model_revision",
    "dtype",
    "diffusion_mode",
    "inversion_mode",
    "scheduler_mode",
    "method",
    "dataset",
    "roles",
    "metadata_path",
    "steps",
    "strength",
    "guidance_scale",
    "shift_space",
    "warp_mode",
    "latent_sampling_mode",
    "padding_mode",
    "view_guided_attention",
    "color_transfer",
    "shift_mode",
    "shift_x",
    "shift_y",
    "shift_magnitude_min",
    "shift_magnitude_max",
    "base_seed",
    "prompt",
    "negative_prompt",
    "debug",
    "save_input_copy",
})

# --------------------------------------------------------------------------- #
# Execution fields — can differ across runs
# --------------------------------------------------------------------------- #
# These are recorded for provenance but do NOT affect attack output.  They are
# excluded from resume config comparison.
EXECUTION_FIELDS = frozenset({
    "output_dir",
    "limit",
    "gpu",
    "overwrite",
    "resume",
    "log_level",
    "save_intermediates",
})


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
    prompt: str = "",
    negative_prompt: str = "",
    debug: bool = False,
    save_input_copy: bool = True,
    save_intermediates: bool = False,
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
        # --- algorithm fields ---
        "method": method.upper(),
        "dataset": dataset,
        "metadata_path": str(metadata_path),
        "roles": list(roles),
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
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "debug": bool(debug),
        "save_input_copy": bool(save_input_copy),
        "model_id": model_id,
        "model_revision": model_revision,
        "dtype": dtype,
        # --- execution fields ---
        "output_dir": str(output_dir),
        "limit": limit,
        "gpu": gpu,
        "overwrite": overwrite,
        "resume": resume,
        "save_intermediates": bool(save_intermediates),
    }
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
        "prompt": None,  # filled per-sample from metadata
        "negative_prompt": config["negative_prompt"],
        "debug": config["debug"],
        "inversion_mode": config["inversion_mode"],
        "save_input_copy": config["save_input_copy"],
    }


def check_config_match(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Return list of *algorithm* fields whose values differ.

    Only algorithm fields are compared; execution fields (output_dir, gpu,
    resume, overwrite, limit, save_intermediates, log_level) are ignored.
    An empty list means resume is safe.
    """
    mismatches = []
    for key in sorted(ALGORITHM_FIELDS):
        stored_val = stored.get(key)
        current_val = current.get(key)
        if stored_val != current_val:
            mismatches.append(key)
    return mismatches
