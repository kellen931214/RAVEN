"""Authoritative protocol and provenance helpers for formal RAVEN evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC_PROTOCOL_VERSION = "raven_formal_eval_v2_aligned_color"

# --------------------------------------------------------------------------- #
# Canonical repository data / output layout (migration 2026-07-26)
# --------------------------------------------------------------------------- #
# The repository has exactly five data/output roots. Nothing else may be created
# at the top of ``data/`` or ``outputs/``:
#
#   data/clean/   original clean images + the metadata that defines them
#   data/tr/      Tree-Ring watermarked images + metadata
#   data/gs/      Gaussian Shading watermarked images + metadata
#   outputs/tr/   every Tree-Ring run artifact (attack/eval/metrics/provenance/log)
#   outputs/gs/   every Gaussian Shading run artifact
#
# Small gates / smoke tests / dry-runs must NOT be written into outputs/; use
# ``scratch_run_root()`` (a /tmp directory) instead.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
CLEAN_DATA_ROOT = DATA_ROOT / "clean"

# method -> (watermarked data root, run output root)
METHOD_DATA_ROOTS: dict[str, Path] = {
    "TR": DATA_ROOT / "tr",
    "GS": DATA_ROOT / "gs",
    # shared_tr_clean_v2 cohorts (Issue #9). These roots hold only the
    # method-specific watermarked images; the clean images stay in data/clean/
    # and are never duplicated per method.
    "GM": DATA_ROOT / "gm",
    "T2S": DATA_ROOT / "t2s",
    "RID": DATA_ROOT / "rid",
    "HSTR": DATA_ROOT / "hstr",
    "HSQR": DATA_ROOT / "hsqr",
}
METHOD_OUTPUT_ROOTS: dict[str, Path] = {
    "TR": OUTPUTS_ROOT / "tr",
    "GS": OUTPUTS_ROOT / "gs",
    "GM": OUTPUTS_ROOT / "gm",
    "T2S": OUTPUTS_ROOT / "t2s",
    "RID": OUTPUTS_ROOT / "rid",
    "HSTR": OUTPUTS_ROOT / "hstr",
    "HSQR": OUTPUTS_ROOT / "hsqr",
}

# Methods whose formal protocol includes the attacked-clean recalibration branch.
# Everything else must never produce attacked-clean artifacts.
ATTACK_CLEAN_METHODS = frozenset({"TR"})


def method_data_root(method: str) -> Path:
    """Canonical watermarked-data root for ``method`` (fail closed on unknown)."""
    key = str(method).upper()
    try:
        return METHOD_DATA_ROOTS[key]
    except KeyError:
        raise ValueError(
            f"no canonical data root for method {method!r}; "
            f"known methods: {sorted(METHOD_DATA_ROOTS)}"
        ) from None


def method_output_root(method: str) -> Path:
    """Canonical run-output root for ``method`` (fail closed on unknown)."""
    key = str(method).upper()
    try:
        return METHOD_OUTPUT_ROOTS[key]
    except KeyError:
        raise ValueError(
            f"no canonical output root for method {method!r}; "
            f"known methods: {sorted(METHOD_OUTPUT_ROOTS)}"
        ) from None


# Methods whose cohort uses the flat layout:
#
#   data/<method>/<dataset>/metadata.csv
#   data/<method>/<dataset>/<run_id>/watermarked.png
#
# Every other method keeps the nested per-method form:
#
#   data/<method>/<dataset>/<METHOD>/metadata.csv
#   data/<method>/<dataset>/<METHOD>/<run_id>/watermarked.png
#
# Tree-Ring moved to the flat layout on 2026-07-27. This is a per-method fact,
# not a global rule: a method is listed here only once its cohort on disk has
# actually been moved.
FLAT_COHORT_METHODS = frozenset({"TR"})


def cohort_dir(method: str, dataset: str) -> Path:
    """Directory holding one method's cohort metadata and per-run image dirs."""
    root = method_data_root(method) / dataset
    if str(method).upper() in FLAT_COHORT_METHODS:
        return root
    return root / str(method).upper()


def source_metadata_path(method: str, dataset: str) -> Path:
    """Canonical source metadata CSV for a generated cohort."""
    return cohort_dir(method, dataset) / "metadata.csv"


def watermarked_image_path(method: str, dataset: str, run_id: int | str) -> Path:
    """Canonical watermarked image path for one run of a cohort."""
    name = f"{int(run_id):06d}" if str(run_id).isdigit() else str(run_id)
    return cohort_dir(method, dataset) / name / "watermarked.png"


def clean_data_dir(dataset: str, method: str | None = None) -> Path:
    """Canonical clean-image directory for a dataset (GS cohorts nest under GS/)."""
    root = CLEAN_DATA_ROOT / dataset
    if method is not None and str(method).upper() == "GS":
        return root / "GS"
    return root


def formal_run_key(source_manifest_sha256: str, attack_config_hash: str) -> str:
    """Stable run key from content hashes — deliberately NOT a timestamp.

    Re-running the same method/dataset/variant with the same source manifest and
    attack config resolves to the same output root, so work is continued with
    ``--resume`` instead of piling up a new timestamped directory each time.
    """
    if not source_manifest_sha256 or not attack_config_hash:
        raise ValueError("run key requires both source manifest and attack config hashes")
    return f"{str(source_manifest_sha256)[:12]}_{str(attack_config_hash)[:12]}"


def formal_output_root(
    method: str, dataset: str, variant: str, run_key: str
) -> Path:
    """Canonical formal-run output root: outputs/<method>/<dataset>/<variant>/<run-key>."""
    for name, value in (("dataset", dataset), ("variant", variant), ("run_key", run_key)):
        text = str(value)
        if not text or "/" in text or text in {".", ".."}:
            raise ValueError(f"invalid {name} path component: {value!r}")
    return method_output_root(method) / dataset / variant / run_key


def scratch_run_root(method: str, purpose: str) -> Path:
    """Throwaway root for gates / smoke tests / dry-runs.

    Small non-formal runs must never be written under ``outputs/``. The caller
    deletes the directory on success and keeps (and reports) it on failure.
    """
    import tempfile

    safe_method = str(method).upper()
    safe_purpose = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(purpose))
    return Path(tempfile.mkdtemp(prefix=f"raven-{safe_method.lower()}-{safe_purpose}-"))


def is_scratch_run_root(path: str | Path, method: str) -> bool:
    """True only for a gate/smoke root created by ``scratch_run_root(method, …)``.

    Gates must live in /tmp rather than under ``outputs/``, so the canonical-root
    guard has to recognise exactly those roots — and nothing else in /tmp.
    """
    import tempfile

    resolved = Path(path).resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    if temporary not in resolved.parents:
        return False
    relative = resolved.relative_to(temporary)
    return relative.parts[0].startswith(f"raven-{str(method).lower()}-")


def assert_canonical_output_root(path: str | Path, method: str) -> Path:
    """Fail closed when a run would write outside its canonical method root."""
    resolved = Path(path).resolve()
    expected = method_output_root(method).resolve()
    if is_scratch_run_root(resolved, method):
        # Explicitly labelled throwaway gate root; never a formal output location.
        return resolved
    if resolved != expected and expected not in resolved.parents:
        raise ValueError(
            f"output root {resolved} is outside the canonical root for method "
            f"{str(method).upper()}: {expected}"
        )
    return resolved

FORMAL_ATTACK_CONFIG: dict[str, Any] = {
    "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
    "model_revision": "c6a5e9bab8d874d081de76fa270ae0aefa5410ff",
    "image_size": [512, 512],
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
    "inversion_mode": "ddim",
    "prompt": "",
    "negative_prompt": "",
    "shift_magnitudes_image_px": list(range(24, 33)),
    "shift_sign_policy": "independent_uniform_random_per_axis",
    "seed_policy": "base_seed_plus_numeric_run_id",
    "base_seed": 42,
}

CLIP_CONFIG: dict[str, Any] = {
    "clip_metric_type": "prompt-image cosine similarity",
    "clip_model_name": "ViT-bigG-14",
    "clip_pretrained": "laion2b_s39b_b160k",
    "preprocess_identifier": "open_clip.create_model_and_transforms",
    "text_source": "original generation prompt",
    "normalization": "L2-normalized image/text embeddings",
    "paper_name_interpretation": (
        "ViT-bigG-14 is an implementation interpretation of the paper's "
        "\"OpenCLIP-ViT/G\" naming, not a fully specified model/tag pair "
        "quoted verbatim from the paper."
    ),
}

TR_PROVIDER_FIELDS = (
    "w_seed",
    "w_channel",
    "w_radius",
    "w_pattern",
    "w_mask_shape",
    "w_measurement",
    "w_injection",
    "w_pattern_const",
)

PROVIDER_FIELDS_BY_METHOD = {
    "TR": TR_PROVIDER_FIELDS,
    "RID": (
        "rid_protocol_mode",
        "rid_bundle_config_sha256",
        "rid_selected_pattern_sha256",
        "rid_mask_sha256",
        "rid_key_index",
    ),
    "HSTR": (
        "hstr_protocol_mode",
        "hstr_bundle_config_sha256",
        "hstr_selected_pattern_sha256",
        "hstr_mask_sha256",
        "hstr_key_index",
    ),
    "HSQR": (
        "hsqr_protocol_mode",
        "hsqr_bundle_config_sha256",
        "hsqr_selected_pattern_sha256",
        "hsqr_mask_sha256",
        "hsqr_key_index",
    ),
    "GS": (
        "gs_protocol_mode",
        "message_width_in_bytes",
        "l",
        "num_replications",
        "gs_channel_copy",
        "gs_hw_copy",
        "gs_fpr",
        "gs_user_number",
    ),
    # GaussMarker: the watermark state is one cohort-wide bundle, so the provider
    # config is the bundle's identity. Every field here is a digest of an
    # artifact the detector must reproduce exactly; none has a meaningful
    # default, which is why they are all listed in
    # PROVIDER_REQUIRED_NONEMPTY_FIELDS below.
    "GM": (
        "gm_protocol_mode",
        "gm_bundle_config_sha256",
        "gm_w1_file_sha256",
        "gm_w2_file_sha256",
        "gm_watermark_sha256",
        "gm_m_sha256",
        "gm_target_sha256",
        "gm_mask_sha256",
    ),
    # T2SMark: the per-sample watermark lives in each row's own portable state
    # artifact, so the cohort-uniform provider config is the encoder/detector
    # profile those states were produced under. The per-sample state is bound
    # separately through t2s_state_sha256.
    "T2S": (
        "t2s_protocol_mode",
        "t2s_rng_mode",
        "t2s_inversion_mode",
        "t2s_num_inversion_steps",
        "t2s_provider_config_sha256",
    ),
}

# Provider fields that must be present and non-empty in the source row. For a
# digest or protocol identity there is no defensible default: silently falling
# back to "" would let a cohort with missing provenance produce a stable-looking
# provider_config_hash. Methods absent from this mapping keep the historical
# default-filling behaviour so existing TR/GS cohort hashes are unchanged.
PROVIDER_REQUIRED_NONEMPTY_FIELDS: dict[str, frozenset[str]] = {
    "GM": frozenset(PROVIDER_FIELDS_BY_METHOD["GM"]),
    "T2S": frozenset(PROVIDER_FIELDS_BY_METHOD["T2S"]),
    "RID": frozenset(PROVIDER_FIELDS_BY_METHOD["RID"]),
    "HSTR": frozenset(PROVIDER_FIELDS_BY_METHOD["HSTR"]),
    "HSQR": frozenset(PROVIDER_FIELDS_BY_METHOD["HSQR"]),
}

PROVIDER_DEFAULTS = {
    "TR": {
        "w_seed": 999999,
        "w_channel": 3,
        "w_radius": 10,
        "w_pattern": "ring",
        "w_mask_shape": "circle",
        "w_measurement": "l1_complex",
        "w_injection": "complex",
        "w_pattern_const": 0.0,
    },
    "RID": {
        "rid_protocol_mode": "",
        "rid_bundle_config_sha256": "",
        "rid_selected_pattern_sha256": "",
        "rid_mask_sha256": "",
        "rid_key_index": 0,
    },
    "HSTR": {
        "hstr_protocol_mode": "",
        "hstr_bundle_config_sha256": "",
        "hstr_selected_pattern_sha256": "",
        "hstr_mask_sha256": "",
        "hstr_key_index": 0,
    },
    "HSQR": {
        "hsqr_protocol_mode": "",
        "hsqr_bundle_config_sha256": "",
        "hsqr_selected_pattern_sha256": "",
        "hsqr_mask_sha256": "",
        "hsqr_key_index": 0,
    },
    "GS": {
        # Default is official_compatible; legacy must be requested explicitly.
        # Only a fallback/type hint here — formal GS rows always carry
        # gs_protocol_mode explicitly, so flipping this does not change existing
        # provider_config hashes. gs_detection_mode is deliberately NOT part of
        # the embedding-config hash (it is a detection-time policy, not a
        # watermark-construction parameter).
        "gs_protocol_mode": "official_compatible",
        "message_width_in_bytes": 32,
        "l": 1,
        "num_replications": 64,
        "gs_channel_copy": 1,
        "gs_hw_copy": 8,
        "gs_fpr": 1e-6,
        "gs_user_number": 1000000,
    },
    # Type hints only. Every GM/T2S field is required non-empty, so these
    # defaults are never substituted for a missing value — they exist so
    # _normalized_scalar knows whether to coerce to str or int.
    "GM": {
        "gm_protocol_mode": "",
        "gm_bundle_config_sha256": "",
        "gm_w1_file_sha256": "",
        "gm_w2_file_sha256": "",
        "gm_watermark_sha256": "",
        "gm_m_sha256": "",
        "gm_target_sha256": "",
        "gm_mask_sha256": "",
    },
    "T2S": {
        "t2s_protocol_mode": "",
        "t2s_rng_mode": "",
        "t2s_inversion_mode": "",
        "t2s_num_inversion_steps": 0,
        "t2s_provider_config_sha256": "",
    },
}

FORMAL_DEBUG_FIELDS = (
    "model_id",
    "model_revision",
    "steps",
    "strength",
    "guidance_scale",
    "inversion_mode",
    "exact_timestep",
    "warp_mode",
    "interpolation_mode",
    "padding_mode",
    "align_corners",
    "normalized_coordinate_formula",
    "pixel_center_offset_image_px",
    "warp_coordinate_convention",
    "warp_implementation_version",
    "color_transfer_mode",
    "inversion_prompt",
    "reconstruction_prompt",
    "transform_config_hash",
    "attack_device_class",
    "attack_dtype",
    "scheduler_class",
    "scheduler_config",
    "scheduler_config_hash",
    "torch_version",
    "diffusers_version",
)


def _reject_non_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains non-finite float: {value!r}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    _reject_non_finite(payload)
    text = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_scheduler_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only deterministic scheduler parameters that affect inference.

    Diffusers materializes ``_use_default_values`` from a set, so its list order
    can differ between otherwise identical scheduler instances. Package and
    class provenance are recorded separately and must not make paired attack
    transform hashes depend on non-semantic private metadata.
    """
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_source_manifest(
    path: str | Path, *, repo_root: str | Path
) -> tuple[dict[str, Any], str]:
    """Validate the immutable formal source manifest against the live files."""
    path = Path(path).resolve()
    companion = path.with_suffix(".sha256")
    if not path.is_file() or not companion.is_file():
        raise FileNotFoundError(f"source manifest or companion SHA is missing: {path}")
    actual_manifest_sha = sha256_path(path)
    recorded_manifest_sha = companion.read_text(encoding="utf-8").strip().split()[0]
    if actual_manifest_sha != recorded_manifest_sha:
        raise RuntimeError(
            "source manifest SHA mismatch: "
            f"recorded={recorded_manifest_sha} actual={actual_manifest_sha}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(repo_root).resolve()
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("formal source manifest has no files")
    seen: set[str] = set()
    for record in files:
        relative = str(record.get("relative_path", ""))
        if not relative or relative in seen:
            raise RuntimeError(f"invalid or duplicate source manifest path: {relative!r}")
        seen.add(relative)
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"source manifest path escapes repository: {relative}") from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"source size drift: {relative}")
        actual_sha = sha256_path(source)
        if actual_sha != record.get("sha256"):
            raise RuntimeError(
                f"source SHA drift: {relative} stored={record.get('sha256')} actual={actual_sha}"
            )
    return payload, actual_manifest_sha


def require_clean_git_worktree(repo_root: str | Path) -> str:
    """Return current commit only when tracked and untracked source state is clean."""
    root = Path(repo_root).resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "formal evaluation requires a clean working tree; commit or remove drift first:\n"
            + status
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def formal_runtime_provenance(
    *, scheduler_mode: str, device_class: str, attack_dtype: str
) -> dict[str, Any]:
    """Build the immutable runtime selector stored before model construction."""
    if device_class != "cuda":
        raise RuntimeError("formal RAVEN attack forbids CPU fallback; device class must be cuda")
    if attack_dtype != "torch.float16":
        raise RuntimeError("formal RAVEN attack dtype must be torch.float16")
    scheduler_classes = {"ddim": "DDIMScheduler", "ddpm": "DDPMScheduler"}
    if scheduler_mode not in scheduler_classes:
        raise ValueError(f"unsupported scheduler mode: {scheduler_mode}")
    scheduler_config = {
        "scheduler_mode": scheduler_mode,
        "scheduler_class": scheduler_classes[scheduler_mode],
        "model_id": FORMAL_ATTACK_CONFIG["model_id"],
        "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "source": "pinned model scheduler config; exact resolved config is recorded per attack",
    }
    return {
        "attack_device_class": device_class,
        "attack_dtype": attack_dtype,
        "scheduler_class": scheduler_classes[scheduler_mode],
        "scheduler_config": scheduler_config,
        "scheduler_config_hash": canonical_json_hash(scheduler_config),
        "torch_version": importlib.metadata.version("torch"),
        "diffusers_version": importlib.metadata.version("diffusers"),
    }


def normalize_formal_attack_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a formal attack variant without weakening the baseline protocol."""
    config = dict(payload)
    required = set(FORMAL_ATTACK_CONFIG)
    missing = sorted(required - set(config))
    extra = sorted(set(config) - required - {"scheduler_mode", "shift_plan_mode", "variant_name"})
    if missing or extra:
        raise ValueError(
            f"invalid formal attack config missing={missing} extra={extra}"
        )
    config.setdefault("scheduler_mode", "ddim")
    config.setdefault("shift_plan_mode", "paper_random_independent_axes")
    config.setdefault("variant_name", "formal_baseline")
    variant_fields = {
        "latent_sampling_mode", "inversion_mode", "scheduler_mode",
        "shift_plan_mode", "variant_name", "warp_mode", "color_transfer_mode",
    }
    drift = {
        key: (FORMAL_ATTACK_CONFIG[key], config[key])
        for key in FORMAL_ATTACK_CONFIG
        if key not in variant_fields and config[key] != FORMAL_ATTACK_CONFIG[key]
    }
    if drift:
        raise ValueError(
            "formal ablation config changes non-variant fields: "
            + ", ".join(sorted(drift))
        )
    if config["inversion_mode"] not in {"ddim", "ddpm"}:
        raise ValueError(f"unsupported inversion_mode={config['inversion_mode']!r}")
    if config["scheduler_mode"] not in {"ddim", "ddpm"}:
        raise ValueError(f"unsupported scheduler_mode={config['scheduler_mode']!r}")
    allowed_scheduler_pairs = {
        ("ddim", "ddim"),
        ("ddpm", "ddpm"),
        ("ddim", "ddpm"),
    }
    pair = (
        config["inversion_mode"],
        config["scheduler_mode"],
    )
    if pair not in allowed_scheduler_pairs:
        raise ValueError(
            "unsupported inversion/scheduler combination: "
            f"inversion_mode={pair[0]!r}, scheduler_mode={pair[1]!r}"
        )
    if config["latent_sampling_mode"] not in {"nearest", "bilinear"}:
        raise ValueError("latent_sampling_mode must be nearest or bilinear")
    if config["color_transfer_mode"] not in {"paper_exact_two_stage_aligned", "paper_exact_two_stage"}:
        raise ValueError("color_transfer_mode must be paper_exact_two_stage_aligned or paper_exact_two_stage")
    if config["shift_plan_mode"] not in {
        "paper_random_independent_axes", "balanced_deterministic_schedule", "zero"
    }:
        raise ValueError(
            "shift_plan_mode must be paper_random_independent_axes, "
            "balanced_deterministic_schedule, or zero"
        )
    if config["warp_mode"] not in {
        "raven_paper_nfpa_gap_fill",
        "raven_paper_nfpa_gap_fill_centered",
    }:
        raise ValueError(
            "formal variants require raven_paper_nfpa_gap_fill or "
            "raven_paper_nfpa_gap_fill_centered"
        )
    if config["padding_mode"] != "reflection":
        raise ValueError("formal variants require reflection padding")
    return config


def load_formal_attack_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize a tracked, immutable formal attack configuration."""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"formal attack config is not an object: {config_path}")
    return normalize_formal_attack_config(payload)


def formal_attack_config_hash(config: Mapping[str, Any] | None = None) -> str:
    return canonical_json_hash(
        normalize_formal_attack_config(FORMAL_ATTACK_CONFIG if config is None else config)
    )


def transform_config_payload(debug_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": debug_info["model_id"],
        "model_revision": debug_info["model_revision"],
        "steps": int(debug_info["steps"]),
        "strength": float(debug_info["strength"]),
        "guidance_scale": float(debug_info["guidance_scale"]),
        "inversion_mode": debug_info["inversion_mode"],
        "exact_timestep": int(debug_info["exact_timestep"]),
        "inversion_prompt": debug_info["inversion_prompt"],
        "reconstruction_prompt": debug_info["reconstruction_prompt"],
        "negative_prompt": debug_info["negative_prompt"],
        "warp_mode": debug_info["warp_mode"],
        "latent_sampling_mode": debug_info["interpolation_mode"],
        "padding_mode": debug_info["padding_mode"],
        "align_corners": debug_info["align_corners"],
        "normalized_coordinate_formula": debug_info["normalized_coordinate_formula"],
        "pixel_center_offset_image_px": (
            None
            if debug_info["pixel_center_offset_image_px"] is None
            else float(debug_info["pixel_center_offset_image_px"])
        ),
        "warp_coordinate_convention": debug_info["warp_coordinate_convention"],
        "warp_implementation_version": debug_info["warp_implementation_version"],
        "planned_flow_dx_image_px": float(debug_info["planned_flow_dx_image_px"]),
        "planned_flow_dy_image_px": float(debug_info["planned_flow_dy_image_px"]),
        "effective_source_flow_dx_image_px": float(
            debug_info["effective_source_flow_dx_image_px"]
        ),
        "effective_source_flow_dy_image_px": float(
            debug_info["effective_source_flow_dy_image_px"]
        ),
        "view_guided_attention": bool(debug_info["view_guided_attention"]),
        "color_transfer": bool(debug_info["color_transfer"]),
        "color_transfer_mode": debug_info["color_transfer_mode"],
        "attack_device_class": debug_info["attack_device_class"],
        "attack_dtype": debug_info["attack_dtype"],
        "scheduler_class": debug_info["scheduler_class"],
        "scheduler_config": debug_info["scheduler_config"],
        "scheduler_config_hash": debug_info["scheduler_config_hash"],
        "torch_version": debug_info["torch_version"],
        "diffusers_version": debug_info["diffusers_version"],
    }


def assert_formal_debug_info(
    debug_info: Mapping[str, Any],
    *,
    planned_flow_dx_image_px: float | None = None,
    planned_flow_dy_image_px: float | None = None,
    attack_config: Mapping[str, Any] | None = None,
) -> str:
    config = normalize_formal_attack_config(
        FORMAL_ATTACK_CONFIG if attack_config is None else attack_config
    )
    missing = [field for field in FORMAL_DEBUG_FIELDS if field not in debug_info]
    if missing:
        raise RuntimeError(f"formal debug info missing fields: {missing}")
    expected = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "steps": config["steps"],
        "strength": config["strength"],
        "guidance_scale": config["guidance_scale"],
        "inversion_mode": config["inversion_mode"],
        "warp_mode": config["warp_mode"],
        "interpolation_mode": config["latent_sampling_mode"],
        "padding_mode": config["padding_mode"],
        "align_corners": False,
        "color_transfer_mode": config["color_transfer_mode"],
        "inversion_prompt": "",
        "reconstruction_prompt": "",
    }
    for field, expected_value in expected.items():
        actual = debug_info[field]
        if isinstance(expected_value, float):
            if not math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"formal debug mismatch: {field}={actual!r}, expected {expected_value!r}"
                )
        elif actual != expected_value:
            raise RuntimeError(
                f"formal debug mismatch: {field}={actual!r}, expected {expected_value!r}"
            )
    runtime_expected = {
        "attack_device_class": "cuda",
        "attack_dtype": "torch.float16",
        "scheduler_class": "DDIMScheduler" if config["scheduler_mode"] == "ddim" else "DDPMScheduler",
    }
    for field, expected_value in runtime_expected.items():
        if debug_info.get(field) != expected_value:
            raise RuntimeError(
                f"formal debug runtime mismatch: {field}={debug_info.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    if canonical_json_hash(debug_info["scheduler_config"]) != debug_info["scheduler_config_hash"]:
        raise RuntimeError("formal debug scheduler_config_hash mismatch")
    if not debug_info.get("torch_version") or not debug_info.get("diffusers_version"):
        raise RuntimeError("formal debug package version provenance is missing")
    if not isinstance(debug_info["exact_timestep"], int):
        raise RuntimeError("formal debug exact_timestep must be an integer")
    if not debug_info["normalized_coordinate_formula"]:
        raise RuntimeError("formal debug normalized_coordinate_formula is empty")
    for field in (
        "effective_source_dx_latent",
        "effective_source_dy_latent",
        "effective_source_flow_dx_image_px",
        "effective_source_flow_dy_image_px",
        "effective_visual_shift_dx_image_px",
        "effective_visual_shift_dy_image_px",
    ):
        value = float(debug_info[field])
        if not math.isfinite(value):
            raise RuntimeError(f"formal debug {field} is non-finite")
    if planned_flow_dx_image_px is not None and not math.isclose(
        float(debug_info["planned_flow_dx_image_px"]),
        float(planned_flow_dx_image_px),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("formal debug planned x flow does not match shift plan")
    if planned_flow_dy_image_px is not None and not math.isclose(
        float(debug_info["planned_flow_dy_image_px"]),
        float(planned_flow_dy_image_px),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("formal debug planned y flow does not match shift plan")
    computed = canonical_json_hash(transform_config_payload(debug_info))
    if debug_info["transform_config_hash"] != computed:
        raise RuntimeError(
            "formal debug transform_config_hash mismatch: "
            f"stored={debug_info['transform_config_hash']} computed={computed}"
        )
    return computed


def _normalized_scalar(value: Any, default: Any) -> Any:
    if value in (None, ""):
        value = default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite provider value: {value!r}")
        return value
    return str(value)


def provider_config(method: str, row: Mapping[str, Any]) -> dict[str, Any]:
    method = method.upper()
    if method not in PROVIDER_FIELDS_BY_METHOD:
        raise ValueError(f"unsupported watermark method: {method}")
    nested = row.get("provider_config")
    if isinstance(nested, str) and nested:
        nested = json.loads(nested)
    source = nested if isinstance(nested, Mapping) else row
    defaults = PROVIDER_DEFAULTS[method]
    required = PROVIDER_REQUIRED_NONEMPTY_FIELDS.get(method, frozenset())
    missing = sorted(
        field
        for field in PROVIDER_FIELDS_BY_METHOD[method]
        if field in required and str(source.get(field, "")).strip() == ""
    )
    if missing:
        raise ValueError(
            f"{method} provider config is missing required fields: {missing}"
        )
    return {
        field: _normalized_scalar(source.get(field), defaults[field])
        for field in PROVIDER_FIELDS_BY_METHOD[method]
    }


def provider_config_hash(method: str, row: Mapping[str, Any]) -> str:
    return canonical_json_hash(provider_config(method, row))


def require_uniform_provider_config(
    method: str, rows: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, Any], str]:
    configs: dict[str, dict[str, Any]] = {}
    count = 0
    for row in rows:
        count += 1
        config = provider_config(method, row)
        configs[canonical_json_hash(config)] = config
    if count == 0:
        raise ValueError("provider cohort is empty")
    if len(configs) != 1:
        raise ValueError(
            "mixed provider configs are forbidden in one formal cohort: "
            f"{sorted(configs)}"
        )
    config_hash, config = next(iter(configs.items()))
    return config, config_hash


def require_uniform_clip_provenance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "clip_metric_type",
        "clip_model_name",
        "clip_pretrained",
        "open_clip_version",
        "preprocess_identifier",
        "text_source",
        "normalization",
        "paper_name_interpretation",
    )
    values = {
        canonical_json_hash({key: record.get(key) for key in keys}): {
            key: record.get(key) for key in keys
        }
        for record in records
    }
    if len(values) != 1:
        raise ValueError(f"mixed CLIP provenance: {sorted(values)}")
    return next(iter(values.values()))


def current_clip_provenance() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("open_clip_torch")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return {**CLIP_CONFIG, "open_clip_version": version}


def validate_resume_record(
    record: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    attack_config: Mapping[str, Any] | None = None,
) -> None:
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise RuntimeError(
                f"resume mismatch for {field}: stored={record.get(field)!r} "
                f"expected={expected_value!r}"
            )
    run_id = str(expected["run_id"])
    for path_field, hash_field in (
        ("attacked_path", "attacked_sha256"),
        ("pre_color_attacked_path", "pre_color_attacked_sha256"),
        ("debug_info_path", "debug_info_sha256"),
    ):
        path = Path(str(record.get(path_field, "")))
        if not path.is_file():
            raise RuntimeError(f"resume missing {path_field} run_id={run_id}: {path}")
        actual = sha256_path(path)
        if actual != record.get(hash_field):
            raise RuntimeError(
                f"resume {hash_field} mismatch run_id={run_id}: "
                f"stored={record.get(hash_field)!r} actual={actual}"
            )
    debug_path = Path(str(record["debug_info_path"]))
    debug_info = json.loads(debug_path.read_text(encoding="utf-8"))
    transform_hash = assert_formal_debug_info(
        debug_info,
        planned_flow_dx_image_px=float(expected["planned_flow_dx_image_px"]),
        planned_flow_dy_image_px=float(expected["planned_flow_dy_image_px"]),
        attack_config=attack_config,
    )
    if record.get("transform_config_hash") != transform_hash:
        raise RuntimeError(f"resume debug transform hash mismatch run_id={run_id}")
    for record_field, debug_field in (
        ("resolved_scheduler_config", "scheduler_config"),
        ("resolved_scheduler_config_hash", "scheduler_config_hash"),
        ("torch_version", "torch_version"),
        ("diffusers_version", "diffusers_version"),
        ("attack_device_class", "attack_device_class"),
        ("attack_dtype", "attack_dtype"),
    ):
        if record.get(record_field) != debug_info.get(debug_field):
            raise RuntimeError(
                f"resume runtime/debug mismatch run_id={run_id}: {record_field}/{debug_field}"
            )


# --------------------------------------------------------------------------- #
# Formal aggregate scalars (2026-07-27)
# --------------------------------------------------------------------------- #
# ``formal_aggregate.json`` used to record only the quality-record path/SHA and
# the raw detector payload, so the experiment table had no field to read for
# PSNR, SSIM or attack success and printed the absent marker instead. These are
# the single authoritative reducers for those scalars: the aggregate stage and
# any backfill of an already-attacked run must both call them so a table cell
# can never disagree with the per-sample records it came from.

# The formal reference for RAVEN PSNR/SSIM is the *watermarked input*, compared
# against the final post-color-transfer attacked image over the verified valid
# overlap. The explicit ``post_color_vs_watermarked_*`` fields carry that
# definition in their name; the bare ``overlap_*`` aliases are not read here so
# a future record that separates them cannot silently change the metric.
QUALITY_PSNR_FIELD = "post_color_vs_watermarked_overlap_psnr"
QUALITY_SSIM_FIELD = "post_color_vs_watermarked_overlap_ssim"
QUALITY_REFERENCE = "watermarked input"
QUALITY_GATE_SCHEMA_VERSION = "raven_quality_gate_v1"
QUALITY_FORMAL_CLASSIFICATION = "formal_quality_metric"
QUALITY_LEGACY_CLASSIFICATION = "legacy_quality_metric"
QUALITY_IDENTITY_SMOKE_CLASSIFICATION = "identity_smoke_control"
QUALITY_FLOW_FIELDS = (
    "planned_flow_dx_image_px",
    "planned_flow_dy_image_px",
    "effective_source_flow_dx_image_px",
    "effective_source_flow_dy_image_px",
)


def _finite_metric(value: Any, field: str, run_id: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite formal quality metric run_id={run_id}: {field}={value}")
    return number


def quality_gate_classification(row: Mapping[str, Any]) -> str:
    """Classify one quality row without weakening formal finite-metric gates."""
    run_id = row.get("run_id")
    psnr = float(row[QUALITY_PSNR_FIELD])
    ssim = float(row[QUALITY_SSIM_FIELD])
    if not math.isfinite(ssim):
        raise RuntimeError(
            f"non-finite formal quality metric run_id={run_id}: "
            f"{QUALITY_SSIM_FIELD}={row[QUALITY_SSIM_FIELD]}"
        )
    if math.isfinite(psnr):
        if row.get("quality_gate_schema_version") == QUALITY_GATE_SCHEMA_VERSION:
            return str(row.get("quality_gate_classification") or QUALITY_FORMAL_CLASSIFICATION)
        return QUALITY_LEGACY_CLASSIFICATION
    if not math.isinf(psnr) or psnr < 0:
        raise RuntimeError(
            f"non-finite formal quality metric run_id={run_id}: "
            f"{QUALITY_PSNR_FIELD}={row[QUALITY_PSNR_FIELD]}"
        )
    if row.get("quality_gate_schema_version") != QUALITY_GATE_SCHEMA_VERSION:
        raise RuntimeError(f"missing quality gate schema for identity PSNR run_id={run_id}")
    missing = [field for field in QUALITY_FLOW_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"missing quality gate flow fields run_id={run_id}: {missing}")
    flows = [float(row[field]) for field in QUALITY_FLOW_FIELDS]
    if not all(math.isfinite(value) for value in flows):
        raise RuntimeError(f"non-finite quality gate flow run_id={run_id}")
    if not all(math.isclose(value, 0.0, abs_tol=0.0) for value in flows):
        raise RuntimeError(f"non-identity quality row has infinite PSNR run_id={run_id}")
    if str(row.get("quality_gate_classification")) != QUALITY_IDENTITY_SMOKE_CLASSIFICATION:
        raise RuntimeError(
            f"identity PSNR row is not classified as {QUALITY_IDENTITY_SMOKE_CLASSIFICATION}"
        )
    return QUALITY_IDENTITY_SMOKE_CLASSIFICATION


def formal_quality_summary(
    quality_records_path: str | Path,
    *,
    expected_count: int,
    expected_run_ids: Iterable[Any] | None = None,
    expected_records_sha256: str | None = None,
) -> dict[str, Any]:
    """Reduce ``quality_records.jsonl`` to the aggregate PSNR/SSIM scalars.

    Fails closed on a SHA, count, run-ID, duplicate, missing-field or non-finite
    problem so a mean can never be published for a cohort that is not exactly
    the verified one.
    """
    path = Path(quality_records_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_path(path)
    if expected_records_sha256 is not None and actual_sha != expected_records_sha256:
        raise RuntimeError(
            f"quality records SHA mismatch: stored={expected_records_sha256} actual={actual_sha}"
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(expected_count):
        raise RuntimeError(
            f"quality record count mismatch: found={len(rows)} expected={expected_count}"
        )
    run_ids = [str(row["run_id"]) for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate run_id in quality records")
    if expected_run_ids is not None:
        expected_ids = {str(value) for value in expected_run_ids}
        if set(run_ids) != expected_ids:
            raise RuntimeError("quality record run-ID set differs from the verified cohort")
    references = {str(row.get("quality_reference", "")) for row in rows}
    if references != {QUALITY_REFERENCE}:
        raise RuntimeError(f"unexpected quality reference definition: {sorted(references)}")
    classifications = [quality_gate_classification(row) for row in rows]
    identity_smoke = any(item == QUALITY_IDENTITY_SMOKE_CLASSIFICATION for item in classifications)
    if identity_smoke and (len(set(classifications)) != 1 or int(expected_count) >= 1000):
        raise RuntimeError("identity smoke quality rows are not eligible for formal aggregation")
    psnr = (
        None
        if identity_smoke
        else [_finite_metric(row[QUALITY_PSNR_FIELD], QUALITY_PSNR_FIELD, row["run_id"]) for row in rows]
    )
    ssim = [_finite_metric(row[QUALITY_SSIM_FIELD], QUALITY_SSIM_FIELD, row["run_id"]) for row in rows]
    overlap_protocols = sorted({str(row.get("overlap_protocol", "")) for row in rows})
    all_current_schema = all(
        row.get("quality_gate_schema_version") == QUALITY_GATE_SCHEMA_VERSION for row in rows
    )
    return {
        "quality_count": len(rows),
        "quality_psnr_mean": None if psnr is None else math.fsum(psnr) / len(psnr),
        "quality_ssim_mean": math.fsum(ssim) / len(ssim),
        "quality_psnr_field": QUALITY_PSNR_FIELD,
        "quality_ssim_field": QUALITY_SSIM_FIELD,
        "quality_reference": QUALITY_REFERENCE,
        "quality_gate_schema_version": QUALITY_GATE_SCHEMA_VERSION if all_current_schema else None,
        "quality_gate_classification": (
            QUALITY_IDENTITY_SMOKE_CLASSIFICATION
            if identity_smoke
            else (
                QUALITY_LEGACY_CLASSIFICATION
                if QUALITY_LEGACY_CLASSIFICATION in classifications
                else QUALITY_FORMAL_CLASSIFICATION
            )
        ),
        "quality_psnr_mean_is_infinite": bool(identity_smoke),
        "quality_overlap_protocols": overlap_protocols,
        "quality_records_sha256": actual_sha,
    }


GS_ATTACK_SUCCESS_FIELD = "attack_success_rate_at_official_onebit_threshold"


def gs_attack_success_summary(detector_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attack success for GS, defined only against the official one-bit threshold.

    ``1 - official_onebit_rates["attacked"]`` — the share of attacked
    watermarked samples that the official beta-tail detector no longer flags.
    This is a different threshold family from the TR clean-calibrated TPR and
    must never be filled in from a TR-style recalibrated rate.
    """
    metric = detector_payload.get("metric")
    if not isinstance(metric, Mapping):
        metric = detector_payload
    rates = metric.get("official_onebit_rates")
    if not isinstance(rates, Mapping) or "attacked" not in rates:
        raise RuntimeError("GS detector payload has no official_onebit_rates.attacked")
    operator = metric.get("official_threshold_comparison_operator")
    if operator not in (">", ">="):
        raise RuntimeError(f"unsupported GS official threshold operator: {operator!r}")
    threshold = float(metric["official_tau_onebit"])
    attacked_rate = float(rates["attacked"])
    if not math.isfinite(attacked_rate) or not 0.0 <= attacked_rate <= 1.0:
        raise RuntimeError(f"invalid official_onebit_rates.attacked: {rates['attacked']!r}")
    if not math.isfinite(threshold):
        raise RuntimeError("non-finite official_tau_onebit")
    return {
        GS_ATTACK_SUCCESS_FIELD: 1.0 - attacked_rate,
        "attack_success_definition": (
            "1 - official_onebit_rates.attacked; share of attacked watermarked samples "
            "no longer detected at the official beta-tail one-bit threshold"
        ),
        "attack_success_threshold_type": "official_beta_tail_tau_onebit",
        "attack_success_threshold": threshold,
        "attack_success_threshold_comparison_operator": operator,
        "attack_success_detected_rate": attacked_rate,
    }


GM_ATTACK_SUCCESS_FIELD = "attack_success_rate_at_clean_calibrated_threshold"
T2S_ATTACK_SUCCESS_FIELD = "attack_success_rate_at_paired_key_comparison"


def _detector_metric(detector_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metric = detector_payload.get("metric")
    return metric if isinstance(metric, Mapping) else detector_payload


def _finite_rate(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"invalid rate for {field}: {value!r}")
    return number


def gm_attack_success_summary(detector_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attack success for GaussMarker, against its clean-calibrated threshold.

    This cohort's GM bundle has no GNR restorer and no ring classifier, so
    GaussMarker's official ensemble score and official threshold do not exist
    for it. The detector family actually used is an empirical threshold
    calibrated from this run's own clean-negative cohort on
    ``gm_raw_bit_accuracy``, and that is what this reduces. It is a different
    threshold family from both the GS official beta-tail threshold and the TR
    clean-calibrated TPR, and must never be filled in from either.
    """
    metric = _detector_metric(detector_payload)
    if str(metric.get("detector_metric")) != "gm_raw_bit_accuracy":
        raise RuntimeError(
            f"unexpected GM detector metric: {metric.get('detector_metric')!r}"
        )
    if metric.get("official_ensemble_threshold_available") is not False:
        raise RuntimeError(
            "GM aggregate expects the ensemble-threshold-unavailable family; "
            "a bundle with GNR/classifier needs its own reducer"
        )
    operator = metric.get("threshold_comparison_operator")
    if operator not in (">", ">="):
        raise RuntimeError(f"unsupported GM threshold operator: {operator!r}")
    attacked_rate = _finite_rate(
        metric["attacked_detection_rate_at_clean_calibrated_threshold"],
        "attacked_detection_rate_at_clean_calibrated_threshold",
    )
    threshold = float(metric["clean_calibrated_threshold"])
    if not math.isfinite(threshold):
        raise RuntimeError("non-finite GM clean_calibrated_threshold")
    return {
        GM_ATTACK_SUCCESS_FIELD: 1.0 - attacked_rate,
        "attack_success_definition": (
            "1 - attacked detection rate at the clean-calibrated empirical "
            "threshold on gm_raw_bit_accuracy"
        ),
        "attack_success_threshold_type": "empirical_clean_1pct_fpr",
        "attack_success_threshold": threshold,
        "attack_success_threshold_comparison_operator": operator,
        "attack_success_detected_rate": attacked_rate,
    }


def t2s_attack_success_summary(detector_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attack success for T2SMark, under its own stored per-image rule.

    T2S's stored decision rule is ``paired_key_comparison``
    (``score_true_key > score_control_key``), whose comparand is each sample's
    own control-key score rather than a cohort scalar — so the threshold is
    reported as null, not as a fabricated number. The rule is a RAVEN
    deployment extension, not a calibrated TPR at a target FPR.
    """
    metric = _detector_metric(detector_payload)
    if str(metric.get("detector_metric")) != "t2s_score_true_key":
        raise RuntimeError(
            f"unexpected T2S detector metric: {metric.get('detector_metric')!r}"
        )
    if str(metric.get("threshold_type")) != "paired_key_comparison_control_key":
        raise RuntimeError(
            f"unexpected T2S threshold family: {metric.get('threshold_type')!r}"
        )
    operator = metric.get("threshold_comparison_operator")
    if operator != ">":
        raise RuntimeError(f"unsupported T2S threshold operator: {operator!r}")
    attacked_rate = _finite_rate(
        metric["attacked_detection_rate_at_paired_key_comparison"],
        "attacked_detection_rate_at_paired_key_comparison",
    )
    return {
        T2S_ATTACK_SUCCESS_FIELD: 1.0 - attacked_rate,
        "attack_success_definition": (
            "1 - attacked detection rate under the stored T2S "
            "paired_key_comparison rule (score_true_key > score_control_key)"
        ),
        "attack_success_threshold_type": "paired_key_comparison_control_key",
        "attack_success_threshold": None,
        "attack_success_threshold_comparison_operator": operator,
        "attack_success_detected_rate": attacked_rate,
    }


FOURIER_ATTACK_SUCCESS_FIELD = "attack_success_rate_at_clean_calibrated_threshold"


def fourier_l1_attack_success_summary(detector_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attack success for RID/HSTR/HSQR canonical score families.

    The detector payload names the method-specific canonical score, which is
    ``-raw_l1`` and therefore higher-is-watermarked. The clean-calibrated
    threshold is in canonical-score space; raw L1 thresholds are never mixed into
    this family.
    """
    metric = _detector_metric(detector_payload)
    method = str(detector_payload.get("method") or "").upper()
    expected_metrics = {
        "RID": "rid_neg_channel_min_complex_l1",
        "HSTR": "hstr_score",
        "HSQR": "hsqr_score",
    }
    if method not in expected_metrics:
        raise RuntimeError(f"unsupported Fourier L1 method for attack success: {method!r}")
    if str(metric.get("detector_metric")) != expected_metrics[method]:
        raise RuntimeError(
            f"unexpected {method} detector metric: {metric.get('detector_metric')!r}"
        )
    if str(metric.get("threshold_score_space")) != "canonical_score":
        raise RuntimeError(f"{method} threshold is not explicitly in canonical_score space")
    if str(metric.get("score_direction")) != "higher_is_watermarked":
        raise RuntimeError(f"{method} canonical score direction is not higher_is_watermarked")
    operator = metric.get("threshold_comparison_operator")
    if operator not in (">", ">="):
        raise RuntimeError(f"unsupported {method} threshold operator: {operator!r}")
    attacked_rate = _finite_rate(
        metric["attacked_detection_rate_at_clean_calibrated_threshold"],
        "attacked_detection_rate_at_clean_calibrated_threshold",
    )
    threshold = float(metric["clean_calibrated_threshold"])
    if not math.isfinite(threshold):
        raise RuntimeError(f"non-finite {method} clean_calibrated_threshold")
    return {
        FOURIER_ATTACK_SUCCESS_FIELD: 1.0 - attacked_rate,
        "attack_success_definition": (
            "1 - attacked detection rate at the clean-calibrated empirical "
            f"threshold on {expected_metrics[method]} (canonical score = -raw L1)"
        ),
        "attack_success_threshold_type": "empirical_clean_1pct_fpr",
        "attack_success_threshold_score_space": "canonical_score",
        "attack_success_threshold": threshold,
        "attack_success_threshold_comparison_operator": operator,
        "attack_success_detected_rate": attacked_rate,
    }


NO_COLOR_FID_ATTACKED_DEFINITION = (
    "formal RAVEN pre-color view_guided_output attacked-watermarked images"
)


def bind_pre_color_attack_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a no-color metric record only after validating explicit pre-color provenance."""
    path = Path(str(record.get("pre_color_attacked_path", "")))
    expected_sha = str(record.get("pre_color_attacked_sha256", ""))
    if not path.is_file() or not expected_sha:
        raise RuntimeError("no-color record is missing explicit pre-color path/SHA provenance")
    actual_sha = sha256_path(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pre-color attacked SHA mismatch: stored={expected_sha} actual={actual_sha}"
        )
    return {
        **dict(record),
        "attacked_path": str(path.resolve()),
        "attacked_sha256": expected_sha,
        "fid_attacked_definition": NO_COLOR_FID_ATTACKED_DEFINITION,
    }


def stage_fid_records(
    records: Sequence[Mapping[str, Any]],
    *,
    formal_output: str | Path,
    quality_config_hash: str,
    expected_count: int,
    reference_definition: str = "original watermarked images from immutable formal snapshots",
    attacked_definition: str = "formal RAVEN final post-color-transfer attacked images",
) -> tuple[Path, dict[str, Any]]:
    fid_root = Path(formal_output) / "metrics" / "fid" / quality_config_hash
    if fid_root.exists():
        raise FileExistsError(f"refusing to reuse FID staging root: {fid_root}")
    run_ids = [str(record["run_id"]) for record in records]
    if len(run_ids) != expected_count or len(set(run_ids)) != expected_count:
        raise ValueError(
            f"FID requires {expected_count} unique completed run IDs, got {len(run_ids)}"
        )
    fid_root.mkdir(parents=True)
    reference_dir = fid_root / "reference_watermarked"
    attacked_dir = fid_root / "attacked"
    reference_dir.mkdir()
    attacked_dir.mkdir()
    manifest_rows = []
    width = max(6, max(len(run_id) for run_id in run_ids))
    for record in sorted(records, key=lambda item: int(item["run_id"])):
        run_id = str(record["run_id"])
        name = f"{int(run_id):0{width}d}.png"
        reference_source = Path(str(record["watermarked_path"])).resolve()
        attacked_source = Path(str(record["attacked_path"])).resolve()
        expected_reference_sha = str(record["watermarked_sha256"])
        expected_attacked_sha = str(record["attacked_sha256"])
        for source, expected_sha, destination in (
            (reference_source, expected_reference_sha, reference_dir / name),
            (attacked_source, expected_attacked_sha, attacked_dir / name),
        ):
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_sha = sha256_path(source)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"FID source SHA mismatch run_id={run_id}: "
                    f"expected={expected_sha} actual={actual_sha}"
                )
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            try:
                os.symlink(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            if not destination.is_file():
                raise FileNotFoundError(f"broken FID staged file: {destination}")
        manifest_rows.append(
            {
                "run_id": run_id,
                "staged_name": name,
                "reference_source_path": str(reference_source),
                "reference_source_sha256": expected_reference_sha,
                "attacked_source_path": str(attacked_source),
                "attacked_source_sha256": expected_attacked_sha,
            }
        )
    expected_names = {row["staged_name"] for row in manifest_rows}
    reference_names = {path.name for path in reference_dir.iterdir()}
    attacked_names = {path.name for path in attacked_dir.iterdir()}
    if reference_names != expected_names or attacked_names != expected_names:
        raise RuntimeError("FID staged run-ID sets do not match")
    manifest_payload = {
        "metric_name": "per_method_fid_watermarked_vs_raven",
        "image_count": expected_count,
        "reference_definition": reference_definition,
        "attacked_definition": attacked_definition,
        "quality_config_hash": quality_config_hash,
        "records": manifest_rows,
    }
    manifest_payload["manifest_hash"] = canonical_json_hash(manifest_payload)
    manifest_path = fid_root / "fid_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest_file_sha = sha256_path(manifest_path)
    with (fid_root / "fid_manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{manifest_file_sha}  fid_manifest.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest_payload["manifest_file_sha256"] = manifest_file_sha
    return fid_root, manifest_payload


def stage_no_color_fid_records(
    records: Sequence[Mapping[str, Any]],
    *,
    formal_output: str | Path,
    quality_config_hash: str,
    expected_count: int,
    reference_definition: str = "original watermarked images from immutable formal snapshots",
) -> tuple[Path, dict[str, Any]]:
    """Reuse strict FID staging with explicitly verified pre-color attack records."""
    bound_records = [bind_pre_color_attack_record(record) for record in records]
    return stage_fid_records(
        bound_records,
        formal_output=formal_output,
        quality_config_hash=quality_config_hash,
        expected_count=expected_count,
        reference_definition=reference_definition,
        attacked_definition=NO_COLOR_FID_ATTACKED_DEFINITION,
    )
