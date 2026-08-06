"""Canonical detector scoring helpers shared across all 7 watermark methods.

Production detectors import individual helpers from this module.  The
original standalone CLI (``raven_repro/scripts/extract_verification_scores.py``)
has been removed; its library functions live here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageOps

# Ensure raven_repro/ and eval_bench_wm/ are importable.
_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "raven_repro"), str(_REPO / "eval_bench_wm")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── tiny utilities (no external deps) ──────────────────────────────────────

def _first(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _integer(row: dict[str, str], names: tuple[str, ...], default: int) -> int:
    value = _first(row, *names)
    return int(value) if value is not None else default


from raven.eval_protocol import sha256_path as _sha256


# ── GS-specific helpers ────────────────────────────────────────────────────

def gs_sampling_provenance(row: dict[str, str], identifier: str) -> dict[str, str]:
    from raven.pairing_provenance import gs_fields_for_protocol

    protocol = str(row.get("protocol", ""))
    protocol_fields = gs_fields_for_protocol(protocol)
    fields = [
        field
        for field in ("gs_sampling_seed", "gs_sampling_uniform_sha256")
        if field in protocol_fields
    ]
    if "gs_sampling_uniform_sha256" not in fields:
        raise RuntimeError(
            f"run_id={identifier}: GS protocol {protocol!r} defines no sampling "
            "uniform provenance"
        )
    resolved: dict[str, str] = {}
    for field in fields:
        if not str(row.get(field, "")):
            raise RuntimeError(f"run_id={identifier}: missing {field}")
        resolved[field] = row[field]
    return resolved


# ── GM helpers ──────────────────────────────────────────────────────────────

def gm_bundle_manifest(row: dict[str, str], identifier: str) -> tuple[Path, dict]:
    bundle_dir = Path(str(row.get("gm_bundle_dir", ""))).resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"run_id={identifier}: GM bundle manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for manifest_field, row_field in (
        ("bundle_config_sha256", "gm_bundle_config_sha256"),
        ("w1_file_sha256", "gm_w1_file_sha256"),
        ("w2_file_sha256", "gm_w2_file_sha256"),
        ("m_sha256", "gm_m_sha256"),
        ("watermark_sha256", "gm_watermark_sha256"),
        ("w2_tensor_sha256", "gm_target_sha256"),
    ):
        expected = str(row.get(row_field, ""))
        actual = str(manifest.get(manifest_field, ""))
        if not expected or expected != actual:
            raise RuntimeError(
                f"run_id={identifier}: GM bundle/source {row_field} mismatch: "
                f"source={expected!r} bundle={actual!r}"
            )
    for digest_field, file_name in (
        ("w1_file_sha256", "w1.pth"),
        ("w2_file_sha256", "w2.pth"),
    ):
        artifact = bundle_dir / file_name
        if not artifact.is_file() or _sha256(artifact) != str(manifest[digest_field]):
            raise RuntimeError(f"run_id={identifier}: GM bundle artifact drift: {artifact}")
    return bundle_dir, manifest


def gm_provider_kwargs(row: dict[str, str], identifier: str) -> dict:
    import math as _math

    bundle_dir, manifest = gm_bundle_manifest(row, identifier)
    if manifest.get("gnr_sha256") is not None or manifest.get("classifier_sha256") is not None:
        raise RuntimeError(
            f"run_id={identifier}: GM bundle declares GNR/classifier artifacts; "
            "ensemble-score detection is not wired up for this cohort"
        )

    def _require_str(key: str) -> str:
        val = manifest.get(key)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest missing or empty {key!r}")
        if not isinstance(val, str):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest {key!r} must be str, got {type(val).__name__}")
        return val

    def _require_int(key: str) -> int:
        val = manifest.get(key)
        if val is None:
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest missing {key!r}")
        if isinstance(val, bool) or not isinstance(val, int):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest {key!r} must be int, got {type(val).__name__}: {val!r}")
        return int(val)

    def _require_float(key: str) -> float:
        val = manifest.get(key)
        if val is None:
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest missing {key!r}")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest {key!r} must be numeric, got {type(val).__name__}: {val!r}")
        f = float(val)
        if not _math.isfinite(f):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest {key!r} must be finite")
        return f

    def _require_bool(key: str) -> bool:
        val = manifest.get(key)
        if val is None:
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest missing {key!r}")
        if not isinstance(val, bool):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest {key!r} must be bool, got {type(val).__name__}: {val!r}")
        return val

    def _optional_strict_int(key: str) -> int | None:
        val = manifest.get(key)
        if val is None:
            return None
        if isinstance(val, bool) or not isinstance(val, int):
            raise RuntimeError(f"run_id={identifier}: GM bundle manifest {key!r} must be int or None, got {type(val).__name__}: {val!r}")
        return val

    prompt_sha = _require_str("inversion_prompt_sha256")
    empty_sha = hashlib.sha256(b"").hexdigest()
    if prompt_sha != empty_sha:
        raise RuntimeError(
            f"run_id={identifier}: GM bundle manifest inversion_prompt_sha256 "
            f"is non-empty, which this detector does not support: {prompt_sha!r} (expected {empty_sha!r})"
        )
    inversion_guidance_val = _require_float("inversion_guidance_scale")
    if not (0.0 < inversion_guidance_val <= 100.0) or not _math.isfinite(inversion_guidance_val):
        raise RuntimeError(f"run_id={identifier}: GM bundle manifest inversion_guidance_scale out of range: {inversion_guidance_val!r}")
    vae_scaling_val = _require_float("vae_scaling_factor")
    if vae_scaling_val <= 0.0 or not _math.isfinite(vae_scaling_val):
        raise RuntimeError(f"run_id={identifier}: GM bundle manifest vae_scaling_factor must be > 0: {vae_scaling_val!r}")

    return {
        "gm_profile": _require_str("profile"),
        "gm_bundle_dir": str(bundle_dir),
        "gm_create_bundle": False,
        "gm_allow_in_memory_state": False,
        "gm_torch_dtype": _require_str("torch_dtype"),
        "gm_channel_copy": _require_int("channel_copy"),
        "gm_w_copy": _require_int("w_copy"),
        "gm_h_copy": _require_int("h_copy"),
        "gm_watermark_bits_seed": _optional_strict_int("watermark_bits_seed"),
        "gm_use_gnr": False,
        "gm_gnr_path": None,
        "gm_model_nf": _require_int("model_nf"),
        "gm_classifier_type": _require_int("classifier_type"),
        "gm_use_classifier": False,
        "gm_classifier_path": None,
        "modelid_target": _require_str("model_id"),
        "model_revision": _require_str("model_revision"),
        "scheduler_target": _require_str("scheduler"),
        "resolution": _require_int("resolution"),
        "gm_inversion_guidance": inversion_guidance_val,
        "gm_inversion_steps": _require_int("inversion_steps"),
        "gm_inversion_seed": 0,
        "gm_inversion_prompt": "",
        "gm_vae_sample": _require_bool("vae_sample"),
        "gm_vae_scaling_factor": vae_scaling_val,
        "gm_profile_is_official": _require_bool("profile_is_official"),
        "w_seed": _require_int("w_seed"),
        "w_channel": _require_int("w_channel"),
        "w_pattern": _require_str("w_pattern"),
        "w_mask_shape": _require_str("w_mask_shape"),
        "w_radius": _require_int("w_radius"),
        "w_measurement": _require_str("w_measurement"),
        "w_injection": _require_str("w_injection"),
    }


# ── T2S adapter ─────────────────────────────────────────────────────────────

class T2SStateDetector:
    """Per-row detector bound to one T2SMark portable state."""

    def __init__(self, state, provider_module, inversion_module):
        self.state = state
        self._provider_module = provider_module
        self._invert_image = inversion_module.invert_image

    def get_wm_type(self) -> str:
        return "T2S"

    def invert_images(self, images, pipe_provider_target=None, num_inference_steps=None, **_):
        if pipe_provider_target is None:
            raise ValueError("T2S inversion requires pipe_provider_target")
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError("T2S inversion supports a single image per call")
            images = images[0]
        zT = self._invert_image(
            pipe_provider_target,
            images,
            inversion_mode=self.state.inversion_mode,
            num_inversion_steps=self.state.num_inversion_steps,
            benchmark_num_inference_steps=self.state.num_inference_steps,
        )
        return {"zT_torch": zT}

    def get_accuracies(self, reversed_latents):
        return self._provider_module.T2SProvider.accuracies_for_state(
            self.state, reversed_latents
        )


# ── Fourier helpers ─────────────────────────────────────────────────────────

def _manifest_value(manifest: dict, field: str, identifier: str, method: str):
    if field not in manifest or manifest[field] in (None, ""):
        raise RuntimeError(f"run_id={identifier}: {method} bundle manifest missing {field}")
    return manifest[field]


def fourier_bundle_manifest(row: dict[str, str], identifier: str, method: str) -> tuple[Path, dict]:
    prefix = method.lower()
    bundle_dir = Path(str(row.get(f"{prefix}_bundle_dir", ""))).resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"run_id={identifier}: {method} bundle manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = [
        (f"{prefix}_bundle_config_sha256", "bundle_config_sha256"),
        (f"{prefix}_selected_pattern_sha256", "selected_pattern_sha256"),
    ]
    if "mask_sha256" in manifest:
        checks.append((f"{prefix}_mask_sha256", "mask_sha256"))
    for row_field, manifest_field in checks:
        expected = str(row.get(row_field, ""))
        actual = str(manifest.get(manifest_field, ""))
        if not expected or expected != actual:
            raise RuntimeError(
                f"run_id={identifier}: {method} bundle/source {row_field} mismatch: "
                f"source={expected!r} bundle={actual!r}"
            )
    return bundle_dir, manifest


def rid_provider_kwargs_from_bundle(row: dict[str, str], identifier: str) -> dict:
    bundle_dir, manifest = fourier_bundle_manifest(row, identifier, "RID")
    return {
        "rid_profile": _manifest_value(manifest, "profile_name", identifier, "RID"),
        "rid_bundle_dir": str(bundle_dir),
        "rid_create_bundle": False,
        "rid_key_index": int(_manifest_value(manifest, "selected_key_index", identifier, "RID")),
        "rid_key_seed": int(_manifest_value(manifest, "rng_seed", identifier, "RID")),
        "rid_key_rng_device": str(_manifest_value(manifest, "rng_device", identifier, "RID")),
        "rid_key_rng_dtype": str(_manifest_value(manifest, "rng_dtype", identifier, "RID")),
        "channel_min": int(_manifest_value(manifest, "channel_min", identifier, "RID")),
        "ring_value_range": int(_manifest_value(manifest, "ring_value_range", identifier, "RID")),
        "quantization_levels": int(_manifest_value(manifest, "quantization_levels", identifier, "RID")),
        "ring_width": int(_manifest_value(manifest, "ring_width", identifier, "RID")),
        "assigned_keys": int(_manifest_value(manifest, "assigned_keys", identifier, "RID")),
        "fix_gt": int(_manifest_value(manifest, "fix_gt", identifier, "RID")),
        "time_shift": int(_manifest_value(manifest, "spatial_shift", identifier, "RID")),
        "time_shift_factor": float(_manifest_value(manifest, "spatial_shift_factor", identifier, "RID")),
        "rid_shift_semantics": str(_manifest_value(manifest, "spatial_shift_factor_semantics", identifier, "RID")),
        "rid_torch_dtype": str(_manifest_value(manifest, "torch_dtype", identifier, "RID")),
        "rid_inversion_prompt": "",
        "rid_inversion_guidance": float(_manifest_value(manifest, "inversion_guidance_scale", identifier, "RID")),
        "rid_inversion_steps": int(_manifest_value(manifest, "inversion_steps", identifier, "RID")),
        "rid_vae_sample": bool(_manifest_value(manifest, "vae_sample", identifier, "RID")),
        "rid_vae_scaling_factor": float(_manifest_value(manifest, "vae_scaling_factor", identifier, "RID")),
        "rid_profile_is_official": manifest.get("profile_is_official"),
        "rid_profile_overrides": manifest.get("profile_overrides") or {},
        "modelid_target": str(_manifest_value(manifest, "model_id", identifier, "RID")),
        "model_revision": str(_manifest_value(manifest, "model_revision", identifier, "RID")),
        "scheduler_target": str(_manifest_value(manifest, "scheduler", identifier, "RID")),
        "resolution": int(_manifest_value(manifest, "resolution", identifier, "RID")),
    }


def hstr_provider_kwargs_from_bundle(row: dict[str, str], identifier: str) -> dict:
    bundle_dir, manifest = fourier_bundle_manifest(row, identifier, "HSTR")
    return {
        "hstr_profile": str(_manifest_value(manifest, "profile_name", identifier, "HSTR")),
        "hstr_bundle_dir": str(bundle_dir),
        "hstr_create_bundle": False,
        "hstr_key_index": int(_manifest_value(manifest, "selected_key_index", identifier, "HSTR")),
        "hstr_rng_device": str(_manifest_value(manifest, "rng_device", identifier, "HSTR")),
        "latent_channel": int(_manifest_value(manifest, "latent_shape", identifier, "HSTR")[1]),
        "hw_latent": int(_manifest_value(manifest, "latent_shape", identifier, "HSTR")[2]),
        "start": int(_manifest_value(manifest, "center_slice", identifier, "HSTR")[0]),
        "end": int(_manifest_value(manifest, "center_slice", identifier, "HSTR")[1]),
        "wm_capacity": int(_manifest_value(manifest, "wm_capacity", identifier, "HSTR")),
        "modelid_target": str(_manifest_value(manifest, "model_id", identifier, "HSTR")),
        "model_revision": str(_manifest_value(manifest, "model_revision", identifier, "HSTR")),
        "scheduler_target": str(_manifest_value(manifest, "scheduler_type", identifier, "HSTR")),
        "resolution": int(_manifest_value(manifest, "resolution", identifier, "HSTR")),
    }


def hsqr_provider_from_bundle(row: dict[str, str], identifier: str, latent_shape, device):
    from utils.wm import sfw_bundle
    from utils.wm.hsqr_provider import HSQRProvider

    bundle_dir, _manifest = fourier_bundle_manifest(row, identifier, "HSQR")
    bundle = sfw_bundle.SfwBundle.load(bundle_dir)
    return HSQRProvider.from_bundle(bundle, latent_shape=latent_shape, device=device)


def t2s_state_for_row(row: dict[str, str], identifier: str):
    from utils.wm.t2s_provider import T2SWatermarkState

    state_path = Path(str(row.get("t2s_state_path", ""))).resolve()
    if not state_path.is_file():
        raise RuntimeError(f"run_id={identifier}: T2S state file not found: {state_path}")
    state = T2SWatermarkState.load(state_path)
    recorded = str(row.get("t2s_state_sha256", ""))
    if not recorded or recorded != state.state_sha256():
        raise RuntimeError(
            f"run_id={identifier}: T2S state SHA mismatch: "
            f"source={recorded!r} state={state.state_sha256()!r}"
        )
    recorded_id = str(row.get("t2s_watermark_id", ""))
    if not recorded_id or recorded_id != state.watermark_id:
        raise RuntimeError(
            f"run_id={identifier}: T2S watermark_id mismatch: "
            f"source={recorded_id!r} state={state.watermark_id!r}"
        )
    for field, row_field in (
        ("rng_mode", "t2s_rng_mode"),
        ("inversion_mode", "t2s_inversion_mode"),
        ("provider_config_sha256", "t2s_provider_config_sha256"),
    ):
        expected = str(row.get(row_field, ""))
        actual = str(getattr(state, field))
        if not expected or expected != actual:
            raise RuntimeError(
                f"run_id={identifier}: T2S {row_field} mismatch: "
                f"source={expected!r} state={actual!r}"
            )
    if int(row["t2s_num_inversion_steps"]) != int(state.num_inversion_steps):
        raise RuntimeError(f"run_id={identifier}: T2S num_inversion_steps mismatch")
    return state


# ── Provider dispatch ───────────────────────────────────────────────────────

def provider_kwargs(method: str, row: dict[str, str]) -> dict:
    if method == "GS":
        secret_index = _integer(row, ("gs_secret_index", "offset"), 0)
        kwargs = {"offset": secret_index, "gs_secret_index": secret_index}
        sampling_seed = _first(row, "gs_sampling_seed")
        if sampling_seed is not None:
            kwargs["gs_sampling_seed"] = int(sampling_seed)
        return kwargs
    if method == "TR":
        return {
            "w_seed": _integer(row, ("w_seed", "watermark_seed"), 999999),
            "w_channel": _integer(row, ("w_channel",), 3),
            "w_radius": _integer(row, ("w_radius",), 10),
            "w_pattern": _first(row, "w_pattern") or "ring",
            "w_mask_shape": _first(row, "w_mask_shape") or "circle",
            "w_measurement": _first(row, "w_measurement") or "l1_complex",
            "w_injection": _first(row, "w_injection") or "complex",
        }
    if method == "RID":
        return {
            "rid_seed": _integer(row, ("rid_seed", "watermark_seed"), 999999),
            "fix_gt": _integer(row, ("fix_gt",), 1),
            "time_shift": _integer(row, ("time_shift",), 1),
        }
    if method == "HSTR":
        return {"hstr_seed": _integer(row, ("hstr_seed", "watermark_seed"), 999999), "fix_gt": _integer(row, ("fix_gt",), 1)}
    if method == "HSQR":
        return {
            "hsqr_seed": _integer(row, ("hsqr_seed", "watermark_seed"), 999999),
            "fix_gt": _integer(row, ("fix_gt",), 1),
            "delta": _integer(row, ("delta",), 0),
        }
    raise ValueError(method)


def provider_class(method: str):
    if method == "TR":
        from utils.wm.tr_provider import TrProvider
        return TrProvider
    if method == "GS":
        from utils.wm.gs_provider import GsProvider
        return GsProvider
    if method == "GM":
        from utils.wm.gm_provider import GmProvider
        return GmProvider
    if method == "RID":
        from utils.wm.ringid_provider import RingIDProvider
        return RingIDProvider
    if method == "HSTR":
        from utils.wm.hstr_provider import HSTRProvider
        return HSTRProvider
    if method == "HSQR":
        from utils.wm.hsqr_provider import HSQRProvider
        return HSQRProvider
    raise ValueError(method)


# ── Score computation ───────────────────────────────────────────────────────

def raw_score(method: str, result: dict) -> float:
    if method == "TR":
        return float(result["p_values"][0])
    if method in {"RID", "HSTR", "HSQR"}:
        return float(result["l1_dist"][0])
    if method == "GM":
        value = result.get("gm_raw_bit_accuracy")
        if value is None:
            raise RuntimeError("GM detector returned no gm_raw_bit_accuracy")
        return float(value)
    if method == "T2S":
        return float(result["t2s_score_true_key"])
    return float(result["bit_accuracies"][0])


def canonical_score(method: str, raw: float, result: dict) -> float:
    if method == "TR":
        diagnostics = result.get("p_value_diagnostics") or []
        log_p = float(diagnostics[0].get("log_p", float("nan"))) if diagnostics else float("nan")
        return -log_p / math.log(10.0) if math.isfinite(log_p) else -math.log10(max(raw, sys.float_info.min))
    return -raw if method in {"RID", "HSTR", "HSQR"} else raw


SCORE_DIRECTION_TEXT = {
    "TR": "lower raw p-value means watermark; canonical=-log10(p)",
    "GS": "higher bit accuracy means watermark",
    "GM": "higher gm_raw_bit_accuracy means watermark",
    "T2S": "higher t2s score_true_key means watermark",
}


def score_direction_text(method: str) -> str:
    return SCORE_DIRECTION_TEXT.get(method, "lower raw L1 means watermark; canonical=-L1")


# ── Image evaluation ────────────────────────────────────────────────────────

def evaluate_image(torch, provider, pipe, path: Path, steps: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    with torch.no_grad():
        inversion = provider.invert_images(image, pipe_provider_target=pipe, num_inference_steps=steps) \
            if hasattr(provider, "invert_images") else pipe.invert_images(image, num_inference_steps=steps)
        recovered = inversion["zT_torch"]
        result = provider.get_accuracies(recovered)
        if provider.get_wm_type() == "TR":
            import scipy.stats

            diagnostics = []
            recovered_fft = torch.fft.fftshift(torch.fft.fft2(recovered), dim=(-1, -2))
            mask = provider.watermarking_mask[0]
            target = provider.gt_patch[0][mask].flatten()
            target = torch.concatenate([target.real, target.imag])
            for latent_fft in recovered_fft:
                observed = latent_fft[mask].flatten()
                observed = torch.concatenate([observed.real, observed.imag])
                sigma = observed.std()
                noncentrality = (target.square() / sigma.square()).sum().item()
                statistic = (((observed - target) / sigma).square()).sum().item()
                raw_log_p = float(scipy.stats.ncx2.logcdf(statistic, df=len(target), nc=noncentrality))
                p_value = float(result["p_values"][len(diagnostics)])
                p_underflow = p_value == 0.0 or not math.isfinite(raw_log_p)
                log_p = raw_log_p if math.isfinite(raw_log_p) else math.log(sys.float_info.min)
                diagnostics.append({
                    "log_p": log_p, "sigma": float(sigma.item()), "lambda": noncentrality,
                    "statistic": statistic, "df": len(target), "p_underflow": p_underflow,
                })
            result["p_value_diagnostics"] = diagnostics
    del inversion
    return result


def add_tr_diagnostics(record: dict, stage: str, result: dict) -> None:
    diagnostics = result.get("p_value_diagnostics") or []
    if not diagnostics:
        return
    item = diagnostics[0]
    mapping = {
        "log_p": "tr_log_p", "sigma": "tr_sigma", "lambda": "tr_lambda",
        "statistic": "tr_statistic", "df": "tr_df", "p_underflow": "tr_p_underflow",
    }
    for source, destination in mapping.items():
        record[f"{stage}_{destination}"] = item.get(source, "")
