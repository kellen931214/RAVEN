#!/usr/bin/env python
"""Stream detector scores for a strict clean/watermarked/attacked manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

STAGES = ("clean", "watermarked", "attacked")
PROVENANCE_FIELDS = [
    "dataset", "run_id", "method", "prompt_id", "prompt", "source",
    "model_id", "model_revision", "vae_id", "vae_scaling_factor",
    "scheduler", "inverse_scheduler", "steps", "resolution", "detector_dtype",
    "score_direction", "provider_parameters", "generation_seed", "attack_seed",
    "watermark_seed", "fix_gt", "offset", "legacy_threshold", "target_fpr",
    "provider_config_hash", "target_watermark_hash", "source_watermark_target_sha256",
    "detector_watermark_target_sha256", "source_watermark_mask_sha256",
    "detector_watermark_mask_sha256", "gs_protocol_mode", "gs_secret_index",
    "gs_message_sha256", "gs_key_sha256", "gs_nonce_sha256",
    "gs_secret_bundle_sha256", "gs_sampling_seed", "gs_sampling_uniform_sha256",
    "gs_official_tau_onebit", "gs_official_tau_bits",
    # GaussMarker: cohort-wide bundle identity the detector reproduced.
    "gm_protocol_mode", "gm_bundle_dir", "gm_bundle_config_sha256",
    "gm_w1_file_sha256", "gm_w2_file_sha256", "gm_state_source",
    "gm_profile_is_official", "gm_report_label", "gm_score_definition",
    "gm_threshold_source", "gm_comparison_operator", "gm_gnr_used",
    "gm_classifier_used",
    # T2SMark: the per-sample portable state this row was scored against.
    "t2s_protocol_mode", "t2s_rng_mode", "t2s_inversion_mode",
    "t2s_num_inversion_steps", "t2s_watermark_id", "t2s_state_path",
    "t2s_state_sha256", "t2s_provider_config_sha256", "t2s_decision_rule",
    "t2s_score_direction",
    # Shared-clean Fourier methods: persisted bundle identity and score family.
    "rid_protocol_mode", "rid_bundle_dir", "rid_bundle_config_sha256",
    "rid_selected_pattern_sha256", "rid_mask_sha256", "rid_key_index",
    "rid_score_definition", "rid_score_direction",
    "hstr_protocol_mode", "hstr_bundle_dir", "hstr_bundle_config_sha256",
    "hstr_selected_pattern_sha256", "hstr_mask_sha256", "hstr_key_index",
    "hstr_score_definition", "hstr_score_direction",
    "hsqr_protocol_mode", "hsqr_bundle_dir", "hsqr_bundle_config_sha256",
    "hsqr_selected_pattern_sha256", "hsqr_mask_sha256", "hsqr_key_index",
    "hsqr_score_definition", "hsqr_score_direction",
]
PATH_FIELDS = [item for stage in STAGES for item in (f"{stage}_path", f"{stage}_sha256")]
SCORE_FIELDS = [
    f"{stage}_{suffix}" for stage in STAGES for suffix in (
        "raw_score", "canonical_score", "tr_log_p", "tr_sigma", "tr_lambda",
        "tr_statistic", "tr_df", "tr_p_underflow",
    )
]
GS_FIELDS = [f"{stage}_decoded_bits_sha256" for stage in STAGES]
# GM emits both of its raw domain scores per stage: the spatial-domain bit
# accuracy (the primary metric, since this bundle carries no GNR/classifier and
# therefore no ensemble score) and the frequency-domain ring L1.
GM_FIELDS = [
    f"{stage}_{suffix}"
    for stage in STAGES
    for suffix in ("gm_raw_bit_accuracy", "gm_raw_ring_l1", "gm_restored_bit_accuracy",
                   "gm_classifier_probability")
]
# T2S needs both key scores per stage: the paired_key_comparison decision is
# score_true_key > score_control_key, so the control score is not a constant and
# must be recorded per sample and per stage.
T2S_FIELDS = [
    f"{stage}_{suffix}"
    for stage in STAGES
    for suffix in ("t2s_score_true_key", "t2s_score_control_key", "t2s_score_margin",
                   "t2s_detection_success", "t2s_key_accuracy", "t2s_message_accuracy")
]
FOURIER_METHOD_FIELDS = [
    f"{stage}_{method.lower()}_{suffix}"
    for stage in STAGES
    for method in ("RID", "HSTR", "HSQR")
    for suffix in ("raw_l1", "canonical_score")
]
FIELDNAMES = (
    PROVENANCE_FIELDS + PATH_FIELDS + SCORE_FIELDS + GS_FIELDS + GM_FIELDS
    + T2S_FIELDS + FOURIER_METHOD_FIELDS + ["error"]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", required=True, choices=["GS", "TR", "GM", "T2S", "RID", "HSTR", "HSQR"]
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Strict pairing manifest CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Append after validated completed rows in an existing output")
    parser.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--scheduler", choices=["DDIM"], default="DDIM")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--min-cpu-mem-gb", type=float, default=92.0)
    parser.add_argument("--warn-cpu-mem-gb", type=float, default=110.0)
    parser.add_argument("--max-process-ram-gb", type=float, default=16.0)
    return parser


def first(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return None


def integer(row: dict[str, str], names: tuple[str, ...], default: int) -> int:
    value = first(row, *names)
    return int(value) if value is not None else default


def gs_sampling_provenance(row: dict[str, str], identifier: str) -> dict[str, str]:
    """Sampling-provenance fields this row's own GS pairing protocol defines.

    Which fields exist is a cohort property: V1 drew the watermark uniforms from
    ``gs_sampling_seed``, while V2 derives them from the shared Tree-Ring latent
    and therefore has no sampling seed at all. Requiring a fixed V1-only pair
    rejects every valid V2 cohort, and defaulting the seed would record a number
    that never existed.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def gm_bundle_manifest(row: dict[str, str], identifier: str) -> tuple[Path, dict]:
    """Load the cohort's GM bundle manifest and bind it to this row's digests.

    The detector configuration is read from the bundle rather than restated
    here, so a detector can never be constructed with copy factors, ring
    parameters or an inversion profile that differ from the ones the cohort was
    embedded with. ``GmProvider`` re-checks the same manifest through
    ``GmBundle.assert_compatible``; this is the outer binding that ties the
    bundle to the digests recorded in the source metadata.
    """
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
        if not artifact.is_file() or sha256(artifact) != str(manifest[digest_field]):
            raise RuntimeError(f"run_id={identifier}: GM bundle artifact drift: {artifact}")
    return bundle_dir, manifest


def gm_provider_kwargs(row: dict[str, str], identifier: str) -> dict:
    """Constructor kwargs derived from the bundle the cohort was embedded with."""
    bundle_dir, manifest = gm_bundle_manifest(row, identifier)
    if manifest.get("gnr_sha256") is not None or manifest.get("classifier_sha256") is not None:
        # This bundle would produce the official ensemble score; supporting it
        # means loading those checkpoints, which this cohort does not have.
        raise RuntimeError(
            f"run_id={identifier}: GM bundle declares GNR/classifier artifacts; "
            "ensemble-score detection is not wired up for this cohort"
        )
    return {
        "gm_profile": str(manifest["profile"]),
        "gm_bundle_dir": str(bundle_dir),
        "gm_create_bundle": False,
        "gm_allow_in_memory_state": False,
        "gm_torch_dtype": str(manifest["torch_dtype"]),
        "gm_channel_copy": int(manifest["channel_copy"]),
        "gm_w_copy": int(manifest["w_copy"]),
        "gm_h_copy": int(manifest["h_copy"]),
        "gm_watermark_bits_seed": manifest.get("watermark_bits_seed"),
        "gm_use_gnr": False,
        "gm_gnr_path": None,
        "gm_use_classifier": False,
        "gm_classifier_path": None,
        "modelid_target": str(manifest["model_id"]),
        "model_revision": str(manifest["model_revision"]),
        "scheduler_target": str(manifest["scheduler"]),
        "resolution": int(manifest["resolution"]),
        "w_seed": int(manifest["w_seed"]),
        "w_channel": int(manifest["w_channel"]),
        "w_pattern": str(manifest["w_pattern"]),
        "w_mask_shape": str(manifest["w_mask_shape"]),
        "w_radius": int(manifest["w_radius"]),
        "w_measurement": str(manifest["w_measurement"]),
        "w_injection": str(manifest["w_injection"]),
    }


class T2SStateDetector:
    """Per-row detector bound to one T2SMark portable state.

    T2S detection is state-bound rather than cohort-uniform: every row carries
    its own session key and message, so there is no single provider instance
    that can score the cohort. This adapter exposes the provider interface
    ``evaluate_image`` expects while delegating to the two authoritative
    standalone implementations — ``t2s_inversion.invert_image`` and
    ``T2SProvider.accuracies_for_state`` — which are the same functions
    ``eval_bench_wm/run_verification.py`` uses. No detector maths lives here.

    Constructing a real ``T2SProvider`` would be wrong as well as unnecessary:
    its ``__init__`` draws master-key/message RNG for generation, which a
    detector must never do.
    """

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
        # The state's own recorded profile wins: this cohort was embedded under
        # t2s_official / 10 steps, and scoring it under the benchmark DDIM
        # profile would be a different detector.
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



def hsqr_center_slice_mask_sha256(provider) -> str:
    from raven.eval_protocol import canonical_json_hash

    return canonical_json_hash({
        "method": "HSQR",
        "mask_identity": "center_slice_protocol",
        "center_slice": [int(provider.start), int(provider.end)],
        "watermark_channels": [int(ch) for ch in provider.watermark_channels],
        "latent_shape": [int(dim) for dim in provider.latent_shape],
        "version": 1,
    })


def _manifest_value(manifest: dict, field: str, identifier: str, method: str):
    if field not in manifest or manifest[field] in (None, ""):
        raise RuntimeError(f"run_id={identifier}: {method} bundle manifest missing {field}")
    return manifest[field]


def fourier_bundle_manifest(row: dict[str, str], identifier: str, method: str) -> tuple[Path, dict]:
    """Load and bind the shared-clean RID/HSTR/HSQR bundle named by one row."""
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
    """Load and bind this row's portable T2S state (fail closed on any drift)."""
    from utils.wm.t2s_provider import T2SWatermarkState

    state_path = Path(str(row.get("t2s_state_path", ""))).resolve()
    if not state_path.is_file():
        raise RuntimeError(f"run_id={identifier}: T2S state file not found: {state_path}")
    # ``load`` already fails closed when the embedded digest does not survive the
    # round trip; this additionally binds it to the digest the cohort recorded.
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


def provider_kwargs(method: str, row: dict[str, str]) -> dict:
    if method == "GS":
        secret_index = integer(row, ("gs_secret_index", "offset"), 0)
        kwargs = {"offset": secret_index, "gs_secret_index": secret_index}
        # Only V1 cohorts carry a GS sampling seed. Detection never uses it (the
        # target payload comes from the secret, not from the uniforms), so a
        # missing seed must stay absent rather than be faked as 0.
        sampling_seed = first(row, "gs_sampling_seed")
        if sampling_seed is not None:
            kwargs["gs_sampling_seed"] = int(sampling_seed)
        return kwargs
    if method == "TR":
        return {
            "w_seed": integer(row, ("w_seed", "watermark_seed"), 999999),
            "w_channel": integer(row, ("w_channel",), 3),
            "w_radius": integer(row, ("w_radius",), 10),
            "w_pattern": first(row, "w_pattern") or "ring",
            "w_mask_shape": first(row, "w_mask_shape") or "circle",
            "w_measurement": first(row, "w_measurement") or "l1_complex",
            "w_injection": first(row, "w_injection") or "complex",
        }
    if method == "RID":
        return {
            "rid_seed": integer(row, ("rid_seed", "watermark_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt",), 1),
            "time_shift": integer(row, ("time_shift",), 1),
        }
    if method == "HSTR":
        return {"hstr_seed": integer(row, ("hstr_seed", "watermark_seed"), 999999), "fix_gt": integer(row, ("fix_gt",), 1)}
    if method == "HSQR":
        return {
            "hsqr_seed": integer(row, ("hsqr_seed", "watermark_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt",), 1),
            "delta": integer(row, ("delta",), 0),
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


def raw_score(method: str, result: dict) -> float:
    if method == "TR":
        return float(result["p_values"][0])
    if method in {"RID", "HSTR", "HSQR"}:
        return float(result["l1_dist"][0])
    if method == "GM":
        # This cohort's bundle carries no GNR and no classifier, so the official
        # ensemble score does not exist and gm_provider correctly refuses to
        # fabricate one. The raw spatial-domain bit accuracy is GM's own primary
        # score; the frequency-domain ring L1 is recorded alongside it.
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
    # GM bit accuracy and T2S score_true_key are both higher-is-watermarked
    # already, so the canonical score is the raw score, exactly as for GS.
    return -raw if method in {"RID", "HSTR", "HSQR"} else raw


SCORE_DIRECTION_TEXT = {
    "TR": "lower raw p-value means watermark; canonical=-log10(p)",
    "GS": "higher bit accuracy means watermark",
    "GM": "higher gm_raw_bit_accuracy means watermark",
    "T2S": "higher t2s score_true_key means watermark",
}


def score_direction_text(method: str) -> str:
    return SCORE_DIRECTION_TEXT.get(method, "lower raw L1 means watermark; canonical=-L1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing score file: {args.output}")
    sys.path.insert(0, str(args.eval_repo.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
    from raven.eval_protocol import canonical_json_hash, require_uniform_provider_config
    from raven.pairing_provenance import tensor_sha256
    from utils.pipe import pipe_utils
    from utils.utils import describe_legacy_detection_threshold

    limit_cpu_threads(1)
    guard = CpuMemoryGuard(args.min_cpu_mem_gb, args.max_process_ram_gb, args.warn_cpu_mem_gb)
    guard.check("detector extraction startup")
    with args.metadata.open(newline="", encoding="utf-8-sig") as source:
        manifest_rows = list(csv.DictReader(source))
    if not manifest_rows:
        raise ValueError(f"No rows in {args.metadata}")
    method = args.method.upper()
    uniform_kwargs, uniform_hash = require_uniform_provider_config(method, manifest_rows)
    recorded_hashes = {row.get("provider_config_hash", "") for row in manifest_rows}
    if recorded_hashes not in ({""}, {uniform_hash}):
        raise ValueError(f"Manifest provider hashes do not match canonical config: {sorted(recorded_hashes)}")

    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    load_options = {"revision": args.model_revision} if args.model_revision else {}
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id, resolution=args.resolution,
        device=device, eager_loading=False, schedulers_name=args.scheduler,
        disable_tqdm=True, **load_options,
    )
    latent_shape = pipe.get_latent_shape()
    vae_scaling = float(pipe.pipe.vae.config.scaling_factor)
    inverse_scheduler = type(pipe.scheduler_inverse).__name__
    legacy = describe_legacy_detection_threshold(args.method, args.model_id)

    provider = None
    target_hash = ""
    # GS and T2S bind their detector to per-sample state, so their provider is
    # rebuilt inside the row loop. Every other method has one cohort-wide
    # detector built once here. GM is cohort-wide too, but its constructor
    # kwargs come from the bundle recorded in the rows, so it is built from the
    # first row rather than from uniform_kwargs alone.
    per_sample_provider_methods = {"GS", "T2S"}
    if method == "GM":
        first_row = manifest_rows[0]
        first_id = first(first_row, "run_id", "sample_id", "id", "index") or "0"
        provider = provider_class(method)(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device,
            **gm_provider_kwargs(first_row, str(first_id)),
        )
        if provider.bundle is None or provider.state_source != "bundle":
            raise RuntimeError(
                "GM verification requires an existing persisted bundle; "
                f"state_source={provider.state_source!r}"
            )
        target = getattr(provider, "gt_patch", None)
        target_hash = tensor_sha256(target.real.contiguous()) if target is not None else ""
    elif method == "RID":
        first_row = manifest_rows[0]
        first_id = first(first_row, "run_id", "sample_id", "id", "index") or "0"
        provider = provider_class(method)(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device,
            **rid_provider_kwargs_from_bundle(first_row, str(first_id)),
        )
        if getattr(provider, "bundle", None) is None or provider.state_source != "bundle":
            raise RuntimeError("RID verification requires an existing persisted bundle")
        target_hash = tensor_sha256(provider.gt_patch)
    elif method == "HSTR":
        first_row = manifest_rows[0]
        first_id = first(first_row, "run_id", "sample_id", "id", "index") or "0"
        provider = provider_class(method)(
            latent_shape=latent_shape,
            dtype=pipe.get_dtype(),
            device=device,
            **hstr_provider_kwargs_from_bundle(first_row, str(first_id)),
        )
        if getattr(provider, "bundle", None) is None or provider.state_source != "bundle":
            raise RuntimeError("HSTR verification requires an existing persisted bundle")
        target_hash = tensor_sha256(provider.gt_patch)
    elif method == "HSQR":
        first_row = manifest_rows[0]
        first_id = first(first_row, "run_id", "sample_id", "id", "index") or "0"
        provider = hsqr_provider_from_bundle(first_row, str(first_id), latent_shape, device)
        if getattr(provider, "bundle", None) is None:
            raise RuntimeError("HSQR verification requires an existing persisted bundle")
        target_hash = tensor_sha256(provider.gt_patch)
    elif method not in per_sample_provider_methods:
        provider = provider_class(method)(
            latent_shape=latent_shape, dtype=pipe.get_dtype(), device=device, **uniform_kwargs
        )
        target = getattr(provider, "gt_patch", None)
        target_hash = tensor_sha256(target) if target is not None else ""
    t2s_modules = None
    if method == "T2S":
        import utils.wm.t2s_provider as t2s_provider_module
        import utils.wm.t2s_inversion as t2s_inversion_module

        t2s_modules = (t2s_provider_module, t2s_inversion_module)
    processed, errors = 0, 0
    completed: set[str] = set()
    output_mode, write_header = "x", True
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as existing:
            existing_reader = csv.DictReader(existing)
            if existing_reader.fieldnames != FIELDNAMES:
                raise ValueError(f"Cannot resume {args.output}: output schema does not match")
            for existing_row in existing_reader:
                if existing_row.get("error"):
                    raise ValueError(f"Cannot resume output containing failed row: {existing_row.get('run_id')}")
                identifier = existing_row.get("run_id")
                if not identifier or identifier in completed:
                    raise ValueError(f"Cannot resume output with missing/duplicate run_id: {identifier!r}")
                completed.add(identifier)
        output_mode, write_header = "a", False
        print(f"resuming {args.output} after {len(completed)} completed rows", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(output_mode, newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            output.flush()
            os.fsync(output.fileno())
        for index, row in enumerate(manifest_rows):
            if args.limit is not None and processed >= args.limit:
                break
            identifier = first(row, "run_id", "sample_id", "id", "index") or str(index)
            if identifier in completed:
                continue
            manifest_model_id = first(row, "model_id")
            if manifest_model_id and manifest_model_id != args.model_id:
                raise ValueError(
                    f"run_id={identifier}: manifest model_id={manifest_model_id!r} "
                    f"does not match --model-id={args.model_id!r}"
                )
            record = {field: "" for field in FIELDNAMES}
            record.update({
                "dataset": first(row, "dataset", "dataset_name") or "unspecified",
                "run_id": identifier, "method": method, "prompt_id": row.get("prompt_id", ""),
                "prompt": first(row, "prompt", "caption", "text") or "", "source": row.get("source", ""),
                "model_id": args.model_id, "model_revision": args.model_revision or row.get("model_revision") or "unspecified",
                "vae_id": row.get("vae_id") or "checkpoint-default", "vae_scaling_factor": vae_scaling,
                "scheduler": args.scheduler, "inverse_scheduler": inverse_scheduler, "steps": args.steps,
                "resolution": args.resolution, "detector_dtype": str(pipe.get_dtype()),
                "score_direction": score_direction_text(method),
                "generation_seed": row.get("generation_seed", ""), "attack_seed": row.get("attack_seed", ""),
                "legacy_threshold": row.get("legacy_threshold") or legacy["threshold"], "target_fpr": args.target_fpr,
                "provider_config_hash": uniform_hash,
                "target_watermark_hash": (
                    row.get("watermark_target_sha256", "") if method == "GS" else target_hash
                ),
            })
            try:
                kwargs = dict(uniform_kwargs)
                if method == "GS":
                    kwargs.update(provider_kwargs(method, row))
                    provider = provider_class(method)(
                        latent_shape=latent_shape,
                        dtype=pipe.get_dtype(),
                        device=device,
                        **kwargs,
                    )
                    secret = provider.secret_provenance()
                    for field in (
                        "gs_secret_index", "gs_message_sha256", "gs_key_sha256",
                        "gs_nonce_sha256", "gs_secret_bundle_sha256",
                    ):
                        source_field = field
                        expected = str(row.get(source_field, ""))
                        actual = str(
                            secret[{
                                "gs_secret_index": "secret_index",
                                "gs_message_sha256": "message_sha256",
                                "gs_key_sha256": "key_sha256",
                                "gs_nonce_sha256": "nonce_sha256",
                                "gs_secret_bundle_sha256": "secret_bundle_sha256",
                            }[field]]
                        )
                        if not expected or expected != actual:
                            raise RuntimeError(
                                f"run_id={identifier}: detector/source {field} mismatch"
                            )
                        record[field] = actual
                    record["gs_protocol_mode"] = provider.gs_protocol_mode
                    record.update(gs_sampling_provenance(row, str(identifier)))
                    thresholds = provider.official_thresholds()
                    record["gs_official_tau_onebit"] = thresholds["tau_onebit"]
                    record["gs_official_tau_bits"] = thresholds["tau_bits"]
                    source_target_hash = str(row.get("watermark_target_sha256", ""))
                    detector_target_hash = tensor_sha256(provider.watermark_target_tensor())
                    source_mask_hash = str(row.get("watermark_mask_sha256", ""))
                    detector_mask_hash = canonical_json_hash(
                        {"method": "GS", "mask": "not_applicable", "version": 1}
                    )
                    if not source_target_hash or source_target_hash != detector_target_hash:
                        raise RuntimeError(
                            f"run_id={identifier}: detector/source target SHA mismatch"
                        )
                    if not source_mask_hash or source_mask_hash != detector_mask_hash:
                        raise RuntimeError(
                            f"run_id={identifier}: detector/source mask SHA mismatch"
                        )
                    record.update({
                        "source_watermark_target_sha256": source_target_hash,
                        "detector_watermark_target_sha256": detector_target_hash,
                        "source_watermark_mask_sha256": source_mask_hash,
                        "detector_watermark_mask_sha256": detector_mask_hash,
                    })
                if method == "GM":
                    # Re-bind every row to the same bundle the detector holds, so
                    # a mixed-bundle manifest cannot slip past the uniform
                    # provider-config check.
                    gm_bundle_manifest(row, str(identifier))
                    source_target_hash = str(row.get("watermark_target_sha256", ""))
                    detector_target_hash = tensor_sha256(provider.gt_patch.real.contiguous())
                    source_mask_hash = str(row.get("watermark_mask_sha256", ""))
                    detector_mask_hash = tensor_sha256(provider.watermarking_mask)
                    if not source_target_hash or source_target_hash != detector_target_hash:
                        raise RuntimeError(
                            f"run_id={identifier}: GM detector/source target SHA mismatch"
                        )
                    if not source_mask_hash or source_mask_hash != detector_mask_hash:
                        raise RuntimeError(
                            f"run_id={identifier}: GM detector/source mask SHA mismatch"
                        )
                    record.update({
                        "source_watermark_target_sha256": source_target_hash,
                        "detector_watermark_target_sha256": detector_target_hash,
                        "source_watermark_mask_sha256": source_mask_hash,
                        "detector_watermark_mask_sha256": detector_mask_hash,
                        "gm_protocol_mode": row.get("gm_protocol_mode", ""),
                        "gm_bundle_dir": row.get("gm_bundle_dir", ""),
                        "gm_bundle_config_sha256": row.get("gm_bundle_config_sha256", ""),
                        "gm_w1_file_sha256": row.get("gm_w1_file_sha256", ""),
                        "gm_w2_file_sha256": row.get("gm_w2_file_sha256", ""),
                        "gm_state_source": provider.state_source,
                        "gm_profile_is_official": provider.profile_is_official,
                    })
                if method == "T2S":
                    state = t2s_state_for_row(row, str(identifier))
                    provider = T2SStateDetector(state, *t2s_modules)
                    record.update({
                        "t2s_protocol_mode": row.get("t2s_protocol_mode", ""),
                        "t2s_rng_mode": row.get("t2s_rng_mode", ""),
                        "t2s_inversion_mode": row.get("t2s_inversion_mode", ""),
                        "t2s_num_inversion_steps": row.get("t2s_num_inversion_steps", ""),
                        "t2s_watermark_id": state.watermark_id,
                        "t2s_state_path": row.get("t2s_state_path", ""),
                        "t2s_state_sha256": state.state_sha256(),
                        "t2s_provider_config_sha256": state.provider_config_sha256,
                        # T2S has no cohort-wide target tensor: the target is the
                        # session key/message inside each row's own state, whose
                        # digest is the per-sample target hash.
                        "source_watermark_target_sha256": row.get("watermark_target_sha256", ""),
                        "detector_watermark_target_sha256": state.state_sha256(),
                    })
                if method in {"RID", "HSTR", "HSQR"}:
                    prefix = method.lower()
                    fourier_bundle_manifest(row, str(identifier), method)
                    if method in {"RID", "HSTR"}:
                        bundle_manifest = getattr(getattr(provider, "bundle", None), "manifest", {})
                        detector_target_hash = getattr(provider, "selected_pattern_sha256", bundle_manifest.get("selected_pattern_sha256", tensor_sha256(provider.gt_patch)))
                        detector_mask_hash = getattr(provider, "watermark_mask_sha256", bundle_manifest.get("mask_sha256", tensor_sha256(provider.watermarking_mask)))
                    else:
                        detector_target_hash = str(provider.bundle.manifest.get("selected_pattern_sha256", ""))
                        detector_mask_hash = getattr(provider, "watermark_mask_sha256", str(row.get("hsqr_mask_sha256", "")))
                    source_target_hash = str(row.get("watermark_target_sha256", ""))
                    source_mask_hash = str(row.get("watermark_mask_sha256", ""))
                    if not source_target_hash or source_target_hash != detector_target_hash:
                        raise RuntimeError(f"run_id={identifier}: {method} detector/source target SHA mismatch")
                    if not source_mask_hash or source_mask_hash != detector_mask_hash:
                        raise RuntimeError(f"run_id={identifier}: {method} detector/source mask SHA mismatch")
                    record.update({
                        f"{prefix}_protocol_mode": row.get(f"{prefix}_protocol_mode", ""),
                        f"{prefix}_bundle_dir": row.get(f"{prefix}_bundle_dir", ""),
                        f"{prefix}_bundle_config_sha256": row.get(f"{prefix}_bundle_config_sha256", ""),
                        f"{prefix}_selected_pattern_sha256": row.get(f"{prefix}_selected_pattern_sha256", ""),
                        f"{prefix}_mask_sha256": row.get(f"{prefix}_mask_sha256", ""),
                        f"{prefix}_key_index": row.get(f"{prefix}_key_index", ""),
                        f"{prefix}_score_definition": {
                            "RID": "rid_neg_channel_min_complex_l1",
                            "HSTR": "hstr_score=-min(channel_0_l1,channel_3_l1)",
                            "HSQR": "hsqr_negative_mean_complex_l1_distance",
                        }[method],
                        f"{prefix}_score_direction": "higher_is_watermarked",
                        "source_watermark_target_sha256": source_target_hash,
                        "detector_watermark_target_sha256": detector_target_hash,
                        "source_watermark_mask_sha256": source_mask_hash,
                        "detector_watermark_mask_sha256": detector_mask_hash,
                    })
                record["provider_parameters"] = json.dumps(uniform_kwargs, sort_keys=True)
                record["watermark_seed"] = next((kwargs[key] for key in ("w_seed", "rid_seed", "hstr_seed", "hsqr_seed") if key in kwargs), "")
                record["fix_gt"], record["offset"] = kwargs.get("fix_gt", ""), kwargs.get("offset", "")
                stage_results = {}
                for stage in STAGES:
                    path = Path(row[f"{stage}_path"]).resolve()
                    record[f"{stage}_path"] = str(path)
                    actual_sha = sha256(path)
                    expected_sha = row.get(f"{stage}_sha256")
                    if expected_sha and expected_sha != actual_sha:
                        raise RuntimeError(f"run_id={identifier}: {stage} SHA mismatch")
                    record[f"{stage}_sha256"] = actual_sha
                    guard.check(f"{method}/{identifier}/{stage}")
                    result = evaluate_image(torch, provider, pipe, path, args.steps)
                    stage_results[stage] = result
                    raw = raw_score(method, result)
                    record[f"{stage}_raw_score"] = raw
                    record[f"{stage}_canonical_score"] = canonical_score(method, raw, result)
                    if method == "TR":
                        add_tr_diagnostics(record, stage, result)
                    if method == "GM":
                        for field in (
                            "gm_raw_bit_accuracy", "gm_raw_ring_l1",
                            "gm_restored_bit_accuracy", "gm_classifier_probability",
                        ):
                            value = result.get(field)
                            record[f"{stage}_{field}"] = "" if value is None else value
                    if method == "T2S":
                        for field in (
                            "t2s_score_true_key", "t2s_score_control_key",
                            "t2s_score_margin", "t2s_key_accuracy",
                        ):
                            value = result.get(field)
                            record[f"{stage}_{field}"] = "" if value is None else value
                        message_accuracy = result.get("message_accuracy")
                        record[f"{stage}_t2s_message_accuracy"] = (
                            "" if message_accuracy is None else message_accuracy
                        )
                        record[f"{stage}_t2s_detection_success"] = bool(
                            result["detection_success"]
                        )
                    if method in {"RID", "HSTR", "HSQR"}:
                        prefix = method.lower()
                        record[f"{stage}_{prefix}_raw_l1"] = raw
                        record[f"{stage}_{prefix}_canonical_score"] = record[f"{stage}_canonical_score"]
                if method == "GS":
                    for stage in STAGES:
                        decoded = stage_results[stage]["message_bits_str_list"][0]
                        record[f"{stage}_decoded_bits_sha256"] = hashlib.sha256(
                            decoded.encode("ascii")
                        ).hexdigest()
                if method == "GM":
                    # Detector self-description, taken from the detector itself
                    # rather than restated, and required to be identical across
                    # the three stages of one row.
                    for field in (
                        "gm_report_label", "gm_score_definition",
                        "gm_threshold_source", "gm_comparison_operator",
                    ):
                        values = {str(stage_results[stage][field]) for stage in STAGES}
                        if len(values) != 1:
                            raise RuntimeError(
                                f"run_id={identifier}: GM {field} differs between stages: "
                                f"{sorted(values)}"
                            )
                        record[field] = next(iter(values))
                    record["gm_gnr_used"] = bool(stage_results["watermarked"]["gm_used_gnr"])
                    record["gm_classifier_used"] = bool(
                        stage_results["watermarked"]["gm_used_classifier"]
                    )
                if method == "T2S":
                    for field, column in (
                        ("decision_rule", "t2s_decision_rule"),
                        ("score_direction", "t2s_score_direction"),
                    ):
                        values = {str(stage_results[stage][field]) for stage in STAGES}
                        if len(values) != 1:
                            raise RuntimeError(
                                f"run_id={identifier}: T2S {field} differs between stages: "
                                f"{sorted(values)}"
                            )
                        record[column] = next(iter(values))
                del stage_results
            except Exception as exc:
                errors += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(record)
            output.flush()
            os.fsync(output.fileno())
            processed += 1
            print(f"[{method}] {processed} run_id={identifier} error={record['error'] or 'none'}", flush=True)
    print(f"wrote {args.output} rows={processed} errors={errors}", flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
