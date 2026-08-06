"""Fail-closed protocol constants and field definitions for watermark detectors.

Every protocol string, field tuple, and hash identity is preserved byte-for-byte
from the original ``pairing_provenance.py``.  Generation-only audit functions
and shared-clean bundle verifiers live in ``generate/provenance.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from raven.protocol import sha256_path

# --------------------------------------------------------------------------- #
# Protocol strings
# --------------------------------------------------------------------------- #
TR_PAIRING_PROTOCOL = "tree_ring_paired_base_latent_v1"
GS_PAIRING_PROTOCOL = "gaussian_shading_shared_uniform_v1"
GS_SHARED_TR_CLEAN_PROTOCOL = "gaussian_shading_shared_tr_clean_v2"
SHARED_CLEAN_PROTOCOL = "tr_canonical_clean_latent_v2"
GS_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
SHARED_CLEAN_SOURCE_METHOD = "TR"
GS_UNIFORM_DERIVATION = "normal_cdf_of_tr_float32_base_latent"

GM_SHARED_TR_CLEAN_PROTOCOL = "gaussmarker_shared_tr_clean_v2"
T2S_SHARED_TR_CLEAN_PROTOCOL = "t2smark_shared_tr_clean_v2"
RID_SHARED_TR_CLEAN_PROTOCOL = "ringid_shared_tr_clean_v2"
HSTR_SHARED_TR_CLEAN_PROTOCOL = "hstr_shared_tr_clean_v2"
HSQR_SHARED_TR_CLEAN_PROTOCOL = "hsqr_shared_tr_clean_v2"

GM_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
T2S_SHARED_TR_CLEAN_MODE = "official_encoder_shared_tr_clean"
RID_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
HSTR_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
HSQR_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
GM_UNIFORM_DERIVATION = "normal_cdf_of_tr_float32_base_latent"

PAIRING_PROTOCOL = TR_PAIRING_PROTOCOL
PAIRING_PROTOCOLS = {"TR": TR_PAIRING_PROTOCOL, "GS": GS_PAIRING_PROTOCOL}
ALLOWED_PAIRING_PROTOCOLS = {
    "TR": (TR_PAIRING_PROTOCOL,),
    "GS": (GS_PAIRING_PROTOCOL, GS_SHARED_TR_CLEAN_PROTOCOL),
    "GM": (GM_SHARED_TR_CLEAN_PROTOCOL,),
    "T2S": (T2S_SHARED_TR_CLEAN_PROTOCOL,),
    "RID": (RID_SHARED_TR_CLEAN_PROTOCOL,),
    "HSTR": (HSTR_SHARED_TR_CLEAN_PROTOCOL,),
    "HSQR": (HSQR_SHARED_TR_CLEAN_PROTOCOL,),
}

SINGLE_TARGET_METHODS = frozenset({"TR", "GM", "RID", "HSTR", "HSQR"})
PER_SAMPLE_TARGET_METHODS = frozenset({"GS", "T2S"})

# --------------------------------------------------------------------------- #
# GS field tuples
# --------------------------------------------------------------------------- #
GS_REQUIRED_FIELDS = (
    "gs_protocol_mode",
    "gs_secret_index",
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "gs_sampling_seed",
    "gs_sampling_uniform_sha256",
    "gs_payload_layout",
    "gs_cipher",
)
GS_CORE_FIELDS = tuple(f for f in GS_REQUIRED_FIELDS if f != "gs_sampling_seed")
GS_SHARED_CLEAN_V2_FIELDS = (
    "shared_clean_protocol",
    "shared_clean_source_method",
    "shared_clean_source_metadata_path",
    "shared_clean_source_metadata_sha256",
    "shared_clean_sample_sha256",
    "gs_uniform_derivation",
    "tr_base_latent_sha256",
    "tr_clean_path",
    "tr_clean_sha256",
)
GS_V2_REQUIRED_FIELDS = GS_CORE_FIELDS + GS_SHARED_CLEAN_V2_FIELDS
GS_V2_PAIRING_FIELDS = tuple(
    field
    for field in GS_V2_REQUIRED_FIELDS
    if field not in {"shared_clean_source_metadata_path", "tr_clean_path"}
)

# --------------------------------------------------------------------------- #
# Shared-clean common fields
# --------------------------------------------------------------------------- #
SHARED_CLEAN_COMMON_FIELDS = (
    "shared_clean_protocol",
    "shared_clean_source_method",
    "shared_clean_source_metadata_path",
    "shared_clean_source_metadata_sha256",
    "shared_clean_sample_sha256",
    "watermark_pre_injection_base_latent_sha256",
    "tr_base_latent_sha256",
    "tr_clean_path",
    "tr_clean_sha256",
    "watermarked_sha256",
)

SHARED_CLEAN_PATH_FIELDS = frozenset(
    {
        "shared_clean_source_metadata_path",
        "tr_clean_path",
        "gm_bundle_dir",
        "t2s_state_path",
        "rid_bundle_dir",
        "hstr_bundle_dir",
        "hsqr_bundle_dir",
    }
)

# --------------------------------------------------------------------------- #
# GM field tuples
# --------------------------------------------------------------------------- #
GM_CORE_FIELDS = (
    "gm_protocol_mode",
    "gm_uniform_derivation",
    "gm_state_source",
    "gm_bundle_dir",
    "gm_bundle_config_sha256",
    "gm_w1_file_sha256",
    "gm_w2_file_sha256",
    "gm_watermark_sha256",
    "gm_m_sha256",
    "gm_target_sha256",
    "gm_mask_sha256",
    "gm_pre_injection_latent_sha256",
    "gm_post_injection_latent_sha256",
    "gm_sampling_uniform_sha256",
    "gm_provider_entrypoint_sha256",
)
GM_REQUIRED_FIELDS = GM_CORE_FIELDS + SHARED_CLEAN_COMMON_FIELDS
GM_PAIRING_FIELDS = tuple(f for f in GM_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)

# --------------------------------------------------------------------------- #
# T2S field tuples
# --------------------------------------------------------------------------- #
T2S_CORE_FIELDS = (
    "t2s_protocol_mode",
    "t2s_rng_mode",
    "t2s_inversion_mode",
    "t2s_watermark_id",
    "t2s_state_path",
    "t2s_state_sha256",
    "t2s_provider_config_sha256",
    "t2s_base_latent_sha256",
    "t2s_abs_magnitude_sha256",
    "t2s_master_key_sha256",
    "t2s_session_key_sha256",
    "t2s_message_sha256",
    "t2s_provider_entrypoint_sha256",
)
T2S_REQUIRED_FIELDS = T2S_CORE_FIELDS + SHARED_CLEAN_COMMON_FIELDS
T2S_PAIRING_FIELDS = tuple(f for f in T2S_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)

# --------------------------------------------------------------------------- #
# Fourier (RID/HSTR/HSQR) field tuples
# --------------------------------------------------------------------------- #
FOURIER_SHARED_CORE_TEMPLATE = (
    "{prefix}_protocol_mode",
    "{prefix}_state_source",
    "{prefix}_bundle_dir",
    "{prefix}_bundle_config_sha256",
    "{prefix}_selected_pattern_sha256",
    "{prefix}_mask_sha256",
    "{prefix}_key_index",
    "{prefix}_pre_injection_latent_sha256",
    "{prefix}_post_injection_latent_sha256",
    "{prefix}_provider_entrypoint_sha256",
)
RID_REQUIRED_FIELDS = tuple(field.format(prefix="rid") for field in FOURIER_SHARED_CORE_TEMPLATE) + SHARED_CLEAN_COMMON_FIELDS
HSTR_REQUIRED_FIELDS = tuple(field.format(prefix="hstr") for field in FOURIER_SHARED_CORE_TEMPLATE) + SHARED_CLEAN_COMMON_FIELDS
HSQR_REQUIRED_FIELDS = tuple(field.format(prefix="hsqr") for field in FOURIER_SHARED_CORE_TEMPLATE) + SHARED_CLEAN_COMMON_FIELDS
RID_PAIRING_FIELDS = tuple(f for f in RID_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)
HSTR_PAIRING_FIELDS = tuple(f for f in HSTR_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)
HSQR_PAIRING_FIELDS = tuple(f for f in HSQR_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)

# --------------------------------------------------------------------------- #
# Method-level field registries
# --------------------------------------------------------------------------- #
METHOD_REQUIRED_FIELDS = {
    "GM": GM_REQUIRED_FIELDS,
    "T2S": T2S_REQUIRED_FIELDS,
    "RID": RID_REQUIRED_FIELDS,
    "HSTR": HSTR_REQUIRED_FIELDS,
    "HSQR": HSQR_REQUIRED_FIELDS,
}
METHOD_PAIRING_FIELDS = {
    "GM": GM_PAIRING_FIELDS,
    "T2S": T2S_PAIRING_FIELDS,
    "RID": RID_PAIRING_FIELDS,
    "HSTR": HSTR_PAIRING_FIELDS,
    "HSQR": HSQR_PAIRING_FIELDS,
}
METHOD_PROTOCOL_MODES = {
    "GM": ("gm_protocol_mode", GM_SHARED_TR_CLEAN_MODE),
    "T2S": ("t2s_protocol_mode", T2S_SHARED_TR_CLEAN_MODE),
    "RID": ("rid_protocol_mode", RID_SHARED_TR_CLEAN_MODE),
    "HSTR": ("hstr_protocol_mode", HSTR_SHARED_TR_CLEAN_MODE),
    "HSQR": ("hsqr_protocol_mode", HSQR_SHARED_TR_CLEAN_MODE),
}
METHOD_COHORT_CONSTANT_FIELDS = {
    "GM": (
        "gm_bundle_config_sha256",
        "gm_w1_file_sha256",
        "gm_w2_file_sha256",
        "gm_watermark_sha256",
        "gm_m_sha256",
        "gm_target_sha256",
        "gm_mask_sha256",
    ),
    "T2S": ("t2s_provider_config_sha256",),
    "RID": ("rid_bundle_config_sha256", "rid_selected_pattern_sha256", "rid_mask_sha256"),
    "HSTR": ("hstr_bundle_config_sha256", "hstr_selected_pattern_sha256", "hstr_mask_sha256"),
    "HSQR": ("hsqr_bundle_config_sha256", "hsqr_selected_pattern_sha256", "hsqr_mask_sha256"),
}
METHOD_PER_SAMPLE_UNIQUE_FIELDS = {
    "GM": ("gm_pre_injection_latent_sha256", "gm_post_injection_latent_sha256",
           "gm_sampling_uniform_sha256"),
    "T2S": ("t2s_watermark_id", "t2s_state_sha256", "t2s_abs_magnitude_sha256"),
    "RID": ("rid_post_injection_latent_sha256",),
    "HSTR": ("hstr_post_injection_latent_sha256",),
    "HSQR": ("hsqr_post_injection_latent_sha256",),
}
METHOD_COLLISION_COUNTED_FIELDS = {
    "T2S": ("t2s_session_key_sha256",),
}

# --------------------------------------------------------------------------- #
# Pairing / attack config fields
# --------------------------------------------------------------------------- #
PAIRING_REQUIRED_FIELDS = (
    "dataset",
    "run_id",
    "prompt",
    "prompt_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "watermarked_base_latent_sha256",
    "watermarked_latent_sha256",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "generation_config_sha256",
    "watermark_config_sha256",
    "pairing_sha256",
    "clean_path",
    "clean_sha256",
    "watermarked_path",
    "watermarked_sha256",
    "model_id",
    "model_revision",
)

PAIRING_HASH_FIELDS = (
    "protocol",
    "dataset",
    "run_id",
    "prompt_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "watermarked_base_latent_sha256",
    "watermarked_latent_sha256",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "generation_config_sha256",
    "watermark_config_sha256",
)

ATTACK_CONFIG_FIELDS = (
    "seed",
    "flow_dx_image_px",
    "flow_dy_image_px",
    "exact_ddim_timestep",
    "steps",
    "strength",
    "guidance_scale",
    "inversion_mode",
    "inversion_prompt",
    "reconstruction_prompt",
    "warp_mode",
    "sampling_mode",
    "padding_mode",
    "normalization_formula",
    "color_transfer_mode",
    "model_id",
    "model_revision",
)

SHARED_CLEAN_IDENTITY_FIELDS = (
    "prompt_sha256",
    "generation_config_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "clean_path",
    "clean_sha256",
)

SHARED_CLEAN_METHOD_PROTOCOLS = {
    "GS": GS_SHARED_TR_CLEAN_PROTOCOL,
    "GM": GM_SHARED_TR_CLEAN_PROTOCOL,
    "T2S": T2S_SHARED_TR_CLEAN_PROTOCOL,
    "RID": RID_SHARED_TR_CLEAN_PROTOCOL,
    "HSTR": HSTR_SHARED_TR_CLEAN_PROTOCOL,
    "HSQR": HSQR_SHARED_TR_CLEAN_PROTOCOL,
}


# --------------------------------------------------------------------------- #
# Protocol helpers
# --------------------------------------------------------------------------- #
def gs_fields_for_protocol(protocol: str) -> tuple[str, ...]:
    """GS provenance field tuple for a recorded pairing protocol (fail closed)."""
    text = str(protocol or "")
    if text == GS_PAIRING_PROTOCOL:
        return GS_REQUIRED_FIELDS
    if text == GS_SHARED_TR_CLEAN_PROTOCOL:
        return GS_V2_REQUIRED_FIELDS
    raise ValueError(f"unsupported GS pairing protocol: {protocol!r}")


def gs_fields_for_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Resolve one GS field tuple for a cohort, rejecting mixed protocols."""
    protocols = {str(row.get("protocol") or "") for row in rows}
    if len(protocols) != 1:
        raise ValueError(f"mixed GS pairing protocols in one cohort: {sorted(protocols)}")
    return gs_fields_for_protocol(next(iter(protocols)))


# --------------------------------------------------------------------------- #
# Tensor hashing
# --------------------------------------------------------------------------- #
def tensor_sha256(tensor) -> str:
    """Hash exact tensor shape, dtype, and bytes without retaining extra copies."""
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(value.view(-1).view(value.dtype).numpy().tobytes(order="C"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Pairing method resolution
# --------------------------------------------------------------------------- #
def pairing_method(row: Mapping[str, Any]) -> str:
    method = str(row.get("wm_type") or row.get("method") or "").upper()
    if method:
        return method
    protocol = str(row.get("protocol") or "")
    if protocol == TR_PAIRING_PROTOCOL:
        return "TR"
    if protocol in {GS_PAIRING_PROTOCOL, GS_SHARED_TR_CLEAN_PROTOCOL}:
        return "GS"
    if protocol == GM_SHARED_TR_CLEAN_PROTOCOL:
        return "GM"
    if protocol == T2S_SHARED_TR_CLEAN_PROTOCOL:
        return "T2S"
    if protocol == RID_SHARED_TR_CLEAN_PROTOCOL:
        return "RID"
    if protocol == HSTR_SHARED_TR_CLEAN_PROTOCOL:
        return "HSTR"
    if protocol == HSQR_SHARED_TR_CLEAN_PROTOCOL:
        return "HSQR"
    return ""


def build_pairing_sha256(row: Mapping[str, Any]) -> str:
    from raven.protocol import canonical_json_hash

    extra: tuple[str, ...] = ()
    method = pairing_method(row)
    if method == "GS":
        protocol = str(row.get("protocol") or "")
        extra = (
            GS_V2_PAIRING_FIELDS
            if protocol == GS_SHARED_TR_CLEAN_PROTOCOL
            else GS_REQUIRED_FIELDS
        )
    elif method in METHOD_PAIRING_FIELDS:
        extra = METHOD_PAIRING_FIELDS[method]
    payload = {field: str(row[field]) for field in PAIRING_HASH_FIELDS + extra}
    payload["base_latent_seed"] = int(row["base_latent_seed"])
    if method == "GS" and extra:
        payload["gs_secret_index"] = int(row["gs_secret_index"])
        if "gs_sampling_seed" in extra:
            payload["gs_sampling_seed"] = int(row["gs_sampling_seed"])
    return canonical_json_hash(payload)
