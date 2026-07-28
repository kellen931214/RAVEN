"""Fail-closed provenance helpers for paired watermark experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


TR_PAIRING_PROTOCOL = "tree_ring_paired_base_latent_v1"
GS_PAIRING_PROTOCOL = "gaussian_shading_shared_uniform_v1"
# V2 cohort: GS embeds from the canonical Tree-Ring clean latent, so the clean
# image and the pre-watermark latent are literally the same artifacts TR used.
GS_SHARED_TR_CLEAN_PROTOCOL = "gaussian_shading_shared_tr_clean_v2"
SHARED_CLEAN_PROTOCOL = "tr_canonical_clean_latent_v2"
# gs_provider.GS_SHARED_TR_CLEAN_MODE — duplicated here because raven/ must not
# import eval_bench_wm. test_gaussian_shading_shared_tr_clean asserts they match.
GS_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
SHARED_CLEAN_SOURCE_METHOD = "TR"
GS_UNIFORM_DERIVATION = "normal_cdf_of_tr_float32_base_latent"

# --------------------------------------------------------------------------- #
# shared_tr_clean_v2 — GaussMarker and T2SMark (Issue #9)
# --------------------------------------------------------------------------- #
# Same contract as the GS V2 cohort: the canonical Tree-Ring clean image and base
# latent are read-only inputs, and only the method-specific watermarked image is
# produced. Each method keeps its own protocol name so a cohort can never be
# silently relabelled as another method's.
GM_SHARED_TR_CLEAN_PROTOCOL = "gaussmarker_shared_tr_clean_v2"
T2S_SHARED_TR_CLEAN_PROTOCOL = "t2smark_shared_tr_clean_v2"
# gm_provider.GM_SHARED_TR_CLEAN_MODE / t2s_provider.T2S_SHARED_TR_CLEAN_MODE —
# duplicated here because raven/ must not import eval_bench_wm. The runners and
# test_shared_tr_clean_gm_t2s assert they match.
GM_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
T2S_SHARED_TR_CLEAN_MODE = "official_encoder_shared_tr_clean"
GM_UNIFORM_DERIVATION = "normal_cdf_of_tr_float32_base_latent"

PAIRING_PROTOCOL = TR_PAIRING_PROTOCOL
# Default protocol written by a new cohort for each method.
PAIRING_PROTOCOLS = {"TR": TR_PAIRING_PROTOCOL, "GS": GS_PAIRING_PROTOCOL}
# Every protocol a method may legitimately carry. V1 GS cohorts stay valid and
# are never relabelled; V2 is an additional, separately-audited protocol.
ALLOWED_PAIRING_PROTOCOLS = {
    "TR": (TR_PAIRING_PROTOCOL,),
    "GS": (GS_PAIRING_PROTOCOL, GS_SHARED_TR_CLEAN_PROTOCOL),
    "GM": (GM_SHARED_TR_CLEAN_PROTOCOL,),
    "T2S": (T2S_SHARED_TR_CLEAN_PROTOCOL,),
}

# Methods whose watermark target is one cohort-wide artifact (Tree-Ring style
# ring pattern) versus methods that derive a fresh target per sample.
SINGLE_TARGET_METHODS = frozenset({"TR", "GM"})
PER_SAMPLE_TARGET_METHODS = frozenset({"GS", "T2S"})

# V1 GS provenance. Field order is preserved byte-for-byte from the original
# cohort so existing metadata column order and consumers are untouched.
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
# Shared by every GS protocol version. V1 additionally records the RNG seed that
# drew the uniforms; V2 has no RNG draw at all (the uniforms are a deterministic
# function of the TR latent), so gs_sampling_seed is deliberately absent from V2
# instead of being faked with an unrelated number.
GS_CORE_FIELDS = tuple(f for f in GS_REQUIRED_FIELDS if f != "gs_sampling_seed")

# Shared-clean identity recorded by every V2 row.
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

# Path fields are locations, not identities: they are required metadata but are
# excluded from the pairing hash so a canonical-layout move cannot invalidate an
# already-validated cohort. The content of everything a path points at is still
# bound into the hash through its SHA-256.
GS_V2_PAIRING_FIELDS = tuple(
    field
    for field in GS_V2_REQUIRED_FIELDS
    if field not in {"shared_clean_source_metadata_path", "tr_clean_path"}
)


# Shared-clean identity recorded by every GM and T2S V2 row. This is the GS V2
# set plus ``watermark_pre_injection_base_latent_sha256``, which Issue #9 makes
# an explicit mandatory identity. GS's own tuple is deliberately left untouched:
# changing it would invalidate every already-validated GS pairing hash.
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
)

# GaussMarker shared-clean provenance: bundle identity (ChaCha20 state, ring
# target, mask), the deterministic uniform derivation, and both sides of the
# ring injection.
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

# T2SMark shared-clean provenance: the portable state artifact and its digest,
# the RNG/inversion profile, and the magnitude-multiset proof that the canonical
# latent really was the encoder's Gaussian source.
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

# Path fields are locations, not identities (same rule as GS V2): required
# metadata, excluded from the pairing hash, but everything they point at is
# still bound through its SHA-256.
SHARED_CLEAN_PATH_FIELDS = frozenset(
    {
        "shared_clean_source_metadata_path",
        "tr_clean_path",
        "gm_bundle_dir",
        "t2s_state_path",
    }
)
GM_PAIRING_FIELDS = tuple(f for f in GM_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)
T2S_PAIRING_FIELDS = tuple(f for f in T2S_REQUIRED_FIELDS if f not in SHARED_CLEAN_PATH_FIELDS)

METHOD_REQUIRED_FIELDS = {
    "GM": GM_REQUIRED_FIELDS,
    "T2S": T2S_REQUIRED_FIELDS,
}
METHOD_PAIRING_FIELDS = {
    "GM": GM_PAIRING_FIELDS,
    "T2S": T2S_PAIRING_FIELDS,
}
# Per-method constants that must hold for every row of a shared-clean cohort.
METHOD_PROTOCOL_MODES = {
    "GM": ("gm_protocol_mode", GM_SHARED_TR_CLEAN_MODE),
    "T2S": ("t2s_protocol_mode", T2S_SHARED_TR_CLEAN_MODE),
}
# Fields that identify one cohort-wide watermark state; exactly one value each.
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
}
# Fields that must be distinct for every row; a repeat means two samples share
# state that is supposed to be per-sample.
METHOD_PER_SAMPLE_UNIQUE_FIELDS = {
    "GM": ("gm_pre_injection_latent_sha256", "gm_post_injection_latent_sha256",
           "gm_sampling_uniform_sha256"),
    "T2S": ("t2s_watermark_id", "t2s_state_sha256", "t2s_abs_magnitude_sha256",
            "t2s_session_key_sha256"),
}


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


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return ""


def build_pairing_sha256(row: Mapping[str, Any]) -> str:
    # CSV is the durable provenance format. Canonicalize scalar types exactly
    # as they will be interpreted after a CSV round trip.
    extra: tuple[str, ...] = ()
    method = pairing_method(row)
    if method == "GS":
        protocol = str(row.get("protocol") or "")
        # V1 hashes exactly the fields it always hashed; V2 additionally binds the
        # full shared-clean identity (source metadata SHA, shared sample SHA, TR
        # base-latent SHA, TR clean SHA, uniform derivation).
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
    return canonical_json_sha256(payload)


def build_attack_config_sha256(record: Mapping[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for field in ATTACK_CONFIG_FIELDS:
        if field not in record or record[field] is None:
            raise ValueError(f"attack record missing required config field: {field}")
        if record[field] == "" and field not in {"inversion_prompt", "reconstruction_prompt"}:
            raise ValueError(f"attack record missing required config field: {field}")
        value = record[field]
        if field in {"seed", "exact_ddim_timestep", "steps"}:
            value = int(value)
        elif field in {"flow_dx_image_px", "flow_dy_image_px", "strength", "guidance_scale"}:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"non-finite attack config value: {field}={value!r}")
        payload[field] = value
    return canonical_json_sha256(payload)


def _required(row: Mapping[str, Any], field: str, run_id: str) -> Any:
    if field not in row or row[field] in (None, ""):
        raise ValueError(f"pairing provenance missing {field} for run_id={run_id}")
    return row[field]


def _assert_shared_clean_identity(
    row: Mapping[str, Any], run_id: str, *, require_pre_injection: bool
) -> None:
    """The V2 promise: this row's clean image and pre-watermark latent ARE the TR ones.

    Shared by every ``shared_tr_clean_v2`` method. Anything else is not
    shared-clean, however well-formed the rest of the row looks.
    """
    if str(row["shared_clean_protocol"]) != SHARED_CLEAN_PROTOCOL:
        raise ValueError(
            f"unsupported shared_clean_protocol run_id={run_id}: "
            f"{row['shared_clean_protocol']!r}"
        )
    if str(row["shared_clean_source_method"]) != SHARED_CLEAN_SOURCE_METHOD:
        raise ValueError(
            f"shared clean source must be {SHARED_CLEAN_SOURCE_METHOD} "
            f"run_id={run_id}: {row['shared_clean_source_method']!r}"
        )
    base_hash = str(row["base_latent_sha256"])
    if str(row["tr_base_latent_sha256"]) != base_hash:
        raise ValueError(f"TR base latent SHA mismatch run_id={run_id}")
    if str(row["tr_clean_sha256"]) != str(row["clean_sha256"]):
        raise ValueError(f"TR clean image SHA mismatch run_id={run_id}")
    if str(row["tr_clean_path"]) != str(row["clean_path"]):
        raise ValueError(f"TR clean path mismatch run_id={run_id}")
    if str(row["shared_clean_sample_sha256"]) != base_hash:
        raise ValueError(
            f"shared_clean_sample_sha256 is not the shared latent SHA run_id={run_id}"
        )
    if require_pre_injection:
        if str(row["watermark_pre_injection_base_latent_sha256"]) != base_hash:
            raise ValueError(
                f"watermark_pre_injection_base_latent_sha256 is not the shared latent "
                f"SHA run_id={run_id}"
            )


def audit_pairing_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: int,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Reject missing provenance, repeated latents, broken pairs, and config drift."""
    seen_run_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_base_hashes: set[str] = set()
    seen_clean_hashes: set[str] = set()
    seen_watermarked_hashes: set[str] = set()
    target_hashes: set[str] = set()
    mask_hashes: set[str] = set()
    generation_hashes: set[str] = set()
    watermark_hashes: set[str] = set()
    revisions: set[str] = set()
    methods: set[str] = set()
    protocols: set[str] = set()
    shared_source_hashes: set[str] = set()
    gs_secret_indexes: set[int] = set()
    gs_secret_hashes: set[str] = set()
    gs_sampling_seeds: set[int] = set()
    gs_sampling_hashes: set[str] = set()
    method_constant_fields: dict[str, set[str]] = {}
    method_unique_fields: dict[str, set[str]] = {}
    count = 0

    for row in rows:
        run_id = str(_required(row, "run_id", "unknown"))
        for field in PAIRING_REQUIRED_FIELDS:
            _required(row, field, run_id)
        method = pairing_method(row)
        if method not in ALLOWED_PAIRING_PROTOCOLS:
            raise ValueError(f"unsupported pairing method run_id={run_id}: {method!r}")
        methods.add(method)
        protocol = str(row.get("protocol") or "")
        allowed = ALLOWED_PAIRING_PROTOCOLS[method]
        if protocol not in allowed:
            raise ValueError(
                f"unsupported pairing protocol run_id={run_id}: {protocol!r}; "
                f"expected one of {list(allowed)!r}"
            )
        protocols.add(protocol)
        if method == "GS":
            shared_tr_clean = protocol == GS_SHARED_TR_CLEAN_PROTOCOL
            for field in gs_fields_for_protocol(protocol):
                _required(row, field, run_id)
            expected_mode = (
                GS_SHARED_TR_CLEAN_MODE if shared_tr_clean else "official_compatible"
            )
            if str(row["gs_protocol_mode"]) != expected_mode:
                raise ValueError(
                    f"protocol {protocol} requires gs_protocol_mode={expected_mode!r} "
                    f"run_id={run_id}: got {row['gs_protocol_mode']!r}"
                )
            secret_index = int(row["gs_secret_index"])
            secret_hash = str(row["gs_secret_bundle_sha256"])
            sampling_hash = str(row["gs_sampling_uniform_sha256"])
            if secret_index in gs_secret_indexes:
                raise ValueError(f"duplicate GS secret index run_id={run_id}: {secret_index}")
            if secret_hash in gs_secret_hashes:
                raise ValueError(f"duplicate GS secret bundle run_id={run_id}: {secret_hash}")
            if sampling_hash in gs_sampling_hashes:
                raise ValueError(f"duplicate GS sampling uniforms run_id={run_id}: {sampling_hash}")
            gs_secret_indexes.add(secret_index)
            gs_secret_hashes.add(secret_hash)
            gs_sampling_hashes.add(sampling_hash)
            if shared_tr_clean:
                if str(row["gs_uniform_derivation"]) != GS_UNIFORM_DERIVATION:
                    raise ValueError(
                        f"unsupported gs_uniform_derivation run_id={run_id}: "
                        f"{row['gs_uniform_derivation']!r}"
                    )
                _assert_shared_clean_identity(row, run_id, require_pre_injection=False)
                shared_source_hashes.add(str(row["shared_clean_source_metadata_sha256"]))
            else:
                sampling_seed = int(row["gs_sampling_seed"])
                if sampling_seed in gs_sampling_seeds:
                    raise ValueError(
                        f"duplicate GS sampling seed run_id={run_id}: {sampling_seed}"
                    )
                gs_sampling_seeds.add(sampling_seed)
        elif method in METHOD_REQUIRED_FIELDS:
            for field in METHOD_REQUIRED_FIELDS[method]:
                _required(row, field, run_id)
            mode_field, expected_mode = METHOD_PROTOCOL_MODES[method]
            if str(row[mode_field]) != expected_mode:
                raise ValueError(
                    f"protocol {protocol} requires {mode_field}={expected_mode!r} "
                    f"run_id={run_id}: got {row[mode_field]!r}"
                )
            if method == "GM" and str(row["gm_uniform_derivation"]) != GM_UNIFORM_DERIVATION:
                raise ValueError(
                    f"unsupported gm_uniform_derivation run_id={run_id}: "
                    f"{row['gm_uniform_derivation']!r}"
                )
            if method == "T2S" and str(row["t2s_base_latent_sha256"]) != str(
                row["base_latent_sha256"]
            ):
                raise ValueError(
                    f"t2s_base_latent_sha256 is not the shared latent SHA run_id={run_id}"
                )
            _assert_shared_clean_identity(row, run_id, require_pre_injection=True)
            for field in METHOD_COHORT_CONSTANT_FIELDS[method]:
                method_constant_fields.setdefault(field, set()).add(str(row[field]))
            for field in METHOD_PER_SAMPLE_UNIQUE_FIELDS[method]:
                value = str(row[field])
                seen = method_unique_fields.setdefault(field, set())
                if value in seen:
                    raise ValueError(
                        f"duplicate {field} run_id={run_id}: {value}"
                    )
                seen.add(value)
            shared_source_hashes.add(str(row["shared_clean_source_metadata_sha256"]))
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id in pairing provenance: {run_id}")
        seen_run_ids.add(run_id)

        seed = int(row["base_latent_seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate base latent seed: {seed}")
        seen_seeds.add(seed)

        base_hash = str(row["base_latent_sha256"])
        if base_hash in seen_base_hashes:
            raise ValueError(f"duplicate base latent hash run_id={run_id}: {base_hash}")
        seen_base_hashes.add(base_hash)
        if str(row["clean_base_latent_sha256"]) != base_hash:
            raise ValueError(f"clean/base latent mismatch run_id={run_id}")
        if str(row["watermarked_base_latent_sha256"]) != base_hash:
            raise ValueError(f"watermarked/base latent mismatch run_id={run_id}")
        if str(row["prompt_sha256"]) != hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest():
            raise ValueError(f"prompt hash mismatch run_id={run_id}")
        if str(row["pairing_sha256"]) != build_pairing_sha256(row):
            raise ValueError(f"pairing hash mismatch run_id={run_id}")

        clean_path = Path(str(row["clean_path"]))
        watermarked_path = Path(str(row["watermarked_path"]))
        if verify_files:
            for label, path, expected in (
                ("clean", clean_path, str(row["clean_sha256"])),
                ("watermarked", watermarked_path, str(row["watermarked_sha256"])),
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual = sha256_path(path)
                if actual != expected:
                    raise ValueError(
                        f"{label} image SHA drift run_id={run_id}: expected {expected}, got {actual}"
                    )

        clean_sha = str(row["clean_sha256"])
        wm_sha = str(row["watermarked_sha256"])
        if clean_sha in seen_clean_hashes:
            raise ValueError(f"duplicate clean image SHA run_id={run_id}: {clean_sha}")
        if wm_sha in seen_watermarked_hashes:
            raise ValueError(f"duplicate watermarked image SHA run_id={run_id}: {wm_sha}")
        if clean_sha == wm_sha:
            raise ValueError(f"clean and watermarked images are identical run_id={run_id}")
        seen_clean_hashes.add(clean_sha)
        seen_watermarked_hashes.add(wm_sha)

        target_hashes.add(str(row["watermark_target_sha256"]))
        mask_hashes.add(str(row["watermark_mask_sha256"]))
        generation_hashes.add(str(row["generation_config_sha256"]))
        watermark_hashes.add(str(row["watermark_config_sha256"]))
        revisions.add(str(row["model_revision"]))
        count += 1

    if count != expected_count:
        raise ValueError(f"expected {expected_count} pairing rows, got {count}")
    if len(methods) != 1:
        raise ValueError(f"mixed pairing methods: {sorted(methods)}")
    method = next(iter(methods))
    if len(protocols) != 1:
        raise ValueError(f"mixed pairing protocols: {sorted(protocols)}")
    protocol = next(iter(protocols))
    if len(shared_source_hashes) > 1:
        raise ValueError(
            f"mixed shared-clean source metadata: {sorted(shared_source_hashes)}"
        )
    for label, values in (
        ("watermark_mask_sha256", mask_hashes),
        ("generation_config_sha256", generation_hashes),
        ("watermark_config_sha256", watermark_hashes),
        ("model_revision", revisions),
    ):
        if len(values) != 1:
            raise ValueError(f"inconsistent {label}: {sorted(values)}")
    if method in SINGLE_TARGET_METHODS and len(target_hashes) != 1:
        raise ValueError(f"inconsistent watermark_target_sha256: {sorted(target_hashes)}")
    if method in PER_SAMPLE_TARGET_METHODS and len(target_hashes) != count:
        raise ValueError(
            f"{method} requires one target per run: unique={len(target_hashes)} count={count}"
        )
    for field, values in sorted(method_constant_fields.items()):
        if len(values) != 1:
            raise ValueError(f"inconsistent {field}: {sorted(values)}")

    result = {
        "passed": True,
        "method": method,
        "protocol": protocol,
        "count": count,
        "unique_run_ids": len(seen_run_ids),
        "unique_base_latent_seeds": len(seen_seeds),
        "unique_base_latent_hashes": len(seen_base_hashes),
        "duplicate_base_latent_hashes": 0,
        "unique_clean_image_hashes": len(seen_clean_hashes),
        "unique_watermarked_image_hashes": len(seen_watermarked_hashes),
        "unique_watermark_target_hashes": len(target_hashes),
        "watermark_target_sha256": (
            next(iter(target_hashes)) if method in SINGLE_TARGET_METHODS else None
        ),
        "watermark_mask_sha256": next(iter(mask_hashes)),
        "generation_config_sha256": next(iter(generation_hashes)),
        "watermark_config_sha256": next(iter(watermark_hashes)),
        "model_revision": next(iter(revisions)),
    }
    if method == "GS":
        result.update({
            "unique_gs_secret_indexes": len(gs_secret_indexes),
            "unique_gs_secret_bundle_hashes": len(gs_secret_hashes),
            "unique_gs_sampling_seeds": len(gs_sampling_seeds),
            "unique_gs_sampling_uniform_hashes": len(gs_sampling_hashes),
        })
        if protocol == GS_SHARED_TR_CLEAN_PROTOCOL:
            result.update({
                "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
                "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
                "shared_clean_source_metadata_sha256": (
                    next(iter(shared_source_hashes)) if shared_source_hashes else None
                ),
                "gs_uniform_derivation": GS_UNIFORM_DERIVATION,
            })
    if method in METHOD_REQUIRED_FIELDS:
        result.update({
            "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
            "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
            "shared_clean_source_metadata_sha256": (
                next(iter(shared_source_hashes)) if shared_source_hashes else None
            ),
            "method_protocol_mode": METHOD_PROTOCOL_MODES[method][1],
            "cohort_constant_fields": {
                field: next(iter(values))
                for field, values in sorted(method_constant_fields.items())
            },
            "per_sample_unique_field_counts": {
                field: len(values)
                for field, values in sorted(method_unique_fields.items())
            },
        })
    return result


SHARED_CLEAN_IDENTITY_FIELDS = (
    "prompt_sha256",
    "generation_config_sha256",
    "base_latent_seed",
    "base_latent_sha256",
    "clean_base_latent_sha256",
    "clean_path",
    "clean_sha256",
)


#: Every method that may take part in a ``shared_tr_clean_v2`` cohort, and the
#: exact protocol name its rows must carry.
SHARED_CLEAN_METHOD_PROTOCOLS = {
    "GS": GS_SHARED_TR_CLEAN_PROTOCOL,
    "GM": GM_SHARED_TR_CLEAN_PROTOCOL,
    "T2S": T2S_SHARED_TR_CLEAN_PROTOCOL,
}


def _run_id_sort_key(run_id: str) -> tuple[int, Any]:
    """Numeric run_ids sort numerically; anything else sorts as text after them."""
    text = str(run_id)
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)


def _index_tr_source(tr_rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    tr_by_id: dict[str, Mapping[str, Any]] = {}
    for row in tr_rows:
        run_id = str(_required(row, "run_id", "unknown"))
        if run_id in tr_by_id:
            raise ValueError(f"duplicate TR run_id in shared-clean source: {run_id}")
        if pairing_method(row) != "TR":
            raise ValueError(f"shared-clean source row is not TR: run_id={run_id}")
        tr_by_id[run_id] = row
    if not tr_by_id:
        raise ValueError("shared-clean audit requires a non-empty TR source cohort")
    return tr_by_id


def _verify_gm_bundle_artifacts(row: Mapping[str, Any], run_id: str) -> dict[str, str]:
    """The GM bundle this row names must still exist, byte-for-byte.

    ``raven/`` must not import ``eval_bench_wm``, so the bundle is checked
    structurally: the three artifacts exist, the manifest's own
    ``bundle_config_sha256`` is the one the row recorded, and the two weight
    files hash to the values the manifest and the row agree on.
    """
    bundle_dir = Path(str(_required(row, "gm_bundle_dir", run_id)))
    manifest_path = bundle_dir / "manifest.json"
    w1_path = bundle_dir / "w1.pth"
    w2_path = bundle_dir / "w2.pth"
    for label, path in (
        ("manifest", manifest_path), ("w1", w1_path), ("w2", w2_path)
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"GM bundle {label} missing for run_id={run_id}: {path}"
            )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"GM bundle manifest is not valid JSON run_id={run_id}: {exc}") from None

    expected_config = str(_required(row, "gm_bundle_config_sha256", run_id))
    if str(manifest.get("bundle_config_sha256") or "") != expected_config:
        raise ValueError(
            f"GM bundle config SHA drift run_id={run_id}: row={expected_config} "
            f"manifest={manifest.get('bundle_config_sha256')}"
        )
    hashes = {}
    for label, path, row_field in (
        ("w1", w1_path, "gm_w1_file_sha256"),
        ("w2", w2_path, "gm_w2_file_sha256"),
    ):
        actual = sha256_path(path)
        expected_row = str(_required(row, row_field, run_id))
        expected_manifest = str(manifest.get(f"{label}_file_sha256") or "")
        if actual != expected_row:
            raise ValueError(
                f"GM bundle {label}.pth SHA drift run_id={run_id}: "
                f"expected {expected_row}, got {actual}"
            )
        if expected_manifest != actual:
            raise ValueError(
                f"GM bundle manifest disagrees with {label}.pth run_id={run_id}: "
                f"manifest={expected_manifest}, file={actual}"
            )
        hashes[row_field] = actual
    return {
        "gm_bundle_dir": str(bundle_dir),
        "gm_bundle_config_sha256": expected_config,
        **hashes,
    }


def _verify_t2s_state_artifact(row: Mapping[str, Any], run_id: str) -> dict[str, str]:
    """The T2S portable state this row names must exist and still be self-signed.

    An unsigned or edited state is rejected exactly as ``T2SWatermarkState``
    rejects it: the recomputed canonical digest of the payload must equal both
    the digest stored inside the file and the one recorded in the metadata row.
    """
    state_path = Path(str(_required(row, "t2s_state_path", run_id)))
    if not state_path.is_file():
        raise FileNotFoundError(f"T2S state artifact missing for run_id={run_id}: {state_path}")
    try:
        record = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"T2S state is not valid JSON run_id={run_id}: {exc}") from None
    if not isinstance(record, dict):
        raise ValueError(f"T2S state is not a JSON object run_id={run_id}: {state_path}")

    declared = record.get("state_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(
            f"T2S state is unsigned or has a malformed state_sha256 run_id={run_id}: "
            f"{declared!r}"
        )
    payload = {key: value for key, value in record.items() if key != "state_sha256"}
    recomputed = canonical_json_sha256(payload)
    if recomputed != declared:
        raise ValueError(
            f"T2S state signature invalid run_id={run_id}: declared={declared} "
            f"recomputed={recomputed}"
        )
    expected_row = str(_required(row, "t2s_state_sha256", run_id))
    if declared != expected_row:
        raise ValueError(
            f"T2S state SHA drift run_id={run_id}: row={expected_row} artifact={declared}"
        )
    return {"t2s_state_path": str(state_path), "t2s_state_sha256": declared}


#: Per-method artifact verification run under ``verify_files=True``. A method
#: cohort is only trustworthy if the state it was generated from still exists
#: unchanged, not merely if its images hash correctly.
METHOD_ARTIFACT_VERIFIERS = {
    "GM": _verify_gm_bundle_artifacts,
    "T2S": _verify_t2s_state_artifact,
}


def audit_shared_clean_cohorts(
    tr_rows: Iterable[Mapping[str, Any]],
    cohorts: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    verify_files: bool = True,
    require_methods: Iterable[str] | None = None,
    expected_run_ids: Iterable[Any] | None = None,
    tr_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prove one or more V2 cohorts really reuse the canonical TR clean source.

    ``cohorts`` maps a method name (``GS`` / ``GM`` / ``T2S``) to its metadata
    rows. For every run_id in a method cohort, the TR and method rows must agree
    on the full shared-clean identity, the method row's TR mirror fields must
    agree with the TR row itself (so a hand-edited row cannot self-certify), and
    both the clean image and the method's watermarked image must exist on disk
    with the recorded SHA-256.

    ``expected_run_ids`` makes coverage explicit and is the difference between
    "the rows present are consistent" and "the cohort is complete". Every
    required method's run_id set must equal it *exactly*: a missing row, an extra
    row and a duplicated row are all rejected. Passing ``None`` for a formal
    audit is not an option worth taking — use the full TR cohort's run_ids.

    ``tr_metadata_path`` binds the cohorts to the actual source file: its
    SHA-256 is recomputed from disk and must equal the
    ``shared_clean_source_metadata_sha256`` recorded by every method row, so a
    cohort generated against a since-changed TR metadata file cannot pass.

    Under ``verify_files`` each method's own state artifact is verified too — the
    GM bundle (manifest + ``w1.pth`` + ``w2.pth``) and the T2S portable state
    JSON, including its signature.

    Where two methods cover the same run_id they must agree with each other on
    the shared clean artifacts and must have produced *different* watermarked
    images. Equal seeds, equal filenames or equal run_ids are never accepted as
    evidence of anything.
    """
    tr_by_id = _index_tr_source(tr_rows)

    requested = {str(name).upper() for name in (require_methods or ())}
    unknown_required = requested - set(SHARED_CLEAN_METHOD_PROTOCOLS)
    if unknown_required:
        raise ValueError(f"unknown shared-clean methods required: {sorted(unknown_required)}")
    missing_required = requested - {str(name).upper() for name in cohorts}
    if missing_required:
        raise ValueError(
            f"shared-clean audit requires cohorts for {sorted(missing_required)}"
        )

    expected_ids: set[str] | None = None
    if expected_run_ids is not None:
        expected_ids = {str(value) for value in expected_run_ids}
        if not expected_ids:
            raise ValueError("expected_run_ids was supplied but is empty")
        unknown_ids = expected_ids - set(tr_by_id)
        if unknown_ids:
            raise ValueError(
                f"expected_run_ids are not in the TR source cohort: {sorted(unknown_ids)}"
            )

    source_metadata_sha256: str | None = None
    if tr_metadata_path is not None:
        source_path = Path(tr_metadata_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"TR source metadata not found: {source_path}")
        source_metadata_sha256 = sha256_path(source_path)

    per_method: dict[str, list[dict[str, Any]]] = {}
    for raw_method, rows in cohorts.items():
        method = str(raw_method).upper()
        expected_protocol = SHARED_CLEAN_METHOD_PROTOCOLS.get(method)
        if expected_protocol is None:
            raise ValueError(
                f"unsupported shared-clean method {raw_method!r}; "
                f"known: {sorted(SHARED_CLEAN_METHOD_PROTOCOLS)}"
            )
        checked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            run_id = str(_required(row, "run_id", "unknown"))
            if run_id in seen:
                raise ValueError(
                    f"duplicate {method} run_id in shared-clean cohort: {run_id}"
                )
            seen.add(run_id)
            if pairing_method(row) != method:
                raise ValueError(
                    f"shared-clean cohort row is not {method}: run_id={run_id}"
                )
            protocol = str(row.get("protocol") or "")
            if protocol != expected_protocol:
                raise ValueError(
                    f"cross-method shared-clean audit requires {expected_protocol}: "
                    f"run_id={run_id} has {protocol!r}"
                )
            tr_row = tr_by_id.get(run_id)
            if tr_row is None:
                raise ValueError(
                    f"{method} run_id={run_id} has no matching TR source row"
                )
            for field in SHARED_CLEAN_IDENTITY_FIELDS:
                tr_value = str(_required(tr_row, field, run_id))
                method_value = str(_required(row, field, run_id))
                if tr_value != method_value:
                    raise ValueError(
                        f"shared-clean mismatch run_id={run_id} field={field}: "
                        f"TR={tr_value!r} {method}={method_value!r}"
                    )
            for mirror, source in (
                ("tr_base_latent_sha256", "base_latent_sha256"),
                ("tr_clean_sha256", "clean_sha256"),
                ("tr_clean_path", "clean_path"),
            ):
                if str(_required(row, mirror, run_id)) != str(tr_row[source]):
                    raise ValueError(f"{mirror} mismatch run_id={run_id}")
            # Issue #9 mandatory identity; GS V2 rows predate the field name and
            # bind the same fact through clean_base_latent_sha256 instead.
            if "watermark_pre_injection_base_latent_sha256" in row:
                if str(row["watermark_pre_injection_base_latent_sha256"]) != str(
                    tr_row["base_latent_sha256"]
                ):
                    raise ValueError(
                        f"watermark_pre_injection_base_latent_sha256 mismatch run_id={run_id}"
                    )
            if str(row["watermarked_sha256"]) == str(tr_row["watermarked_sha256"]):
                raise ValueError(
                    f"{method} and TR watermarked images are identical run_id={run_id}"
                )
            if str(row["watermarked_sha256"]) == str(row["clean_sha256"]):
                raise ValueError(
                    f"{method} watermarked image is the clean image run_id={run_id}"
                )
            # The row must name the source file this audit actually read.
            if source_metadata_sha256 is not None:
                recorded_source = str(
                    _required(row, "shared_clean_source_metadata_sha256", run_id)
                )
                if recorded_source != source_metadata_sha256:
                    raise ValueError(
                        f"shared_clean_source_metadata_sha256 drift run_id={run_id} "
                        f"method={method}: row={recorded_source} "
                        f"actual={source_metadata_sha256}"
                    )
            artifacts: dict[str, str] = {}
            if verify_files:
                verifier = METHOD_ARTIFACT_VERIFIERS.get(method)
                if verifier is not None:
                    artifacts = verifier(row, run_id)
                for label, path_field, hash_field in (
                    ("clean", "clean_path", "clean_sha256"),
                    (f"{method.lower()}_watermarked", "watermarked_path", "watermarked_sha256"),
                ):
                    path = Path(str(row[path_field]))
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    actual = sha256_path(path)
                    if actual != str(row[hash_field]):
                        raise ValueError(
                            f"{label} SHA drift run_id={run_id}: "
                            f"expected {row[hash_field]}, got {actual}"
                        )
            checked.append({
                "run_id": run_id,
                "method": method,
                "base_latent_seed": int(row["base_latent_seed"]),
                "base_latent_sha256": str(row["base_latent_sha256"]),
                "clean_path": str(row["clean_path"]),
                "clean_sha256": str(row["clean_sha256"]),
                "watermarked_path": str(row["watermarked_path"]),
                "watermarked_sha256": str(row["watermarked_sha256"]),
                "artifacts": artifacts,
            })
        if not checked:
            raise ValueError(
                f"cross-method shared-clean audit requires at least one {method} row"
            )
        # Coverage: the cohort must be exactly the expected set, not a subset
        # that happens to be internally consistent.
        if expected_ids is not None and (method in requested or not requested):
            missing = sorted(expected_ids - seen, key=_run_id_sort_key)
            extra = sorted(seen - expected_ids, key=_run_id_sort_key)
            if missing or extra:
                raise ValueError(
                    f"{method} cohort does not cover the expected run_ids: "
                    f"missing={missing} unexpected={extra} "
                    f"(expected {len(expected_ids)} rows, got {len(seen)})"
                )
        per_method[method] = checked

    # Cross-method agreement for every run_id covered by more than one method.
    overlap: dict[str, dict[str, dict[str, Any]]] = {}
    for method, checked in per_method.items():
        for item in checked:
            overlap.setdefault(item["run_id"], {})[method] = item
    shared_run_ids = sorted(
        run_id for run_id, items in overlap.items() if len(items) > 1
    )
    for run_id in shared_run_ids:
        items = overlap[run_id]
        for field in ("clean_path", "clean_sha256", "base_latent_sha256"):
            values = {item[field] for item in items.values()}
            if len(values) != 1:
                raise ValueError(
                    f"cross-method shared-clean disagreement run_id={run_id} "
                    f"field={field}: {sorted(values)}"
                )
        watermarked = [item["watermarked_sha256"] for item in items.values()]
        if len(set(watermarked)) != len(watermarked):
            raise ValueError(
                f"two methods produced the identical watermarked image run_id={run_id}"
            )

    return {
        "passed": True,
        "audit": "cross_method_shared_clean",
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "verified_files": bool(verify_files),
        "compared_fields": list(SHARED_CLEAN_IDENTITY_FIELDS),
        "tr_source_rows": len(tr_by_id),
        "tr_metadata_path": None if tr_metadata_path is None else str(tr_metadata_path),
        "tr_metadata_sha256": source_metadata_sha256,
        "source_metadata_sha256_verified": source_metadata_sha256 is not None,
        "expected_run_ids": (
            None if expected_ids is None
            else sorted(expected_ids, key=_run_id_sort_key)
        ),
        "expected_run_id_count": None if expected_ids is None else len(expected_ids),
        "run_id_coverage_verified": expected_ids is not None,
        "method_artifacts_verified": sorted(
            method for method in per_method
            if verify_files and method in METHOD_ARTIFACT_VERIFIERS
        ),
        "methods": sorted(per_method),
        "method_protocols": {
            method: SHARED_CLEAN_METHOD_PROTOCOLS[method] for method in sorted(per_method)
        },
        "rows_checked": {method: len(items) for method, items in sorted(per_method.items())},
        "unique_clean_sha256": {
            method: len({item["clean_sha256"] for item in items})
            for method, items in sorted(per_method.items())
        },
        "unique_base_latent_sha256": {
            method: len({item["base_latent_sha256"] for item in items})
            for method, items in sorted(per_method.items())
        },
        "unique_watermarked_sha256": {
            method: len({item["watermarked_sha256"] for item in items})
            for method, items in sorted(per_method.items())
        },
        "cross_method_run_ids": shared_run_ids,
        "rows": {method: items for method, items in sorted(per_method.items())},
    }


def audit_tr_gs_shared_clean(
    tr_rows: Iterable[Mapping[str, Any]],
    gs_rows: Iterable[Mapping[str, Any]],
    *,
    verify_files: bool = True,
    expected_run_ids: Iterable[Any] | None = None,
    tr_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """GS-shaped view of :func:`audit_shared_clean_cohorts` (unchanged contract)."""
    gs_rows = list(gs_rows)
    result = audit_shared_clean_cohorts(
        tr_rows,
        {"GS": gs_rows},
        verify_files=verify_files,
        require_methods=("GS",),
        expected_run_ids=expected_run_ids,
        tr_metadata_path=tr_metadata_path,
    )
    by_id = {str(row["run_id"]): row for row in gs_rows}
    checked = [
        {
            "run_id": item["run_id"],
            "base_latent_seed": item["base_latent_seed"],
            "base_latent_sha256": item["base_latent_sha256"],
            "clean_path": item["clean_path"],
            "clean_sha256": item["clean_sha256"],
            "gs_watermarked_path": item["watermarked_path"],
            "gs_watermarked_sha256": item["watermarked_sha256"],
            "gs_sampling_uniform_sha256": str(by_id[item["run_id"]]["gs_sampling_uniform_sha256"]),
            "gs_secret_bundle_sha256": str(by_id[item["run_id"]]["gs_secret_bundle_sha256"]),
        }
        for item in result["rows"]["GS"]
    ]
    return {
        "passed": True,
        "audit": "cross_method_shared_clean",
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "gs_protocol": GS_SHARED_TR_CLEAN_PROTOCOL,
        "gs_uniform_derivation": GS_UNIFORM_DERIVATION,
        "verified_files": bool(verify_files),
        "compared_fields": list(SHARED_CLEAN_IDENTITY_FIELDS),
        "tr_source_rows": result["tr_source_rows"],
        "gs_rows_checked": len(checked),
        "unique_clean_sha256": result["unique_clean_sha256"]["GS"],
        "unique_base_latent_sha256": result["unique_base_latent_sha256"]["GS"],
        "unique_gs_watermarked_sha256": result["unique_watermarked_sha256"]["GS"],
        "rows": checked,
    }


def assert_attack_pair_config_match(
    clean_attack: Mapping[str, Any], watermarked_attack: Mapping[str, Any], run_id: str
) -> str:
    clean_hash = build_attack_config_sha256(clean_attack)
    watermarked_hash = build_attack_config_sha256(watermarked_attack)
    for label, record, computed in (
        ("clean", clean_attack, clean_hash),
        ("watermarked", watermarked_attack, watermarked_hash),
    ):
        stored = record.get("attack_config_sha256")
        if not stored:
            raise ValueError(f"{label} attack missing attack_config_sha256 run_id={run_id}")
        if str(stored) != computed:
            raise ValueError(f"{label} attack config hash mismatch run_id={run_id}")
    if clean_hash != watermarked_hash:
        raise ValueError(
            f"attacked clean/watermarked config mismatch run_id={run_id}: "
            f"clean={clean_hash} watermarked={watermarked_hash}"
        )
    clean_pair = str(clean_attack.get("pairing_sha256") or "")
    wm_pair = str(watermarked_attack.get("pairing_sha256") or "")
    if not clean_pair or clean_pair != wm_pair:
        raise ValueError(f"attack source pairing mismatch run_id={run_id}")
    return clean_hash
