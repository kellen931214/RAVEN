"""Fail-closed provenance helpers for paired watermark experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


TR_PAIRING_PROTOCOL = "tree_ring_paired_base_latent_v1"
GS_PAIRING_PROTOCOL = "gaussian_shading_shared_uniform_v1"
PAIRING_PROTOCOL = TR_PAIRING_PROTOCOL
PAIRING_PROTOCOLS = {"TR": TR_PAIRING_PROTOCOL, "GS": GS_PAIRING_PROTOCOL}
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
    if protocol == GS_PAIRING_PROTOCOL:
        return "GS"
    return ""


def build_pairing_sha256(row: Mapping[str, Any]) -> str:
    # CSV is the durable provenance format. Canonicalize scalar types exactly
    # as they will be interpreted after a CSV round trip.
    fields = PAIRING_HASH_FIELDS + (GS_REQUIRED_FIELDS if pairing_method(row) == "GS" else ())
    payload = {field: str(row[field]) for field in fields}
    payload["base_latent_seed"] = int(row["base_latent_seed"])
    if pairing_method(row) == "GS":
        payload["gs_secret_index"] = int(row["gs_secret_index"])
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
    gs_secret_indexes: set[int] = set()
    gs_secret_hashes: set[str] = set()
    gs_sampling_seeds: set[int] = set()
    gs_sampling_hashes: set[str] = set()
    count = 0

    for row in rows:
        run_id = str(_required(row, "run_id", "unknown"))
        for field in PAIRING_REQUIRED_FIELDS:
            _required(row, field, run_id)
        method = pairing_method(row)
        if method not in PAIRING_PROTOCOLS:
            raise ValueError(f"unsupported pairing method run_id={run_id}: {method!r}")
        methods.add(method)
        protocol = str(row.get("protocol") or "")
        expected_protocol = PAIRING_PROTOCOLS[method]
        if protocol != expected_protocol:
            raise ValueError(
                f"unsupported pairing protocol run_id={run_id}: {protocol!r}; "
                f"expected {expected_protocol!r}"
            )
        if method == "GS":
            for field in GS_REQUIRED_FIELDS:
                _required(row, field, run_id)
            if str(row["gs_protocol_mode"]) != "official_compatible":
                raise ValueError(f"formal GS requires official_compatible mode run_id={run_id}")
            secret_index = int(row["gs_secret_index"])
            sampling_seed = int(row["gs_sampling_seed"])
            secret_hash = str(row["gs_secret_bundle_sha256"])
            sampling_hash = str(row["gs_sampling_uniform_sha256"])
            if secret_index in gs_secret_indexes:
                raise ValueError(f"duplicate GS secret index run_id={run_id}: {secret_index}")
            if secret_hash in gs_secret_hashes:
                raise ValueError(f"duplicate GS secret bundle run_id={run_id}: {secret_hash}")
            if sampling_seed in gs_sampling_seeds:
                raise ValueError(f"duplicate GS sampling seed run_id={run_id}: {sampling_seed}")
            if sampling_hash in gs_sampling_hashes:
                raise ValueError(f"duplicate GS sampling uniforms run_id={run_id}: {sampling_hash}")
            gs_secret_indexes.add(secret_index)
            gs_secret_hashes.add(secret_hash)
            gs_sampling_seeds.add(sampling_seed)
            gs_sampling_hashes.add(sampling_hash)
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
    for label, values in (
        ("watermark_mask_sha256", mask_hashes),
        ("generation_config_sha256", generation_hashes),
        ("watermark_config_sha256", watermark_hashes),
        ("model_revision", revisions),
    ):
        if len(values) != 1:
            raise ValueError(f"inconsistent {label}: {sorted(values)}")
    if method == "TR" and len(target_hashes) != 1:
        raise ValueError(f"inconsistent watermark_target_sha256: {sorted(target_hashes)}")
    if method == "GS" and len(target_hashes) != count:
        raise ValueError(
            f"GS requires one target per run: unique={len(target_hashes)} count={count}"
        )

    result = {
        "passed": True,
        "method": method,
        "protocol": PAIRING_PROTOCOLS[method],
        "count": count,
        "unique_run_ids": len(seen_run_ids),
        "unique_base_latent_seeds": len(seen_seeds),
        "unique_base_latent_hashes": len(seen_base_hashes),
        "duplicate_base_latent_hashes": 0,
        "unique_clean_image_hashes": len(seen_clean_hashes),
        "unique_watermarked_image_hashes": len(seen_watermarked_hashes),
        "unique_watermark_target_hashes": len(target_hashes),
        "watermark_target_sha256": next(iter(target_hashes)) if method == "TR" else None,
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
    return result


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
