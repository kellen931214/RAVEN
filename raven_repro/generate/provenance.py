"""Generation-only provenance: pairing audit, shared-clean validation, bundle verification.

These functions validate that generated cohorts are internally consistent and
faithful to their canonical source.  Attack and eval runtime never import this
module — only generation orchestrators and shared-clean cohort builders use it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from raven.detectors.protocols import (
    ALLOWED_PAIRING_PROTOCOLS,
    GM_SHARED_TR_CLEAN_MODE,
    GM_UNIFORM_DERIVATION,
    GS_PAIRING_PROTOCOL,
    GS_SHARED_TR_CLEAN_MODE,
    GS_SHARED_TR_CLEAN_PROTOCOL,
    GS_UNIFORM_DERIVATION,
    METHOD_COHORT_CONSTANT_FIELDS,
    METHOD_COLLISION_COUNTED_FIELDS,
    METHOD_PAIRING_FIELDS,
    METHOD_PER_SAMPLE_UNIQUE_FIELDS,
    METHOD_PROTOCOL_MODES,
    METHOD_REQUIRED_FIELDS,
    PAIRING_REQUIRED_FIELDS,
    PER_SAMPLE_TARGET_METHODS,
    SHARED_CLEAN_IDENTITY_FIELDS,
    SHARED_CLEAN_METHOD_PROTOCOLS,
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    SINGLE_TARGET_METHODS,
    build_pairing_sha256,
    gs_fields_for_protocol,
    pairing_method,
    sha256_path,
)
from raven.protocol import canonical_json_hash


# --------------------------------------------------------------------------- #
# Shared Fourier method config
# --------------------------------------------------------------------------- #
RID_SHARED_TR_CLEAN_PROTOCOL = "ringid_shared_tr_clean_v2"
HSTR_SHARED_TR_CLEAN_PROTOCOL = "hstr_shared_tr_clean_v2"
HSQR_SHARED_TR_CLEAN_PROTOCOL = "hsqr_shared_tr_clean_v2"
RID_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
HSTR_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"
HSQR_SHARED_TR_CLEAN_MODE = "official_math_shared_tr_clean"

SHARED_FOURIER_METHOD_CONFIG = {
    "RID": {
        "protocol": RID_SHARED_TR_CLEAN_PROTOCOL,
        "mode_field": "rid_protocol_mode",
        "mode": RID_SHARED_TR_CLEAN_MODE,
        "bundle_dir_field": "rid_bundle_dir",
        "bundle_config_field": "rid_bundle_config_sha256",
        "pattern_field": "rid_selected_pattern_sha256",
        "mask_field": "rid_mask_sha256",
    },
    "HSTR": {
        "protocol": HSTR_SHARED_TR_CLEAN_PROTOCOL,
        "mode_field": "hstr_protocol_mode",
        "mode": HSTR_SHARED_TR_CLEAN_MODE,
        "bundle_dir_field": "hstr_bundle_dir",
        "bundle_config_field": "hstr_bundle_config_sha256",
        "pattern_field": "hstr_selected_pattern_sha256",
        "mask_field": "hstr_mask_sha256",
    },
    "HSQR": {
        "protocol": HSQR_SHARED_TR_CLEAN_PROTOCOL,
        "mode_field": "hsqr_protocol_mode",
        "mode": HSQR_SHARED_TR_CLEAN_MODE,
        "bundle_dir_field": "hsqr_bundle_dir",
        "bundle_config_field": "hsqr_bundle_config_sha256",
        "pattern_field": "hsqr_selected_pattern_sha256",
        "mask_field": "hsqr_mask_sha256",
    },
}


# --------------------------------------------------------------------------- #
# Shared-clean identity assertion
# --------------------------------------------------------------------------- #
def _required(row: Mapping[str, Any], field: str, run_id: str) -> Any:
    if field not in row or row[field] in (None, ""):
        raise ValueError(f"pairing provenance missing {field} for run_id={run_id}")
    return row[field]


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


# --------------------------------------------------------------------------- #
# Pairing row audit
# --------------------------------------------------------------------------- #
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
    method_collision_counted_fields: dict[str, Counter[str]] = {}
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
            if method in {"RID", "HSTR", "HSQR"}:
                prefix = method.lower()
                if str(row[f"{prefix}_pre_injection_latent_sha256"]) != str(row["base_latent_sha256"]):
                    raise ValueError(
                        f"{prefix}_pre_injection_latent_sha256 is not the shared latent SHA run_id={run_id}"
                    )
                if str(row[f"{prefix}_post_injection_latent_sha256"]) != str(row["watermarked_latent_sha256"]):
                    raise ValueError(
                        f"{prefix}_post_injection_latent_sha256 disagrees with watermarked_latent_sha256 run_id={run_id}"
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
            for field in METHOD_COLLISION_COUNTED_FIELDS.get(method, ()):
                counts = method_collision_counted_fields.setdefault(field, Counter())
                counts[str(_required(row, field, run_id))] += 1
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
        if str(row["prompt_sha256"]) != __import__("hashlib").sha256(str(row["prompt"]).encode("utf-8")).hexdigest():
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
            verifier = METHOD_ARTIFACT_VERIFIERS.get(method)
            if verifier is not None:
                verifier(row, run_id)

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
            "collision_counted_field_stats": {
                field: {
                    "distinct_values": len(counts),
                    "colliding_pairs": sum(
                        n * (n - 1) // 2 for n in counts.values()
                    ),
                    "max_repeat": max(counts.values()),
                }
                for field, counts in sorted(method_collision_counted_fields.items())
            },
        })
    return result


# --------------------------------------------------------------------------- #
# Bundle artifact verifiers
# --------------------------------------------------------------------------- #
def _verify_bundle_manifest_fields(row: Mapping[str, Any], run_id: str, *, method: str) -> dict[str, str]:
    config = SHARED_FOURIER_METHOD_CONFIG[method]
    bundle_dir = Path(str(_required(row, config["bundle_dir_field"], run_id)))
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{method} bundle manifest missing for run_id={run_id}: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{method} bundle manifest is not valid JSON run_id={run_id}: {exc}") from None
    checks = {
        config["bundle_config_field"]: "bundle_config_sha256",
        config["pattern_field"]: "selected_pattern_sha256",
    }
    if "mask_sha256" in manifest:
        checks[config["mask_field"]] = "mask_sha256"
    artifacts = {config["bundle_dir_field"]: str(bundle_dir)}
    for row_field, manifest_field in checks.items():
        expected = str(_required(row, row_field, run_id))
        actual = str(manifest.get(manifest_field) or "")
        if expected != actual:
            raise ValueError(
                f"{method} bundle manifest drift run_id={run_id} field={row_field}: "
                f"row={expected} manifest={actual}"
            )
        artifacts[row_field] = actual
    return artifacts


def _verify_rid_bundle_artifacts(row: Mapping[str, Any], run_id: str) -> dict[str, str]:
    return _verify_bundle_manifest_fields(row, run_id, method="RID")


def _verify_hstr_bundle_artifacts(row: Mapping[str, Any], run_id: str) -> dict[str, str]:
    return _verify_bundle_manifest_fields(row, run_id, method="HSTR")


def _verify_hsqr_bundle_artifacts(row: Mapping[str, Any], run_id: str) -> dict[str, str]:
    return _verify_bundle_manifest_fields(row, run_id, method="HSQR")


def _verify_gm_bundle_artifacts(row: Mapping[str, Any], run_id: str) -> dict[str, str]:
    """The GM bundle this row names must still exist, byte-for-byte."""
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
    """The T2S portable state this row names must exist and still be self-signed."""
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
    recomputed = canonical_json_hash(payload)
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


METHOD_ARTIFACT_VERIFIERS = {
    "GM": _verify_gm_bundle_artifacts,
    "T2S": _verify_t2s_state_artifact,
    "RID": _verify_rid_bundle_artifacts,
    "HSTR": _verify_hstr_bundle_artifacts,
    "HSQR": _verify_hsqr_bundle_artifacts,
}


# --------------------------------------------------------------------------- #
# Cross-method shared-clean audit
# --------------------------------------------------------------------------- #
def audit_shared_clean_cohorts(
    tr_rows: Iterable[Mapping[str, Any]],
    cohorts: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    verify_files: bool = True,
    require_methods: Iterable[str] | None = None,
    expected_run_ids: Iterable[Any] | None = None,
    tr_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prove one or more V2 cohorts really reuse the canonical TR clean source."""
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
