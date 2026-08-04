"""Gaussian Shading detector adapter.

Every source sample gets its own canonical ``GsProvider``, constructed once
and cached per ``(run_id, role)``.  The adapter validates required metadata,
provider configuration identity, pipe profile uniformity, secret provenance,
watermark target/mask identity, and scoring outputs — all before returning a
scored row.  No GS algorithm is reimplemented here.

Failure taxonomy
----------------
* missing required metadata / secret index / bundle → ``DetectorMissingStateError``
  → ``failed_missing_required_state`` (allowable under ``--allow-missing-metrics``)
* SHA / target / mask / protocol / config mismatch → ``DetectorStateValidationError``
  → ``failed_state_validation`` (hard failure)
* runtime inversion / decoding / scoring failure → ``DetectorScoringError``
  → ``failed_scoring`` (hard failure)
* missing image → ``FileNotFoundError`` → ``failed_missing_image`` (hard failure)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from . import (
    DetectorMissingStateError,
    DetectorDependencyError,
    DetectorProviderInitializationError,
    DetectorStateValidationError,
    DetectorScoringError,
)

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "gs_secret_index",
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "gs_protocol_mode",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "provider_config_hash",
})


# ---------------------------------------------------------------------------
# Required metadata fields that must be present before provider_kwargs
# ---------------------------------------------------------------------------
_GS_REQUIRED_METADATA_FIELDS: tuple[str, ...] = (
    "run_id",
    "gs_secret_index",
    "gs_message_sha256",
    "gs_key_sha256",
    "gs_nonce_sha256",
    "gs_secret_bundle_sha256",
    "gs_protocol_mode",
    "watermark_target_sha256",
    "watermark_mask_sha256",
    "provider_config_hash",
)

# Pipe-level fields validated for cohort uniformity (not in PROVIDER_FIELDS_BY_METHOD["GS"])
_GS_PIPE_CONFIG_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_revision",
    "scheduler",
    "resolution",
)


def describe_required_artifacts() -> list[str]:
    return [
        "GS secret bundle directory",
        "gs_secret_index per row in source metadata (non-negative integer)",
        "gs_message_sha256, gs_key_sha256, gs_nonce_sha256",
        "gs_secret_bundle_sha256",
        "gs_protocol_mode",
        "watermark_target_sha256",
        "watermark_mask_sha256",
        "provider_config_hash",
        "Stable Diffusion inversion pipe",
    ]


def _ensure_paths():
    repo = Path(__file__).resolve().parents[3]
    for p in [str(repo / "eval_bench_wm"), str(repo / "raven_repro" / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Metadata preflight — must run BEFORE any provider_kwargs call
# ---------------------------------------------------------------------------
def _validate_required_gs_metadata(record: dict[str, Any]) -> None:
    """Validate that *record* carries every required GS metadata field.

    Runs before ``provider_kwargs("GS", row)`` so canonical defaults cannot
    mask missing or invalid values.  ``gs_secret_index`` must be a
    non-negative integer — anything else (float, negative, non-numeric
    string) is ``DetectorStateValidationError``.
    """
    run_id = str(record.get("run_id", ""))
    if not run_id.strip():
        raise DetectorMissingStateError("record missing run_id")

    role = record.get("role") or record.get("source_role", "")
    if not str(role).strip():
        raise DetectorMissingStateError(
            f"run_id={run_id}: missing role/source_role"
        )

    for field in _GS_REQUIRED_METADATA_FIELDS:
        if field == "run_id":
            continue
        value = record.get(field)
        if value is None or str(value).strip() == "":
            raise DetectorMissingStateError(
                f"run_id={run_id}: missing required GS metadata field: {field}"
            )

    secret_index_raw = record.get("gs_secret_index")
    try:
        idx = int(secret_index_raw)
        if idx < 0:
            raise ValueError
    except (ValueError, TypeError):
        raise DetectorStateValidationError(
            f"run_id={run_id}: gs_secret_index must be a non-negative "
            f"integer, got {secret_index_raw!r}"
        )


# ---------------------------------------------------------------------------
# Metadata index — prevents record cross-use
# ---------------------------------------------------------------------------
def _build_metadata_index(
    enriched_records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index enriched records by ``(run_id, role)``.

    Every key must be unique; duplicate keys are
    ``DetectorStateValidationError``.  Every record must carry a non-empty
    ``run_id`` and ``role``.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in enriched_records:
        run_id = str(rec.get("run_id", ""))
        role = str(rec.get("role", "watermarked"))
        if not run_id.strip():
            raise DetectorMissingStateError(
                "enriched record has no run_id"
            )
        key = (run_id, role)
        if key in index:
            raise DetectorStateValidationError(
                f"duplicate metadata key (run_id={run_id!r}, role={role!r})"
            )
        index[key] = dict(rec)
    return index


# ---------------------------------------------------------------------------
# Provider configuration identity
# ---------------------------------------------------------------------------
def _validate_pipe_config_uniformity(
    enriched_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract pipe-level config from resolved metadata, verify uniformity.

    Returns a dict with *model_id*, *model_revision*, *scheduler*,
    *resolution* (all guaranteed to be identical across the cohort).
    """
    pipe_hashes: dict[str, dict[str, Any]] = {}
    for rec in enriched_records:
        model_id = str(rec.get("model_id",
                               "RedbeardNZ/stable-diffusion-2-1-base"))
        model_revision = str(rec.get("model_revision", ""))
        scheduler = str(rec.get("scheduler", "DDIM"))
        resolution_raw = rec.get("resolution", "512")
        try:
            resolution = int(resolution_raw)
        except (ValueError, TypeError):
            raise DetectorStateValidationError(
                f"run_id={rec.get('run_id')}: resolution must be an "
                f"integer, got {resolution_raw!r}"
            )

        from raven.eval_protocol import canonical_json_hash

        h = canonical_json_hash({
            "model_id": model_id,
            "model_revision": model_revision if model_revision else None,
            "scheduler": scheduler,
            "resolution": resolution,
        })
        pipe_hashes[h] = {
            "model_id": model_id,
            "model_revision": model_revision if model_revision else None,
            "scheduler": scheduler,
            "resolution": resolution,
        }
    if len(pipe_hashes) != 1:
        raise DetectorStateValidationError(
            f"GS pipe config not uniform across cohort: "
            f"{len(pipe_hashes)} distinct pipe profiles"
        )
    _, config = next(iter(pipe_hashes.items()))
    return config


def _validate_gs_provider_config(
    enriched_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Compute canonical GS provider config and hash from enriched records.

    Uses ``require_uniform_provider_config`` from the formal extraction path.
    Returns ``(canonical_config, canonical_hash)``.
    """
    from raven.eval_protocol import (
        require_uniform_provider_config,
        provider_config_hash,
        canonical_json_hash,
    )

    try:
        canonical_config, canonical_hash = require_uniform_provider_config(
            "GS", enriched_records,
        )
    except ValueError as exc:
        raise DetectorStateValidationError(
            f"GS provider config not uniform: {exc}"
        ) from exc

    # Augment with gs_detection_mode and pipe fields for full identity
    augmented = dict(canonical_config)
    pipe_cfg = _validate_pipe_config_uniformity(enriched_records)
    augmented.update(pipe_cfg)
    # gs_detection_mode is detection-time policy but part of adapter identity
    detection_modes = {
        str(r.get("gs_detection_mode", "official_onebit"))
        for r in enriched_records
    }
    if len(detection_modes) != 1:
        raise DetectorStateValidationError(
            f"GS detection mode not uniform: {sorted(detection_modes)}"
        )
    augmented["gs_detection_mode"] = next(iter(detection_modes))

    augmented_hash = canonical_json_hash(augmented)
    return augmented, augmented_hash


# ---------------------------------------------------------------------------
# Per-source provider construction
# ---------------------------------------------------------------------------
def _construct_provider(
    pipe: Any,
    GsProvider: type,
    device_obj: Any,
    metadata: dict[str, Any],
    canonical_config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Construct a canonical GS provider for one source sample.

    Uses ``provider_kwargs("GS", metadata)`` from the formal extraction path.
    Validates secret provenance, target, mask, and protocol identity against
    the resolved metadata.  Returns ``(provider, provenance_record)``.

    Raises:
        DetectorMissingStateError: secret index out of range or bundle missing.
        DetectorStateValidationError: any identity mismatch.
        DetectorProviderInitializationError: constructor failure.
    """
    from extract_verification_scores import provider_kwargs as _canonical_gs_kwargs

    run_id = str(metadata.get("run_id", ""))

    gs_kwargs = _canonical_gs_kwargs("GS", metadata)
    secret_index = gs_kwargs.get("gs_secret_index")

    try:
        provider = GsProvider(
            latent_shape=pipe.get_latent_shape(),
            dtype=pipe.get_dtype(),
            device=device_obj,
            **gs_kwargs,
        )
    except IndexError:
        raise DetectorMissingStateError(
            f"run_id={run_id}: gs_secret_index {secret_index} is out of "
            "range for the available secret bundle"
        )
    except TypeError as exc:
        raise DetectorProviderInitializationError(
            f"run_id={run_id}: GS provider construction type error: {exc}"
        ) from exc
    except ValueError as exc:
        raise DetectorProviderInitializationError(
            f"run_id={run_id}: GS provider construction value error: {exc}"
        ) from exc
    except DetectorMissingStateError:
        raise
    except DetectorStateValidationError:
        raise
    except DetectorProviderInitializationError:
        raise
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"run_id={run_id}: GS provider construction failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # ---- secret provenance ----
    try:
        secret = provider.secret_provenance()
    except IndexError:
        raise DetectorMissingStateError(
            f"run_id={run_id}: secret_provenance index out of range for "
            f"secret_index={secret_index}"
        )
    except Exception as exc:
        raise DetectorStateValidationError(
            f"run_id={run_id}: secret_provenance failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    recorded_index = int(secret_index)
    actual_index = int(secret.get("secret_index", -1))
    if recorded_index != actual_index:
        raise DetectorStateValidationError(
            f"run_id={run_id}: GS secret_index mismatch: "
            f"recorded={recorded_index!r} actual={actual_index!r}"
        )

    for row_field, secret_field in (
        ("gs_message_sha256", "message_sha256"),
        ("gs_key_sha256", "key_sha256"),
        ("gs_nonce_sha256", "nonce_sha256"),
        ("gs_secret_bundle_sha256", "secret_bundle_sha256"),
    ):
        recorded = str(metadata.get(row_field, ""))
        actual = str(secret.get(secret_field, ""))
        if recorded != actual:
            raise DetectorStateValidationError(
                f"run_id={run_id}: GS {row_field} mismatch: "
                f"recorded={recorded!r} actual={actual!r}"
            )

    # ---- protocol mode ----
    recorded_protocol = str(metadata.get("gs_protocol_mode", ""))
    actual_protocol = str(getattr(provider, "gs_protocol_mode", ""))
    if recorded_protocol != actual_protocol:
        raise DetectorStateValidationError(
            f"run_id={run_id}: GS protocol_mode mismatch: "
            f"recorded={recorded_protocol!r} actual={actual_protocol!r}"
        )

    # ---- target identity ----
    from raven.pairing_provenance import tensor_sha256

    source_target = str(metadata.get("watermark_target_sha256", ""))
    detector_target = tensor_sha256(provider.watermark_target_tensor())
    if source_target != detector_target:
        raise DetectorStateValidationError(
            f"run_id={run_id}: GS target SHA mismatch: "
            f"recorded={source_target!r} detector={detector_target!r}"
        )

    # ---- mask identity (GS has no mask — canonical sentinel) ----
    from raven.eval_protocol import canonical_json_hash

    source_mask = str(metadata.get("watermark_mask_sha256", ""))
    detector_mask = canonical_json_hash(
        {"method": "GS", "mask": "not_applicable", "version": 1}
    )
    if source_mask != detector_mask:
        raise DetectorStateValidationError(
            f"run_id={run_id}: GS mask SHA mismatch: "
            f"recorded={source_mask!r} detector={detector_mask!r}"
        )

    provenance_record = {
        "gs_secret_index": int(secret_index),
        "gs_message_sha256": secret["message_sha256"],
        "gs_key_sha256": secret["key_sha256"],
        "gs_nonce_sha256": secret["nonce_sha256"],
        "gs_secret_bundle_sha256": secret["secret_bundle_sha256"],
        "gs_protocol_mode": actual_protocol,
        "source_watermark_target_sha256": source_target,
        "detector_watermark_target_sha256": detector_target,
        "source_watermark_mask_sha256": source_mask,
        "detector_watermark_mask_sha256": detector_mask,
    }
    return provider, provenance_record


# ---------------------------------------------------------------------------
# Scoring output validation
# ---------------------------------------------------------------------------
def _validate_scoring_result(result: dict[str, Any], run_id: str) -> None:
    """Validate that *result* from ``evaluate_image`` contains required GS
    scoring outputs.  Raises ``DetectorScoringError`` on missing/illegal
    values — no fabricated defaults."""
    # bit_accuracies
    bit_accuracies = result.get("bit_accuracies")
    if not isinstance(bit_accuracies, list) or len(bit_accuracies) == 0:
        raise DetectorScoringError(
            f"run_id={run_id}: GS scoring result missing or empty "
            "bit_accuracies list"
        )
    raw_bit = bit_accuracies[0]
    try:
        bit_val = float(raw_bit)
    except (ValueError, TypeError):
        raise DetectorScoringError(
            f"run_id={run_id}: GS bit accuracy not convertible to float: "
            f"{raw_bit!r}"
        )
    if not math.isfinite(bit_val):
        raise DetectorScoringError(
            f"run_id={run_id}: GS bit accuracy is non-finite: {bit_val!r}"
        )
    if not 0.0 <= bit_val <= 1.0:
        raise DetectorScoringError(
            f"run_id={run_id}: GS bit accuracy out of range [0,1]: "
            f"{bit_val!r}"
        )

    # message_bits_str_list
    msg_list = result.get("message_bits_str_list")
    if not isinstance(msg_list, list) or len(msg_list) == 0:
        raise DetectorScoringError(
            f"run_id={run_id}: GS scoring result missing or empty "
            "message_bits_str_list"
        )
    decoded_str = msg_list[0]
    if not isinstance(decoded_str, str) or decoded_str.strip() == "":
        raise DetectorScoringError(
            f"run_id={run_id}: GS decoded bits string is empty or "
            f"non-string: {decoded_str!r}"
        )


def _validate_thresholds(
    thresholds: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    """Validate official thresholds and return canonical threshold record.

    Raises ``DetectorScoringError`` if required thresholds are missing,
    non-finite, or out of range.
    """
    tau_onebit = thresholds.get("tau_onebit")
    tau_bits = thresholds.get("tau_bits")

    if tau_onebit is None or tau_bits is None:
        raise DetectorScoringError(
            f"run_id={run_id}: GS official_thresholds missing tau_onebit "
            f"or tau_bits"
        )
    try:
        t1 = float(tau_onebit)
        tb = float(tau_bits)
    except (ValueError, TypeError):
        raise DetectorScoringError(
            f"run_id={run_id}: GS official thresholds not convertible to "
            f"float: tau_onebit={tau_onebit!r} tau_bits={tau_bits!r}"
        )
    if not math.isfinite(t1) or not math.isfinite(tb):
        raise DetectorScoringError(
            f"run_id={run_id}: GS official thresholds non-finite: "
            f"tau_onebit={t1!r} tau_bits={tb!r}"
        )

    record = {
        "gs_official_tau_onebit": t1,
        "gs_official_tau_bits": tb,
    }
    for extra_key in ("fpr", "user_number", "comparison_operator", "source"):
        if extra_key in thresholds:
            record[f"gs_official_{extra_key}"] = thresholds[extra_key]
    return record


# ===========================================================================
# Public adapter API
# ===========================================================================
def load_state(records: list[dict[str, Any]], device: str,
               **extra) -> dict[str, Any]:
    """Load GS pipe and build the per-source provider cache infrastructure.

    Returns a ``provider_info`` dict with:

    * ``pipe`` — the SD inversion pipe (built from verified config)
    * ``provider_class`` — ``GsProvider``
    * ``device_obj`` — torch device
    * ``provider_cache`` — ``dict[(run_id, role), provider]`` (populated lazily)
    * ``metadata_index`` — ``dict[(run_id, role), resolved_metadata]``
    * ``canonical_config`` — uniform provider config dict
    * ``detector_provider_config_hash`` — SHA-256 of the canonical config
    * ``pipe_config`` — resolved pipe profile
    """
    import torch

    _ensure_paths()

    try:
        from eval_bench_wm.utils.pipe import pipe_utils
        from eval_bench_wm.utils.wm.gs_provider import GsProvider
    except ImportError as exc:
        raise DetectorDependencyError(
            f"GS dependencies not available: {exc}"
        ) from exc

    # ---- preflight: validate required metadata on every record ----
    for rec in records:
        _validate_required_gs_metadata(rec)

    # ---- metadata index ----
    metadata_index = _build_metadata_index(records)

    # ---- provider config identity ----
    canonical_config, detector_provider_hash = _validate_gs_provider_config(
        records,
    )

    # Validate source provider_config_hash against detector hash
    for rec in records:
        run_id = str(rec.get("run_id", ""))
        source_hash = str(rec.get("provider_config_hash", ""))
        if source_hash != detector_provider_hash:
            raise DetectorStateValidationError(
                f"run_id={run_id}: provider_config_hash mismatch: "
                f"source={source_hash!r} detector={detector_provider_hash!r}"
            )

    # ---- pipe from verified config ----
    pipe_cfg = _validate_pipe_config_uniformity(records)
    try:
        device_obj = torch.device(device)
        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": pipe_cfg["model_id"],
            "resolution": pipe_cfg["resolution"],
            "device": device_obj,
            "eager_loading": False,
            "schedulers_name": pipe_cfg["scheduler"],
            "disable_tqdm": True,
        }
        if pipe_cfg.get("model_revision"):
            load_kwargs["model_revision"] = pipe_cfg["model_revision"]
        pipe = pipe_utils.get_pipe_provider(**load_kwargs)
    except Exception as exc:
        raise DetectorProviderInitializationError(
            f"GS pipe init failed: {type(exc).__name__}: {exc}"
        ) from exc

    return {
        "pipe": pipe,
        "provider_class": GsProvider,
        "device_obj": device_obj,
        "provider_cache": {},
        "metadata_index": metadata_index,
        "canonical_config": canonical_config,
        "detector_provider_config_hash": detector_provider_hash,
        "pipe_config": pipe_cfg,
    }


def score_image(provider_info: dict[str, Any], image_path: str, *,
                record: dict[str, Any] | None = None,
                evaluation_entry: dict[str, Any] | None = None,
                steps: int = 50) -> dict[str, Any]:
    """Score one GS image through the canonical per-source provider.

    Uses ``(evaluation_entry.run_id, evaluation_entry.source_role)`` to
    look up canonical metadata from the index.  The provider for that source
    is constructed once and cached; subsequent cohorts for the same source
    reuse the cached provider.
    """
    import torch
    from PIL import Image, ImageOps
    from raven.eval_protocol import canonical_json_hash
    from raven.pairing_provenance import tensor_sha256

    _ensure_paths()
    from extract_verification_scores import (
        evaluate_image,
        raw_score as _canonical_raw_score,
        canonical_score as _canonical_canonical_score,
        provider_kwargs as _canonical_gs_kwargs,
    )

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(image_path)

    if evaluation_entry is None:
        raise DetectorMissingStateError(
            "GS requires evaluation_entry with run_id and source_role"
        )

    run_id = str(evaluation_entry.get("run_id", ""))
    source_role = str(evaluation_entry.get("source_role", ""))
    if not run_id or not source_role:
        raise DetectorMissingStateError(
            "evaluation_entry missing run_id or source_role"
        )

    source_key = (run_id, source_role)
    metadata_index = provider_info.get("metadata_index", {})
    if not isinstance(metadata_index, dict) or source_key not in metadata_index:
        raise DetectorMissingStateError(
            f"no metadata indexed for (run_id={run_id!r}, "
            f"source_role={source_role!r})"
        )
    canonical_metadata = metadata_index[source_key]

    # Cross-validate passed record against indexed metadata
    if record is not None:
        for field in ("run_id",):
            rec_val = str(record.get(field, ""))
            meta_val = str(canonical_metadata.get(field, ""))
            if rec_val and meta_val and rec_val != meta_val:
                raise DetectorStateValidationError(
                    f"run_id={run_id}: record.{field}={rec_val!r} "
                    f"disagrees with indexed metadata={meta_val!r}"
                )
        for field in ("role", "source_role"):
            rec_role = str(record.get(field, ""))
            if rec_role and rec_role != source_role:
                raise DetectorStateValidationError(
                    f"run_id={run_id}: record role={rec_role!r} "
                    f"disagrees with evaluation_entry.source_role="
                    f"{source_role!r}"
                )
        for field in (
            "gs_secret_index", "gs_message_sha256", "gs_key_sha256",
            "gs_nonce_sha256", "gs_secret_bundle_sha256",
            "provider_config_hash",
        ):
            rec_val = str(record.get(field, ""))
            meta_val = str(canonical_metadata.get(field, ""))
            if rec_val and meta_val and rec_val != meta_val:
                raise DetectorStateValidationError(
                    f"run_id={run_id}: record.{field}={rec_val!r} "
                    f"disagrees with indexed metadata={meta_val!r}"
                )

    # ---- provider cache: construct once per source ----
    cache: dict[tuple[str, str], Any] = provider_info.get("provider_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        provider_info["provider_cache"] = cache

    cached = cache.get(source_key)
    if cached is not None:
        provider, provenance_record = cached
    else:
        pipe = provider_info["pipe"]
        GsProvider = provider_info["provider_class"]
        device_obj = provider_info["device_obj"]
        canonical_config = provider_info.get("canonical_config", {})

        provider, provenance_record = _construct_provider(
            pipe, GsProvider, device_obj, canonical_metadata, canonical_config,
        )
        cache[source_key] = (provider, provenance_record)

    # ---- canonical scoring via formal extraction helpers ----
    try:
        result = evaluate_image(torch, provider, provider_info["pipe"],
                                path, steps)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise DetectorScoringError(
            f"GS scoring failed for {image_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Validate required scoring outputs
    _validate_scoring_result(result, run_id)

    raw = _canonical_raw_score("GS", result)
    canonical = _canonical_canonical_score("GS", raw, result)

    # Cross-validate: raw_score must equal bit_accuracy
    bit_val = float(result["bit_accuracies"][0])
    if raw != bit_val:
        raise DetectorScoringError(
            f"run_id={run_id}: raw_score ({raw}) != bit_accuracy "
            f"({bit_val})"
        )

    decoded_str = result["message_bits_str_list"][0]
    decoded_sha = __import__("hashlib").sha256(
        decoded_str.encode("ascii")
    ).hexdigest()

    # ---- official thresholds (fail closed) ----
    try:
        thresholds = provider.official_thresholds()
    except Exception as exc:
        raise DetectorScoringError(
            f"run_id={run_id}: official_thresholds() failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    threshold_record = _validate_thresholds(thresholds, run_id)

    # ---- provider config identity ----
    detector_config_hash = provider_info.get(
        "detector_provider_config_hash", "")
    source_config_hash = str(canonical_metadata.get(
        "provider_config_hash", ""))

    # ---- build scored row ----
    scored: dict[str, Any] = {
        "raw_score": raw,
        "canonical_score": canonical,
        "bit_accuracy": float(bit_val),
        "decoded_bits_sha256": decoded_sha,
        "score_direction": "higher_is_watermarked",
        # per-sample secret provenance
        "gs_secret_index": provenance_record["gs_secret_index"],
        "gs_message_sha256": provenance_record["gs_message_sha256"],
        "gs_key_sha256": provenance_record["gs_key_sha256"],
        "gs_nonce_sha256": provenance_record["gs_nonce_sha256"],
        "gs_secret_bundle_sha256": provenance_record["gs_secret_bundle_sha256"],
        "gs_protocol_mode": provenance_record["gs_protocol_mode"],
        # explicit source/detector target/mask pairs
        "source_watermark_target_sha256": provenance_record[
            "source_watermark_target_sha256"],
        "detector_watermark_target_sha256": provenance_record[
            "detector_watermark_target_sha256"],
        "source_watermark_mask_sha256": provenance_record[
            "source_watermark_mask_sha256"],
        "detector_watermark_mask_sha256": provenance_record[
            "detector_watermark_mask_sha256"],
        # backwards-compatible merged fields
        "watermark_target_sha256": provenance_record[
            "detector_watermark_target_sha256"],
        "watermark_mask_sha256": provenance_record[
            "detector_watermark_mask_sha256"],
        # provider config identity
        "source_provider_config_hash": source_config_hash,
        "detector_provider_config_hash": detector_config_hash,
        # verification flags
        "gs_secret_verified": True,
        "gs_target_verified": True,
        "gs_mask_verified": True,
        "provider_config_verified": True,
        # official thresholds
        **threshold_record,
    }
    return scored


def aggregate(detector_rows: list[dict[str, Any]], **extra) -> dict[str, Any]:
    """Aggregate GS detector rows."""
    from raven.metrics import summarize_detection
    from . import ROW_STATUS_SCORED

    cohorts: dict[str, list[float]] = {}
    for row in detector_rows:
        if row.get("status") != ROW_STATUS_SCORED:
            continue
        cohort = row.get("evaluation_cohort", "")
        cs = row.get("canonical_score")
        if cs is not None and math.isfinite(float(cs)):
            cohorts.setdefault(cohort, []).append(float(cs))

    scored = sum(1 for r in detector_rows if r.get("status") == ROW_STATUS_SCORED)
    failed = len(detector_rows) - scored

    required = {"original_watermarked", "attacked_watermarked"}
    missing = sorted(required - set(cohorts))

    result: dict[str, Any] = {
        "method": "GS",
        "requested_count": len(detector_rows),
        "scored_count": scored,
        "failed_count": failed,
        "cohort_counts": {c: len(v) for c, v in cohorts.items()},
        "missing_cohorts": missing,
        "score_direction": "higher_is_watermarked",
    }

    clean = cohorts.get("original_clean", [])
    watermarked = cohorts.get("original_watermarked", [])
    attacked = cohorts.get("attacked_watermarked", [])

    if clean and watermarked and attacked:
        summary = summarize_detection(clean, watermarked, attacked, target_fpr=0.01)
        result["detection_summary"] = {
            "target_fpr": 0.01,
            "clean_calibrated_threshold": summary.calibration.threshold,
            "clean_calibrated_actual_fpr": summary.calibration.actual_fpr,
            "original_watermarked_tpr": summary.watermarked_tpr,
            "attacked_watermarked_tpr": summary.attacked_tpr,
            "attack_success": 1.0 - summary.attacked_tpr,
        }

    return result
