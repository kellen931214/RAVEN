#!/usr/bin/env python3
"""Deterministic experiment-result table updater.

Reads only structured JSON/JSONL result files under a completed run root and
upserts a single Markdown row into ``reports/runtime/experiment_results.md``.

Policy source: ``.agents/skills/raven-experiment-table/SKILL.md``.

This program never parses console output, never invents metrics, never
substitutes zero for a missing value, and never relabels a method-specific
detector metric with another method's definition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TABLE = Path("reports/runtime/experiment_results.md")

MISSING = "—"

COLUMNS = [
    "Finished UTC",
    "Method",
    "Dataset",
    "Experiment",
    "Stage",
    "N",
    "Attack",
    "Detector Metric",
    "Score Direction",
    "Threshold Type",
    "Threshold",
    "Nominal FPR",
    "Before Score",
    "After Score",
    "Before Detection Rate",
    "After Detection Rate",
    "Attack Success",
    "Empirical Clean FPR",
    "ROC-AUC",
    "FID",
    "CLIP",
    "PSNR",
    "SSIM",
    "Status",
    "Run Root",
]

IDENTITY_COLUMNS = ("Method", "Dataset", "Experiment", "Stage")

STAGE_WATERMARK_GENERATION = "watermark_generation"
STAGE_ATTACK_ONLY = "attack_only"
STAGE_EVALUATION = "evaluation"
STAGE_FORMAL_EVALUATION = "formal_evaluation"

STATUS_VALIDATED = "validated_formal_result"

QUALITY_DECIMALS = 6

# Console/log artifacts are never metric sources.
LOG_SUFFIXES = (".log", ".out", ".err", ".txt")


class UpdaterError(RuntimeError):
    """Fail-closed error: nothing is written to the table."""


class ConflictError(UpdaterError):
    """Two structured sources disagree about the same field."""


# --------------------------------------------------------------------------
# structured source loading
# --------------------------------------------------------------------------


@dataclass
class Sources:
    """Structured result files discovered under a run root."""

    run_root: Path
    files: dict[str, Path] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        return self.data.get(name)

    def path(self, name: str) -> Path | None:
        return self.files.get(name)

    def rel(self, name: str) -> str:
        path = self.files.get(name)
        return "<missing>" if path is None else relative_to_repo(path)


def load_json(path: Path) -> Any:
    if path.suffix.lower() in LOG_SUFFIXES:
        raise UpdaterError(f"refusing to parse a log file as a metric source: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise UpdaterError(f"invalid JSON in structured source {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise UpdaterError(
                    f"invalid JSONL in structured source {path} line {lineno}: {exc}"
                ) from exc
    return rows


def find_first(run_root: Path, relative_candidates: Iterable[str]) -> Path | None:
    """Return the first existing candidate path, searching shallow-first."""
    for candidate in relative_candidates:
        direct = run_root / candidate
        if direct.is_file():
            return direct
    for candidate in relative_candidates:
        name = Path(candidate).name
        matches = sorted(
            (p for p in run_root.rglob(name) if p.is_file()),
            key=lambda p: (len(p.relative_to(run_root).parts), str(p)),
        )
        if matches:
            return matches[0]
    return None


SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    # priority 1..9 per the recording policy
    "validated": ("VALIDATED.json",),
    "formal_aggregate": ("formal_aggregate.json",),
    "aggregate": ("aggregate_results.json", "aggregate.json"),
    "verification": (
        "verification/verification_result.json",
        "verification_result.json",
        "verification/verification_aggregate.json",
        "verification/detector.json",
    ),
    "verification_samples": (
        "verification/verification_records.jsonl",
        "verification/records.jsonl",
        "verification/scores.jsonl",
    ),
    "quality": (
        "metrics/quality_aggregate.json",
        "quality_aggregate.json",
        "metrics/quality.json",
    ),
    "fid": ("metrics/fid_aggregate.json", "fid_aggregate.json", "metrics/fid.json"),
    "clip": ("metrics/clip_aggregate.json", "clip_aggregate.json", "metrics/clip.json"),
    "run_config": ("run_config.json",),
    "generation_complete": (
        "generation_complete.json",
        "generation_completion.json",
        "watermark_generation_complete.json",
    ),
    "attack_complete": (
        "attack_complete.json",
        "attack_completion.json",
    ),
}


def collect_sources(run_root: Path) -> Sources:
    sources = Sources(run_root=run_root)
    for name, candidates in SOURCE_CANDIDATES.items():
        path = find_first(run_root, candidates)
        if path is None:
            continue
        sources.files[name] = path
        sources.data[name] = (
            load_jsonl(path) if path.suffix == ".jsonl" else load_json(path)
        )
    return sources


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def relative_to_repo(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def check_finite(value: Any, field_name: str, source: str) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise UpdaterError(
                f"non-finite value for {field_name!r} in {source}: {value!r}"
            )
    return value


def dig(obj: Any, *keys: str) -> Any:
    current = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_present(mapping: Any, keys: Iterable[str]) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def merge_scalar(
    field_name: str, candidates: list[tuple[str, Any]]
) -> Any:
    """Return one agreed value, or fail closed on structured-source conflict."""
    seen: list[tuple[str, Any]] = []
    for source, value in candidates:
        if value is None:
            continue
        check_finite(value, field_name, source)
        seen.append((source, value))
    if not seen:
        return None
    baseline = seen[0][1]
    conflicting = [item for item in seen if not values_equal(item[1], baseline)]
    if conflicting:
        detail = ", ".join(f"{src}:{val!r}" for src, val in seen)
        raise ConflictError(
            f"conflicting structured sources for field {field_name!r}: {detail}"
        )
    return baseline


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
        except (TypeError, ValueError):
            return left == right
    return left == right


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def fmt_value(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise UpdaterError(f"refusing to format non-finite value: {value!r}")
        return repr(value)
    text = str(value).strip()
    if not text:
        return MISSING
    return escape_pipes(text)


def fmt_quality(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise UpdaterError(f"refusing to format non-finite value: {value!r}")
        return f"{number:.{QUALITY_DECIMALS}f}"
    return fmt_value(value)


def escape_pipes(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


@dataclass
class Identity:
    method: str
    dataset: str
    experiment: str
    run_key: str
    stage: str

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        return (self.method, self.dataset, self.experiment, self.run_key, self.stage)

    def describe(self) -> str:
        return (
            f"method={self.method} dataset={self.dataset} "
            f"experiment={self.experiment} run_key={self.run_key} stage={self.stage}"
        )


GENERIC_SLUGS = {
    "formal",
    "experiment",
    "test",
    "run",
    "latest",
    "final",
    "new",
    "temp",
}


def resolve_method(sources: Sources) -> str:
    candidates = [
        (sources.rel("run_config"), dig(sources.get("run_config"), "method")),
        (sources.rel("formal_aggregate"), dig(sources.get("formal_aggregate"), "method")),
        (sources.rel("verification"), dig(sources.get("verification"), "method")),
        (sources.rel("aggregate"), dig(sources.get("aggregate"), "method")),
        (
            sources.rel("generation_complete"),
            dig(sources.get("generation_complete"), "method"),
        ),
        (sources.rel("attack_complete"), dig(sources.get("attack_complete"), "method")),
    ]
    method = merge_scalar("method", [(s, normalize_method(v)) for s, v in candidates])
    if method:
        return method
    inferred = method_from_path(sources.run_root)
    if inferred:
        return inferred
    raise UpdaterError(
        "required identity field 'method' is missing from all structured sources "
        f"under {relative_to_repo(sources.run_root)}"
    )


def normalize_method(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper()


def method_from_path(run_root: Path) -> str | None:
    parts = [part.lower() for part in Path(run_root).resolve().parts]
    if "outputs" in parts:
        index = parts.index("outputs")
        if index + 1 < len(parts) and parts[index + 1] in {"tr", "gs"}:
            return parts[index + 1].upper()
    return None


def resolve_dataset(sources: Sources) -> str:
    candidates = [
        (sources.rel("run_config"), dig(sources.get("run_config"), "dataset")),
        (
            sources.rel("formal_aggregate"),
            dig(sources.get("formal_aggregate"), "dataset"),
        ),
        (sources.rel("verification"), dig(sources.get("verification"), "dataset")),
        (sources.rel("aggregate"), dig(sources.get("aggregate"), "dataset")),
        (
            sources.rel("generation_complete"),
            dig(sources.get("generation_complete"), "dataset"),
        ),
        (
            sources.rel("attack_complete"),
            dig(sources.get("attack_complete"), "dataset"),
        ),
    ]
    dataset = merge_scalar("dataset", candidates)
    if isinstance(dataset, str) and dataset.strip():
        return dataset.strip()
    inferred = layout_component(sources.run_root, 2)
    if inferred:
        return inferred
    raise UpdaterError(
        "required identity field 'dataset' is missing from all structured sources "
        f"under {relative_to_repo(sources.run_root)}"
    )


def layout_component(run_root: Path, offset: int) -> str | None:
    """Return outputs/<method>/<dataset>/<slug>/<run-key> component at offset."""
    parts = Path(run_root).resolve().parts
    lowered = [part.lower() for part in parts]
    if "outputs" not in lowered:
        return None
    index = lowered.index("outputs")
    wanted = index + offset
    if wanted < len(parts):
        return parts[wanted]
    return None


def resolve_experiment(sources: Sources) -> str:
    for key in ("run_config", "formal_aggregate", "aggregate", "attack_complete",
                "generation_complete"):
        payload = sources.get(key)
        value = first_present(payload, ("experiment_name", "experiment", "variant"))
        if isinstance(value, str) and value.strip():
            return value.strip()
    from_path = layout_component(sources.run_root, 3)
    if from_path:
        return from_path
    parent = Path(sources.run_root).resolve().parent.name
    if parent and parent not in GENERIC_SLUGS:
        return parent
    raise UpdaterError(
        "required identity field 'experiment' could not be resolved for "
        f"{relative_to_repo(sources.run_root)}"
    )


def resolve_run_key(sources: Sources) -> str:
    for key in ("run_config", "formal_aggregate", "aggregate"):
        value = first_present(sources.get(key), ("run_key",))
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = Path(sources.run_root).resolve().name
    if not name:
        raise UpdaterError("required identity field 'run_key' could not be resolved")
    return name


def resolve_stage(sources: Sources) -> str:
    has_validated = "validated" in sources.data
    has_aggregate = "formal_aggregate" in sources.data or "aggregate" in sources.data

    if has_validated and not has_aggregate:
        raise UpdaterError(
            "fail-closed: VALIDATED.json present without a structured aggregate at "
            f"{sources.rel('validated')}"
        )
    if has_aggregate:
        explicit = merge_scalar(
            "stage",
            [
                (sources.rel("run_config"), dig(sources.get("run_config"), "stage")),
                (
                    sources.rel("formal_aggregate"),
                    dig(sources.get("formal_aggregate"), "stage"),
                ),
                (sources.rel("aggregate"), dig(sources.get("aggregate"), "stage")),
            ],
        )
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        return STAGE_FORMAL_EVALUATION if has_validated else STAGE_EVALUATION
    if "attack_complete" in sources.data:
        return STAGE_ATTACK_ONLY
    if "generation_complete" in sources.data:
        return STAGE_WATERMARK_GENERATION
    raise UpdaterError(
        "no structured completion record found under "
        f"{relative_to_repo(sources.run_root)}; refusing to record this run"
    )


# --------------------------------------------------------------------------
# detector extraction
# --------------------------------------------------------------------------


@dataclass
class DetectorMetrics:
    """Normalized detector row fields.

    This is the only shared schema. Every watermark method keeps its own raw
    detector schema and its own extractor; normalization happens here, after
    method-specific extraction, never before it. GS (bit accuracy against an
    official beta-tail threshold) and TR (complex-L1 TPR against an empirically
    calibrated clean threshold) are two registered families among others: any
    further method must register its own extractor rather than reuse either
    family's field meanings.
    """

    detector_metric: Any = None
    score_direction: Any = None
    threshold_type: Any = None
    threshold: Any = None
    nominal_fpr: Any = None
    before_score: Any = None
    after_score: Any = None
    before_detection_rate: Any = None
    after_detection_rate: Any = None
    attack_success: Any = None
    empirical_clean_fpr: Any = None
    roc_auc: Any = None


VALID_SCORE_DIRECTIONS = ("higher_is_watermarked", "lower_is_watermarked")


GS_REQUIRED_METRIC_KEYS = (
    "macro_bit_accuracy_before",
    "macro_bit_accuracy_attacked",
)

GS_OFFICIAL_KEYS = (
    "official_tau_onebit",
    "official_threshold_comparison_operator",
    "official_onebit_rates",
)


def gs_metric_block(sources: Sources) -> tuple[dict[str, Any], str] | None:
    for key in ("verification", "formal_aggregate", "aggregate"):
        payload = sources.get(key)
        block = dig(payload, "metric")
        if isinstance(block, dict):
            return block, sources.rel(key)
        if isinstance(payload, dict) and all(
            name in payload for name in GS_REQUIRED_METRIC_KEYS
        ):
            return payload, sources.rel(key)
    return None


def extract_gs_detector_metrics(sources: Sources) -> DetectorMetrics:
    found = gs_metric_block(sources)
    if found is None:
        return DetectorMetrics()
    metric, source = found

    missing = [key for key in GS_REQUIRED_METRIC_KEYS if key not in metric]
    if missing:
        raise UpdaterError(
            f"unknown GS detector schema in {source}: missing {sorted(missing)}"
        )
    official_missing = [key for key in GS_OFFICIAL_KEYS if key not in metric]
    if official_missing:
        raise UpdaterError(
            "unknown GS threshold schema in "
            f"{source}: missing official threshold fields {sorted(official_missing)}"
        )

    rates = metric.get("official_onebit_rates")
    if not isinstance(rates, dict):
        raise UpdaterError(
            f"unknown GS detector schema in {source}: official_onebit_rates is not an object"
        )
    operator = metric.get("official_threshold_comparison_operator")
    if operator not in (">", ">="):
        raise UpdaterError(
            f"unsupported GS threshold comparison operator {operator!r} in {source}"
        )

    result = DetectorMetrics(
        detector_metric="bit_accuracy",
        score_direction="higher_is_watermarked",
        threshold_type="official_beta_tail_tau_onebit",
        threshold=check_finite(metric.get("official_tau_onebit"), "official_tau_onebit", source),
        before_score=check_finite(
            metric.get("macro_bit_accuracy_before"), "macro_bit_accuracy_before", source
        ),
        after_score=check_finite(
            metric.get("macro_bit_accuracy_attacked"),
            "macro_bit_accuracy_attacked",
            source,
        ),
        before_detection_rate=check_finite(
            rates.get("watermarked"), "official_onebit_rates.watermarked", source
        ),
        after_detection_rate=check_finite(
            rates.get("attacked"), "official_onebit_rates.attacked", source
        ),
        roc_auc=check_finite(metric.get("attacked_roc_auc"), "attacked_roc_auc", source),
    )

    # Empirical clean FPR only when a clean-negative cohort was actually scored,
    # measured against the same official threshold reported above.
    clean_n = dig(metric, "stages", "clean", "N")
    if isinstance(clean_n, int) and clean_n > 0 and "clean" in rates:
        result.empirical_clean_fpr = check_finite(
            rates.get("clean"), "official_onebit_rates.clean", source
        )

    # Nominal (configured, theoretical) GS FPR, only when explicitly stored.
    result.nominal_fpr = gs_nominal_fpr(sources, metric)

    # Attack success only when the authoritative aggregate provides it.
    result.attack_success = authoritative_attack_success(sources)
    return result


def gs_nominal_fpr(sources: Sources, metric: dict[str, Any]) -> Any:
    explicit = first_present(metric, ("nominal_fpr", "gs_fpr", "official_nominal_fpr"))
    if explicit is not None:
        return check_finite(explicit, "nominal_fpr", "detector metric")
    params = dig(sources.get("verification"), "provenance", "provider_parameters")
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return None
    if isinstance(params, dict) and params.get("gs_fpr") is not None:
        return check_finite(params["gs_fpr"], "gs_fpr", sources.rel("verification"))
    return None


TR_PROTOCOL_KEYS = (
    "full_precision_protocol",
    "nfpa_rounded2_protocol",
)

TR_REQUIRED_PROTOCOL_KEYS = (
    "before_tpr",
    "attacked_tpr_at_original_clean_threshold",
)

TR_CLEAN_CALIBRATION_KEYS = (
    "original_clean_threshold",
    "original_clean_actual_fpr",
    "original_clean_target_fpr",
)


def tr_protocol_block(sources: Sources) -> tuple[dict[str, Any], str, str] | None:
    for key in ("aggregate", "formal_aggregate", "verification"):
        payload = sources.get(key)
        detector = dig(payload, "detector")
        container = detector if isinstance(detector, dict) else payload
        if not isinstance(container, dict):
            continue
        for name in TR_PROTOCOL_KEYS:
            block = container.get(name)
            if isinstance(block, dict):
                return block, name, sources.rel(key)
        if all(field in container for field in TR_REQUIRED_PROTOCOL_KEYS):
            return container, "detector", sources.rel(key)
    return None


def tr_detector_metric_name(sources: Sources, source: str) -> str:
    for key in ("aggregate", "formal_aggregate", "verification", "run_config"):
        payload = sources.get(key)
        explicit = first_present(payload, ("detector_metric",))
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        protocol = first_present(payload, ("detector_protocol",))
        if isinstance(protocol, str):
            lowered = protocol.lower()
            if "complex-l1" in lowered or "complex_l1" in lowered or "l1_complex" in lowered:
                return "l1_complex"
    raise UpdaterError(
        f"unknown TR detector metric: no detector_metric or recognized "
        f"detector_protocol recorded in {source}"
    )


def tr_score_direction(sources: Sources, source: str) -> str:
    for key in ("aggregate", "formal_aggregate", "verification", "run_config"):
        payload = sources.get(key)
        explicit = first_present(payload, ("score_direction",))
        if isinstance(explicit, str) and explicit.strip():
            return normalize_score_direction(explicit, source)
        protocol = first_present(payload, ("detector_protocol",))
        if isinstance(protocol, str):
            lowered = protocol.lower()
            if "score < threshold" in lowered or "lower" in lowered:
                return "lower_is_watermarked"
            if "score > threshold" in lowered or "higher" in lowered:
                return "higher_is_watermarked"
    raise UpdaterError(
        f"unknown TR score direction: not recorded in {source}"
    )


def normalize_score_direction(value: str, source: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"lower_is_watermarked", "higher_is_watermarked"}:
        return lowered
    if "lower" in lowered:
        return "lower_is_watermarked"
    if "higher" in lowered:
        return "higher_is_watermarked"
    raise UpdaterError(f"unknown score direction {value!r} in {source}")


def extract_tr_detector_metrics(sources: Sources) -> DetectorMetrics:
    found = tr_protocol_block(sources)
    if found is None:
        return DetectorMetrics()
    block, protocol_name, source = found

    missing = [key for key in TR_REQUIRED_PROTOCOL_KEYS if key not in block]
    if missing:
        raise UpdaterError(
            f"unknown TR detector schema in {source} ({protocol_name}): "
            f"missing {sorted(missing)}"
        )

    result = DetectorMetrics(
        detector_metric=tr_detector_metric_name(sources, source),
        score_direction=tr_score_direction(sources, source),
        before_score=check_finite(
            first_present(block, ("before_score", "macro_score_before")),
            "before_score",
            source,
        ),
        after_score=check_finite(
            first_present(block, ("after_score", "macro_score_attacked")),
            "after_score",
            source,
        ),
        before_detection_rate=check_finite(block.get("before_tpr"), "before_tpr", source),
        after_detection_rate=check_finite(
            block.get("attacked_tpr_at_original_clean_threshold"),
            "attacked_tpr_at_original_clean_threshold",
            source,
        ),
        roc_auc=check_finite(block.get("attacked_roc_auc"), "attacked_roc_auc", source),
    )

    calibrated = all(key in block for key in TR_CLEAN_CALIBRATION_KEYS)
    if calibrated:
        target = check_finite(
            block.get("original_clean_target_fpr"), "original_clean_target_fpr", source
        )
        result.threshold = check_finite(
            block.get("original_clean_threshold"), "original_clean_threshold", source
        )
        result.threshold_type = clean_threshold_type(target)
        result.empirical_clean_fpr = check_finite(
            block.get("original_clean_actual_fpr"), "original_clean_actual_fpr", source
        )
        result.nominal_fpr = target
    else:
        explicit_threshold = first_present(block, ("threshold", "before_threshold"))
        if explicit_threshold is not None:
            result.threshold = check_finite(explicit_threshold, "threshold", source)
            threshold_type = first_present(block, ("threshold_type",))
            if isinstance(threshold_type, str) and threshold_type.strip():
                result.threshold_type = threshold_type.strip()

    result.attack_success = authoritative_attack_success(sources, block)
    return result


def clean_threshold_type(target_fpr: Any) -> str:
    if isinstance(target_fpr, (int, float)) and not isinstance(target_fpr, bool):
        if math.isclose(float(target_fpr), 0.01, rel_tol=1e-12, abs_tol=1e-12):
            return "empirical_clean_1pct_fpr"
        return f"empirical_clean_fpr_target_{float(target_fpr)!r}"
    return "empirical_clean_fpr"


ATTACK_SUCCESS_KEYS = (
    "attack_success",
    "attack_success_rate",
    "attack_success_rate_at_recalibrated_threshold",
)


def authoritative_attack_success(
    sources: Sources, block: dict[str, Any] | None = None
) -> Any:
    if block is not None:
        value = first_present(block, ATTACK_SUCCESS_KEYS)
        if value is not None:
            return check_finite(value, "attack_success", "detector protocol block")
    for key in ("formal_aggregate", "aggregate", "verification"):
        payload = sources.get(key)
        value = first_present(payload, ATTACK_SUCCESS_KEYS)
        if value is not None:
            return check_finite(value, "attack_success", sources.rel(key))
        metric = dig(payload, "metric")
        value = first_present(metric, ATTACK_SUCCESS_KEYS)
        if value is not None:
            return check_finite(value, "attack_success", sources.rel(key))
    return None


DETECTOR_EXTRACTORS: dict[str, Any] = {
    "GS": extract_gs_detector_metrics,
    "TR": extract_tr_detector_metrics,
}


def register_detector_extractor(method: str, extractor: Any) -> None:
    """Register a method-specific extractor for an additional watermark method.

    The extractor receives ``Sources`` and returns ``DetectorMetrics``. It must
    read that method's own structured detector schema and must not borrow
    another method's metric, threshold family or detection-rate definition.
    """

    DETECTOR_EXTRACTORS[normalize_method(method) or method] = extractor


def extract_detector_metrics(method: str, sources: Sources) -> DetectorMetrics:
    extractor = DETECTOR_EXTRACTORS.get(method)
    if extractor is None:
        if has_detector_payload(sources):
            raise UpdaterError(
                f"no detector extractor registered for method {method!r}; "
                "refusing to guess detector fields. Register a method-specific "
                "extractor with register_detector_extractor()."
            )
        return DetectorMetrics()
    return validate_detector_metrics(method, extractor(sources))


def validate_detector_metrics(
    method: str, metrics: DetectorMetrics
) -> DetectorMetrics:
    """Enforce the shared contract every method-specific extractor must meet."""

    if not isinstance(metrics, DetectorMetrics):
        raise UpdaterError(
            f"detector extractor for {method!r} did not return DetectorMetrics"
        )
    direction = metrics.score_direction
    if direction is not None and direction not in VALID_SCORE_DIRECTIONS:
        raise UpdaterError(
            f"detector extractor for {method!r} returned unknown score direction "
            f"{direction!r}; expected one of {list(VALID_SCORE_DIRECTIONS)}"
        )
    if metrics.detector_metric is None and metrics.threshold is not None:
        raise UpdaterError(
            f"detector extractor for {method!r} reported a threshold without "
            "naming the detector metric it applies to"
        )
    if metrics.threshold is not None and metrics.threshold_type is None:
        raise UpdaterError(
            f"detector extractor for {method!r} reported a threshold without a "
            "threshold type; the threshold family must never be implied"
        )
    for name in (
        "threshold",
        "nominal_fpr",
        "before_score",
        "after_score",
        "before_detection_rate",
        "after_detection_rate",
        "attack_success",
        "empirical_clean_fpr",
        "roc_auc",
    ):
        check_finite(getattr(metrics, name), name, f"{method} detector extractor")
    return metrics


def has_detector_payload(sources: Sources) -> bool:
    for key in ("verification", "formal_aggregate", "aggregate"):
        payload = sources.get(key)
        if isinstance(payload, dict) and (
            "detector" in payload or "metric" in payload or "detector_result" in payload
        ):
            return True
    return False


# --------------------------------------------------------------------------
# quality metrics
# --------------------------------------------------------------------------


@dataclass
class QualityMetrics:
    fid: Any = None
    clip: Any = None
    psnr: Any = None
    ssim: Any = None


def extract_quality_metrics(sources: Sources) -> QualityMetrics:
    fid = merge_scalar(
        "fid",
        [
            (sources.rel("fid"), first_present(sources.get("fid"), ("value", "fid"))),
            (
                sources.rel("formal_aggregate"),
                dig(sources.get("formal_aggregate"), "fid_result", "value"),
            ),
            (sources.rel("aggregate"), dig(sources.get("aggregate"), "fid", "value")),
        ],
    )
    clip = merge_scalar(
        "clip",
        [
            (sources.rel("clip"), first_present(sources.get("clip"), ("mean", "clip"))),
            (
                sources.rel("formal_aggregate"),
                dig(sources.get("formal_aggregate"), "clip_result", "mean"),
            ),
            (sources.rel("aggregate"), dig(sources.get("aggregate"), "clip", "mean")),
        ],
    )
    psnr = merge_scalar(
        "psnr",
        [
            (
                sources.rel("quality"),
                first_present(sources.get("quality"), ("psnr_mean", "quality_psnr_mean")),
            ),
            (
                sources.rel("formal_aggregate"),
                first_present(
                    sources.get("formal_aggregate"),
                    ("quality_psnr_mean",),
                ),
            ),
            (
                sources.rel("aggregate"),
                first_present(sources.get("aggregate"), ("quality_psnr_mean",)),
            ),
        ],
    )
    ssim = merge_scalar(
        "ssim",
        [
            (
                sources.rel("quality"),
                first_present(sources.get("quality"), ("ssim_mean", "quality_ssim_mean")),
            ),
            (
                sources.rel("formal_aggregate"),
                first_present(
                    sources.get("formal_aggregate"),
                    ("quality_ssim_mean",),
                ),
            ),
            (
                sources.rel("aggregate"),
                first_present(sources.get("aggregate"), ("quality_ssim_mean",)),
            ),
        ],
    )
    return QualityMetrics(fid=fid, clip=clip, psnr=psnr, ssim=ssim)


# --------------------------------------------------------------------------
# row construction
# --------------------------------------------------------------------------


def resolve_sample_count(sources: Sources) -> Any:
    return merge_scalar(
        "N",
        [
            (
                sources.rel("formal_aggregate"),
                first_present(sources.get("formal_aggregate"), ("N", "sample_count")),
            ),
            (
                sources.rel("aggregate"),
                first_present(sources.get("aggregate"), ("N", "sample_count")),
            ),
            (
                sources.rel("validated"),
                first_present(sources.get("validated"), ("N", "sample_count")),
            ),
            (
                sources.rel("generation_complete"),
                first_present(sources.get("generation_complete"), ("N", "sample_count")),
            ),
            (
                sources.rel("attack_complete"),
                first_present(sources.get("attack_complete"), ("N", "sample_count")),
            ),
        ],
    )


def resolve_attack(sources: Sources) -> Any:
    for key in ("formal_aggregate", "aggregate", "run_config", "attack_complete"):
        payload = sources.get(key)
        config = first_present(payload, ("attack_config", "formal_attack_config"))
        name = first_present(config, ("variant_name", "attack_name", "variant"))
        config_hash = first_present(
            payload, ("attack_config_hash", "formal_attack_config_hash")
        )
        if name and config_hash:
            return f"{name} ({str(config_hash)[:12]})"
        if name:
            return str(name)
        if config_hash:
            return str(config_hash)[:12]
        variant = first_present(payload, ("variant",))
        if isinstance(variant, str) and variant.strip():
            return variant.strip()
    return None


def resolve_finished_utc(sources: Sources) -> Any:
    for key, fields in (
        ("validated", ("validated_utc", "finished_utc")),
        ("formal_aggregate", ("finished_utc", "created_utc", "completed_utc")),
        ("aggregate", ("finished_utc", "created_utc", "completed_utc")),
        ("attack_complete", ("finished_utc", "completed_utc", "created_utc")),
        ("generation_complete", ("finished_utc", "completed_utc", "created_utc")),
        ("run_config", ("finished_utc", "created_utc")),
    ):
        value = first_present(sources.get(key), fields)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_status(sources: Sources, stage: str) -> str:
    validated = sources.get("validated")
    if isinstance(validated, dict):
        status = validated.get("status")
        if isinstance(status, str) and status.strip().startswith("validated"):
            return STATUS_VALIDATED
    for key in ("attack_complete", "generation_complete", "formal_aggregate", "aggregate"):
        status = first_present(sources.get(key), ("status",))
        if isinstance(status, str) and status.strip():
            if status.strip() == STATUS_VALIDATED:
                # Only VALIDATED.json may grant validated_formal_result.
                continue
            return status.strip()
    return {
        STAGE_WATERMARK_GENERATION: "completed_generation",
        STAGE_ATTACK_ONLY: "completed_attack",
        STAGE_EVALUATION: "completed_evaluation",
        STAGE_FORMAL_EVALUATION: "completed_evaluation",
    }[stage]


def build_row(sources: Sources) -> tuple[Identity, dict[str, str]]:
    method = resolve_method(sources)
    dataset = resolve_dataset(sources)
    experiment = resolve_experiment(sources)
    run_key = resolve_run_key(sources)
    stage = resolve_stage(sources)
    identity = Identity(method, dataset, experiment, run_key, stage)

    detector = (
        DetectorMetrics()
        if stage in (STAGE_WATERMARK_GENERATION, STAGE_ATTACK_ONLY)
        else extract_detector_metrics(method, sources)
    )
    quality = (
        QualityMetrics()
        if stage == STAGE_WATERMARK_GENERATION
        else extract_quality_metrics(sources)
    )

    row = {
        "Finished UTC": fmt_value(resolve_finished_utc(sources)),
        "Method": fmt_value(method),
        "Dataset": fmt_value(dataset),
        "Experiment": fmt_value(experiment),
        "Stage": fmt_value(stage),
        "N": fmt_value(resolve_sample_count(sources)),
        "Attack": fmt_value(resolve_attack(sources)),
        "Detector Metric": fmt_value(detector.detector_metric),
        "Score Direction": fmt_value(detector.score_direction),
        "Threshold Type": fmt_value(detector.threshold_type),
        "Threshold": fmt_value(detector.threshold),
        "Nominal FPR": fmt_value(detector.nominal_fpr),
        "Before Score": fmt_value(detector.before_score),
        "After Score": fmt_value(detector.after_score),
        "Before Detection Rate": fmt_value(detector.before_detection_rate),
        "After Detection Rate": fmt_value(detector.after_detection_rate),
        "Attack Success": fmt_value(detector.attack_success),
        "Empirical Clean FPR": fmt_value(detector.empirical_clean_fpr),
        "ROC-AUC": fmt_value(detector.roc_auc),
        "FID": fmt_quality(quality.fid),
        "CLIP": fmt_quality(quality.clip),
        "PSNR": fmt_quality(quality.psnr),
        "SSIM": fmt_quality(quality.ssim),
        "Status": fmt_value(resolve_status(sources, stage)),
        "Run Root": fmt_value(relative_to_repo(sources.run_root)),
    }
    return identity, row


# --------------------------------------------------------------------------
# markdown table i/o
# --------------------------------------------------------------------------


TABLE_TITLE = "# RAVEN experiment results"

TABLE_NOTE = (
    "Generated by `experiments/update_experiment_table.py`. "
    "Rows are upserted on method + dataset + experiment + run key + stage. "
    "`—` means the metric was absent or not applicable; it never means zero."
)


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            current.append(char)
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def unescape_cell(text: str) -> str:
    return text.replace("\\|", "|").replace("\\\\", "\\")


def read_rows(table_path: Path) -> list[dict[str, str]]:
    if not table_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in table_path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = split_markdown_row(line)
        if len(cells) != len(COLUMNS):
            continue
        if cells == COLUMNS:
            continue
        if all(set(cell) <= {"-", ":"} and cell for cell in cells):
            continue
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


def row_identity(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    run_root = unescape_cell(row.get("Run Root", ""))
    run_key = Path(run_root).name if run_root and run_root != MISSING else ""
    values = [unescape_cell(row.get(column, "")) for column in IDENTITY_COLUMNS]
    return (values[0], values[1], values[2], run_key, values[3])


def render_table(rows: list[dict[str, str]]) -> str:
    lines = [TABLE_TITLE, "", TABLE_NOTE, ""]
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, MISSING) for col in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def atomic_write(table_path: Path, text: str) -> None:
    directory = table_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(directory),
        prefix=f".{table_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, table_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(dir_fd)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(dir_fd)


def upsert(table_path: Path, identity: Identity, row: dict[str, str]) -> str:
    rows = read_rows(table_path)
    target = identity.as_tuple()
    action = "inserted"
    for index, existing in enumerate(rows):
        if row_identity(existing) == target:
            rows[index] = row
            action = "updated"
            break
    else:
        rows.append(row)
    atomic_write(table_path, render_table(rows))
    return action


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------


def update_experiment_table(run_root: Path, table_path: Path) -> tuple[str, Identity]:
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise UpdaterError(f"run root does not exist or is not a directory: {run_root}")
    sources = collect_sources(run_root)
    identity, row = build_row(sources)
    action = upsert(Path(table_path), identity, row)
    return action, identity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically record a completed RAVEN experiment run."
    )
    parser.add_argument(
        "--run-root", required=True, type=Path, help="completed experiment run root"
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=REPO_ROOT / DEFAULT_TABLE,
        help=f"Markdown table path (default: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="also print the complete Markdown table after updating",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        action, identity = update_experiment_table(args.run_root, args.table)
    except UpdaterError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(action)
    print(relative_to_repo(args.table))
    print(identity.describe())
    if args.print_table:
        print(Path(args.table).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
