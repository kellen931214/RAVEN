"""Bit Accuracy extraction, reporting, aggregation, and cache versioning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


METRIC_SCHEMA_VERSION = 2
STATUS_OK = "OK"
STATUS_NOT_AVAILABLE = "N/A"
STATUS_ERROR = "ERROR"


@dataclass(frozen=True)
class BitAccuracyMetric:
    value: Optional[float]
    status: str
    error: Optional[str] = None


def extract_bit_accuracy(accuracy_results: Mapping[str, Any]) -> BitAccuracyMetric:
    """Return a real provider metric, or N/A when the provider has no decoder."""
    values = accuracy_results.get("bit_accuracies")
    if values is None:
        return BitAccuracyMetric(None, STATUS_NOT_AVAILABLE)

    try:
        value = float(values[0])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return BitAccuracyMetric(None, STATUS_ERROR, f"invalid bit_accuracies: {exc}")

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return BitAccuracyMetric(
            None,
            STATUS_ERROR,
            f"bit accuracy must be finite and within [0, 1], got {value!r}",
        )
    return BitAccuracyMetric(value, STATUS_OK)


def run_bit_decoder(decoder: Callable[[], Mapping[str, Any]]) -> BitAccuracyMetric:
    """Convert a decoder exception into an explicit ERROR metric."""
    try:
        return extract_bit_accuracy(decoder())
    except Exception as exc:  # decoder implementations can raise library-specific errors
        return BitAccuracyMetric(None, STATUS_ERROR, f"{type(exc).__name__}: {exc}")


def format_bit_accuracy(metric: BitAccuracyMetric) -> str:
    if metric.status == STATUS_OK and metric.value is not None:
        return f"{metric.value:.4f}"
    return metric.status


def format_bit_accuracy_value(
    value: Optional[float], status: str, error: Optional[str] = None
) -> str:
    return format_bit_accuracy(BitAccuracyMetric(value, status, error))


def summarize_bit_accuracy(metrics: Iterable[BitAccuracyMetric]) -> dict[str, Any]:
    """Average valid measurements only; keep decoder errors separate."""
    items = list(metrics)
    values = [item.value for item in items if item.status == STATUS_OK and item.value is not None]
    error_count = sum(item.status == STATUS_ERROR for item in items)
    if values:
        metric = BitAccuracyMetric(sum(values) / len(values), STATUS_OK)
    elif error_count:
        metric = BitAccuracyMetric(None, STATUS_ERROR)
    else:
        metric = BitAccuracyMetric(None, STATUS_NOT_AVAILABLE)
    return {
        "value": metric.value,
        "status": metric.status,
        "display": format_bit_accuracy(metric),
        "valid_count": len(values),
        "error_count": error_count,
    }


def format_staged_bit_accuracy_rows(
    rows: Sequence[Mapping[str, Any]], stages: Sequence[str] = ("before", "after")
) -> list[dict[str, Any]]:
    """Prepare table/CSV rows while keeping Before and After independent."""
    formatted_rows = []
    for row in rows:
        formatted = dict(row)
        for stage in stages:
            formatted[f"{stage}_bit_accuracy"] = format_bit_accuracy_value(
                row.get(f"{stage}_bit_accuracy"),
                row.get(f"{stage}_bit_accuracy_status", STATUS_NOT_AVAILABLE),
                row.get(f"{stage}_bit_accuracy_error"),
            )
        formatted_rows.append(formatted)
    return formatted_rows


def format_bit_accuracy_rows(
    rows: Sequence[Mapping[str, Any]], key: str = "bit_accuracy"
) -> list[dict[str, Any]]:
    """Format an unprefixed Bit Accuracy column for CSV output."""
    formatted_rows = []
    for row in rows:
        formatted = dict(row)
        formatted[key] = format_bit_accuracy_value(
            row.get(key),
            row.get(f"{key}_status", STATUS_NOT_AVAILABLE),
            row.get(f"{key}_error"),
        )
        formatted_rows.append(formatted)
    return formatted_rows


def cache_has_current_metric_schema(payload: Mapping[str, Any]) -> bool:
    """Old cache entries without the current schema must be recomputed."""
    return payload.get("metric_schema_version") == METRIC_SCHEMA_VERSION


def add_metric_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    versioned = dict(payload)
    versioned["metric_schema_version"] = METRIC_SCHEMA_VERSION
    return versioned
