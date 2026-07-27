"""Regression tests for the aggregate-level quality and attack-success scalars.

Historical bug (2026-07-27): ``aggregate_stage`` recorded only the path and SHA
of ``quality_records.jsonl`` and published no attack-success field, so every GS
row in ``experiment_results.md`` rendered PSNR, SSIM and Attack Success as the
absent marker even though the per-sample records were complete and valid.

All fixtures are synthetic JSON. No GPU, no model, no image and no existing
formal output is read.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "raven_repro"))

from experiments.update_experiment_table import (  # noqa: E402
    MISSING,
    UpdaterError,
    read_rows,
    update_experiment_table,
)
from raven.eval_protocol import (  # noqa: E402
    GS_ATTACK_SUCCESS_FIELD,
    QUALITY_PSNR_FIELD,
    QUALITY_SSIM_FIELD,
    formal_quality_summary,
    gs_attack_success_summary,
    sha256_path,
)

BACKFILL = REPO / "experiments" / "backfill_formal_aggregate_metrics.py"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def quality_row(run_id, psnr, ssim, **overrides):
    row = {
        "run_id": str(run_id),
        "quality_reference": "watermarked input",
        "overlap_protocol": "inverse_warp_valid_correspondence",
        QUALITY_PSNR_FIELD: psnr,
        QUALITY_SSIM_FIELD: ssim,
        "overlap_psnr": psnr,
        "overlap_ssim": ssim,
    }
    row.update(overrides)
    return row


def write_quality(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def gs_metric(attacked_rate=0.0, **overrides):
    metric = {
        "N": 3,
        "num_bits_per_sample": 256,
        "stages": {
            "clean": {"N": 3, "macro_bit_accuracy": 0.5},
            "watermarked": {"N": 3, "macro_bit_accuracy": 1.0},
            "attacked": {"N": 3, "macro_bit_accuracy": 0.5037},
        },
        "macro_bit_accuracy_before": 1.0,
        "macro_bit_accuracy_attacked": 0.5037,
        "official_tau_onebit": 0.6484375,
        "official_tau_bits": 0.71484375,
        "official_threshold_comparison_operator": ">=",
        "official_onebit_rates": {
            "clean": 0.0, "watermarked": 1.0, "attacked": attacked_rate
        },
        "attacked_roc_auc": 0.5152,
    }
    metric.update(overrides)
    return metric


# ---------------------------------------------------------------------------
# formal_quality_summary
# ---------------------------------------------------------------------------


def test_quality_summary_means_match_manual_reduction(tmp_path):
    rows = [quality_row(i, 20.0 + i, 0.80 + i / 100) for i in range(3)]
    path = write_quality(tmp_path / "quality_records.jsonl", rows)
    summary = formal_quality_summary(path, expected_count=3)
    assert summary["quality_count"] == 3
    assert summary["quality_psnr_mean"] == pytest.approx(21.0, abs=1e-12)
    assert summary["quality_ssim_mean"] == pytest.approx(0.81, abs=1e-12)
    # The reduction must name the post-color-vs-watermarked definition, not the
    # bare overlap aliases, so the metric identity is visible in the aggregate.
    assert summary["quality_psnr_field"] == QUALITY_PSNR_FIELD
    assert summary["quality_ssim_field"] == QUALITY_SSIM_FIELD
    assert summary["quality_reference"] == "watermarked input"
    assert summary["quality_records_sha256"] == sha256_path(path)


def test_quality_summary_reads_the_named_field_not_the_alias(tmp_path):
    # A record whose alias drifted from the named field must not be averaged
    # through the alias.
    rows = [quality_row(0, 20.0, 0.8), quality_row(1, 30.0, 0.9)]
    rows[0]["overlap_psnr"] = 999.0
    rows[0]["overlap_ssim"] = 0.0
    path = write_quality(tmp_path / "quality_records.jsonl", rows)
    summary = formal_quality_summary(path, expected_count=2)
    assert summary["quality_psnr_mean"] == pytest.approx(25.0, abs=1e-12)
    assert summary["quality_ssim_mean"] == pytest.approx(0.85, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs, rows, match",
    [
        ({"expected_count": 4}, [quality_row(i, 20.0, 0.8) for i in range(3)], "count mismatch"),
        (
            {"expected_count": 2},
            [quality_row(0, 20.0, 0.8), quality_row(0, 21.0, 0.8)],
            "duplicate run_id",
        ),
        (
            {"expected_count": 2, "expected_run_ids": {"0", "7"}},
            [quality_row(0, 20.0, 0.8), quality_row(1, 21.0, 0.8)],
            "run-ID set differs",
        ),
        (
            {"expected_count": 1},
            [quality_row(0, float("nan"), 0.8)],
            "non-finite",
        ),
        (
            {"expected_count": 1},
            [quality_row(0, float("inf"), 0.8)],
            "non-finite",
        ),
        (
            {"expected_count": 1},
            [quality_row(0, 20.0, 0.8, quality_reference="paired clean image")],
            "quality reference",
        ),
    ],
)
def test_quality_summary_fails_closed(tmp_path, kwargs, rows, match):
    path = write_quality(tmp_path / "quality_records.jsonl", rows)
    with pytest.raises(RuntimeError, match=match):
        formal_quality_summary(path, **kwargs)


def test_quality_summary_rejects_sha_drift(tmp_path):
    path = write_quality(tmp_path / "quality_records.jsonl", [quality_row(0, 20.0, 0.8)])
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        formal_quality_summary(
            path, expected_count=1, expected_records_sha256="0" * 64
        )


def test_quality_summary_rejects_missing_named_field(tmp_path):
    row = quality_row(0, 20.0, 0.8)
    del row[QUALITY_PSNR_FIELD]
    path = write_quality(tmp_path / "quality_records.jsonl", [row])
    with pytest.raises(KeyError):
        formal_quality_summary(path, expected_count=1)


# ---------------------------------------------------------------------------
# gs_attack_success_summary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attacked_rate", [0.0, 0.017, 0.5, 1.0])
def test_gs_attack_success_is_one_minus_official_onebit_attacked(attacked_rate):
    summary = gs_attack_success_summary({"metric": gs_metric(attacked_rate)})
    assert summary[GS_ATTACK_SUCCESS_FIELD] == pytest.approx(1.0 - attacked_rate, abs=1e-12)
    assert summary["attack_success_threshold_type"] == "official_beta_tail_tau_onebit"
    assert summary["attack_success_threshold"] == 0.6484375
    assert summary["attack_success_threshold_comparison_operator"] == ">="
    assert summary["attack_success_detected_rate"] == attacked_rate


def test_gs_attack_success_accepts_a_bare_metric_block():
    assert gs_attack_success_summary(gs_metric(0.25))[GS_ATTACK_SUCCESS_FIELD] == 0.75


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"metric": {"official_tau_onebit": 0.6}}, "official_onebit_rates"),
        (
            {"metric": gs_metric(0.0, official_threshold_comparison_operator="~")},
            "operator",
        ),
        (
            {"metric": gs_metric(0.0, official_onebit_rates={"attacked": 1.5})},
            "invalid official_onebit_rates.attacked",
        ),
        (
            {"metric": gs_metric(0.0, official_onebit_rates={"attacked": float("nan")})},
            "invalid official_onebit_rates.attacked",
        ),
    ],
)
def test_gs_attack_success_fails_closed(payload, match):
    with pytest.raises(RuntimeError, match=match):
        gs_attack_success_summary(payload)


# ---------------------------------------------------------------------------
# table updater
# ---------------------------------------------------------------------------


def make_run(tmp_path: Path, *, aggregate_extra, attacked_rate=0.0):
    root = tmp_path / "outputs" / "gs" / "diffusiondb" / "variant" / "rk"
    (root / "verification").mkdir(parents=True)
    (root / "run_config.json").write_text(
        json.dumps({"method": "GS", "dataset": "diffusiondb"}), encoding="utf-8"
    )
    (root / "verification" / "verification_result.json").write_text(
        json.dumps(
            {
                "method": "GS",
                "dataset": "diffusiondb",
                "provenance": {
                    "method": "GS",
                    "score_direction": "higher bit accuracy means watermark",
                    "provider_parameters": json.dumps({"gs_fpr": 1e-06}),
                },
                "metric": gs_metric(attacked_rate),
            }
        ),
        encoding="utf-8",
    )
    (root / "formal_aggregate.json").write_text(
        json.dumps(
            {
                "method": "GS",
                "dataset": "diffusiondb",
                "N": 3,
                "attack_config": {"variant_name": "unit"},
                "attack_config_hash": "a" * 12,
                "fid_result": {"value": 23.0},
                "clip_result": {"mean": 0.46},
                **aggregate_extra,
            }
        ),
        encoding="utf-8",
    )
    (root / "VALIDATED.json").write_text(
        json.dumps(
            {"status": "validated_formal_result", "validated_utc": "2026-07-27T00:00:00Z"}
        ),
        encoding="utf-8",
    )
    return root


def test_table_renders_the_backfilled_scalars(tmp_path):
    root = make_run(
        tmp_path,
        attacked_rate=0.017,
        aggregate_extra={
            "quality_psnr_mean": 28.5,
            "quality_ssim_mean": 0.923,
            GS_ATTACK_SUCCESS_FIELD: 0.983,
        },
    )
    table = tmp_path / "experiment_results.md"
    update_experiment_table(root, table)
    row = read_rows(table)[0]
    assert float(row["PSNR"]) == 28.5
    assert float(row["SSIM"]) == 0.923
    assert row["Attack Success"] == "0.983"
    assert row["After Detection Rate"] == "0.017"


def test_table_still_marks_absent_scalars_rather_than_zero(tmp_path):
    # A run aggregated before the fix must render the absent marker, never 0.0.
    root = make_run(tmp_path, aggregate_extra={})
    table = tmp_path / "experiment_results.md"
    update_experiment_table(root, table)
    row = read_rows(table)[0]
    assert row["PSNR"] == MISSING
    assert row["SSIM"] == MISSING
    assert row["Attack Success"] == MISSING


def test_table_rejects_attack_success_that_contradicts_the_detection_rate(tmp_path):
    root = make_run(
        tmp_path,
        attacked_rate=0.017,
        # A TR-style recalibrated rate pasted into a GS aggregate.
        aggregate_extra={GS_ATTACK_SUCCESS_FIELD: 0.857},
    )
    with pytest.raises(UpdaterError, match="not 1 - official_onebit_rates.attacked"):
        update_experiment_table(root, tmp_path / "experiment_results.md")


# ---------------------------------------------------------------------------
# backfill tool
# ---------------------------------------------------------------------------


def gs_metric_n(attacked_rate, n):
    metric = gs_metric(attacked_rate, N=n)
    metric["stages"] = {
        key: {**value, "N": n} for key, value in metric["stages"].items()
    }
    return metric


def make_backfillable_run(tmp_path: Path, *, attacked_rate=0.0, psnr=(20.0, 30.0)):
    root = tmp_path / "run"
    root.mkdir(parents=True)
    quality = write_quality(
        root / "metrics" / "quality" / "qh" / "quality_records.jsonl",
        [quality_row(i, value, 0.9) for i, value in enumerate(psnr)],
    )
    detector = root / "verification" / "verification_result.json"
    detector.parent.mkdir(parents=True, exist_ok=True)
    detector.write_text(
        json.dumps({"metric": gs_metric_n(attacked_rate, len(psnr))}), encoding="utf-8"
    )
    (root / "attack_records_watermarked.jsonl").write_text(
        "".join(json.dumps({"run_id": str(i)}) + "\n" for i in range(len(psnr))),
        encoding="utf-8",
    )
    (root / "VALIDATED.json").write_text(json.dumps({"status": "validated_formal_result"}))
    (root / "formal_aggregate.json").write_text(
        json.dumps(
            {
                "method": "GS",
                "N": len(psnr),
                "quality_records": str(quality),
                "quality_records_sha256": sha256_path(quality),
                "detector_result": str(detector),
                "detector_result_sha256": sha256_path(detector),
            }
        ),
        encoding="utf-8",
    )
    return root


def run_backfill(root: Path, *extra):
    return subprocess.run(
        [sys.executable, str(BACKFILL), str(root), *extra],
        capture_output=True, text=True, cwd=REPO,
    )


def test_backfill_writes_the_recomputed_scalars(tmp_path):
    root = make_backfillable_run(tmp_path, attacked_rate=0.017)
    assert run_backfill(root, "--apply").returncode == 0
    payload = json.loads((root / "formal_aggregate.json").read_text())
    assert payload["quality_psnr_mean"] == pytest.approx(25.0, abs=1e-12)
    assert payload["quality_ssim_mean"] == pytest.approx(0.9, abs=1e-12)
    assert payload[GS_ATTACK_SUCCESS_FIELD] == pytest.approx(0.983, abs=1e-12)
    assert payload["aggregate_scalar_backfill"][0]["fields"]


def test_backfill_without_apply_leaves_the_aggregate_untouched(tmp_path):
    root = make_backfillable_run(tmp_path)
    before = (root / "formal_aggregate.json").read_text()
    assert run_backfill(root).returncode == 0
    assert (root / "formal_aggregate.json").read_text() == before


def test_backfill_is_idempotent(tmp_path):
    root = make_backfillable_run(tmp_path)
    assert run_backfill(root, "--apply").returncode == 0
    first = (root / "formal_aggregate.json").read_text()
    assert run_backfill(root, "--apply").returncode == 0
    assert (root / "formal_aggregate.json").read_text() == first


def test_backfill_rejects_quality_records_that_no_longer_hash_correctly(tmp_path):
    root = make_backfillable_run(tmp_path)
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    Path(aggregate["quality_records"]).write_text(
        json.dumps(quality_row(0, 99.0, 0.9), sort_keys=True) + "\n"
        + json.dumps(quality_row(1, 99.0, 0.9), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = run_backfill(root, "--apply")
    assert result.returncode != 0
    assert "SHA mismatch" in result.stderr


def test_backfill_rejects_detector_payload_drift(tmp_path):
    root = make_backfillable_run(tmp_path)
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    Path(aggregate["detector_result"]).write_text(
        json.dumps({"metric": gs_metric_n(0.9, 2)}), encoding="utf-8"
    )
    result = run_backfill(root, "--apply")
    assert result.returncode != 0
    assert "detector result SHA mismatch" in result.stderr


def test_backfill_rejects_a_cohort_size_disagreement(tmp_path):
    root = make_backfillable_run(tmp_path)
    (root / "attack_records_watermarked.jsonl").write_text(
        json.dumps({"run_id": "0"}) + "\n", encoding="utf-8"
    )
    result = run_backfill(root, "--apply")
    assert result.returncode != 0
    assert "aggregate N" in result.stderr


def test_backfill_refuses_to_overwrite_a_conflicting_stored_scalar(tmp_path):
    root = make_backfillable_run(tmp_path)
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    aggregate["quality_psnr_mean"] = 99.0
    (root / "formal_aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    result = run_backfill(root, "--apply")
    assert result.returncode != 0
    assert "conflict" in result.stderr


def test_backfill_requires_a_validated_run(tmp_path):
    root = make_backfillable_run(tmp_path)
    (root / "VALIDATED.json").unlink()
    result = run_backfill(root, "--apply")
    assert result.returncode != 0
    assert "not a validated formal run" in result.stderr


def test_backfilled_scalars_equal_the_aggregate_stage_reducers(tmp_path):
    """A backfilled run and a freshly aggregated run must not disagree."""
    root = make_backfillable_run(tmp_path, attacked_rate=0.25)
    assert run_backfill(root, "--apply").returncode == 0
    payload = json.loads((root / "formal_aggregate.json").read_text())
    fresh_quality = formal_quality_summary(payload["quality_records"], expected_count=2)
    fresh_attack = gs_attack_success_summary(
        json.loads(Path(payload["detector_result"]).read_text())
    )
    for key, value in {**fresh_quality, **fresh_attack}.items():
        assert payload[key] == value or (
            isinstance(value, float) and math.isclose(payload[key], value, abs_tol=1e-12)
        )
