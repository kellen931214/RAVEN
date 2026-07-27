"""Unit tests for the deterministic experiment-table updater.

All fixtures are synthetic structured JSON. No GPU, no model, and no existing
formal output is read.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.update_experiment_table import (  # noqa: E402
    COLUMNS,
    MISSING,
    STATUS_VALIDATED,
    DetectorMetrics,
    UpdaterError,
    ConflictError,
    read_rows,
    register_detector_extractor,
    update_experiment_table,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def gs_metric_block(**overrides):
    metric = {
        "N": 10,
        "num_bits_per_sample": 256,
        "stages": {
            "clean": {"N": 10, "macro_bit_accuracy": 0.5066},
            "watermarked": {"N": 10, "macro_bit_accuracy": 1.0},
            "attacked": {"N": 10, "macro_bit_accuracy": 0.5015625},
        },
        "macro_bit_accuracy_before": 1.0,
        "macro_bit_accuracy_attacked": 0.5015625,
        "official_tau_onebit": 0.6484375,
        "official_tau_bits": 0.71484375,
        "official_threshold_comparison_operator": ">=",
        "official_onebit_rates": {"clean": 0.0, "watermarked": 1.0, "attacked": 0.0},
    }
    metric.update(overrides)
    return metric


def make_gs_run(tmp_path: Path, *, run_key="rk_gs_1", experiment="shared_clean_gs_from_tr",
                metric=None, clean_stage=True, provider_fpr=1e-06):
    root = tmp_path / "outputs" / "gs" / "diffusiondb" / experiment / run_key
    root.mkdir(parents=True)
    write_json(root / "run_config.json", {"method": "GS", "dataset": "diffusiondb"})
    block = gs_metric_block() if metric is None else metric
    if not clean_stage:
        block = dict(block)
        block["stages"] = {
            key: value for key, value in block["stages"].items() if key != "clean"
        }
        block["official_onebit_rates"] = {
            key: value
            for key, value in block["official_onebit_rates"].items()
            if key != "clean"
        }
    provenance = {
        "method": "GS",
        "score_direction": "higher bit accuracy means watermark",
        "provider_parameters": json.dumps({"gs_fpr": provider_fpr}),
    }
    write_json(
        root / "verification" / "verification_result.json",
        {"method": "GS", "dataset": "diffusiondb", "provenance": provenance,
         "metric": block},
    )
    write_json(
        root / "formal_aggregate.json",
        {
            "method": "GS",
            "dataset": "diffusiondb",
            "N": 10,
            "attack_config": {"variant_name": "formal_baseline"},
            "attack_config_hash": "20c33008fba829b580cf5ab1b6defa9b5f7c0a45c",
            "fid_result": {"value": 66.36420818661253},
            "clip_result": {"mean": 0.4389688760042191},
            "quality_psnr_mean": 30.25,
            "quality_ssim_mean": 0.91,
        },
    )
    write_json(
        root / "VALIDATED.json",
        {"status": "validated_formal_result", "validated_utc": "2026-07-25T06:38:39Z"},
    )
    return root


def tr_protocol_block(**overrides):
    block = {
        "before_tpr": 1.0,
        "attacked_tpr_at_original_clean_threshold": 0.1048951048951049,
        "attacked_tpr_at_attacked_clean_recalibrated_threshold": 0.14285714285714285,
        "attack_success_rate_at_recalibrated_threshold": 0.8571428571428572,
        "attacked_roc_auc": 0.8413724137999862,
        "original_clean_threshold": 75.68119049072266,
        "original_clean_actual_fpr": 0.008991008991008992,
        "original_clean_target_fpr": 0.01,
    }
    block.update(overrides)
    return block


def make_tr_run(tmp_path: Path, *, run_key="rk_tr_1",
                experiment="ddim_nearest_reflection_aligned_color", block=None,
                validated=True):
    root = tmp_path / "outputs" / "tr" / "diffusiondb" / experiment / run_key
    root.mkdir(parents=True)
    write_json(root / "run_config.json", {"method": "TR", "dataset": "diffusiondb"})
    write_json(
        root / "aggregate_results.json",
        {
            "method": "TR",
            "dataset": "diffusiondb",
            "sample_count": 1001,
            "created_utc": "2026-07-20T15:20:28Z",
            "detector_protocol": "formal Tree-Ring complex-L1, strict score < threshold",
            "detector": {
                "N": 1001,
                "full_precision_protocol": tr_protocol_block() if block is None else block,
            },
            "formal_attack_config": {"variant_name": "shift_aligned_color_transfer"},
            "formal_attack_config_hash": "554518d0f1bab8e96769da290809f6cd60fd05c1",
            "fid": {"value": 23.430311208461887},
            "clip": {"mean": 0.4606021070754254},
            "quality_psnr_mean": 30.489567385243156,
            "quality_ssim_mean": 0.9161351040883974,
        },
    )
    if validated:
        write_json(
            root / "VALIDATED.json",
            {
                "status": "validated_aligned_color_evaluation",
                "validated_utc": "2026-07-20T15:21:00Z",
            },
        )
    return root


def row_by_experiment(table: Path, experiment: str, stage: str | None = None):
    rows = [row for row in read_rows(table) if row["Experiment"] == experiment]
    if stage is not None:
        rows = [row for row in rows if row["Stage"] == stage]
    assert rows, f"no row for experiment={experiment} stage={stage}"
    assert len(rows) == 1, f"expected one row, found {len(rows)}"
    return rows[0]


# ---------------------------------------------------------------------------
# core table behavior
# ---------------------------------------------------------------------------


def test_first_insert_creates_table_with_exact_columns(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "reports" / "runtime" / "experiment_results.md"

    action, _table, identity = update_experiment_table(root, table)

    assert action == "inserted"
    assert identity.as_tuple() == (
        "GS",
        "diffusiondb",
        "shared_clean_gs_from_tr",
        "rk_gs_1",
        "formal_evaluation",
    )
    header = [line for line in table.read_text().splitlines() if line.startswith("| ")][0]
    assert header == "| " + " | ".join(COLUMNS) + " |"
    assert len(read_rows(table)) == 1


def test_idempotent_update_does_not_duplicate(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"

    assert update_experiment_table(root, table)[0] == "inserted"
    assert update_experiment_table(root, table)[0] == "updated"
    assert update_experiment_table(root, table)[0] == "updated"

    assert len(read_rows(table)) == 1


def test_same_run_processed_twice_keeps_one_row_with_new_values(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    aggregate["fid_result"]["value"] = 42.5
    write_json(root / "formal_aggregate.json", aggregate)
    action, _table, _identity = update_experiment_table(root, table)

    rows = read_rows(table)
    assert action == "updated"
    assert len(rows) == 1
    assert rows[0]["FID"] == "42.500000"


def test_different_experiment_creates_new_row(tmp_path):
    table = tmp_path / "table.md"
    first = make_gs_run(tmp_path, experiment="shared_clean_gs_from_tr")
    second = make_gs_run(tmp_path, experiment="zero_shift_ablation", run_key="rk_gs_2")

    update_experiment_table(first, table)
    update_experiment_table(second, table)

    experiments = {row["Experiment"] for row in read_rows(table)}
    assert experiments == {"shared_clean_gs_from_tr", "zero_shift_ablation"}


def test_different_stage_creates_new_row(tmp_path):
    table = tmp_path / "table.md"
    evaluation_root = make_gs_run(tmp_path)
    update_experiment_table(evaluation_root, table)

    attack_root = (
        tmp_path / "outputs" / "gs" / "diffusiondb" / "shared_clean_gs_from_tr" / "rk_gs_a"
    )
    attack_root.mkdir(parents=True)
    write_json(
        attack_root / "attack_complete.json",
        {"method": "GS", "dataset": "diffusiondb", "N": 10,
         "status": "completed_attack", "finished_utc": "2026-07-26T00:00:00Z"},
    )
    update_experiment_table(attack_root, table)

    stages = sorted(row["Stage"] for row in read_rows(table))
    assert stages == ["attack_only", "formal_evaluation"]


def test_missing_optional_metric_is_em_dash_not_zero(tmp_path):
    root = make_gs_run(tmp_path)
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    del aggregate["fid_result"]
    del aggregate["quality_ssim_mean"]
    write_json(root / "formal_aggregate.json", aggregate)
    table = tmp_path / "table.md"

    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["FID"] == MISSING
    assert row["SSIM"] == MISSING
    assert row["FID"] != "0"
    assert row["Attack Success"] == MISSING


def test_missing_required_identity_fails(tmp_path):
    root = tmp_path / "somewhere" / "unlabelled_run"
    root.mkdir(parents=True)
    write_json(root / "attack_complete.json", {"status": "completed_attack"})
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="method"):
        update_experiment_table(root, table)
    assert not table.exists()


def test_no_structured_completion_record_fails(tmp_path):
    root = tmp_path / "outputs" / "gs" / "diffusiondb" / "slug" / "rk"
    root.mkdir(parents=True)
    write_json(root / "run_config.json", {"method": "GS", "dataset": "diffusiondb"})
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="no structured completion record"):
        update_experiment_table(root, table)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_detector_value_fails_closed(tmp_path, bad):
    root = make_gs_run(tmp_path, metric=gs_metric_block(macro_bit_accuracy_attacked=bad))
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError):
        update_experiment_table(root, table)
    assert not table.exists()


def test_repository_relative_run_root_when_inside_repo(tmp_path):
    import experiments.update_experiment_table as updater

    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"
    original = updater.REPO_ROOT
    updater.REPO_ROOT = tmp_path
    try:
        update_experiment_table(root, table)
    finally:
        updater.REPO_ROOT = original

    row = read_rows(table)[0]
    assert row["Run Root"] == (
        "outputs/gs/diffusiondb/shared_clean_gs_from_tr/rk_gs_1"
    )
    assert not row["Run Root"].startswith("/")


def test_atomic_write_leaves_no_temp_files_and_preserves_prior_table(tmp_path):
    table = tmp_path / "reports" / "runtime" / "experiment_results.md"
    first = make_gs_run(tmp_path)
    update_experiment_table(first, table)
    before = table.read_text()

    broken = make_gs_run(
        tmp_path,
        run_key="rk_broken",
        experiment="broken_run",
        metric=gs_metric_block(macro_bit_accuracy_before=float("nan")),
    )
    with pytest.raises(UpdaterError):
        update_experiment_table(broken, table)

    assert table.read_text() == before
    leftovers = [p.name for p in table.parent.iterdir() if p.name != table.name]
    assert leftovers == []


def test_console_logs_are_never_parsed_for_metrics(tmp_path):
    root = make_gs_run(tmp_path)
    (root / "launcher.log").write_text("FID: 999.0\nbit accuracy: 0.99\n")
    (root / "run.log").write_text("PSNR 12.0 SSIM 0.1\n")
    table = tmp_path / "table.md"

    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert "999" not in row["FID"]
    assert row["FID"] == "66.364208"
    assert row["PSNR"] == "30.250000"


def test_conflicting_structured_sources_fail_closed(tmp_path):
    root = make_gs_run(tmp_path)
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    aggregate["N"] = 25
    write_json(root / "formal_aggregate.json", aggregate)
    validated = json.loads((root / "VALIDATED.json").read_text())
    validated["N"] = 10
    write_json(root / "VALIDATED.json", validated)
    table = tmp_path / "table.md"

    with pytest.raises(ConflictError) as excinfo:
        update_experiment_table(root, table)
    assert "'N'" in str(excinfo.value)
    assert not table.exists()


# ---------------------------------------------------------------------------
# generation-only / attack-only / incomplete
# ---------------------------------------------------------------------------


def test_generation_only_completed_row(tmp_path):
    root = tmp_path / "outputs" / "gs" / "diffusiondb" / "shared_clean_gs_from_tr" / "rk_g"
    root.mkdir(parents=True)
    write_json(
        root / "generation_complete.json",
        {"method": "GS", "dataset": "diffusiondb", "N": 1001,
         "status": "completed_generation", "finished_utc": "2026-07-24T10:00:00Z"},
    )
    table = tmp_path / "table.md"

    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Stage"] == "watermark_generation"
    assert row["Status"] == "completed_generation"
    assert row["Status"] != STATUS_VALIDATED
    assert row["N"] == "1001"
    for column in ("Detector Metric", "Threshold", "FID", "CLIP", "PSNR", "SSIM"):
        assert row[column] == MISSING


def test_attack_only_completed_row(tmp_path):
    root = tmp_path / "outputs" / "tr" / "diffusiondb" / "zero_shift_ablation" / "rk_a"
    root.mkdir(parents=True)
    write_json(
        root / "attack_complete.json",
        {"method": "TR", "dataset": "diffusiondb", "N": 50,
         "status": "completed_attack", "finished_utc": "2026-07-24T12:00:00Z",
         "attack_config_hash": "abcdef0123456789"},
    )
    table = tmp_path / "table.md"

    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Stage"] == "attack_only"
    assert row["Status"] == "completed_attack"
    assert row["Status"] != STATUS_VALIDATED
    assert row["Attack"] == "abcdef012345"
    assert row["Before Detection Rate"] == MISSING


def test_incomplete_formal_result_is_not_marked_validated(tmp_path):
    root = make_tr_run(tmp_path, validated=False)
    table = tmp_path / "table.md"

    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Stage"] == "evaluation"
    assert row["Status"] != STATUS_VALIDATED


def test_validated_json_without_aggregate_fails_closed(tmp_path):
    root = tmp_path / "outputs" / "tr" / "diffusiondb" / "slug" / "rk_v"
    root.mkdir(parents=True)
    write_json(root / "run_config.json", {"method": "TR", "dataset": "diffusiondb"})
    write_json(root / "VALIDATED.json", {"status": "validated_formal_result"})
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="VALIDATED.json present without"):
        update_experiment_table(root, table)


# ---------------------------------------------------------------------------
# Gaussian Shading
# ---------------------------------------------------------------------------


def test_gs_detector_metric_and_direction(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Detector Metric"] == "bit_accuracy"
    assert row["Score Direction"] == "higher_is_watermarked"
    assert row["Before Score"] == "1.0"
    assert row["After Score"] == "0.5015625"


def test_gs_official_threshold_type_is_preserved(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Threshold Type"] == "official_beta_tail_tau_onebit"
    assert row["Threshold"] == "0.6484375"
    assert "1pct" not in row["Threshold Type"]
    assert "empirical" not in row["Threshold Type"]


def test_gs_nominal_fpr_is_not_written_as_empirical_clean_fpr(tmp_path):
    root = make_gs_run(tmp_path, provider_fpr=1e-06)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Nominal FPR"] == "1e-06"
    assert row["Empirical Clean FPR"] == "0.0"
    assert row["Empirical Clean FPR"] != row["Nominal FPR"]


def test_gs_detection_rate_is_not_labelled_tpr_at_1pct_fpr(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Before Detection Rate"] == "1.0"
    assert row["After Detection Rate"] == "0.0"
    text = table.read_text()
    assert "TPR@1%FPR" not in text
    assert "TPR" not in "".join(COLUMNS)


def test_gs_without_clean_cohort_has_no_empirical_clean_fpr(tmp_path):
    root = make_gs_run(tmp_path, clean_stage=False)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Empirical Clean FPR"] == MISSING
    assert row["Nominal FPR"] == "1e-06"


def test_gs_unknown_detector_schema_fails_rather_than_guessing(tmp_path):
    metric = gs_metric_block()
    del metric["official_tau_onebit"]
    root = make_gs_run(tmp_path, metric=metric)
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="unknown GS threshold schema"):
        update_experiment_table(root, table)


def test_gs_unsupported_comparison_operator_fails(tmp_path):
    metric = gs_metric_block(official_threshold_comparison_operator="~=")
    root = make_gs_run(tmp_path, metric=metric)
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="comparison operator"):
        update_experiment_table(root, table)


def test_gs_attack_success_only_from_authoritative_aggregate(tmp_path):
    root = make_gs_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)
    assert read_rows(table)[0]["Attack Success"] == MISSING

    metric = gs_metric_block(attack_success_rate=0.75)
    root2 = make_gs_run(tmp_path, metric=metric, run_key="rk_gs_as",
                        experiment="gs_with_attack_success")
    update_experiment_table(root2, table)
    assert row_by_experiment(table, "gs_with_attack_success")["Attack Success"] == "0.75"


# ---------------------------------------------------------------------------
# Tree-Ring
# ---------------------------------------------------------------------------


def test_tr_empirical_clean_fpr_is_preserved(tmp_path):
    root = make_tr_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Detector Metric"] == "l1_complex"
    assert row["Score Direction"] == "lower_is_watermarked"
    assert row["Threshold Type"] == "empirical_clean_1pct_fpr"
    assert row["Threshold"] == "75.68119049072266"
    assert row["Empirical Clean FPR"] == "0.008991008991008992"


def test_tr_tpr_values_are_normalized_into_generic_detection_rate_columns(tmp_path):
    root = make_tr_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Before Detection Rate"] == "1.0"
    assert row["After Detection Rate"] == "0.1048951048951049"
    assert row["Attack Success"] == "0.8571428571428572"
    assert row["ROC-AUC"] == "0.8413724137999862"


def test_tr_without_clean_calibration_has_no_1pct_threshold_type(tmp_path):
    block = tr_protocol_block()
    for key in ("original_clean_threshold", "original_clean_actual_fpr",
                "original_clean_target_fpr"):
        del block[key]
    root = make_tr_run(tmp_path, block=block)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["Threshold Type"] == MISSING
    assert row["Empirical Clean FPR"] == MISSING


def test_tr_unknown_detector_fields_fail_rather_than_being_guessed(tmp_path):
    block = tr_protocol_block()
    del block["attacked_tpr_at_original_clean_threshold"]
    root = make_tr_run(tmp_path, block=block)
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="unknown TR detector schema"):
        update_experiment_table(root, table)


def test_tr_unknown_detector_metric_name_fails(tmp_path):
    root = make_tr_run(tmp_path)
    aggregate = json.loads((root / "aggregate_results.json").read_text())
    aggregate["detector_protocol"] = "some unnamed detector protocol"
    write_json(root / "aggregate_results.json", aggregate)
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="unknown TR detector metric"):
        update_experiment_table(root, table)


# ---------------------------------------------------------------------------
# additional watermark methods beyond GS and TR
# ---------------------------------------------------------------------------


def make_other_method_run(tmp_path, method="XX", run_key="rk_x"):
    root = tmp_path / "outputs" / "xx" / "diffusiondb" / "other_method_eval" / run_key
    root.mkdir(parents=True)
    write_json(root / "run_config.json", {"method": method, "dataset": "diffusiondb"})
    write_json(
        root / "formal_aggregate.json",
        {"method": method, "dataset": "diffusiondb", "N": 8,
         "detector": {"custom_score_before": 0.9, "custom_score_after": 0.2}},
    )
    return root


def test_unregistered_method_with_detector_payload_fails_closed(tmp_path):
    root = make_other_method_run(tmp_path)
    table = tmp_path / "table.md"

    with pytest.raises(UpdaterError, match="no detector extractor registered"):
        update_experiment_table(root, table)
    assert not table.exists()


def test_registered_method_specific_extractor_fills_generic_columns(tmp_path):
    def extract_xx(sources):
        detector = sources.get("formal_aggregate")["detector"]
        return DetectorMetrics(
            detector_metric="xx_correlation",
            score_direction="higher_is_watermarked",
            threshold_type="xx_fixed_key_threshold",
            threshold=0.5,
            before_score=detector["custom_score_before"],
            after_score=detector["custom_score_after"],
        )

    register_detector_extractor("XX", extract_xx)
    try:
        root = make_other_method_run(tmp_path)
        table = tmp_path / "table.md"
        update_experiment_table(root, table)
    finally:
        import experiments.update_experiment_table as updater

        updater.DETECTOR_EXTRACTORS.pop("XX", None)

    row = read_rows(table)[0]
    assert row["Detector Metric"] == "xx_correlation"
    assert row["Threshold Type"] == "xx_fixed_key_threshold"
    assert row["Before Score"] == "0.9"
    assert row["Empirical Clean FPR"] == MISSING


def test_extractor_returning_foreign_score_direction_fails(tmp_path):
    def bad_extractor(sources):
        return DetectorMetrics(
            detector_metric="xx_correlation", score_direction="bit_accuracy"
        )

    register_detector_extractor("XX", bad_extractor)
    try:
        root = make_other_method_run(tmp_path)
        table = tmp_path / "table.md"
        with pytest.raises(UpdaterError, match="unknown score direction"):
            update_experiment_table(root, table)
    finally:
        import experiments.update_experiment_table as updater

        updater.DETECTOR_EXTRACTORS.pop("XX", None)


def test_extractor_threshold_without_threshold_type_fails(tmp_path):
    def bad_extractor(sources):
        return DetectorMetrics(detector_metric="xx_correlation", threshold=0.5)

    register_detector_extractor("XX", bad_extractor)
    try:
        root = make_other_method_run(tmp_path)
        table = tmp_path / "table.md"
        with pytest.raises(UpdaterError, match="threshold type"):
            update_experiment_table(root, table)
    finally:
        import experiments.update_experiment_table as updater

        updater.DETECTOR_EXTRACTORS.pop("XX", None)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def test_pipe_characters_in_text_fields_are_escaped(tmp_path):
    root = make_gs_run(tmp_path)
    aggregate = json.loads((root / "formal_aggregate.json").read_text())
    aggregate["attack_config"]["variant_name"] = "shift|color"
    write_json(root / "formal_aggregate.json", aggregate)
    table = tmp_path / "table.md"

    update_experiment_table(root, table)

    rows = read_rows(table)
    assert len(rows) == 1
    assert rows[0]["Attack"].startswith("shift\\|color")


def test_rates_are_not_multiplied_by_100(tmp_path):
    root = make_tr_run(tmp_path)
    table = tmp_path / "table.md"
    update_experiment_table(root, table)

    row = read_rows(table)[0]
    assert row["After Detection Rate"] == "0.1048951048951049"
    assert row["Empirical Clean FPR"].startswith("0.0089")


# ---------------------------------------------------------------------------
# canonical per-method/per-dataset table location
# ---------------------------------------------------------------------------


def test_default_table_is_per_method_and_dataset(tmp_path, monkeypatch):
    import experiments.update_experiment_table as updater

    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    root = make_gs_run(tmp_path)
    action, table_path, identity = updater.update_experiment_table(root)

    assert action == "inserted"
    assert table_path == tmp_path / "outputs/gs/diffusiondb/_table/experiment_results.md"
    assert table_path.is_file()
    assert identity.method == "GS" and identity.dataset == "diffusiondb"


def test_default_table_separates_methods_and_datasets(tmp_path, monkeypatch):
    import experiments.update_experiment_table as updater

    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    gs = updater.update_experiment_table(make_gs_run(tmp_path))[1]
    tr = updater.update_experiment_table(make_tr_run(tmp_path))[1]
    assert gs != tr
    assert gs.parent == tmp_path / "outputs/gs/diffusiondb/_table"
    assert tr.parent == tmp_path / "outputs/tr/diffusiondb/_table"
    # one row each, never mixed across methods
    assert len(updater.read_rows(gs)) == 1
    assert len(updater.read_rows(tr)) == 1


def test_table_path_fails_closed_on_bad_components():
    import experiments.update_experiment_table as updater

    for method, dataset in (("GS", "../escape"), ("", "diffusiondb"), ("GS", "")):
        with pytest.raises(updater.UpdaterError, match="invalid"):
            updater.method_dataset_table_path(method, dataset)


def test_two_variants_of_one_dataset_share_one_table(tmp_path, monkeypatch):
    """Both sampling-mode variants upsert into the same summary table."""
    import experiments.update_experiment_table as updater

    monkeypatch.setattr(updater, "REPO_ROOT", tmp_path)
    nearest = make_gs_run(
        tmp_path, experiment="ddim_inverse_ddpm_forward_nearest", run_key="rk_n"
    )
    bilinear = make_gs_run(
        tmp_path, experiment="ddim_inverse_ddpm_forward_bilinear", run_key="rk_b"
    )
    table = updater.update_experiment_table(nearest)[1]
    assert updater.update_experiment_table(bilinear)[1] == table
    rows = updater.read_rows(table)
    assert len(rows) == 2
    assert {row["Experiment"] for row in rows} == {
        "ddim_inverse_ddpm_forward_nearest",
        "ddim_inverse_ddpm_forward_bilinear",
    }
