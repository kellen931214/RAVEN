import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.build_raven_formal_eval_table import (
    FIELDS,
    table_row,
    validate_color_transfer_comparison,
)


def test_formal_tr_table_reports_both_actual_fprs_and_threshold_bases(tmp_path):
    root = tmp_path / "formal"
    verification = root / "verification"
    quality = root / "metrics" / "quality.jsonl"
    verification.mkdir(parents=True)
    quality.parent.mkdir(parents=True)
    (root / "VALIDATED.json").write_text(
        json.dumps({"status": "validated_formal_result"})
    )
    metric = {
        "target_fpr": 0.01,
        "original_clean_actual_fpr": 0.009,
        "attacked_clean_actual_fpr": 0.008,
        "before_tpr": 0.9,
        "attacked_tpr_at_original_clean_threshold": 0.2,
        "attacked_tpr_at_attacked_clean_recalibrated_threshold": 0.3,
        "attack_success_rate_at_recalibrated_threshold": 0.7,
        "attacked_roc_auc": 0.6,
    }
    detector = verification / "detector.json"
    detector.write_text(json.dumps({"nfpa_rounded2_protocol": metric}))
    (verification / "manifest.csv").write_text("run_id\n0\n")
    quality.write_text(json.dumps({
        "post_color_vs_watermarked_overlap_psnr": 20.0,
        "post_color_vs_watermarked_overlap_ssim": 0.8,
    }) + "\n")
    aggregate = {
        "dataset": "diffusiondb",
        "method": "TR",
        "attack_config": {"variant_name": "unit"},
        "N": 1,
        "metric_protocol_version": "unit",
        "detector_result": str(detector),
        "quality_records": str(quality),
        "clip_result": {
            "clip_model_name": "ViT-bigG-14",
            "clip_pretrained": "laion2b_s39b_b160k",
            "mean": 0.5,
        },
        "fid_result": {"reference_definition": "watermarked", "value": 1.0},
        "attack_config_hash": "attack",
        "detector_config_hash": "detector",
        "git_head": "head",
    }
    (root / "formal_aggregate.json").write_text(json.dumps(aggregate))
    row = table_row(root)
    assert "Actual FPR" not in FIELDS
    assert row["Original-clean actual FPR"] == 0.009
    assert row["Attacked-clean recalibrated actual FPR"] == 0.008
    assert row["Attack success at original-clean threshold"] == 0.8
    assert row["Attack success at attacked-clean recalibrated threshold"] == 0.7



def _comparison_record(run_id="0", *, pre_color_sha="pre", source_hash="source"):
    return {
        "run_id": run_id,
        "pairing_sha256": "pair",
        "attack_seed": 42,
        "planned_flow_dx_image_px": 27.0,
        "planned_flow_dy_image_px": -29.0,
        "pre_color_attacked_sha256": pre_color_sha,
        "source_attack_config_hash": source_hash,
    }


def test_color_transfer_comparison_requires_identical_pre_color_attack(tmp_path):
    aligned = tmp_path / "aligned"
    paper = tmp_path / "paper"
    aligned.mkdir()
    paper.mkdir()
    (aligned / "attack_records_color_watermarked.jsonl").write_text(
        json.dumps(_comparison_record()) + "\n"
    )
    (paper / "attack_records_color_watermarked.jsonl").write_text(
        json.dumps(_comparison_record()) + "\n"
    )
    validate_color_transfer_comparison([aligned, paper])

    drifted = _comparison_record(pre_color_sha="replaced")
    (paper / "attack_records_color_watermarked.jsonl").write_text(
        json.dumps(drifted) + "\n"
    )
    import pytest
    with pytest.raises(RuntimeError, match="pre_color_attacked_sha256 mismatch"):
        validate_color_transfer_comparison([aligned, paper])


def test_paper_exact_color_table_status_and_variant(tmp_path):
    root = tmp_path / "paper"
    verification = root / "verification"
    verification.mkdir(parents=True)
    (verification / "manifest.csv").write_text("run_id\n0\n")
    (root / "VALIDATED.json").write_text(json.dumps({
        "status": "validated_paper_exact_color_evaluation",
    }))
    protocol = {
        "target_fpr": 0.01,
        "original_clean_actual_fpr": 0.0,
        "attacked_clean_actual_fpr": 0.0,
        "before_tpr": 1.0,
        "attacked_tpr_at_original_clean_threshold": 0.2,
        "attacked_tpr_at_attacked_clean_recalibrated_threshold": 0.3,
        "attack_success_rate_at_recalibrated_threshold": 0.7,
        "attacked_roc_auc": 0.6,
    }
    aggregate = {
        "sample_count": 1,
        "detector": {"protocol": "tr", "nfpa_rounded2_protocol": protocol},
        "fid": {"reference_definition": "watermarked", "value": 1.0},
        "clip": {
            "clip_model_name": "ViT-bigG-14",
            "clip_pretrained": "laion2b_s39b_b160k",
            "mean": 0.5,
        },
        "quality_reference": "watermarked input",
        "quality_overlap": "effective source flow inverse warp",
        "quality_psnr_mean": 20.0,
        "quality_ssim_mean": 0.8,
        "formal_attack_config_hash": "paper-hash",
        "git_head": "current-head",
        "result_table": {"Variant": "shift + paper_exact_two_stage"},
    }
    (root / "aggregate_results.json").write_text(json.dumps(aggregate))
    row = table_row(root)
    assert row["Variant"] == "shift + paper_exact_two_stage"
    assert row["Git SHA"] == "current-head"
    assert row["Attack config hash"] == "paper-hash"
