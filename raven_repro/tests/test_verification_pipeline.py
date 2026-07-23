import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


manifest = load_script("build_verification_manifest")
evaluator = load_script("evaluate_verification")


def semantic_row(run_id, clean, before, attacked):
    return {
        "dataset": "diffusiondb",
        "run_id": str(run_id),
        "method": "TR",
        "model_id": "model",
        "model_revision": "revision",
        "vae_id": "checkpoint-default",
        "vae_scaling_factor": "0.18215",
        "scheduler": "DDIM",
        "inverse_scheduler": "DDIMInverseScheduler",
        "steps": "50",
        "resolution": "512",
        "detector_dtype": "torch.float32",
        "score_direction": "lower raw p-value means watermark; canonical=-log10(p)",
        "provider_parameters": json.dumps({"w_seed": 999999}),
        "legacy_threshold": "0.024",
        "clean_raw_score": str(clean),
        "watermarked_raw_score": str(before),
        "attacked_raw_score": str(attacked),
        "clean_canonical_score": str(-__import__("math").log10(clean)),
        "watermarked_canonical_score": str(-__import__("math").log10(before)),
        "attacked_canonical_score": str(-__import__("math").log10(attacked)),
    }


def test_semantic_report_separates_legacy_and_calibrated_rates():
    rows = [semantic_row(i, 0.5 + i / 1000, 1e-8, 0.01 if i < 50 else 0.9) for i in range(100)]
    report = evaluator.semantic_report("TR", rows, target_fpr=0.01, threshold_override=None)
    assert report["actual_empirical_fpr"] == pytest.approx(0.01)
    assert report["before_tpr"] == 1.0
    assert report["attacked_tpr_at_original_clean_threshold"] == pytest.approx(0.5)
    assert report["legacy_fixed_threshold_attacked_detect_rate"] == pytest.approx(0.5)
    assert report["legacy_actual_clean_fpr"] == 0.0
    assert report["N"] == {"clean": 100, "watermarked": 100, "attacked": 100}


def test_gs_separates_legacy_official_and_clean_calibrated_thresholds():
    rows = [{
        "run_id": "0",
        "clean_raw_score": "0.50",
        "watermarked_raw_score": "1.0",
        "attacked_raw_score": "0.75",
        "legacy_threshold": "0.70703125",
        "gs_official_tau_onebit": "0.65",
        "gs_official_tau_bits": "0.75",
        "gs_secret_index": "0",
        "gs_secret_bundle_sha256": "secret-0",
        "clean_decoded_bits_sha256": "clean-0",
        "watermarked_decoded_bits_sha256": "wm-0",
        "attacked_decoded_bits_sha256": "attack-0",
    }, {
        "run_id": "1",
        "clean_raw_score": "0.55",
        "watermarked_raw_score": "1.0",
        "attacked_raw_score": "0.50",
        "legacy_threshold": "0.70703125",
        "gs_official_tau_onebit": "0.65",
        "gs_official_tau_bits": "0.75",
        "gs_secret_index": "1",
        "gs_secret_bundle_sha256": "secret-1",
        "clean_decoded_bits_sha256": "clean-1",
        "watermarked_decoded_bits_sha256": "wm-1",
        "attacked_decoded_bits_sha256": "attack-1",
    }]
    report, audited = evaluator.gs_report(rows, expected_bits=256, target_fpr=0.01)
    assert report["macro_bit_accuracy_attacked"] == pytest.approx(0.625)
    assert report["legacy_fixed_threshold_rates"]["attacked"] == pytest.approx(0.5)
    assert report["official_onebit_rates"]["attacked"] == pytest.approx(0.5)
    assert "clean_calibrated_threshold_at_target_fpr" in report
    assert report["statistically_valid_for_target_fpr"] is False
    assert "key_hex" not in audited[0]
    assert audited[0]["attacked_decoded_bits_sha256"] == "attack-0"
