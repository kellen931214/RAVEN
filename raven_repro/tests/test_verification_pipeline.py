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


def test_provider_defaults_keep_global_sample_mapping():
    assert manifest.provider_defaults("GS")["offset"] == 0
    assert manifest.provider_defaults("HSTR")["fix_gt"] == 1
    assert manifest.provider_defaults("HSQR")["fix_gt"] == 1
    assert manifest.provider_defaults("TR")["w_seed"] == 999999


def test_resolve_recorded_path_repairs_old_workspace_prefix(tmp_path):
    target = tmp_path / "data" / "image.png"
    target.parent.mkdir()
    target.write_bytes(b"image")
    resolved = manifest.resolve_recorded_path("/workspace/data/image.png", tmp_path)
    assert resolved == target.resolve()


def test_semantic_report_separates_legacy_and_calibrated_rates():
    rows = [semantic_row(i, 0.5 + i / 1000, 1e-8, 0.01 if i < 50 else 0.9) for i in range(100)]
    report = evaluator.semantic_report("TR", rows, target_fpr=0.01, threshold_override=None)
    assert report["actual_FPR"] == pytest.approx(0.01)
    assert report["calibrated_before_TPR"] == 1.0
    assert report["calibrated_TPR_at_1pct_FPR"] == pytest.approx(0.5)
    assert report["legacy_fixed_threshold_detect_rate"] == pytest.approx(0.5)
    assert report["legacy_actual_clean_FPR"] == 0.0
    assert report["N"] == {"clean": 100, "watermarked": 100, "attacked": 100}


def test_gs_reports_macro_micro_and_per_sample_errors():
    rows = [{
        "run_id": "0", "ground_truth_bits": "0011", "clean_predicted_bits": "0011",
        "watermarked_predicted_bits": "0011", "attacked_predicted_bits": "0111",
        "key_hex": "aa", "nonce_hex": "bb", "offset": "0",
    }, {
        "run_id": "1", "ground_truth_bits": "1111", "clean_predicted_bits": "1111",
        "watermarked_predicted_bits": "1111", "attacked_predicted_bits": "1100",
        "key_hex": "aa", "nonce_hex": "bb", "offset": "0",
    }]
    report, audited = evaluator.gs_report(rows, expected_bits=4)
    assert report["macro_bit_accuracy_attacked"] == pytest.approx(0.625)
    assert report["micro_bit_accuracy_attacked"] == pytest.approx(0.625)
    assert audited[0]["attacked_bit_errors"] == [1]
    assert audited[1]["attacked_bit_errors"] == [2, 3]
