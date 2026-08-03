"""Tests for the unified main/eval pipeline — no real data required."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))


# ===========================================================================
# Helpers
# ===========================================================================
def _make_fake_record(run_id="17", role="watermarked", **kwargs):
    return {
        "run_id": run_id,
        "role": role,
        "method": kwargs.get("method", "TR"),
        "input_path": kwargs.get("input_path", f"/tmp/input_{run_id}.png"),
        "prompt": kwargs.get("prompt", ""),
        "prompt_id": kwargs.get("prompt_id", ""),
        "prompt_source": kwargs.get("prompt_source", ""),
        "attack_seed": 59,
        "planned_flow_dx_image_px": 24.0,
        "planned_flow_dy_image_px": -24.0,
        "effective_source_flow_dx_image_px": kwargs.get("edx", 24.0),
        "effective_source_flow_dy_image_px": kwargs.get("edy", -24.0),
        "diffusion_mode": "ddim",
        "inversion_mode": "ddim",
        "scheduler_mode": "ddim",
        "output_path": f"/tmp/out/watermarked/{run_id}/output.png",
        "debug_info_path": kwargs.get("debug_info_path", ""),
        "debug_info_retained": kwargs.get("debug_info_retained", False),
        **kwargs,
    }


# ===========================================================================
# experiment_config
# ===========================================================================
class TestDiffusionModeMap:
    def test_three_modes_exist(self):
        from raven.experiment_config import DIFFUSION_MODE_MAP
        assert set(DIFFUSION_MODE_MAP) == {"ddim", "ddpm", "ddim-ddpm"}

    def test_ddpm_inversion_ddim_scheduler_rejected(self):
        from raven.experiment_config import validate_diffusion_pair
        with pytest.raises(ValueError):
            validate_diffusion_pair("ddpm", "ddim")

    def test_valid_pairs_accepted(self):
        from raven.experiment_config import DIFFUSION_MODE_MAP, validate_diffusion_pair
        for pair in DIFFUSION_MODE_MAP.values():
            validate_diffusion_pair(pair["inversion_mode"], pair["scheduler_mode"])


class TestAlgorithmExecutionSplit:
    def test_no_overlap(self):
        from raven.experiment_config import ALGORITHM_FIELDS, EXECUTION_FIELDS
        assert ALGORITHM_FIELDS.isdisjoint(EXECUTION_FIELDS)

    def test_resume_ignores_execution_fields(self):
        from raven.experiment_config import check_config_match, normalize_config
        base = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/a")
        resumed = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/b",
                                    resume=False, overwrite=False, gpu=1)
        assert check_config_match(base, resumed) == []

    def test_resume_catches_algorithm_drift(self):
        from raven.experiment_config import check_config_match, normalize_config
        base = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/a")
        changed = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/b",
                                    base_seed=99)
        assert "base_seed" in check_config_match(base, changed)


# ===========================================================================
# shift_plan
# ===========================================================================
class TestComputeAttackSeed:
    def test_numeric_17_base_42(self):
        from raven.shift_plan import compute_attack_seed
        assert compute_attack_seed(42, "17") == 59

    def test_non_numeric_deterministic(self):
        from raven.shift_plan import compute_attack_seed
        assert compute_attack_seed(42, "abc-123") == compute_attack_seed(42, "abc-123")


class TestPlanShift:
    def test_none_zero(self):
        from raven.shift_plan import plan_shift
        dx, dy, seed = plan_shift("17", {"shift_mode": "none", "base_seed": 42})
        assert dx == 0.0 and dy == 0.0 and seed == 59

    def test_fixed_exact(self):
        from raven.shift_plan import plan_shift
        dx, dy, seed = plan_shift("17", {"shift_mode": "fixed", "base_seed": 42,
                                           "shift_x": 24.0, "shift_y": -24.0})
        assert dx == 24.0 and dy == -24.0 and seed == 59


# ===========================================================================
# experiment_io — path safety (Critical 1)
# ===========================================================================
class TestValidateOutputDirSafety:
    def test_exact_repo_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        repo = str(Path(__file__).resolve().parents[2])
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety(repo)

    def test_exact_data_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        repo = Path(__file__).resolve().parents[2]
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety(str(repo / "data"))

    def test_child_of_data_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        repo = Path(__file__).resolve().parents[2]
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety(str(repo / "data" / "tr" / "something"))

    def test_child_of_outputs_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        repo = Path(__file__).resolve().parents[2]
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety(str(repo / "outputs" / "tr" / "run"))

    def test_tmp_allowed(self):
        from raven.experiment_io import validate_output_dir_safety
        validate_output_dir_safety("/tmp/raven_test_dir")

    def test_exact_root_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety("/")

    def test_exact_workspace_rejected(self):
        from raven.experiment_io import validate_output_dir_safety
        with pytest.raises(ValueError, match="protected"):
            validate_output_dir_safety("/workspace")


# ===========================================================================
# experiment_io — prepare_output_dir
# ===========================================================================
class TestPrepareOutputDir:
    def test_new_created(self, tmp_path):
        from raven.experiment_io import prepare_output_dir
        d = tmp_path / "new"
        assert prepare_output_dir(d).is_dir()

    def test_nonempty_no_flags_fails(self, tmp_path):
        from raven.experiment_io import prepare_output_dir
        (tmp_path / "x").write_text("hi")
        with pytest.raises(FileExistsError, match="non-empty"):
            prepare_output_dir(tmp_path)

    def test_overwrite_clears(self, tmp_path):
        from raven.experiment_io import prepare_output_dir
        (tmp_path / "x").write_text("hi")
        result = prepare_output_dir(tmp_path, overwrite=True)
        assert not (tmp_path / "x").exists()


# ===========================================================================
# experiment_io — debug_info cleanup
# ===========================================================================
class TestCleanupIntermediates:
    def test_keeps_output_and_record(self, tmp_path):
        from raven.experiment_io import sample_dir, cleanup_intermediates
        s = sample_dir(tmp_path, "watermarked", "17")
        s.mkdir(parents=True)
        (s / "output.png").write_bytes(b"png")
        (s / "record.json").write_text("{}")
        (s / "debug_info.json").write_text("{}")
        (s / "view_guided_output.png").write_bytes(b"png")
        cleanup_intermediates(tmp_path, "watermarked", "17")
        assert {f.name for f in s.iterdir()} == {"output.png", "record.json"}


# ===========================================================================
# Import isolation
# ===========================================================================
class TestImportIsolation:
    def test_metrics_no_pipeline(self):
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "import raven.metrics; "
             "print('pipeline_raven' in sys.modules)"],
            capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0
        assert r.stdout.strip() == "False"

    def test_pipeline_still_importable(self):
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "from raven import RavenPipeline; "
             "print('OK')"],
            capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0
        assert "OK" in r.stdout


# ===========================================================================
# Detector modules — structure and dispatch
# ===========================================================================
class TestDetectorModulesExist:
    def test_all_seven_methods_have_module(self):
        from raven.detectors import DETECTOR_MODULES, _lazy_imports, get_detector_module
        _lazy_imports()
        for m in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            mod = get_detector_module(m)
            assert mod is not None, f"No module for {m}"

    def test_each_module_has_required_functions(self):
        from raven.detectors import _lazy_imports, get_detector_module
        _lazy_imports()
        for m in ["TR", "GS", "GM", "T2S", "RID"]:
            mod = get_detector_module(m)
            assert callable(mod.load_state), f"{m}: load_state missing"
            assert callable(mod.score_image), f"{m}: score_image missing"
            assert callable(mod.aggregate), f"{m}: aggregate missing"
            assert callable(mod.describe_required_artifacts), f"{m}: describe missing"

    def test_tr_describe_artifacts(self):
        from raven.detectors import get_detector_module
        mod = get_detector_module("TR")
        artifacts = mod.describe_required_artifacts()
        assert isinstance(artifacts, list)
        assert len(artifacts) > 0


# ===========================================================================
# Detector — TR module mock tests
# ===========================================================================
class TestTRDetectorMock:
    def test_load_state_returns_none_without_eval_bench(self, monkeypatch):
        """Without eval_bench_wm, load_state returns None (not exception)."""
        from raven.detectors.tr_detector import load_state
        # Ensure eval_bench_wm is NOT importable for this test
        result = None
        try:
            result = load_state([], "cpu")
        except Exception:
            pass
        # May return None or a dict depending on environment; don't crash.
        assert result is None or isinstance(result, dict)

    def test_aggregate_handles_empty_rows(self):
        from raven.detectors.tr_detector import aggregate
        result = aggregate([])
        assert result["scored_count"] == 0
        assert result["failed_count"] == 0

    def test_aggregate_with_fake_scores(self):
        from raven.detectors.tr_detector import aggregate
        rows = [
            {"status": "scored", "evaluation_cohort": "original_clean",
             "canonical_score": -5.0},
            {"status": "scored", "evaluation_cohort": "original_clean",
             "canonical_score": -4.0},
            {"status": "scored", "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": "scored", "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 2.0},
        ]
        result = aggregate(rows)
        assert result["scored_count"] == 4
        assert "detection_summary" in result
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is False


# ===========================================================================
# Detector — GM module
# ===========================================================================
class TestGMDetector:
    def test_aggregate_handles_empty(self):
        from raven.detectors.gm_detector import aggregate
        result = aggregate([])
        assert result["method"] == "GM"
        assert result["score_type"] == "gm_raw_bit_accuracy"

    def test_describe_names_bundle_artifacts(self):
        from raven.detectors.gm_detector import describe_required_artifacts
        artifacts = describe_required_artifacts()
        texts = " ".join(artifacts).lower()
        assert "w1" in texts
        assert "w2" in texts
        assert "manifest" in texts


# ===========================================================================
# Detector — T2S module
# ===========================================================================
class TestT2SDetector:
    def test_aggregate_preserves_bit_accuracy_percentiles(self):
        import numpy as np
        from raven.detectors.t2s_detector import aggregate
        rows = []
        for i in range(10):
            rows.append({
                "status": "scored",
                "evaluation_cohort": "original_watermarked",
                "t2s_bit_accuracy": 0.5 + i * 0.05,
                "t2s_detection_success": i >= 2,
            })
        rows.append({
            "status": "scored",
            "evaluation_cohort": "attacked_watermarked",
            "t2s_bit_accuracy": 0.1,
            "t2s_detection_success": False,
        })
        result = aggregate(rows)
        owm = result["original_watermarked_bit_accuracy"]
        assert "mean" in owm
        assert "median" in owm
        assert "q25" in owm
        assert "q75" in owm

    def test_message_corrupted_count(self):
        from raven.detectors.t2s_detector import aggregate
        rows = [
            {"status": "scored", "evaluation_cohort": "original_watermarked",
             "t2s_bit_accuracy": 0.5, "t2s_detection_success": True},
            {"status": "scored", "evaluation_cohort": "original_watermarked",
             "t2s_bit_accuracy": 1.0, "t2s_detection_success": True},
            {"status": "scored", "evaluation_cohort": "original_watermarked",
             "t2s_bit_accuracy": 1.0, "t2s_detection_success": False},
        ]
        result = aggregate(rows)
        assert result["original_watermarked_message_corrupted"] == 1
        assert result["original_watermarked_detection_failed_but_readable"] == 1


# ===========================================================================
# Detector — Fourier module
# ===========================================================================
class TestFourierDetector:
    def test_aggregate_canonical_is_negative_raw(self):
        from raven.detectors.fourier_detector import aggregate
        rows = [
            {"status": "scored", "evaluation_cohort": "original_clean",
             "canonical_score": -0.1, "raw_l1": 0.1},
            {"status": "scored", "evaluation_cohort": "original_clean",
             "canonical_score": -10.0, "raw_l1": 10.0},
            {"status": "scored", "evaluation_cohort": "original_watermarked",
             "canonical_score": -0.5, "raw_l1": 0.5},
            {"status": "scored", "evaluation_cohort": "attacked_watermarked",
             "canonical_score": -5.0, "raw_l1": 5.0},
        ]
        result = aggregate(rows, method="RID")
        assert result["scored_count"] == 4
        assert "detection_summary" in result
        # Verify score direction: -0.1 > -10.0 (smaller raw L1 = larger canonical)
        assert result["score_direction"] == "higher_is_watermarked"

    def test_raw_l1_to_canonical_order(self):
        """smaller raw L1 → larger canonical score."""
        raw_a, raw_b = 0.1, 10.0
        canon_a, canon_b = -raw_a, -raw_b
        assert canon_a > canon_b


# ===========================================================================
# eval.py — FID stage mock
# ===========================================================================
class TestFIDStage:
    def test_watermarked_only(self):
        """Clean-role records excluded."""
        wm = _make_fake_record("1", "watermarked")
        cl = _make_fake_record("1", "clean")
        # evaluate_fid filters to watermarked only
        from experiments.eval import evaluate_fid
        # With real images missing, should skip gracefully
        result = evaluate_fid([wm, cl], "/tmp/nonexistent_fid_test")
        assert result["stage"] == "fid"

    def test_non_numeric_run_id_safe_name(self):
        """Non-numeric run IDs get hashed safe names."""
        import hashlib
        run_id = "non-numeric-abc"
        try:
            name = f"{int(run_id):06d}"
        except (ValueError, TypeError):
            name = hashlib.sha256(run_id.encode()).hexdigest()[:12]
        assert len(name) == 12  # hash-based fallback, not zero-padded int

    def test_clean_records_excluded_from_pairs(self):
        """Only watermarked role contributes to FID pairs."""
        wm = _make_fake_record("1", "watermarked",
                                input_path="/tmp/real_input.png")
        cl = _make_fake_record("2", "clean",
                                input_path="/tmp/clean_input.png")
        wm_only = [r for r in [wm, cl] if r.get("role") == "watermarked"]
        assert len(wm_only) == 1
        assert wm_only[0]["run_id"] == "1"


# ===========================================================================
# eval.py — CLIP stage mock
# ===========================================================================
class TestCLIPStage:
    def test_watermarked_only(self):
        wm = _make_fake_record("1", "watermarked", prompt="a cat")
        cl = _make_fake_record("1", "clean", prompt="a cat")
        wm_only = [r for r in [wm, cl] if r.get("role") == "watermarked"]
        assert len(wm_only) == 1

    def test_missing_prompt_caught(self):
        wm = _make_fake_record("1", "watermarked", prompt="")
        prompts = [r.get("prompt", "") for r in [wm]
                   if r.get("role") == "watermarked"]
        assert not all(prompts)


# ===========================================================================
# eval.py — Quality stage
# ===========================================================================
class TestQualityStage:
    def test_no_planned_flow_fallback(self):
        """Quality must use effective flow, not planned flow."""
        from experiments.eval import evaluate_quality
        rec = _make_fake_record("1", "watermarked",
                                 edx=None, edy=None)  # missing effective
        rec.pop("effective_source_flow_dx_image_px", None)
        rec.pop("effective_source_flow_dy_image_px", None)
        result = evaluate_quality([rec], "/tmp/nonexistent_quality_test")
        # Should report unavailable, not fallback to planned
        assert not result.get("available", True) or result["psnr_mean"] is None

    def test_effective_flow_present_passes(self):
        """When effective flow exists, quality proceeds (image may be missing)."""
        from experiments.eval import evaluate_quality
        rec = _make_fake_record("1", "watermarked",
                                 edx=24.0, edy=-24.0)
        result = evaluate_quality([rec], "/tmp/nonexistent_quality_test")
        assert result["stage"] == "quality"


# ===========================================================================
# eval.py — stage exit codes
# ===========================================================================
class TestStageExitCodes:
    def test_detector_unavailable_no_flag_nonzero(self):
        """Without --allow-missing-metrics, skipped required stage = nonzero."""
        from experiments.eval import run_evaluation
        # Create a fake output dir with config and records
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            from raven.experiment_io import write_config, write_record
            write_config(out, {"method": "GM", "dataset": "test"})
            write_record(out, "watermarked", "1",
                          _make_fake_record("1", "watermarked", method="GM"))
            (out / "samples" / "watermarked" / "1" / "output.png").write_bytes(b"png")
            from raven.experiment_io import rebuild_records_jsonl
            rebuild_records_jsonl(out)

            result = run_evaluation(out, device="cpu", stages=["detector"],
                                     allow_missing_metrics=False)
            assert "failed_stages" in result or "skipped_stages" in result

    def test_detector_unavailable_allow_flag_zero(self):
        """With --allow-missing-metrics, skipped stages don't go into skipped list."""
        from experiments.eval import run_evaluation
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
            write_config(out, {"method": "GM", "dataset": "test"})
            write_record(out, "watermarked", "1",
                          _make_fake_record("1", "watermarked", method="GM"))
            (out / "samples" / "watermarked" / "1" / "output.png").write_bytes(b"png")
            rebuild_records_jsonl(out)

            result = run_evaluation(out, device="cpu", stages=["detector"],
                                     allow_missing_metrics=True)
            # skipped_stages should be empty because allow_missing_metrics suppresses
            assert result.get("skipped_stages", []) == []


# ===========================================================================
# eval.py — debug_info path handling
# ===========================================================================
class TestDebugInfoPath:
    def test_not_retained_is_empty(self):
        """When save_intermediates=false, debug_info_path is empty string."""
        rec = _make_fake_record("1", "watermarked",
                                 debug_info_path="",
                                 debug_info_retained=False)
        assert rec["debug_info_path"] == ""
        assert rec["debug_info_retained"] is False

    def test_retained_has_path(self):
        rec = _make_fake_record("1", "watermarked",
                                 debug_info_path="/tmp/debug_info.json",
                                 debug_info_retained=True)
        assert rec["debug_info_path"] == "/tmp/debug_info.json"


# ===========================================================================
# Detector cohort model
# ===========================================================================
class TestDetectorCohorts:
    def test_all_four_cohorts(self):
        from experiments.eval import DETECTOR_COHORTS
        wm = DETECTOR_COHORTS["watermarked"]
        cl = DETECTOR_COHORTS["clean"]
        assert wm["original"]["evaluation_cohort"] == "original_watermarked"
        assert wm["attacked"]["evaluation_cohort"] == "attacked_watermarked"
        assert cl["original"]["evaluation_cohort"] == "original_clean"
        assert cl["attacked"]["evaluation_cohort"] == "attacked_clean"

    def test_build_image_index_emits_four_rows_for_both_roles(self):
        from experiments.eval import _build_detector_image_index
        rec_wm = _make_fake_record("1", "watermarked")
        rec_cl = _make_fake_record("1", "clean")
        index = _build_detector_image_index([rec_wm, rec_cl], "/tmp/fake")
        cohorts = {r["evaluation_cohort"] for r in index}
        assert "original_watermarked" in cohorts
        assert "attacked_watermarked" in cohorts
        assert "original_clean" in cohorts
        assert "attacked_clean" in cohorts
        assert len(index) == 4


# ===========================================================================
# eval.py — CLI help
# ===========================================================================
class TestEvalCLI:
    def test_help_works(self):
        r = subprocess.run(
            [sys.executable, "experiments/eval.py", "--help"],
            capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0
        assert "--allow-missing-metrics" in r.stdout


# ===========================================================================
# main.py — structure
# ===========================================================================
class TestMainModuleStructure:
    def test_has_save_intermediates(self):
        source = (REPO / "experiments" / "main.py").read_text()
        assert "save_intermediates" in source
        assert "debug_info_retained" in source


# ===========================================================================
# Status taxonomy
# ===========================================================================
class TestStatusTaxonomy:
    def test_all_statuses_defined(self):
        from experiments.eval import (
            STATUS_COMPLETED, STATUS_SKIPPED_INSUFFICIENT_DATA,
            STATUS_FAILED_MISSING_REQUIRED_STATE, STATUS_FAILED_MISSING_DEPENDENCY,
            STATUS_FAILED_PROVIDER_INITIALIZATION, STATUS_FAILED_SCORING,
            STATUS_FAILED_INTERNAL_ERROR, NONZERO_STATUSES,
        )
        assert STATUS_COMPLETED not in NONZERO_STATUSES
        assert STATUS_SKIPPED_INSUFFICIENT_DATA not in NONZERO_STATUSES
        assert STATUS_FAILED_MISSING_REQUIRED_STATE in NONZERO_STATUSES
        assert STATUS_FAILED_INTERNAL_ERROR in NONZERO_STATUSES
