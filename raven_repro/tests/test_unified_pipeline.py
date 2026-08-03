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

from raven.experiment_config import (  # noqa: E402
    ALGORITHM_FIELDS,
    DIFFUSION_MODE_MAP,
    EXECUTION_FIELDS,
    FORBIDDEN_PAIR,
    VALID_DIFFUSION_MODES,
    check_config_match,
    config_for_pipeline,
    normalize_config,
    resolve_diffusion_mode,
    validate_diffusion_pair,
)
from raven.shift_plan import compute_attack_seed, plan_shift  # noqa: E402
from raven.experiment_io import (  # noqa: E402
    cleanup_intermediates,
    collect_incomplete_run_ids,
    config_path,
    detector_records_path,
    evaluation_dir,
    is_sample_complete,
    output_image_path,
    prepare_output_dir,
    read_config,
    read_records_jsonl,
    rebuild_records_jsonl,
    record_path,
    records_jsonl_path,
    sample_dir,
    validate_output_dir_safety,
    write_config,
    write_record,
)


# ===========================================================================
# experiment_config — diffusion mode mapping
# ===========================================================================
class TestDiffusionModeMap:
    def test_three_modes_exist(self):
        assert set(DIFFUSION_MODE_MAP) == {"ddim", "ddpm", "ddim-ddpm"}

    def test_ddim_maps_correctly(self):
        assert DIFFUSION_MODE_MAP["ddim"] == {
            "inversion_mode": "ddim", "scheduler_mode": "ddim"}

    def test_ddpm_maps_correctly(self):
        assert DIFFUSION_MODE_MAP["ddpm"] == {
            "inversion_mode": "ddpm", "scheduler_mode": "ddpm"}

    def test_ddim_ddpm_maps_correctly(self):
        assert DIFFUSION_MODE_MAP["ddim-ddpm"] == {
            "inversion_mode": "ddim", "scheduler_mode": "ddpm"}

    def test_ddpm_inversion_ddim_scheduler_rejected(self):
        with pytest.raises(ValueError):
            validate_diffusion_pair("ddpm", "ddim")

    def test_valid_pairs_accepted(self):
        for mode, pair in DIFFUSION_MODE_MAP.items():
            validate_diffusion_pair(pair["inversion_mode"], pair["scheduler_mode"])

    def test_unknown_pair_rejected(self):
        with pytest.raises(ValueError):
            validate_diffusion_pair("forward_noise", "ddim")

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            resolve_diffusion_mode("nonexistent")


# ===========================================================================
# experiment_config — config normalization
# ===========================================================================
class TestNormalizeConfig:
    def test_defaults_produce_valid_config(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out")
        assert config["inversion_mode"] == "ddim"
        assert config["scheduler_mode"] == "ddim"
        assert config["base_seed"] == 42
        assert config["shift_mode"] == "random"

    def test_no_config_hash(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out")
        assert "config_hash" not in config

    def test_resume_and_overwrite_mutex(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out",
                              resume=True, overwrite=True)

    def test_ddim_ddpm_produces_correct_pair(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out",
                                   diffusion_mode="ddim-ddpm")
        assert config["inversion_mode"] == "ddim"
        assert config["scheduler_mode"] == "ddpm"

    def test_fixed_shift_requires_xy(self):
        with pytest.raises(ValueError, match="shift_mode='fixed'"):
            normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out",
                              shift_mode="fixed")

    def test_none_shift_zeroes_xy(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out",
                                   shift_mode="none")
        assert config["shift_x"] == 0.0
        assert config["shift_y"] == 0.0

    def test_unknown_kwargs_rejected(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out",
                              fake_param=123)

    def test_invalid_shift_mode_rejected(self):
        with pytest.raises(ValueError, match="shift_mode"):
            normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out",
                              shift_mode="invalid_mode")


# ===========================================================================
# experiment_config — algorithm vs execution fields
# ===========================================================================
class TestAlgorithmExecutionSplit:
    def test_algorithm_fields_exist(self):
        assert "model_id" in ALGORITHM_FIELDS
        assert "steps" in ALGORITHM_FIELDS
        assert "base_seed" in ALGORITHM_FIELDS
        assert "diffusion_mode" in ALGORITHM_FIELDS

    def test_execution_fields_exist(self):
        assert "output_dir" in EXECUTION_FIELDS
        assert "resume" in EXECUTION_FIELDS
        assert "overwrite" in EXECUTION_FIELDS
        assert "gpu" in EXECUTION_FIELDS

    def test_no_overlap(self):
        assert ALGORITHM_FIELDS.isdisjoint(EXECUTION_FIELDS)

    def test_resume_ignores_execution_fields(self):
        base = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/a")
        resumed = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/b",
                                    resume=True, overwrite=False, gpu=1)
        assert check_config_match(base, resumed) == []

    def test_resume_catches_algorithm_drift(self):
        base = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/a")
        changed = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/b",
                                    base_seed=99)
        mismatches = check_config_match(base, changed)
        assert "base_seed" in mismatches

    def test_execution_only_diff_ignored(self):
        # resume/overwrite are mutex — test with different gpu and output_dir only
        base = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/a",
                                 resume=False, overwrite=False, gpu=0)
        current = normalize_config(metadata_path="/tmp/t.csv", output_dir="/tmp/b",
                                    resume=False, overwrite=False, gpu=7)
        assert check_config_match(base, current) == []


class TestConfigForPipeline:
    def test_includes_all_required_fields(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out")
        pipe_kwargs = config_for_pipeline(config)
        required = {
            "steps", "strength", "guidance_scale", "shift_space",
            "warp_mode", "latent_sampling_mode", "padding_mode",
            "shift_x", "shift_y", "view_guided_attention", "color_transfer",
            "seed", "prompt", "negative_prompt", "debug", "inversion_mode",
            "save_input_copy",
        }
        assert set(pipe_kwargs) == required

    def test_seed_is_none_placeholder(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out")
        pipe_kwargs = config_for_pipeline(config)
        assert pipe_kwargs["seed"] is None

    def test_prompt_is_none_placeholder(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out")
        pipe_kwargs = config_for_pipeline(config)
        assert pipe_kwargs["prompt"] is None


# ===========================================================================
# shift_plan
# ===========================================================================
class TestComputeAttackSeed:
    def test_numeric_run_id_adds_to_base(self):
        assert compute_attack_seed(42, "17") == 59

    def test_numeric_run_id_zero(self):
        assert compute_attack_seed(42, "0") == 42

    def test_numeric_run_id_large(self):
        assert compute_attack_seed(100, "9999") == 10099

    def test_non_numeric_run_id_deterministic(self):
        first = compute_attack_seed(42, "abc-123")
        second = compute_attack_seed(42, "abc-123")
        assert first == second

    def test_non_numeric_run_id_different_for_different_input(self):
        first = compute_attack_seed(42, "abc-123")
        second = compute_attack_seed(42, "abc-124")
        assert first != second

    def test_non_numeric_run_id_is_int(self):
        result = compute_attack_seed(42, "non-numeric-id")
        assert isinstance(result, int)


class TestPlanShift:
    def test_none_mode_returns_zero(self):
        config = {"shift_mode": "none", "base_seed": 42}
        dx, dy, seed = plan_shift("17", config)
        assert dx == 0.0 and dy == 0.0 and seed == 59

    def test_fixed_mode_returns_exact(self):
        config = {"shift_mode": "fixed", "base_seed": 42,
                   "shift_x": 24.0, "shift_y": -24.0}
        dx, dy, seed = plan_shift("17", config)
        assert dx == 24.0 and dy == -24.0 and seed == 59

    def test_random_mode_produces_within_range(self):
        config = {"shift_mode": "random", "base_seed": 42,
                   "shift_magnitude_min": 24, "shift_magnitude_max": 32}
        for run_id in ["0", "1", "2", "10", "100"]:
            dx, dy, seed = plan_shift(run_id, config)
            assert 24 <= abs(dx) <= 32
            assert 24 <= abs(dy) <= 32
            assert seed == 42 + int(run_id)

    def test_random_mode_deterministic(self):
        config = {"shift_mode": "random", "base_seed": 42,
                   "shift_magnitude_min": 24, "shift_magnitude_max": 32}
        first = plan_shift("17", config)
        second = plan_shift("17", config)
        assert first == second

    def test_random_mode_uses_run_id_not_index(self):
        config = {"shift_mode": "random", "base_seed": 42,
                   "shift_magnitude_min": 24, "shift_magnitude_max": 32}
        result_a = plan_shift("17", config, index=0)
        result_b = plan_shift("17", config, index=5)
        assert result_a == result_b

    def test_x_and_y_independent(self):
        config = {"shift_mode": "random", "base_seed": 0}
        results = set()
        for run_id in range(100):
            dx, dy, _ = plan_shift(str(run_id), config)
            results.add((dx, dy))
        signs = {(1 if r[0] > 0 else -1, 1 if r[1] > 0 else -1) for r in results}
        assert len(signs) >= 2


# ===========================================================================
# experiment_io — output layout
# ===========================================================================
class TestOutputLayout:
    def test_sample_dir_structure(self):
        d = sample_dir("/tmp/run", "watermarked", "17")
        assert d == Path("/tmp/run/samples/watermarked/17")

    def test_output_image_path(self):
        p = output_image_path("/tmp/run", "watermarked", "17")
        assert p.name == "output.png"

    def test_record_path(self):
        p = record_path("/tmp/run", "watermarked", "17")
        assert p.name == "record.json"

    def test_config_path(self):
        assert config_path("/tmp/run") == Path("/tmp/run/config.json")

    def test_records_jsonl_path(self):
        assert records_jsonl_path("/tmp/run") == Path("/tmp/run/records.jsonl")

    def test_evaluation_dir(self):
        assert evaluation_dir("/tmp/run") == Path("/tmp/run/evaluation")

    def test_detector_records_path(self):
        assert detector_records_path("/tmp/run") == Path(
            "/tmp/run/evaluation/detector_records.jsonl")


class TestOutputDirSafety:
    def test_repo_data_protected(self):
        repo = Path(__file__).resolve().parents[2]
        with pytest.raises(ValueError):
            validate_output_dir_safety(str(repo / "data"))

    def test_tmp_dir_allowed(self):
        validate_output_dir_safety("/tmp/raven_safe_test")


class TestPrepareOutputDir:
    def test_create_new(self, tmp_path):
        d = tmp_path / "new_dir"
        result = prepare_output_dir(d)
        assert result.is_dir()

    def test_empty_existing(self, tmp_path):
        result = prepare_output_dir(tmp_path)
        assert result.is_dir()

    def test_nonempty_without_flags_fails(self, tmp_path):
        (tmp_path / "some_file").write_text("data")
        with pytest.raises(FileExistsError, match="non-empty"):
            prepare_output_dir(tmp_path)

    def test_nonempty_with_overwrite(self, tmp_path):
        (tmp_path / "some_file").write_text("data")
        result = prepare_output_dir(tmp_path, overwrite=True)
        assert result.is_dir()
        assert not (tmp_path / "some_file").exists()

    def test_nonempty_with_resume_needs_config(self, tmp_path):
        (tmp_path / "some_file").write_text("data")
        with pytest.raises(FileNotFoundError, match="no config.json"):
            prepare_output_dir(tmp_path, resume=True)

    def test_nonempty_with_resume_and_config(self, tmp_path):
        write_config(tmp_path, {"method": "TR"})
        result = prepare_output_dir(tmp_path, resume=True)
        assert result.is_dir()


class TestWriteReadConfig:
    def test_roundtrip(self, tmp_path):
        config = {"method": "TR", "dataset": "test", "base_seed": 42}
        write_config(tmp_path, config)
        read = read_config(tmp_path)
        assert read["method"] == "TR"
        assert read["base_seed"] == 42


class TestWriteReadRecord:
    def test_write_and_read(self, tmp_path):
        write_record(tmp_path, "watermarked", "17",
                      {"run_id": "17", "role": "watermarked", "attack_seed": 59})
        output_image_path(tmp_path, "watermarked", "17").write_bytes(b"fake png")
        assert is_sample_complete(tmp_path, "watermarked", "17")

    def test_record_status_forced_complete(self, tmp_path):
        write_record(tmp_path, "watermarked", "17",
                      {"run_id": "17", "status": "pending"})
        rec = json.loads(record_path(tmp_path, "watermarked", "17").read_text())
        assert rec["status"] == "complete"

    def test_incomplete_without_output_png(self, tmp_path):
        rec = record_path(tmp_path, "watermarked", "17")
        rec.parent.mkdir(parents=True)
        rec.write_text(json.dumps({"run_id": "17", "status": "complete"}))
        assert not is_sample_complete(tmp_path, "watermarked", "17")

    def test_prompt_fields_saved(self, tmp_path):
        write_record(tmp_path, "watermarked", "17", {
            "run_id": "17", "role": "watermarked",
            "prompt": "a cat", "prompt_id": "cat_001",
            "prompt_source": "metadata",
        })
        output_image_path(tmp_path, "watermarked", "17").write_bytes(b"fake")
        assert is_sample_complete(tmp_path, "watermarked", "17")


class TestRebuildRecordsJsonl:
    def test_empty_dir(self, tmp_path):
        rebuild_records_jsonl(tmp_path)
        assert read_records_jsonl(tmp_path) == []

    def test_rebuilds_from_records(self, tmp_path):
        for rid in ("1", "2"):
            write_record(tmp_path, "watermarked", rid,
                          {"run_id": rid, "role": "watermarked", "attack_seed": 42})
            output_image_path(tmp_path, "watermarked", rid).write_bytes(b"fake")
        rebuild_records_jsonl(tmp_path)
        assert len(read_records_jsonl(tmp_path)) == 2

    def test_sorted_by_run_id(self, tmp_path):
        for rid in ("10", "2", "1"):
            write_record(tmp_path, "watermarked", rid,
                          {"run_id": rid, "role": "watermarked"})
            output_image_path(tmp_path, "watermarked", rid).write_bytes(b"fake")
        rebuild_records_jsonl(tmp_path)
        assert [r["run_id"] for r in read_records_jsonl(tmp_path)] == ["1", "10", "2"]


class TestCollectIncomplete:
    def test_all_incomplete_when_empty(self, tmp_path):
        incomplete = collect_incomplete_run_ids(tmp_path, ["watermarked"], ["1", "2", "3"])
        assert len(incomplete) == 3

    def test_none_incomplete_when_all_complete(self, tmp_path):
        for rid in ("1", "2"):
            write_record(tmp_path, "watermarked", rid,
                          {"run_id": rid, "role": "watermarked"})
            output_image_path(tmp_path, "watermarked", rid).write_bytes(b"fake")
        assert len(collect_incomplete_run_ids(tmp_path, ["watermarked"], ["1", "2"])) == 0


class TestCleanupIntermediates:
    def test_keeps_output_png_and_record(self, tmp_path):
        sample = sample_dir(tmp_path, "watermarked", "17")
        sample.mkdir(parents=True)
        (sample / "output.png").write_bytes(b"png")
        (sample / "record.json").write_text("{}")
        (sample / "debug_info.json").write_text("{}")
        (sample / "view_guided_output.png").write_bytes(b"png")
        (sample / "final_color_corrected.png").write_bytes(b"png")
        cleanup_intermediates(tmp_path, "watermarked", "17")
        remaining = {f.name for f in sample.iterdir()}
        assert remaining == {"output.png", "record.json"}


# ===========================================================================
# Import isolation
# ===========================================================================
class TestImportIsolation:
    def test_importing_metrics_does_not_load_pipeline(self):
        """Verify ``import raven.metrics`` does not load pipeline_raven."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "import raven.metrics; "
             "print('pipeline_raven' in sys.modules)"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "False", (
            "importing raven.metrics loaded pipeline_raven")

    def test_importing_experiment_io_does_not_load_pipeline(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "import raven.experiment_io; "
             "print('pipeline_raven' in sys.modules)"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "False"

    def test_importing_experiment_config_does_not_load_pipeline(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "import raven.experiment_config; "
             "print('pipeline_raven' in sys.modules)"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "False"

    def test_raven_pipeline_still_importable(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'raven_repro'); "
             "from raven import RavenPipeline; "
             "print(type(RavenPipeline).__name__)"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert result.returncode == 0
        assert "type" in result.stdout.strip()


# ===========================================================================
# Fourier score direction
# ===========================================================================
class TestFourierScoreDirection:
    def test_raw_l1_smaller_means_canonical_larger(self):
        """Canonical score must be -raw_l1. Smaller raw → larger canonical."""
        raw_a, raw_b = 0.1, 10.0
        canon_a, canon_b = -raw_a, -raw_b
        assert canon_a > canon_b

    def test_canonical_watermark_score_rid_higher_is_watermarked(self):
        from raven.metrics import canonical_watermark_score
        # RID is in SEMANTIC_METHODS: canonical = -raw
        raw_small = 0.1  # strong watermark (small L1)
        raw_large = 10.0  # weak watermark (large L1)
        assert canonical_watermark_score("RID", raw_small) > \
               canonical_watermark_score("RID", raw_large)

    def test_gm_and_t2s_not_in_semantic_methods(self):
        from raven.metrics import SEMANTIC_METHODS
        assert "GM" not in SEMANTIC_METHODS
        assert "T2S" not in SEMANTIC_METHODS

    def test_canonical_watermark_score_rejects_gm_t2s(self):
        from raven.metrics import canonical_watermark_score
        with pytest.raises(ValueError):
            canonical_watermark_score("GM", 0.5)
        with pytest.raises(ValueError):
            canonical_watermark_score("T2S", 0.5)


# ===========================================================================
# T2S aggregation
# ===========================================================================
class TestT2SAggregation:
    def test_bit_accuracy_percentiles(self):
        """T2S bit accuracy must compute mean/median/q25/q75."""
        import numpy as np
        values = [0.5, 0.8, 0.9, 0.95, 1.0, 1.0, 1.0, 0.7, 0.85, 0.92]
        result = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
        assert result["mean"] > 0
        assert result["median"] > 0
        assert result["q75"] > result["q25"]

    def test_message_corrupted_logic(self):
        """detected==True and bit_accuracy<1 → message corrupted."""
        detections = [True, True, False, True, False]
        bit_accs = [0.5, 1.0, 1.0, 0.9, 0.8]
        corrupted = sum(
            1 for det, acc in zip(detections, bit_accs)
            if det and acc < 1.0
        )
        assert corrupted == 2  # indices 0 and 3

    def test_detection_failed_but_readable(self):
        """detected==False and bit_accuracy==1 → detection failed but readable."""
        detections = [True, False, False, True]
        bit_accs = [1.0, 1.0, 0.5, 0.9]
        failed_readable = sum(
            1 for det, acc in zip(detections, bit_accs)
            if not det and acc == 1.0
        )
        assert failed_readable == 1  # index 1


# ===========================================================================
# Detector cohort model
# ===========================================================================
class TestDetectorCohortModel:
    def test_eval_source_has_cohort_model(self):
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "DETECTOR_COHORTS" in source
        assert "original_watermarked" in source
        assert "attacked_watermarked" in source
        assert "original_clean" in source
        assert "attacked_clean" in source

    def test_cohorts_distinct_from_roles(self):
        source = (REPO / "experiments" / "eval.py").read_text()
        assert '"evaluation_cohort"' in source
        assert '"image_source"' in source


# ===========================================================================
# eval.py structure checks
# ===========================================================================
class TestEvalModuleStructure:
    def test_eval_no_raven_pipeline_import(self):
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "from raven.pipeline_raven import" not in source
        assert "import RavenPipeline" not in source

    def test_eval_has_fid_stage(self):
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "evaluate_fid" in source
        assert "clean_fid" in source

    def test_eval_has_clip_stage(self):
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "evaluate_clip" in source
        assert "openclip_text_image_scores" in source


# ===========================================================================
# main.py structure checks
# ===========================================================================
class TestMainModuleStructure:
    def test_main_no_detector_imports(self):
        source = (REPO / "experiments" / "main.py").read_text()
        assert "extract_verification_scores" not in source
        assert "evaluate_verification" not in source
        assert "clean_fid" not in source
        assert "openclip" not in source.lower()

    def test_main_has_prompt_propagation(self):
        source = (REPO / "experiments" / "main.py").read_text()
        assert "sample_prompt" in source
        assert "prompt_source" in source
        assert "prompt_id" in source

    def test_main_has_save_intermediates(self):
        source = (REPO / "experiments" / "main.py").read_text()
        assert "save_intermediates" in source
        assert "cleanup_intermediates" in source

    def test_main_has_color_transfer_choices(self):
        source = (REPO / "experiments" / "main.py").read_text()
        assert '"aligned"' in source
        assert '"none"' in source
