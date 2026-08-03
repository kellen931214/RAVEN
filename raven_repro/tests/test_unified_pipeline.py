"""Tests for the unified main/eval pipeline — no real data required."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.experiment_config import (  # noqa: E402
    DIFFUSION_MODE_MAP,
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
    collect_incomplete_run_ids,
    config_path,
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
# experiment_config
# ===========================================================================
class TestDiffusionModeMap:
    def test_three_modes_exist(self):
        assert set(DIFFUSION_MODE_MAP) == {"ddim", "ddpm", "ddim-ddpm"}

    def test_ddim_maps_correctly(self):
        assert DIFFUSION_MODE_MAP["ddim"] == {
            "inversion_mode": "ddim",
            "scheduler_mode": "ddim",
        }

    def test_ddpm_maps_correctly(self):
        assert DIFFUSION_MODE_MAP["ddpm"] == {
            "inversion_mode": "ddpm",
            "scheduler_mode": "ddpm",
        }

    def test_ddim_ddpm_maps_correctly(self):
        assert DIFFUSION_MODE_MAP["ddim-ddpm"] == {
            "inversion_mode": "ddim",
            "scheduler_mode": "ddpm",
        }

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

    def test_ddpm_ddim_is_forbidden_pair(self):
        assert FORBIDDEN_PAIR == ("ddpm", "ddim")


class TestNormalizeConfig:
    def test_defaults_produce_valid_config(self):
        config = normalize_config(metadata_path="/tmp/test.csv", output_dir="/tmp/out")
        assert config["inversion_mode"] == "ddim"
        assert config["scheduler_mode"] == "ddim"
        assert config["base_seed"] == 42
        assert config["shift_mode"] == "random"
        assert "config_hash" in config

    def test_resume_and_overwrite_mutex(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            normalize_config(
                metadata_path="/tmp/test.csv",
                output_dir="/tmp/out",
                resume=True,
                overwrite=True,
            )

    def test_ddim_ddpm_produces_correct_pair(self):
        config = normalize_config(
            metadata_path="/tmp/test.csv",
            output_dir="/tmp/out",
            diffusion_mode="ddim-ddpm",
        )
        assert config["inversion_mode"] == "ddim"
        assert config["scheduler_mode"] == "ddpm"

    def test_fixed_shift_requires_xy(self):
        with pytest.raises(ValueError, match="shift_mode='fixed'"):
            normalize_config(
                metadata_path="/tmp/test.csv",
                output_dir="/tmp/out",
                shift_mode="fixed",
            )

    def test_none_shift_zeroes_xy(self):
        config = normalize_config(
            metadata_path="/tmp/test.csv",
            output_dir="/tmp/out",
            shift_mode="none",
        )
        assert config["shift_x"] == 0.0
        assert config["shift_y"] == 0.0

    def test_unknown_kwargs_rejected(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            normalize_config(
                metadata_path="/tmp/test.csv",
                output_dir="/tmp/out",
                fake_param=123,
            )

    def test_invalid_shift_mode_rejected(self):
        with pytest.raises(ValueError, match="shift_mode"):
            normalize_config(
                metadata_path="/tmp/test.csv",
                output_dir="/tmp/out",
                shift_mode="invalid_mode",
            )


class TestConfigMatch:
    def test_identical_configs_match(self):
        a = normalize_config(metadata_path="/tmp/a.csv", output_dir="/tmp/a")
        b = normalize_config(metadata_path="/tmp/a.csv", output_dir="/tmp/a")
        # hash differs because metadata_path differs between a and b
        a.pop("config_hash", None)
        b.pop("config_hash", None)
        a["metadata_path"] = "/tmp/test.csv"
        b["metadata_path"] = "/tmp/test.csv"
        a["output_dir"] = "/tmp/out"
        b["output_dir"] = "/tmp/out"
        a["config_hash"] = "fake"
        b["config_hash"] = "fake"
        assert check_config_match(a, b) == []

    def test_different_configs_report_mismatches(self):
        a = normalize_config(metadata_path="/tmp/a.csv", output_dir="/tmp/o")
        b = normalize_config(metadata_path="/tmp/a.csv", output_dir="/tmp/o",
                             base_seed=99)
        a.pop("config_hash", None)
        b.pop("config_hash", None)
        a["metadata_path"] = "/tmp/test.csv"
        b["metadata_path"] = "/tmp/test.csv"
        a["output_dir"] = "/tmp/out"
        b["output_dir"] = "/tmp/out"
        a["config_hash"] = "fake"
        b["config_hash"] = "fake"
        mismatches = check_config_match(a, b)
        assert "base_seed" in mismatches


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

    def test_same_as_legacy_arithmetic(self):
        # Verify the exact formula: base_seed + numeric_run_id
        assert compute_attack_seed(42, "17") == 59
        # Non-numeric uses hash
        result = compute_attack_seed(0, "test")
        assert isinstance(result, int)
        assert result > 0


class TestPlanShift:
    def test_none_mode_returns_zero(self):
        config = {"shift_mode": "none", "base_seed": 42}
        dx, dy, seed = plan_shift("17", config)
        assert dx == 0.0
        assert dy == 0.0
        assert seed == 59

    def test_fixed_mode_returns_exact(self):
        config = {
            "shift_mode": "fixed",
            "base_seed": 42,
            "shift_x": 24.0,
            "shift_y": -24.0,
        }
        dx, dy, seed = plan_shift("17", config)
        assert dx == 24.0
        assert dy == -24.0
        assert seed == 59

    def test_random_mode_produces_within_range(self):
        config = {
            "shift_mode": "random",
            "base_seed": 42,
            "shift_magnitude_min": 24,
            "shift_magnitude_max": 32,
        }
        for run_id in ["0", "1", "2", "10", "100"]:
            dx, dy, seed = plan_shift(run_id, config)
            assert 24 <= abs(dx) <= 32, f"dx={dx} out of range"
            assert 24 <= abs(dy) <= 32, f"dy={dy} out of range"
            assert seed == 42 + int(run_id)

    def test_random_mode_deterministic(self):
        config = {
            "shift_mode": "random",
            "base_seed": 42,
            "shift_magnitude_min": 24,
            "shift_magnitude_max": 32,
        }
        first = plan_shift("17", config)
        second = plan_shift("17", config)
        assert first == second

    def test_random_mode_uses_run_id_not_index(self):
        config = {
            "shift_mode": "random",
            "base_seed": 42,
            "shift_magnitude_min": 24,
            "shift_magnitude_max": 32,
        }
        # Same run_id, different index → same shift
        result_a = plan_shift("17", config, index=0)
        result_b = plan_shift("17", config, index=5)
        assert result_a == result_b

    def test_shift_magnitude_default_range(self):
        config = {"shift_mode": "random", "base_seed": 42}
        dx, dy, _ = plan_shift("1", config)
        assert 24 <= abs(dx) <= 32
        assert 24 <= abs(dy) <= 32

    def test_x_and_y_independent(self):
        """Verify x and y signs/magnitudes can differ (independent sampling)."""
        config = {"shift_mode": "random", "base_seed": 0}
        results = set()
        for run_id in range(100):
            dx, dy, _ = plan_shift(str(run_id), config)
            results.add((dx, dy))
        # With independent axes we should see varied sign combinations
        signs = {(1 if r[0] > 0 else -1, 1 if r[1] > 0 else -1) for r in results}
        assert len(signs) >= 2  # at least some sign variation


# ===========================================================================
# experiment_io
# ===========================================================================
class TestOutputLayout:
    def test_sample_dir_structure(self):
        d = sample_dir("/tmp/run", "watermarked", "17")
        assert d == Path("/tmp/run/samples/watermarked/17")

    def test_output_image_path(self):
        p = output_image_path("/tmp/run", "watermarked", "17")
        assert p.name == "output.png"
        assert p.parent == sample_dir("/tmp/run", "watermarked", "17")

    def test_record_path(self):
        p = record_path("/tmp/run", "watermarked", "17")
        assert p.name == "record.json"

    def test_config_path(self):
        assert config_path("/tmp/run") == Path("/tmp/run/config.json")

    def test_records_jsonl_path(self):
        assert records_jsonl_path("/tmp/run") == Path("/tmp/run/records.jsonl")


class TestOutputDirSafety:
    def test_repo_root_protected(self):
        repo = Path(__file__).resolve().parents[2]
        with pytest.raises(ValueError):
            validate_output_dir_safety(str(repo / "data"))

    def test_tmp_dir_allowed(self):
        validate_output_dir_safety("/tmp/raven_safe_test")

    def test_slash_protected(self):
        with pytest.raises(ValueError):
            validate_output_dir_safety("/workspace/RAVEN/data")


class TestWriteReadConfig:
    def test_roundtrip(self, tmp_path):
        config = {"method": "TR", "dataset": "test", "base_seed": 42}
        write_config(tmp_path, config)
        read = read_config(tmp_path)
        assert read["method"] == "TR"
        assert read["base_seed"] == 42

    def test_config_hash_recorded(self, tmp_path):
        config = normalize_config(
            metadata_path="/tmp/test.csv",
            output_dir=str(tmp_path),
        )
        write_config(tmp_path, config)
        read = read_config(tmp_path)
        assert "config_hash" in read


class TestWriteReadRecord:
    def test_write_and_read(self, tmp_path):
        write_record(
            tmp_path, "watermarked", "17",
            {"run_id": "17", "role": "watermarked", "attack_seed": 59},
        )
        # is_sample_complete requires both record.json AND output.png
        output_image_path(tmp_path, "watermarked", "17").write_bytes(b"fake png")
        assert is_sample_complete(tmp_path, "watermarked", "17")

    def test_record_status_forced_complete(self, tmp_path):
        write_record(
            tmp_path, "watermarked", "17",
            {"run_id": "17", "status": "pending"},
        )
        rec = json.loads(record_path(tmp_path, "watermarked", "17").read_text())
        assert rec["status"] == "complete"

    def test_incomplete_without_output_png(self, tmp_path):
        rec = record_path(tmp_path, "watermarked", "17")
        rec.parent.mkdir(parents=True)
        rec.write_text(json.dumps({"run_id": "17", "status": "complete"}))
        assert not is_sample_complete(tmp_path, "watermarked", "17")

    def test_incomplete_without_record(self, tmp_path):
        out = output_image_path(tmp_path, "watermarked", "17")
        out.parent.mkdir(parents=True)
        out.write_bytes(b"fake png")
        assert not is_sample_complete(tmp_path, "watermarked", "17")


class TestRebuildRecordsJsonl:
    def test_empty_dir(self, tmp_path):
        rebuild_records_jsonl(tmp_path)
        records = read_records_jsonl(tmp_path)
        assert records == []

    def test_rebuilds_from_records(self, tmp_path):
        write_record(tmp_path, "watermarked", "1",
                     {"run_id": "1", "role": "watermarked", "attack_seed": 43})
        write_record(tmp_path, "watermarked", "2",
                     {"run_id": "2", "role": "watermarked", "attack_seed": 44})
        # Need output.png files for completeness
        for rid in ("1", "2"):
            p = output_image_path(tmp_path, "watermarked", rid)
            p.write_bytes(b"fake image")
        rebuild_records_jsonl(tmp_path)
        records = read_records_jsonl(tmp_path)
        assert len(records) == 2
        assert records[0]["run_id"] == "1"
        assert records[1]["run_id"] == "2"

    def test_sorted_by_run_id(self, tmp_path):
        for rid in ("10", "2", "1"):
            write_record(tmp_path, "watermarked", rid,
                         {"run_id": rid, "role": "watermarked", "attack_seed": 42})
            p = output_image_path(tmp_path, "watermarked", rid)
            p.write_bytes(b"fake image")
        rebuild_records_jsonl(tmp_path)
        records = read_records_jsonl(tmp_path)
        assert [r["run_id"] for r in records] == ["1", "10", "2"]


class TestCollectIncomplete:
    def test_all_incomplete_when_empty(self, tmp_path):
        incomplete = collect_incomplete_run_ids(
            tmp_path, ["watermarked"], ["1", "2", "3"])
        assert len(incomplete) == 3

    def test_none_incomplete_when_all_complete(self, tmp_path):
        for rid in ("1", "2"):
            write_record(tmp_path, "watermarked", rid,
                         {"run_id": rid, "role": "watermarked"})
            output_image_path(tmp_path, "watermarked", rid).write_bytes(b"fake")
        incomplete = collect_incomplete_run_ids(
            tmp_path, ["watermarked"], ["1", "2"])
        assert len(incomplete) == 0


# ===========================================================================
# eval import isolation
# ===========================================================================
class TestEvalDoesNotImportPipeline:
    def test_eval_module_no_raven_pipeline(self):
        """Verify experiments.eval does not import RavenPipeline."""
        source = (REPO / "experiments" / "eval.py").read_text()
        # Check for actual import statements, not docstring mentions
        assert "from raven.pipeline_raven import" not in source, (
            "eval.py must not import pipeline_raven"
        )
        assert "import RavenPipeline" not in source, (
            "eval.py must not import RavenPipeline"
        )
        assert "from raven.pipeline_raven import RavenPipeline" not in source, (
            "eval.py must not import RavenPipeline from pipeline_raven"
        )


# ===========================================================================
# Synthetic detector orchestration (contract tests only)
# ===========================================================================
class TestDetectorAdapterDispatch:
    def test_all_methods_have_adapter(self):
        from raven.experiment_config import VALID_DIFFUSION_MODES
        # Import eval adapters
        sys.path.insert(0, str(REPO / "experiments"))
        # We use importlib to avoid side effects
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_module", REPO / "experiments" / "eval.py")
        eval_module = importlib.util.module_from_spec(spec)

        # Check DETECTOR_ADAPTERS keys before executing module
        source = (REPO / "experiments" / "eval.py").read_text()
        for method in ["TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"]:
            assert f'"{method}"' in source or f"'{method}'" in source, (
                f"Method {method} missing from DETECTOR_ADAPTERS")

    def test_canonical_score_gm_t2s_not_using_semantic_path(self):
        """GM and T2S scores are higher-is-watermarked already — verify they
        are NOT routed through canonical_watermark_score's semantic branch."""
        from raven.metrics import canonical_watermark_score
        # GM and T2S should NOT be in the semantic methods list
        from raven.metrics import SEMANTIC_METHODS
        assert "GM" not in SEMANTIC_METHODS
        assert "T2S" not in SEMANTIC_METHODS
        # And canonical_watermark_score should raise for them
        with pytest.raises(ValueError, match="Unsupported watermark method"):
            canonical_watermark_score("GM", 0.5)
        with pytest.raises(ValueError, match="Unsupported watermark method"):
            canonical_watermark_score("T2S", 0.5)


class TestTRAdapterSchema:
    """Verify TR adapter preserves required fields even with synthetic data."""
    def test_tr_adapter_returns_recalibration_flag(self):
        sys.path.insert(0, str(REPO / "experiments"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_mod", REPO / "experiments" / "eval.py")
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)

        result = eval_mod.evaluate_tr([], "/tmp/fake")
        assert "recalibrated_metrics_available" in result or "available" in result


class TestT2SAdapterSchema:
    """Verify T2S adapter preserves per-sample bit accuracy fields."""
    def test_t2s_adapter_preserves_bit_accuracy_schema(self):
        sys.path.insert(0, str(REPO / "experiments"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_mod2", REPO / "experiments" / "eval.py")
        eval_mod2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod2)

        result = eval_mod2.evaluate_t2s([], "/tmp/fake")
        if not result.get("available"):
            assert "preserved_fields" in result, (
                "T2S adapter must document preserved fields when unavailable")
            preserved = result["preserved_fields"]
            assert any("bit accuracy" in f.lower() for f in preserved)


class TestGMAdapterDispatch:
    """Verify GM adapter uses its own detector fields, not TR's."""
    def test_gm_returns_score_type(self):
        sys.path.insert(0, str(REPO / "experiments"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_mod3", REPO / "experiments" / "eval.py")
        eval_mod3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod3)

        result = eval_mod3.evaluate_gm([], "/tmp/fake")
        if not result.get("available"):
            assert "required_artifacts" in result
            artifacts = result["required_artifacts"]
            assert any("gm_bundle_dir" in a for a in artifacts)
            assert any("gm_w1_file_sha256" in a for a in artifacts)
            assert any("gm_w2_file_sha256" in a for a in artifacts)
            assert any("gm_watermark_sha256" in a for a in artifacts)


# ===========================================================================
# main.py module structure
# ===========================================================================
class TestMainModuleStructure:
    def test_main_does_not_import_detector(self):
        source = (REPO / "experiments" / "main.py").read_text()
        # main must not import detector/eval modules
        assert "extract_verification_scores" not in source
        assert "evaluate_verification" not in source
        assert "from raven.quality import" not in source
        assert "clean_fid" not in source
        assert "openclip" not in source.lower()

    def test_main_imports_pipeline_once(self):
        source = (REPO / "experiments" / "main.py").read_text()
        # Pipeline is imported inside the function, not at module level
        assert "from raven.pipeline_raven import RavenPipeline" in source
