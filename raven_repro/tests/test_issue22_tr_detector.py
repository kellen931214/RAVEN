"""Issue #22 regression tests — fail-closed TR cohort configuration.

All tests call the real ``tr_detector.load_state`` (mocking only the
heavy imports: pipe / TrProvider / extract_module) to exercise the
record-by-record validation, uniform-config enforcement, and identity-field
checks.  score_image and aggregate are tested with synthetic records.

Run:  pytest -q raven_repro/tests/test_issue22_tr_detector.py
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Synthetic TR metadata — canonical 7-field complete record
# ---------------------------------------------------------------------------
TR_META_COMPLETE: dict[str, str] = {
    "w_seed": "99",
    "w_channel": "3",
    "w_radius": "10",
    "w_pattern": "ring",
    "w_mask_shape": "circle",
    "w_measurement": "l1_complex",
    "w_injection": "complex",
}


from contextlib import contextmanager


def _make_record(run_id="1", role="watermarked", method="TR", **kw):
    """Build a synthetic record with TR provider fields at the top level.

    In the real pipeline ``MetadataResolver.enrich_record`` copies metadata
    fields to the record's top level.  Tests that call ``load_state`` directly
    must provide those fields at the top level.
    """
    source_meta = kw.pop("source_metadata", dict(TR_META_COMPLETE))
    record = {
        "run_id": run_id,
        "role": role,
        "method": method,
        "input_path": kw.get("input_path", f"/tmp/in_{run_id}.png"),
        "output_path": f"/tmp/out/{role}/{run_id}/output.png",
        "prompt": kw.get("prompt", ""),
        "attack_seed": 59,
        "planned_flow_dx_image_px": 24.0,
        "planned_flow_dy_image_px": -24.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -24.0,
        "debug_info_path": "",
        "debug_info_retained": False,
        "source_metadata": source_meta,
        **kw,
    }
    # Copy TR provider fields from source_metadata to record top level
    # (mimics MetadataResolver.enrich_record behaviour)
    for key in TR_META_COMPLETE:
        if key in source_meta:
            record[key] = source_meta[key]
    # Also copy identity fields if present in source_metadata
    for key in ("model_id", "model_revision", "scheduler",
                "inverse_scheduler", "watermark_target_sha256",
                "watermark_mask_sha256", "provider_config_hash"):
        if key in source_meta:
            record[key] = source_meta[key]
    return record


@contextmanager
def _mock_load_state_deps(monkeypatch):
    """Replace heavy imports so load_state can run CPU-only.

    Patches ``_get_extract_module`` and all ``eval_bench_wm`` imports
    inside ``load_state``.  No real package is ever touched.
    """
    import raven.detectors.tr_detector as mod
    import builtins

    fake_extract = mock.MagicMock()
    monkeypatch.setattr(mod, "_extract_module", fake_extract)
    monkeypatch.setattr(mod, "_get_extract_module",
                        lambda: fake_extract)

    fake_pipe = mock.MagicMock()
    fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
    fake_pipe.get_dtype.return_value = "torch.float32"

    fake_pipe_utils = mock.MagicMock()
    fake_pipe_utils.get_pipe_provider.return_value = fake_pipe

    fake_tr_provider_class = mock.MagicMock()
    fake_tr_module = mock.MagicMock(TrProvider=fake_tr_provider_class)

    fake_wm = mock.MagicMock(tr_provider=fake_tr_module)
    fake_utils = mock.MagicMock(pipe=mock.MagicMock(pipe_utils=fake_pipe_utils),
                                wm=fake_wm)
    fake_eb = mock.MagicMock(utils=fake_utils)
    fake_eb.__path__ = []  # prevents pkgutil from recursing

    _import_modules = {
        "eval_bench_wm": fake_eb,
        "eval_bench_wm.utils": fake_utils,
        "eval_bench_wm.utils.pipe": fake_utils.pipe,
        "eval_bench_wm.utils.wm": fake_wm,
        "eval_bench_wm.utils.wm.tr_provider": fake_tr_module,
    }

    original_import = builtins.__import__

    def _mock_import(name, globals=None, locals=None, fromlist=(),
                     level=0):
        if name in _import_modules:
            return _import_modules[name]
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    yield


# ===========================================================================
# 1 — Each missing required field → DetectorMissingStateError
# ===========================================================================
class TestMissingRequiredFields:
    REQUIRED = sorted(TR_META_COMPLETE)

    def test_all_fields_present_passes(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        records = [_make_record("1", source_metadata=dict(TR_META_COMPLETE))]
        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert "provider" in result
        assert "provider_config_hash" in result

    @pytest.mark.parametrize("field", sorted(TR_META_COMPLETE))
    def test_missing_field_raises(self, monkeypatch, field):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE)
        del meta[field]
        records = [_make_record("1", source_metadata=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match=field):
                load_state(records, "cpu")

    def test_empty_field_raises(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE)
        meta["w_seed"] = ""
        records = [_make_record("1", source_metadata=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="w_seed"):
                load_state(records, "cpu")

    def test_whitespace_field_raises(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta = dict(TR_META_COMPLETE)
        meta["w_channel"] = "   "
        records = [_make_record("1", source_metadata=meta)]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="w_channel"):
                load_state(records, "cpu")

    def test_empty_records_raises(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError):
                load_state([], "cpu")

    def test_second_record_missing_field_raises(self, monkeypatch):
        """Not just first — every record is validated."""
        from raven.detectors.tr_detector import load_state, DetectorMissingStateError

        meta_bad = dict(TR_META_COMPLETE)
        del meta_bad["w_radius"]
        records = [
            _make_record("1", source_metadata=dict(TR_META_COMPLETE)),
            _make_record("2", source_metadata=meta_bad),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorMissingStateError, match="record index 1"):
                load_state(records, "cpu")


# ===========================================================================
# 2 — Mixed w_seed → DetectorStateValidationError
# ===========================================================================
class TestMixedSeed:
    def test_different_w_seed_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE)
        meta_b = dict(TR_META_COMPLETE)
        meta_b["w_seed"] = "88888"

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_same_w_seed_ok(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [
            _make_record("1", source_metadata=dict(TR_META_COMPLETE)),
            _make_record("2", source_metadata=dict(TR_META_COMPLETE)),
        ]

        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert "provider" in result


# ===========================================================================
# 3 — Mixed radius / pattern → DetectorStateValidationError
# ===========================================================================
class TestMixedRadiusPattern:
    def test_different_radius_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE, w_radius="10")
        meta_b = dict(TR_META_COMPLETE, w_radius="15")

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_different_pattern_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE, w_pattern="ring")
        meta_b = dict(TR_META_COMPLETE, w_pattern="zeros")

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_different_mask_shape_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE, w_mask_shape="circle")
        meta_b = dict(TR_META_COMPLETE, w_mask_shape="square")

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_different_measurement_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE, w_measurement="l1_complex")
        meta_b = dict(TR_META_COMPLETE, w_measurement="l2_complex")

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_different_injection_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE, w_injection="complex")
        meta_b = dict(TR_META_COMPLETE, w_injection="seed")

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")

    def test_different_channel_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        meta_a = dict(TR_META_COMPLETE, w_channel="3")
        meta_b = dict(TR_META_COMPLETE, w_channel="0")

        records = [
            _make_record("1", source_metadata=meta_a),
            _make_record("2", source_metadata=meta_b),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="Mixed TR provider"):
                load_state(records, "cpu")


# ===========================================================================
# 4 — Mixed target / mask identity → DetectorStateValidationError
# ===========================================================================
class TestMixedTargetMask:
    def test_mixed_target_sha_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE,
                watermark_target_sha256="aaa111",
            )),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE,
                watermark_target_sha256="bbb222",
            )),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="watermark_target_sha256"):
                load_state(records, "cpu")

    def test_mixed_mask_sha_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE,
                watermark_mask_sha256="ccc333",
            )),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE,
                watermark_mask_sha256="ddd444",
            )),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="watermark_mask_sha256"):
                load_state(records, "cpu")

    def test_same_target_and_mask_accepted(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE,
                watermark_target_sha256="eee555",
                watermark_mask_sha256="fff666",
            )),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE,
                watermark_target_sha256="eee555",
                watermark_mask_sha256="fff666",
            )),
        ]

        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert "provider" in result

    def test_mixed_model_id_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE, model_id="sd-2-1-base")),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE, model_id="sd-2-0-base")),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="model_id"):
                load_state(records, "cpu")

    def test_mixed_provider_config_hash_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash="hash111")),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash="hash222")),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="provider_config_hash"):
                load_state(records, "cpu")


# ===========================================================================
# 5 — Uniform provider configuration
# ===========================================================================
class TestUniformProvider:
    def test_single_cohort_one_provider(self, monkeypatch):
        """A uniform cohort constructs exactly one provider."""
        from raven.detectors.tr_detector import load_state

        records = [
            _make_record("1", source_metadata=dict(TR_META_COMPLETE)),
            _make_record("2", source_metadata=dict(TR_META_COMPLETE)),
            _make_record("3", source_metadata=dict(TR_META_COMPLETE)),
        ]

        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert "provider" in result
        assert "provider_config_hash" in result
        assert len(result["provider_kwargs"]) == 7

    def test_single_record_cohort_one_provider(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_make_record("1", source_metadata=dict(TR_META_COMPLETE))]

        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert "provider" in result

    def test_provider_kwargs_match_source(self, monkeypatch):
        from raven.detectors.tr_detector import load_state

        records = [_make_record("1", source_metadata=dict(
            TR_META_COMPLETE, w_seed="42", w_radius="8"))]

        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert result["provider_kwargs"]["w_seed"] == 42
        assert result["provider_kwargs"]["w_radius"] == 8


# ===========================================================================
# 6 — Canonical helper delegation (no reimplementation)
# ===========================================================================
class TestCanonicalDelegation:
    def test_score_image_delegates_to_evaluate_image(self, monkeypatch):
        """score_image calls mod.evaluate_image / raw_score / canonical_score."""
        import raven.detectors.tr_detector as tr_mod

        fake_mod = mock.MagicMock()
        fake_mod.evaluate_image.return_value = {
            "p_values": [0.001],
            "p_value_diagnostics": [{"log_p": -20.0, "sigma": 1.0,
                                     "lambda": 100.0, "statistic": 50.0,
                                     "df": 100, "p_underflow": False}],
        }
        fake_mod.raw_score.return_value = 0.001
        fake_mod.canonical_score.return_value = 10.0

        provider_info = {
            "provider": mock.MagicMock(),
            "pipe": mock.MagicMock(),
            "extract_module": fake_mod,
        }

        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "test.png"
            img.write_bytes(b"fake png")
            from PIL import Image
            Image.new("RGB", (64, 64)).save(img)

            result = tr_mod.score_image(provider_info, str(img))

        fake_mod.evaluate_image.assert_called_once()
        fake_mod.raw_score.assert_called_once_with("TR",
                                                    fake_mod.evaluate_image.return_value)
        fake_mod.canonical_score.assert_called_once_with(
            "TR", 0.001, fake_mod.evaluate_image.return_value)
        assert result["raw_score"] == 0.001
        assert result["canonical_score"] == 10.0
        assert result.get("tr_log_p") == -20.0

    def test_tr_math_not_reimplemented(self):
        """TR detector source never reimplements scipy.stats.ncx2."""
        source = (REPO / "raven_repro" / "raven" / "detectors"
                  / "tr_detector.py").read_text()
        assert "scipy.stats" not in source
        assert "ncx2" not in source
        assert "torch.fft" not in source


# ===========================================================================
# 7 — Threshold metrics without attacked-clean
# ===========================================================================
class TestThresholdWithoutAttackedClean:
    def test_three_cohorts_no_attacked_clean(self):
        """original_clean + original_watermarked + attacked_watermarked
        produce detection_summary but tr_recalibrated unavailable."""
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 4.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 11.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 6.0},
        ]

        result = aggregate(rows)
        assert result["scored_count"] == 6
        assert result["failed_count"] == 0
        assert "detection_summary" in result
        assert result["detection_summary"]["target_fpr"] == 0.01
        # Recalibrated not available without attacked_clean
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is False

    def test_no_clean_yields_missing_cohort(self):
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
        ]

        result = aggregate(rows)
        assert "original_clean" not in result.get("cohort_counts", {})
        assert "detection_summary" not in result


# ===========================================================================
# 8 — Recalibration with attacked-clean
# ===========================================================================
class TestRecalibrationWithAttackedClean:
    def test_all_four_cohorts_recalibrated_available(self):
        """attacked_clean present + clean → tr_recalibrated with valid metrics."""
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            # original_clean — 2 scored
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 4.0},
            # original_watermarked — 2 scored
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 11.0},
            # attacked_watermarked — 2 scored
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 6.0},
            # attacked_clean — 2 scored
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_clean",
             "canonical_score": 3.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_clean",
             "canonical_score": 2.0},
        ]

        result = aggregate(rows)
        assert result["scored_count"] == 8
        assert "detection_summary" in result
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is True
        assert result["tr_recalibrated"]["attacked_clean_count"] == 2
        assert "attacked_clean_recalibrated_threshold" in result["tr_recalibrated"]

    def test_attacked_clean_without_original_clean_no_recal(self):
        """attacked_clean present but NO original_clean → no recalibration."""
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_clean",
             "canonical_score": 3.0},
        ]

        result = aggregate(rows)
        # attacked_clean exists but no original_clean
        assert result["tr_recalibrated"]["recalibrated_metrics_available"] is False

    def test_failed_rows_excluded_from_cohorts(self):
        """Rows with status != scored are excluded from aggregate."""
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import (
            ROW_STATUS_SCORED, ROW_STATUS_FAILED_SCORING,
        )

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_FAILED_SCORING, "evaluation_cohort": "original_clean",
             "canonical_score": None, "error": "boom"},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
        ]

        result = aggregate(rows)
        assert result["scored_count"] == 3
        assert result["failed_count"] == 1


# ===========================================================================
# 9 — Mixed records with same provider config hash but different
#     raw field values (can't happen with hash but exercise the path)
# ===========================================================================
class TestProviderConfigHashRecorded:
    def test_recorded_hash_matches_computed(self, monkeypatch):
        """When records carry an explicit provider_config_hash it must match."""
        from raven.detectors.tr_detector import load_state

        # The computed hash from canonical helpers
        from raven.eval_protocol import provider_config_hash
        expected = provider_config_hash("TR", TR_META_COMPLETE)

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash=expected)),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash=expected)),
        ]

        with _mock_load_state_deps(monkeypatch):
            result = load_state(records, "cpu")
        assert result["provider_config_hash"] == expected

    def test_recorded_hash_mismatch_computed(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash="deadbeef0000")),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError,
                               match="provider_config_hash"):
                load_state(records, "cpu")

    def test_mixed_recorded_hashes_rejected(self, monkeypatch):
        from raven.detectors.tr_detector import load_state, DetectorStateValidationError
        from raven.eval_protocol import provider_config_hash

        expected = provider_config_hash("TR", TR_META_COMPLETE)

        records = [
            _make_record("1", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash=expected)),
            _make_record("2", source_metadata=dict(
                TR_META_COMPLETE, provider_config_hash="mixed1111")),
        ]

        with _mock_load_state_deps(monkeypatch):
            with pytest.raises(DetectorStateValidationError):
                load_state(records, "cpu")


# ===========================================================================
# 10 — Aggregate field completeness
# ===========================================================================
class TestAggregateFields:
    def test_required_result_fields(self):
        from raven.detectors.tr_detector import aggregate
        from raven.detectors import ROW_STATUS_SCORED

        rows = [
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 5.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_clean",
             "canonical_score": 4.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 10.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "original_watermarked",
             "canonical_score": 11.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 7.0},
            {"status": ROW_STATUS_SCORED, "evaluation_cohort": "attacked_watermarked",
             "canonical_score": 6.0},
        ]

        result = aggregate(rows)
        for field in ("method", "requested_count", "scored_count",
                      "failed_count", "cohort_counts", "missing_cohorts"):
            assert field in result, f"Missing field: {field}"

        summary = result["detection_summary"]
        for field in ("target_fpr", "threshold_comparison_operator",
                      "original_clean_threshold", "original_clean_target_fpr",
                      "original_clean_actual_fpr", "original_watermarked_tpr",
                      "attacked_watermarked_tpr_at_original_threshold",
                      "attack_success_at_original_threshold"):
            assert field in summary, f"Missing summary field: {field}"
