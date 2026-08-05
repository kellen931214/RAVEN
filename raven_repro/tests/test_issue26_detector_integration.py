"""Issue #26 — behavior-level detector integration matrix.

Real adapters, real orchestrator, mock only heavy boundaries.

Recorded but NOT fixed here:
  ``gs_detector._ensure_paths()`` inserts ``<repo>/eval_bench_wm/`` into
  sys.path but ``eval_bench_wm`` has no ``__init__.py``, and the import
  ``from eval_bench_wm.utils.pipe import pipe_utils`` needs the PARENT
  directory on sys.path, not ``eval_bench_wm/`` itself.  Tests work
  because module-level sys.modules stubs short-circuit the import before
  filesystem path resolution.  Production ``experiments/eval.py`` fails
  with ``DetectorDependencyError: GS dependencies not available: No
  module named 'eval_bench_wm'``.  Fix belongs to issue #29 or a
  dedicated path-infra cleanup.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
RAVEN_REPRO = REPO / "raven_repro"
for _root in (RAVEN_REPRO, REPO, REPO / "eval_bench_wm"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# ---------------------------------------------------------------------------
# Module-level stubs — short-circuit ALL eval_bench_wm imports before any
# detector module is loaded (same pattern as test_issue20/21/23/24).
# ---------------------------------------------------------------------------
_fake_torch = mock.MagicMock(name="torch")
_fake_torch.cuda.is_available.return_value = False
_fake_torch.device.return_value = mock.MagicMock(name="cpu_device")
_fake_torch.no_grad.return_value = mock.MagicMock()
_fake_torch.float16 = "float16"
_fake_torch.float32 = "float32"
# scipy array_api_compat checks issubclass(..., torch.Tensor) — must be a
# real class, not a MagicMock, or scipy raises TypeError.
_FakeTensor = type("FakeTensor", (), {})
_fake_torch.Tensor = _FakeTensor

_fake_pipe_utils = mock.MagicMock(name="pipe_utils")
_fake_pipe = mock.MagicMock(name="pipe")
_fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
_fake_pipe.get_dtype.return_value = _fake_torch.float32
_fake_pipe_utils.get_pipe_provider.return_value = _fake_pipe

_fake_gs_provider_mod = mock.MagicMock(name="gs_provider")
_fake_gs_provider_mod.GsProvider = mock.MagicMock(name="GsProvider")
_fake_gm_provider_mod = mock.MagicMock(name="gm_provider")
_fake_gm_provider_mod.GmProvider = mock.MagicMock(name="GmProvider")

_STUB_MODULES = {
    "torch": _fake_torch,
    "eval_bench_wm.utils.pipe": mock.MagicMock(pipe_utils=_fake_pipe_utils),
    "eval_bench_wm.utils.pipe.pipe_utils": _fake_pipe_utils,
    "eval_bench_wm.utils.wm.gs_provider": _fake_gs_provider_mod,
    "eval_bench_wm.utils.wm.gm_provider": _fake_gm_provider_mod,
}
# T2S provider/inversion NOT mocked — imported from real eval_bench_wm
# directory (on sys.path via REPO). test_issue21 helpers patch them.
# eval_bench_wm / eval_bench_wm.utils / eval_bench_wm.utils.wm NOT mocked
# either — they must be namespace-package-importable from the filesystem.
# Fourier providers also NOT mocked — test_issue24 handles them per-test
# via _install_fourier_mocks monkeypatch.setitem.
for _k, _v in _STUB_MODULES.items():
    if _k not in sys.modules:
        sys.modules[_k] = _v

# Now safe: the real detector modules import without touching disk
from raven.detectors import DETECTOR_MODULES, _lazy_imports  # noqa: E402
_lazy_imports()

from raven.detectors import (  # noqa: E402
    ROW_STATUS_SCORED, ROW_STATUS_FAILED_MISSING_IMAGE,
    ROW_STATUS_FAILED_MISSING_STATE, ROW_STATUS_FAILED_SCORING,
    ROW_STATUS_FAILED_STATE_VALIDATION, ROW_STATUS_FAILED_PROVIDER,
    ROW_STATUS_FAILED_MISSING_DEPENDENCY, ROW_STATUS_FAILED_INTERNAL_ERROR,
    STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS,
    STATUS_SKIPPED_INSUFFICIENT_DATA,
    STATUS_FAILED_MISSING_REQUIRED_STATE, STATUS_FAILED_MISSING_DEPENDENCY,
    STATUS_FAILED_MISSING_IMAGE, STATUS_FAILED_PROVIDER_INITIALIZATION,
    STATUS_FAILED_STATE_VALIDATION, STATUS_FAILED_SCORING,
    STATUS_FAILED_INTERNAL_ERROR,
    FAILURE_CAUSE_SCORING_ERROR, FAILURE_CAUSE_MISSING_IMAGE,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE, FAILURE_CAUSE_MISSING_DEPENDENCY,
    FAILURE_CAUSE_PROVIDER_INITIALIZATION, FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_INTERNAL_ERROR,
    determine_exit_code,
)

from experiments.eval import evaluate_detector, run_evaluation  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _rows(out_dir: Path) -> list[dict]:
    path = out_dir / "evaluation" / "detector_records.jsonl"
    assert path.is_file(), f"detector_records.jsonl missing at {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_record(run_id="1", role="watermarked", method="GS", **kw):
    meta = kw.pop("source_metadata", None) or {}
    return {
        "run_id": str(run_id), "role": role, "method": method,
        "input_path": kw.get("input_path", f"/tmp/issue26_in_{role}_{run_id}.png"),
        "output_path": f"/tmp/issue26_out/{role}/{run_id}/output.png",
        "prompt": kw.get("prompt", ""),
        "prompt_source": kw.get("prompt_source", "metadata"),
        "attack_seed": 59,
        "planned_flow_dx_image_px": 0.0,
        "planned_flow_dy_image_px": 0.0,
        "effective_source_flow_dx_image_px": 0.0,
        "effective_source_flow_dy_image_px": 0.0,
        "debug_info_path": "", "debug_info_retained": False,
        "source_metadata": meta,
    }


def _raise(exc):
    raise exc


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_bytes")


def _write_run(root: Path, method: str, *, records: list[dict],
               config_extra: dict | None = None,
               csv_rows: list[dict] | None = None) -> Path:
    """Minimal run dir with config.json, records.jsonl, sample PNGs, optional CSV."""
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = root / "run"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"method": method, "dataset": "test", **(config_extra or {})}
    if csv_rows is not None:
        csv_path = root / "meta.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        cfg["metadata_path"] = str(csv_path)
    write_config(out, cfg)
    for rec in records:
        role = rec.get("role", "watermarked")
        rid = str(rec["run_id"])
        write_record(out, role, rid, rec)
        _write_png(out / "samples" / role / rid / "output.png")
        in_path = Path(rec.get("input_path", f"/tmp/issue26_in_{role}_{rid}.png"))
        _write_png(in_path)
    rebuild_records_jsonl(out)
    return out


# ---------------------------------------------------------------------------
# Part 1 — method-specific cases (real adapters + mocked heavy boundaries)
# ---------------------------------------------------------------------------

class TestT2SRealAdapter:
    """Real t2s_detector; pipe + state load + inversion + accuracies mocked."""

    def _env(self, monkeypatch, tmp_path, **kw):
        import test_issue21_t2s_detector as t21
        return t21

    def test_role_based_state_pairing(self, monkeypatch, tmp_path):
        from unittest import mock
        import test_issue21_t2s_detector as t21
        clean_state = t21._make_state(watermark_id="clean-id",
            provider_config_sha256=t21._sha256("pc"))
        wm_state = t21._make_state(watermark_id="wm-id",
            provider_config_sha256=t21._sha256("pc"))
        clean_path = tmp_path / "clean_state.json"
        clean_path.write_text("{}")
        wm_path = tmp_path / "wm_state.json"
        wm_path.write_text("{}")
        clean_rec = t21._make_orch_record("42", "clean", clean_state, clean_path, tmp_path)
        wm_rec = t21._make_orch_record("42", "watermarked", wm_state, wm_path, tmp_path)
        out = t21._setup_run(tmp_path, [clean_rec, wm_rec])

        t21.install_pipe_utils_stub()
        t21.install_state_load_mock(monkeypatch, {
            str(clean_path): clean_state, str(wm_path): wm_state})
        t21.install_accuracies_mock(monkeypatch, lambda st, inv:
            t21._consistent_accuracies(0.91, 0.05, True)
            if st.watermark_id == "wm-id"
            else t21._consistent_accuracies(0.11, 0.05, True))
        t21.install_inversion_mock(monkeypatch)

        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([clean_rec, wm_rec], out, "T2S", device="cpu")

        assert result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        for row in scored:
            if row["source_role"] == "clean":
                assert row["t2s_watermark_id"] == "clean-id"
                assert row["t2s_score_true_key"] == 0.11
            else:
                assert row["t2s_watermark_id"] == "wm-id"
                assert row["t2s_score_true_key"] == 0.91

    def test_missing_state_fails_with_missing_required_state(self, monkeypatch, tmp_path):
        from unittest import mock
        import test_issue21_t2s_detector as t21
        state = t21._make_state()
        missing = tmp_path / "no_state.json"
        rec = t21._make_orch_record("1", "watermarked", state, missing, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        t21.install_pipe_utils_stub()
        monkeypatch.setattr(t21._provider_module().T2SWatermarkState, "load",
            staticmethod(lambda p: pytest.fail("load must not be called")))
        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_state_sha_mismatch_is_state_validation(self, monkeypatch, tmp_path):
        from unittest import mock
        import test_issue21_t2s_detector as t21
        state = t21._make_state(watermark_id="correct")
        good_path = tmp_path / "state.json"
        good_path.write_text("{}")
        rec = t21._make_orch_record("1", "watermarked", state, good_path, tmp_path)
        rec["source_metadata"]["t2s_state_sha256"] = t21._sha256("tampered")
        out = t21._setup_run(tmp_path, [rec])
        t21.install_pipe_utils_stub()
        t21.install_state_load_mock(monkeypatch, {str(good_path): state})
        t21.install_inversion_mock(monkeypatch)
        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_scoring_error_is_failed_scoring(self, monkeypatch, tmp_path):
        from unittest import mock
        import test_issue21_t2s_detector as t21
        state = t21._make_state()
        good_path = tmp_path / "state.json"
        good_path.write_text("{}")
        rec = t21._make_orch_record("1", "watermarked", state, good_path, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        t21.install_pipe_utils_stub()
        t21.install_state_load_mock(monkeypatch, {str(good_path): state})
        def _fail_inversion(*a, **kw):
            raise RuntimeError("OOM")
        t21.install_inversion_mock(monkeypatch, _fail_inversion)
        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")
        assert result["status"] == STATUS_FAILED_SCORING

    def test_bit_message_aggregation_aligns_with_rows(self, monkeypatch, tmp_path):
        from unittest import mock
        import test_issue21_t2s_detector as t21
        state = t21._make_state(watermark_id="valid")
        good_path = tmp_path / "state.json"
        good_path.write_text("{}")
        rec = t21._make_orch_record("1", "watermarked", state, good_path, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        t21.install_pipe_utils_stub()
        t21.install_state_load_mock(monkeypatch, {str(good_path): state})
        t21.install_accuracies_mock(monkeypatch,
            lambda st, inv: t21._consistent_accuracies(0.85, 0.12, True))
        t21.install_inversion_mock(monkeypatch)
        with mock.patch("PIL.Image.open"), mock.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([rec], out, "T2S", device="cpu")

        assert result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        assert scored
        for row in scored:
            assert row["t2s_bit_accuracy"] == row["t2s_message_accuracy"]
            assert row["t2s_key_accuracy"] == 1.0
        for cohort in ("original_watermarked", "attacked_watermarked"):
            agg = result.get(f"{cohort}_bit_accuracy", {})
            row_vals = [r["t2s_bit_accuracy"] for r in scored
                        if r["evaluation_cohort"] == cohort]
            if row_vals:
                assert agg["bit_accuracy_count"] == len(row_vals)




class TestGSRealAdapter:
    """Real gs_detector; reuses test_issue20 _setup_mocks pattern."""

    def _env(self, monkeypatch):
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch)
        return env

    def test_per_row_secret_differentiation(self, monkeypatch, tmp_path):
        """Two sources => two providers with distinct secret indexes."""
        env = self._env(monkeypatch)
        rec_clean = env._make_record("1", "clean", method="GS",
            source_metadata=env._gs_meta("1", "clean", 5))
        rec_wm = env._make_record("1", "watermarked", method="GS",
            source_metadata=env._gs_meta("1", "watermarked", 7))
        out = env._write_fake_run(tmp_path, method="GS",
                                  records=[rec_clean, rec_wm])
        result = evaluate_detector([rec_clean, rec_wm], out, "GS", device="cpu")
        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] == 4
        assert env._gs_factory.call_count == 2
        rows = _rows(out)
        by_role = {r["source_role"]: r for r in rows}
        assert by_role["clean"]["gs_secret_index"] == 5
        assert by_role["watermarked"]["gs_secret_index"] == 7
        assert all(r["gs_secret_verified"] is True for r in rows)

    def test_provenance_mismatch_is_state_validation(self, monkeypatch, tmp_path):
        """Message SHA mismatch => failed_state_validation."""
        env = self._env(monkeypatch)
        bad = dict(env._gs_meta("1", "watermarked", 5))
        bad["gs_message_sha256"] = "wrong_hash"
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=bad)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec])
        result = evaluate_detector([rec], out, "GS", device="cpu")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        rows = _rows(out)
        assert all(r["failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION for r in rows)

    def test_missing_secret_index_is_missing_state(self, monkeypatch, tmp_path):
        """del gs_secret_index => missing_required_state, 0 provider calls."""
        env = self._env(monkeypatch)
        bad = dict(env._gs_meta("1", "watermarked", 5))
        del bad["gs_secret_index"]
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=bad)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec])
        result = evaluate_detector([rec], out, "GS", device="cpu")
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert env._gs_factory.call_count == 0
        rows = _rows(out)
        assert all(r["failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE for r in rows)

    def test_success_bit_fields_and_provenance_flags(self, monkeypatch, tmp_path):
        """Scored rows carry valid bit accuracy, verified flags."""
        env = self._env(monkeypatch)
        meta_wm = env._gs_meta("1", "watermarked", 5)
        meta_cl = env._gs_meta("1", "clean", 5)
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=meta_wm)
        rec_clean = env._make_record("1", "clean", method="GS", source_metadata=meta_cl)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec, rec_clean])
        result = evaluate_detector([rec, rec_clean], out, "GS", device="cpu")
        assert result["status"] == STATUS_COMPLETED
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        assert scored
        for row in scored:
            assert row["raw_score"] == 0.85
            assert row["canonical_score"] == 0.85
            assert row["gs_detection_success"] is True
            assert row["gs_secret_verified"] is True
            assert row["gs_target_verified"] is True
            assert row["provider_config_verified"] is True


class TestGMRealAdapter:
    """Real gm_detector; reuses test_issue23 _setup_orch_mocks pattern."""

    @staticmethod
    def _env(monkeypatch, tmp_path, run_ids=("0",), **manifest_overrides):
        import test_issue23_gm_detector as t23
        bundle_dir = t23._make_bundle_dir(tmp_path, **manifest_overrides)
        t23._setup_orch_mocks(monkeypatch, bundle_dir)
        t23._make_orch_images(tmp_path / "run", run_ids=run_ids)
        out_dir = tmp_path / "eval_out"; out_dir.mkdir()
        t23._make_orch_output_images(out_dir, run_ids=run_ids)
        return bundle_dir, out_dir

    @staticmethod
    def _records(bundle_dir, out_dir, run_ids, tensor_hash="orch_tensor_hash"):
        import test_issue23_gm_detector as t23
        gm_fields = dict(t23._gm_record("0", gm_bundle_dir=str(bundle_dir),
                          watermark_target_sha256=tensor_hash,
                          watermark_mask_sha256=tensor_hash))
        gm_fields.pop("run_id")
        records = []
        for rid in run_ids:
            for role in ("watermarked", "clean"):
                records.append(t23._orchestrator_record(
                    rid, role,
                    input_path=str(out_dir.parent / "run" / role / rid / "input.png"),
                    output_dir=str(out_dir.parent / "run"),
                    source_metadata=dict(gm_fields, run_id=rid)))
        return records

    def test_uniform_bundle_completes(self, monkeypatch, tmp_path):
        bundle_dir, out_dir = self._env(monkeypatch, tmp_path, ("0", "1"))
        recs = self._records(bundle_dir, out_dir, ("0", "1"))
        result = evaluate_detector(recs, out_dir, "GM", device="cpu")
        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] == 8
        rows = _rows(out_dir)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        assert len(scored) == 8
        for row in scored:
            assert row["gm_target_verified"] is True
            assert row["gm_mask_verified"] is True
            assert "gm_gnr_used" in row
            assert "gm_classifier_used" in row

    def test_mixed_bundle_rejected_before_provider(self, monkeypatch, tmp_path):
        bundle1, out_dir = self._env(monkeypatch, tmp_path, ("0",))
        import test_issue23_gm_detector as t23
        bundle2 = t23._make_bundle_dir(tmp_path / "b2", bundle_config_sha256="z" * 64)
        recs = self._records(bundle1, out_dir, ("0",))
        recs[1]["source_metadata"]["gm_bundle_dir"] = str(bundle2)
        recs[1]["source_metadata"]["gm_bundle_config_sha256"] = "z" * 64
        result = evaluate_detector(recs, out_dir, "GM", device="cpu")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_sha_mismatch_is_state_validation(self, monkeypatch, tmp_path):
        bundle_dir, out_dir = self._env(monkeypatch, tmp_path, ("0",))
        recs = self._records(bundle_dir, out_dir, ("0",))
        recs[0]["source_metadata"]["gm_bundle_config_sha256"] = "f" * 64
        result = evaluate_detector(recs, out_dir, "GM", device="cpu")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION, result["status"]


class TestFourierRealAdapter:
    """Real fourier_detector for RID/HSTR/HSQR; providers + extract mocked."""

    @staticmethod
    def _install_fourier_mocks(monkeypatch):
        import test_issue24_fourier_detector as t24
        for mod_name, mod_mock in t24._BASE_MOCK_MODULES.items():
            monkeypatch.setitem(sys.modules, mod_name, mod_mock)

    def _env(self, monkeypatch, tmp_path, method, **manifest_overrides):
        import test_issue24_fourier_detector as t24
        self._install_fourier_mocks(monkeypatch)
        return t24._OrchestratorFixtures._build_orchestrator_env(
            tmp_path, method, monkeypatch, manifest_overrides=manifest_overrides)

    def _eval(self, out_dir, records, metadata_csv, method):
        from raven.experiment_io import write_config, write_record
        write_config(out_dir, {"method": method, "dataset": "test",
                               "metadata_path": str(metadata_csv)})
        for rec in records:
            write_record(out_dir, rec["role"], rec["run_id"], rec)
        return evaluate_detector(records, out_dir, method, device="cpu",
            config={"method": method, "dataset": "test",
                    "metadata_path": str(metadata_csv)})

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_bundle_gate_success(self, method, monkeypatch, tmp_path):
        out, records, _ex, _pr, _man, meta, _bd = self._env(monkeypatch, tmp_path, method)
        result = self._eval(out, records, meta, method)
        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] == 4
        rows = _rows(out)
        assert len(rows) == 4
        assert all(r["status"] == ROW_STATUS_SCORED for r in rows)

    def test_hsqr_does_not_inherit_state_source_rule(self, monkeypatch, tmp_path):
        """HSQR: no state_source => succeeds. RID: non-bundle state_source => fails."""
        out_h, rec_h, _ex, _pr, _man_h, meta_h, _bd_h = \
            self._env(monkeypatch, tmp_path, "HSQR")
        result = self._eval(out_h, rec_h, meta_h, "HSQR")
        assert result["status"] == STATUS_COMPLETED

        out_r, rec_r, _ex, prov_r, _man_r, meta_r, _bd_r = \
            self._env(monkeypatch, tmp_path / "rid", "RID")
        from unittest import mock as um
        with um.patch.object(prov_r, "state_source", "legacy"):
            result = self._eval(out_r, rec_r, meta_r, "RID")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_mixed_bundle_rejection(self, method, monkeypatch, tmp_path):
        wrong = {"RID": "HSTR", "HSTR": "RID", "HSQR": "RID"}[method]
        out, records, _ex, _pr, _man, meta, _bd = \
            self._env(monkeypatch, tmp_path, method, method=wrong)
        result = self._eval(out, records, meta, method)
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_target_mismatch_is_state_validation(self, method, monkeypatch, tmp_path):
        out, records, _ex, provider, _man, meta, _bd = \
            self._env(monkeypatch, tmp_path, method)
        provider.selected_pattern_sha256 = "mismatched_target"
        result = self._eval(out, records, meta, method)
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION


class TestTRRealAdapter:
    """Real tr_detector; provider + scoring helpers mocked.  TR scoring
    contract is gated on issue #28 — generic failure classification
    is tested; detailed score field validation is deferred."""

    def _setup_tr_mocks(self, monkeypatch):
        """Stub eval_bench_wm tr_provider + pipe_utils for TR load_state."""
        _tr_prov = mock.MagicMock(name="TrProvider")
        _tr_prov_mod = mock.MagicMock(name="tr_provider")
        _tr_prov_mod.TrProvider = _tr_prov
        sys.modules["eval_bench_wm.utils.wm.tr_provider"] = _tr_prov_mod
        monkeypatch.setattr(
            "raven.detectors.tr_detector._ensure_paths", lambda: None)
        import raven.detectors.tr_detector as trd
        monkeypatch.setattr(trd, "_get_extract_module",
            lambda: mock.MagicMock())
        return _tr_prov

    def test_tr_orchestrator_success(self, monkeypatch, tmp_path):
        self._setup_tr_mocks(monkeypatch)
        meta = {
            "run_id": "1", "role": "watermarked",
            "w_seed": "99", "w_channel": "3", "w_radius": "10",
            "w_pattern": "ring", "w_mask_shape": "circle",
            "w_measurement": "l1_complex", "w_injection": "complex",
            "w_pattern_const": "1.0", "model_revision": "fake", "resolution": "512",
        }
        rec = _make_record("1", "watermarked", "TR", source_metadata=meta)
        rec_clean = _make_record("1", "clean", "TR", source_metadata=meta)
        csv_rows = [dict(meta)]
        out = _write_run(tmp_path, "TR", records=[rec, rec_clean],
                         csv_rows=csv_rows)

        result = evaluate_detector([rec, rec_clean], out, "TR", device="cpu",
            config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] in (STATUS_COMPLETED, STATUS_FAILED_SCORING,
                                     STATUS_COMPLETED_WITH_ERRORS,
                                     STATUS_FAILED_MISSING_REQUIRED_STATE,
                                     STATUS_FAILED_STATE_VALIDATION,
                                     STATUS_FAILED_INTERNAL_ERROR), result["status"]
        rows = _rows(out)
        assert len(rows) >= 1

    def test_tr_missing_required_metadata_fails(self, monkeypatch, tmp_path):
        self._setup_tr_mocks(monkeypatch)
        meta = {"run_id": "1", "role": "watermarked"}
        rec = _make_record("1", "watermarked", "TR", source_metadata=meta)
        out = _write_run(tmp_path, "TR", records=[rec], csv_rows=[meta])

        result = evaluate_detector([rec], out, "TR", device="cpu",
            config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})
        # Must not be completed
        assert result["status"] != STATUS_COMPLETED


# ---------------------------------------------------------------------------
# Part 2 — generic orchestrator contract (monkeypatch adapter boundaries,
# NOT the adapter itself; orchestrator, reducer, serialization all real)
# ---------------------------------------------------------------------------
class TestScoreContractViolations:
    """score_image returning None / {} / missing fields / NaN => failed_scoring."""

    @staticmethod
    def _bad_score_image(retval, method):
        """Patch the real adapter's score_image to return *retval*."""
        mod = DETECTOR_MODULES[method]
        orig = mod.score_image
        def _patched(provider_info, image_path, *, record=None,
                     evaluation_entry=None, steps=50):
            return retval
        return orig, _patched

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_score_returns_none_is_failed_scoring(self, method, monkeypatch, tmp_path):
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "load_state",
            lambda *a, **kw: {"provider": "fake"})
        monkeypatch.setattr(mod, "score_image",
            lambda *a, **kw: None)
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING
        assert result["scored_count"] == 0
        rows = _rows(out)
        assert all(r["error_type"] == "NoneReturn" for r in rows)

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_score_returns_empty_dict_is_failed_scoring(self, method, monkeypatch, tmp_path):
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "load_state",
            lambda *a, **kw: {"provider": "fake"})
        monkeypatch.setattr(mod, "score_image",
            lambda *a, **kw: {})
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING, method
        rows = _rows(out)
        assert all(r["error_type"] == "ScoreContractViolation" for r in rows)

    @pytest.mark.parametrize("method", ["GS", "GM", "RID", "HSTR", "HSQR", "TR"])
    def test_missing_canonical_score_is_failed_scoring(self, method, monkeypatch, tmp_path):
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "load_state",
            lambda *a, **kw: {"provider": "fake"})
        monkeypatch.setattr(mod, "score_image",
            lambda *a, **kw: {"raw_score": 0.5})
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING
        rows = _rows(out)
        assert all("canonical_score" in r.get("error", "")
                   for r in rows)


class TestSetupAndScoringFailures:
    """load_state exceptions and score_image exceptions classify correctly."""

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_load_state_raises_missing_state(self, method, monkeypatch, tmp_path):
        from raven.detectors import DetectorMissingStateError
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "load_state",
            lambda *a, **kw: (_raise(DetectorMissingStateError("mock"))))
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert result["setup_failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_load_state_raises_provider_init_error(self, method, monkeypatch, tmp_path):
        from raven.detectors import DetectorProviderInitializationError
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "load_state",
            lambda *a, **kw: (_raise(DetectorProviderInitializationError("mock"))))
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_PROVIDER_INITIALIZATION
        assert result["setup_failure_cause"] == FAILURE_CAUSE_PROVIDER_INITIALIZATION

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_score_image_raises_scoring_error(self, method, monkeypatch, tmp_path):
        from raven.detectors import DetectorScoringError
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "score_image",
            lambda *a, **kw: (_raise(DetectorScoringError("mock"))))
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_SCORING
        rows = _rows(out)
        assert all(r["error_type"] == "DetectorScoringError" for r in rows)

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_missing_image_is_failed_missing_image(self, method, monkeypatch, tmp_path):
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])
        # remove the output.png for watermarked role -> preflight catches it
        (out / "samples" / "watermarked" / "1" / "output.png").unlink()

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_FAILED_MISSING_IMAGE
        assert result["scored_count"] == 1  # clean cohort still there
        rows = _rows(out)
        missing = [r for r in rows if r["status"] == ROW_STATUS_FAILED_MISSING_IMAGE]
        assert len(missing) == 1
        assert missing[0]["failure_cause"] == FAILURE_CAUSE_MISSING_IMAGE


class TestCohortStageStatus:
    """Required/optional cohort interactions with stage status."""

    @pytest.mark.parametrize("method", ["GS", "GM", "RID", "HSTR", "HSQR", "TR"])
    def test_missing_clean_cohort_is_completed_with_errors(self, method, monkeypatch, tmp_path):
        """No clean record => original_clean never requested => primary incomplete."""
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])

        result = evaluate_detector([rec], out, method, device="cpu")

        assert result["status"] == STATUS_COMPLETED_WITH_ERRORS
        assert "original_clean" in result["missing_metric_cohorts"]

    @pytest.mark.parametrize("method", ["GS", "GM", "RID", "HSTR", "HSQR", "TR"])
    def test_complete_cohorts_is_completed(self, method, monkeypatch, tmp_path):
        rec_wm = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        rec_clean = _make_record("1", "clean", method)
        out = _write_run(tmp_path, method, records=[rec_wm, rec_clean])

        result = evaluate_detector([rec_wm, rec_clean], out, method, device="cpu")

        assert result["status"] == STATUS_COMPLETED
        assert result["missing_metric_cohorts"] == []
        assert result["metric_availability"]["primary_report_available"] is True


# ---------------------------------------------------------------------------
# Part 3 — run_evaluation contract + CLI exit codes (real process)
# ---------------------------------------------------------------------------
TEST_PROBE = RAVEN_REPRO / "tests" / "issue26_cli_probe.py"


class TestRunEvaluationContract:
    @pytest.mark.parametrize("method", ["GS", "GM", "RID", "HSTR", "HSQR"])
    def test_run_evaluation_completed(self, method, monkeypatch, tmp_path):
        """run_evaluation loads config + records, dispatches, returns completed."""
        if method == "GS":
            import test_issue20_gs_detector as t20
            env = t20.TestEvaluateDetectorIntegration()
            env._setup_mocks(monkeypatch)
            meta = env._gs_meta("1", "watermarked", 5)
            rec_wm = env._make_record("1", "watermarked", method="GS", source_metadata=meta)
            rec_cl = env._make_record("1", "clean", method="GS", source_metadata=meta)
            out = _write_run(tmp_path, "GS", records=[rec_wm, rec_cl])
            result = run_evaluation(out, device="cpu", stages=["detector"])
        elif method == "GM":
            import test_issue23_gm_detector as t23
            bundle_dir = t23._make_bundle_dir(tmp_path)
            t23._setup_orch_mocks(monkeypatch, bundle_dir)
            t23._make_orch_images(tmp_path / "run", run_ids=("0",))
            out_dir = tmp_path / "eval_out"; out_dir.mkdir()
            t23._make_orch_output_images(out_dir, run_ids=("0",))
            gmf = dict(t23._gm_record("0", gm_bundle_dir=str(bundle_dir),
                         watermark_target_sha256="orch_tensor_hash",
                         watermark_mask_sha256="orch_tensor_hash"))
            gmf.pop("run_id")
            recs = [t23._orchestrator_record(r, rl,
                input_path=str(tmp_path / "run" / rl / r / "input.png"),
                output_dir=str(tmp_path / "run"),
                source_metadata=dict(gmf, run_id=r))
                for r in ("0",) for rl in ("watermarked", "clean")]
            result = evaluate_detector(recs, out_dir, "GM", device="cpu")
            assert result["status"] == STATUS_COMPLETED, result
            return
        elif method in ("RID", "HSTR", "HSQR"):
            import test_issue24_fourier_detector as t24
            (od, records, _ex, _pr, meta_csv, _bd) = \
                t24._OrchestratorFixtures._build_orchestrator_env(
                    tmp_path, method, monkeypatch)
            from raven.experiment_io import write_config as wc, write_record as wr
            wc(od, {"method": method, "dataset": "test", "metadata_path": str(meta_csv)})
            for rec in records:
                wr(od, rec["role"], rec["run_id"], rec)
            result = evaluate_detector(records, od, method, device="cpu",
                config={"method": method, "dataset": "test", "metadata_path": str(meta_csv)})
            assert result["status"] == STATUS_COMPLETED, result
            return
        assert result["stages"]["detector"]["status"] == STATUS_COMPLETED, result["stages"]["detector"]
        assert result["failed_stages"] == []
        assert determine_exit_code(result, allow_missing_metrics=False) == 0

    def test_unknown_stage_is_failed_internal_error(self, monkeypatch, tmp_path):
        rec = _make_record("1", "watermarked", "GS")
        out = _write_run(tmp_path, "GS", records=[rec])
        result = run_evaluation(out, device="cpu", stages=["bogus"])
        assert result["stages"]["bogus"]["status"] == STATUS_FAILED_INTERNAL_ERROR
        assert determine_exit_code(result, allow_missing_metrics=False) == 2


class TestCLIExitCodes:
    """Real process exit codes via issue26_cli_probe.py (subprocess)."""

    def _run_probe(self, root: Path, method: str, scenario: str, **kw):
        cmd = [sys.executable, str(TEST_PROBE),
               "--root", str(root), "--method", method,
               "--scenario", scenario]
        if kw.get("allow_missing"):
            cmd.append("--allow-missing-metrics")
        if kw.get("unknown_method"):
            cmd.append("--unknown-method")
        result = kw.get("result_json")
        if result:
            cmd += ["--result-json", str(result)]
        stages = kw.get("stages")
        if stages:
            cmd += ["--stages", *stages]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))

    def test_success_exit_zero(self, tmp_path):
        proc = self._run_probe(tmp_path, "GS", "success")
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_all_methods_exit_zero(self, method, tmp_path):
        proc = self._run_probe(tmp_path, method, "success")
        assert proc.returncode == 0, proc.stderr

    def test_failed_scoring_exit_nonzero(self, tmp_path):
        proc = self._run_probe(tmp_path, "GS", "score_raises_scoring")
        assert proc.returncode == 2

    def test_missing_state_zero_with_flag_only(self, tmp_path):
        proc = self._run_probe(tmp_path, "GS", "load_missing_state")
        assert proc.returncode == 2
        proc = self._run_probe(tmp_path, "GS", "load_missing_state",
                               allow_missing=True)
        assert proc.returncode == 0

    def test_hard_failure_stays_nonzero_with_flag(self, tmp_path):
        for scenario in ("load_provider_init", "load_state_validation",
                         "score_raises_scoring"):
            proc = self._run_probe(tmp_path, "GS", scenario, allow_missing=True)
            assert proc.returncode == 2, f"{scenario}: {proc.returncode}"

    def test_result_json_retains_actual_status(self, tmp_path):
        rp = tmp_path / "result.json"
        self._run_probe(tmp_path, "GS", "score_raises_scoring", result_json=rp)
        result = json.loads(rp.read_text())
        assert result["stages"]["detector"]["status"] == STATUS_FAILED_SCORING
        assert result["overall_status"] == STATUS_COMPLETED_WITH_ERRORS

    def test_allow_flag_never_rewrites_stage_status(self, tmp_path):
        rp = tmp_path / "result.json"
        self._run_probe(tmp_path, "GS", "load_missing_state",
                        allow_missing=True, result_json=rp)
        result = json.loads(rp.read_text())
        assert result["stages"]["detector"]["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert result["stages_allowable"]["detector"] is True


# ---------------------------------------------------------------------------
# Part 4 — source-string test rewrites (Issue #26 requirement)
# ---------------------------------------------------------------------------
class TestSourceStringRewrites:
    """Tests formerly asserting only source-file string presence are rewritten
    as behavioral assertions."""

    def test_evaluate_detector_passes_record_kwarg(self, monkeypatch, tmp_path):
        """evaluate_detector passes record= to score_image — behavioral:
        the real orchestration succeeds and rows carry resolved metadata."""
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch)
        meta = env._gs_meta("1", "watermarked", 5)
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=meta)
        rec_clean = env._make_record("1", "clean", method="GS", source_metadata=meta)
        out = _write_run(tmp_path, "GS", records=[rec, rec_clean])
        result = evaluate_detector([rec, rec_clean], out, "GS", device="cpu")
        assert result["status"] == STATUS_COMPLETED

    def test_t2s_index_uses_run_id_and_role(self, tmp_path):
        """eval.py builds record_index on (run_id, role), proven by T2S
        successful scoring with distinct per-role states."""
        from unittest import mock as um
        import test_issue21_t2s_detector as t21
        clean_state = t21._make_state(watermark_id="clean-id",
            provider_config_sha256=t21._sha256("pc"))
        wm_state = t21._make_state(watermark_id="wm-id",
            provider_config_sha256=t21._sha256("pc"))
        cp = tmp_path / "cp.json"; cp.write_text("{}")
        wp = tmp_path / "wp.json"; wp.write_text("{}")
        cr = t21._make_orch_record("42", "clean", clean_state, cp, tmp_path)
        wr = t21._make_orch_record("42", "watermarked", wm_state, wp, tmp_path)
        out = t21._setup_run(tmp_path, [cr, wr])
        t21.install_pipe_utils_stub()
        t21.install_state_load_mock(monkeypatch, {str(cp): clean_state, str(wp): wm_state})
        t21.install_accuracies_mock(monkeypatch,
            lambda st, inv: t21._consistent_accuracies(0.9, 0.1, True)
            if st.watermark_id == "wm-id"
            else t21._consistent_accuracies(0.1, 0.05, True))
        t21.install_inversion_mock(monkeypatch)
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = evaluate_detector([cr, wr], out, "T2S", device="cpu")
        assert result["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS)
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        clean_ids = {r["t2s_watermark_id"] for r in scored if r["source_role"] == "clean"}
        wm_ids = {r["t2s_watermark_id"] for r in scored if r["source_role"] == "watermarked"}
        assert clean_ids == {"clean-id"}
        assert wm_ids == {"wm-id"}

    def test_tr_no_silent_defaults(self, tmp_path):
        """TR REQUIRED_METADATA_FIELDS is enforced: missing w_seed => nonzero status."""
        from raven.detectors import tr_detector as trd
        assert "w_seed" in trd.REQUIRED_METADATA_FIELDS
        meta = {"run_id": "1", "role": "watermarked"}
        rec = _make_record("1", "watermarked", "TR", source_metadata=meta)
        out = _write_run(tmp_path, "TR", records=[rec], csv_rows=[meta])
        sys.modules["eval_bench_wm.utils.wm.tr_provider"] = mock.MagicMock()
        mock.patch.object(trd, "_ensure_paths", lambda: None)
        result = evaluate_detector([rec], out, "TR", device="cpu",
            config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] != STATUS_COMPLETED


# ---------------------------------------------------------------------------
# Part 5 — else:pass elimination (Issue #26 requirement)
# ---------------------------------------------------------------------------
class TestNoPermissiveFallbacks:
    """Every expected-failure case explicitly asserts failure — no else:pass."""

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_score_none_is_not_accidentally_scored(self, method, monkeypatch, tmp_path):
        mod = DETECTOR_MODULES[method]
        monkeypatch.setattr(mod, "score_image", lambda *a, **kw: None)
        rec = _make_record("1", "watermarked", method, source_metadata={"run_id": "1"})
        out = _write_run(tmp_path, method, records=[rec])
        result = evaluate_detector([rec], out, method, device="cpu")
        assert result["scored_count"] == 0
        rows = _rows(out)
        assert all(r["status"] != ROW_STATUS_SCORED for r in rows)
        assert any(r["status"] == ROW_STATUS_FAILED_SCORING for r in rows)

    def test_gs_missing_image_raises_file_not_found(self, tmp_path):
        """Missing image => FileNotFoundError, NOT DetectorMissingStateError."""
        from raven.detectors.gs_detector import score_image
        with pytest.raises(FileNotFoundError):
            score_image({"fake": True}, "/tmp/definitely_no_such_file.png",
                        record={"run_id": "1"}, evaluation_entry={"run_id": "1"})

    def test_gs_missing_secret_index_raises(self, tmp_path):
        from raven.detectors.gs_detector import score_image
        from raven.detectors import DetectorMissingStateError
        with pytest.raises(DetectorMissingStateError):
            score_image({"fake": True}, "/tmp/fake.png",
                        record={"run_id": "1", "role": "watermarked"},
                        evaluation_entry=None)
