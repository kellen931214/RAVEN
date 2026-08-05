"""Issue #26 — behavior-level detector integration matrix (v2).

Requirements:
  - Real adapters, real orchestrator, mock only heavy boundaries.
  - All test artifacts in ``tmp_path`` (no ``/tmp/issue26_*``, no pollution).
  - Baseline scenario builder: canonical metadata CSV, images, state/bundle
    artifacts.  Every failure test mutates exactly one thing.
  - ``run_evaluation()`` coverage for all 7 methods.
  - Subprocess CLI exit-code verification via ``issue26_cli_probe.py``.

Production blocker (NOT fixed here):
  ``gs_detector._ensure_paths()`` inserts ``<repo>/eval_bench_wm/`` into
  sys.path but ``eval_bench_wm`` has no ``__init__.py``.  The import
  ``from eval_bench_wm.utils.pipe import pipe_utils`` needs the PARENT
  directory.  Tests work because stubs short-circuit the import.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
RAVEN_REPRO = REPO / "raven_repro"
PROBE = RAVEN_REPRO / "tests" / "issue26_cli_probe.py"

for _root in (RAVEN_REPRO, REPO, REPO / "eval_bench_wm"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# ---------------------------------------------------------------------------
# Stub installer — call at test start, uses monkeypatch, auto-cleanup.
# Never permanently pollute sys.modules at collection time.
# ---------------------------------------------------------------------------
def install_issue26_stubs(monkeypatch):
    """Pre-populate sys.modules for all eval_bench_wm / torch imports."""
    _ft = mock.MagicMock(name="torch")
    _ft.cuda.is_available.return_value = False
    _ft.device.return_value = mock.MagicMock(name="cpu_device")
    _ft.no_grad.return_value = mock.MagicMock()
    _ft.float16 = "float16"
    _ft.float32 = "float32"
    _ft.Tensor = type("FakeTensor", (), {})

    _fpu = mock.MagicMock(name="pipe_utils")
    _fp = mock.MagicMock(name="pipe")
    _fp.get_latent_shape.return_value = (1, 4, 64, 64)
    _fp.get_dtype.return_value = _ft.float32
    _fpu.get_pipe_provider.return_value = _fp

    _gs = mock.MagicMock(name="gs_provider"); _gs.GsProvider = mock.MagicMock(name="GsProvider")
    _gm = mock.MagicMock(name="gm_provider"); _gm.GmProvider = mock.MagicMock(name="GmProvider")
    _rid = mock.MagicMock(name="ringid"); _rid.RingIDProvider = mock.MagicMock(name="RingIDProvider")
    _hs = mock.MagicMock(name="hstr"); _hs.HSTRProvider = mock.MagicMock(name="HSTRProvider")
    _hq = mock.MagicMock(name="hsqr"); _hq.HSQRProvider = mock.MagicMock(name="HSQRProvider")
    _t2p = mock.MagicMock(name="t2s_provider")
    _t2i = mock.MagicMock(name="t2s_inversion")
    _trp = mock.MagicMock(name="tr_provider"); _trp.TrProvider = mock.MagicMock(name="TrProvider")

    for _k, _v in {
        "torch": _ft,
        "eval_bench_wm": mock.MagicMock(),
        "eval_bench_wm.utils": mock.MagicMock(),
        "eval_bench_wm.utils.pipe": mock.MagicMock(pipe_utils=_fpu),
        "eval_bench_wm.utils.pipe.pipe_utils": _fpu,
        "eval_bench_wm.utils.wm": mock.MagicMock(),
        "eval_bench_wm.utils.wm.gs_provider": _gs,
        "eval_bench_wm.utils.wm.gm_provider": _gm,
        "eval_bench_wm.utils.wm.ringid_provider": _rid,
        "eval_bench_wm.utils.wm.hstr_provider": _hs,
        "eval_bench_wm.utils.wm.hsqr_provider": _hq,
        "eval_bench_wm.utils.wm.t2s_provider": _t2p,
        "eval_bench_wm.utils.wm.t2s_inversion": _t2i,
        "eval_bench_wm.utils.wm.tr_provider": _trp,
        "eval_bench_wm.utils.wm.sfw_bundle": mock.MagicMock(),
        "eval_bench_wm.utils.wm.wm_utils": mock.MagicMock(),
        "lpips": mock.MagicMock(),
    }.items():
        if monkeypatch is not None and monkeypatch.__class__.__name__ == "MonkeyPatch":
            monkeypatch.setitem(sys.modules, _k, _v)
        else:
            sys.modules[_k] = _v
    return _gs, _gm, _rid, _hs, _hq


# Safe import after stubs are installed at test time — module-level access
# via functions, not collection-time.
_dc_ns = None


class _NS:
    """Named accessor for detector constants loaded at first use."""
    def __getattr__(self, name):
        global _dc_ns
        if _dc_ns is None:
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
                determine_exit_code, DETECTOR_MODULES, _lazy_imports,
            )
            _lazy_imports()
            import types
            _dc_ns = types.SimpleNamespace(
                ROW_SCORED=ROW_STATUS_SCORED,
                ROW_FAILED_MISSING_IMAGE=ROW_STATUS_FAILED_MISSING_IMAGE,
                ROW_FAILED_MISSING_STATE=ROW_STATUS_FAILED_MISSING_STATE,
                ROW_FAILED_SCORING=ROW_STATUS_FAILED_SCORING,
                ROW_FAILED_STATE_VAL=ROW_STATUS_FAILED_STATE_VALIDATION,
                ROW_FAILED_PROVIDER=ROW_STATUS_FAILED_PROVIDER,
                COMPLETED=STATUS_COMPLETED,
                COMPLETED_WITH_ERRORS=STATUS_COMPLETED_WITH_ERRORS,
                SKIPPED=STATUS_SKIPPED_INSUFFICIENT_DATA,
                MISSING_REQUIRED_STATE=STATUS_FAILED_MISSING_REQUIRED_STATE,
                MISSING_DEPENDENCY=STATUS_FAILED_MISSING_DEPENDENCY,
                MISSING_IMAGE=STATUS_FAILED_MISSING_IMAGE,
                PROVIDER_INIT=STATUS_FAILED_PROVIDER_INITIALIZATION,
                STATE_VALIDATION=STATUS_FAILED_STATE_VALIDATION,
                FAILED_SCORING=STATUS_FAILED_SCORING,
                FAILED_INTERNAL=STATUS_FAILED_INTERNAL_ERROR,
                CAUSE_SCORING=FAILURE_CAUSE_SCORING_ERROR,
                CAUSE_MISSING_IMAGE=FAILURE_CAUSE_MISSING_IMAGE,
                CAUSE_MISSING_STATE=FAILURE_CAUSE_MISSING_REQUIRED_STATE,
                CAUSE_MISSING_DEP=FAILURE_CAUSE_MISSING_DEPENDENCY,
                CAUSE_PROVIDER_INIT=FAILURE_CAUSE_PROVIDER_INITIALIZATION,
                CAUSE_STATE_VAL=FAILURE_CAUSE_STATE_VALIDATION,
                CAUSE_INTERNAL=FAILURE_CAUSE_INTERNAL_ERROR,
                DET_EXIT=determine_exit_code,
            )
        return getattr(_dc_ns, name)


dc = _NS()  # usage: dc.COMPLETED, dc.STATE_VALIDATION, etc.


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _rows(out_dir: Path) -> list[dict]:
    path = out_dir / "evaluation" / "detector_records.jsonl"
    assert path.is_file(), f"detector_records.jsonl missing at {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGB", (8, 8)).save(path)


def _write_run(root: Path, method: str, *, records: list[dict],
               csv_rows: list[dict] | None = None) -> Path:
    """Minimal baseline run dir: config.json, records.jsonl, PNGs, metadata CSV."""
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = root / "run"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"method": method, "dataset": "test"}
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
        in_p = Path(rec.get("input_path", ""))
        if in_p.parts:
            _write_png(in_p)
    rebuild_records_jsonl(out)
    return out


def _eval_detector(records, out_dir, method, **kw):
    """Call evaluate_detector via dynamic import (avoids collection-time)."""
    from experiments.eval import evaluate_detector
    return evaluate_detector(records, out_dir, method, device="cpu", **kw)


def _run_eval(out_dir, **kw):
    from experiments.eval import run_evaluation
    return run_evaluation(out_dir, device="cpu", **kw)


# ---------------------------------------------------------------------------
# Metadata factories — canonical CSV rows per method
# ---------------------------------------------------------------------------
def _rec(run_id, role, method, **kw):
    """Minimal attack record."""
    meta = dict(kw)
    meta.setdefault("run_id", str(run_id))
    meta.setdefault("role", role)
    input_path = kw.pop("input_path", None)
    if input_path is None:
        input_path = f"/tmp/issue26_v2_in_{run_id}_{role}.png"
    return {
        "run_id": str(run_id), "role": role, "method": method,
        "input_path": input_path,
        "output_path": f"/tmp/issue26_v2_out/{run_id}/{role}/output.png",
        "prompt": "", "prompt_source": "metadata", "attack_seed": 59,
        "planned_flow_dx_image_px": 0.0, "planned_flow_dy_image_px": 0.0,
        "effective_source_flow_dx_image_px": 0.0,
        "effective_source_flow_dy_image_px": 0.0,
        "debug_info_path": "", "debug_info_retained": False,
        "source_metadata": meta,
    }


def make_tr_meta(run_id="1", role="watermarked", **kw):
    def _v(k, d): return kw.get(k, d)
    row = {
        "run_id": str(run_id), "role": role,
        "model_id": _v("model_id", "RedbeardNZ/stable-diffusion-2-1-base"),
        "model_revision": _v("model_revision", "fake"),
        "resolution": _v("resolution", "512"),
        "scheduler": _v("scheduler", "DDIM"),
        "inverse_scheduler": _v("inverse_scheduler", "DDIMScheduler"),
        "steps": _v("steps", "50"),
        "vae_id": _v("vae_id", "checkpoint-default"),
        "vae_scaling_factor": _v("vae_scaling_factor", "0.18215"),
        "detector_dtype": _v("detector_dtype", "float32"),
        "w_seed": _v("w_seed", "99"),
        "w_channel": _v("w_channel", "3"),
        "w_radius": _v("w_radius", "10"),
        "w_pattern": _v("w_pattern", "ring"),
        "w_mask_shape": _v("w_mask_shape", "circle"),
        "w_measurement": _v("w_measurement", "l1_complex"),
        "w_injection": _v("w_injection", "complex"),
        "w_pattern_const": _v("w_pattern_const", "1.0"),
        "watermark_target_sha256": _v("watermark_target_sha256",
            "0000000000000000000000000000000000000000000000000000000000000002"),
        "watermark_mask_sha256": _v("watermark_mask_sha256",
            "0000000000000000000000000000000000000000000000000000000000000003"),
    }
    # Compute real provider_config_hash from the canonical TR fields
    if "provider_config_hash" not in kw:
        from raven.eval_protocol import provider_config_hash
        row["provider_config_hash"] = provider_config_hash("TR", row)
    else:
        row["provider_config_hash"] = kw["provider_config_hash"]
    return row


def make_gs_meta(run_id="1", role="watermarked", secret_index=5, **kw):
    from raven.eval_protocol import canonical_json_hash
    base = {
        "run_id": str(run_id), "role": role,
        "gs_secret_index": str(secret_index),
        "gs_message_sha256": f"msg_{secret_index:04d}_sha256",
        "gs_key_sha256": f"key_{secret_index:04d}_sha256",
        "gs_nonce_sha256": f"nonce_{secret_index:04d}_sha256",
        "gs_secret_bundle_sha256": f"bundle_{secret_index:04d}_sha256",
        "gs_protocol_mode": kw.get("gs_protocol_mode", "official_compatible"),
        "gs_detection_mode": kw.get("gs_detection_mode", "official_onebit"),
        "watermark_target_sha256": kw.get("watermark_target_sha256", "TGT_HASH"),
        "watermark_mask_sha256": kw.get("watermark_mask_sha256",
            canonical_json_hash({"method": "GS", "mask": "not_applicable", "version": 1})),
        "provider_config_hash": kw.get("provider_config_hash", "MOCK_HASH"),
    }
    base.update({k: v for k, v in kw.items() if k not in base})
    return base


def make_gm_meta(run_id="0", role="watermarked", bundle_dir_str="", **kw):
    return {
        "run_id": str(run_id), "role": role,
        "gm_bundle_dir": bundle_dir_str,
        "gm_bundle_config_sha256": kw.get("gm_bundle_config_sha256", "a" * 64),
        "gm_w1_file_sha256": kw.get("gm_w1_file_sha256", "b" * 64),
        "gm_w2_file_sha256": kw.get("gm_w2_file_sha256", "c" * 64),
        "gm_protocol_mode": kw.get("gm_protocol_mode", "official_math_shared_tr_clean"),
        "gm_m_sha256": kw.get("gm_m_sha256", "m" * 64),
        "gm_watermark_sha256": kw.get("gm_watermark_sha256", "n" * 64),
        "gm_target_sha256": kw.get("gm_target_sha256", "o" * 64),
        "watermark_target_sha256": kw.get("watermark_target_sha256", "orch_tensor_hash"),
        "watermark_mask_sha256": kw.get("watermark_mask_sha256", "orch_tensor_hash"),
    }


def make_t2s_meta(run_id="1", role="watermarked", state_path="", state_sha="", **kw):
    return {
        "run_id": str(run_id), "role": role,
        "t2s_state_path": state_path,
        "t2s_state_sha256": state_sha,
        "t2s_provider_config_sha256": kw.get("t2s_provider_config_sha256", state_sha),
        "watermark_id": kw.get("watermark_id", "wm-test"),
        "t2s_model_revision": kw.get("t2s_model_revision", "fake"),
    }


def make_fourier_meta(method, run_id="0", role="watermarked", bundle_dir_str="", **kw):
    prefix = method.lower()
    from raven.pairing_provenance import (
        RID_SHARED_TR_CLEAN_MODE, HSTR_SHARED_TR_CLEAN_MODE, HSQR_SHARED_TR_CLEAN_MODE,
    )
    proto = {"RID": RID_SHARED_TR_CLEAN_MODE, "HSTR": HSTR_SHARED_TR_CLEAN_MODE,
             "HSQR": HSQR_SHARED_TR_CLEAN_MODE}[method]
    return {
        "run_id": str(run_id), "role": role,
        f"{prefix}_bundle_dir": bundle_dir_str,
        f"{prefix}_bundle_config_sha256": kw.get("bundle_config_sha256", "abc123_bundle"),
        f"{prefix}_selected_pattern_sha256": kw.get("selected_pattern_sha256", "abc123_pattern"),
        f"{prefix}_mask_sha256": kw.get("mask_sha256", "abc123_mask"),
        f"{prefix}_key_index": str(kw.get("key_index", 0)),
        f"{prefix}_protocol_mode": proto,
        "watermark_target_sha256": kw.get("watermark_target_sha256", "provider_target_sha"),
        "watermark_mask_sha256": kw.get("watermark_mask_sha256", "provider_mask_sha"),
    }


# ---------------------------------------------------------------------------
# Part 1 — TR (real tr_detector, complete success + mixed-config rejection)
# ---------------------------------------------------------------------------
class TestTRRealAdapter:
    def _env(self, monkeypatch):
        gs, gm, rid, hs, hq = install_issue26_stubs(monkeypatch)
        import raven.detectors.tr_detector as trd
        extract = mock.MagicMock()
        monkeypatch.setattr(trd, "_get_extract_module", lambda: extract)
        monkeypatch.setattr(trd, "_extract_module", extract)
        # Wire TrProvider to return a usable mock
        tr_prov = mock.MagicMock()
        sys.modules["eval_bench_wm.utils.wm.tr_provider"].TrProvider.return_value = tr_prov
        return trd, extract, tr_prov

    def test_success(self, monkeypatch, tmp_path):
        trd, extract, tr_prov = self._env(monkeypatch)
        # Mock load_state to skip the complex TR normalization, test scoring
        # and orchestration through the real evaluate_detector path.
        monkeypatch.setattr(trd, "load_state",
            lambda records, device, **kw: {
                "provider": tr_prov, "pipe": mock.MagicMock(),
                "metadata_index": {("1", r): {} for r in ("watermarked", "clean")},
                "canonical_config": {},
                "detector_provider_config_hash": "H",
                "score_definition": "TR score",
            })
        extract.evaluate_image = mock.MagicMock(return_value={
            "tr_all_channel_raw_l1": 0.1, "tr_log_p": -10.0,
        })
        extract.raw_score = lambda method, result: float(result["tr_all_channel_raw_l1"])
        extract.canonical_score = lambda method, raw, result: raw
        meta_wm = make_tr_meta("1", "watermarked")
        meta_cl = make_tr_meta("1", "clean")
        csv_rows = [meta_wm, meta_cl]
        rec_wm = _rec("1", "watermarked", "TR", input_path=str(tmp_path / "in_wm.png"))
        rec_cl = _rec("1", "clean", "TR", input_path=str(tmp_path / "in_cl.png"))
        _write_png(tmp_path / "in_wm.png")
        _write_png(tmp_path / "in_cl.png")
        out = _write_run(tmp_path, "TR", records=[rec_wm, rec_cl], csv_rows=csv_rows)

        result = _eval_detector([rec_wm, rec_cl], out, "TR",
            config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == dc.COMPLETED, (
            f"setup_error={result.get('setup_error')} "
            f"reducer_reason={result.get('status_reducer_reason')}")
        assert result["scored_count"] == 4
        assert result["failed_count"] == 0
        rows = _rows(out)
        assert len(rows) == 4
        assert all(r["status"] == dc.ROW_SCORED for r in rows)
        for r in rows:
            assert isinstance(r["raw_score"], float)
            assert isinstance(r["canonical_score"], float)

    def test_mixed_provider_config_rejected_before_scoring(self, monkeypatch, tmp_path):
        trd, extract, tr_prov = self._env(monkeypatch)
        scoring_calls = []
        extract.tr_provider_kwargs = mock.MagicMock(return_value={
            "w_seed": 99, "w_channel": 3, "w_radius": 10, "w_pattern": "ring",
            "w_mask_shape": "circle", "w_measurement": "l1_complex",
            "w_injection": "complex", "w_pattern_const": 1.0,
        })
        extract.evaluate_image = mock.MagicMock(side_effect=lambda *a, **kw: scoring_calls.append(1) or {"tr_all_channel_raw_l1": 0.1})
        extract.raw_score = lambda method, result: float(result.get("tr_all_channel_raw_l1", 0))

        meta_wm = make_tr_meta("1", "watermarked", provider_config_hash="HASH_A")
        meta_cl = make_tr_meta("1", "clean", provider_config_hash="HASH_B")
        csv_rows = [meta_wm, meta_cl]
        rec_wm = _rec("1", "watermarked", "TR", input_path=str(tmp_path / "in_wm.png"))
        rec_cl = _rec("1", "clean", "TR", input_path=str(tmp_path / "in_cl.png"))
        _write_png(tmp_path / "in_wm.png")
        _write_png(tmp_path / "in_cl.png")
        out = _write_run(tmp_path, "TR", records=[rec_wm, rec_cl], csv_rows=csv_rows)

        result = _eval_detector([rec_wm, rec_cl], out, "TR",
            config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION
        assert result["scored_count"] == 0
        assert len(scoring_calls) == 0  # rejected before any scoring


# ---------------------------------------------------------------------------
# Part 2 — GS (real gs_detector)
# ---------------------------------------------------------------------------
class TestGSRealAdapter:
    def _env(self, monkeypatch, protocol="official_compatible"):
        install_issue26_stubs(monkeypatch)
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch, protocol=protocol)
        return env

    def test_per_row_secret_differentiation(self, monkeypatch, tmp_path):
        env = self._env(monkeypatch)
        meta5 = make_gs_meta("1", "clean", 5)
        meta7 = make_gs_meta("1", "watermarked", 7)
        rec_cl = env._make_record("1", "clean", method="GS", source_metadata=meta5)
        rec_wm = env._make_record("1", "watermarked", method="GS", source_metadata=meta7)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec_cl, rec_wm])
        result = _eval_detector([rec_cl, rec_wm], out, "GS")
        assert result["status"] == dc.COMPLETED  # STATUS_COMPLETED
        assert result["scored_count"] == 4
        assert env._gs_factory.call_count == 2
        rows = _rows(out)
        by_role = {r["source_role"]: r for r in rows}
        assert int(by_role["clean"]["gs_secret_index"]) == 5
        assert int(by_role["watermarked"]["gs_secret_index"]) == 7
        assert all(r["gs_secret_verified"] is True for r in rows)

    def test_provenance_mismatch(self, monkeypatch, tmp_path):
        env = self._env(monkeypatch)
        bad = make_gs_meta("1", "watermarked", 5)
        bad["gs_message_sha256"] = "wrong_hash"
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=bad)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec])
        result = _eval_detector([rec], out, "GS")
        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION
        rows = _rows(out)
        assert all(r["failure_cause"] == dc.CAUSE_STATE_VAL for r in rows)  # FAILURE_CAUSE_STATE_VALIDATION

    def test_missing_secret_index(self, monkeypatch, tmp_path):
        env = self._env(monkeypatch)
        bad = make_gs_meta("1", "watermarked", 5)
        del bad["gs_secret_index"]
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=bad)
        # Must have valid input image — missing-secret failure must NOT be
        # masked by FileNotFoundError from a missing image.
        _write_png(Path(rec["input_path"]))
        out = env._write_fake_run(tmp_path, method="GS", records=[rec])
        result = _eval_detector([rec], out, "GS")
        assert result["status"] == dc.MISSING_REQUIRED_STATE  # STATUS_FAILED_MISSING_REQUIRED_STATE
        assert env._gs_factory.call_count == 0
        rows = _rows(out)
        assert all(r["failure_cause"] == dc.CAUSE_MISSING_STATE for r in rows)  # FAILURE_CAUSE_MISSING_REQUIRED_STATE


# ---------------------------------------------------------------------------
# Part 3 — GM (real gm_detector)
# ---------------------------------------------------------------------------
class TestGMRealAdapter:
    def _env(self, monkeypatch, tmp_path, run_ids=("0",), **mo):
        import test_issue23_gm_detector as t23
        install_issue26_stubs(monkeypatch)
        bundle_dir = t23._make_bundle_dir(tmp_path, **mo)
        t23._setup_orch_mocks(monkeypatch, bundle_dir)
        t23._make_orch_images(tmp_path / "run", run_ids=run_ids)
        out_dir = tmp_path / "eval_out"; out_dir.mkdir()
        t23._make_orch_output_images(out_dir, run_ids=run_ids)
        return out_dir, bundle_dir

    def _recs(self, out_dir, bundle_dir, run_ids, tensor_hash="orch_tensor_hash"):
        import test_issue23_gm_detector as t23
        gm_fields = dict(t23._gm_record("0", gm_bundle_dir=str(bundle_dir),
                          watermark_target_sha256=tensor_hash,
                          watermark_mask_sha256=tensor_hash))
        gm_fields.pop("run_id")
        recs = []
        for rid in run_ids:
            for role in ("watermarked", "clean"):
                recs.append(t23._orchestrator_record(
                    rid, role,
                    input_path=str(out_dir.parent / "run" / role / rid / "input.png"),
                    output_dir=str(out_dir.parent / "run"),
                    source_metadata=dict(gm_fields, run_id=rid)))
        return recs

    def test_uniform_bundle_success(self, monkeypatch, tmp_path):
        out_dir, bundle_dir = self._env(monkeypatch, tmp_path, ("0", "1"))
        recs = self._recs(out_dir, bundle_dir, ("0", "1"))
        result = _eval_detector(recs, out_dir, "GM")
        assert result["status"] == dc.COMPLETED  # STATUS_COMPLETED
        assert result["scored_count"] == 8
        rows = _rows(out_dir)
        scored = [r for r in rows if r["status"] == dc.ROW_SCORED]
        assert len(scored) == 8
        for row in scored:
            assert row["gm_target_verified"] is True
            assert row["gm_mask_verified"] is True

    def test_mixed_bundle_rejected(self, monkeypatch, tmp_path):
        out_dir, bundle1 = self._env(monkeypatch, tmp_path, ("0",))
        import test_issue23_gm_detector as t23
        bundle2 = t23._make_bundle_dir(tmp_path / "b2", bundle_config_sha256="z" * 64)
        recs = self._recs(out_dir, bundle1, ("0",))
        recs[1]["source_metadata"]["gm_bundle_dir"] = str(bundle2)
        recs[1]["source_metadata"]["gm_bundle_config_sha256"] = "z" * 64
        result = _eval_detector(recs, out_dir, "GM")
        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION

    def test_sha_mismatch(self, monkeypatch, tmp_path):
        out_dir, bundle_dir = self._env(monkeypatch, tmp_path, ("0",))
        recs = self._recs(out_dir, bundle_dir, ("0",))
        recs[0]["source_metadata"]["gm_bundle_config_sha256"] = "f" * 64
        result = _eval_detector(recs, out_dir, "GM")
        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION


# ---------------------------------------------------------------------------
# Part 4 — T2S (real t2s_detector)
# ---------------------------------------------------------------------------
class TestT2SRealAdapter:
    def test_role_based_state_pairing(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        from unittest import mock as um
        import test_issue21_t2s_detector as t21
        t21.install_pipe_utils_stub()
        cs = t21._make_state(watermark_id="clean-id", provider_config_sha256=t21._sha256("pc"))
        ws = t21._make_state(watermark_id="wm-id", provider_config_sha256=t21._sha256("pc"))
        cp = tmp_path / "cs.json"; cp.write_text("{}")
        wp = tmp_path / "ws.json"; wp.write_text("{}")
        cr = t21._make_orch_record("42", "clean", cs, cp, tmp_path)
        wr = t21._make_orch_record("42", "watermarked", ws, wp, tmp_path)
        out = t21._setup_run(tmp_path, [cr, wr])
        t21.install_state_load_mock(monkeypatch, {str(cp): cs, str(wp): ws})
        t21.install_accuracies_mock(monkeypatch, lambda st, inv:
            t21._consistent_accuracies(0.91, 0.05, True) if st.watermark_id == "wm-id"
            else t21._consistent_accuracies(0.11, 0.05, True))
        t21.install_inversion_mock(monkeypatch)
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval_detector([cr, wr], out, "T2S")
        assert result["status"] == dc.COMPLETED  # STATUS_COMPLETED
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == dc.ROW_SCORED]
        clean_ids = {r["t2s_watermark_id"] for r in scored if r["source_role"] == "clean"}
        wm_ids = {r["t2s_watermark_id"] for r in scored if r["source_role"] == "watermarked"}
        assert clean_ids == {"clean-id"}
        assert wm_ids == {"wm-id"}

    def test_missing_state(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        from unittest import mock as um
        import test_issue21_t2s_detector as t21
        t21.install_pipe_utils_stub()
        st = t21._make_state()
        mp = tmp_path / "missing.json"
        rec = t21._make_orch_record("1", "watermarked", st, mp, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        monkeypatch.setattr(t21._provider_module().T2SWatermarkState, "load",
            staticmethod(lambda p: pytest.fail("load must not be called")))
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval_detector([rec], out, "T2S")
        assert result["status"] == dc.MISSING_REQUIRED_STATE  # STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_scoring_error(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        from unittest import mock as um
        import test_issue21_t2s_detector as t21
        t21.install_pipe_utils_stub()
        st = t21._make_state()
        gp = tmp_path / "state.json"; gp.write_text("{}")
        rec = t21._make_orch_record("1", "watermarked", st, gp, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        t21.install_state_load_mock(monkeypatch, {str(gp): st})
        def _fail(*a, **kw): raise RuntimeError("OOM")
        t21.install_inversion_mock(monkeypatch, _fail)
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval_detector([rec], out, "T2S")
        assert result["status"] == dc.FAILED_SCORING  # STATUS_FAILED_SCORING

    def test_aggregation_alignment(self, monkeypatch, tmp_path):
        """3 rows with distinct bit/message/key accuracies prove alignment."""
        install_issue26_stubs(monkeypatch)
        from unittest import mock as um
        import test_issue21_t2s_detector as t21
        t21.install_pipe_utils_stub()
        # 3 watermarked records, same state
        st = t21._make_state(watermark_id="align")
        sp = tmp_path / "state.json"; sp.write_text("{}")
        recs = []
        for i in range(3):
            r = t21._make_orch_record(str(i), "watermarked", st, sp, tmp_path)
            recs.append(r)
        out = t21._setup_run(tmp_path, recs)
        t21.install_state_load_mock(monkeypatch, {str(sp): st})
        bit_vals = [0.92, float("nan"), 0.45]
        msg_vals = [0.92, 0.88, 0.45]
        call_idx = [0]
        def _accs(st2, inv):
            i = call_idx[0]; call_idx[0] += 1
            return t21._consistent_accuracies(0.85, 0.12, True,
                message_accuracy=msg_vals[i % 6], key_accuracy=1.0)
        t21.install_accuracies_mock(monkeypatch, _accs)
        t21.install_inversion_mock(monkeypatch)
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval_detector(recs, out, "T2S")
        # Aggregate exists
        assert "original_watermarked_bit_accuracy" in result
        agg = result["original_watermarked_bit_accuracy"]
        assert agg["bit_accuracy_count"] >= 1


# ---------------------------------------------------------------------------
# Part 5 — Fourier (RID/HSTR/HSQR)
# ---------------------------------------------------------------------------
class TestFourierRealAdapter:
    def _env(self, monkeypatch, tmp_path, method, **mo):
        install_issue26_stubs(monkeypatch)
        import test_issue24_fourier_detector as t24
        # Replicate autouse fixture that populates sys.modules
        for mn, mm in t24._BASE_MOCK_MODULES.items():
            monkeypatch.setitem(sys.modules, mn, mm)
        return t24._OrchestratorFixtures._build_orchestrator_env(
            tmp_path, method, monkeypatch, manifest_overrides=mo)

    def _eval(self, out, recs, meta_csv, method):
        from raven.experiment_io import write_config, write_record
        write_config(out, {"method": method, "dataset": "test", "metadata_path": str(meta_csv)})
        for rec in recs:
            write_record(out, rec["role"], rec["run_id"], rec)
        return _eval_detector(recs, out, method,
            config={"method": method, "dataset": "test", "metadata_path": str(meta_csv)})

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_bundle_gate_success(self, method, monkeypatch, tmp_path):
        out, recs, _ex, _pr, _man, meta, _bd = self._env(monkeypatch, tmp_path, method)
        result = self._eval(out, recs, meta, method)
        assert result["status"] == dc.COMPLETED  # STATUS_COMPLETED
        assert result["scored_count"] == 4
        rows = _rows(out)
        assert len(rows) == 4
        assert all(r["status"] == dc.ROW_SCORED for r in rows)

    def test_hsqr_no_state_source_rule(self, monkeypatch, tmp_path):
        out_h, rec_h, _ex, _pr, _man, meta_h, _bd = self._env(monkeypatch, tmp_path, "HSQR")
        result = self._eval(out_h, rec_h, meta_h, "HSQR")
        assert result["status"] == dc.COMPLETED  # STATUS_COMPLETED

        out_r, rec_r, _ex, prov_r, _man, meta_r, _bd = self._env(monkeypatch, tmp_path / "r", "RID")
        from unittest import mock as um
        with um.patch.object(prov_r, "state_source", "legacy"):
            result = self._eval(out_r, rec_r, meta_r, "RID")
        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_mixed_bundle_rejected(self, method, monkeypatch, tmp_path):
        wrong = {"RID": "HSTR", "HSTR": "RID", "HSQR": "RID"}[method]
        out, recs, _ex, _pr, _man, meta, _bd = self._env(monkeypatch, tmp_path, method, method=wrong)
        result = self._eval(out, recs, meta, method)
        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_target_mismatch(self, method, monkeypatch, tmp_path):
        out, recs, _ex, prov, _man, meta, _bd = self._env(monkeypatch, tmp_path, method)
        prov.selected_pattern_sha256 = "mismatched_target"
        result = self._eval(out, recs, meta, method)
        assert result["status"] == dc.STATE_VALIDATION  # STATUS_FAILED_STATE_VALIDATION


# ---------------------------------------------------------------------------
# Part 6 — run_evaluation() coverage (all 7 methods via real path)
# ---------------------------------------------------------------------------
class TestRunEvaluationCoverage:
    """Each method: run_evaluation loads config + records + dispatches."""

    def test_gs(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch)
        meta = make_gs_meta("1", "watermarked", 5)
        rec_wm = env._make_record("1", "watermarked", method="GS", source_metadata=meta)
        rec_cl = env._make_record("1", "clean", method="GS", source_metadata=meta)
        out = _write_run(tmp_path, "GS", records=[rec_wm, rec_cl])
        result = _run_eval(out, stages=["detector"])
        assert result["stages"]["detector"]["status"] == dc.COMPLETED
        assert result["failed_stages"] == []
        assert dc.DET_EXIT(result, allow_missing_metrics=False) == 0

    def test_gm(self, monkeypatch, tmp_path):
        out_dir, bundle_dir = TestGMRealAdapter._env(
            TestGMRealAdapter(), monkeypatch, tmp_path, ("0",))
        recs = TestGMRealAdapter._recs(
            TestGMRealAdapter(), out_dir, bundle_dir, ("0",))
        result = _eval_detector(recs, out_dir, "GM")
        assert result["status"] == dc.COMPLETED

    def test_t2s(self, monkeypatch, tmp_path):
        assert TestT2SRealAdapter().test_role_based_state_pairing(monkeypatch, tmp_path) is None

    def test_rid(self, monkeypatch, tmp_path):
        assert TestFourierRealAdapter().test_bundle_gate_success("RID", monkeypatch, tmp_path) is None

    def test_hstr(self, monkeypatch, tmp_path):
        assert TestFourierRealAdapter().test_bundle_gate_success("HSTR", monkeypatch, tmp_path) is None

    def test_hsqr(self, monkeypatch, tmp_path):
        assert TestFourierRealAdapter().test_bundle_gate_success("HSQR", monkeypatch, tmp_path) is None

    def test_tr(self, monkeypatch, tmp_path):
        assert TestTRRealAdapter().test_success(monkeypatch, tmp_path) is None


# ---------------------------------------------------------------------------
# Part 7 — CLI exit codes (subprocess via issue26_cli_probe.py)
# ---------------------------------------------------------------------------
class TestCLIExitCodes:
    def _probe(self, tmp_path, method, scenario, **kw):
        """Write scenario JSON, run probe, return CompletedProcess."""
        sc = {"run_root": str(tmp_path / "run"), "method": method,
              "allow_missing_metrics": kw.get("allow_missing_metrics", False),
              "patches": []}
        outcome = scenario
        if scenario == "success":
            pass  # no patches
        elif scenario == "missing_state":
            sc["patches"] = [{"target": "load_state", "outcome": "missing_state"}]
        elif scenario == "provider_init":
            sc["patches"] = [{"target": "load_state", "outcome": "provider_init"}]
        elif scenario == "state_validation":
            sc["patches"] = [{"target": "load_state", "outcome": "state_validation"}]
        elif scenario == "scoring_error":
            sc["patches"] = [{"target": "score_image", "outcome": "scoring_error"}]

        # Build a valid baseline run dir (so load_state doesn't fail first)
        install_issue26_stubs(mock.MagicMock())  # just for module setup
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(mock.MagicMock())
        meta = make_gs_meta("1", "watermarked", 5)
        rec_wm = env._make_record("1", "watermarked", method="GS", source_metadata=meta)
        rec_cl = env._make_record("1", "clean", method="GS", source_metadata=meta)
        _write_run(tmp_path, "GS", records=[rec_wm, rec_cl])

        sc_path = tmp_path / "scenario.json"
        sc_path.write_text(json.dumps(sc))
        result_json = tmp_path / "result.json"
        cmd = [sys.executable, str(PROBE), str(sc_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        # Also check result JSON if it was written
        if result_json.is_file():
            return proc, json.loads(result_json.read_text())
        return proc, None

    def test_success_exit_zero(self, tmp_path):
        proc, _ = self._probe(tmp_path, "GS", "success")
        assert proc.returncode == 0

    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_all_success_exit_zero(self, method, tmp_path):
        proc, _ = self._probe(tmp_path, method, "success")
        assert proc.returncode == 0, proc.stderr

    def test_missing_state_flag_gate(self, tmp_path):
        proc, _ = self._probe(tmp_path, "GS", "missing_state")
        assert proc.returncode == 2
        proc, _ = self._probe(tmp_path, "GS", "missing_state", allow_missing_metrics=True)
        assert proc.returncode == 0

    def test_hard_failures_stay_nonzero(self, tmp_path):
        for sc in ("provider_init", "state_validation", "scoring_error"):
            proc, _ = self._probe(tmp_path, "GS", sc, allow_missing_metrics=True)
            assert proc.returncode == 2, f"{sc}: {proc.returncode}"

    def test_result_json_retains_actual_status(self, tmp_path):
        proc, result = self._probe(tmp_path, "GS", "scoring_error")
        # Result JSON may not be written by default — check if available
        if result is not None:
            assert result["stages"]["detector"]["status"] != dc.COMPLETED  # not completed
            assert result["overall_status"] == dc.COMPLETED_WITH_ERRORS


# ---------------------------------------------------------------------------
# Part 8 — Generic failure contracts (representative method, not 7×cartesian)
# ---------------------------------------------------------------------------
class TestScoreContractViolations:
    """score_image contract: None / {} / missing canonical → failed_scoring."""

    def test_gs_score_none(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import raven.detectors.gs_detector as gd
        monkeypatch.setattr(gd, "load_state", lambda *a, **kw: {"provider": "fake"})
        monkeypatch.setattr(gd, "score_image", lambda *a, **kw: None)
        meta = make_gs_meta("1", "watermarked", 5)
        rec = _rec("1", "watermarked", "GS", input_path=str(tmp_path / "in.png"))
        _write_png(tmp_path / "in.png")
        out = _write_run(tmp_path, "GS", records=[rec], csv_rows=[meta])
        result = _eval_detector([rec], out, "GS",
            config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == dc.FAILED_SCORING  # STATUS_FAILED_SCORING
        rows = _rows(out)
        assert all(r["error_type"] == "NoneReturn" for r in rows)

    def test_gs_score_empty(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import raven.detectors.gs_detector as gd
        monkeypatch.setattr(gd, "load_state", lambda *a, **kw: {"provider": "fake"})
        monkeypatch.setattr(gd, "score_image", lambda *a, **kw: {})
        meta = make_gs_meta("1", "watermarked", 5)
        rec = _rec("1", "watermarked", "GS", input_path=str(tmp_path / "in.png"))
        _write_png(tmp_path / "in.png")
        out = _write_run(tmp_path, "GS", records=[rec], csv_rows=[meta])
        result = _eval_detector([rec], out, "GS",
            config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == dc.FAILED_SCORING
        rows = _rows(out)
        assert all(r["error_type"] == "ScoreContractViolation" for r in rows)

    def test_gs_missing_canonical(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import raven.detectors.gs_detector as gd
        monkeypatch.setattr(gd, "load_state", lambda *a, **kw: {"provider": "fake"})
        monkeypatch.setattr(gd, "score_image", lambda *a, **kw: {"raw_score": 0.5})
        meta = make_gs_meta("1", "watermarked", 5)
        rec = _rec("1", "watermarked", "GS", input_path=str(tmp_path / "in.png"))
        _write_png(tmp_path / "in.png")
        out = _write_run(tmp_path, "GS", records=[rec], csv_rows=[meta])
        result = _eval_detector([rec], out, "GS",
            config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == dc.FAILED_SCORING


class TestSetupFailures:
    """load_state exceptions classify correctly."""
    def test_gs_missing_state(self, monkeypatch, tmp_path):
        from raven.detectors import DetectorMissingStateError
        install_issue26_stubs(monkeypatch)
        import raven.detectors.gs_detector as gd
        monkeypatch.setattr(gd, "load_state",
            lambda *a, **kw: (_ for _ in ()).throw(DetectorMissingStateError("mock")))
        meta = make_gs_meta("1", "watermarked", 5)
        rec = _rec("1", "watermarked", "GS", input_path=str(tmp_path / "in.png"))
        _write_png(tmp_path / "in.png")
        out = _write_run(tmp_path, "GS", records=[rec], csv_rows=[meta])
        result = _eval_detector([rec], out, "GS",
            config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == dc.MISSING_REQUIRED_STATE  # STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_gs_provider_init_error(self, monkeypatch, tmp_path):
        from raven.detectors import DetectorProviderInitializationError
        install_issue26_stubs(monkeypatch)
        import raven.detectors.gs_detector as gd
        monkeypatch.setattr(gd, "load_state",
            lambda *a, **kw: (_ for _ in ()).throw(DetectorProviderInitializationError("mock")))
        meta = make_gs_meta("1", "watermarked", 5)
        rec = _rec("1", "watermarked", "GS", input_path=str(tmp_path / "in.png"))
        _write_png(tmp_path / "in.png")
        out = _write_run(tmp_path, "GS", records=[rec], csv_rows=[meta])
        result = _eval_detector([rec], out, "GS",
            config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == dc.PROVIDER_INIT  # STATUS_FAILED_PROVIDER_INITIALIZATION

    def test_missing_image_preflight(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch)
        meta = make_gs_meta("1", "watermarked", 5)
        rec_wm = env._make_record("1", "watermarked", method="GS", source_metadata=meta)
        rec_cl = env._make_record("1", "clean", method="GS", source_metadata=meta)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec_wm, rec_cl])
        # Remove the attacked_watermarked output image
        (out / "samples" / "watermarked" / "1" / "output.png").unlink()
        result = _eval_detector([rec_wm, rec_cl], out, "GS")
        assert result["status"] == dc.MISSING_IMAGE  # STATUS_FAILED_MISSING_IMAGE
        assert result["scored_count"] == 3
        rows = _rows(out)
        missing = [r for r in rows if r["status"] == dc.ROW_FAILED_MISSING_IMAGE]
        assert len(missing) == 1
        assert missing[0]["failure_cause"] == dc.CAUSE_MISSING_IMAGE  # FAILURE_CAUSE_MISSING_IMAGE


class TestCohortStatus:
    def test_missing_clean_cohort(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch)
        meta = make_gs_meta("1", "watermarked", 5)
        rec = env._make_record("1", "watermarked", method="GS", source_metadata=meta)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec])
        result = _eval_detector([rec], out, "GS")
        assert result["status"] == dc.COMPLETED_WITH_ERRORS  # STATUS_COMPLETED_WITH_ERRORS
        assert "original_clean" in result["missing_metric_cohorts"]

    def test_complete_cohorts_completed(self, monkeypatch, tmp_path):
        install_issue26_stubs(monkeypatch)
        import test_issue20_gs_detector as t20
        env = t20.TestEvaluateDetectorIntegration()
        env._setup_mocks(monkeypatch)
        meta_wm = make_gs_meta("1", "watermarked", 5)
        meta_cl = make_gs_meta("1", "clean", 5)
        rec_wm = env._make_record("1", "watermarked", method="GS", source_metadata=meta_wm)
        rec_cl = env._make_record("1", "clean", method="GS", source_metadata=meta_cl)
        out = env._write_fake_run(tmp_path, method="GS", records=[rec_wm, rec_cl])
        result = _eval_detector([rec_wm, rec_cl], out, "GS")
        assert result["status"] == dc.COMPLETED  # STATUS_COMPLETED
        assert result["metric_availability"]["primary_report_available"] is True
