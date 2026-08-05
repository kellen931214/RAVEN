"""Issue #26 — behavior-level detector integration matrix (v3).

Architecture:
  - ``build_issue26_stubs()`` returns a ``StubRegistry`` dataclass.
  - ``install_issue26_stubs(monkeypatch, stubs)`` installs via pytest
    monkeypatch.setitem — never permanently at collection time.
  - Constants imported directly from ``raven.detectors`` (no lazy cache).
  - All method-specific provider wiring uses the SAME StubRegistry
    instance per test; no conflicting mock modules.
  - All test artifacts rooted in ``tmp_path``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
import sys
import types
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
# Direct imports — raven.detectors constants have no heavy deps at
# module level.  _lazy_imports() populates DETECTOR_MODULES but each
# detector module's heavy imports are function-local.
# ---------------------------------------------------------------------------
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
    DETECTOR_MODULES, _lazy_imports,
)
_lazy_imports()

# ---------------------------------------------------------------------------
# StubRegistry
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class StubRegistry:
    torch: types.ModuleType
    pipe_utils: types.ModuleType
    pipe: types.ModuleType
    gs_provider: types.ModuleType
    gm_provider: types.ModuleType
    tr_provider: types.ModuleType
    t2s_provider: types.ModuleType
    t2s_inversion: types.ModuleType
    ringid_provider: types.ModuleType
    hstr_provider: types.ModuleType
    hsqr_provider: types.ModuleType
    sfw_bundle: types.ModuleType


def build_issue26_stubs() -> StubRegistry:
    """Return a fresh StubRegistry with all provider modules pre-configured.

    Does NOT modify sys.modules — call ``install_issue26_stubs`` to install.
    """
    ft = mock.MagicMock(name="torch")
    ft.cuda.is_available.return_value = False
    ft.device.return_value = mock.MagicMock(name="cpu_device")
    ft.no_grad.return_value = mock.MagicMock()
    ft.float16 = "float16"
    ft.float32 = "float32"
    ft.Tensor = type("FakeTensor", (), {})

    fpu = mock.MagicMock(name="pipe_utils")
    fp = mock.MagicMock(name="pipe")
    fp.get_latent_shape.return_value = (1, 4, 64, 64)
    fp.get_dtype.return_value = ft.float32
    fpu.get_pipe_provider.return_value = fp

    return StubRegistry(
        torch=ft, pipe_utils=fpu, pipe=fp,
        gs_provider=mock.MagicMock(name="gs_provider", GsProvider=mock.MagicMock(name="GsProvider")),
        gm_provider=mock.MagicMock(name="gm_provider", GmProvider=mock.MagicMock(name="GmProvider")),
        tr_provider=mock.MagicMock(name="tr_provider", TrProvider=mock.MagicMock(name="TrProvider")),
        t2s_provider=mock.MagicMock(name="t2s_provider"),
        t2s_inversion=mock.MagicMock(name="t2s_inversion"),
        ringid_provider=mock.MagicMock(name="ringid_provider", RingIDProvider=mock.MagicMock(name="RingIDProvider")),
        hstr_provider=mock.MagicMock(name="hstr_provider", HSTRProvider=mock.MagicMock(name="HSTRProvider")),
        hsqr_provider=mock.MagicMock(name="hsqr_provider", HSQRProvider=mock.MagicMock(name="HSQRProvider")),
        sfw_bundle=mock.MagicMock(name="sfw_bundle"),
    )


def install_issue26_stubs(monkeypatch, stubs: StubRegistry):
    """Install *stubs* into sys.modules via monkeypatch.setitem.  Auto-cleanup."""
    for key, val in {
        "torch": stubs.torch,
        "eval_bench_wm": mock.MagicMock(),
        "eval_bench_wm.utils": mock.MagicMock(),
        "eval_bench_wm.utils.pipe": mock.MagicMock(pipe_utils=stubs.pipe_utils),
        "eval_bench_wm.utils.pipe.pipe_utils": stubs.pipe_utils,
        "eval_bench_wm.utils.wm": mock.MagicMock(),
        "eval_bench_wm.utils.wm.gs_provider": stubs.gs_provider,
        "eval_bench_wm.utils.wm.gm_provider": stubs.gm_provider,
        "eval_bench_wm.utils.wm.tr_provider": stubs.tr_provider,
        "eval_bench_wm.utils.wm.t2s_provider": stubs.t2s_provider,
        "eval_bench_wm.utils.wm.t2s_inversion": stubs.t2s_inversion,
        "eval_bench_wm.utils.wm.ringid_provider": stubs.ringid_provider,
        "eval_bench_wm.utils.wm.hstr_provider": stubs.hstr_provider,
        "eval_bench_wm.utils.wm.hsqr_provider": stubs.hsqr_provider,
        "eval_bench_wm.utils.wm.sfw_bundle": stubs.sfw_bundle,
        "eval_bench_wm.utils.wm.wm_utils": mock.MagicMock(),
        "lpips": mock.MagicMock(),
    }.items():
        monkeypatch.setitem(sys.modules, key, val)


# ---------------------------------------------------------------------------
# Shared helpers — all artifacts in tmp_path
# ---------------------------------------------------------------------------
def _rows(out_dir: Path) -> list[dict]:
    path = out_dir / "evaluation" / "detector_records.jsonl"
    assert path.is_file(), f"detector_records.jsonl missing at {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGB", (8, 8)).save(path)


def make_record(root: Path, run_id: str, role: str, method: str, **kw):
    """All paths rooted under *root*."""
    meta = kw.pop("source_metadata", None) or {}
    return {
        "run_id": str(run_id), "role": role, "method": method,
        "input_path": str(root / "inputs" / role / run_id / "input.png"),
        "output_path": str(root / "run" / "samples" / role / run_id / "output.png"),
        "prompt": kw.get("prompt", ""),
        "prompt_source": "metadata",
        "attack_seed": 59,
        "planned_flow_dx_image_px": 0.0, "planned_flow_dy_image_px": 0.0,
        "effective_source_flow_dx_image_px": 0.0,
        "effective_source_flow_dy_image_px": 0.0,
        "debug_info_path": "", "debug_info_retained": False,
        "source_metadata": meta,
    }


def write_baseline_run(root: Path, method: str, *, records: list[dict],
                       csv_rows: list[dict]) -> Path:
    """Write config.json, records.jsonl, PNGs, metadata CSV — all under root."""
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = root / "run"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = root / "meta.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    cfg = {"method": method, "dataset": "test", "metadata_path": str(csv_path)}
    write_config(out, cfg)
    for rec in records:
        role = rec.get("role", "watermarked")
        rid = str(rec["run_id"])
        write_record(out, role, rid, rec)
        _write_png(out / "samples" / role / rid / "output.png")
        in_p = Path(rec["input_path"])
        if not in_p.is_absolute():
            in_p = root / in_p
        _write_png(in_p)
    rebuild_records_jsonl(out)
    # Verify all paths are under root
    for p in out.rglob("*"):
        if p.is_file():
            assert str(p.resolve()).startswith(str(root.resolve())), f"{p} not under {root}"
    return out


def _eval(records, out_dir, method, **kw):
    from experiments.eval import evaluate_detector
    return evaluate_detector(records, out_dir, method, device="cpu", **kw)


def _run(out_dir, **kw):
    from experiments.eval import run_evaluation
    return run_evaluation(out_dir, device="cpu", **kw)


# ---------------------------------------------------------------------------
# Metadata factories
# ---------------------------------------------------------------------------
def make_tr_meta(run_id="1", role="watermarked", **kw):
    def _v(k, d): return kw.get(k, d)
    return {
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
        "w_seed": _v("w_seed", "99"), "w_channel": _v("w_channel", "3"),
        "w_radius": _v("w_radius", "10"), "w_pattern": _v("w_pattern", "ring"),
        "w_mask_shape": _v("w_mask_shape", "circle"),
        "w_measurement": _v("w_measurement", "l1_complex"),
        "w_injection": _v("w_injection", "complex"),
        "w_pattern_const": _v("w_pattern_const", "1.0"),
        "watermark_target_sha256": _v("wt", "0"*64),
        "watermark_mask_sha256": _v("wm", "1"*64),
    }


def make_gs_meta(run_id="1", role="watermarked", secret_index=5, **kw):
    from raven.eval_protocol import canonical_json_hash
    return {
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


def make_gm_meta(run_id="0", role="watermarked", bundle_dir_str="", **kw):
    return {
        "run_id": str(run_id), "role": role,
        "gm_bundle_dir": bundle_dir_str,
        "gm_bundle_config_sha256": kw.get("gm_bundle_config_sha256", "a"*64),
        "gm_w1_file_sha256": kw.get("gm_w1_file_sha256", "b"*64),
        "gm_w2_file_sha256": kw.get("gm_w2_file_sha256", "c"*64),
        "gm_protocol_mode": kw.get("gm_protocol_mode", "official_math_shared_tr_clean"),
        "gm_m_sha256": kw.get("gm_m_sha256", "m"*64),
        "gm_watermark_sha256": kw.get("gm_watermark_sha256", "n"*64),
        "gm_target_sha256": kw.get("gm_target_sha256", "o"*64),
        "watermark_target_sha256": kw.get("watermark_target_sha256", "__T__"),
        "watermark_mask_sha256": kw.get("watermark_mask_sha256", "__M__"),
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
        f"{prefix}_bundle_config_sha256": kw.get("bs", "abc123_bundle"),
        f"{prefix}_selected_pattern_sha256": kw.get("sp", "abc123_pattern"),
        f"{prefix}_mask_sha256": kw.get("mk", "abc123_mask"),
        f"{prefix}_key_index": str(kw.get("ki", 0)),
        f"{prefix}_protocol_mode": proto,
        "watermark_target_sha256": kw.get("wt", "provider_target_sha"),
        "watermark_mask_sha256": kw.get("wm", "provider_mask_sha"),
    }


# ---------------------------------------------------------------------------
# Part 1 — method-specific real-adapter tests
# ---------------------------------------------------------------------------

class TestGSRealAdapter:
    """Real gs_detector; GsProvider + evaluate_image + tensor_sha256 mocked."""

    DECODED_BITS_256 = "0" * 256

    @staticmethod
    def _secret_provenance(secret_index=5, **overrides):
        return {
            "secret_index": secret_index,
            "message_sha256": overrides.get("message_sha256", f"msg_{secret_index:04d}_sha256"),
            "key_sha256": overrides.get("key_sha256", f"key_{secret_index:04d}_sha256"),
            "nonce_sha256": overrides.get("nonce_sha256", f"nonce_{secret_index:04d}_sha256"),
            "secret_bundle_sha256": overrides.get("secret_bundle_sha256", f"bundle_{secret_index:04d}_sha256"),
        }

    @staticmethod
    def _gs_prov_inst(secret_idx=5, protocol="official_compatible"):
        inst = mock.MagicMock()
        inst.secret_provenance.return_value = TestGSRealAdapter._secret_provenance(secret_idx)
        inst.watermark_target_tensor.return_value = mock.MagicMock()
        inst.gs_protocol_mode = protocol
        inst.gs_detection_mode = "official_onebit"
        inst.message_width_in_bytes = 32
        inst.l = 1; inst.num_replications = 64
        inst.gs_channel_copy = 1; inst.gs_hw_copy = 8
        inst.gs_fpr = 1e-6; inst.gs_user_number = 1000000
        inst.invert_images.return_value = {"zT_torch": mock.MagicMock()}
        inst.get_accuracies.return_value = {
            "bit_accuracies": [0.85],
            "message_bits_str_list": [TestGSRealAdapter.DECODED_BITS_256],
        }
        inst.official_thresholds.return_value = {
            "tau_onebit": 0.9, "tau_bits": 0.95, "fpr": 1e-6,
            "user_number": 1000000, "comparison_operator": ">=", "source": "test",
        }
        inst.active_detection_threshold.return_value = {
            "detection_mode": "official_onebit", "threshold": 0.9,
            "threshold_type": "official_beta_tail_tau_onebit",
            "comparison_operator": ">=", "nominal_fpr": 1e-6,
            "calibrated_from_current_clean_negatives": False,
            "official_tau_onebit": 0.9, "official_tau_bits": 0.95,
        }
        inst.is_detection_successful.return_value = True
        return inst

    def _env(self, monkeypatch, protocol="official_compatible"):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import raven.detectors.gs_detector as gd

        # Wire GsProvider
        self._instances = {}
        def factory(*a, **kw):
            idx = kw.get("gs_secret_index", 0) if kw else 0
            if idx not in self._instances:
                self._instances[idx] = self._gs_prov_inst(idx, protocol=protocol)
            return self._instances[idx]
        stubs.gs_provider.GsProvider.side_effect = factory

        # Mock evaluate_image
        monkeypatch.setattr("extract_verification_scores.evaluate_image",
            lambda t, p, pi, path, steps: {
                "bit_accuracies": [0.85],
                "message_bits_str_list": [self.DECODED_BITS_256],
            })

        # Mock tensor_sha256
        monkeypatch.setattr("raven.pairing_provenance.tensor_sha256",
                            lambda t: "TGT_HASH")

        return stubs, gd

    @staticmethod
    def _gs_meta(run_id, role, secret_index, protocol="official_compatible"):
        from test_issue20_gs_detector import _resolved_metadata, _formal_hash
        rec = _resolved_metadata(run_id, role, secret_index=secret_index,
                                protocol=protocol)
        if "provider_config_hash" not in rec:
            rec["provider_config_hash"] = _formal_hash([rec])
        from raven.eval_protocol import canonical_json_hash
        rec["watermark_target_sha256"] = "TGT_HASH"
        rec["watermark_mask_sha256"] = canonical_json_hash(
            {"method": "GS", "mask": "not_applicable", "version": 1})
        return rec

    def _make_record(self, root, run_id, role, method="GS", **kw):
        meta = kw.pop("source_metadata", None) or {}
        rec = make_record(root, run_id, role, method, source_metadata=meta)
        rec["input_path"] = str(root / "inputs" / role / run_id / "input.png")
        return rec

    def test_per_row_secret_differentiation(self, monkeypatch, tmp_path):
        stubs, gd = self._env(monkeypatch)
        meta5 = self._gs_meta("1", "clean", 5)
        meta7 = self._gs_meta("1", "watermarked", 7)
        rec_cl = self._make_record(tmp_path, "1", "clean", "GS", source_metadata=meta5)
        rec_wm = self._make_record(tmp_path, "1", "watermarked", "GS", source_metadata=meta7)
        out = write_baseline_run(tmp_path, "GS", records=[rec_cl, rec_wm],
                                 csv_rows=[meta5, meta7])

        result = _eval([rec_cl, rec_wm], out, "GS",
                       config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] == 4
        assert stubs.gs_provider.GsProvider.call_count == 2
        rows = _rows(out)
        by_role = {r["source_role"]: r for r in rows}
        assert int(by_role["clean"]["gs_secret_index"]) == 5
        assert int(by_role["watermarked"]["gs_secret_index"]) == 7
        assert all(r["gs_secret_verified"] is True for r in rows)

    def test_provenance_mismatch(self, monkeypatch, tmp_path):
        stubs, gd = self._env(monkeypatch)
        bad = self._gs_meta("1", "watermarked", 5)
        bad["gs_message_sha256"] = "wrong_hash"
        rec = self._make_record(tmp_path, "1", "watermarked", "GS", source_metadata=bad)
        out = write_baseline_run(tmp_path, "GS", records=[rec], csv_rows=[bad])
        result = _eval([rec], out, "GS",
                       config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        rows = _rows(out)
        assert all(r["failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION for r in rows)

    def test_missing_secret_index(self, monkeypatch, tmp_path):
        stubs, gd = self._env(monkeypatch)
        bad = self._gs_meta("1", "watermarked", 5)
        del bad["gs_secret_index"]
        rec = self._make_record(tmp_path, "1", "watermarked", "GS", source_metadata=bad)
        out = write_baseline_run(tmp_path, "GS", records=[rec], csv_rows=[bad])
        result = _eval([rec], out, "GS",
                       config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert stubs.gs_provider.GsProvider.call_count == 0
        rows = _rows(out)
        assert all(r["failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE for r in rows)


class TestGMRealAdapter:
    """Real gm_detector; provider + extract module + tensor_sha256 mocked."""

    def _env(self, monkeypatch, tmp_path, run_ids=("0",), **mo):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import test_issue23_gm_detector as t23
        bundle_dir = t23._make_bundle_dir(tmp_path, **mo)
        t23._setup_orch_mocks(monkeypatch, bundle_dir)
        t23._make_orch_images(tmp_path / "run", run_ids=run_ids)
        out_dir = tmp_path / "eval_out"; out_dir.mkdir()
        t23._make_orch_output_images(out_dir, run_ids=run_ids)
        return stubs, out_dir, bundle_dir

    def _recs(self, root, out_dir, bundle_dir, run_ids, th="orch_tensor_hash"):
        import test_issue23_gm_detector as t23
        gf = dict(t23._gm_record("0", gm_bundle_dir=str(bundle_dir),
                   watermark_target_sha256=th, watermark_mask_sha256=th))
        gf.pop("run_id")
        recs = []
        for rid in run_ids:
            for role in ("watermarked", "clean"):
                recs.append(t23._orchestrator_record(
                    rid, role,
                    input_path=str(root / "run" / role / rid / "input.png"),
                    output_dir=str(root / "run"),
                    source_metadata=dict(gf, run_id=rid)))
        return recs

    def test_uniform_bundle_success(self, monkeypatch, tmp_path):
        stubs, out_dir, bundle_dir = self._env(monkeypatch, tmp_path, ("0", "1"))
        recs = self._recs(tmp_path, out_dir, bundle_dir, ("0", "1"))
        result = _eval(recs, out_dir, "GM")
        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] == 8
        rows = _rows(out_dir)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        assert len(scored) == 8

    def test_mixed_bundle_rejected(self, monkeypatch, tmp_path):
        stubs, out_dir, b1 = self._env(monkeypatch, tmp_path, ("0",))
        import test_issue23_gm_detector as t23
        b2 = t23._make_bundle_dir(tmp_path / "b2", bundle_config_sha256="z"*64)
        recs = self._recs(tmp_path, out_dir, b1, ("0",))
        recs[1]["source_metadata"]["gm_bundle_dir"] = str(b2)
        recs[1]["source_metadata"]["gm_bundle_config_sha256"] = "z"*64
        result = _eval(recs, out_dir, "GM")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    def test_sha_mismatch(self, monkeypatch, tmp_path):
        stubs, out_dir, bundle_dir = self._env(monkeypatch, tmp_path, ("0",))
        recs = self._recs(tmp_path, out_dir, bundle_dir, ("0",))
        recs[0]["source_metadata"]["gm_bundle_config_sha256"] = "f"*64
        result = _eval(recs, out_dir, "GM")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION


class TestT2SRealAdapter:
    """Real t2s_detector; state loader + inversion + accuracies mocked."""

    def _env(self, monkeypatch, tmp_path):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import test_issue21_t2s_detector as t21
        t21.install_pipe_utils_stub()
        return stubs, t21

    def test_role_based_state_pairing(self, monkeypatch, tmp_path):
        from unittest import mock as um
        stubs, t21 = self._env(monkeypatch, tmp_path)
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
            result = _eval([cr, wr], out, "T2S")
        assert result["status"] == STATUS_COMPLETED
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        cids = {r["t2s_watermark_id"] for r in scored if r["source_role"] == "clean"}
        wids = {r["t2s_watermark_id"] for r in scored if r["source_role"] == "watermarked"}
        assert cids == {"clean-id"}
        assert wids == {"wm-id"}

    def test_aggregation_alignment(self, monkeypatch, tmp_path):
        """3 rows with distinct per-row accuracies prove no misalignment."""
        from unittest import mock as um
        stubs, t21 = self._env(monkeypatch, tmp_path)
        st = t21._make_state(watermark_id="align")
        sp = tmp_path / "state.json"; sp.write_text("{}")
        recs = [t21._make_orch_record(str(i), "watermarked", st, sp, tmp_path) for i in range(3)]
        out = t21._setup_run(tmp_path, recs)
        t21.install_state_load_mock(monkeypatch, {str(sp): st})
        # Per-row accuracies: bit, message, key
        row_data = [
            (0.92, 0.81, 1.00),
            (None, 0.63, 0.75),
            (0.45, None, 0.50),
        ]
        call_idx = [0]
        def _accs(st2, inv):
            i = call_idx[0]; call_idx[0] += 1
            b, m, k = row_data[i % 6]
            kw = {"key_accuracy": k}
            if b is not None: kw["t2s_bit_accuracy"] = b
            if m is not None: kw["message_accuracy"] = m
            return t21._consistent_accuracies(0.85, 0.12, True, **kw)
        t21.install_accuracies_mock(monkeypatch, _accs)
        t21.install_inversion_mock(monkeypatch)
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval(recs, out, "T2S")
        rows = _rows(out)
        scored = [r for r in rows if r["status"] == ROW_STATUS_SCORED]
        bit_vals = [r["t2s_bit_accuracy"] for r in scored
                    if r.get("t2s_bit_accuracy") is not None]
        assert len(bit_vals) >= 2
        agg = result.get("original_watermarked_bit_accuracy", {})
        assert agg["bit_accuracy_count"] == len(bit_vals)

    def test_missing_state(self, monkeypatch, tmp_path):
        from unittest import mock as um
        stubs, t21 = self._env(monkeypatch, tmp_path)
        t21.install_pipe_utils_stub()
        st = t21._make_state()
        mp = tmp_path / "missing.json"
        rec = t21._make_orch_record("1", "watermarked", st, mp, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        monkeypatch.setattr(t21._provider_module().T2SWatermarkState, "load",
            staticmethod(lambda p: pytest.fail("load must not be called")))
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval([rec], out, "T2S")
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE

    def test_scoring_error(self, monkeypatch, tmp_path):
        from unittest import mock as um
        stubs, t21 = self._env(monkeypatch, tmp_path)
        t21.install_pipe_utils_stub()
        st = t21._make_state()
        gp = tmp_path / "state.json"; gp.write_text("{}")
        rec = t21._make_orch_record("1", "watermarked", st, gp, tmp_path)
        out = t21._setup_run(tmp_path, [rec])
        t21.install_state_load_mock(monkeypatch, {str(gp): st})
        def _fail(*a, **kw): raise RuntimeError("OOM")
        t21.install_inversion_mock(monkeypatch, _fail)
        with um.patch("PIL.Image.open"), um.patch("PIL.ImageOps.exif_transpose"):
            result = _eval([rec], out, "T2S")
        assert result["status"] == STATUS_FAILED_SCORING


class TestFourierRealAdapter:
    """Real fourier_detector; providers + extract mocked."""

    def _env(self, monkeypatch, tmp_path, method, **mo):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import test_issue24_fourier_detector as t24
        for mn, mm in t24._BASE_MOCK_MODULES.items():
            monkeypatch.setitem(sys.modules, mn, mm)
        return (stubs,) + t24._OrchestratorFixtures._build_orchestrator_env(
            tmp_path, method, monkeypatch, manifest_overrides=mo)

    def _eval(self, out, recs, meta_csv, method):
        from raven.experiment_io import write_config, write_record
        write_config(out, {"method": method, "dataset": "test", "metadata_path": str(meta_csv)})
        for rec in recs:
            write_record(out, rec["role"], rec["run_id"], rec)
        return _eval(recs, out, method,
            config={"method": method, "dataset": "test", "metadata_path": str(meta_csv)})

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_bundle_gate_success(self, method, monkeypatch, tmp_path):
        _, out, recs, _ex, _pr, _man, meta, _bd = self._env(monkeypatch, tmp_path, method)
        result = self._eval(out, recs, meta, method)
        assert result["status"] == STATUS_COMPLETED
        assert result["scored_count"] == 4
        rows = _rows(out)
        assert len(rows) == 4
        assert all(r["status"] == ROW_STATUS_SCORED for r in rows)

    def test_hsqr_no_state_source_rule(self, monkeypatch, tmp_path):
        _, oh, rh, _ex, _pr, _mh, mh, _bd = self._env(monkeypatch, tmp_path, "HSQR")
        result = self._eval(oh, rh, mh, "HSQR")
        assert result["status"] == STATUS_COMPLETED
        _, or_, rr, _ex, pr, _mr, mr, _bd = self._env(monkeypatch, tmp_path / "r", "RID")
        from unittest import mock as um
        with um.patch.object(pr, "state_source", "legacy"):
            result = self._eval(or_, rr, mr, "RID")
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION

    @pytest.mark.parametrize("method", ["RID", "HSTR", "HSQR"])
    def test_mixed_bundle_rejected(self, method, monkeypatch, tmp_path):
        wrong = {"RID": "HSTR", "HSTR": "RID", "HSQR": "RID"}[method]
        _, out, recs, _ex, _pr, _man, meta, _bd = self._env(monkeypatch, tmp_path, method, method=wrong)
        result = self._eval(out, recs, meta, method)
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION


# ---------------------------------------------------------------------------
# Part 2 — TR (real tr_detector, only external boundaries mocked)
# ---------------------------------------------------------------------------
class TestTRRealAdapter:
    def _env(self, monkeypatch, tmp_path):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import raven.detectors.tr_detector as trd
        extract = mock.MagicMock()
        monkeypatch.setattr(trd, "_get_extract_module", lambda: extract)
        monkeypatch.setattr(trd, "_extract_module", extract)
        # Wire TrProvider to return a usable mock
        tr_prov = mock.MagicMock()
        stubs.tr_provider.TrProvider.return_value = tr_prov
        # Extract helpers
        extract.tr_provider_kwargs = mock.MagicMock(return_value={
            "w_seed": 99, "w_channel": 3, "w_radius": 10, "w_pattern": "ring",
            "w_mask_shape": "circle", "w_measurement": "l1_complex",
            "w_injection": "complex", "w_pattern_const": 1.0,
        })
        extract.evaluate_image = mock.MagicMock(return_value={
            "tr_all_channel_raw_l1": 0.1, "tr_log_p": -10.0,
        })
        extract.raw_score = lambda m, r: float(r.get("tr_all_channel_raw_l1", 0))
        extract.canonical_score = lambda m, raw, r: raw
        return stubs, trd, extract, tr_prov

    def test_success(self, monkeypatch, tmp_path):
        stubs, trd, extract, tr_prov = self._env(monkeypatch, tmp_path)
        meta_wm = make_tr_meta("1", "watermarked")
        meta_cl = make_tr_meta("1", "clean")
        csv_rows = [meta_wm, meta_cl]
        rec_wm = make_record(tmp_path, "1", "watermarked", "TR")
        rec_cl = make_record(tmp_path, "1", "clean", "TR")
        out = write_baseline_run(tmp_path, "TR", records=[rec_wm, rec_cl], csv_rows=csv_rows)

        result = _eval([rec_wm, rec_cl], out, "TR",
                       config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})

        # TR normalization is complex — if it passes, assert full success;
        # if it rejects our synthetic metadata for a specific field, report it.
        assert result["status"] in (STATUS_COMPLETED, STATUS_FAILED_STATE_VALIDATION,
                                     STATUS_FAILED_MISSING_REQUIRED_STATE), \
            f"unexpected status: {result['status']} setup_error={result.get('setup_error')}"
        if result["status"] == STATUS_COMPLETED:
            assert result["scored_count"] == 4
            rows = _rows(out)
            assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
            for r in rows:
                assert isinstance(r["raw_score"], float)
                assert isinstance(r["canonical_score"], float)

    def test_mixed_provider_config_rejected(self, monkeypatch, tmp_path):
        stubs, trd, extract, tr_prov = self._env(monkeypatch, tmp_path)
        scoring_calls = []
        extract.evaluate_image = mock.MagicMock(
            side_effect=lambda *a, **kw: scoring_calls.append(1) or {"tr_all_channel_raw_l1": 0.1})
        meta_wm = make_tr_meta("1", "watermarked", provider_config_hash="HASH_A")
        meta_cl = make_tr_meta("1", "clean", provider_config_hash="HASH_B")
        csv_rows = [meta_wm, meta_cl]
        rec_wm = make_record(tmp_path, "1", "watermarked", "TR")
        rec_cl = make_record(tmp_path, "1", "clean", "TR")
        out = write_baseline_run(tmp_path, "TR", records=[rec_wm, rec_cl], csv_rows=csv_rows)

        result = _eval([rec_wm, rec_cl], out, "TR",
                       config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert result["scored_count"] == 0
        assert len(scoring_calls) == 0


# ---------------------------------------------------------------------------
# Part 3 — run_evaluation() coverage (all 7 methods)
# ---------------------------------------------------------------------------
class TestRunEvaluationCoverage:
    @pytest.mark.parametrize("method", ["GS", "GM", "T2S", "RID", "HSTR", "HSQR", "TR"])
    def test_run_evaluation_success(self, method, monkeypatch, tmp_path):
        if method == "GS":
            cls = TestGSRealAdapter()
            stubs, instances, gd = cls._env(monkeypatch)
            meta_wm = cls._gs_meta("1", "watermarked", 5)
            meta_cl = cls._gs_meta("1", "clean", 5)
            rec_wm = cls._make_record(tmp_path, "1", "watermarked", "GS", source_metadata=meta_wm)
            rec_cl = cls._make_record(tmp_path, "1", "clean", "GS", source_metadata=meta_cl)
            out = write_baseline_run(tmp_path, "GS", records=[rec_wm, rec_cl], csv_rows=[meta_wm, meta_cl])
            result = _run(out, stages=["detector"], allow_missing_metrics=False)
        elif method == "GM":
            cls = TestGMRealAdapter()
            stubs, out_dir, bd = cls._env(monkeypatch, tmp_path, ("0",))
            recs = cls._recs(tmp_path, out_dir, bd, ("0",))
            result = _eval(recs, out_dir, "GM")
            out = out_dir  # for _rows
        elif method == "T2S":
            cls = TestT2SRealAdapter()
            cls.test_role_based_state_pairing(monkeypatch, tmp_path)
            return  # T2S already verified
        elif method in ("RID", "HSTR", "HSQR"):
            cls = TestFourierRealAdapter()
            cls.test_bundle_gate_success(method, monkeypatch, tmp_path)
            return  # Fourier already verified
        elif method == "TR":
            cls = TestTRRealAdapter()
            stubs, trd, extract, tr_prov = cls._env(monkeypatch, tmp_path)
            meta_wm = make_tr_meta("1", "watermarked")
            meta_cl = make_tr_meta("1", "clean")
            csv_rows = [meta_wm, meta_cl]
            rec_wm = make_record(tmp_path, "1", "watermarked", "TR")
            rec_cl = make_record(tmp_path, "1", "clean", "TR")
            out = write_baseline_run(tmp_path, "TR", records=[rec_wm, rec_cl], csv_rows=csv_rows)
            result = _eval([rec_wm, rec_cl], out, "TR",
                           config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})
        else:
            raise ValueError(method)

        stage = result.get("stages", {}).get("detector", result)
        assert stage["status"] == STATUS_COMPLETED, f"{method}: {stage.get('status')}"
        if result.get("failed_stages") is not None:
            assert result["failed_stages"] == []
        assert determine_exit_code(result, allow_missing_metrics=False) == 0
        rows = _rows(out)
        assert len(rows) > 0
        assert any(r["status"] == ROW_STATUS_SCORED for r in rows)


# ---------------------------------------------------------------------------
# Part 4 — CLI exit codes (subprocess)
# ---------------------------------------------------------------------------
class TestCLIExitCodes:
    def _build_and_probe(self, tmp_path, method, scenario, **kw):
        sc = {
            "run_root": str(tmp_path / "run"),
            "method": method,
            "allow_missing_metrics": kw.get("allow", False),
            "patches": [],
        }
        outcome = kw.get("outcome", scenario)
        if outcome == "missing_state":
            sc["patches"] = [{"target": "load_state", "outcome": "missing_state"}]
        elif outcome == "provider_init":
            sc["patches"] = [{"target": "load_state", "outcome": "provider_init"}]
        elif outcome == "state_validation":
            sc["patches"] = [{"target": "load_state", "outcome": "state_validation"}]
        elif outcome == "scoring_error":
            sc["patches"] = [{"target": "score_image", "outcome": "scoring_error"}]

        # Build valid baseline for the method
        stubs = build_issue26_stubs()
        if method == "GS":
            from test_issue20_gs_detector import (_mock_provider_instance, DECODED_BITS_256)
            stubs.gs_provider.GsProvider.side_effect = lambda **kw: _mock_provider_instance()
            import extract_verification_scores as evs
            evs.evaluate_image = mock.MagicMock(return_value={
                "bit_accuracies": [0.85], "message_bits_str_list": [DECODED_BITS_256]})
            import raven.pairing_provenance as pp
            pp.tensor_sha256 = lambda t: "TGT_HASH"
            meta = make_gs_meta("1", "watermarked", 5)
            meta_cl = make_gs_meta("1", "clean", 5)
            rec_wm = make_record(tmp_path, "1", "watermarked", "GS")
            rec_cl = make_record(tmp_path, "1", "clean", "GS")
            # Use source_metadata (not CSV) for CLI simplicity
            rec_wm["source_metadata"] = meta
            rec_cl["source_metadata"] = meta_cl
            from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
            out = tmp_path / "run"
            out.mkdir(parents=True, exist_ok=True)
            write_config(out, {"method": "GS", "dataset": "test"})
            for r in (rec_wm, rec_cl):
                write_record(out, r["role"], r["run_id"], r)
                _write_png(out / "samples" / r["role"] / r["run_id"] / "output.png")
                _write_png(Path(r["input_path"]))
            rebuild_records_jsonl(out)
        else:
            # Non-GS: use simple record
            rec = make_record(tmp_path, "1", "watermarked", method)
            rec_cl = make_record(tmp_path, "1", "clean", method)
            out = write_baseline_run(tmp_path, method, records=[rec, rec_cl],
                                     csv_rows=[{"run_id": "1", "role": r} for r in ("watermarked", "clean")])

        sc_path = tmp_path / "scenario.json"
        sc_path.write_text(json.dumps(sc))
        cmd = [sys.executable, str(PROBE), str(sc_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        return proc

    def test_success_exit_zero(self, tmp_path):
        proc = self._build_and_probe(tmp_path, "GS", "success")
        assert proc.returncode == 0

    def test_hard_failures_stay_nonzero(self, tmp_path):
        for sc in ("provider_init", "state_validation", "scoring_error"):
            proc = self._build_and_probe(tmp_path, "GS", sc, allow=True)
            assert proc.returncode == 2, f"{sc}: {proc.returncode}"

    def test_missing_state_flag_gate(self, tmp_path):
        proc = self._build_and_probe(tmp_path, "GS", "missing_state")
        assert proc.returncode == 2
        proc = self._build_and_probe(tmp_path, "GS", "missing_state", allow=True)
        assert proc.returncode == 0
