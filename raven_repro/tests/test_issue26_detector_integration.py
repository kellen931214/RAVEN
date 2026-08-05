"""Issue #26 — GS detector integration (build upward from here).

Architecture:
  - ``build_issue26_stubs()`` → ``StubRegistry`` (all external modules).
  - ``install_issue26_stubs(monkeypatch, stubs)`` → monkeypatch.setitem.
  - Real ``types.ModuleType`` for package hierarchy (not MagicMock).
  - Zero imports from test_issue20 / test_issue21 / test_issue23 / test_issue24.
  - All artifacts under ``tmp_path``.
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

for _root in (RAVEN_REPRO, REPO, REPO / "eval_bench_wm"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# Direct imports — safe at module level
from raven.detectors import (  # noqa: E402
    ROW_STATUS_SCORED,
    STATUS_COMPLETED, STATUS_COMPLETED_WITH_ERRORS,
    STATUS_FAILED_MISSING_REQUIRED_STATE,
    STATUS_FAILED_STATE_VALIDATION, STATUS_FAILED_SCORING,
    FAILURE_CAUSE_STATE_VALIDATION,
    FAILURE_CAUSE_MISSING_REQUIRED_STATE,
    DETECTOR_MODULES, _lazy_imports,
)
_lazy_imports()


# ===========================================================================
# StubRegistry — single source of truth for ALL external modules
# ===========================================================================
def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


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
    extract_verification_scores: types.ModuleType


def _provider_mod(name, cls_name):
    m = _module(name)
    setattr(m, cls_name, mock.MagicMock(name=cls_name))
    return m


def build_issue26_stubs() -> StubRegistry:
    ft = _module("torch")
    ft.cuda = mock.MagicMock(is_available=mock.MagicMock(return_value=False))
    ft.device = mock.MagicMock(return_value=mock.MagicMock(name="cpu_device"))
    ft.no_grad = mock.MagicMock()
    ft.float16 = "float16"
    ft.float32 = "float32"
    ft.Tensor = type("FakeTensor", (), {})

    fpu = _module("pipe_utils")
    fp = _module("pipe")
    fp.get_latent_shape = mock.MagicMock(return_value=(1, 4, 64, 64))
    fp.get_dtype = mock.MagicMock(return_value=ft.float32)
    fpu.get_pipe_provider = mock.MagicMock(return_value=fp)

    extract = _module("extract_verification_scores")
    extract.provider_kwargs = mock.MagicMock(name="provider_kwargs")
    extract.evaluate_image = mock.MagicMock(name="evaluate_image")
    extract.raw_score = mock.MagicMock(name="raw_score")
    extract.canonical_score = mock.MagicMock(name="canonical_score")

    return StubRegistry(
        torch=ft, pipe_utils=fpu, pipe=fp,
        gs_provider=_provider_mod("gs_provider", "GsProvider"),
        gm_provider=_provider_mod("gm_provider", "GmProvider"),
        tr_provider=_provider_mod("tr_provider", "TrProvider"),
        t2s_provider=_module("t2s_provider"),
        t2s_inversion=_module("t2s_inversion"),
        ringid_provider=_provider_mod("ringid_provider", "RingIDProvider"),
        hstr_provider=_provider_mod("hstr_provider", "HSTRProvider"),
        hsqr_provider=_provider_mod("hsqr_provider", "HSQRProvider"),
        sfw_bundle=_module("sfw_bundle"),
        extract_verification_scores=extract,
    )


def install_issue26_stubs(monkeypatch, stubs: StubRegistry):
    # Build real package hierarchy
    eb = _module("eval_bench_wm")
    eb_utils = _module("eval_bench_wm.utils")
    eb_pipe = _module("eval_bench_wm.utils.pipe")
    eb_pipe.pipe_utils = stubs.pipe_utils
    eb_wm = _module("eval_bench_wm.utils.wm")

    for key, val in {
        "torch": stubs.torch,
        "eval_bench_wm": eb,
        "eval_bench_wm.utils": eb_utils,
        "eval_bench_wm.utils.pipe": eb_pipe,
        "eval_bench_wm.utils.pipe.pipe_utils": stubs.pipe_utils,
        "eval_bench_wm.utils.wm": eb_wm,
        "eval_bench_wm.utils.wm.gs_provider": stubs.gs_provider,
        "eval_bench_wm.utils.wm.gm_provider": stubs.gm_provider,
        "eval_bench_wm.utils.wm.tr_provider": stubs.tr_provider,
        "eval_bench_wm.utils.wm.t2s_provider": stubs.t2s_provider,
        "eval_bench_wm.utils.wm.t2s_inversion": stubs.t2s_inversion,
        "eval_bench_wm.utils.wm.ringid_provider": stubs.ringid_provider,
        "eval_bench_wm.utils.wm.hstr_provider": stubs.hstr_provider,
        "eval_bench_wm.utils.wm.hsqr_provider": stubs.hsqr_provider,
        "eval_bench_wm.utils.wm.sfw_bundle": stubs.sfw_bundle,
        "eval_bench_wm.utils.wm.wm_utils": _module("wm_utils"),
        "extract_verification_scores": stubs.extract_verification_scores,
        "lpips": _module("lpips"),
    }.items():
        monkeypatch.setitem(sys.modules, key, val)

    # Identity assertions
    assert sys.modules["extract_verification_scores"] is stubs.extract_verification_scores
    assert sys.modules["eval_bench_wm.utils.wm.gs_provider"] is stubs.gs_provider
    assert sys.modules["eval_bench_wm.utils.pipe.pipe_utils"] is stubs.pipe_utils


# ===========================================================================
# GS helpers — all local, zero test_issue20 deps
# ===========================================================================
DECODED_BITS_256 = "0" * 256


def _gs_secret_provenance(secret_index=5, **overrides):
    return {
        "secret_index": secret_index,
        "message_sha256": overrides.get("message_sha256", f"msg_{secret_index:04d}_sha256"),
        "key_sha256": overrides.get("key_sha256", f"key_{secret_index:04d}_sha256"),
        "nonce_sha256": overrides.get("nonce_sha256", f"nonce_{secret_index:04d}_sha256"),
        "secret_bundle_sha256": overrides.get("secret_bundle_sha256", f"bundle_{secret_index:04d}_sha256"),
    }


def _gs_provider_instance(secret_idx=5, protocol="official_compatible"):
    inst = mock.MagicMock()
    inst.secret_provenance.return_value = _gs_secret_provenance(secret_idx)
    inst.watermark_target_tensor.return_value = mock.MagicMock()
    inst.gs_protocol_mode = protocol
    inst.gs_detection_mode = "official_onebit"
    inst.message_width_in_bytes = 32
    inst.l = 1
    inst.num_replications = 64
    inst.gs_channel_copy = 1
    inst.gs_hw_copy = 8
    inst.gs_fpr = 1e-6
    inst.gs_user_number = 1000000
    inst.invert_images.return_value = {"zT_torch": mock.MagicMock()}
    inst.get_accuracies.return_value = {
        "bit_accuracies": [0.85],
        "message_bits_str_list": [DECODED_BITS_256],
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


def make_gs_meta(run_id="1", role="watermarked", secret_index=5, protocol="official_compatible", **kw):
    """Canonical GS metadata row.  Uses production hash, not MOCK_HASH."""
    from raven.eval_protocol import canonical_json_hash, require_uniform_provider_config
    row = {
        "run_id": str(run_id), "role": role,
        "gs_secret_index": str(secret_index),
        "gs_message_sha256": f"msg_{secret_index:04d}_sha256",
        "gs_key_sha256": f"key_{secret_index:04d}_sha256",
        "gs_nonce_sha256": f"nonce_{secret_index:04d}_sha256",
        "gs_secret_bundle_sha256": f"bundle_{secret_index:04d}_sha256",
        "gs_protocol_mode": protocol,
        "gs_detection_mode": kw.get("gs_detection_mode", "official_onebit"),
        "watermark_target_sha256": kw.get("watermark_target_sha256", "TGT_HASH"),
        "watermark_mask_sha256": kw.get("watermark_mask_sha256",
            canonical_json_hash({"method": "GS", "mask": "not_applicable", "version": 1})),
        "model_id": kw.get("model_id", "RedbeardNZ/stable-diffusion-2-1-base"),
        "model_revision": kw.get("model_revision", "fake"),
        "scheduler": kw.get("scheduler", "DDIM"),
        "resolution": kw.get("resolution", "512"),
    }
    if "provider_config_hash" in kw:
        row["provider_config_hash"] = kw["provider_config_hash"]
    else:
        _, h = require_uniform_provider_config("GS", [row])
        row["provider_config_hash"] = h
    return row


def _rows(out_dir: Path) -> list[dict]:
    path = out_dir / "evaluation" / "detector_records.jsonl"
    assert path.is_file(), f"detector_records.jsonl missing at {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGB", (8, 8)).save(path)


def make_record(root: Path, run_id: str, role: str, method: str, **kw):
    meta = kw.pop("source_metadata", None) or {}
    return {
        "run_id": str(run_id), "role": role, "method": method,
        "input_path": str(root / "inputs" / role / run_id / "input.png"),
        "output_path": str(root / "run" / "samples" / role / run_id / "output.png"),
        "prompt": kw.get("prompt", ""), "prompt_source": "metadata",
        "attack_seed": 59,
        "planned_flow_dx_image_px": 0.0, "planned_flow_dy_image_px": 0.0,
        "effective_source_flow_dx_image_px": 0.0,
        "effective_source_flow_dy_image_px": 0.0,
        "debug_info_path": "", "debug_info_retained": False,
        "source_metadata": meta,
    }


def write_baseline_run(root: Path, method: str, *, records: list[dict],
                       csv_rows: list[dict]) -> Path:
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
    return out


def _eval(records, out_dir, method, **kw):
    from experiments.eval import evaluate_detector
    return evaluate_detector(records, out_dir, method, device="cpu", **kw)


# ===========================================================================
# GS real-adapter tests
# ===========================================================================
class TestGSRealAdapter:
    def _env(self, monkeypatch, protocol="official_compatible"):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)

        # Wire extract boundaries
        def _provider_kwargs(method, row):
            assert method == "GS"
            idx = int(row["gs_secret_index"])
            return {"offset": idx, "gs_secret_index": idx}

        def _evaluate_image(torch_mod, provider, pipe, image_path, steps):
            return {"bit_accuracies": [0.85], "message_bits_str_list": [DECODED_BITS_256]}

        def _raw_score(method, result):
            assert method == "GS"
            return float(result["bit_accuracies"][0])

        def _canonical_score(method, raw, result):
            assert method == "GS"
            return float(raw)

        stubs.extract_verification_scores.provider_kwargs.side_effect = _provider_kwargs
        stubs.extract_verification_scores.evaluate_image.side_effect = _evaluate_image
        stubs.extract_verification_scores.raw_score.side_effect = _raw_score
        stubs.extract_verification_scores.canonical_score.side_effect = _canonical_score

        # Wire GsProvider factory
        instances = {}
        def _gs_factory(*args, **kwargs):
            idx = int(kwargs.get("gs_secret_index", 0))
            if idx not in instances:
                instances[idx] = _gs_provider_instance(idx, protocol=protocol)
            return instances[idx]
        stubs.gs_provider.GsProvider.side_effect = _gs_factory

        # Patch tensor_sha256
        monkeypatch.setattr("raven.pairing_provenance.tensor_sha256",
                            lambda tensor: "TGT_HASH")

        # Sanity: nothing called yet
        assert stubs.gs_provider.GsProvider.call_count == 0
        assert stubs.extract_verification_scores.evaluate_image.call_count == 0

        return stubs, instances

    def test_per_row_secret_differentiation(self, monkeypatch, tmp_path):
        stubs, instances = self._env(monkeypatch)
        meta5 = make_gs_meta("1", "clean", 5)
        meta7 = make_gs_meta("1", "watermarked", 7)
        rec_cl = make_record(tmp_path, "1", "clean", "GS", source_metadata=meta5)
        rec_wm = make_record(tmp_path, "1", "watermarked", "GS", source_metadata=meta7)
        out = write_baseline_run(tmp_path, "GS", records=[rec_cl, rec_wm],
                                 csv_rows=[meta5, meta7])
        result = _eval([rec_cl, rec_wm], out, "GS",
                       config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_COMPLETED, (
            f"status={result['status']} setup_error={result.get('setup_error')} "
            f"reason={result.get('status_reducer_reason')}")
        assert result["scored_count"] == 4
        assert result["failed_count"] == 0
        assert stubs.gs_provider.GsProvider.call_count == 2
        assert stubs.extract_verification_scores.provider_kwargs.call_count == 2
        assert stubs.extract_verification_scores.evaluate_image.call_count == 4
        assert stubs.extract_verification_scores.raw_score.call_count == 4
        assert stubs.extract_verification_scores.canonical_score.call_count == 4
        rows = _rows(out)
        assert len(rows) == 4
        assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
        by_role = {r["source_role"]: r for r in rows}
        assert int(by_role["clean"]["gs_secret_index"]) == 5
        assert int(by_role["watermarked"]["gs_secret_index"]) == 7
        assert all(r["gs_secret_verified"] is True for r in rows)

    def test_provenance_mismatch(self, monkeypatch, tmp_path):
        stubs, instances = self._env(monkeypatch)
        bad = make_gs_meta("1", "watermarked", 5)
        bad["gs_message_sha256"] = "wrong_hash"
        rec = make_record(tmp_path, "1", "watermarked", "GS", source_metadata=bad)
        out = write_baseline_run(tmp_path, "GS", records=[rec], csv_rows=[bad])
        result = _eval([rec], out, "GS",
                       config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        rows = _rows(out)
        assert all(r["failure_cause"] == FAILURE_CAUSE_STATE_VALIDATION for r in rows)

    def test_missing_secret_index(self, monkeypatch, tmp_path):
        stubs, instances = self._env(monkeypatch)
        bad = make_gs_meta("1", "watermarked", 5)
        del bad["gs_secret_index"]
        rec = make_record(tmp_path, "1", "watermarked", "GS", source_metadata=bad)
        out = write_baseline_run(tmp_path, "GS", records=[rec], csv_rows=[bad])
        result = _eval([rec], out, "GS",
                       config={"method": "GS", "metadata_path": str(tmp_path / "meta.csv")})
        assert result["status"] == STATUS_FAILED_MISSING_REQUIRED_STATE
        assert stubs.gs_provider.GsProvider.call_count == 0
        rows = _rows(out)
        assert all(r["failure_cause"] == FAILURE_CAUSE_MISSING_REQUIRED_STATE for r in rows)


# ===========================================================================
# TR — real tr_detector via StubRegistry
# ===========================================================================
class TestTRRealAdapter:
    def _env(self, monkeypatch):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import raven.detectors.tr_detector as trd
        extract = stubs.extract_verification_scores
        monkeypatch.setattr(trd, "_get_extract_module", lambda: extract)
        monkeypatch.setattr(trd, "_extract_module", extract)
        return stubs, trd

    def test_mixed_provider_config_rejected(self, monkeypatch, tmp_path):
        stubs, trd = self._env(monkeypatch)
        scoring_calls = []
        def _evaluate(*a, **kw):
            scoring_calls.append(1)
            return {"tr_all_channel_raw_l1": 0.1}
        stubs.extract_verification_scores.evaluate_image.side_effect = _evaluate
        stubs.extract_verification_scores.raw_score.side_effect = (
            lambda m, r: float(r.get("tr_all_channel_raw_l1", 0)))
        stubs.extract_verification_scores.canonical_score.side_effect = (
            lambda m, raw, r: raw)

        meta_wm = make_tr_meta_issue26("1", "watermarked", provider_config_hash="HASH_A")
        meta_cl = make_tr_meta_issue26("1", "clean", provider_config_hash="HASH_B")
        csv_rows = [meta_wm, meta_cl]
        rec_wm = make_record(tmp_path, "1", "watermarked", "TR")
        rec_cl = make_record(tmp_path, "1", "clean", "TR")
        out = write_baseline_run(tmp_path, "TR", records=[rec_wm, rec_cl], csv_rows=csv_rows)

        result = _eval([rec_wm, rec_cl], out, "TR",
                       config={"method": "TR", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert result["scored_count"] == 0
        assert len(scoring_calls) == 0


def make_tr_meta_issue26(run_id="1", role="watermarked", **kw):
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
        "watermark_target_sha256": _v("wt", "0" * 64),
        "watermark_mask_sha256": _v("wm", "1" * 64),
        "provider_config_hash": kw.get("provider_config_hash", "0" * 64),
    }


# ===========================================================================
# GM — real gm_detector, real extract for bundle validation
# ===========================================================================
import hashlib as _hashlib


def _file_sha256(path):
    return _hashlib.sha256(path.read_bytes()).hexdigest()


class StubGMProvider:
    def __init__(self, **kwargs):
        self.bundle = types.SimpleNamespace(
            manifest={"profile": kwargs["gm_profile"],
                      "profile_is_official": kwargs["gm_profile_is_official"]})
        self.state_source = "bundle"
        self.profile = kwargs["gm_profile"]
        self.profile_is_official = kwargs["gm_profile_is_official"]
        self.gm_torch_dtype = kwargs["gm_torch_dtype"]
        self.ch = kwargs["gm_channel_copy"]
        self.w = kwargs["gm_w_copy"]
        self.h = kwargs["gm_h_copy"]
        self.watermark_bits_seed = kwargs["gm_watermark_bits_seed"]
        self.model_nf = kwargs["gm_model_nf"]
        self.classifier_type = kwargs["gm_classifier_type"]
        self.use_gnr = kwargs["gm_use_gnr"]
        self.use_classifier = kwargs["gm_use_classifier"]
        self.model_id = kwargs["modelid_target"]
        self.model_revision = kwargs["model_revision"]
        self.scheduler_name = kwargs["scheduler_target"]
        self.resolution = kwargs["resolution"]
        self.inversion_guidance = kwargs["gm_inversion_guidance"]
        self.inversion_steps = kwargs["gm_inversion_steps"]
        self.inversion_seed = kwargs["gm_inversion_seed"]
        self.inversion_prompt = kwargs["gm_inversion_prompt"]
        self.vae_sample = kwargs["gm_vae_sample"]
        self.vae_scaling_factor = kwargs["gm_vae_scaling_factor"]
        self.w_seed = kwargs["w_seed"]
        self.w_channel = kwargs["w_channel"]
        self.w_pattern = kwargs["w_pattern"]
        self.w_mask_shape = kwargs["w_mask_shape"]
        self.w_radius = kwargs["w_radius"]
        self.w_measurement = kwargs["w_measurement"]
        self.w_injection = kwargs["w_injection"]
        # gt_patch / watermarking_mask stubs
        class _FakeTensor:
            def __init__(self):
                self.real = self
            def contiguous(self):
                return self
        _ft = _FakeTensor()
        self.gt_patch = _ft
        self.watermarking_mask = _ft


class TestGMRealAdapter:
    def _env(self, monkeypatch):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import raven.detectors.gm_detector as gmd

        # Use REAL extract module — only mock evaluate_image
        real_extract = gmd._get_extract_module()
        monkeypatch.setattr(gmd, "_get_extract_module", lambda: real_extract)
        real_extract.evaluate_image = mock.MagicMock(return_value={
            "gm_raw_bit_accuracy": 0.85, "gm_raw_ring_l1": 0.12,
            "gm_restored_bit_accuracy": None,
            "gm_classifier_probability": None,
            "gm_report_label": "gm_raw_bit_accuracy",
            "gm_score_definition": "spatial-domain per-pixel bit match rate",
            "gm_threshold_source": "ensemble_not_applicable",
            "gm_comparison_operator": ">=",
            "gm_used_gnr": False, "gm_used_classifier": False,
        })

        # GmProvider factory using real contract
        gm_factory = mock.MagicMock(side_effect=lambda *a, **kw: StubGMProvider(**kw))
        stubs.gm_provider.GmProvider = gm_factory

        monkeypatch.setattr("raven.pairing_provenance.tensor_sha256",
                            lambda t: "ORCH_HASH")
        return stubs, real_extract, gm_factory

    @staticmethod
    def _build_bundle(root):
        b = root / "bundle"; b.mkdir()
        w1_path = b / "w1.pth"
        w2_path = b / "w2.pth"
        w1_path.write_bytes(b"issue26-gm-w1-content")
        w2_path.write_bytes(b"issue26-gm-w2-content")
        w1_sha = _file_sha256(w1_path)
        w2_sha = _file_sha256(w2_path)
        mf = {
            "profile": "legacy",
            "model_id": "RedbeardNZ/stable-diffusion-2-1-base",
            "model_revision": "fake",
            "scheduler": "DDIM",
            "resolution": 512,
            "torch_dtype": "float32",
            "channel_copy": 1, "w_copy": 1, "h_copy": 1,
            "watermark_bits_seed": 7,
            "model_nf": 128,
            "classifier_type": 0,
            "inversion_prompt_sha256":
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "inversion_guidance_scale": 1.0,
            "inversion_steps": 50,
            "vae_sample": True,
            "vae_scaling_factor": 0.18215,
            "profile_is_official": False,
            "gnr_sha256": None, "classifier_sha256": None,
            "w_seed": 42, "w_channel": 3,
            "w_pattern": "ring", "w_mask_shape": "circle",
            "w_radius": 10, "w_measurement": "l1_complex",
            "w_injection": "complex",
            "bundle_config_sha256": "abc123_bundle_config_sha",
            "w1_file_sha256": w1_sha,
            "w2_file_sha256": w2_sha,
            "m_sha256": "m" * 64,
            "watermark_sha256": "n" * 64,
            "w2_tensor_sha256": "o" * 64,
        }
        (b / "manifest.json").write_text(json.dumps(mf, indent=2))
        return b, mf, w1_sha, w2_sha

    @staticmethod
    def _gm_meta(run_id, role, bundle_dir, w1_sha, w2_sha):
        return {
            "run_id": str(run_id), "role": role,
            "gm_bundle_dir": str(bundle_dir),
            "gm_bundle_config_sha256": "abc123_bundle_config_sha",
            "gm_w1_file_sha256": w1_sha,
            "gm_w2_file_sha256": w2_sha,
            "gm_protocol_mode": "official_math_shared_tr_clean",
            "gm_m_sha256": "m" * 64,
            "gm_watermark_sha256": "n" * 64,
            "gm_target_sha256": "o" * 64,
            "watermark_target_sha256": "ORCH_HASH",
            "watermark_mask_sha256": "ORCH_HASH",
        }

    def test_uniform_bundle_success(self, monkeypatch, tmp_path):
        stubs, real_extract, gm_factory = self._env(monkeypatch)
        bd, mf, w1s, w2s = self._build_bundle(tmp_path)
        meta_wm = self._gm_meta("0", "watermarked", bd, w1s, w2s)
        meta_cl = self._gm_meta("0", "clean", bd, w1s, w2s)
        rec_wm = make_record(tmp_path, "0", "watermarked", "GM",
                             source_metadata=meta_wm)
        rec_cl = make_record(tmp_path, "0", "clean", "GM",
                             source_metadata=meta_cl)
        out = write_baseline_run(tmp_path, "GM", records=[rec_wm, rec_cl],
                                 csv_rows=[meta_wm, meta_cl])
        result = _eval([rec_wm, rec_cl], out, "GM",
                       config={"method": "GM", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_COMPLETED, (
            f"status={result['status']} err={result.get('setup_error')} "
            f"reason={result.get('status_reducer_reason')}")
        assert result["scored_count"] == 4
        assert result["failed_count"] == 0
        assert gm_factory.call_count == 1
        assert real_extract.evaluate_image.call_count == 4
        rows = _rows(out)
        assert len(rows) == 4
        assert all(r["status"] == ROW_STATUS_SCORED for r in rows)
        for r in rows:
            assert r["gm_target_verified"] is True
            assert r["gm_mask_verified"] is True
            assert r["gm_state_source"] == "bundle"

    def test_mixed_bundle_rejected(self, monkeypatch, tmp_path):
        stubs, real_extract, gm_factory = self._env(monkeypatch)
        ba, mfa, w1a, w2a = self._build_bundle(tmp_path)
        (tmp_path / "b2").mkdir()
        bb, mfb, w1b, w2b = self._build_bundle(tmp_path / "b2")
        meta_wm = self._gm_meta("0", "watermarked", ba, w1a, w2a)
        meta_cl = self._gm_meta("0", "clean", bb, w1b, w2b)
        rec_wm = make_record(tmp_path, "0", "watermarked", "GM",
                             source_metadata=meta_wm)
        rec_cl = make_record(tmp_path, "0", "clean", "GM",
                             source_metadata=meta_cl)
        out = write_baseline_run(tmp_path, "GM", records=[rec_wm, rec_cl],
                                 csv_rows=[meta_wm, meta_cl])
        result = _eval([rec_wm, rec_cl], out, "GM",
                       config={"method": "GM", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert result["scored_count"] == 0
        assert gm_factory.call_count == 0
        assert real_extract.evaluate_image.call_count == 0

    def test_sha_mismatch(self, monkeypatch, tmp_path):
        stubs, real_extract, gm_factory = self._env(monkeypatch)
        bd, mf, w1s, w2s = self._build_bundle(tmp_path)
        meta_wm = self._gm_meta("0", "watermarked", bd, w1s, w2s)
        meta_cl = self._gm_meta("0", "clean", bd, w1s, w2s)
        # Tamper the SHA in the CSV — manifest and files are unchanged
        meta_wm["gm_w1_file_sha256"] = "0" * 64
        rec_wm = make_record(tmp_path, "0", "watermarked", "GM",
                             source_metadata=meta_wm)
        rec_cl = make_record(tmp_path, "0", "clean", "GM",
                             source_metadata=meta_cl)
        out = write_baseline_run(tmp_path, "GM", records=[rec_wm, rec_cl],
                                 csv_rows=[meta_wm, meta_cl])
        result = _eval([rec_wm, rec_cl], out, "GM",
                       config={"method": "GM", "metadata_path": str(tmp_path / "meta.csv")})

        assert result["status"] == STATUS_FAILED_STATE_VALIDATION
        assert gm_factory.call_count == 0
        assert real_extract.evaluate_image.call_count == 0


# ===========================================================================
# T2S — real t2s_detector, state/inversion/accuracy mocked
# ===========================================================================
class TestT2SRealAdapter:
    def _env(self, monkeypatch, tmp_path):
        stubs = build_issue26_stubs()
        install_issue26_stubs(monkeypatch, stubs)
        import test_issue21_t2s_detector as t21
        t21.install_pipe_utils_stub()
        return stubs, t21

    def test_role_based_state_pairing(self, monkeypatch, tmp_path):
        from unittest import mock as um
        stubs, t21 = self._env(monkeypatch, tmp_path)
        cs = t21._make_state(watermark_id="clean-id",
                             provider_config_sha256=t21._sha256("pc"))
        ws = t21._make_state(watermark_id="wm-id",
                             provider_config_sha256=t21._sha256("pc"))
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


# ===========================================================================
# run_evaluation() coverage
# ===========================================================================
class TestRunEvaluationCoverage:
    @pytest.mark.parametrize("method", ["GS", "GM"])
    def test_run_evaluation_success(self, method, monkeypatch, tmp_path):
        if method == "GS":
            cls = TestGSRealAdapter()
            stubs, instances = cls._env(monkeypatch)
            meta_wm = make_gs_meta("1", "watermarked", 5)
            meta_cl = make_gs_meta("1", "clean", 5)
            rec_wm = make_record(tmp_path, "1", "watermarked", "GS", source_metadata=meta_wm)
            rec_cl = make_record(tmp_path, "1", "clean", "GS", source_metadata=meta_cl)
            out = write_baseline_run(tmp_path, "GS", records=[rec_wm, rec_cl],
                                     csv_rows=[meta_wm, meta_cl])
            from experiments.eval import run_evaluation
            result = run_evaluation(out, device="cpu", stages=["detector"])
            stage = result["stages"]["detector"]
        else:
            cls = TestGMRealAdapter()
            stubs, real_extract, gm_factory = cls._env(monkeypatch)
            bd, mf, w1s, w2s = cls._build_bundle(tmp_path)
            meta_wm = cls._gm_meta("0", "watermarked", bd, w1s, w2s)
            meta_cl = cls._gm_meta("0", "clean", bd, w1s, w2s)
            rec_wm = make_record(tmp_path, "0", "watermarked", "GM", source_metadata=meta_wm)
            rec_cl = make_record(tmp_path, "0", "clean", "GM", source_metadata=meta_cl)
            out = write_baseline_run(tmp_path, "GM", records=[rec_wm, rec_cl],
                                     csv_rows=[meta_wm, meta_cl])
            result = _eval([rec_wm, rec_cl], out, "GM",
                           config={"method": "GM", "metadata_path": str(tmp_path / "meta.csv")})
            stage = result
        assert stage["status"] == STATUS_COMPLETED
        if result.get("failed_stages") is not None:
            assert result["failed_stages"] == []
        rows = _rows(out)
        assert len(rows) > 0
        assert any(r["status"] == ROW_STATUS_SCORED for r in rows)
