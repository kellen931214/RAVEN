#!/usr/bin/env python3
"""Child-process probe for the issue #26 CLI exit-code matrix.

Runs ``experiments.eval.main()`` in a fresh process with module-level
eval_bench_wm stubs so the real orchestrator path (config loading, records,
dispatch, reducer, serialization, exit code) runs without touching disk
for heavy dependencies.

Usage:
    python3 issue26_cli_probe.py --root DIR --method GS --scenario success
        [--allow-missing-metrics] [--result-json PATH] [--stages detector]
        [--unknown-method]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "raven_repro", REPO):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

# ---- module-level stubs (mirror test_issue26_detector_integration) ----
_fake_torch = mock.MagicMock(name="torch")
_fake_torch.cuda.is_available.return_value = False
_fake_torch.device.return_value = mock.MagicMock()
_fake_torch.no_grad.return_value = mock.MagicMock()
_fake_torch.float16 = "float16"
_fake_torch.float32 = "float32"
_fake_torch.Tensor = type("FakeTensor", (), {})
_fake_pipe_utils = mock.MagicMock(name="pipe_utils")
_fake_pipe = mock.MagicMock(name="pipe")
_fake_pipe.get_latent_shape.return_value = (1, 4, 64, 64)
_fake_pipe.get_dtype.return_value = _fake_torch.float32
_fake_pipe_utils.get_pipe_provider.return_value = _fake_pipe
_STUBS = {
    "torch": _fake_torch,
    "eval_bench_wm.utils.pipe": mock.MagicMock(pipe_utils=_fake_pipe_utils),
    "eval_bench_wm.utils.pipe.pipe_utils": _fake_pipe_utils,
    "eval_bench_wm.utils.wm.gs_provider": mock.MagicMock(GsProvider=mock.MagicMock()),
    "eval_bench_wm.utils.wm.gm_provider": mock.MagicMock(GmProvider=mock.MagicMock()),
    "eval_bench_wm.utils.wm.sfw_bundle": mock.MagicMock(),
    "eval_bench_wm.utils.wm.ringid_provider": mock.MagicMock(),
    "eval_bench_wm.utils.wm.hstr_provider": mock.MagicMock(),
    "eval_bench_wm.utils.wm.hsqr_provider": mock.MagicMock(),
}
for _k, _v in _STUBS.items():
    if _k not in sys.modules:
        sys.modules[_k] = _v


def _raise(exc):
    raise exc


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_bytes")


def _make_record(run_id="1", role="watermarked", method="GS"):
    return {
        "run_id": run_id, "role": role, "method": method,
        "input_path": f"/tmp/issue26_cli_in_{role}_{run_id}.png",
        "output_path": f"/tmp/issue26_cli_out/{role}/{run_id}/output.png",
        "prompt": "", "prompt_source": "metadata",
        "attack_seed": 59,
        "planned_flow_dx_image_px": 0.0, "planned_flow_dy_image_px": 0.0,
        "effective_source_flow_dx_image_px": 0.0,
        "effective_source_flow_dy_image_px": 0.0,
        "debug_info_path": "", "debug_info_retained": False,
        "source_metadata": {},
    }


SCENARIOS = {
    "success": {},
    "score_raises_scoring": {},
    "score_none": {},
    "score_empty": {},
    "load_missing_state": {},
    "load_provider_init": {},
    "load_state_validation": {},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", default="GS")
    parser.add_argument("--scenario", default="success")
    parser.add_argument("--allow-missing-metrics", action="store_true")
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--stages", nargs="+", default=["detector"])
    parser.add_argument("--unknown-method", action="store_true")
    args = parser.parse_args()

    # ---- write run dir ----
    root = args.root
    if args.unknown_method:
        rec = _make_record("1", "watermarked", "NOPE")
    else:
        rec = _make_record("1", "watermarked", args.method)
    from raven.experiment_io import write_config, write_record, rebuild_records_jsonl
    out = root / "run"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"method": args.method, "dataset": "test"}
    write_config(out, cfg)
    for role in ("watermarked", "clean"):
        r = dict(rec)
        r["role"] = role
        write_record(out, role, r["run_id"], r)
    _write_png(out / "samples" / "watermarked" / rec["run_id"] / "output.png")
    _write_png(out / "samples" / "clean" / rec["run_id"] / "output.png")
    rebuild_records_jsonl(out)

    # ---- patch adapter for scenario ----
    if not args.unknown_method:
        from raven.detectors import DETECTOR_MODULES, _lazy_imports
        _lazy_imports()
        mod = DETECTOR_MODULES.get(args.method.upper())
        if mod and args.scenario in SCENARIOS:
            _patch = SCENARIOS[args.scenario]
            if args.scenario == "score_raises_scoring":
                from raven.detectors import DetectorScoringError
                mod.score_image = lambda *a, **kw: (_raise(DetectorScoringError("mock")))
            elif args.scenario == "score_none":
                mod.score_image = lambda *a, **kw: None
            elif args.scenario == "score_empty":
                mod.score_image = lambda *a, **kw: {}
            elif args.scenario == "load_missing_state":
                from raven.detectors import DetectorMissingStateError
                mod.load_state = lambda *a, **kw: (_raise(DetectorMissingStateError("mock")))
            elif args.scenario == "load_provider_init":
                from raven.detectors import DetectorProviderInitializationError
                mod.load_state = lambda *a, **kw: (_raise(DetectorProviderInitializationError("mock")))
            elif args.scenario == "load_state_validation":
                from raven.detectors import DetectorStateValidationError
                mod.load_state = lambda *a, **kw: (_raise(DetectorStateValidationError("mock")))

    from experiments.eval import main as eval_main
    argv = ["--output-dir", str(out), "--device", "cpu",
            "--stages", *args.stages]
    if args.allow_missing_metrics:
        argv.append("--allow-missing-metrics")
    if args.result_json is not None:
        argv += ["--output", str(args.result_json)]
    return eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
