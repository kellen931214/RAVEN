#!/usr/bin/env python3
"""Issue #26 CLI probe — subprocess exit-code verification.

Reads a scenario JSON (written by the parent test) that describes:
  - run_root: path to a complete baseline run dir
  - patches: list of {"target": "load_state"|"score_image", "outcome": ...}

Installs module stubs, applies the requested patches, and calls
``experiments.eval.main()``.  Exit code is the process exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "raven_repro", REPO, REPO / "eval_bench_wm"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def _install_stubs():
    _ft = mock.MagicMock(name="torch")
    _ft.cuda.is_available.return_value = False
    _ft.device.return_value = mock.MagicMock()
    _ft.no_grad.return_value = mock.MagicMock()
    _ft.float16 = "float16"
    _ft.float32 = "float32"
    _ft.Tensor = type("FakeTensor", (), {})
    _fpu = mock.MagicMock(name="pipe_utils")
    _fp = mock.MagicMock(name="pipe")
    _fp.get_latent_shape.return_value = (1, 4, 64, 64)
    _fp.get_dtype.return_value = _ft.float32
    _fpu.get_pipe_provider.return_value = _fp
    stubs = {
        "torch": _ft,
        "eval_bench_wm.utils.pipe": mock.MagicMock(pipe_utils=_fpu),
        "eval_bench_wm.utils.pipe.pipe_utils": _fpu,
        "eval_bench_wm.utils.wm.gs_provider": mock.MagicMock(GsProvider=mock.MagicMock()),
        "eval_bench_wm.utils.wm.gm_provider": mock.MagicMock(GmProvider=mock.MagicMock()),
        "eval_bench_wm.utils.wm.sfw_bundle": mock.MagicMock(),
        "eval_bench_wm.utils.wm.ringid_provider": mock.MagicMock(RingIDProvider=mock.MagicMock()),
        "eval_bench_wm.utils.wm.hstr_provider": mock.MagicMock(HSTRProvider=mock.MagicMock()),
        "eval_bench_wm.utils.wm.hsqr_provider": mock.MagicMock(HSQRProvider=mock.MagicMock()),
        "eval_bench_wm.utils.wm.t2s_provider": mock.MagicMock(),
        "eval_bench_wm.utils.wm.t2s_inversion": mock.MagicMock(),
        "eval_bench_wm.utils.wm.tr_provider": mock.MagicMock(TrProvider=mock.MagicMock()),
        "eval_bench_wm.utils.wm.wm_utils": mock.MagicMock(),
        "lpips": mock.MagicMock(),
    }
    for k, v in stubs.items():
        if k not in sys.modules:
            sys.modules[k] = v
    # Ensure parent namespace packages are importable
    for pkg in ("eval_bench_wm", "eval_bench_wm.utils", "eval_bench_wm.utils.wm"):
        if pkg not in sys.modules:
            sys.modules[pkg] = mock.MagicMock()


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: issue26_cli_probe.py <scenario.json>", file=sys.stderr)
        return 1
    scenario_path = Path(sys.argv[1])
    scenario = json.loads(scenario_path.read_text())
    run_root = Path(scenario["run_root"])
    allow_missing = scenario.get("allow_missing_metrics", False)
    patches = scenario.get("patches", [])

    _install_stubs()

    from raven.detectors import DETECTOR_MODULES, _lazy_imports
    _lazy_imports()
    from raven.detectors import (DetectorScoringError, DetectorMissingStateError,
                                  DetectorProviderInitializationError,
                                  DetectorStateValidationError)

    for patch in patches:
        mod = DETECTOR_MODULES.get(scenario["method"].upper())
        if mod is None:
            continue
        target = patch["target"]
        outcome = patch["outcome"]
        if target == "load_state":
            if outcome == "missing_state":
                mod.load_state = lambda *a, **kw: (_ for _ in ()).throw(DetectorMissingStateError("mock"))
            elif outcome == "provider_init":
                mod.load_state = lambda *a, **kw: (_ for _ in ()).throw(DetectorProviderInitializationError("mock"))
            elif outcome == "state_validation":
                mod.load_state = lambda *a, **kw: (_ for _ in ()).throw(DetectorStateValidationError("mock"))
        elif target == "score_image":
            if outcome == "scoring_error":
                mod.score_image = lambda *a, **kw: (_ for _ in ()).throw(DetectorScoringError("mock"))
            elif outcome == "none":
                mod.score_image = lambda *a, **kw: None
            elif outcome == "empty":
                mod.score_image = lambda *a, **kw: {}

    from experiments.eval import main as eval_main
    argv = ["--output-dir", str(run_root), "--device", "cpu", "--stages", "detector"]
    if allow_missing:
        argv.append("--allow-missing-metrics")
    return eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
