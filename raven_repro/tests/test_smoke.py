"""Minimal CPU-only smoke test for the public RAVEN package.

Verifies:
- ``raven`` package import
- Detector registry has all 7 methods
- ``raven_repro/main.py`` and ``raven_repro/eval.py`` parsers are constructable
- ``normalize_config`` basic round-trip
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "raven_repro"))


def test_raven_import() -> None:
    import raven
    assert raven


def test_detector_registry_all_seven_methods() -> None:
    from raven.detectors import get_detector_module
    for method in ("TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"):
        mod = get_detector_module(method)
        assert mod is not None, f"missing detector module: {method}"
        assert hasattr(mod, "load_state"), f"{method}: missing load_state"
        assert hasattr(mod, "score_image"), f"{method}: missing score_image"
        assert hasattr(mod, "aggregate"), f"{method}: missing aggregate"


def test_main_parser() -> None:
    from main import build_parser
    parser = build_parser()
    assert parser is not None


def test_eval_parser() -> None:
    from eval import build_parser
    parser = build_parser()
    assert parser is not None


def test_normalize_config_roundtrip() -> None:
    from raven.experiment_config import normalize_config, ALGORITHM_FIELDS
    config = normalize_config(
        diffusion_mode="ddim",
        method="TR",
        dataset="smoke",
        metadata_path="/tmp/metadata.csv",
        output_dir="/tmp/out",
    )
    for field in ALGORITHM_FIELDS:
        assert field in config, f"missing algorithm field: {field}"
    assert config["diffusion_mode"] == "ddim"
    assert config["method"] == "TR"
    assert config["dataset"] == "smoke"
