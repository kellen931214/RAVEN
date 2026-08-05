"""Tests for TR RAVEN storage-light mode (no input.png copy, no attack-clean).

TR scoring semantics come from the package modules directly; the legacy
``raven_nfpa_tr_eval.py`` script is not loaded here.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import raven.pipeline_raven as pipeline_raven
from raven.pipeline_raven import RavenPipeline

REPO = Path(__file__).resolve().parents[2]
FORMAL_EVAL_PATH = REPO / "experiments" / "run_raven_formal_eval.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


formal_eval = _load_module("run_raven_formal_eval_under_test", FORMAL_EVAL_PATH)


class _Stop(RuntimeError):
    pass


def _run_pipeline_until_encode(tmp_path, monkeypatch, save_input_copy):
    """Drive RavenPipeline.run() up to the first heavy step and stop.

    This exercises only the early input.png save guard without any model.
    """
    import types

    pipe = RavenPipeline.__new__(RavenPipeline)
    pipe.torch = object()
    pipe.device = "cpu"
    pipe.pipe = types.SimpleNamespace(unet=object())
    pipe._default_attn_processors = {}

    monkeypatch.setattr(pipeline_raven, "restore_default_attention", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_raven, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(pipe, "_make_generator", lambda seed: None, raising=False)

    def _stop(*args, **kwargs):
        raise _Stop("stop before inference")

    monkeypatch.setattr(pipe, "_encode_prompt", _stop, raising=False)

    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    output_dir = tmp_path / "out"
    with pytest.raises(_Stop):
        pipe.run(
            input_image=image,
            output_dir=output_dir,
            save_input_copy=save_input_copy,
        )
    return output_dir


def test_default_mode_saves_input_png(tmp_path, monkeypatch):
    output_dir = _run_pipeline_until_encode(tmp_path, monkeypatch, save_input_copy=True)
    assert (output_dir / "input.png").is_file()


def test_storage_light_omits_input_png(tmp_path, monkeypatch):
    output_dir = _run_pipeline_until_encode(tmp_path, monkeypatch, save_input_copy=False)
    assert not (output_dir / "input.png").exists()


def test_run_default_signature_keeps_input_copy():
    import inspect

    signature = inspect.signature(RavenPipeline.run)
    assert signature.parameters["save_input_copy"].default is True


def test_view_and_final_saves_are_unconditional():
    """view_guided_output.png and final_*.png are never gated by save_input_copy."""
    source = Path(pipeline_raven.__file__).read_text(encoding="utf-8")
    # The only save guarded by save_input_copy is the input.png copy.
    assert 'if save_input_copy:' in source
    assert 'output_dir / "view_guided_output.png"' in source
    assert 'output_dir / final_name' in source
    # The retained outputs are emitted outside any save_input_copy branch.
    guarded_block = source.split("if save_input_copy:")[1]
    guarded_line = guarded_block.splitlines()[1]
    assert "input.png" in guarded_line
    assert "view_guided_output.png" not in guarded_line
    assert "final" not in guarded_line


def test_storage_mode_metadata_default_full_protocol():
    meta = formal_eval.storage_mode_metadata(
        method="TR", expected_count=100, storage_light=False, attack_clean_enabled=True
    )
    assert meta["storage_light"] is False
    assert meta["attack_clean_enabled"] is True
    assert meta["attacked_clean_count"] == 100
    assert meta["recalibrated_metrics_available"] is True
    assert meta["formal_protocol_complete"] is True
    assert meta["result_classification"] == "formal_complete"


def test_storage_mode_metadata_storage_light_tr():
    meta = formal_eval.storage_mode_metadata(
        method="TR", expected_count=100, storage_light=True, attack_clean_enabled=False
    )
    assert meta["storage_light"] is True
    assert meta["attack_clean_enabled"] is False
    assert meta["attacked_clean_count"] == 0
    assert meta["recalibrated_metrics_available"] is False
    assert meta["formal_protocol_complete"] is False
    assert meta["result_classification"] == "TR STORAGE-LIGHT / NO ATTACK-CLEAN"


def test_storage_mode_metadata_non_tr_default():
    meta = formal_eval.storage_mode_metadata(
        method="GS", expected_count=50, storage_light=False, attack_clean_enabled=True
    )
    # GS never had attacked-clean recalibration; still a complete formal result.
    assert meta["attacked_clean_count"] == 0
    assert meta["recalibrated_metrics_available"] is False
    assert meta["formal_protocol_complete"] is True
    assert meta["result_classification"] == "formal_complete"


def test_build_parser_defaults_and_flags():
    base = [
        "--dataset", "diffusiondb", "--method", "TR",
        "--source-metadata", "meta.csv", "--source-manifest", "manifest.json",
        "--output-root", "out", "--expected-count", "10", "--stage", "attack-watermarked",
    ]
    parser = formal_eval.build_parser()
    default_args = parser.parse_args(base)
    assert default_args.storage_light is False
    assert default_args.attack_clean_enabled is True

    light_args = parser.parse_args(base + ["--storage-light", "--attack-clean-enabled", "false"])
    assert light_args.storage_light is True
    assert light_args.attack_clean_enabled is False


@pytest.mark.parametrize(
    "storage_light, attack_clean_enabled",
    [
        (False, True),   # full formal
        (True, False),   # storage-light
    ],
)
def test_require_valid_storage_mode_accepts_legal_pairs(storage_light, attack_clean_enabled):
    # Should not raise.
    formal_eval.require_valid_storage_mode(storage_light, attack_clean_enabled)


@pytest.mark.parametrize(
    "storage_light, attack_clean_enabled",
    [
        (True, True),    # storage-light without disabling attack-clean
        (False, False),  # attack-clean disabled without storage-light
    ],
)
def test_require_valid_storage_mode_rejects_illegal_pairs(storage_light, attack_clean_enabled):
    with pytest.raises(RuntimeError, match="storage-light mode requires"):
        formal_eval.require_valid_storage_mode(storage_light, attack_clean_enabled)


def test_parse_bool_flag_rejects_garbage():
    with pytest.raises(Exception):
        formal_eval.parse_bool_flag("maybe")


def test_aggregate_no_clean_omits_recalibration():
    """Storage-light TR (no attacked-clean rows): the unified detector
    aggregate reports the original-clean threshold report and marks the
    recalibrated block unavailable — never fabricates recalibration."""
    from raven.detectors import ROW_STATUS_SCORED
    from raven.detectors.tr_detector import aggregate

    rows = []
    for i in range(100):
        rows.append({"status": ROW_STATUS_SCORED,
                     "evaluation_cohort": "original_clean",
                     "canonical_score": -(1.0 + i * 0.01)})
        rows.append({"status": ROW_STATUS_SCORED,
                     "evaluation_cohort": "original_watermarked",
                     "canonical_score": -(0.1 + i * 0.01)})
        rows.append({"status": ROW_STATUS_SCORED,
                     "evaluation_cohort": "attacked_watermarked",
                     "canonical_score": -(0.2 + i * 0.01)})

    agg = aggregate(rows)
    assert agg["tr_recalibrated"]["recalibrated_metrics_available"] is False
    # original-clean calibration and TPR at that threshold are still finite.
    summary = agg["detection_summary"]
    assert math.isfinite(summary["original_clean_threshold"])
    assert math.isfinite(summary["original_watermarked_tpr"])
    assert math.isfinite(summary["attacked_watermarked_tpr_at_original_threshold"])
    assert summary["threshold_comparison_operator"] == ">="
    assert agg["score_definition"] == "complex_l1_mean"
