import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from raven.eval_protocol import canonical_json_hash, sha256_path
from experiments.run_raven_aligned_color_eval import (
    build_aligned_records,
    configure_single_gpu,
    paired_effective_source_flow,
    select_expected_run_ids,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (512, 512), color).save(path)
    return sha256_path(path)


def test_aligned_adapter_is_declared():
    from raven.color_transfer import PAPER_EXACT_TWO_STAGE_ALIGNED

    assert PAPER_EXACT_TWO_STAGE_ALIGNED == "paper_exact_two_stage_aligned"


def test_aligned_adapter_isolates_the_requested_physical_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    configure_single_gpu(8)
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "8"


def test_paired_effective_flow_ignores_planned_flow():
    wm = {
        "planned_flow_dx_image_px": 27.0,
        "planned_flow_dy_image_px": -29.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
    }
    clean = {
        "planned_flow_dx_image_px": 999.0,
        "planned_flow_dy_image_px": 999.0,
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
    }
    assert paired_effective_source_flow(wm, clean, "0") == (24.0, -32.0)


def test_paired_effective_flow_rejects_clean_watermarked_drift():
    wm = {
        "effective_source_flow_dx_image_px": 24.0,
        "effective_source_flow_dy_image_px": -32.0,
    }
    clean = {
        "effective_source_flow_dx_image_px": 32.0,
        "effective_source_flow_dy_image_px": -32.0,
    }
    import pytest

    with pytest.raises(RuntimeError, match="attacked pair effective_source_flow"):
        paired_effective_source_flow(wm, clean, "0")


def test_gate_cohort_selection_validates_full_coverage_before_subsetting():
    source = {str(index): {} for index in range(12)}
    watermarked = dict(source)
    clean = dict(source)

    assert select_expected_run_ids(source, watermarked, clean, 2) == {"0", "1"}
    assert select_expected_run_ids(source, watermarked, clean, 10) == {
        str(index) for index in range(10)
    }


def test_gate_cohort_selection_rejects_coverage_drift_and_invalid_count():
    import pytest

    source = {"0": {}, "1": {}}
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        select_expected_run_ids(source, {"0": {}}, dict(source), 1)
    with pytest.raises(RuntimeError, match="between 1 and 2"):
        select_expected_run_ids(source, dict(source), dict(source), 3)
