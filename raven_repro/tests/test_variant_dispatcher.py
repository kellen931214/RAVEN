import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.wait_and_run_raven_protocol_variants import (
    JOBS,
    idle_candidates,
)
from raven.eval_protocol import load_formal_attack_config


def test_dispatcher_reuses_four_attacks_for_five_evaluations():
    assert len(JOBS) == 4
    color_results = sum(
        "validate" in job["formal_stages"] for job in JOBS.values()
    )
    no_color_results = sum(job["no_color"] for job in JOBS.values())
    assert color_results == 3
    assert no_color_results == 2
    assert color_results + no_color_results == 5


def test_idle_candidate_requires_no_process_low_utilization_and_free_memory():
    gpus = [
        {"index": 0, "uuid": "a", "free_mib": 24000, "total_mib": 24576, "utilization": 0},
        {"index": 1, "uuid": "b", "free_mib": 23000, "total_mib": 24576, "utilization": 0},
        {"index": 2, "uuid": "c", "free_mib": 22000, "total_mib": 24576, "utilization": 20},
        {"index": 3, "uuid": "d", "free_mib": 10000, "total_mib": 24576, "utilization": 0},
    ]
    result = idle_candidates(
        gpus,
        {"a"},
        min_free_mib=18000,
        max_utilization=5,
        locally_reserved=set(),
    )
    assert [gpu["index"] for gpu in result] == [1]


def test_all_tracked_dispatch_configs_are_formally_valid():
    root = Path(__file__).resolve().parents[2]
    names = {job["config"] for job in JOBS.values()}
    assert names == {
        "ddim_no_shift.json",
        "ddpm_reflection.json",
        "nfpa_bilinear_reflection.json",
        "nfpa_nearest_reflection_ddim_aligned.json",
    }
    for name in names:
        config = load_formal_attack_config(
            root / "experiments" / "raven_ablation_configs" / name
        )
        assert config["shift_magnitudes_image_px"] == list(range(24, 33))
        if config["shift_plan_mode"] != "zero":
            assert config["shift_plan_mode"] == "paper_random_independent_axes"



def test_allowed_gpu_filter_excludes_incompatible_devices():
    gpus = [
        {"index": 4, "uuid": "a", "free_mib": 24000, "total_mib": 24576, "utilization": 0},
        {"index": 6, "uuid": "blackwell", "free_mib": 96000, "total_mib": 98304, "utilization": 0},
    ]
    allowed = {4, 5, 8}
    filtered = [gpu for gpu in gpus if int(gpu["index"]) in allowed]
    result = idle_candidates(
        filtered, set(), min_free_mib=18000, max_utilization=5,
        locally_reserved=set(),
    )
    assert [gpu["index"] for gpu in result] == [4]
