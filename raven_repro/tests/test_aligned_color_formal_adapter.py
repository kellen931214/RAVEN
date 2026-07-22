import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from raven.eval_protocol import canonical_json_hash, sha256_path
from experiments.run_raven_aligned_color_eval import (
    build_aligned_records,
    configure_single_gpu,
    create_evaluation_snapshot,
    paired_effective_source_flow,
    select_expected_run_ids,
    write_variant_attack_config,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (512, 512), color).save(path)
    return sha256_path(path)


def test_color_transfer_adapter_declares_paper_exact_and_aligned_modes():
    from raven.color_transfer import (
        PAPER_EXACT_TWO_STAGE,
        PAPER_EXACT_TWO_STAGE_ALIGNED,
    )

    assert PAPER_EXACT_TWO_STAGE == "paper_exact_two_stage"
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


def test_evaluation_snapshot_contains_exact_selected_run_ids(tmp_path):
    formal_root = tmp_path / "formal"
    snapshots = formal_root / "snapshots"
    snapshots.mkdir(parents=True)
    source_snapshot = snapshots / "source.csv"
    rows = [
        {"run_id": str(index), "dataset": "diffusiondb", "method": "TR"}
        for index in range(4)
    ]
    with source_snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_sha = sha256_path(source_snapshot)
    source_index = snapshots / "snapshot_index.jsonl"
    source_index.write_text(
        json.dumps(
            {
                "batch_id": 0,
                "row_count": 4,
                "snapshot_path": str(source_snapshot),
                "snapshot_sha256": source_sha,
                "source_metadata_path": "/immutable/metadata.csv",
                "source_metadata_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index_path, snapshot_sha, index_sha = create_evaluation_snapshot(
        formal_root, tmp_path / "output", {"0", "2"}
    )

    with (index_path.parent / "cohort.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        selected = list(csv.DictReader(handle))
    entry = json.loads(index_path.read_text(encoding="utf-8"))
    assert [row["run_id"] for row in selected] == ["0", "2"]
    assert entry["row_count"] == 2
    assert entry["snapshot_sha256"] == snapshot_sha
    assert sha256_path(index_path) == index_sha



def test_variant_attack_config_is_persisted_for_strict_manifest_validation(tmp_path):
    from raven.eval_protocol import (
        FORMAL_ATTACK_CONFIG,
        formal_attack_config_hash,
        normalize_formal_attack_config,
    )

    config = normalize_formal_attack_config({
        **FORMAL_ATTACK_CONFIG,
        "color_transfer_mode": "paper_exact_two_stage",
        "variant_name": "paper_exact_test",
    })
    config_hash = formal_attack_config_hash(config)
    records = [
        {"formal_attack_config": config, "attack_config_hash": config_hash},
        {"formal_attack_config": config, "attack_config_hash": config_hash},
    ]
    path = write_variant_attack_config(tmp_path, records, config_hash)
    assert json.loads(path.read_text()) == config
    assert sha256_path(path)


def test_variant_attack_config_rejects_hash_drift(tmp_path):
    import pytest
    from raven.eval_protocol import FORMAL_ATTACK_CONFIG, normalize_formal_attack_config

    config = normalize_formal_attack_config({
        **FORMAL_ATTACK_CONFIG,
        "color_transfer_mode": "paper_exact_two_stage",
        "variant_name": "paper_exact_test",
    })
    records = [{"formal_attack_config": config, "attack_config_hash": "wrong"}]
    with pytest.raises(RuntimeError, match="variant config hash mismatch"):
        write_variant_attack_config(tmp_path, records, "wrong")
