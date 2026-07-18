from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from scripts import run_diffusiondb_chain_after_clean as chain


class RecordingGuard:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def check(self, label: str) -> None:
        self.labels.append(label)


def test_completed_shard_metadata_resume_gate(tmp_path):
    metadata_dir = tmp_path / "data" / "watermarked" / "diffusiondb" / "TR"
    metadata_dir.mkdir(parents=True)
    for shard_index, run_ids in ((0, [0, 2, 4]), (1, [1, 3])):
        path = metadata_dir / f"metadata.shard-{shard_index:03d}-of-002.csv"
        path.write_text(
            "run_id,prompt\n"
            + "".join(f"{run_id},prompt-{run_id}\n" for run_id in run_ids),
            encoding="utf-8",
        )

    assert chain.paired_shard_metadata_complete(tmp_path, 5, 2)
    with (
        metadata_dir / "metadata.shard-001-of-002.csv"
    ).open("a", encoding="utf-8") as handle:
        handle.write("5,unexpected\n")
    assert not chain.paired_shard_metadata_complete(tmp_path, 5, 2)


def test_parallel_attack_stages_use_distinct_gpu_uuids(tmp_path, monkeypatch):
    selected = [
        {"index": "2", "uuid": "GPU-test-a", "name": "A", "free_mib": 24000},
        {"index": "3", "uuid": "GPU-test-b", "name": "B", "free_mib": 24000},
    ]
    requested_counts: list[int] = []

    def fake_wait_for_gpus(requested, count, poll_seconds, min_free_mib):
        requested_counts.append(count)
        return selected[:count]

    monkeypatch.setattr(chain, "wait_for_gpus", fake_wait_for_gpus)
    args = SimpleNamespace(
        visible_gpu="auto",
        gpu_poll_seconds=1,
        min_free_gpu_mib=20000,
    )
    state = {"completed_stages": []}
    guard = RecordingGuard()
    command = [
        sys.executable,
        "-c",
        "import os, time; print(os.environ['CUDA_VISIBLE_DEVICES']); time.sleep(0.05)",
    ]

    chain.run_parallel_gpu_stages(
        [("attacked_watermarked", command), ("attacked_clean", command)],
        args=args,
        root=tmp_path,
        state=state,
        guard=guard,
        env={},
    )

    assert requested_counts == [2]
    assert set(state["completed_stages"]) == {
        "attacked_watermarked",
        "attacked_clean",
    }
    assert state["attack_workers"]["attacked_watermarked"]["status"] == "completed"
    assert state["attack_workers"]["attacked_clean"]["status"] == "completed"
    assert state["stage_gpus"]["attacked_watermarked"]["uuid"] == "GPU-test-a"
    assert state["stage_gpus"]["attacked_clean"]["uuid"] == "GPU-test-b"
    assert "GPU-test-a" in (tmp_path / "logs" / "attacked_watermarked.log").read_text()
    assert "GPU-test-b" in (tmp_path / "logs" / "attacked_clean.log").read_text()
    persisted = json.loads((tmp_path / "run_state.json").read_text())
    assert set(persisted["completed_stages"]) == {
        "attacked_watermarked",
        "attacked_clean",
    }
    assert guard.labels == [
        "before parallel clean/watermarked attacks",
        "after parallel clean/watermarked attacks",
    ]


def test_parallel_attack_resume_only_requests_missing_gpu(tmp_path, monkeypatch):
    requested_counts: list[int] = []

    def fake_wait_for_gpus(requested, count, poll_seconds, min_free_mib):
        requested_counts.append(count)
        return [
            {"index": "7", "uuid": "GPU-test-only", "name": "A", "free_mib": 24000}
        ]

    monkeypatch.setattr(chain, "wait_for_gpus", fake_wait_for_gpus)
    args = SimpleNamespace(
        visible_gpu="auto",
        gpu_poll_seconds=1,
        min_free_gpu_mib=20000,
    )
    state = {"completed_stages": ["attacked_watermarked"]}
    guard = RecordingGuard()

    chain.run_parallel_gpu_stages(
        [
            ("attacked_watermarked", [sys.executable, "-c", "raise SystemExit(99)"]),
            ("attacked_clean", [sys.executable, "-c", "print('clean-only')"]),
        ],
        args=args,
        root=tmp_path,
        state=state,
        guard=guard,
        env={},
    )

    assert requested_counts == [1]
    assert state["completed_stages"] == ["attacked_watermarked", "attacked_clean"]
    assert set(state["attack_workers"]) == {"attacked_clean"}
