import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import color_alignment_waiter as waiter


def base_args(tmp_path: Path):
    generation_log = tmp_path / "chain.log"
    generation_log.write_text("chain waiting\n", encoding="utf-8")
    return SimpleNamespace(
        generation_pid=os.getpid(),
        generation_pid_file=None,
        generation_log=generation_log,
        completion_marker=tmp_path / "chain_state.json",
        completion_key="stage",
        completion_value="attacked_clean_complete",
        success_log_token="chain complete",
        source_p1_dir=tmp_path / "p1",
        source_nfpa_dir=tmp_path / "nfpa",
        expected_count=1,
        waiter_audit=tmp_path / "audit.json",
        python_executable=Path(sys.executable),
        repo_root=tmp_path,
        experiment_output_dir=tmp_path / "experiment",
        experiment_log=tmp_path / "experiment.log",
        experiment_pid_file=tmp_path / "experiment.pid",
        started_marker=tmp_path / ".experiment_started",
        completed_marker=tmp_path / ".experiment_completed",
        eval_repo=tmp_path / "eval",
        device="cuda",
        experiment_count=1,
        validation_count=1,
        min_cpu_mem_gb=1,
        warn_cpu_mem_gb=2,
        max_process_ram_gb=3,
    )


def test_waiter_does_not_launch_when_data_incomplete(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    monkeypatch.setattr(waiter, "launch_once", lambda _: pytest.fail("launched early"))
    assert waiter.one_iteration(args) == "waiting"


def test_waiter_fails_if_generation_exits_without_completion(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    args.generation_pid = 999999999
    monkeypatch.setattr(waiter, "launch_once", lambda _: pytest.fail("launched after failure"))
    with pytest.raises(RuntimeError, match="exited before completion marker"):
        waiter.one_iteration(args)


def test_waiter_fails_on_fatal_generation_log(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    args.generation_log.write_text("Traceback (most recent call last):\nboom\n", encoding="utf-8")
    monkeypatch.setattr(waiter, "launch_once", lambda _: pytest.fail("launched after fatal log"))
    with pytest.raises(RuntimeError, match="fatal error"):
        waiter.one_iteration(args)


def test_complete_inputs_are_audited_then_launched(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    args.generation_pid = None
    args.completion_marker.write_text(
        json.dumps({"stage": "attacked_clean_complete"}), encoding="utf-8"
    )
    args.generation_log.write_text("chain complete\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        waiter,
        "audit_sources",
        lambda p1, nfpa, count: {
            "counts": {"manifest": count, "shift_plan": count, "attacked_watermarked": count, "attacked_clean": count},
            "config_and_hash_audit": "passed",
        },
    )
    monkeypatch.setattr(waiter, "launch_once", lambda _: calls.append("launch") or 123)
    assert waiter.one_iteration(args) == "launched"
    assert calls == ["launch"]
    assert json.loads(args.waiter_audit.read_text())["config_and_hash_audit"] == "passed"


def test_launch_once_creates_markers_and_refuses_duplicate(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    args.generation_pid = None
    calls = []

    class FakeProcess:
        pid = os.getpid()
        returncode = None

        def poll(self):
            return None

    def fake_popen(*positional, **keywords):
        calls.append((positional, keywords))
        return FakeProcess()

    monkeypatch.setattr(waiter, "build_experiment_command", lambda _: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(waiter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(waiter.time, "sleep", lambda _: None)
    assert waiter.launch_once(args) == os.getpid()
    assert args.experiment_pid_file.read_text().strip() == str(os.getpid())
    assert args.started_marker.is_file()
    assert waiter.launch_once(args) == 0
    assert len(calls) == 1


def test_existing_completed_marker_refuses_duplicate_launch(tmp_path, monkeypatch):
    args = base_args(tmp_path)
    args.completed_marker.write_text("done\n", encoding="utf-8")
    monkeypatch.setattr(
        waiter.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("duplicate process launched"),
    )
    assert waiter.launch_once(args) == 0
