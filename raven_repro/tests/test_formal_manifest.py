import json

import pytest

from raven.eval_protocol import (
    load_and_validate_source_manifest,
    require_clean_git_worktree,
    sha256_path,
)
from scripts import build_verification_manifest as manifest


def test_snapshot_loader_rejects_hash_drift_and_duplicate_ids(tmp_path):
    snapshot = tmp_path / "batch.csv"
    snapshot.write_text("run_id,prompt\n1,a\n1,b\n")
    index = tmp_path / "index.jsonl"
    entry = {
        "batch_id": 0, "snapshot_path": str(snapshot), "snapshot_sha256": sha256_path(snapshot),
        "source_metadata_sha256": "source", "row_count": 2,
    }
    index.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="duplicate run_id"):
        manifest.load_snapshots(index)
    entry["snapshot_sha256"] = "bad"
    index.write_text(json.dumps(entry) + "\n")
    with pytest.raises(RuntimeError, match="snapshot file/hash mismatch"):
        manifest.load_snapshots(index)


def test_source_manifest_validates_live_file_and_companion_sha(tmp_path):
    source = tmp_path / "formal.py"
    source.write_text("value = 1\n")
    payload = {
        "git_head": "head",
        "files": [{
            "relative_path": "formal.py",
            "tracked_or_untracked": "untracked",
            "sha256": sha256_path(source),
            "size_bytes": source.stat().st_size,
            "git_head": "head",
            "git_dirty": True,
        }],
    }
    path = tmp_path / "formal_source_manifest.json"
    path.write_text(json.dumps(payload))
    path.with_suffix(".sha256").write_text(sha256_path(path) + "  formal_source_manifest.json\n")
    loaded, actual_sha = load_and_validate_source_manifest(path, repo_root=tmp_path)
    assert loaded == payload
    assert actual_sha == sha256_path(path)
    source.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source SHA drift"):
        load_and_validate_source_manifest(path, repo_root=tmp_path)



def test_source_manifest_head_is_informational_but_companion_and_files_are_strict(tmp_path):
    source = tmp_path / "formal.py"
    source.write_text("value = 1\n")
    payload = {
        "git_head": "intentionally-stale-build-head",
        "files": [{
            "relative_path": "formal.py",
            "sha256": sha256_path(source),
            "size_bytes": source.stat().st_size,
        }],
    }
    path = tmp_path / "formal_source_manifest.json"
    path.write_text(json.dumps(payload))
    path.with_suffix(".sha256").write_text(f"{sha256_path(path)}  {path.name}\n")
    loaded, _ = load_and_validate_source_manifest(path, repo_root=tmp_path)
    assert loaded["git_head"] == "intentionally-stale-build-head"
    path.with_suffix(".sha256").write_text(f"{'0' * 64}  {path.name}\n")
    with pytest.raises(RuntimeError, match="source manifest SHA mismatch"):
        load_and_validate_source_manifest(path, repo_root=tmp_path)


def test_formal_git_preflight_rejects_dirty_tree(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert require_clean_git_worktree(tmp_path) == head
    tracked.write_text("dirty\n")
    with pytest.raises(RuntimeError, match="requires a clean working tree"):
        require_clean_git_worktree(tmp_path)
