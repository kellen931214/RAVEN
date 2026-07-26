"""Canonical data/output layout policy (migration 2026-07-26).

These tests are static/lightweight: they never launch a model, never write into
``outputs/``, and never regenerate images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.eval_protocol import (
    ATTACK_CLEAN_METHODS,
    CLEAN_DATA_ROOT,
    METHOD_DATA_ROOTS,
    METHOD_OUTPUT_ROOTS,
    assert_canonical_output_root,
    clean_data_dir,
    formal_output_root,
    formal_run_key,
    method_data_root,
    method_output_root,
    scratch_run_root,
    source_metadata_path,
)


# --------------------------------------------------------------------------- #
# Canonical roots
# --------------------------------------------------------------------------- #
def test_canonical_roots_exist_on_disk():
    for root in (CLEAN_DATA_ROOT, *METHOD_DATA_ROOTS.values(), *METHOD_OUTPUT_ROOTS.values()):
        assert root.is_dir(), f"missing canonical root: {root}"


def test_data_and_outputs_tops_contain_only_canonical_entries():
    assert {p.name for p in (REPO / "data").iterdir()} == {"clean", "tr", "gs"}
    assert {p.name for p in (REPO / "outputs").iterdir()} == {"tr", "gs"}


def test_method_roots_are_method_specific():
    assert method_data_root("TR") == REPO / "data" / "tr"
    assert method_data_root("gs") == REPO / "data" / "gs"
    assert method_output_root("TR") == REPO / "outputs" / "tr"
    assert method_output_root("gs") == REPO / "outputs" / "gs"


def test_unknown_method_fails_closed():
    for fn in (method_data_root, method_output_root):
        with pytest.raises(ValueError, match="canonical"):
            fn("NOPE")


# --------------------------------------------------------------------------- #
# Source data resolution
# --------------------------------------------------------------------------- #
def test_source_metadata_paths_resolve_to_real_cohorts():
    tr = source_metadata_path("TR", "diffusiondb")
    gs = source_metadata_path("GS", "gs_diffusiondb_1001_match_tr")
    assert tr == REPO / "data/tr/diffusiondb/TR/metadata.csv"
    assert gs == REPO / "data/gs/gs_diffusiondb_1001_match_tr/GS/metadata.csv"
    assert tr.is_file() and gs.is_file()


def test_clean_dirs_keep_tr_and_gs_cohorts_separate():
    tr_clean = clean_data_dir("diffusiondb")
    gs_clean = clean_data_dir("gs_diffusiondb_1001_match_tr", "GS")
    assert tr_clean != gs_clean
    assert tr_clean.is_dir() and gs_clean.is_dir()
    # Same filenames, different image sets -> must never share a directory.
    assert (tr_clean / "000000.png").is_file()
    assert (gs_clean / "000000.png").is_file()
    assert (tr_clean / "000000.png").read_bytes() != (gs_clean / "000000.png").read_bytes()


# --------------------------------------------------------------------------- #
# Output routing policy
# --------------------------------------------------------------------------- #
def test_formal_output_root_is_method_aware():
    tr = formal_output_root("TR", "diffusiondb", "formal", "abc123_def456")
    gs = formal_output_root("GS", "gs_diffusiondb_1001_match_tr", "formal", "abc123_def456")
    assert tr == REPO / "outputs/tr/diffusiondb/formal/abc123_def456"
    assert gs == REPO / "outputs/gs/gs_diffusiondb_1001_match_tr/formal/abc123_def456"


def test_formal_output_root_rejects_path_traversal():
    for bad in ("..", "a/b", ""):
        with pytest.raises(ValueError, match="invalid"):
            formal_output_root("TR", "diffusiondb", bad, "key")


def test_run_key_is_content_addressed_not_timestamped():
    key = formal_run_key("8a204ca1c95df8983ae6f0ad", "20c33008fba829b580cf5ab1")
    assert key == "8a204ca1c95d_20c33008fba8"
    # stable across calls, and derived only from content hashes
    assert key == formal_run_key("8a204ca1c95df8983ae6f0ad", "20c33008fba829b580cf5ab1")
    assert not any(ch.isspace() for ch in key)
    with pytest.raises(ValueError):
        formal_run_key("", "abc")


def test_output_root_guard_rejects_cross_method_and_non_canonical():
    ok = REPO / "outputs/gs/gs_diffusiondb_1001_match_tr/formal/x"
    assert assert_canonical_output_root(ok, "GS") == ok.resolve()
    with pytest.raises(ValueError, match="outside the canonical root"):
        assert_canonical_output_root(REPO / "outputs/tr/diffusiondb/x", "GS")
    with pytest.raises(ValueError, match="outside the canonical root"):
        assert_canonical_output_root(REPO / "outputs/scratch_run", "TR")


def test_gates_and_smoke_runs_go_to_tmp_not_outputs():
    root = scratch_run_root("GS", "gate10")
    try:
        assert root.is_dir()
        assert str(root).startswith("/tmp/")
        assert REPO / "outputs" not in root.parents
        assert "raven-gs-gate10" in root.name
    finally:
        root.rmdir()


# --------------------------------------------------------------------------- #
# Attack-clean is TR-only
# --------------------------------------------------------------------------- #
def test_attack_clean_is_tr_only():
    assert ATTACK_CLEAN_METHODS == frozenset({"TR"})
    assert "GS" not in ATTACK_CLEAN_METHODS


def test_non_tr_methods_get_no_attack_clean_and_no_input_png():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_raven_formal_eval", REPO / "experiments" / "run_raven_formal_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    gs = module.storage_mode_metadata(
        method="GS", expected_count=10, storage_light=True, attack_clean_enabled=False
    )
    assert gs["attack_clean_enabled"] is False
    assert gs["attacked_clean_count"] == 0
    assert gs["recalibrated_metrics_available"] is False
    # GS omitting attack-clean/input.png is the COMPLETE GS protocol, and must not be
    # mislabelled with the TR storage-light classification.
    assert gs["formal_protocol_complete"] is True
    assert gs["result_classification"] == "formal_complete"
    assert "TR" not in gs["result_classification"]

    # TR keeps both the full protocol and its explicit storage-light variant.
    tr_full = module.storage_mode_metadata(
        method="TR", expected_count=10, storage_light=False, attack_clean_enabled=True
    )
    assert tr_full["attacked_clean_count"] == 10
    assert tr_full["result_classification"] == "formal_complete"
    tr_light = module.storage_mode_metadata(
        method="TR", expected_count=10, storage_light=True, attack_clean_enabled=False
    )
    assert tr_light["attacked_clean_count"] == 0
    assert tr_light["result_classification"] == "TR STORAGE-LIGHT / NO ATTACK-CLEAN"


def test_formal_runner_output_root_is_optional():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_raven_formal_eval", REPO / "experiments" / "run_raven_formal_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    actions = {a.dest: a for a in module.build_parser()._actions}
    assert actions["output_root"].required is False
    assert actions["output_root"].default is None
    assert actions["variant"].default == "formal"
    assert actions["run_key"].default is None
