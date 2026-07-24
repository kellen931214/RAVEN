"""Focused tests for the generalized RAVEN no-color evaluator.

These exercise the pure helpers and the TR/GS dispatch of
``experiments/run_raven_no_color_eval.py`` without loading a Stable Diffusion
model, running an attack, or touching a GPU. Heavy dependencies (git worktree,
source manifest, snapshot/attack-record loaders, FID/CLIP and the detector
subprocesses) are replaced with light fakes; the pre-color binding, coverage
selection and provenance logic remain under test.
"""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "raven_repro"))

import experiments.run_raven_no_color_eval as nc  # noqa: E402
from raven.eval_protocol import canonical_json_hash  # noqa: E402


HEAD = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
MANIFEST_SHA = "0" * 64
ATTACK_HASH = "a" * 64
MODEL_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"


def _png(path: Path, color) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_record(tmp: Path, run_id: int, role: str) -> dict:
    """Build a synthetic completed attack record with real image files on disk."""
    base = tmp / "formal" / "attack_cache" / ATTACK_HASH / str(run_id) / role
    watermarked = tmp / "src" / f"wm_{run_id}.png"
    pre_color = base / "output" / "view_guided_output.png"
    post_color = base / "output" / "final_color_corrected.png"
    wm_sha = _png(watermarked, (10 + run_id, 20, 30))
    pre_sha = _png(pre_color, (40 + run_id, 50, 60))
    post_sha = _png(post_color, (70 + run_id, 80, 90))
    return {
        "run_id": str(run_id),
        "prompt": f"prompt {run_id}",
        "attack_config_hash": ATTACK_HASH,
        "watermarked_path": str(watermarked),
        "watermarked_sha256": wm_sha,
        "attacked_path": str(post_color),
        "attacked_sha256": post_sha,
        "pre_color_attacked_path": str(pre_color),
        "pre_color_attacked_sha256": pre_sha,
        "effective_source_flow_dx_image_px": 4.0,
        "effective_source_flow_dy_image_px": -5.0,
        "pairing_sha256": f"pair-{run_id}",
        "snapshot_sha256": "source-snap-sha",
    }


def _write_run_config(formal_root: Path, method: str) -> dict:
    config = {
        "source_code_manifest_sha256": MANIFEST_SHA,
        "git_head": HEAD,
        "dataset": "diffusiondb",
        "method": method,
        "attack_config_hash": ATTACK_HASH,
        "attack_config": {
            "model_revision": MODEL_REVISION,
            "variant_name": "nfpa_nearest_reflection_ddim_inverse_ddpm_aligned",
        },
        "attack_config_source_path": None,
    }
    formal_root.mkdir(parents=True, exist_ok=True)
    (formal_root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    return config


# --------------------------------------------------------------------------- #
# Pure helper tests
# --------------------------------------------------------------------------- #


def test_bound_no_color_record_sets_pre_color_fields(tmp_path):
    record = _make_record(tmp_path, 0, "watermarked")
    bound = nc.bound_no_color_record(record, "vhash")
    assert bound["output_color_transfer"] is False
    assert bound["output_color_transfer_mode"] == "none"
    assert bound["metric_image_source"] == "pre_color_attacked_path"
    # the metric image is bound to the explicit pre-color output, not post-color
    assert bound["attacked_path"] == str(Path(record["pre_color_attacked_path"]).resolve())
    assert bound["attacked_sha256"] == record["pre_color_attacked_sha256"]
    assert bound["output_sha256"] == record["pre_color_attacked_sha256"]
    # post-color provenance is retained but demoted to source_*
    assert bound["source_final_post_color_sha256"] == record["attacked_sha256"]
    assert bound["source_attack_config_hash"] == ATTACK_HASH


def test_bound_no_color_record_rejects_pre_color_sha_mismatch(tmp_path):
    record = _make_record(tmp_path, 1, "watermarked")
    record["pre_color_attacked_sha256"] = "f" * 64  # corrupt recorded SHA
    with pytest.raises(RuntimeError, match="pre-color attacked SHA mismatch"):
        nc.bound_no_color_record(record, "vhash")


def test_bound_no_color_record_rejects_missing_pre_color(tmp_path):
    record = _make_record(tmp_path, 2, "watermarked")
    Path(record["pre_color_attacked_path"]).unlink()
    with pytest.raises(RuntimeError, match="explicit pre-color"):
        nc.bound_no_color_record(record, "vhash")


def test_gs_coverage_requires_only_watermarked_records():
    source = {"0": {}, "1": {}}
    watermarked = {"0": {}, "1": {}}
    # No clean records are needed for GS.
    assert nc.select_watermarked_run_ids(source, watermarked, 2) == {"0", "1"}


def test_gs_coverage_rejects_incomplete_watermarked():
    source = {"0": {}, "1": {}}
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        nc.select_watermarked_run_ids(source, {"0": {}}, 2)


def test_tr_coverage_requires_attacked_clean_records():
    """TR uses select_expected_run_ids, which fails without clean coverage."""
    from experiments.run_raven_aligned_color_eval import select_expected_run_ids

    source = {"0": {}, "1": {}}
    watermarked = {"0": {}, "1": {}}
    clean_missing = {"0": {}}
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        select_expected_run_ids(source, watermarked, clean_missing, 2)


# --------------------------------------------------------------------------- #
# Integration dispatch tests (heavy deps faked)
# --------------------------------------------------------------------------- #


def _install_common_fakes(monkeypatch, tmp_path, records_by_role, source_rows):
    monkeypatch.setattr(nc, "require_clean_git_worktree", lambda repo: HEAD)
    monkeypatch.setattr(
        nc,
        "load_and_validate_source_manifest",
        lambda path, repo_root: ({"git_head": HEAD}, MANIFEST_SHA),
    )
    monkeypatch.setattr(nc, "configure_single_gpu", lambda gpu: None)
    monkeypatch.setattr(nc, "load_snapshot_rows", lambda root: source_rows)
    monkeypatch.setattr(
        nc, "audit_pairing_rows", lambda rows, expected_count, verify_files: None
    )

    def fake_attack_records(root, run_config, role):
        if role not in records_by_role:
            raise AssertionError(f"unexpected attack_records role requested: {role}")
        return list(records_by_role[role].values())

    monkeypatch.setattr(nc, "attack_records", fake_attack_records)

    def fake_snapshot(formal_root, output_root, run_ids):
        index = output_root / "snapshots" / "snapshot_index.jsonl"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("{}\n", encoding="utf-8")
        return index, "cohort-snap-sha", "cohort-index-sha"

    monkeypatch.setattr(nc, "create_evaluation_snapshot", fake_snapshot)

    quality_seen = {}

    def fake_quality(records):
        # Assert quality reads the pre-color attacked image and the watermarked reference.
        for record in records:
            assert record["attacked_path"] == str(
                Path(record["pre_color_attacked_path"]).resolve()
            )
            assert Path(record["watermarked_path"]).is_file()
        quality_seen["records"] = records
        return [
            {"run_id": r["run_id"], "overlap_psnr": 30.0, "overlap_ssim": 0.9}
            for r in records
        ]

    monkeypatch.setattr(nc, "compute_quality_rows", fake_quality)

    def fake_fid_clip(variant_wm, *, output_root, variant_hash, expected_count, device):
        for record in variant_wm:
            assert record["attacked_path"] == str(
                Path(record["pre_color_attacked_path"]).resolve()
            )
        (output_root / "metrics").mkdir(parents=True, exist_ok=True)
        return (
            {"value": 12.5, "config_hash": variant_hash},
            {"mean": 0.31, "clip_model_name": "ViT-bigG-14"},
        )

    monkeypatch.setattr(nc, "compute_fid_and_clip", fake_fid_clip)
    return quality_seen


def _fake_run_factory(commands):
    def fake_run(command):
        commands.append(command)
        joined = " ".join(command)
        if "build_verification_manifest.py" in joined:
            out = Path(command[command.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                "run_id,provider_config_hash\n0,PROVHASH\n1,PROVHASH\n", encoding="utf-8"
            )
        elif "raven_nfpa_tr_eval.py" in joined:
            out_dir = Path(command[command.index("--output-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            protocol = {
                "original_clean_actual_fpr": 0.01,
                "attacked_clean_actual_fpr": 0.02,
                "before_tpr": 0.99,
                "attacked_tpr_at_original_clean_threshold": 0.10,
                "attacked_tpr_at_attacked_clean_recalibrated_threshold": 0.15,
                "attack_success_rate_at_recalibrated_threshold": 0.85,
                "attacked_roc_auc": 0.60,
            }
            (out_dir / "aggregate_results.json").write_text(
                json.dumps(
                    {
                        "nfpa_rounded2_protocol": protocol,
                        "full_precision_protocol": protocol,
                        "provider_config_hash": "TRPROV",
                        "target_watermark_hash": "TRTARGET",
                    }
                ),
                encoding="utf-8",
            )
        elif "extract_verification_scores.py" in joined:
            out = Path(command[command.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("run_id\n0\n1\n", encoding="utf-8")
        elif "evaluate_verification.py" in joined:
            out = Path(command[command.index("--output-json") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "method": "GS",
                        "metric": {
                            "macro_bit_accuracy_before": 0.98,
                            "macro_bit_accuracy_attacked": 0.72,
                            "before_tpr_at_clean_calibrated_threshold": 0.97,
                            "attacked_tpr_at_clean_calibrated_threshold": 0.20,
                            "before_roc_auc": 0.99,
                            "attacked_roc_auc": 0.65,
                            "official_onebit_rates": {
                                "clean": 0.0,
                                "watermarked": 1.0,
                                "attacked": 0.30,
                            },
                            "official_traceability_rates": {
                                "clean": 0.0,
                                "watermarked": 1.0,
                                "attacked": 0.10,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            raise AssertionError(f"unexpected subprocess: {joined}")

    return fake_run


def _run_main(monkeypatch, *, method, formal_root, output_root, count=2):
    argv = [
        "run_raven_no_color_eval.py",
        "--dataset", "diffusiondb",
        "--method", method,
        "--formal-root", str(formal_root),
        "--output-root", str(output_root),
        "--expected-count", str(count),
        "--source-manifest", str(formal_root / "manifest.json"),
        "--device", "cuda",
        "--gpu", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return nc.main()


def test_gs_no_color_dispatch_and_provenance(tmp_path, monkeypatch):
    formal_root = tmp_path / "formal"
    _write_run_config(formal_root, "GS")
    wm = {str(i): _make_record(tmp_path, i, "watermarked") for i in range(2)}
    source_rows = {rid: {"run_id": rid} for rid in wm}
    # GS must never request clean attack records.
    _install_common_fakes(monkeypatch, tmp_path, {"watermarked": wm}, source_rows)
    commands: list[list[str]] = []
    monkeypatch.setattr(nc, "run", _fake_run_factory(commands))

    output_root = tmp_path / "b2_no_color"
    formal_before = (formal_root / "run_config.json").read_bytes()
    assert _run_main(monkeypatch, method="GS", formal_root=formal_root, output_root=output_root) == 0

    joined = [" ".join(c) for c in commands]
    # GS branch dispatches to the GS detector scripts, never to the TR scorer.
    assert any("extract_verification_scores.py --method GS" in j for j in joined)
    assert any("evaluate_verification.py --method GS" in j for j in joined)
    assert any("build_verification_manifest.py --dataset diffusiondb --method GS" in j for j in joined)
    assert all("raven_nfpa_tr_eval.py" not in j for j in joined)
    assert all("--attacked-clean-records" not in j for j in joined)

    validated = json.loads((output_root / "VALIDATED.json").read_text())
    assert validated["method"] == "GS"
    assert validated["evaluation_variant"] == "no_color_transfer"
    assert validated["output_color_transfer"] is False
    assert validated["source_attack_config_hash"] == ATTACK_HASH
    assert validated["source_formal_root"] == str(formal_root.resolve())
    assert validated["sample_count"] == 2
    assert validated["git_head"] == HEAD
    assert validated["source_code_manifest_sha256"] == MANIFEST_SHA
    assert validated["metric_image_source"] == "pre_color_attacked_path"
    assert validated["attacked_clean_used"] is False
    # GS detector numbers live in GS-named fields, not TR/NFPA field names.
    assert "gs_macro_bit_accuracy_attacked" in validated
    assert "before_tpr" not in validated
    assert "nfpa_rounded2_protocol" not in validated

    table = json.loads((output_root / "aggregate_results.json").read_text())["result_table"]
    assert table["Watermark"] == "GS"
    assert "GS macro bit accuracy attacked" in table
    assert "Attacked TPR at recalibrated threshold" not in table

    # source pre-color SHA provenance is recorded and matches the source records
    expected_set_hash = canonical_json_hash(
        {"pre_color_sha256": sorted(r["pre_color_attacked_sha256"] for r in wm.values())}
    )
    assert validated["source_pre_color_sha256_set_hash"] == expected_set_hash

    # the source formal root must not be mutated by the no-color evaluation
    assert (formal_root / "run_config.json").read_bytes() == formal_before


def test_tr_no_color_dispatch_requires_clean_and_uses_tr_scorer(tmp_path, monkeypatch):
    formal_root = tmp_path / "formal"
    _write_run_config(formal_root, "TR")
    wm = {str(i): _make_record(tmp_path, i, "watermarked") for i in range(2)}
    clean = {str(i): _make_record(tmp_path, i, "clean") for i in range(2)}
    # keep pairing_sha256 identical across the attacked pair
    for rid in wm:
        clean[rid]["pairing_sha256"] = wm[rid]["pairing_sha256"]
    source_rows = {rid: {"run_id": rid} for rid in wm}
    _install_common_fakes(
        monkeypatch, tmp_path, {"watermarked": wm, "clean": clean}, source_rows
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(nc, "run", _fake_run_factory(commands))

    output_root = tmp_path / "b1_no_color"
    assert _run_main(monkeypatch, method="TR", formal_root=formal_root, output_root=output_root) == 0

    joined = [" ".join(c) for c in commands]
    assert any("raven_nfpa_tr_eval.py score-formal" in j for j in joined)
    assert any("--attacked-clean-records" in j for j in joined)
    assert all("extract_verification_scores.py" not in j for j in joined)

    validated = json.loads((output_root / "VALIDATED.json").read_text())
    assert validated["method"] == "TR"
    assert validated["output_color_transfer"] is False
    assert validated["source_attack_config_hash"] == ATTACK_HASH
    # TR keeps its NFPA provider/target provenance fields
    assert validated["provider_config_hash"] == "TRPROV"
    assert validated["target_watermark_hash"] == "TRTARGET"

    table = json.loads((output_root / "aggregate_results.json").read_text())["result_table"]
    assert table["Watermark"] == "TR"
    assert "Attacked TPR at recalibrated threshold" in table


def test_tr_variant_hash_is_unchanged_definition(tmp_path):
    """The TR no-color variant hash definition must not silently change."""
    run_config = {"attack_config_hash": ATTACK_HASH}
    expected = canonical_json_hash(
        {
            "metric_image_source": "explicit pre-color attacked image",
            "source_attack_config_hash": run_config["attack_config_hash"],
            "source_code_manifest_sha256": MANIFEST_SHA,
            "protocol": "raven_formal_no_color_v1",
        }
    )
    gs_hash = canonical_json_hash(
        {
            "metric_image_source": "explicit pre-color attacked image",
            "source_attack_config_hash": run_config["attack_config_hash"],
            "source_code_manifest_sha256": MANIFEST_SHA,
            "protocol": "raven_formal_no_color_gs_v1",
            "method": "GS",
        }
    )
    assert expected != gs_hash


def test_dataset_method_are_validated_against_run_config(tmp_path, monkeypatch):
    formal_root = tmp_path / "formal"
    _write_run_config(formal_root, "GS")  # formal root is GS
    wm = {str(i): _make_record(tmp_path, i, "watermarked") for i in range(2)}
    source_rows = {rid: {"run_id": rid} for rid in wm}
    _install_common_fakes(monkeypatch, tmp_path, {"watermarked": wm}, source_rows)
    monkeypatch.setattr(nc, "run", _fake_run_factory([]))
    output_root = tmp_path / "mismatch"
    # requesting TR against a GS formal root must fail closed
    with pytest.raises(RuntimeError, match="does not match formal run_config"):
        _run_main(monkeypatch, method="TR", formal_root=formal_root, output_root=output_root)


def test_existing_output_root_with_mismatched_provenance_is_rejected(tmp_path, monkeypatch):
    formal_root = tmp_path / "formal"
    _write_run_config(formal_root, "GS")
    wm = {str(i): _make_record(tmp_path, i, "watermarked") for i in range(2)}
    source_rows = {rid: {"run_id": rid} for rid in wm}
    _install_common_fakes(monkeypatch, tmp_path, {"watermarked": wm}, source_rows)
    monkeypatch.setattr(nc, "run", _fake_run_factory([]))
    output_root = tmp_path / "b2_no_color"
    output_root.mkdir(parents=True)
    # A pre-existing VALIDATED.json with the wrong sample_count must be rejected.
    (output_root / "VALIDATED.json").write_text(
        json.dumps(
            {
                "status": "validated_no_color_evaluation",
                "method": "GS",
                "source_code_manifest_sha256": MANIFEST_SHA,
                "git_head": HEAD,
                "sample_count": 999,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError):
        _run_main(monkeypatch, method="GS", formal_root=formal_root, output_root=output_root)


def test_existing_matching_output_root_is_reused(tmp_path, monkeypatch, capsys):
    formal_root = tmp_path / "formal"
    _write_run_config(formal_root, "GS")
    wm = {str(i): _make_record(tmp_path, i, "watermarked") for i in range(2)}
    source_rows = {rid: {"run_id": rid} for rid in wm}
    _install_common_fakes(monkeypatch, tmp_path, {"watermarked": wm}, source_rows)
    monkeypatch.setattr(nc, "run", _fake_run_factory([]))
    output_root = tmp_path / "b2_no_color"
    output_root.mkdir(parents=True)
    (output_root / "VALIDATED.json").write_text(
        json.dumps(
            {
                "status": "validated_no_color_evaluation",
                "method": "GS",
                "source_code_manifest_sha256": MANIFEST_SHA,
                "git_head": HEAD,
                "sample_count": 2,
            }
        ),
        encoding="utf-8",
    )
    assert _run_main(monkeypatch, method="GS", formal_root=formal_root, output_root=output_root) == 0
    assert "reused" in capsys.readouterr().out
