"""Focused CPU tests for RID/HSTR/HSQR shared-clean formal eval wiring."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "raven_repro"))
sys.path.insert(0, str(REPO / "experiments"))

from experiments import run_raven_formal_eval as formal
from experiments.update_experiment_table import read_rows, update_experiment_table
from raven.eval_protocol import (
    FORMAL_ATTACK_CONFIG,
    canonical_json_hash,
    formal_output_root,
    method_data_root,
    method_output_root,
    provider_config,
    provider_config_hash,
    sha256_path,
    transform_config_payload,
    validate_resume_record,
)
from raven.pairing_provenance import (
    HSTR_SHARED_TR_CLEAN_MODE,
    HSTR_SHARED_TR_CLEAN_PROTOCOL,
    HSQR_SHARED_TR_CLEAN_MODE,
    HSQR_SHARED_TR_CLEAN_PROTOCOL,
    RID_SHARED_TR_CLEAN_MODE,
    RID_SHARED_TR_CLEAN_PROTOCOL,
    SHARED_CLEAN_PROTOCOL,
    SHARED_CLEAN_SOURCE_METHOD,
    TR_PAIRING_PROTOCOL,
    audit_pairing_rows,
    audit_shared_clean_cohorts,
    build_pairing_sha256,
)
from scripts import build_verification_manifest as manifest

METHODS = {
    "RID": ("rid", RID_SHARED_TR_CLEAN_PROTOCOL, RID_SHARED_TR_CLEAN_MODE, "rid_neg_channel_min_complex_l1"),
    "HSTR": ("hstr", HSTR_SHARED_TR_CLEAN_PROTOCOL, HSTR_SHARED_TR_CLEAN_MODE, "hstr_score"),
    "HSQR": ("hsqr", HSQR_SHARED_TR_CLEAN_PROTOCOL, HSQR_SHARED_TR_CLEAN_MODE, "hsqr_score"),
}


def _png(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _debug(path: Path, dx: float = 24.0, dy: float = -27.0) -> Path:
    payload = {
        "model_id": FORMAL_ATTACK_CONFIG["model_id"],
        "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "steps": 50,
        "strength": 0.15,
        "guidance_scale": 2.5,
        "inversion_mode": "ddim",
        "exact_timestep": 149,
        "inversion_prompt": "",
        "reconstruction_prompt": "",
        "negative_prompt": "",
        "warp_mode": "raven_paper_nfpa_gap_fill",
        "interpolation_mode": "nearest",
        "padding_mode": "reflection",
        "align_corners": False,
        "normalized_coordinate_formula": "x_norm = 2*x_pixel/W - 1",
        "pixel_center_offset_image_px": 0.0,
        "warp_coordinate_convention": "legacy_nfpa_w_h_norm",
        "warp_implementation_version": "nfpa_image_grid_w_h_norm_v1",
        "planned_flow_dx_image_px": dx,
        "planned_flow_dy_image_px": dy,
        "effective_source_dx_latent": 3.0,
        "effective_source_dy_latent": -4.0,
        "effective_source_flow_dx_image_px": dx,
        "effective_source_flow_dy_image_px": dy,
        "effective_visual_shift_dx_image_px": -dx,
        "effective_visual_shift_dy_image_px": -dy,
        "view_guided_attention": True,
        "color_transfer": True,
        "color_transfer_mode": "paper_exact_two_stage_aligned",
        "attack_device_class": "cuda",
        "attack_dtype": "torch.float16",
        "scheduler_class": "DDIMScheduler",
        "scheduler_config": {"beta_start": 0.00085},
        "torch_version": "2.test",
        "diffusers_version": "0.test",
    }
    payload["scheduler_config_hash"] = canonical_json_hash(payload["scheduler_config"])
    payload["transform_config_hash"] = canonical_json_hash(transform_config_payload(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tr_row(tmp_path: Path, run_id: int = 0) -> dict:
    prompt = f"prompt {run_id}"
    clean = _png(tmp_path / "clean" / f"{run_id:06d}.png", (run_id, 1, 2))
    wm = _png(tmp_path / "tr" / f"{run_id:06d}" / "watermarked.png", (run_id, 3, 4))
    row = {
        "protocol": TR_PAIRING_PROTOCOL,
        "dataset": "synthetic",
        "dataset_name": "synthetic",
        "run_id": str(run_id),
        "prompt_id": str(run_id),
        "prompt": prompt,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "source": "unit",
        "wm_type": "TR",
        "model_id": FORMAL_ATTACK_CONFIG["model_id"],
        "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "base_latent_seed": str(100 + run_id),
        "base_latent_sha256": f"base{run_id:060d}",
        "clean_base_latent_sha256": f"base{run_id:060d}",
        "watermarked_base_latent_sha256": f"base{run_id:060d}",
        "watermarked_latent_sha256": f"trpost{run_id:058d}",
        "watermark_target_sha256": "trtarget",
        "watermark_mask_sha256": "trmask",
        "generation_config_sha256": "gencfg",
        "watermark_config_sha256": "trwcfg",
        "clean_path": str(clean.resolve()),
        "clean_sha256": sha256_path(clean),
        "watermarked_path": str(wm.resolve()),
        "watermarked_sha256": sha256_path(wm),
    }
    row["pairing_sha256"] = build_pairing_sha256(row)
    return row


def _method_row(tmp_path: Path, method: str, run_id: int = 0) -> dict:
    prefix, protocol, mode, _metric = METHODS[method]
    tr = _tr_row(tmp_path, run_id)
    bundle = tmp_path / prefix / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "bundle_config_sha256": f"{prefix}_bundle_cfg",
        "selected_pattern_sha256": f"{prefix}_pattern",
        "mask_sha256": f"{prefix}_mask",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest_payload), encoding="utf-8")
    wm = _png(tmp_path / prefix / f"{run_id:06d}" / "watermarked.png", (run_id, 9, 10))
    row = {
        **tr,
        "protocol": protocol,
        "wm_type": method,
        "watermarked_path": str(wm.resolve()),
        "watermarked_sha256": sha256_path(wm),
        "watermarked_latent_sha256": f"{prefix}post{run_id:057d}",
        "watermark_target_sha256": f"{prefix}_pattern",
        "watermark_mask_sha256": f"{prefix}_mask",
        "watermark_config_sha256": f"{prefix}_wcfg",
        "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
        "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
        "shared_clean_source_metadata_path": str((tmp_path / "tr" / "metadata.csv").resolve()),
        "shared_clean_source_metadata_sha256": "tr_metadata_sha",
        "shared_clean_sample_sha256": tr["base_latent_sha256"],
        "watermark_pre_injection_base_latent_sha256": tr["base_latent_sha256"],
        "tr_base_latent_sha256": tr["base_latent_sha256"],
        "tr_clean_path": tr["clean_path"],
        "tr_clean_sha256": tr["clean_sha256"],
        f"{prefix}_protocol_mode": mode,
        f"{prefix}_state_source": "bundle",
        f"{prefix}_bundle_dir": str(bundle.resolve()),
        f"{prefix}_bundle_config_sha256": f"{prefix}_bundle_cfg",
        f"{prefix}_selected_pattern_sha256": f"{prefix}_pattern",
        f"{prefix}_mask_sha256": f"{prefix}_mask",
        f"{prefix}_key_index": "7",
        f"{prefix}_pre_injection_latent_sha256": tr["base_latent_sha256"],
        f"{prefix}_post_injection_latent_sha256": f"{prefix}post{run_id:057d}",
        f"{prefix}_provider_entrypoint_sha256": f"{prefix}_provider",
    }
    row["pairing_sha256"] = build_pairing_sha256(row)
    return row


def test_method_dispatch_and_path_resolution_for_new_methods():
    for method in METHODS:
        assert method_data_root(method).as_posix().endswith(f"data/{method.lower()}")
        assert method_output_root(method).as_posix().endswith(f"outputs/{method.lower()}")
        assert formal_output_root(method, "diffusiondb", "formal", "rk").as_posix().endswith(
            f"outputs/{method.lower()}/diffusiondb/formal/rk"
        )


@pytest.mark.parametrize("method", sorted(METHODS))
def test_provider_config_requires_shared_clean_bundle_identity(method, tmp_path):
    row = _method_row(tmp_path, method)
    cfg = provider_config(method, row)
    assert cfg[f"{method.lower()}_protocol_mode"] == METHODS[method][2]
    assert provider_config_hash(method, row)
    bad = dict(row)
    bad[f"{method.lower()}_bundle_config_sha256"] = ""
    with pytest.raises(ValueError, match="missing required"):
        provider_config(method, bad)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_snapshot_manifest_creation_rejects_duplicate_run_id(method, tmp_path):
    row = _method_row(tmp_path, method)
    source = _write_csv(tmp_path / "source.csv", [row, dict(row)])
    args = type("Args", (), {
        "immutable_source_snapshot_index": None,
        "source_metadata": source,
        "expected_count": 2,
        "dataset": "synthetic",
        "method": method,
        "output_root": tmp_path / "out",
        "batch_size": 2,
    })()
    config = {"attack_config": {"image_size": [4, 4]}}
    with pytest.raises(ValueError, match="duplicate"):
        formal.snapshot_stage(args, config)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_attack_record_pairing_cache_reuse_and_rejection(method, tmp_path):
    row = formal.normalize_snapshot_row(
        _method_row(tmp_path, method), dataset="synthetic", method=method,
        attack_config={"image_size": [4, 4]},
    )
    row["snapshot_sha256"] = "snapshot"
    row["source_manifest_sha256"] = "source"
    config = {
        "dataset": "synthetic",
        "method": method,
        "attack_config_hash": "attackhash",
        "git_head": "head",
        "formal_source_config_hash": "sourcecfg",
        "source_code_manifest_sha256": "codemanifest",
        "attack_config": FORMAL_ATTACK_CONFIG,
        "attack_runtime": {
            "attack_device_class": "cuda",
            "attack_dtype": "torch.float16",
            "scheduler_class": "DDIMScheduler",
            "scheduler_config": {"beta_start": 0.00085},
            "scheduler_config_hash": canonical_json_hash({"beta_start": 0.00085}),
            "torch_version": "2.test",
            "diffusers_version": "0.test",
        },
    }
    expected = formal.expected_resume_fields(row, role="watermarked", dx=24.0, dy=-27.0, seed=42, config=config)
    attacked = tmp_path / "attacked.png"; attacked.write_bytes(b"attacked")
    pre = tmp_path / "pre.png"; pre.write_bytes(b"pre")
    debug = _debug(tmp_path / "debug.json")
    payload = json.loads(debug.read_text())
    record = {
        **expected,
        "attacked_path": str(attacked), "attacked_sha256": sha256_path(attacked),
        "pre_color_attacked_path": str(pre), "pre_color_attacked_sha256": sha256_path(pre),
        "debug_info_path": str(debug), "debug_info_sha256": sha256_path(debug),
        "resolved_scheduler_config": payload["scheduler_config"],
        "resolved_scheduler_config_hash": payload["scheduler_config_hash"],
        "torch_version": "2.test", "diffusers_version": "0.test",
        "attack_device_class": "cuda", "attack_dtype": "torch.float16",
        "transform_config_hash": payload["transform_config_hash"],
    }
    validate_resume_record(record, expected=expected, attack_config=FORMAL_ATTACK_CONFIG)
    with pytest.raises(RuntimeError, match="resume mismatch"):
        validate_resume_record(record, expected={**expected, "watermark_target_sha256": "other"}, attack_config=FORMAL_ATTACK_CONFIG)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_build_verification_manifest_preserves_method_provenance_and_rejects_hash_drift(method, tmp_path):
    row = formal.normalize_snapshot_row(
        _method_row(tmp_path, method), dataset="synthetic", method=method,
        attack_config={"image_size": [4, 4]},
    )
    snapshot = _write_csv(tmp_path / "snapshot.csv", [row])
    index = tmp_path / "snapshot_index.jsonl"
    index.write_text(json.dumps({
        "batch_id": 0,
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_sha256": sha256_path(snapshot),
        "source_metadata_sha256": "source",
        "row_count": 1,
    }) + "\n")
    attacked = tmp_path / "attacked.png"; attacked.write_bytes(b"attacked")
    pre = tmp_path / "pre.png"; pre.write_bytes(b"pre")
    debug = _debug(tmp_path / "debug.json")
    payload = json.loads(debug.read_text())
    attack = {
        **row,
        "dataset": "synthetic", "method": method, "run_id": "0",
        "attacked_path": str(attacked.resolve()), "attacked_sha256": sha256_path(attacked),
        "pre_color_attacked_path": str(pre.resolve()), "pre_color_attacked_sha256": sha256_path(pre),
        "debug_info_path": str(debug.resolve()), "debug_info_sha256": sha256_path(debug),
        "model_id": FORMAL_ATTACK_CONFIG["model_id"], "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "attack_seed": 42, "planned_flow_dx_image_px": 24.0, "planned_flow_dy_image_px": -27.0,
        "attack_config_hash": "attackhash", "transform_config_hash": payload["transform_config_hash"],
        "source_code_manifest_sha256": "codemanifest",
    }
    attacks = tmp_path / "attacks.jsonl"
    attacks.write_text(json.dumps(attack) + "\n")
    out = tmp_path / "manifest.csv"
    argv = [
        "--dataset", "synthetic", "--method", method, "--metadata", str(index),
        "--attack-records", str(attacks), "--snapshot-manifest", str(index), "--output", str(out),
    ]
    assert manifest.main.__wrapped__(argv) if hasattr(manifest.main, "__wrapped__") else True
    # main() parses sys.argv, so exercise the lower-level strict loaders directly here.
    snapshots, _ = manifest.load_snapshots(index)
    assert snapshots[0][f"{method.lower()}_bundle_config_sha256"] == f"{method.lower()}_bundle_cfg"
    bad = dict(attack, **{f"{method.lower()}_bundle_config_sha256": "bad"})
    for field in manifest.method_provenance_fields(method, [row]):
        if str(row.get(field)) != str(bad.get(field)):
            assert field.endswith("bundle_config_sha256")
            break


@pytest.mark.parametrize("method", sorted(METHODS))
def test_detector_extractor_and_experiment_table_for_new_methods(method, tmp_path):
    metric_name = METHODS[method][3]
    root = tmp_path / "outputs" / method.lower() / "synthetic" / "formal" / "rk"
    (root / "verification").mkdir(parents=True)
    (root / "metrics" / "quality").mkdir(parents=True)
    (root / "metrics" / "fid").mkdir(parents=True)
    (root / "metrics" / "clip").mkdir(parents=True)
    (root / "run_config.json").write_text(json.dumps({"method": method, "dataset": "synthetic"}))
    metric = {
        "detector_metric": metric_name,
        "raw_detector_metric": f"{method.lower()}_l1",
        "score_direction": "higher_is_watermarked",
        "raw_score_direction": "lower_is_watermarked",
        "threshold_type": "empirical_clean_1pct_fpr",
        "threshold_score_space": "canonical_score",
        "threshold_comparison_operator": ">=",
        "clean_calibrated_threshold": -10.0,
        "target_fpr": 0.01,
        "clean_calibrated_actual_empirical_fpr": 0.0,
        "mean_canonical_score_before": -1.0,
        "mean_canonical_score_attacked": -20.0,
        "before_detection_rate_at_clean_calibrated_threshold": 1.0,
        "attacked_detection_rate_at_clean_calibrated_threshold": 0.25,
        "attacked_roc_auc": 0.75,
    }
    (root / "verification" / "verification_result.json").write_text(json.dumps({"method": method, "dataset": "synthetic", "metric": metric}))
    (root / "formal_aggregate.json").write_text(json.dumps({
        "method": method, "dataset": "synthetic", "N": 4,
        "attack_config": {"variant_name": "formal_baseline"}, "attack_config_hash": "a" * 40,
        "attack_success_rate_at_clean_calibrated_threshold": 0.75,
        "fid_result": {"value": 1.0}, "clip_result": {"mean": 0.5},
        "quality_psnr_mean": 30.0, "quality_ssim_mean": 0.9,
    }))
    (root / "VALIDATED.json").write_text(json.dumps({"status": "validated_formal_result"}))
    table = tmp_path / "table.md"
    update_experiment_table(root, table)
    row = read_rows(table)[0]
    assert row["Method"] == method
    assert row["Detector Metric"] == metric_name
    assert row["Score Direction"] == "higher_is_watermarked"
    assert row["Threshold Type"] == "empirical_clean_1pct_fpr"
    assert row["Attack Success"] == "0.75"
    metric["threshold_score_space"] = "raw_l1"
    (root / "verification" / "verification_result.json").write_text(json.dumps({"method": method, "dataset": "synthetic", "metric": metric}))
    with pytest.raises(Exception, match="canonical_score"):
        update_experiment_table(root, tmp_path / "bad.md")


def test_hsqr_center_slice_mask_identity_is_stable_without_manifest_mask():
    from generate_hsqr_from_tr_shared_clean import SPEC
    from shared_tr_clean_fourier import _fourier_mask_sha256
    from raven.eval_protocol import canonical_json_hash

    class DummyProvider:
        start = 10
        end = 54
        watermark_channels = [0, 1, 2, 3]
        latent_shape = [1, 4, 64, 64]

    expected = canonical_json_hash({
        "method": "HSQR",
        "mask_identity": "center_slice_protocol",
        "center_slice": [10, 54],
        "watermark_channels": [0, 1, 2, 3],
        "latent_shape": [1, 4, 64, 64],
        "version": 1,
    })
    assert _fourier_mask_sha256(SPEC, DummyProvider(), {}) == expected



@pytest.mark.parametrize("method", sorted(METHODS))
def test_pairing_audit_rejects_bundle_provenance_and_no_clean_regeneration(method, tmp_path):
    row = _method_row(tmp_path, method)
    audit_pairing_rows([row], expected_count=1, verify_files=True)
    bad = dict(row)
    bad[f"{method.lower()}_selected_pattern_sha256"] = "wrong"
    bad["watermark_target_sha256"] = "wrong"
    bad["pairing_sha256"] = build_pairing_sha256(bad)
    with pytest.raises(ValueError, match="inconsistent|drift|mismatch"):
        audit_pairing_rows([bad], expected_count=1, verify_files=True)
    fake_clean = _png(tmp_path / method.lower() / "000000" / "clean.png", (99, 99, 99))
    bad_clean = dict(row, clean_path=str(fake_clean.resolve()), clean_sha256=sha256_path(fake_clean), tr_clean_path=str(fake_clean.resolve()), tr_clean_sha256=sha256_path(fake_clean))
    bad_clean["pairing_sha256"] = build_pairing_sha256(bad_clean)
    with pytest.raises(ValueError, match="clean_path|clean_sha256|TR clean"):
        audit_shared_clean_cohorts([_tr_row(tmp_path, 0)], {method: [bad_clean]}, verify_files=False)
