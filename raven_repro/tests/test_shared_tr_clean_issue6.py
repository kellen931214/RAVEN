"""CPU-only shared-clean coverage for RID/HSTR/HSQR (Issue #6)."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for _root in (REPO / "raven_repro", REPO / "eval_bench_wm", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from raven.pairing_provenance import (  # noqa: E402
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
    canonical_json_sha256,
    sha256_path,
    tensor_sha256,
)
from shared_clean_tr import rebuild_shared_clean_latent  # noqa: E402

MODEL_ID = "RedbeardNZ/stable-diffusion-2-1-base"
MODEL_REVISION = "c6a5e9bab8d874d081de76fa270ae0aefa5410ff"
LATENT_SHAPE = (1, 4, 64, 64)
GENERATION_CONFIG = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "scheduler": "DDIM",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "resolution": 512,
    "dtype": str(torch.float32),
}
GENERATION_CONFIG_SHA256 = canonical_json_sha256(GENERATION_CONFIG)
METHODS = {
    "RID": (RID_SHARED_TR_CLEAN_PROTOCOL, RID_SHARED_TR_CLEAN_MODE),
    "HSTR": (HSTR_SHARED_TR_CLEAN_PROTOCOL, HSTR_SHARED_TR_CLEAN_MODE),
    "HSQR": (HSQR_SHARED_TR_CLEAN_PROTOCOL, HSQR_SHARED_TR_CLEAN_MODE),
}


def _image(tag: str) -> Image.Image:
    rng = np.random.RandomState(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))
    return Image.fromarray(rng.randint(0, 256, (16, 16, 3), dtype=np.uint8))


def _latent(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(LATENT_SHAPE, generator=g, dtype=torch.float32)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tr_rows(root: Path):
    rows = []
    clean_dir = root / "clean"
    tr_dir = root / "tr"
    clean_dir.mkdir(parents=True)
    tr_dir.mkdir(parents=True)
    for run_id, seed in enumerate((10, 11)):
        base = _latent(seed)
        base_sha = tensor_sha256(base)
        prompt = f"prompt {run_id}"
        clean = clean_dir / f"{run_id:06d}.png"
        wm = tr_dir / f"{run_id:06d}" / "watermarked.png"
        wm.parent.mkdir(parents=True)
        _image(f"clean-{run_id}").save(clean)
        _image(f"tr-{run_id}").save(wm)
        row = {
            "protocol": TR_PAIRING_PROTOCOL,
            "dataset_name": "synthetic",
            "dataset": "synthetic",
            "run_id": run_id,
            "prompt_id": run_id,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "source": "synthetic",
            "wm_type": "TR",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "scheduler_target": "DDIM",
            "num_inference_steps_target": 50,
            "guidance_scale_target": 7.5,
            "resolution": 512,
            "base_latent_seed": seed,
            "base_latent_sha256": base_sha,
            "clean_base_latent_sha256": base_sha,
            "watermarked_base_latent_sha256": base_sha,
            "watermarked_latent_sha256": tensor_sha256(base + 1),
            "watermark_target_sha256": "tr_target",
            "watermark_mask_sha256": "tr_mask",
            "generation_config_sha256": GENERATION_CONFIG_SHA256,
            "watermark_config_sha256": "tr_config",
            "clean_path": str(clean.resolve()),
            "clean_sha256": sha256_path(clean),
            "watermarked_path": str(wm.resolve()),
            "watermarked_image_path": str(wm.resolve()),
            "watermarked_sha256": sha256_path(wm),
        }
        row["pairing_sha256"] = build_pairing_sha256(row)
        rows.append(row)
    meta = tr_dir / "metadata.csv"
    _write_csv(meta, rows)
    return rows, meta


def _bundle(root: Path, method: str):
    path = root / f"{method.lower()}_bundle"
    path.mkdir(parents=True)
    manifest = {
        "bundle_config_sha256": f"{method.lower()}_bundle_cfg",
        "selected_pattern_sha256": f"{method.lower()}_pattern",
        "mask_sha256": f"{method.lower()}_mask",
    }
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path, manifest


def _method_rows(root: Path, tr_rows, tr_meta: Path, method: str):
    protocol, mode = METHODS[method]
    prefix = method.lower()
    bundle, manifest = _bundle(root, method)
    rows = []
    for tr in tr_rows:
        run_id = int(tr["run_id"])
        wm = root / method.lower() / f"{run_id:06d}" / "watermarked.png"
        wm.parent.mkdir(parents=True, exist_ok=True)
        _image(f"{method}-{run_id}").save(wm)
        row = {
            "protocol": protocol,
            "dataset_name": "synthetic",
            "dataset": "synthetic",
            "run_id": run_id,
            "prompt_id": tr["prompt_id"],
            "prompt": tr["prompt"],
            "prompt_sha256": tr["prompt_sha256"],
            "source": tr.get("source", ""),
            "wm_type": method,
            "wm_name": method,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_latent_seed": tr["base_latent_seed"],
            "base_latent_sha256": tr["base_latent_sha256"],
            "clean_base_latent_sha256": tr["clean_base_latent_sha256"],
            "watermarked_base_latent_sha256": tr["base_latent_sha256"],
            "watermarked_latent_sha256": f"{method.lower()}_post_{run_id}",
            "watermark_target_sha256": manifest["selected_pattern_sha256"],
            "watermark_mask_sha256": manifest["mask_sha256"],
            "generation_config_sha256": tr["generation_config_sha256"],
            "watermark_config_sha256": f"{method.lower()}_wm_config",
            "clean_path": tr["clean_path"],
            "clean_sha256": tr["clean_sha256"],
            "watermarked_path": str(wm.resolve()),
            "watermarked_image_path": str(wm.resolve()),
            "watermarked_sha256": sha256_path(wm),
            "shared_clean_protocol": SHARED_CLEAN_PROTOCOL,
            "shared_clean_source_method": SHARED_CLEAN_SOURCE_METHOD,
            "shared_clean_source_metadata_path": str(tr_meta.resolve()),
            "shared_clean_source_metadata_sha256": sha256_path(tr_meta),
            "shared_clean_sample_sha256": tr["base_latent_sha256"],
            "watermark_pre_injection_base_latent_sha256": tr["base_latent_sha256"],
            "tr_base_latent_sha256": tr["base_latent_sha256"],
            "tr_clean_path": tr["clean_path"],
            "tr_clean_sha256": tr["clean_sha256"],
            f"{prefix}_protocol_mode": mode,
            f"{prefix}_state_source": "bundle",
            f"{prefix}_bundle_dir": str(bundle.resolve()),
            f"{prefix}_bundle_config_sha256": manifest["bundle_config_sha256"],
            f"{prefix}_selected_pattern_sha256": manifest["selected_pattern_sha256"],
            f"{prefix}_mask_sha256": manifest["mask_sha256"],
            f"{prefix}_key_index": 1,
            f"{prefix}_pre_injection_latent_sha256": tr["base_latent_sha256"],
            f"{prefix}_post_injection_latent_sha256": f"{method.lower()}_post_{run_id}",
            f"{prefix}_provider_entrypoint_sha256": f"{method.lower()}_provider",
        }
        row["pairing_sha256"] = build_pairing_sha256(row)
        rows.append(row)
    audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
    return rows


def test_canonical_latent_reconstruction_and_wrong_seed_rejection(tmp_path):
    rows, _ = _tr_rows(tmp_path)
    _, latent, sha = rebuild_shared_clean_latent(torch, rows[0], resolution=512, device=torch.device("cpu"), dtype=torch.float32)
    assert sha == rows[0]["base_latent_sha256"] == tensor_sha256(latent)
    bad = dict(rows[0], base_latent_seed=int(rows[0]["base_latent_seed"]) + 1)
    with pytest.raises(Exception, match="does not match"):
        rebuild_shared_clean_latent(torch, bad, resolution=512, device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_cross_method_audit_accepts_new_methods_and_rejects_drift(tmp_path, method):
    tr_rows, tr_meta = _tr_rows(tmp_path / "src")
    rows = _method_rows(tmp_path / "out", tr_rows, tr_meta, method)
    result = audit_shared_clean_cohorts(
        tr_rows,
        {method: rows},
        verify_files=True,
        require_methods=(method,),
        expected_run_ids=[r["run_id"] for r in tr_rows],
        tr_metadata_path=tr_meta,
    )
    assert result["rows_checked"][method] == 2
    drifted = copy.deepcopy(rows)
    drifted[0]["clean_sha256"] = "0" * 64
    drifted[0]["pairing_sha256"] = build_pairing_sha256(drifted[0])
    with pytest.raises(ValueError, match="clean_sha256"):
        audit_shared_clean_cohorts(tr_rows, {method: drifted}, verify_files=False, tr_metadata_path=tr_meta)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_method_rows_are_relocatable_but_artifact_identity_bound(tmp_path, method):
    tr_rows, tr_meta = _tr_rows(tmp_path / "src")
    rows = _method_rows(tmp_path / "out", tr_rows, tr_meta, method)
    moved = copy.deepcopy(rows[0])
    original_hash = moved["pairing_sha256"]
    moved["clean_path"] = "/relocated/clean.png"
    moved["tr_clean_path"] = "/relocated/clean.png"
    moved[f"{method.lower()}_bundle_dir"] = "/relocated/bundle"
    assert build_pairing_sha256(moved) == original_hash
    moved["watermarked_sha256"] = "f" * 64
    assert build_pairing_sha256(moved) != original_hash


@pytest.mark.parametrize("method", sorted(METHODS))
def test_regenerated_method_specific_clean_image_is_rejected(tmp_path, method):
    tr_rows, tr_meta = _tr_rows(tmp_path / "src")
    rows = _method_rows(tmp_path / "out", tr_rows, tr_meta, method)
    fake_clean = tmp_path / "out" / method.lower() / "000000" / "clean.png"
    _image("method-clean").save(fake_clean)
    rows[0]["clean_path"] = str(fake_clean.resolve())
    rows[0]["clean_sha256"] = sha256_path(fake_clean)
    rows[0]["pairing_sha256"] = build_pairing_sha256(rows[0])
    with pytest.raises(ValueError, match="clean_path|clean_sha256"):
        audit_shared_clean_cohorts(tr_rows, {method: rows}, verify_files=False, tr_metadata_path=tr_meta)


def test_issue6_runner_specs_expose_protocol_modes():
    from generate_hsqr_from_tr_shared_clean import SPEC as hsqr_spec
    from generate_hstr_from_tr_shared_clean import SPEC as hstr_spec
    from generate_rid_from_tr_shared_clean import SPEC as rid_spec

    assert rid_spec.protocol_mode == RID_SHARED_TR_CLEAN_MODE
    assert hstr_spec.protocol_mode == HSTR_SHARED_TR_CLEAN_MODE
    assert hsqr_spec.protocol_mode == HSQR_SHARED_TR_CLEAN_MODE
