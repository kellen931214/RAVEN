#!/usr/bin/env python
"""NFPA-style Tree-Ring complex-L1 TPR@1%FPR for completed RAVEN P1 outputs.

This script reuses existing attacked-watermarked images, generates only the
missing attacked-clean images with the same RAVEN settings/shift plan, and
computes the NFPA Tree-Ring score:

    mean(abs(decoded_watermark - target_watermark))

Lower scores indicate a stronger Tree-Ring watermark.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.metrics import crop_overlap_inverse_warp, roc_auc
from raven.pairing_provenance import (
    assert_attack_pair_config_match as assert_attack_pair_provenance,
    audit_pairing_rows,
    build_attack_config_sha256,
    tensor_sha256,
)
from raven.pipeline_raven import RavenPipeline, require_effective_source_flow
from raven.eval_protocol import (
    FORMAL_ATTACK_CONFIG as CENTRAL_FORMAL_ATTACK_CONFIG,
    provider_config,
    require_uniform_provider_config,
    sha256_path as formal_sha256_path,
)
from raven.utils import load_image
from scripts import raven_p1_full as p1

MODEL_ID = p1.MODEL_ID
MODEL_REVISION = p1.MODEL_REVISION
P1_MODE = p1.P1_MODE
VAE_SCALE_FACTOR = p1.VAE_SCALE_FACTOR
TARGET_FPR = 0.01


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_json(path: Path, payload: Any, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_stats(values) -> dict:
    vals = [float(v) for v in values]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        raise ValueError("no finite values")
    return {
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def quality_pair(
    reference: Image.Image,
    attacked: Image.Image,
    effective_source_dx: float,
    effective_source_dy: float,
    suffix: str,
) -> dict:
    first = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(attacked.convert("RGB"), dtype=np.float32) / 255.0
    overlap_first, overlap_second = crop_overlap_inverse_warp(
        first, second, effective_source_dx, effective_source_dy
    )
    return {
        f"psnr_vs_{suffix}": float(peak_signal_noise_ratio(overlap_first, overlap_second, data_range=1.0)),
        f"ssim_vs_{suffix}": float(structural_similarity(overlap_first, overlap_second, channel_axis=2, data_range=1.0)),
        f"valid_overlap_width_vs_{suffix}": int(overlap_first.shape[1]),
        f"valid_overlap_height_vs_{suffix}": int(overlap_first.shape[0]),
        f"valid_overlap_area_ratio_vs_{suffix}": float(overlap_first.shape[0] * overlap_first.shape[1] / (first.shape[0] * first.shape[1])),
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_protocol(output_dir: Path) -> tuple[dict, dict[str, dict[str, str]], dict]:
    plan = json.loads((output_dir / "shift_plan.json").read_text())
    rows = load_csv_rows(output_dir / "diagnostic_manifest.csv")
    pairing_audit = audit_pairing_rows(
        rows, expected_count=len(plan["samples"]), verify_files=True
    )
    manifest = {str(row["run_id"]): row for row in rows}
    provenance = json.loads((output_dir / "provenance.json").read_text())
    if provenance.get("pairing_audit") != pairing_audit:
        raise ValueError("stored pairing audit does not match current source provenance")
    return plan, manifest, provenance


FORMAL_ATTACK_CONFIG = {
    "warp_mode": "raven_paper_nfpa_gap_fill",
    "padding_mode": "reflection",
    "sampling_mode": "nearest",
    "color_transfer_mode": "none",
    "inversion_mode": "ddim",
    "strength": 0.15,
    "guidance_scale": 2.5,
    "steps": 50,
    "prompt": "",
    "inversion_prompt": "",
    "reconstruction_prompt": "",
}


def assert_formal_debug_config(debug_info: dict, run_id: str, stage: str) -> None:
    checks = {
        "warp_mode": debug_info.get("warp_mode"),
        "padding_mode": debug_info.get("padding_mode"),
        "sampling_mode": debug_info.get("interpolation_mode"),
        "color_transfer_mode": debug_info.get("color_transfer_mode"),
        "inversion_mode": debug_info.get("inversion_mode"),
        "strength": float(debug_info.get("strength")),
        "guidance_scale": float(debug_info.get("guidance_scale")),
        "inversion_prompt": debug_info.get("inversion_prompt"),
        "reconstruction_prompt": debug_info.get("reconstruction_prompt"),
    }
    for key, expected in FORMAL_ATTACK_CONFIG.items():
        if key in {"steps", "prompt"}:
            continue
        actual = checks.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-9:
                raise RuntimeError(f"{stage} config mismatch run_id={run_id}: {key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            raise RuntimeError(f"{stage} config mismatch run_id={run_id}: {key}={actual!r}, expected {expected!r}")
    if not debug_info.get("transform_config_hash"):
        raise RuntimeError(f"{stage} missing transform_config_hash run_id={run_id}")


def attack_config_from_record(record: dict) -> dict:
    return {
        "run_id": str(record["run_id"]),
        "seed": int(record["seed"]),
        "flow_dx_image_px": float(record["flow_dx_image_px"]),
        "flow_dy_image_px": float(record["flow_dy_image_px"]),
        "exact_ddim_timestep": int(record["exact_ddim_timestep"]),
        "steps": int(record.get("steps", 50)),
        "strength": float(record.get("strength", 0.15)),
        "guidance_scale": float(record.get("guidance_scale", 2.5)),
        "inversion_mode": record.get("inversion_mode", "ddim"),
        "inversion_prompt": record.get("inversion_prompt", ""),
        "reconstruction_prompt": record.get("reconstruction_prompt", ""),
        "warp_mode": record.get("warp_mode"),
        "sampling_mode": record.get("sampling_mode"),
        "padding_mode": record.get("padding_mode"),
        "normalization_formula": record.get("normalization_formula"),
        "color_transfer_mode": record["color_transfer_mode"],
        "transform_config_hash": record.get("transform_config_hash"),
    }


def assert_attack_pair_config_match(clean_attack: dict, wm_attack: dict, run_id: str) -> None:
    clean_cfg = attack_config_from_record(clean_attack)
    wm_cfg = attack_config_from_record(wm_attack)
    for key in (
        "run_id", "seed", "flow_dx_image_px", "flow_dy_image_px", "exact_ddim_timestep",
        "steps", "strength", "guidance_scale", "inversion_mode", "inversion_prompt",
        "reconstruction_prompt", "warp_mode", "sampling_mode", "padding_mode",
        "normalization_formula", "color_transfer_mode", "transform_config_hash",
    ):
        if clean_cfg[key] != wm_cfg[key]:
            raise RuntimeError(
                f"attacked clean/watermarked config mismatch run_id={run_id}: "
                f"{key} clean={clean_cfg[key]!r} wm={wm_cfg[key]!r}"
            )
    for key, expected in FORMAL_ATTACK_CONFIG.items():
        if key == "prompt":
            continue
        actual = clean_cfg.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-9:
                raise RuntimeError(f"formal config mismatch run_id={run_id}: {key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            raise RuntimeError(f"formal config mismatch run_id={run_id}: {key}={actual!r}, expected {expected!r}")
    assert_attack_pair_provenance(clean_attack, wm_attack, run_id)


def validate_p1_protocol_source(
    p1_dir: Path, expected_count: int
) -> tuple[dict, list[dict], dict]:
    manifest = load_csv_rows(p1_dir / "diagnostic_manifest.csv")
    plan = json.loads((p1_dir / "shift_plan.json").read_text())
    if len(manifest) != expected_count:
        raise ValueError(f"{p1_dir}: expected {expected_count} manifest rows, got {len(manifest)}")
    if len(plan["samples"]) != expected_count:
        raise ValueError(f"{p1_dir}: expected {expected_count} shift samples, got {len(plan['samples'])}")
    if len({str(row["run_id"]) for row in manifest}) != expected_count:
        raise ValueError(f"{p1_dir}: duplicate run_id in manifest")
    if len({str(row["run_id"]) for row in plan["samples"]}) != expected_count:
        raise ValueError(f"{p1_dir}: duplicate run_id in shift plan")
    if {str(row["run_id"]) for row in manifest} != {
        str(row["run_id"]) for row in plan["samples"]
    }:
        raise ValueError(f"{p1_dir}: manifest/shift-plan run_id mismatch")
    pairing_audit = audit_pairing_rows(
        manifest, expected_count=expected_count, verify_files=True
    )
    source_provenance = json.loads((p1_dir / "provenance.json").read_text())
    if source_provenance.get("pairing_audit") != pairing_audit:
        raise ValueError(f"{p1_dir}: stored pairing audit does not match current files")
    return plan, manifest, pairing_audit


def validate_p1_source(p1_dir: Path, expected_count: int) -> tuple[dict, list[dict], list[dict]]:
    plan, manifest, _ = validate_p1_protocol_source(p1_dir, expected_count)
    attacks = load_jsonl(p1_dir / "attack_records.jsonl")
    if len(attacks) != expected_count:
        raise ValueError(f"{p1_dir}: expected {expected_count} attacked-watermarked rows, got {len(attacks)}")
    if len({str(row["run_id"]) for row in attacks}) != expected_count:
        raise ValueError(f"{p1_dir}: duplicate run_id in attacked-watermarked records")
    for attack in attacks:
        path = Path(attack["attacked_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_path(path) != attack["attacked_sha256"]:
            raise ValueError(f"attacked-watermarked SHA drift: {path}")
        if attack.get("attack_config_sha256") != build_attack_config_sha256(attack):
            raise ValueError(f"attacked-watermarked config SHA drift: {path}")
    return plan, manifest, attacks


def prepare_protocol_workspace(args) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    for name in ("configs", "logs", "state", "attacked_clean_outputs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    plan, manifest, pairing_audit = validate_p1_protocol_source(
        args.p1_dir.resolve(), args.expected_count
    )
    shutil.copy2(args.p1_dir / "diagnostic_manifest.csv", output_dir / "diagnostic_manifest.csv")
    shutil.copy2(args.p1_dir / "shift_plan.json", output_dir / "shift_plan.json")

    provenance = {
        "protocol": "raven_paper_nfpa_gap_fill_tree_ring_complex_l1_v2",
        "dataset": args.dataset,
        "expected_count": args.expected_count,
        "source_p1_dir": str(args.p1_dir.resolve()),
        "source_p1_manifest_sha256": sha256_path(args.p1_dir / "diagnostic_manifest.csv"),
        "source_p1_shift_plan_sha256": sha256_path(args.p1_dir / "shift_plan.json"),
        "source_p1_attacked_watermarked_records_sha256": None,
        "attacked_watermarked_sync_status": "pending",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pairing_audit": pairing_audit,
        "target_fpr": TARGET_FPR,
        "score_name": "NFPA-style Tree-Ring complex L1",
        "score_definition": "torch.abs(decoded_watermark - target_watermark).mean(-1)",
        "score_direction": "lower means more likely watermarked",
        "threshold_rule": "sorted_clean_scores[max(0, int(N*0.01)-1)], detection uses score < threshold",
        "nfpa_source_references": {
            "NFPA/utils.py:_decode_batch_raw": "lines 1497-1517",
            "NFPA/utils.py:err": "lines 1523-1525",
            "NFPA/utils.py:compute_threshold_at_fpr": "lines 1640-1663",
            "NFPA/utils.py:compute_tpr_at_fpr": "lines 1665-1693",
        },
        "created_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "python": sys.executable,
        "argv": sys.argv,
    }
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "source_counts.json", {
        "manifest_rows": len(manifest),
        "shift_samples": len(plan["samples"]),
        "attacked_watermarked_records": 0,
    })
    return provenance


def sync_attacked_watermarked(args) -> dict:
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    p1_dir = args.p1_dir.resolve()
    plan, manifest, attacks = validate_p1_source(p1_dir, args.expected_count)
    pairing_audit = audit_pairing_rows(
        manifest, expected_count=args.expected_count, verify_files=True
    )
    provenance_path = output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    expected_source = str(p1_dir)
    if provenance.get("source_p1_dir") != expected_source:
        raise ValueError(
            f"NFPA protocol source mismatch: {provenance.get('source_p1_dir')} != {expected_source}"
        )
    expected_hashes = {
        "source_p1_manifest_sha256": sha256_path(p1_dir / "diagnostic_manifest.csv"),
        "source_p1_shift_plan_sha256": sha256_path(p1_dir / "shift_plan.json"),
    }
    for key, actual in expected_hashes.items():
        if provenance.get(key) != actual:
            raise ValueError(f"NFPA protocol source hash mismatch: {key}")
    if provenance.get("pairing_audit") != pairing_audit:
        raise ValueError("NFPA protocol pairing audit drift before watermarked sync")

    source_records = p1_dir / "attack_records.jsonl"
    source_bytes = source_records.read_bytes()
    for destination in (
        output_dir / "attack_records.jsonl",
        output_dir / "attacked_watermarked_records.jsonl",
    ):
        if destination.exists() and destination.read_bytes() != source_bytes:
            raise ValueError(f"refusing to replace mismatched existing records: {destination}")
        if not destination.exists():
            shutil.copy2(source_records, destination)
    legacy_source = p1_dir / "per_sample_results.jsonl"
    legacy_destination = output_dir / "legacy_log10p_fixed_threshold_results.jsonl"
    if legacy_source.is_file() and not legacy_destination.exists():
        shutil.copy2(legacy_source, legacy_destination)

    records_sha = sha256_path(source_records)
    provenance["source_p1_attacked_watermarked_records_sha256"] = records_sha
    provenance["attacked_watermarked_sync_status"] = "complete"
    provenance["attacked_watermarked_synced_utc"] = time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )
    write_json(provenance_path, provenance, exclusive=False)
    write_json(
        output_dir / "source_counts.json",
        {
            "manifest_rows": len(manifest),
            "shift_samples": len(plan["samples"]),
            "attacked_watermarked_records": len(attacks),
        },
        exclusive=False,
    )
    return provenance


def command_prepare_protocol(args) -> int:
    provenance = prepare_protocol_workspace(args)
    print(json.dumps(provenance, indent=2, ensure_ascii=False), flush=True)
    return 0


def command_sync_watermarked(args) -> int:
    provenance = sync_attacked_watermarked(args)
    print(json.dumps(provenance, indent=2, ensure_ascii=False), flush=True)
    return 0


def command_prepare(args) -> int:
    prepare_protocol_workspace(args)
    provenance = sync_attacked_watermarked(args)
    print(json.dumps(provenance, indent=2, ensure_ascii=False), flush=True)
    return 0


def command_attack_clean(args) -> int:
    import torch

    plan, manifest, provenance = load_protocol(args.output_dir)
    records_path = args.output_dir / "attacked_clean_records.jsonl"
    completed: dict[str, dict] = {}
    open_mode = "x"
    if records_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {records_path}")
        for row in load_jsonl(records_path):
            run_id = str(row["run_id"])
            attacked_path = Path(row["attacked_clean_path"])
            if not attacked_path.is_file() or sha256_path(attacked_path) != row["attacked_clean_sha256"]:
                raise ValueError(f"completed attacked-clean hash mismatch run_id={run_id}")
            if row.get("attack_config_sha256") != build_attack_config_sha256(row):
                raise ValueError(f"completed attacked-clean config hash mismatch run_id={run_id}")
            completed[run_id] = row
        open_mode = "a"

    pipe = RavenPipeline(model_id=MODEL_ID, revision=MODEL_REVISION, device=args.device, dtype=args.dtype)
    done = len(completed)
    total = len(plan["samples"])
    with records_path.open(open_mode, encoding="utf-8") as output:
        for sample in plan["samples"]:
            run_id = str(sample["run_id"])
            if run_id in completed:
                continue
            row = manifest[run_id]
            shift = sample["shift"]
            clean_path = Path(row["clean_path"])
            watermarked_path = Path(row["watermarked_path"])
            if sha256_path(clean_path) != row["clean_sha256"]:
                raise ValueError(f"clean SHA drift run_id={run_id}")
            if sha256_path(watermarked_path) != row["watermarked_sha256"]:
                raise ValueError(f"watermarked SHA drift run_id={run_id}")
            item_dir = args.output_dir / "attacked_clean_outputs" / f"{int(run_id):06d}"
            if item_dir.exists():
                raise FileExistsError(f"unrecorded attacked-clean output exists: {item_dir}")

            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            clean = load_image(clean_path, size=512)
            watermarked = load_image(watermarked_path, size=512)
            pipe.run(
                input_image=clean,
                output_dir=item_dir,
                steps=50,
                strength=0.15,
                guidance_scale=2.5,
                shift_space="image_pixels",
                warp_mode="raven_paper_nfpa_gap_fill",
                padding_mode="reflection",
                latent_sampling_mode="nearest",
                shift_x=shift["flow_dx_image_px"],
                shift_y=shift["flow_dy_image_px"],
                view_guided_attention=True,
                color_transfer=False,
                seed=int(row["attack_seed"]),
                prompt="",
                negative_prompt="",
                debug=args.debug,
                inversion_mode="ddim",
            )
            final_path = item_dir / "final.png"
            attacked = load_image(final_path, size=None)
            debug_info_path = item_dir / "debug_info.json"
            debug_info = json.loads(debug_info_path.read_text())
            if debug_info["inversion_prompt"] != "" or debug_info["reconstruction_prompt"] != "":
                raise RuntimeError(f"non-empty prompt in attacked-clean run_id={run_id}")
            assert_formal_debug_config(debug_info, run_id, "attacked-clean")
            effective_dx, effective_dy = require_effective_source_flow(debug_info)
            record = {
                "dataset": provenance["dataset"],
                "sample_id": run_id,
                "run_id": run_id,
                "mode": P1_MODE,
                "clean_path": row["clean_path"],
                "clean_sha256": row["clean_sha256"],
                "watermarked_path": row["watermarked_path"],
                "watermarked_sha256": row["watermarked_sha256"],
                "pairing_sha256": row["pairing_sha256"],
                "base_latent_seed": int(row["base_latent_seed"]),
                "base_latent_sha256": row["base_latent_sha256"],
                "watermark_target_sha256": row["watermark_target_sha256"],
                "generation_config_sha256": row["generation_config_sha256"],
                "watermark_config_sha256": row["watermark_config_sha256"],
                "attacked_clean_path": str(final_path.resolve()),
                "attacked_clean_sha256": sha256_path(final_path),
                "seed": int(row["attack_seed"]),
                "watermark_seed": int(row["watermark_seed"]),
                **shift,
                "prompt": "",
                "inversion_prompt": debug_info["inversion_prompt"],
                "reconstruction_prompt": debug_info["reconstruction_prompt"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "exact_ddim_timestep": int(debug_info["exact_timestep"]),
                "active_denoising_steps": len(debug_info["timesteps"]),
                "steps": 50,
                "strength": 0.15,
                "guidance_scale": 2.5,
                "inversion_mode": debug_info["inversion_mode"],
                "warp_mode": debug_info["warp_mode"],
                "sampling_mode": debug_info["interpolation_mode"],
                "padding_mode": debug_info["padding_mode"],
                "normalization_formula": debug_info["normalized_coordinate_formula"],
                "color_transfer_mode": debug_info["color_transfer_mode"],
                "transform_config_hash": debug_info.get("transform_config_hash"),
                "effective_source_flow_dx_image_px": effective_dx,
                "effective_source_flow_dy_image_px": effective_dy,
                **quality_pair(clean, attacked, effective_dx, effective_dy, "clean"),
                **quality_pair(
                    watermarked, attacked, effective_dx, effective_dy, "watermarked"
                ),
                "quality_primary_reference": "clean_input_for_attacked_clean_stage",
                "runtime_seconds": float(time.monotonic() - started),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "attention_processor_count": int(debug_info.get("attention_processor_count") or 0),
                "debug_info_path": str(debug_info_path.resolve()),
            }
            record["attack_config_sha256"] = build_attack_config_sha256(record)
            append_jsonl(output, record)
            done += 1
            print(f"[attack-clean {done}/{total}] dataset={provenance['dataset']} run_id={run_id} gpu={record['peak_gpu_memory_bytes']/2**30:.2f}GiB", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def complex_l1_score(torch, provider, pipe, path: Path, steps: int) -> dict:
    from PIL import ImageOps

    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    with torch.no_grad():
        inversion = provider.invert_images(image, pipe_provider_target=pipe, num_inference_steps=steps) \
            if hasattr(provider, "invert_images") else pipe.invert_images(image, num_inference_steps=steps)
        recovered = inversion["zT_torch"]
        recovered_fft = torch.fft.fftshift(torch.fft.fft2(recovered), dim=(-1, -2))
        mask = provider.watermarking_mask[0]
        target = provider.gt_patch[0][mask].flatten()
        scores = []
        decoded_norms = []
        target_norms = []
        for latent_fft in recovered_fft:
            decoded = latent_fft[mask].flatten()
            distance = torch.abs(decoded - target)
            scores.append(float(distance.mean().detach().cpu().item()))
            decoded_norms.append(float(torch.abs(decoded).mean().detach().cpu().item()))
            target_norms.append(float(torch.abs(target).mean().detach().cpu().item()))
    del inversion, recovered, recovered_fft
    return {
        "score": scores[0],
        "decoded_abs_mean": decoded_norms[0],
        "target_abs_mean": target_norms[0],
        "nan": not math.isfinite(scores[0]) or math.isnan(scores[0]),
        "inf": math.isinf(scores[0]),
    }


def command_score_l1(args) -> int:
    import torch

    if not (args.eval_repo / "utils" / "pipe" / "pipe_utils.py").is_file():
        raise FileNotFoundError(args.eval_repo)
    sys.path.insert(0, str(args.eval_repo.resolve()))
    from raven.resource_guard import limit_cpu_threads
    from scripts.extract_verification_scores import provider_class, provider_kwargs
    from utils.pipe import pipe_utils

    limit_cpu_threads(1)
    plan, manifest, provenance = load_protocol(args.output_dir)
    attacked_clean = {str(row["run_id"]): row for row in load_jsonl(args.output_dir / "attacked_clean_records.jsonl")}
    attacked_wm = {str(row["run_id"]): row for row in load_jsonl(args.output_dir / "attacked_watermarked_records.jsonl")}
    if len(attacked_clean) != len(plan["samples"]):
        raise ValueError(f"expected {len(plan['samples'])} attacked-clean rows, found {len(attacked_clean)}")
    if len(attacked_wm) != len(plan["samples"]):
        raise ValueError(f"expected {len(plan['samples'])} attacked-watermarked rows, found {len(attacked_wm)}")

    records_path = args.output_dir / "l1_scores.jsonl"
    completed: dict[str, dict] = {}
    open_mode = "x"
    if records_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue {records_path}")
        for row in load_jsonl(records_path):
            completed[str(row["run_id"])] = row
        open_mode = "a"

    device = torch.device(args.device)
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=MODEL_ID,
        resolution=512,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
        revision=MODEL_REVISION,
    )
    first = manifest[str(plan["samples"][0]["run_id"])]
    provider = provider_class("TR")(
        latent_shape=pipe.get_latent_shape(),
        dtype=pipe.get_dtype(),
        device=device,
        **provider_kwargs("TR", first),
    )

    rows = list(completed.values())
    total = len(plan["samples"])
    with records_path.open(open_mode, encoding="utf-8") as output:
        for index, sample in enumerate(plan["samples"], start=1):
            run_id = str(sample["run_id"])
            if run_id in completed:
                continue
            row = manifest[run_id]
            clean_attack = attacked_clean[run_id]
            wm_attack = attacked_wm[run_id]
            assert_attack_pair_config_match(clean_attack, wm_attack, run_id)
            paths = {
                "original_clean": Path(row["clean_path"]),
                "watermarked": Path(row["watermarked_path"]),
                "attacked_clean": Path(clean_attack["attacked_clean_path"]),
                "attacked_watermarked": Path(wm_attack["attacked_path"]),
            }
            expected_hash = {
                "original_clean": row["clean_sha256"],
                "watermarked": row["watermarked_sha256"],
                "attacked_clean": clean_attack["attacked_clean_sha256"],
                "attacked_watermarked": wm_attack["attacked_sha256"],
            }
            for stage, path in paths.items():
                if sha256_path(path) != expected_hash[stage]:
                    raise ValueError(f"{stage} SHA drift run_id={run_id}: {path}")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            scores = {stage: complex_l1_score(torch, provider, pipe, path, 50) for stage, path in paths.items()}
            record = {
                "dataset": provenance["dataset"],
                "sample_id": run_id,
                "run_id": run_id,
                "score_protocol": "NFPA-style Tree-Ring complex L1",
                "score_definition": "torch.abs(decoded_watermark - target_watermark).mean(-1)",
                "score_direction": "lower means more watermark",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "watermark_seed": int(row["watermark_seed"]),
                "original_clean_path": str(paths["original_clean"].resolve()),
                "watermarked_path": str(paths["watermarked"].resolve()),
                "attacked_clean_path": str(paths["attacked_clean"].resolve()),
                "attacked_watermarked_path": str(paths["attacked_watermarked"].resolve()),
                "original_clean_sha256": expected_hash["original_clean"],
                "watermarked_sha256": expected_hash["watermarked"],
                "attacked_clean_sha256": expected_hash["attacked_clean"],
                "attacked_watermarked_sha256": expected_hash["attacked_watermarked"],
                "original_clean_l1": scores["original_clean"]["score"],
                "watermarked_l1": scores["watermarked"]["score"],
                "attacked_clean_l1": scores["attacked_clean"]["score"],
                "attacked_watermarked_l1": scores["attacked_watermarked"]["score"],
                "original_clean_decoded_abs_mean": scores["original_clean"]["decoded_abs_mean"],
                "watermarked_decoded_abs_mean": scores["watermarked"]["decoded_abs_mean"],
                "attacked_clean_decoded_abs_mean": scores["attacked_clean"]["decoded_abs_mean"],
                "attacked_watermarked_decoded_abs_mean": scores["attacked_watermarked"]["decoded_abs_mean"],
                "target_abs_mean": scores["watermarked"]["target_abs_mean"],
                "attack_config_hash": clean_attack.get("transform_config_hash"),
                "warp_mode": clean_attack.get("warp_mode"),
                "sampling_mode": clean_attack.get("sampling_mode"),
                "padding_mode": clean_attack.get("padding_mode"),
                "normalization_formula": clean_attack.get("normalization_formula"),
                "color_transfer_mode": clean_attack.get("color_transfer_mode"),
                "flow_dx_image_px": float(clean_attack["flow_dx_image_px"]),
                "flow_dy_image_px": float(clean_attack["flow_dy_image_px"]),
                "visual_content_direction": clean_attack.get("visual_content_direction"),
                "detector_nan": any(item["nan"] for item in scores.values()),
                "detector_inf": any(item["inf"] for item in scores.values()),
                "p_value_underflow": None,
                "runtime_seconds": float(time.monotonic() - started),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_cpu_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
            if record["detector_nan"] or record["detector_inf"]:
                raise ValueError(f"non-finite NFPA L1 score run_id={run_id}")
            append_jsonl(output, record)
            rows.append(record)
            print(
                f"[score-l1 {index}/{total}] dataset={provenance['dataset']} run_id={run_id} "
                f"clean={record['original_clean_l1']:.6f} wm={record['watermarked_l1']:.6f} "
                f"att_clean={record['attacked_clean_l1']:.6f} att_wm={record['attacked_watermarked_l1']:.6f}",
                flush=True,
            )

    rows.sort(key=lambda x: int(x["run_id"]))
    write_csv(args.output_dir / "per_sample_results.csv", rows)
    write_csv(args.output_dir / "l1_scores.csv", rows)
    del provider, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def nfpa_threshold(clean_scores: list[float], target_fpr: float = TARGET_FPR) -> float:
    sorted_scores = np.sort(np.asarray(clean_scores, dtype=np.float64))
    k = max(0, int(len(clean_scores) * target_fpr) - 1)
    return float(sorted_scores[k])


def nfpa_round2(value: float) -> float:
    """Match the pinned NFPA NumPy-array `.round(2)` score serialization."""
    return float(np.asarray(value, dtype=np.float64).round(2))


def nfpa_detection_rate(scores: list[float], threshold: float) -> float:
    return float(np.mean(np.asarray(scores, dtype=np.float64) < threshold))


nfpa_rate = nfpa_detection_rate


def aggregate_nfpa_protocol(rows: list[dict], suffix: str) -> dict:
    columns = {
        stage: [float(row[f"{stage}_l1_{suffix}"]) for row in rows]
        for stage in (
            "original_clean", "watermarked", "attacked_clean", "attacked_watermarked"
        )
    }
    before_threshold = nfpa_threshold(columns["original_clean"])
    after_threshold = nfpa_threshold(columns["attacked_clean"])
    before_fp = int(np.sum(np.asarray(columns["original_clean"]) < before_threshold))
    after_fp = int(np.sum(np.asarray(columns["attacked_clean"]) < after_threshold))
    before_tpr = nfpa_detection_rate(columns["watermarked"], before_threshold)
    recalibrated_tpr = nfpa_detection_rate(columns["attacked_watermarked"], after_threshold)
    original_threshold_tpr = nfpa_detection_rate(
        columns["attacked_watermarked"], before_threshold
    )
    return {
        "target_fpr": TARGET_FPR,
        "original_clean_target_fpr": TARGET_FPR,
        "original_clean_threshold": before_threshold,
        "original_clean_actual_fpr": before_fp / len(rows),
        "original_clean_fp_count": before_fp,
        "before_threshold": before_threshold,
        "before_actual_fpr": before_fp / len(rows),
        "before_false_positives": before_fp,
        "before_tpr": before_tpr,
        "before_roc_auc": roc_auc(
            [-value for value in columns["watermarked"]],
            [-value for value in columns["original_clean"]],
        ),
        "attacked_clean_recalibrated_threshold": after_threshold,
        "attacked_clean_target_fpr": TARGET_FPR,
        "attacked_clean_actual_fpr": after_fp / len(rows),
        "attacked_clean_fp_count": after_fp,
        "attacked_actual_fpr": after_fp / len(rows),
        "attacked_false_positives": after_fp,
        "attacked_tpr_at_original_clean_threshold": original_threshold_tpr,
        "attacked_tpr_at_attacked_clean_recalibrated_threshold": recalibrated_tpr,
        "attacked_roc_auc": roc_auc(
            [-value for value in columns["attacked_watermarked"]],
            [-value for value in columns["attacked_clean"]],
        ),
        "attack_success_rate_at_recalibrated_threshold": 1.0 - recalibrated_tpr,
    }


def aggregate_nfpa_protocol_no_clean(rows: list[dict], suffix: str) -> dict:
    """Storage-light aggregate: original-clean calibration only, no recalibration.

    Uses the original-clean and watermarked scores that are always available, and
    reports attacked-watermarked TPR at the original-clean threshold. All
    attacked-clean recalibration fields are null because attack-clean was skipped.
    """
    columns = {
        stage: [float(row[f"{stage}_l1_{suffix}"]) for row in rows]
        for stage in ("original_clean", "watermarked", "attacked_watermarked")
    }
    before_threshold = nfpa_threshold(columns["original_clean"])
    before_fp = int(np.sum(np.asarray(columns["original_clean"]) < before_threshold))
    before_tpr = nfpa_detection_rate(columns["watermarked"], before_threshold)
    original_threshold_tpr = nfpa_detection_rate(
        columns["attacked_watermarked"], before_threshold
    )
    return {
        "target_fpr": TARGET_FPR,
        "attack_clean_enabled": False,
        "recalibrated_metrics_available": False,
        "original_clean_target_fpr": TARGET_FPR,
        "original_clean_threshold": before_threshold,
        "original_clean_actual_fpr": before_fp / len(rows),
        "original_clean_fp_count": before_fp,
        "before_threshold": before_threshold,
        "before_actual_fpr": before_fp / len(rows),
        "before_false_positives": before_fp,
        "before_tpr": before_tpr,
        "before_roc_auc": roc_auc(
            [-value for value in columns["watermarked"]],
            [-value for value in columns["original_clean"]],
        ),
        "attacked_tpr_at_original_clean_threshold": original_threshold_tpr,
        "attacked_clean_recalibrated_threshold": None,
        "attacked_clean_actual_fpr": None,
        "attacked_clean_fp_count": None,
        "attacked_tpr_at_attacked_clean_recalibrated_threshold": None,
        "attacked_roc_auc": None,
        "attack_success_rate_at_recalibrated_threshold": None,
    }


def assert_detector_source_tensor_hashes(
    rows: list[dict[str, Any]], *, detector_target_sha256: str, detector_mask_sha256: str
) -> None:
    """Fail before scoring when detector tensors differ from generation tensors."""
    source_target_hashes = {row.get("watermark_target_sha256", "") for row in rows}
    source_mask_hashes = {row.get("watermark_mask_sha256", "") for row in rows}
    if source_target_hashes != {detector_target_sha256}:
        raise RuntimeError(
            "detector/source target tensor SHA mismatch before scoring: "
            f"detector={detector_target_sha256} source={sorted(source_target_hashes)}"
        )
    if source_mask_hashes != {detector_mask_sha256}:
        raise RuntimeError(
            "detector/source mask tensor SHA mismatch before scoring: "
            f"detector={detector_mask_sha256} source={sorted(source_mask_hashes)}"
        )


def command_score_formal(args) -> int:
    import torch

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse formal TR output: {args.output_dir}")
    attack_clean_enabled = not args.no_attacked_clean
    if attack_clean_enabled and args.attacked_clean_records is None:
        raise ValueError("attacked-clean scoring requires --attacked-clean-records")
    if not attack_clean_enabled and args.attacked_clean_records is not None:
        raise ValueError("--no-attacked-clean forbids --attacked-clean-records")
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError("formal TR manifest is empty")
    manifest = {str(row["run_id"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate run ID in formal TR inputs")
    if attack_clean_enabled:
        clean_rows = load_jsonl(args.attacked_clean_records)
        if len(manifest_rows) != len(clean_rows):
            raise ValueError("formal TR manifest and attacked-clean record counts differ")
        clean = {str(row["run_id"]): row for row in clean_rows}
        if len(clean) != len(clean_rows):
            raise ValueError("duplicate run ID in formal TR inputs")
        if set(manifest) != set(clean):
            raise ValueError("formal TR manifest and attacked-clean run-ID sets differ")
    else:
        clean = {}
    config, config_hash = require_uniform_provider_config("TR", manifest_rows)
    recorded_hashes = {row.get("provider_config_hash", "") for row in manifest_rows}
    if recorded_hashes != {config_hash}:
        raise ValueError(f"formal TR provider hash mismatch: {sorted(recorded_hashes)}")

    sys.path.insert(0, str(args.eval_repo.resolve()))
    from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
    from scripts.extract_verification_scores import provider_class
    from utils.pipe import pipe_utils

    limit_cpu_threads(1)
    CpuMemoryGuard(24.0, 48.0, 40.0).check("formal TR score startup")
    device = torch.device(args.device)
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=CENTRAL_FORMAL_ATTACK_CONFIG["model_id"],
        resolution=512,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
        revision=CENTRAL_FORMAL_ATTACK_CONFIG["model_revision"],
    )
    provider = provider_class("TR")(
        latent_shape=pipe.get_latent_shape(),
        dtype=pipe.get_dtype(),
        device=device,
        **config,
    )
    target_hash = tensor_sha256(provider.gt_patch)
    mask_hash = tensor_sha256(provider.watermarking_mask)
    assert_detector_source_tensor_hashes(
        manifest_rows,
        detector_target_sha256=target_hash,
        detector_mask_sha256=mask_hash,
    )
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "l1_scores.jsonl"
    rows = []
    with output_path.open("x", encoding="utf-8") as output:
        for index, run_id in enumerate(sorted(manifest, key=int), start=1):
            row = manifest[run_id]
            paths = {
                "original_clean": Path(row["clean_path"]),
                "watermarked": Path(row["watermarked_path"]),
                "attacked_watermarked": Path(row["attacked_path"]),
            }
            expected = {
                "original_clean": row["clean_sha256"],
                "watermarked": row["watermarked_sha256"],
                "attacked_watermarked": row["attacked_sha256"],
            }
            if attack_clean_enabled:
                clean_attack = clean[run_id]
                if clean_attack.get("input_role") != "clean":
                    raise ValueError(f"run_id={run_id}: attacked-clean record has wrong role")
                if clean_attack.get("attack_config_hash") != row.get("attack_config_hash"):
                    raise ValueError(f"run_id={run_id}: attacked clean/watermarked config mismatch")
                for field in (
                    "attack_seed", "planned_flow_dx_image_px", "planned_flow_dy_image_px",
                    "transform_config_hash", "pairing_sha256",
                ):
                    if str(clean_attack.get(field)) != str(row.get(field)):
                        raise ValueError(f"run_id={run_id}: attacked pair {field} mismatch")
                paths["attacked_clean"] = Path(clean_attack["attacked_path"])
                expected["attacked_clean"] = clean_attack["attacked_sha256"]
            for stage, path in paths.items():
                if formal_sha256_path(path) != expected[stage]:
                    raise RuntimeError(f"run_id={run_id}: {stage} SHA mismatch")
            scores = {
                stage: complex_l1_score(torch, provider, pipe, path, 50)["score"]
                for stage, path in paths.items()
            }
            if not all(math.isfinite(value) for value in scores.values()):
                raise ValueError(f"run_id={run_id}: non-finite complex-L1 score")
            record = {
                "dataset": row["dataset"],
                "method": "TR",
                "run_id": run_id,
                "provider_config": config,
                "provider_config_hash": config_hash,
                "target_watermark_hash": target_hash,
                "detector_target_watermark_sha256": target_hash,
                "source_watermark_target_sha256": row["watermark_target_sha256"],
                "detector_watermark_mask_sha256": mask_hash,
                "source_watermark_mask_sha256": row["watermark_mask_sha256"],
                "source_target_watermark_hash": row["watermark_target_sha256"],
                "pairing_sha256": row["pairing_sha256"],
                "score_direction": "lower means watermark",
            }
            for stage, value in scores.items():
                record[f"{stage}_l1_full_precision"] = value
                record[f"{stage}_l1_nfpa_rounded2"] = nfpa_round2(value)
            record["tr_l1_full_precision"] = scores["attacked_watermarked"]
            record["tr_l1_rounded2"] = nfpa_round2(scores["attacked_watermarked"])
            append_jsonl(output, record)
            rows.append(record)
            print(f"[formal-tr {index}/{len(manifest)}] run_id={run_id}", flush=True)
    protocol_aggregate = (
        aggregate_nfpa_protocol if attack_clean_enabled else aggregate_nfpa_protocol_no_clean
    )
    aggregate = {
        "protocol": "formal_tree_ring_complex_l1_v3",
        "N": len(rows),
        "sample_count": len(rows),
        "gate_only": len(rows) < 1000,
        "paper_comparable": False,
        "attack_clean_enabled": attack_clean_enabled,
        "recalibrated_metrics_available": attack_clean_enabled,
        "result_classification": (
            "formal_complete" if attack_clean_enabled else "TR STORAGE-LIGHT / NO ATTACK-CLEAN"
        ),
        "statistical_validity": (
            "gate_only_insufficient_for_1pct_fpr"
            if len(rows) < 100
            else "requires_final_protocol_review"
        ),
        "score_definition": "torch.abs(decoded_watermark - target_watermark).mean()",
        "threshold_rule": "NFPA order statistic; strict score < threshold",
        "provider_config": config,
        "provider_config_hash": config_hash,
        "target_watermark_hash": target_hash,
        "detector_target_watermark_sha256": target_hash,
        "source_watermark_target_sha256": target_hash,
        "detector_watermark_mask_sha256": mask_hash,
        "source_watermark_mask_sha256": mask_hash,
        "full_precision_protocol": protocol_aggregate(rows, "full_precision"),
        "nfpa_rounded2_protocol": protocol_aggregate(rows, "nfpa_rounded2"),
        "primary_protocol": "nfpa_rounded2_protocol",
    }
    write_json(args.output_dir / "aggregate_results.json", aggregate)
    del provider, pipe
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


def command_aggregate(args) -> int:
    plan, _, provenance = load_protocol(args.output_dir)
    rows = load_jsonl(args.output_dir / "l1_scores.jsonl")
    if len(rows) != len(plan["samples"]):
        raise ValueError(f"expected {len(plan['samples'])} NFPA L1 rows, found {len(rows)}")
    if len({str(row["run_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate run_id in NFPA L1 scores")

    original_clean = [float(row["original_clean_l1"]) for row in rows]
    watermarked = [float(row["watermarked_l1"]) for row in rows]
    attacked_clean = [float(row["attacked_clean_l1"]) for row in rows]
    attacked_watermarked = [float(row["attacked_watermarked_l1"]) for row in rows]

    before_threshold = nfpa_threshold(original_clean)
    after_threshold = nfpa_threshold(attacked_clean)
    before_fpr = nfpa_rate(original_clean, before_threshold)
    before_tpr = nfpa_rate(watermarked, before_threshold)
    after_fpr = nfpa_rate(attacked_clean, after_threshold)
    after_tpr = nfpa_rate(attacked_watermarked, after_threshold)
    attacked_tpr_at_before_threshold = nfpa_rate(attacked_watermarked, before_threshold)
    before_fp = int(np.sum(np.asarray(original_clean, dtype=np.float64) < before_threshold))
    after_fp = int(np.sum(np.asarray(attacked_clean, dtype=np.float64) < after_threshold))
    before_auc = roc_auc([-v for v in watermarked], [-v for v in original_clean])
    after_auc = roc_auc([-v for v in attacked_watermarked], [-v for v in attacked_clean])

    direction_stats = {}
    for direction in sorted({str(row.get("visual_content_direction")) for row in rows}):
        items = [row for row in rows if str(row.get("visual_content_direction")) == direction]
        direction_stats[direction] = {
            "N": len(items),
            "after_tpr": nfpa_rate([float(row["attacked_watermarked_l1"]) for row in items], after_threshold),
            "mean_attacked_watermarked_l1": finite_stats(row["attacked_watermarked_l1"] for row in items)["mean"],
        }

    duplicate_score_audit = {}
    for key, values in (
        ("original_clean_l1", original_clean),
        ("watermarked_l1", watermarked),
        ("attacked_clean_l1", attacked_clean),
        ("attacked_watermarked_l1", attacked_watermarked),
    ):
        rounded = {}
        exact = {}
        for row, value in zip(rows, values):
            rounded.setdefault(f"{value:.6f}", []).append(str(row["run_id"]))
            exact.setdefault(repr(float(value)), []).append(str(row["run_id"]))
        duplicate_score_audit[key] = {
            "rounded_6dp_duplicate_groups": {k: v for k, v in rounded.items() if len(v) > 1},
            "exact_duplicate_groups": {k: v for k, v in exact.items() if len(v) > 1},
        }

    score_histograms = {
        "original_clean_l1": p1.histogram(original_clean),
        "watermarked_l1": p1.histogram(watermarked),
        "attacked_clean_l1": p1.histogram(attacked_clean),
        "attacked_watermarked_l1": p1.histogram(attacked_watermarked),
    }
    write_json(args.output_dir / "score_histograms.json", score_histograms, exclusive=False)
    write_json(args.output_dir / "failed_samples.json", {"failed": []}, exclusive=False)

    aggregate = {
        "protocol": "raven_paper_nfpa_gap_fill_tree_ring_complex_l1_v2",
        "result_name": "NFPA-style Tree-Ring TPR@1%FPR",
        "dataset": provenance["dataset"],
        "N": len(rows),
        "score_definition": "torch.abs(decoded_watermark - target_watermark).mean(-1)",
        "score_direction": "lower means more watermark",
        "threshold_rule": "sorted_scores[max(0, int(N*0.01)-1)], detection uses score < threshold",
        "before_attack": {
            "threshold": before_threshold,
            "actual_fpr": before_fpr,
            "false_positives": before_fp,
            "tpr": before_tpr,
            "roc_auc": before_auc,
            "clean_score_source": "original_clean_l1",
            "watermarked_score_source": "watermarked_l1",
        },
        "after_attack": {
            "threshold": after_threshold,
            "actual_fpr": after_fpr,
            "false_positives": after_fp,
            "tpr": after_tpr,
            "attack_success_rate": 1.0 - after_tpr,
            "roc_auc": after_auc,
            "attacked_tpr_at_fixed_before_threshold": attacked_tpr_at_before_threshold,
            "clean_score_source": "attacked_clean_l1",
            "watermarked_score_source": "attacked_watermarked_l1",
        },
        "mean_scores": {
            "original_clean": finite_stats(original_clean)["mean"],
            "watermarked": finite_stats(watermarked)["mean"],
            "attacked_clean": finite_stats(attacked_clean)["mean"],
            "attacked_watermarked": finite_stats(attacked_watermarked)["mean"],
        },
        "score_stats": {
            "original_clean": finite_stats(original_clean),
            "watermarked": finite_stats(watermarked),
            "attacked_clean": finite_stats(attacked_clean),
            "attacked_watermarked": finite_stats(attacked_watermarked),
        },
        "score_histograms": score_histograms,
        "shift_direction_analysis": direction_stats,
        "duplicate_score_audit": duplicate_score_audit,
        "quality": {
            "primary_reference": "watermarked_input",
            "post_color_overlap_psnr_vs_watermarked": finite_stats(row.get("psnr_vs_watermarked", float("nan")) for row in load_jsonl(args.output_dir / "attacked_watermarked_records.jsonl")),
            "post_color_overlap_ssim_vs_watermarked": finite_stats(row.get("ssim_vs_watermarked", float("nan")) for row in load_jsonl(args.output_dir / "attacked_watermarked_records.jsonl")),
        },
        "numeric_diagnostics": {
            "nan_count": int(sum(bool(row["detector_nan"]) for row in rows)),
            "inf_count": int(sum(bool(row["detector_inf"]) for row in rows)),
            "p_value_underflow_count": None,
        },
        "old_log10p_fixed_threshold_reference": {
            "path": str((args.output_dir / "legacy_log10p_fixed_threshold_results.jsonl").resolve())
            if (args.output_dir / "legacy_log10p_fixed_threshold_results.jsonl").is_file() else None,
            "note": "Separate -log10(p) fixed-threshold analysis; not mixed with NFPA-style complex L1.",
        },
    }
    write_json(args.output_dir / "aggregate_results.json", aggregate, exclusive=False)
    lines = [
        f"# NFPA-style Tree-Ring TPR@1%FPR - {provenance['dataset']}",
        "",
        "Score: `torch.abs(decoded_watermark - target_watermark).mean(-1)`; lower means more likely watermarked.",
        "",
        "| Metric | Before | After |",
        "| --- | ---: | ---: |",
        f"| Threshold | {before_threshold:.8f} | {after_threshold:.8f} |",
        f"| Actual FPR | {before_fpr:.6f} ({before_fp} FP) | {after_fpr:.6f} ({after_fp} FP) |",
        f"| TPR@1%FPR | {before_tpr:.6f} | {after_tpr:.6f} |",
        f"| Attack success rate | - | {1.0 - after_tpr:.6f} |",
        f"| Mean clean L1 | {aggregate['mean_scores']['original_clean']:.8f} | {aggregate['mean_scores']['attacked_clean']:.8f} |",
        f"| Mean watermarked L1 | {aggregate['mean_scores']['watermarked']:.8f} | {aggregate['mean_scores']['attacked_watermarked']:.8f} |",
        f"| ROC-AUC | {before_auc:.6f} | {after_auc:.6f} |",
        "",
        "The copied `legacy_log10p_fixed_threshold_results.jsonl` is retained only as separate analysis.",
    ]
    (args.output_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
    return 0


def run_stage(argv: list[str]) -> None:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), *argv]
    print("$ " + " ".join(str(x) for x in command), flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_state(root: Path) -> dict:
    path = root / "run_state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_stages": []}


def save_state(root: Path, state: dict) -> None:
    write_json(root / "run_state.json", state, exclusive=False)


def mark_done(root: Path, state: dict, stage: str) -> None:
    if stage not in state["completed_stages"]:
        state["completed_stages"].append(stage)
    save_state(root, state)


def write_combined(root: Path, db_dir: Path, coco_dir: Path) -> None:
    db = json.loads((db_dir / "aggregate_results.json").read_text())
    coco = json.loads((coco_dir / "aggregate_results.json").read_text())
    rows = []
    for label, agg in (("DiffusionDB", db), ("MS-COCO", coco)):
        rows.append({
            "dataset": label,
            "N": agg["N"],
            "before_attack_threshold": agg["before_attack"]["threshold"],
            "before_attack_actual_fpr": agg["before_attack"]["actual_fpr"],
            "before_attack_tpr": agg["before_attack"]["tpr"],
            "after_attack_threshold": agg["after_attack"]["threshold"],
            "after_attack_actual_fpr": agg["after_attack"]["actual_fpr"],
            "after_attack_tpr": agg["after_attack"]["tpr"],
            "attack_success_rate": agg["after_attack"]["attack_success_rate"],
            "original_clean_mean_score": agg["mean_scores"]["original_clean"],
            "watermarked_mean_score": agg["mean_scores"]["watermarked"],
            "attacked_clean_mean_score": agg["mean_scores"]["attacked_clean"],
            "attacked_watermarked_mean_score": agg["mean_scores"]["attacked_watermarked"],
        })
    payload = {
        "protocol": "nfpa_style_tree_ring_complex_l1_combined_v1",
        "result_name": "NFPA-style Tree-Ring TPR@1%FPR",
        "diffusiondb_output": str(db_dir.resolve()),
        "mscoco_output": str(coco_dir.resolve()),
        "datasets": rows,
        "note": "Dataset thresholds are calibrated separately. -log10(p) fixed-threshold results are not mixed with this table.",
    }
    write_json(root / "combined_summary.json", payload, exclusive=False)
    lines = [
        "# NFPA-style Tree-Ring TPR@1%FPR Combined Summary",
        "",
        "| Dataset | N | Before threshold | Before FPR | Before TPR | After threshold | After FPR | After TPR | Attack success | Clean mean | WM mean | Att-clean mean | Att-WM mean |",
        "| ------- | -: | ---------------: | ---------: | ---------: | --------------: | --------: | --------: | -------------: | ---------: | ------: | -------------: | ----------: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['N']} | {row['before_attack_threshold']:.8f} | {row['before_attack_actual_fpr']:.6f} | "
            f"{row['before_attack_tpr']:.6f} | {row['after_attack_threshold']:.8f} | {row['after_attack_actual_fpr']:.6f} | "
            f"{row['after_attack_tpr']:.6f} | {row['attack_success_rate']:.6f} | {row['original_clean_mean_score']:.8f} | "
            f"{row['watermarked_mean_score']:.8f} | {row['attacked_clean_mean_score']:.8f} | {row['attacked_watermarked_mean_score']:.8f} |"
        )
    (root / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_run_sequential(args) -> int:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = load_state(root)
    if "created_utc" not in state:
        state["created_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        state["diffusiondb_dir"] = str((root / "diffusiondb" / state["created_utc"]).resolve())
        state["mscoco_dir"] = str((root / "mscoco" / state["created_utc"]).resolve())
        save_state(root, state)
    write_json(root / "orchestrator_provenance.json", {"python": sys.executable, "argv": sys.argv, "cwd": str(Path.cwd())}, exclusive=not (root / "orchestrator_provenance.json").exists())
    stages = state["completed_stages"]
    db_dir = Path(state["diffusiondb_dir"])
    coco_dir = Path(state["mscoco_dir"])

    if "diffusiondb_prepare" not in stages:
        run_stage(["prepare", "--dataset", "diffusiondb", "--p1-dir", str(args.diffusiondb_p1_dir), "--output-dir", str(db_dir), "--expected-count", "1001"])
        mark_done(root, state, "diffusiondb_prepare")
    if "diffusiondb_attack_clean" not in stages:
        run_stage(["attack-clean", "--output-dir", str(db_dir), "--device", "cuda", "--dtype", "float16", "--resume"])
        mark_done(root, state, "diffusiondb_attack_clean")
    if "diffusiondb_score_l1" not in stages:
        run_stage(["score-l1", "--output-dir", str(db_dir), "--device", "cuda", "--resume"])
        mark_done(root, state, "diffusiondb_score_l1")
    if "diffusiondb_aggregate" not in stages:
        run_stage(["aggregate", "--output-dir", str(db_dir)])
        mark_done(root, state, "diffusiondb_aggregate")

    if "mscoco_prepare" not in stages:
        run_stage(["prepare", "--dataset", "mscoco", "--p1-dir", str(args.mscoco_p1_dir), "--output-dir", str(coco_dir), "--expected-count", "1000"])
        mark_done(root, state, "mscoco_prepare")
    if "mscoco_attack_clean" not in stages:
        run_stage(["attack-clean", "--output-dir", str(coco_dir), "--device", "cuda", "--dtype", "float16", "--resume"])
        mark_done(root, state, "mscoco_attack_clean")
    if "mscoco_score_l1" not in stages:
        run_stage(["score-l1", "--output-dir", str(coco_dir), "--device", "cuda", "--resume"])
        mark_done(root, state, "mscoco_score_l1")
    if "mscoco_aggregate" not in stages:
        run_stage(["aggregate", "--output-dir", str(coco_dir)])
        mark_done(root, state, "mscoco_aggregate")
    if "combined" not in stages:
        write_combined(root, db_dir, coco_dir)
        mark_done(root, state, "combined")
    print(json.dumps(load_state(root), indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--p1-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--expected-count", type=int, required=True)
    prepare_protocol = sub.add_parser("prepare-protocol")
    prepare_protocol.add_argument("--dataset", required=True)
    prepare_protocol.add_argument("--p1-dir", type=Path, required=True)
    prepare_protocol.add_argument("--output-dir", type=Path, required=True)
    prepare_protocol.add_argument("--expected-count", type=int, required=True)
    sync_watermarked = sub.add_parser("sync-watermarked")
    sync_watermarked.add_argument("--p1-dir", type=Path, required=True)
    sync_watermarked.add_argument("--output-dir", type=Path, required=True)
    sync_watermarked.add_argument("--expected-count", type=int, required=True)
    attack = sub.add_parser("attack-clean")
    attack.add_argument("--output-dir", type=Path, required=True)
    attack.add_argument("--device", default="cuda")
    attack.add_argument("--dtype", choices=["float16"], default="float16")
    attack.add_argument("--resume", action="store_true")
    attack.add_argument("--debug", action="store_true")
    score = sub.add_parser("score-l1")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    score.add_argument("--device", choices=["cuda"], default="cuda")
    score.add_argument("--resume", action="store_true")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, required=True)
    formal = sub.add_parser("score-formal")
    formal.add_argument("--manifest", type=Path, required=True)
    formal.add_argument("--attacked-clean-records", type=Path, default=None)
    formal.add_argument(
        "--no-attacked-clean",
        action="store_true",
        help=(
            "Storage-light TR: score watermarked detection only, without "
            "attacked-clean FPR recalibration or recalibrated TPR."
        ),
    )
    formal.add_argument("--output-dir", type=Path, required=True)
    formal.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    formal.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run = sub.add_parser("run-sequential")
    run.add_argument("--root", type=Path, default=Path("outputs/raven_nfpa_tr_eval"))
    run.add_argument("--diffusiondb-p1-dir", type=Path, required=True)
    run.add_argument("--mscoco-p1-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "prepare-protocol":
        return command_prepare_protocol(args)
    if args.command == "sync-watermarked":
        return command_sync_watermarked(args)
    if args.command == "attack-clean":
        return command_attack_clean(args)
    if args.command == "score-l1":
        return command_score_l1(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    if args.command == "score-formal":
        return command_score_formal(args)
    if args.command == "run-sequential":
        return command_run_sequential(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
