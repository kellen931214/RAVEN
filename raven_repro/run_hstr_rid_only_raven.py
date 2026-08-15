#!/usr/bin/env python3
"""Run RAVEN against a completed official-math HSTR RID-only cohort.

The workflow waits for complete direct generation, attacks only the
watermarked role, then adds the untouched clean images as a reference-only
cohort. Evaluation calibrates one threshold from original clean images and
applies it to original/attacked watermarked images only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_bench_wm"))
sys.path.insert(0, str(REPO / "raven_repro"))

RID_ONLY_PROFILE = "official_math_rid_only"
RID_ONLY_PROTOCOL = "hstr_rid_only_sfwmark_ablation_paired_direct_generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=1001)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_image_path(source_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (source_dir.parents[1] / path).resolve()


def wait_for_generation(source_dir: Path, expected_samples: int, poll_seconds: int) -> list[dict[str, Any]]:
    results_path = source_dir / "results.jsonl"
    while True:
        if results_path.is_file():
            rows = read_jsonl(results_path)
            complete = len(rows) == expected_samples
            pairs_exist = complete and all(
                source_image_path(source_dir, row["clean_image_path"]).is_file()
                and source_image_path(source_dir, row["watermarked_image_path"]).is_file()
                for row in rows
            )
            if pairs_exist:
                return rows
            print(f"Waiting for direct HSTR generation: {len(rows)}/{expected_samples} complete", flush=True)
        else:
            print("Waiting for direct HSTR generation manifest/results.jsonl", flush=True)
        time.sleep(poll_seconds)


def cuda_gpu_ready(gpu: int) -> bool:
    """Probe the exact physical GPU mapping used by raven_repro/main.py."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    probe = [
        sys.executable, "-c",
        "import torch; assert torch.cuda.is_available(); "
        "x=torch.ones(1, device='cuda:0'); torch.cuda.synchronize(); print(x.item())",
    ]
    return subprocess.run(probe, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def wait_for_cuda(gpu: int, poll_seconds: int) -> None:
    """Wait without model allocation until the selected GPU can execute PyTorch."""
    while not cuda_gpu_ready(gpu):
        print(f"Waiting for compatible CUDA GPU {gpu}; no attack process has been started", flush=True)
        time.sleep(poll_seconds)
    print(f"CUDA GPU {gpu} is compatible; starting watermarked-only attack", flush=True)


def validated_manifest(source_dir: Path) -> tuple[Path, dict[str, Any]]:
    bundle_dir = source_dir / "hstr_bundle"
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "method": "HSTR",
        "profile_name": RID_ONLY_PROFILE,
        "pattern_variant": "rid_only",
        "score_mode": "center_channel_0_complex_l1",
        "scheduler_type": "DDIM",
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"bundle manifest {field}={manifest.get(field)!r}, expected {expected!r}")
    if manifest.get("watermark_channels") != [0] or manifest.get("heterogeneous_channels") != [0]:
        raise ValueError("RID-only bundle must contain only watermark/heterogeneous channel 0")
    return bundle_dir.resolve(), manifest


def hstr_mask_sha256(bundle_dir: Path, manifest: dict[str, Any]) -> str:
    """Reuse HSTRProvider/bundle compatibility, without loading a diffusion model."""
    import torch
    from eval_bench_wm.utils.wm.hstr_provider import HSTRProvider

    provider = HSTRProvider(
        latent_shape=tuple(manifest["latent_shape"]),
        dtype=torch.float32,
        device=torch.device("cpu"),
        hstr_profile=manifest["profile_name"],
        hstr_bundle_dir=str(bundle_dir),
        hstr_key_index=int(manifest["selected_key_index"]),
        hstr_rng_device=str(manifest["rng_device"]),
        latent_channel=int(manifest["latent_shape"][1]),
        hw_latent=int(manifest["latent_shape"][2]),
        start=int(manifest["center_slice"][0]),
        end=int(manifest["center_slice"][1]),
        wm_capacity=int(manifest["wm_capacity"]),
        modelid_target=str(manifest["model_id"]),
        model_revision=str(manifest["model_revision"]),
        scheduler_target=str(manifest["scheduler_type"]),
        resolution=int(manifest["resolution"]),
    )
    if provider.profile != RID_ONLY_PROFILE or provider.score_definition != "hstr_rid_only_score=-channel_0_l1":
        raise ValueError("loaded bundle is not the channel-0-only HSTR profile")
    return provider.watermark_mask_sha256


def build_metadata(rows: list[dict[str, Any]], source_dir: Path, bundle_dir: Path, manifest: dict[str, Any], metadata_path: Path) -> None:
    mask_sha256 = hstr_mask_sha256(bundle_dir, manifest)
    fieldnames = [
        "run_id", "watermarked_path", "clean_path", "prompt", "method",
        "hstr_bundle_dir", "hstr_bundle_config_sha256", "hstr_selected_pattern_sha256",
        "hstr_mask_sha256", "hstr_key_index", "hstr_protocol_mode",
        "watermark_target_sha256", "watermark_mask_sha256",
    ]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in rows:
            if source.get("profile_name") != RID_ONLY_PROFILE or source.get("protocol") != RID_ONLY_PROTOCOL:
                raise ValueError(f"sample {source.get('sample_name')}: source provenance is not HSTR RID-only direct generation")
            writer.writerow({
                "run_id": source["sample_name"],
                "watermarked_path": str(source_image_path(source_dir, source["watermarked_image_path"])),
                "clean_path": str(source_image_path(source_dir, source["clean_image_path"])),
                "prompt": source.get("prompt", ""),
                "method": "HSTR",
                "hstr_bundle_dir": str(bundle_dir),
                "hstr_bundle_config_sha256": manifest["bundle_config_sha256"],
                "hstr_selected_pattern_sha256": manifest["selected_pattern_sha256"],
                "hstr_mask_sha256": mask_sha256,
                "hstr_key_index": manifest["selected_key_index"],
                "hstr_protocol_mode": RID_ONLY_PROTOCOL,
                "watermark_target_sha256": manifest["selected_pattern_sha256"],
                "watermark_mask_sha256": mask_sha256,
            })


def run_watermarked_attack(args: argparse.Namespace, metadata_path: Path) -> None:
    command = [
        sys.executable, str(REPO / "raven_repro" / "main.py"),
        "--dataset", "hstr_rid_only_direct_1001", "--method", "HSTR",
        "--metadata", str(metadata_path), "--output-dir", str(args.output_dir),
        "--roles", "watermarked", "--diffusion-mode", "ddim-ddpm",
        "--sampling", "nearest", "--gpu", str(args.gpu),
        "--limit", str(args.expected_samples),
    ]
    if args.resume:
        command.append("--resume")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def add_untouched_clean_records(rows: list[dict[str, Any]], source_dir: Path, output_dir: Path, metadata_path: Path) -> None:
    from raven.experiment_io import output_image_path, rebuild_records_jsonl, write_record

    for source in rows:
        run_id = str(source["sample_name"])
        clean_path = source_image_path(source_dir, source["clean_image_path"])
        output_path = output_image_path(output_dir, "clean", run_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() or output_path.is_symlink():
            if not output_path.is_symlink() or output_path.resolve() != clean_path:
                raise FileExistsError(f"refusing to replace non-reference clean output: {output_path}")
        else:
            output_path.symlink_to(clean_path)
        write_record(output_dir, "clean", run_id, {
            "run_id": run_id, "role": "clean", "dataset": "hstr_rid_only_direct_1001",
            "method": "HSTR", "input_path": str(clean_path), "output_path": str(output_path),
            "prompt": source.get("prompt", ""), "reference_only_clean": True,
            "threshold_calibration_role": "original_clean_only",
            "metadata_path": str(metadata_path),
            "effective_source_flow_dx_image_px": None,
            "effective_source_flow_dy_image_px": None,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    rebuild_records_jsonl(output_dir)


def run_evaluation(output_dir: Path, gpu: int) -> None:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HUB_OFFLINE"] = "1"
    subprocess.run(
        [sys.executable, str(REPO / "raven_repro" / "eval.py"),
         "--output-dir", str(output_dir), "--device", "cuda",
         "--stages", "quality", "detector", "fid", "clip", "lpips"],
        cwd=REPO, env=env, check=True,
    )


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    rows = wait_for_generation(source_dir, args.expected_samples, args.poll_seconds)
    bundle_dir, manifest = validated_manifest(source_dir)
    metadata_path = output_dir.parent / f"{output_dir.name}.hstr_rid_only_metadata.csv"
    build_metadata(rows, source_dir, bundle_dir, manifest, metadata_path)
    wait_for_cuda(args.gpu, args.poll_seconds)
    run_watermarked_attack(args, metadata_path)
    add_untouched_clean_records(rows, source_dir, output_dir, metadata_path)
    run_evaluation(output_dir, args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
