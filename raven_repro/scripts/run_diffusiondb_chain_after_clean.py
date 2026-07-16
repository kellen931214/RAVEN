#!/usr/bin/env python
"""Wait for clean DiffusionDB generation, then run TR watermark and RAVEN attack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.gpu_utils import setup_run_logging


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def clean_complete(clean_dir: Path, expected: int) -> bool:
    manifest = clean_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    png_count = len(list(clean_dir.glob("*.png")))
    return len(rows) == expected and png_count >= expected


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def write_results(path: Path, expected_count: int, stages: list[str]) -> None:
    payload = {
        "expected_count": expected_count,
        "completed_utc": utc_now(),
        "stages": stages,
        "status": "completed",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1001)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--watermark-gpu", default="0")
    parser.add_argument("--attack-visible-gpu", default="6")
    args = parser.parse_args()

    root = args.root.resolve()
    setup_run_logging(root)
    clean_dir = root / "data/generated/diffusiondb"
    prompts_csv = root / "inputs/diffusiondb_1001_prompts.csv"
    marker_path = root / "CHAIN_COMPLETE"
    results_path = root / "results.json"
    stages: list[str] = []

    log(f"chain waiting for clean generation: {clean_dir}")
    while not clean_complete(clean_dir, args.expected_count):
        log(f"waiting for clean generation png_count={len(list(clean_dir.glob('*.png'))) if clean_dir.is_dir() else 0}/{args.expected_count}")
        time.sleep(args.poll_seconds)

    stages.append("clean_generation_complete")
    log("clean generation complete; starting Tree-Ring watermark generation")

    watermark_cmd = [
        sys.executable, "-u", "experiments/generate_watermarked_images.py",
        "--dataset_name", "diffusiondb",
        "--prompts_csv", str(prompts_csv),
        "--output_dir", str(root / "data/watermarked"),
        "--device", "cuda",
        "--gpu", args.watermark_gpu,
        "--require_free_gpu", "false",
        "--min_cpu_mem_gb", "64",
        "--warn_cpu_mem_gb", "96",
        "--max_process_ram_gb", "16",
        "--wm_types", "TR",
        "--num_pairs", str(args.expected_count),
        "--start_index", "0",
        "--seed", "42",
        "--modelid_target", "RedbeardNZ/stable-diffusion-2-1-base",
        "--scheduler_target", "DDIM",
        "--num_inference_steps_target", "50",
        "--guidance_scale_target", "7.5",
        "--resolution", "512",
        "--validate_before", "true",
    ]
    run(watermark_cmd)
    stages.append("watermark_generation_complete")

    log("building P1 manifest")
    manifest_cmd = [
        sys.executable, "raven_repro/scripts/build_diffusiondb_tr_manifest.py",
        "--generated-manifest", str(clean_dir / "manifest.json"),
        "--watermarked-metadata", str(root / "data/watermarked/diffusiondb/TR/metadata.csv"),
        "--watermarked-root", str(root / "data/watermarked/diffusiondb/TR"),
        "--output", str(root / "inputs/diffusiondb_tr_manifest_1001.csv"),
    ]
    run(manifest_cmd)
    stages.append("manifest_built")

    log("planning P1 shifts")
    plan_cmd = [
        sys.executable, "-u", "raven_repro/scripts/raven_p1_full.py", "plan-dataset",
        "--dataset", "diffusiondb",
        "--manifest", str(root / "inputs/diffusiondb_tr_manifest_1001.csv"),
        "--output-dir", str(root / "p1_1001"),
        "--expected-count", str(args.expected_count),
        "--plan-seed", "2026071401",
    ]
    run(plan_cmd)
    stages.append("p1_plan_complete")

    log("starting RAVEN attacked-watermarked on visible GPU " + args.attack_visible_gpu)
    attack_env = os.environ.copy()
    attack_env["CUDA_VISIBLE_DEVICES"] = args.attack_visible_gpu
    attack_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    attack_cmd = [
        sys.executable, "-u", "raven_repro/scripts/raven_p1_full.py", "attack",
        "--output-dir", str(root / "p1_1001"),
        "--device", "cuda",
        "--dtype", "float16",
    ]
    run(attack_cmd, env=attack_env)
    stages.append("p1_attack_watermarked_complete")

    log("preparing NFPA/TR attacked-clean workspace")
    nfpa_dir = root / "nfpa_1001"
    prepare_clean_cmd = [
        sys.executable, "-u", "raven_repro/scripts/raven_nfpa_tr_eval.py", "prepare",
        "--dataset", "diffusiondb",
        "--p1-dir", str(root / "p1_1001"),
        "--output-dir", str(nfpa_dir),
        "--expected-count", str(args.expected_count),
    ]
    run(prepare_clean_cmd)
    stages.append("nfpa_prepare_complete")

    log("starting RAVEN attacked-clean on visible GPU " + args.attack_visible_gpu)
    attack_clean_cmd = [
        sys.executable, "-u", "raven_repro/scripts/raven_nfpa_tr_eval.py", "attack-clean",
        "--output-dir", str(nfpa_dir),
        "--device", "cuda",
        "--dtype", "float16",
        "--resume",
    ]
    run(attack_clean_cmd, env=attack_env)
    stages.append("attacked_clean_complete")

    marker_path.write_text(utc_now() + "\n", encoding="utf-8")
    write_results(results_path, args.expected_count, stages)
    log("chain complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
