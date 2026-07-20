#!/usr/bin/env python3
"""Single auditable entrypoint for formal RAVEN evaluation.

Stages are deliberately separate so every expensive operation consumes only an
immutable snapshot or committed per-sample records from the preceding stage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
RAVEN_REPRO = REPO / "raven_repro"
sys.path.insert(0, str(RAVEN_REPRO))

from raven.eval_protocol import (  # noqa: E402
    CLIP_CONFIG,
    FORMAL_ATTACK_CONFIG,
    METRIC_PROTOCOL_VERSION,
    assert_formal_debug_info,
    canonical_json_hash,
    current_clip_provenance,
    formal_attack_config_hash,
    load_and_validate_source_manifest,
    provider_config,
    provider_config_hash,
    require_uniform_clip_provenance,
    require_uniform_provider_config,
    sha256_path,
    stage_fid_records,
    validate_resume_record,
)
from raven.metrics import pair_quality_metrics  # noqa: E402


STAGES = (
    "snapshot",
    "attack-watermarked",
    "attack-clean",
    "verify",
    "quality",
    "fid",
    "clip",
    "aggregate",
    "validate",
)
SHIFT_MAGNITUDES = (24, 27, 28, 29, 32)
SHIFT_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_dir(path.parent)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_committed_csv(path: Path) -> tuple[list[dict[str, str]], bool]:
    data = path.read_bytes()
    if not data:
        raise ValueError(f"empty source metadata: {path}")
    had_partial_tail = not data.endswith((b"\n", b"\r"))
    complete = data if not had_partial_tail else data[: data.rfind(b"\n") + 1]
    text = complete.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"no committed rows in {path}")
    return rows, had_partial_tail


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def verified_image(path_value: str, run_id: str, label: str) -> tuple[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"run_id={run_id}: missing {label}: {path}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image.verify()
    with Image.open(path) as opened:
        decoded = ImageOps.exif_transpose(opened).convert("RGB")
        decoded.load()
        if list(decoded.size) != FORMAL_ATTACK_CONFIG["image_size"]:
            raise ValueError(
                f"run_id={run_id}: {label} size={decoded.size}, "
                f"expected={tuple(FORMAL_ATTACK_CONFIG['image_size'])}"
            )
    return str(path), sha256_path(path)


def normalize_snapshot_row(
    row: dict[str, str], *, dataset: str, method: str
) -> dict[str, Any]:
    run_id = first(row, "run_id", "sample_id", "id")
    if not run_id:
        raise ValueError("source row is missing run_id")
    recorded_dataset = first(row, "dataset", "dataset_name") or dataset
    recorded_method = (first(row, "method", "wm_type") or method).upper()
    if recorded_dataset != dataset or recorded_method != method:
        raise ValueError(
            f"run_id={run_id}: cohort mismatch dataset={recorded_dataset} method={recorded_method}"
        )
    prompt = first(row, "prompt", "source_prompt", "caption", "text")
    prompt_id = first(row, "prompt_id", "source_id")
    if not prompt or not prompt_id:
        raise ValueError(f"run_id={run_id}: prompt and prompt_id are required")
    clean_path, clean_sha = verified_image(
        first(row, "clean_path", "clean_image_path"), run_id, "clean image"
    )
    watermarked_path, watermarked_sha = verified_image(
        first(row, "watermarked_path", "watermarked_image_path"), run_id, "watermarked image"
    )
    config = provider_config(method, row)
    return {
        **row,
        "run_id": str(run_id),
        "dataset": dataset,
        "method": method,
        "prompt": prompt,
        "prompt_id": prompt_id,
        "clean_path": clean_path,
        "clean_sha256": clean_sha,
        "watermarked_path": watermarked_path,
        "watermarked_sha256": watermarked_sha,
        "provider_config": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "provider_config_hash": provider_config_hash(method, row),
    }


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    source_manifest, source_manifest_sha = load_and_validate_source_manifest(
        args.source_manifest, repo_root=REPO
    )
    head = git_head()
    if source_manifest.get("git_head") != head:
        raise RuntimeError(
            f"source manifest git HEAD mismatch: {source_manifest.get('git_head')} != {head}"
        )
    return {
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "dataset": args.dataset,
        "method": args.method,
        "expected_count": args.expected_count,
        "source_metadata": str(args.source_metadata.resolve()),
        "attack_config": FORMAL_ATTACK_CONFIG,
        "attack_config_hash": formal_attack_config_hash(),
        "detector_config_hash": canonical_json_hash(
            {"method": args.method, "target_fpr": 0.01, "score_source": "strict manifest"}
        ),
        "quality_config_hash": canonical_json_hash(
            {
                "reference": "watermarked input",
                "comparison": "final post-color-transfer attacked image",
                "overlap": "effective source flow inverse warp",
                "fid": "clean-fid watermarked-vs-raven",
                "clip": CLIP_CONFIG,
            }
        ),
        "git_head": head,
        "source_code_manifest_path": str(args.source_manifest.resolve()),
        "source_code_manifest_sha256": source_manifest_sha,
        "formal_source_config_hash": source_manifest_sha,
    }


def initialize_or_validate_run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    path = output / "run_config.json"
    expected = run_config(args)
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"formal output exists; use --resume only for an identical run: {output}")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != expected:
            raise RuntimeError(f"formal run config mismatch: stored={stored} expected={expected}")
        return stored
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty uninitialized output root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(path, expected)
    return expected


def load_snapshot_index(root: Path) -> list[dict[str, Any]]:
    path = root / "snapshots" / "snapshot_index.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({row["batch_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate snapshot batch_id")
    return rows


def load_snapshot_rows(root: Path) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for entry in load_snapshot_index(root):
        path = Path(entry["snapshot_path"])
        if not path.is_file() or sha256_path(path) != entry["snapshot_sha256"]:
            raise RuntimeError(f"snapshot file/hash drift: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(entry["row_count"]):
            raise RuntimeError(f"snapshot row count drift: {path}")
        for row in rows:
            run_id = str(row["run_id"])
            if run_id in seen:
                raise RuntimeError(f"duplicate snapshotted run_id={run_id}")
            seen.add(run_id)
            row["snapshot_sha256"] = entry["snapshot_sha256"]
            row["source_manifest_sha256"] = entry["source_metadata_sha256"]
            result.append(row)
    return result


def snapshot_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    source_rows, partial_tail_ignored = read_committed_csv(args.source_metadata)
    source_rows = source_rows[: args.expected_count]
    normalized = [
        normalize_snapshot_row(row, dataset=args.dataset, method=args.method) for row in source_rows
    ]
    source_ids = [row["run_id"] for row in normalized]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate run IDs in committed source metadata")
    existing = {row["run_id"]: row for row in load_snapshot_rows(args.output_root)}
    source_by_id = {row["run_id"]: row for row in normalized}
    for run_id, old in existing.items():
        if run_id not in source_by_id:
            raise RuntimeError(f"live metadata lost already snapshotted run_id={run_id}")
        new = source_by_id[run_id]
        for field in (
            "prompt",
            "prompt_id",
            "clean_path",
            "clean_sha256",
            "watermarked_path",
            "watermarked_sha256",
            "provider_config_hash",
        ):
            if str(old[field]) != str(new[field]):
                raise RuntimeError(f"snapshotted source drift run_id={run_id}: {field}")
    new_rows = [row for row in normalized if row["run_id"] not in existing]
    remaining = max(0, args.expected_count - len(existing))
    new_rows = new_rows[:remaining]
    snapshots = args.output_root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    index_path = snapshots / "snapshot_index.jsonl"
    batch_number = len(load_snapshot_index(args.output_root))
    source_sha = sha256_path(args.source_metadata)
    for start in range(0, len(new_rows), args.batch_size):
        batch = new_rows[start : start + args.batch_size]
        ids = [int(row["run_id"]) for row in batch]
        name = f"batch_{min(ids):06d}_{max(ids):06d}.csv"
        path = snapshots / name
        if path.exists():
            raise FileExistsError(path)
        fields: list[str] = []
        for row in batch:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(batch)
            handle.flush()
            os.fsync(handle.fileno())
        entry = {
            "batch_id": batch_number,
            "run_id_min": str(min(ids)),
            "run_id_max": str(max(ids)),
            "row_count": len(batch),
            "source_metadata_path": str(args.source_metadata.resolve()),
            "source_metadata_sha256": source_sha,
            "snapshot_path": str(path.resolve()),
            "snapshot_sha256": sha256_path(path),
            "created_utc": utc_now(),
            "partial_tail_ignored": partial_tail_ignored,
        }
        append_jsonl(index_path, entry)
        batch_number += 1
    print(json.dumps({"snapshotted_total": len(existing) + len(new_rows), "new": len(new_rows)}))
    return 0


def planned_shift(index: int, run_id: str, base_seed: int | None = None) -> tuple[float, float, int]:
    if base_seed is None:
        base_seed = int(FORMAL_ATTACK_CONFIG["base_seed"])
    magnitude_x = SHIFT_MAGNITUDES[index % len(SHIFT_MAGNITUDES)]
    magnitude_y = SHIFT_MAGNITUDES[(index // len(SHIFT_MAGNITUDES)) % len(SHIFT_MAGNITUDES)]
    sign_x, sign_y = SHIFT_SIGNS[index % len(SHIFT_SIGNS)]
    try:
        numeric_id = int(run_id)
    except ValueError:
        numeric_id = int(canonical_json_hash({"run_id": run_id})[:8], 16)
    return float(sign_x * magnitude_x), float(sign_y * magnitude_y), base_seed + numeric_id


def expected_resume_fields(
    row: dict[str, Any], *, role: str, dx: float, dy: float, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    input_path_field = "watermarked_path" if role == "watermarked" else "clean_path"
    input_sha_field = "watermarked_sha256" if role == "watermarked" else "clean_sha256"
    return {
        "run_id": str(row["run_id"]),
        "dataset": config["dataset"],
        "method": config["method"],
        "input_role": role,
        "input_path": str(row[input_path_field]),
        "input_sha256": str(row[input_sha_field]),
        "snapshot_sha256": str(row["snapshot_sha256"]),
        "source_manifest_sha256": str(row["source_manifest_sha256"]),
        "attack_config_hash": config["attack_config_hash"],
        "git_head": config["git_head"],
        "formal_source_config_hash": config["formal_source_config_hash"],
        "source_code_manifest_sha256": config["source_code_manifest_sha256"],
        "model_id": FORMAL_ATTACK_CONFIG["model_id"],
        "model_revision": FORMAL_ATTACK_CONFIG["model_revision"],
        "attack_seed": seed,
        "planned_flow_dx_image_px": dx,
        "planned_flow_dy_image_px": dy,
    }


def attack_stage(args: argparse.Namespace, config: dict[str, Any], role: str) -> int:
    if role == "clean" and args.method != "TR":
        raise ValueError("attack-clean is required only for the formal TR/NFPA protocol")
    rows = load_snapshot_rows(args.output_root)
    if not rows:
        raise RuntimeError("no immutable snapshots are available")
    cache = args.output_root / "attack_cache" / config["attack_config_hash"]
    pending = []
    reused_count = 0
    for index, row in enumerate(rows):
        dx, dy, seed = planned_shift(index, str(row["run_id"]))
        item = cache / str(row["run_id"]) / role
        record_path = item / "record.json"
        expected = expected_resume_fields(row, role=role, dx=dx, dy=dy, seed=seed, config=config)
        if record_path.exists():
            if not args.resume:
                raise FileExistsError(record_path)
            validate_resume_record(json.loads(record_path.read_text(encoding="utf-8")), expected=expected)
            reused_count += 1
            continue
        if item.exists():
            raise RuntimeError(f"uncommitted attack cache exists for run_id={row['run_id']}: {item}")
        pending.append((row, item, expected))
    if not pending:
        summary = {
            "role": role,
            "reused_count": reused_count,
            "recomputed_count": 0,
            "rejected_count": 0,
            "created_utc": utc_now(),
        }
        write_json_atomic(args.output_root / "state" / f"attack-{role}-last.json", summary)
        print(json.dumps(summary))
        return 0

    import torch
    from raven.pipeline_raven import RavenPipeline
    from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads

    limit_cpu_threads(1)
    guard = CpuMemoryGuard(24.0, 48.0, 40.0)
    guard.check(f"formal attack-{role} startup")
    pipe = RavenPipeline(
        model_id=FORMAL_ATTACK_CONFIG["model_id"],
        revision=FORMAL_ATTACK_CONFIG["model_revision"],
        device=args.device,
        dtype="float16" if args.device.startswith("cuda") else "float32",
    )
    for position, (row, item, expected) in enumerate(pending, start=1):
        input_path = Path(expected["input_path"])
        if sha256_path(input_path) != expected["input_sha256"]:
            raise RuntimeError(f"input SHA drift run_id={row['run_id']}")
        with Image.open(input_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
        output_dir = item / "output"
        attacked = pipe.run(
            input_image=image,
            output_dir=output_dir,
            steps=FORMAL_ATTACK_CONFIG["steps"],
            strength=FORMAL_ATTACK_CONFIG["strength"],
            guidance_scale=FORMAL_ATTACK_CONFIG["guidance_scale"],
            shift_space=FORMAL_ATTACK_CONFIG["shift_space"],
            warp_mode=FORMAL_ATTACK_CONFIG["warp_mode"],
            padding_mode=FORMAL_ATTACK_CONFIG["padding_mode"],
            latent_sampling_mode=FORMAL_ATTACK_CONFIG["latent_sampling_mode"],
            shift_x=expected["planned_flow_dx_image_px"],
            shift_y=expected["planned_flow_dy_image_px"],
            view_guided_attention=FORMAL_ATTACK_CONFIG["view_guided_attention"],
            color_transfer=FORMAL_ATTACK_CONFIG["color_transfer"],
            seed=expected["attack_seed"],
            prompt=FORMAL_ATTACK_CONFIG["prompt"],
            negative_prompt=FORMAL_ATTACK_CONFIG["negative_prompt"],
            debug=False,
            inversion_mode=FORMAL_ATTACK_CONFIG["inversion_mode"],
        )
        del attacked, image
        attacked_path = output_dir / "final_color_corrected.png"
        debug_path = output_dir / "debug_info.json"
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        transform_hash = assert_formal_debug_info(
            debug,
            planned_flow_dx_image_px=expected["planned_flow_dx_image_px"],
            planned_flow_dy_image_px=expected["planned_flow_dy_image_px"],
        )
        record = {
            **expected,
            "metric_protocol_version": METRIC_PROTOCOL_VERSION,
            "clean_path": row["clean_path"],
            "clean_sha256": row["clean_sha256"],
            "watermarked_path": row["watermarked_path"],
            "watermarked_sha256": row["watermarked_sha256"],
            "prompt": row["prompt"],
            "prompt_id": row["prompt_id"],
            "provider_config": json.loads(row["provider_config"]),
            "provider_config_hash": row["provider_config_hash"],
            "target_watermark_hash": row.get("watermark_target_sha256", ""),
            "attacked_path": str(attacked_path.resolve()),
            "attacked_sha256": sha256_path(attacked_path),
            "debug_info_path": str(debug_path.resolve()),
            "debug_info_sha256": sha256_path(debug_path),
            "exact_ddim_timestep": int(debug["exact_timestep"]),
            "effective_source_dx_latent": float(debug["effective_source_dx_latent"]),
            "effective_source_dy_latent": float(debug["effective_source_dy_latent"]),
            "effective_source_flow_dx_image_px": float(debug["effective_source_flow_dx_image_px"]),
            "effective_source_flow_dy_image_px": float(debug["effective_source_flow_dy_image_px"]),
            "effective_visual_shift_dx_image_px": float(debug["effective_visual_shift_dx_image_px"]),
            "effective_visual_shift_dy_image_px": float(debug["effective_visual_shift_dy_image_px"]),
            "formal_attack_config": FORMAL_ATTACK_CONFIG,
            "transform_config_hash": transform_hash,
            "formal_config_hash": expected["attack_config_hash"],
            "source_code_manifest_sha": expected["source_code_manifest_sha256"],
            "seed": expected["attack_seed"],
            "planned_dx": expected["planned_flow_dx_image_px"],
            "planned_dy": expected["planned_flow_dy_image_px"],
            "effective_source_dx_image_px": float(debug["effective_source_flow_dx_image_px"]),
            "effective_source_dy_image_px": float(debug["effective_source_flow_dy_image_px"]),
            "effective_visual_dx_image_px": float(debug["effective_visual_shift_dx_image_px"]),
            "effective_visual_dy_image_px": float(debug["effective_visual_shift_dy_image_px"]),
            "output_sha256": sha256_path(attacked_path),
            "debug_sha256": sha256_path(debug_path),
            "transform_hash": transform_hash,
            "created_utc": utc_now(),
        }
        write_json_exclusive(item / "record.json", record)
        guard.check(f"attack-{role} {position}/{len(pending)}")
    del pipe
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    summary = {
        "role": role,
        "reused_count": reused_count,
        "recomputed_count": len(pending),
        "rejected_count": 0,
        "created_utc": utc_now(),
    }
    write_json_atomic(args.output_root / "state" / f"attack-{role}-last.json", summary)
    print(json.dumps(summary))
    return 0


def attack_records(root: Path, config: dict[str, Any], role: str) -> list[dict[str, Any]]:
    rows = load_snapshot_rows(root)
    records = []
    for index, row in enumerate(rows):
        dx, dy, seed = planned_shift(index, str(row["run_id"]))
        path = root / "attack_cache" / config["attack_config_hash"] / str(row["run_id"]) / role / "record.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_resume_record(
            record,
            expected=expected_resume_fields(row, role=role, dx=dx, dy=dy, seed=seed, config=config),
        )
        records.append(record)
    return records


def require_complete_records(args: argparse.Namespace, config: dict[str, Any], role: str) -> list[dict[str, Any]]:
    rows = attack_records(args.output_root, config, role)
    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"final metric stage requires {args.expected_count} immutable records, found {len(rows)}"
        )
    if len({row["run_id"] for row in rows}) != args.expected_count:
        raise RuntimeError("duplicate attack record run IDs")
    return rows


def quality_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    records = require_complete_records(args, config, "watermarked")
    root = args.output_root / "metrics" / "quality" / config["quality_config_hash"]
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    rows = []
    for record in records:
        with Image.open(record["watermarked_path"]) as reference, Image.open(record["attacked_path"]) as attacked:
            metric = pair_quality_metrics(
                reference.convert("RGB"),
                attacked.convert("RGB"),
                record["effective_source_flow_dx_image_px"],
                record["effective_source_flow_dy_image_px"],
            )
        rows.append(
            {
                "run_id": record["run_id"],
                "quality_reference": "watermarked input",
                "overlap_protocol": "inverse warp using effective source flow from actual grid",
                "post_color_vs_watermarked_overlap_psnr": metric["overlap_psnr"],
                "post_color_vs_watermarked_overlap_ssim": metric["overlap_ssim"],
                **metric,
            }
        )
    with (root / "quality_records.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return 0


def fid_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    records = require_complete_records(args, config, "watermarked")
    fid_root, manifest = stage_fid_records(
        records,
        formal_output=args.output_root,
        quality_config_hash=config["quality_config_hash"],
        expected_count=args.expected_count,
    )
    from raven.quality import clean_fid

    result = clean_fid(
        fid_root / "reference_watermarked", fid_root / "attacked", device=args.device
    )
    import importlib.metadata

    result.update(
        {
            "clean_fid_version": importlib.metadata.version("clean-fid"),
            "mode": "clean",
            "image_count": args.expected_count,
            "reference_definition": manifest["reference_definition"],
            "attacked_definition": manifest["attacked_definition"],
            "manifest_hash": manifest["manifest_hash"],
            "manifest_file_sha256": manifest["manifest_file_sha256"],
            "config_hash": config["quality_config_hash"],
            "metric_name": "per_method_fid_watermarked_vs_raven",
        }
    )
    write_json_exclusive(fid_root / "fid_result.json", result)
    return 0


def clip_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    records = require_complete_records(args, config, "watermarked")
    root = args.output_root / "metrics" / "clip" / config["quality_config_hash"]
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    from raven.quality import openclip_text_image_scores

    result = openclip_text_image_scores(
        [row["attacked_path"] for row in records],
        [row["prompt"] for row in records],
        device=args.device,
        model_name=CLIP_CONFIG["clip_model_name"],
        pretrained=CLIP_CONFIG["clip_pretrained"],
    )
    provenance = current_clip_provenance()
    rows = [
        {"run_id": record["run_id"], "score": score, **provenance}
        for record, score in zip(records, result["scores"])
    ]
    require_uniform_clip_provenance(rows)
    with (root / "clip_records.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_json_exclusive(root / "clip_result.json", {**result, **provenance})
    return 0


def run_subprocess(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def verify_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    wm_records = require_complete_records(args, config, "watermarked")
    records_path = args.output_root / "attack_records_watermarked.jsonl"
    if records_path.exists():
        raise FileExistsError(records_path)
    with records_path.open("x", encoding="utf-8") as handle:
        for row in wm_records:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    snapshot_index = args.output_root / "snapshots" / "snapshot_index.jsonl"
    manifest = args.output_root / "verification" / "manifest.csv"
    run_subprocess(
        [
            sys.executable,
            str(RAVEN_REPRO / "scripts" / "build_verification_manifest.py"),
            "--dataset",
            args.dataset,
            "--method",
            args.method,
            "--metadata",
            str(snapshot_index),
            "--attack-records",
            str(records_path),
            "--snapshot-manifest",
            str(snapshot_index),
            "--output",
            str(manifest),
        ]
    )
    verification = args.output_root / "verification"
    if args.method == "TR":
        clean_records = require_complete_records(args, config, "clean")
        clean_path = verification / "attacked_clean_records.jsonl"
        with clean_path.open("x", encoding="utf-8") as handle:
            for row in clean_records:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        run_subprocess(
            [
                sys.executable,
                str(RAVEN_REPRO / "scripts" / "raven_nfpa_tr_eval.py"),
                "score-formal",
                "--manifest",
                str(manifest),
                "--attacked-clean-records",
                str(clean_path),
                "--output-dir",
                str(verification / "tr_nfpa"),
                "--device",
                args.device,
            ]
        )
    else:
        scores = verification / "scores.csv"
        run_subprocess(
            [
                sys.executable,
                str(RAVEN_REPRO / "scripts" / "extract_verification_scores.py"),
                "--method",
                args.method,
                "--metadata",
                str(manifest),
                "--output",
                str(scores),
                "--model-revision",
                FORMAL_ATTACK_CONFIG["model_revision"],
                "--device",
                args.device,
            ]
        )
        run_subprocess(
            [
                sys.executable,
                str(RAVEN_REPRO / "scripts" / "evaluate_verification.py"),
                "--method",
                args.method,
                "--records",
                str(scores),
                "--output-json",
                str(verification / "verification_result.json"),
            ]
        )
    return 0


def aggregate_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    require_complete_records(args, config, "watermarked")
    detector = (
        args.output_root / "verification" / "tr_nfpa" / "aggregate_results.json"
        if args.method == "TR"
        else args.output_root / "verification" / "verification_result.json"
    )
    quality = args.output_root / "metrics" / "quality" / config["quality_config_hash"] / "quality_records.jsonl"
    fid = args.output_root / "metrics" / "fid" / config["quality_config_hash"] / "fid_result.json"
    clip = args.output_root / "metrics" / "clip" / config["quality_config_hash"] / "clip_result.json"
    for path in (detector, quality, fid, clip):
        if not path.is_file():
            raise FileNotFoundError(path)
    aggregate = {
        **config,
        "status": "formal_complete_pending_validation",
        "N": args.expected_count,
        "sample_count": args.expected_count,
        "gate_only": args.expected_count < 1000,
        "paper_comparable": False,
        "statistical_validity": (
            "gate_only_insufficient_for_1pct_fpr"
            if args.expected_count < 100
            else "requires_final_protocol_review"
        ),
        "detector_result": str(detector.resolve()),
        "detector_result_sha256": sha256_path(detector),
        "quality_records": str(quality.resolve()),
        "quality_records_sha256": sha256_path(quality),
        "fid_result": json.loads(fid.read_text(encoding="utf-8")),
        "clip_result": json.loads(clip.read_text(encoding="utf-8")),
    }
    write_json_exclusive(args.output_root / "formal_aggregate.json", aggregate)
    return 0


def validate_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    records = require_complete_records(args, config, "watermarked")
    run_ids = {str(row["run_id"]) for row in records}
    hashes = {row["attack_config_hash"] for row in records}
    providers = {row["provider_config_hash"] for row in records}
    if hashes != {config["attack_config_hash"]}:
        raise RuntimeError(f"mixed attack config hashes: {sorted(hashes)}")
    if len(providers) != 1:
        raise RuntimeError(f"mixed provider configs: {sorted(providers)}")
    require_uniform_provider_config(args.method, records)
    target_hashes = {str(row.get("target_watermark_hash", "")) for row in records}
    if "" in target_hashes or len(target_hashes) != 1:
        raise RuntimeError(f"missing or mixed target watermark hashes: {sorted(target_hashes)}")
    alias_pairs = (
        ("formal_config_hash", "attack_config_hash"),
        ("source_code_manifest_sha", "source_code_manifest_sha256"),
        ("seed", "attack_seed"),
        ("planned_dx", "planned_flow_dx_image_px"),
        ("planned_dy", "planned_flow_dy_image_px"),
        ("effective_source_dx_image_px", "effective_source_flow_dx_image_px"),
        ("effective_source_dy_image_px", "effective_source_flow_dy_image_px"),
        ("effective_visual_dx_image_px", "effective_visual_shift_dx_image_px"),
        ("effective_visual_dy_image_px", "effective_visual_shift_dy_image_px"),
        ("output_sha256", "attacked_sha256"),
        ("debug_sha256", "debug_info_sha256"),
        ("transform_hash", "transform_config_hash"),
    )
    for record in records:
        if record.get("source_code_manifest_sha256") != config["source_code_manifest_sha256"]:
            raise RuntimeError(f"run_id={record['run_id']}: source manifest mismatch")
        for left, right in alias_pairs:
            if record.get(left) != record.get(right):
                raise RuntimeError(f"run_id={record['run_id']}: alias mismatch {left}/{right}")
        for path_field, hash_field in (
            ("input_path", "input_sha256"),
            ("clean_path", "clean_sha256"),
            ("watermarked_path", "watermarked_sha256"),
            ("attacked_path", "attacked_sha256"),
            ("debug_info_path", "debug_info_sha256"),
        ):
            path = Path(record[path_field])
            if not path.is_file() or sha256_path(path) != record[hash_field]:
                raise RuntimeError(f"run_id={record['run_id']}: {path_field} SHA mismatch")
    if args.method == "TR":
        clean_records = require_complete_records(args, config, "clean")
        clean_by_id = {str(row["run_id"]): row for row in clean_records}
        if set(clean_by_id) != run_ids:
            raise RuntimeError("attacked-clean/attacked-watermarked run-ID sets differ")
        for watermarked in records:
            clean = clean_by_id[str(watermarked["run_id"])]
            for field in (
                "attack_config_hash", "source_code_manifest_sha256", "model_id",
                "model_revision", "attack_seed", "planned_flow_dx_image_px",
                "planned_flow_dy_image_px", "transform_config_hash",
            ):
                if clean.get(field) != watermarked.get(field):
                    raise RuntimeError(
                        f"run_id={watermarked['run_id']}: attacked pair {field} mismatch"
                    )
    quality_path = (
        args.output_root / "metrics" / "quality" / config["quality_config_hash"]
        / "quality_records.jsonl"
    )
    quality_rows = [json.loads(line) for line in quality_path.read_text().splitlines() if line]
    if len(quality_rows) != args.expected_count or {str(row["run_id"]) for row in quality_rows} != run_ids:
        raise RuntimeError("quality record count/run-ID mismatch")
    for row in quality_rows:
        for field in (
            "post_color_vs_watermarked_overlap_psnr",
            "post_color_vs_watermarked_overlap_ssim",
        ):
            value = float(row[field])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite formal quality metric: {field}={value}")
    clip_path = (
        args.output_root / "metrics" / "clip" / config["quality_config_hash"]
        / "clip_records.jsonl"
    )
    clip_rows = [json.loads(line) for line in clip_path.read_text().splitlines() if line]
    if len(clip_rows) != args.expected_count or {str(row["run_id"]) for row in clip_rows} != run_ids:
        raise RuntimeError("CLIP record count/run-ID mismatch")
    require_uniform_clip_provenance(clip_rows)
    if any(not math.isfinite(float(row["score"])) for row in clip_rows):
        raise RuntimeError("non-finite CLIP score")
    fid_root = args.output_root / "metrics" / "fid" / config["quality_config_hash"]
    fid_manifest_path = fid_root / "fid_manifest.json"
    fid_sha_path = fid_root / "fid_manifest.sha256"
    fid_manifest = json.loads(fid_manifest_path.read_text(encoding="utf-8"))
    if sha256_path(fid_manifest_path) != fid_sha_path.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("FID manifest file SHA mismatch")
    stored_manifest_hash = fid_manifest.pop("manifest_hash")
    if canonical_json_hash(fid_manifest) != stored_manifest_hash:
        raise RuntimeError("FID canonical manifest hash mismatch")
    fid_ids = {str(row["run_id"]) for row in fid_manifest["records"]}
    if fid_ids != run_ids or fid_manifest["image_count"] != args.expected_count:
        raise RuntimeError("FID manifest count/run-ID mismatch")
    expected_names = {row["staged_name"] for row in fid_manifest["records"]}
    reference_dir = fid_root / "reference_watermarked"
    attacked_dir = fid_root / "attacked"
    if {path.name for path in reference_dir.iterdir()} != expected_names:
        raise RuntimeError("FID reference staging contains missing or extra files")
    if {path.name for path in attacked_dir.iterdir()} != expected_names:
        raise RuntimeError("FID attacked staging contains missing or extra files")
    for row in fid_manifest["records"]:
        reference = reference_dir / row["staged_name"]
        attacked = attacked_dir / row["staged_name"]
        if not reference.is_file() or not attacked.is_file():
            raise RuntimeError(f"broken FID staging link run_id={row['run_id']}")
        if sha256_path(reference) != row["reference_source_sha256"]:
            raise RuntimeError(f"FID reference SHA mismatch run_id={row['run_id']}")
        if sha256_path(attacked) != row["attacked_source_sha256"]:
            raise RuntimeError(f"FID attacked SHA mismatch run_id={row['run_id']}")
    detector = (
        args.output_root / "verification" / "tr_nfpa" / "aggregate_results.json"
        if args.method == "TR"
        else args.output_root / "verification" / "verification_result.json"
    )
    detector_payload = json.loads(detector.read_text(encoding="utf-8"))
    if args.method == "TR":
        score_path = args.output_root / "verification" / "tr_nfpa" / "l1_scores.jsonl"
        score_rows = [json.loads(line) for line in score_path.read_text().splitlines() if line]
        if len(score_rows) != args.expected_count or {str(row["run_id"]) for row in score_rows} != run_ids:
            raise RuntimeError("TR verification count/run-ID mismatch")
        if {row["provider_config_hash"] for row in score_rows} != providers:
            raise RuntimeError("TR verification provider config mismatch")
        if len({row["target_watermark_hash"] for row in score_rows}) != 1:
            raise RuntimeError("mixed TR detector target hashes")
        for protocol_name in ("full_precision_protocol", "nfpa_rounded2_protocol"):
            protocol = detector_payload[protocol_name]
            for key in (
                "original_clean_threshold", "original_clean_actual_fpr",
                "attacked_clean_recalibrated_threshold", "attacked_clean_actual_fpr",
                "before_tpr", "attacked_tpr_at_original_clean_threshold",
                "attacked_tpr_at_attacked_clean_recalibrated_threshold", "attacked_roc_auc",
            ):
                if not math.isfinite(float(protocol[key])):
                    raise RuntimeError(f"non-finite TR aggregate {protocol_name}.{key}")
    aggregate = args.output_root / "formal_aggregate.json"
    if not aggregate.is_file():
        raise FileNotFoundError(aggregate)
    aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
    if aggregate_payload.get("source_code_manifest_sha256") != config["source_code_manifest_sha256"]:
        raise RuntimeError("formal aggregate source manifest mismatch")
    if aggregate_payload.get("sample_count") != args.expected_count:
        raise RuntimeError("formal aggregate sample count mismatch")
    if args.expected_count < 1000 and (
        aggregate_payload.get("gate_only") is not True
        or aggregate_payload.get("paper_comparable") is not False
    ):
        raise RuntimeError("small formal gate is not marked gate-only/non-paper-comparable")
    validation = {
        "status": "validated_formal_result",
        "N": len(records),
        "duplicate_run_ids": 0,
        "missing_paths": 0,
        "sha_mismatches": 0,
        "config_mismatches": 0,
        "mixed_provider_configs": 0,
        "mixed_attack_config_hashes": 0,
        "nan_count": 0,
        "inf_count": 0,
        "mixed_clip_provenance": 0,
        "mixed_source_manifests": 0,
        "mixed_model_revisions": 0,
        "attacked_clean_count": args.expected_count if args.method == "TR" else 0,
        "attacked_watermarked_count": args.expected_count,
        "verification_count": args.expected_count,
        "quality_count": args.expected_count,
        "clip_count": args.expected_count,
        "fid_reference_count": args.expected_count,
        "fid_attacked_count": args.expected_count,
        "provider_config_hash": next(iter(providers)),
        "target_watermark_hash": next(iter(target_hashes)),
        "source_code_manifest_sha256": config["source_code_manifest_sha256"],
        "gate_only": args.expected_count < 1000,
        "paper_comparable": False,
        "validated_utc": utc_now(),
    }
    write_json_exclusive(args.output_root / "VALIDATED.json", validation)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--stage", required=True, choices=STAGES)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_count <= 0 or args.batch_size <= 0:
        raise ValueError("expected-count and batch-size must be positive")
    args.method = args.method.upper()
    args.output_root = args.output_root.resolve()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = initialize_or_validate_run(args)
    if args.stage == "snapshot":
        return snapshot_stage(args, config)
    if args.stage == "attack-watermarked":
        return attack_stage(args, config, "watermarked")
    if args.stage == "attack-clean":
        return attack_stage(args, config, "clean")
    if args.stage == "verify":
        return verify_stage(args, config)
    if args.stage == "quality":
        return quality_stage(args, config)
    if args.stage == "fid":
        return fid_stage(args, config)
    if args.stage == "clip":
        return clip_stage(args, config)
    if args.stage == "aggregate":
        return aggregate_stage(args, config)
    if args.stage == "validate":
        return validate_stage(args, config)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
