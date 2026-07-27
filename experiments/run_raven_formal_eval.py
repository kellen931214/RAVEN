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
import random
import shutil
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
    ATTACK_CLEAN_METHODS,
    CLIP_CONFIG,
    FORMAL_ATTACK_CONFIG,
    METRIC_PROTOCOL_VERSION,
    assert_canonical_output_root,
    assert_formal_debug_info,
    canonical_json_hash,
    formal_output_root,
    formal_run_key,
    current_clip_provenance,
    formal_attack_config_hash,
    formal_runtime_provenance,
    load_and_validate_source_manifest,
    load_formal_attack_config,
    normalize_formal_attack_config,
    provider_config,
    provider_config_hash,
    require_uniform_clip_provenance,
    require_clean_git_worktree,
    require_uniform_provider_config,
    sha256_path,
    stage_fid_records,
    validate_resume_record,
)
from raven.metrics import pair_quality_metrics  # noqa: E402
from raven.pairing_provenance import (  # noqa: E402
    PAIRING_REQUIRED_FIELDS,
    audit_pairing_rows,
    gs_fields_for_rows,
)


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
SHIFT_MAGNITUDES = tuple(range(24, 33))
BALANCED_SHIFT_MAGNITUDES = (24, 27, 28, 29, 32)
SHIFT_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_bool_flag(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def require_valid_storage_mode(storage_light: bool, attack_clean_enabled: bool) -> None:
    """Fail closed: only full formal and storage-light modes are legal.

    Full formal:   storage_light=false, attack_clean_enabled=true
    Storage-light: storage_light=true,  attack_clean_enabled=false
    """
    if bool(storage_light) != (not bool(attack_clean_enabled)):
        raise RuntimeError(
            "storage-light mode requires both --storage-light "
            "and --attack-clean-enabled false; full formal mode requires "
            "neither --storage-light nor --attack-clean-enabled false"
        )


def storage_mode_metadata(
    *, method: str, expected_count: int, storage_light: bool, attack_clean_enabled: bool
) -> dict[str, Any]:
    """Derive the storage-light / attack-clean provenance flags.

    Kept pure so callers (run_config, aggregate, validate) and tests all agree on
    the exact classification. Defaults reproduce the legacy formal protocol.
    """
    method = method.upper()
    tr_recalibrated = bool(attack_clean_enabled) and method == "TR"
    if method not in ATTACK_CLEAN_METHODS:
        # attacked-clean recalibration is not part of this method's protocol at all,
        # and the per-sample input.png copy is redundant (the source watermarked image
        # is already SHA-pinned), so omitting both is the COMPLETE protocol here — not
        # a reduced "storage-light" variant of it.
        return {
            "storage_light": bool(storage_light),
            "attack_clean_enabled": False,
            "attacked_clean_count": 0,
            "recalibrated_metrics_available": False,
            "formal_protocol_complete": True,
            "result_classification": "formal_complete",
        }
    return {
        "storage_light": bool(storage_light),
        "attack_clean_enabled": bool(attack_clean_enabled),
        "attacked_clean_count": int(expected_count) if tr_recalibrated else 0,
        "recalibrated_metrics_available": tr_recalibrated,
        "formal_protocol_complete": bool(attack_clean_enabled) and not bool(storage_light),
        "result_classification": (
            "TR STORAGE-LIGHT / NO ATTACK-CLEAN" if storage_light else "formal_complete"
        ),
    }


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


def verified_image(
    path_value: str, run_id: str, label: str, image_size: list[int]
) -> tuple[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"run_id={run_id}: missing {label}: {path}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image.verify()
    with Image.open(path) as opened:
        decoded = ImageOps.exif_transpose(opened).convert("RGB")
        decoded.load()
        if list(decoded.size) != image_size:
            raise ValueError(
                f"run_id={run_id}: {label} size={decoded.size}, "
                f"expected={tuple(image_size)}"
            )
    return str(path), sha256_path(path)


def normalize_snapshot_row(
    row: dict[str, str], *, dataset: str, method: str, attack_config: dict[str, Any]
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
        first(row, "clean_path", "clean_image_path"), run_id, "clean image", attack_config["image_size"]
    )
    watermarked_path, watermarked_sha = verified_image(
        first(row, "watermarked_path", "watermarked_image_path"), run_id, "watermarked image", attack_config["image_size"]
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


def prepare_runtime_source_manifest(
    args: argparse.Namespace, output: Path
) -> tuple[Path, str, str]:
    """Copy and validate the exact source manifest used by this output root."""
    head = require_clean_git_worktree(REPO)
    _, supplied_sha = load_and_validate_source_manifest(args.source_manifest, repo_root=REPO)
    runtime_dir = output / "provenance"
    runtime_path = runtime_dir / "formal_source_manifest.json"
    runtime_sha_path = runtime_path.with_suffix(".sha256")
    if runtime_path.exists() or runtime_sha_path.exists():
        if not runtime_path.is_file() or not runtime_sha_path.is_file():
            raise RuntimeError("runtime source manifest is incomplete")
        _, runtime_sha = load_and_validate_source_manifest(runtime_path, repo_root=REPO)
        if runtime_sha != supplied_sha:
            raise RuntimeError(
                f"runtime/supplied source manifest mismatch: {runtime_sha} != {supplied_sha}"
            )
        return runtime_path, runtime_sha, head
    runtime_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source_manifest.resolve(), runtime_path)
    shutil.copyfile(args.source_manifest.resolve().with_suffix(".sha256"), runtime_sha_path)
    _, runtime_sha = load_and_validate_source_manifest(runtime_path, repo_root=REPO)
    if runtime_sha != supplied_sha:
        raise RuntimeError("copied runtime source manifest SHA changed")
    fsync_dir(runtime_dir)
    return runtime_path, runtime_sha, head


def run_config(
    args: argparse.Namespace,
    *,
    runtime_source_manifest: Path,
    source_manifest_sha: str,
    head: str,
) -> dict[str, Any]:
    attack_config = (
        load_formal_attack_config(args.attack_config)
        if args.attack_config is not None
        else normalize_formal_attack_config(FORMAL_ATTACK_CONFIG)
    )
    attack_config_source = (
        args.attack_config.resolve() if args.attack_config is not None else None
    )
    runtime = formal_runtime_provenance(
        scheduler_mode=attack_config["scheduler_mode"],
        device_class="cuda",
        attack_dtype="torch.float16",
    )
    return {
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "dataset": args.dataset,
        "method": args.method,
        "expected_count": args.expected_count,
        "source_metadata": str(args.source_metadata.resolve()),
        "source_metadata_sha256": sha256_path(args.source_metadata),
        "immutable_source_snapshot_index": (
            str(args.immutable_source_snapshot_index.resolve())
            if args.immutable_source_snapshot_index is not None
            else None
        ),
        "immutable_source_snapshot_index_sha256": (
            sha256_path(args.immutable_source_snapshot_index)
            if args.immutable_source_snapshot_index is not None
            else None
        ),
        "attack_config": attack_config,
        "attack_config_hash": formal_attack_config_hash(attack_config),
        "attack_config_source_path": (
            str(attack_config_source) if attack_config_source is not None else None
        ),
        "attack_config_source_sha256": (
            sha256_path(attack_config_source) if attack_config_source is not None else None
        ),
        "attack_runtime": runtime,
        **runtime,
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
                "attack_config_hash": formal_attack_config_hash(attack_config),
            }
        ),
        "git_head": head,
        "source_manifest_build_git_head": load_and_validate_source_manifest(
            runtime_source_manifest, repo_root=REPO
        )[0].get("git_head"),
        "source_code_manifest_path": str(runtime_source_manifest),
        "source_code_manifest_sha256": source_manifest_sha,
        "formal_source_config_hash": source_manifest_sha,
        **storage_mode_metadata(
            method=args.method,
            expected_count=args.expected_count,
            storage_light=args.storage_light,
            attack_clean_enabled=args.attack_clean_enabled,
        ),
    }


def initialize_or_validate_run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    path = output / "run_config.json"
    if not path.exists() and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty uninitialized output root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runtime_path, runtime_sha, head = prepare_runtime_source_manifest(args, output)
    expected = run_config(
        args,
        runtime_source_manifest=runtime_path,
        source_manifest_sha=runtime_sha,
        head=head,
    )
    if path.exists():
        if not args.resume:
            raise FileExistsError(
                f"formal output exists; use --resume only for an identical run: {output}"
            )
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != expected:
            raise RuntimeError(f"formal run config mismatch: stored={stored} expected={expected}")
        return stored
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


def load_immutable_source_rows(
    snapshot_index: Path, *, source_metadata: Path, expected_count: int
) -> tuple[list[dict[str, str]], str]:
    """Load a prior immutable cohort and fail closed on file or source drift."""
    entries = [
        json.loads(line)
        for line in snapshot_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"empty immutable source snapshot index: {snapshot_index}")
    source_sha = sha256_path(source_metadata)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.get("source_metadata_path") != str(source_metadata.resolve()):
            raise RuntimeError("immutable source snapshot metadata path mismatch")
        if entry.get("source_metadata_sha256") != source_sha:
            raise RuntimeError("immutable source snapshot metadata SHA mismatch")
        batch_path = Path(entry["snapshot_path"])
        if not batch_path.is_file() or sha256_path(batch_path) != entry.get("snapshot_sha256"):
            raise RuntimeError(f"immutable source snapshot file/hash drift: {batch_path}")
        batch_rows, partial = read_committed_csv(batch_path)
        if partial or len(batch_rows) != int(entry["row_count"]):
            raise RuntimeError(f"immutable source snapshot row-count drift: {batch_path}")
        for row in batch_rows:
            run_id = str(row.get("run_id", ""))
            if not run_id or run_id in seen:
                raise RuntimeError(f"immutable source duplicate/empty run_id={run_id!r}")
            seen.add(run_id)
            rows.append(row)
    if len(rows) < expected_count:
        raise RuntimeError(
            f"immutable source cohort has {len(rows)} rows, expected at least {expected_count}"
        )
    return rows[:expected_count], source_sha


def snapshot_stage(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.immutable_source_snapshot_index is None:
        source_rows, partial_tail_ignored = read_committed_csv(args.source_metadata)
        source_rows = source_rows[: args.expected_count]
    else:
        source_rows, source_sha = load_immutable_source_rows(
            args.immutable_source_snapshot_index,
            source_metadata=args.source_metadata,
            expected_count=args.expected_count,
        )
        if source_sha != config["source_metadata_sha256"]:
            raise RuntimeError("immutable source metadata SHA changed after run initialization")
        partial_tail_ignored = False
    normalized = [
        normalize_snapshot_row(
            row,
            dataset=args.dataset,
            method=args.method,
            attack_config=config["attack_config"],
        )
        for row in source_rows
    ]
    if args.method in {"TR", "GS"}:
        audit_pairing_rows(
            normalized, expected_count=len(normalized), verify_files=True
        )
    source_ids = [row["run_id"] for row in normalized]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate run IDs in committed source metadata")
    existing = {row["run_id"]: row for row in load_snapshot_rows(args.output_root)}
    source_by_id = {row["run_id"]: row for row in normalized}
    for run_id, old in existing.items():
        if run_id not in source_by_id:
            raise RuntimeError(f"live metadata lost already snapshotted run_id={run_id}")
        new = source_by_id[run_id]
        drift_fields = [
            "prompt",
            "prompt_id",
            "clean_path",
            "clean_sha256",
            "watermarked_path",
            "watermarked_sha256",
            "provider_config_hash",
        ]
        if args.method in {"TR", "GS"}:
            drift_fields.extend(PAIRING_REQUIRED_FIELDS)
        if args.method == "GS":
            # V1 and V2 GS cohorts carry different provenance field sets; use the
            # one this cohort's recorded protocol actually defines.
            drift_fields.extend(gs_fields_for_rows([new]))
        for field in dict.fromkeys(drift_fields):
            if str(old.get(field, "")) != str(new.get(field, "")):
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
            "immutable_source_snapshot_index": config["immutable_source_snapshot_index"],
            "immutable_source_snapshot_index_sha256": config[
                "immutable_source_snapshot_index_sha256"
            ],
            "snapshot_path": str(path.resolve()),
            "snapshot_sha256": sha256_path(path),
            "created_utc": utc_now(),
            "partial_tail_ignored": partial_tail_ignored,
        }
        append_jsonl(index_path, entry)
        batch_number += 1
    print(json.dumps({"snapshotted_total": len(existing) + len(new_rows), "new": len(new_rows)}))
    return 0


def planned_shift(
    index: int, run_id: str, attack_config: dict[str, Any]
) -> tuple[float, float, int]:
    base_seed = int(attack_config["base_seed"])
    try:
        numeric_id = int(run_id)
    except ValueError:
        numeric_id = int(canonical_json_hash({"run_id": run_id})[:8], 16)
    attack_seed = base_seed + numeric_id
    mode = attack_config["shift_plan_mode"]
    if mode == "zero":
        return 0.0, 0.0, attack_seed
    if mode == "balanced_deterministic_schedule":
        magnitude_x = BALANCED_SHIFT_MAGNITUDES[index % len(BALANCED_SHIFT_MAGNITUDES)]
        magnitude_y = BALANCED_SHIFT_MAGNITUDES[
            (index // len(BALANCED_SHIFT_MAGNITUDES)) % len(BALANCED_SHIFT_MAGNITUDES)
        ]
        sign_x, sign_y = SHIFT_SIGNS[index % len(SHIFT_SIGNS)]
        return float(sign_x * magnitude_x), float(sign_y * magnitude_y), attack_seed
    if mode != "paper_random_independent_axes":
        raise ValueError(f"unsupported shift plan mode: {mode}")
    rng_seed = int(
        canonical_json_hash(
            {"protocol": mode, "run_id": run_id, "attack_seed": attack_seed}
        )[:16],
        16,
    )
    rng = random.Random(rng_seed)
    dx = rng.choice(SHIFT_MAGNITUDES) * rng.choice((-1, 1))
    dy = rng.choice(SHIFT_MAGNITUDES) * rng.choice((-1, 1))
    return float(dx), float(dy), attack_seed


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
        "model_id": config["attack_config"]["model_id"],
        "model_revision": config["attack_config"]["model_revision"],
        "attack_seed": seed,
        "planned_flow_dx_image_px": dx,
        "planned_flow_dy_image_px": dy,
        "pairing_sha256": str(row.get("pairing_sha256", "")),
        "provider_config_hash": str(row.get("provider_config_hash", "")),
        "watermark_target_sha256": str(row.get("watermark_target_sha256", "")),
        "watermark_mask_sha256": str(row.get("watermark_mask_sha256", "")),
        **(
            {field: row.get(field, "") for field in gs_fields_for_rows([row])}
            if config["method"] == "GS"
            else {}
        ),
        **config["attack_runtime"],
    }


def attack_stage(args: argparse.Namespace, config: dict[str, Any], role: str) -> int:
    if role == "clean" and args.method != "TR":
        raise ValueError("attack-clean is required only for the formal TR/NFPA protocol")
    if role == "clean" and not config["attack_clean_enabled"]:
        raise ValueError(
            "attack-clean is disabled in storage-light mode "
            "(--attack-clean-enabled false); the TR result is NO ATTACK-CLEAN"
        )
    rows = load_snapshot_rows(args.output_root)
    if not rows:
        raise RuntimeError("no immutable snapshots are available")
    cache = args.output_root / "attack_cache" / config["attack_config_hash"]
    pending = []
    reused_count = 0
    for index, row in enumerate(rows):
        dx, dy, seed = planned_shift(index, str(row["run_id"]), config["attack_config"])
        item = cache / str(row["run_id"]) / role
        record_path = item / "record.json"
        expected = expected_resume_fields(row, role=role, dx=dx, dy=dy, seed=seed, config=config)
        if record_path.exists():
            if not args.resume:
                raise FileExistsError(record_path)
            validate_resume_record(
                json.loads(record_path.read_text(encoding="utf-8")),
                expected=expected,
                attack_config=config["attack_config"],
            )
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
        model_id=config["attack_config"]["model_id"],
        revision=config["attack_config"]["model_revision"],
        scheduler_mode=config["attack_config"]["scheduler_mode"],
        device=args.device,
        dtype="float16",
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
            steps=config["attack_config"]["steps"],
            strength=config["attack_config"]["strength"],
            guidance_scale=config["attack_config"]["guidance_scale"],
            shift_space=config["attack_config"]["shift_space"],
            warp_mode=config["attack_config"]["warp_mode"],
            padding_mode=config["attack_config"]["padding_mode"],
            latent_sampling_mode=config["attack_config"]["latent_sampling_mode"],
            shift_x=expected["planned_flow_dx_image_px"],
            shift_y=expected["planned_flow_dy_image_px"],
            view_guided_attention=config["attack_config"]["view_guided_attention"],
            color_transfer=config["attack_config"]["color_transfer"],
            seed=expected["attack_seed"],
            prompt=config["attack_config"]["prompt"],
            negative_prompt=config["attack_config"]["negative_prompt"],
            debug=False,
            inversion_mode=config["attack_config"]["inversion_mode"],
            save_input_copy=not config["storage_light"],
        )
        del attacked, image
        attacked_path = output_dir / "final_color_corrected.png"
        pre_color_path = output_dir / "view_guided_output.png"
        debug_path = output_dir / "debug_info.json"
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        transform_hash = assert_formal_debug_info(
            debug,
            planned_flow_dx_image_px=expected["planned_flow_dx_image_px"],
            planned_flow_dy_image_px=expected["planned_flow_dy_image_px"],
            attack_config=config["attack_config"],
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
            "pairing_sha256": row.get("pairing_sha256", ""),
            "base_latent_seed": row.get("base_latent_seed", ""),
            "base_latent_sha256": row.get("base_latent_sha256", ""),
            "watermark_target_sha256": row.get("watermark_target_sha256", ""),
            "watermark_mask_sha256": row.get("watermark_mask_sha256", ""),
            "generation_config_sha256": row.get("generation_config_sha256", ""),
            "watermark_config_sha256": row.get("watermark_config_sha256", ""),
            **(
                {field: row.get(field, "") for field in gs_fields_for_rows([row])}
                if args.method == "GS"
                else {}
            ),
            "pre_color_attacked_path": str(pre_color_path.resolve()),
            "pre_color_attacked_sha256": sha256_path(pre_color_path),
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
            "formal_attack_config": config["attack_config"],
            **config["attack_runtime"],
            "resolved_scheduler_config": debug["scheduler_config"],
            "resolved_scheduler_config_hash": debug["scheduler_config_hash"],
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
        dx, dy, seed = planned_shift(index, str(row["run_id"]), config["attack_config"])
        path = root / "attack_cache" / config["attack_config_hash"] / str(row["run_id"]) / role / "record.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_resume_record(
            record,
            expected=expected_resume_fields(row, role=role, dx=dx, dy=dy, seed=seed, config=config),
            attack_config=config["attack_config"],
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
        reference_definition="original watermarked images from immutable formal snapshots",
        attacked_definition="formal RAVEN final post-color-transfer attacked-watermarked images",
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
    manifest_command = [
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
    ]
    if config["attack_config_source_path"] is not None:
        manifest_command.extend(
            ["--attack-config", str(config["attack_config_source_path"])]
        )
    manifest_command.extend(["--output", str(manifest)])
    run_subprocess(manifest_command)
    verification = args.output_root / "verification"
    if args.method == "TR":
        tr_command = [
            sys.executable,
            str(RAVEN_REPRO / "scripts" / "raven_nfpa_tr_eval.py"),
            "score-formal",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(verification / "tr_nfpa"),
            "--device",
            args.device,
        ]
        if config["attack_clean_enabled"]:
            clean_records = require_complete_records(args, config, "clean")
            clean_path = verification / "attacked_clean_records.jsonl"
            with clean_path.open("x", encoding="utf-8") as handle:
                for row in clean_records:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            tr_command.extend(["--attacked-clean-records", str(clean_path)])
        else:
            # storage-light: no attacked-clean recalibration is performed.
            tr_command.append("--no-attacked-clean")
        run_subprocess(tr_command)
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
                config["attack_config"]["model_revision"],
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
        "status": (
            "storage_light_no_attack_clean_pending_validation"
            if config["storage_light"]
            else "formal_complete_pending_validation"
        ),
        "result_classification": config["result_classification"],
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
    snapshot_rows = load_snapshot_rows(args.output_root)
    if len(snapshot_rows) != args.expected_count:
        raise RuntimeError("final pairing audit requires the full expected snapshot cohort")
    pairing_summary = (
        audit_pairing_rows(
            snapshot_rows, expected_count=args.expected_count, verify_files=True
        )
        if args.method in {"TR", "GS"}
        else None
    )
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
    if "" in target_hashes:
        raise RuntimeError("missing target watermark hash")
    if args.method == "TR" and len(target_hashes) != 1:
        raise RuntimeError(f"mixed Tree-Ring target watermark hashes: {sorted(target_hashes)}")
    if args.method == "GS" and len(target_hashes) != args.expected_count:
        raise RuntimeError(
            f"GS requires one target per run: unique={len(target_hashes)} "
            f"expected={args.expected_count}"
        )
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
            ("pre_color_attacked_path", "pre_color_attacked_sha256"),
            ("debug_info_path", "debug_info_sha256"),
        ):
            path = Path(record[path_field])
            if not path.is_file() or sha256_path(path) != record[hash_field]:
                raise RuntimeError(f"run_id={record['run_id']}: {path_field} SHA mismatch")
    if args.method == "TR" and config["attack_clean_enabled"]:
        clean_records = require_complete_records(args, config, "clean")
        clean_by_id = {str(row["run_id"]): row for row in clean_records}
        if set(clean_by_id) != run_ids:
            raise RuntimeError("attacked-clean/attacked-watermarked run-ID sets differ")
        for watermarked in records:
            clean = clean_by_id[str(watermarked["run_id"])]
            for field in (
                "attack_config_hash", "source_code_manifest_sha256", "model_id",
                "model_revision", "attack_seed", "planned_flow_dx_image_px",
                "planned_flow_dy_image_px", "pairing_sha256", "transform_config_hash",
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
        source_target_hash = next(iter(target_hashes))
        source_mask_hashes = {str(row["watermark_mask_sha256"]) for row in records}
        if len(source_mask_hashes) != 1:
            raise RuntimeError("missing or mixed source watermark mask hashes")
        source_mask_hash = next(iter(source_mask_hashes))
        for score in score_rows:
            if score.get("detector_target_watermark_sha256") != source_target_hash:
                raise RuntimeError("TR detector/source target hash mismatch")
            if score.get("source_watermark_target_sha256") != source_target_hash:
                raise RuntimeError("TR score/source target hash mismatch")
            if score.get("detector_watermark_mask_sha256") != source_mask_hash:
                raise RuntimeError("TR detector/source mask hash mismatch")
            if score.get("source_watermark_mask_sha256") != source_mask_hash:
                raise RuntimeError("TR score/source mask hash mismatch")
        finite_keys = [
            "original_clean_threshold", "original_clean_actual_fpr",
            "before_tpr", "attacked_tpr_at_original_clean_threshold",
        ]
        if config["attack_clean_enabled"]:
            # attacked-clean recalibration is only present in the full protocol.
            finite_keys.extend([
                "attacked_clean_recalibrated_threshold", "attacked_clean_actual_fpr",
                "attacked_tpr_at_attacked_clean_recalibrated_threshold", "attacked_roc_auc",
            ])
        for protocol_name in ("full_precision_protocol", "nfpa_rounded2_protocol"):
            protocol = detector_payload[protocol_name]
            for key in finite_keys:
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
        "storage_light": config["storage_light"],
        "attack_clean_enabled": config["attack_clean_enabled"],
        "recalibrated_metrics_available": config["recalibrated_metrics_available"],
        "formal_protocol_complete": config["formal_protocol_complete"],
        "result_classification": config["result_classification"],
        "attacked_clean_count": config["attacked_clean_count"],
        "attacked_watermarked_count": args.expected_count,
        "verification_count": args.expected_count,
        "quality_count": args.expected_count,
        "clip_count": args.expected_count,
        "fid_reference_count": args.expected_count,
        "fid_attacked_count": args.expected_count,
        "provider_config_hash": next(iter(providers)),
        "target_watermark_hash": next(iter(target_hashes)),
        "pairing_audit": pairing_summary,
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
    parser.add_argument(
        "--immutable-source-snapshot-index", type=Path, default=None,
        help="Optional verified prior formal snapshot index; prevents reading a mutable cohort.",
    )
    parser.add_argument("--attack-config", type=Path, default=None)
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=(
            "Explicit run output root. Omit to use the canonical method-aware root "
            "outputs/<tr|gs>/<dataset>/<variant>/<run-key>. Any explicit value must "
            "still live under the canonical root for --method."
        ),
    )
    parser.add_argument(
        "--variant", default="formal",
        help="Variant folder under the canonical method/dataset root (default: formal).",
    )
    parser.add_argument(
        "--run-key", default=None,
        help=(
            "Stable run key (default: <source-manifest-short-sha>_<attack-config-short-hash>). "
            "Deliberately not a timestamp so re-runs resume the same root."
        ),
    )
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--storage-light",
        action="store_true",
        help=(
            "Skip the redundant per-sample input.png copy. Combined with "
            "--attack-clean-enabled false this produces a TR STORAGE-LIGHT / "
            "NO ATTACK-CLEAN result (no attacked-clean FPR recalibration)."
        ),
    )
    parser.add_argument(
        "--attack-clean-enabled",
        type=parse_bool_flag,
        default=True,
        help=(
            "Default true reproduces the full formal TR protocol. Set false to "
            "skip attack-clean, attacked-clean FPR, recalibrated threshold and "
            "recalibrated TPR (storage-light TR)."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16"])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--stage", required=True, choices=STAGES)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_count <= 0 or args.batch_size <= 0:
        raise ValueError("expected-count and batch-size must be positive")
    args.method = args.method.upper()
    # Methods outside the TR/NFPA protocol never run the attacked-clean branch and
    # never need the redundant per-sample input.png copy, so their storage-light /
    # no-attack-clean posture is the default rather than something to remember.
    if args.method not in ATTACK_CLEAN_METHODS:
        args.attack_clean_enabled = False
        args.storage_light = True
    require_valid_storage_mode(args.storage_light, args.attack_clean_enabled)
    if args.stage == "attack-clean" and not args.attack_clean_enabled:
        raise RuntimeError("attack-clean stage requested but attack-clean is disabled")
    if not args.device.startswith("cuda"):
        raise RuntimeError("formal RAVEN evaluation forbids CPU fallback")
    if args.dtype != "float16":
        raise RuntimeError("formal RAVEN evaluation requires float16")
    if args.output_root is None:
        # Canonical, method-aware, content-addressed root (never a timestamp dir).
        run_key = args.run_key or formal_run_key(
            sha256_path(args.source_manifest), formal_attack_config_hash(
                load_formal_attack_config(args.attack_config) if args.attack_config else None
            ),
        )
        args.output_root = formal_output_root(
            args.method, args.dataset, args.variant, run_key
        )
    args.output_root = assert_canonical_output_root(args.output_root, args.method)
    if args.gpu is not None:
        # Keep --gpu tied to the physical nvidia-smi index used in provenance.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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
