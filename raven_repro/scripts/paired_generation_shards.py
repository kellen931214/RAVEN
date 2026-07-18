#!/usr/bin/env python
"""Prepare and merge fail-closed multi-GPU paired-generation shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.pairing_provenance import audit_pairing_rows, sha256_path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def shard_suffix(num_shards: int, shard_index: int) -> str:
    return f".shard-{shard_index:03d}-of-{num_shards:03d}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def canonical_shard_fieldnames(fieldnames: list[str]) -> list[str]:
    required = {"run_id", "num_shards", "shard_index"}
    if not required.issubset(fieldnames):
        raise ValueError(f"shard metadata missing fields: {sorted(required - set(fieldnames))}")
    ordered = [
        field for field in fieldnames if field not in {"num_shards", "shard_index"}
    ]
    insertion = ordered.index("run_id") + 1
    ordered[insertion:insertion] = ["num_shards", "shard_index"]
    return ordered


def order_shard_row(
    row: dict[str, object], num_shards: int, shard_index: int
) -> dict[str, object]:
    normalized = dict(row)
    normalized["num_shards"] = num_shards
    normalized["shard_index"] = shard_index
    fields = canonical_shard_fieldnames(list(normalized))
    return {field: normalized.get(field, "") for field in fields}


def read_shard_csv_recover_schema(
    path: Path, num_shards: int, shard_index: int
) -> tuple[list[dict[str, str]], int]:
    if not path.is_file():
        return [], 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            stored_header = next(reader)
        except StopIteration:
            return [], 0
        canonical_header = canonical_shard_fieldnames(stored_header)
        layouts = [stored_header]
        if canonical_header != stored_header:
            layouts.append(canonical_header)
        rows: list[dict[str, str]] = []
        repaired = 0
        for csv_line, values in enumerate(reader, start=2):
            candidates: list[tuple[list[str], dict[str, str]]] = []
            errors: list[str] = []
            for layout in layouts:
                if len(values) != len(layout):
                    errors.append(f"{len(values)} values for {len(layout)} fields")
                    continue
                candidate = dict(zip(layout, values))
                try:
                    if int(candidate.get("num_shards", 0)) != num_shards:
                        raise ValueError("num_shards mismatch")
                    if int(candidate.get("shard_index", -1)) != shard_index:
                        raise ValueError("shard_index mismatch")
                    audit_pairing_rows([candidate], expected_count=1, verify_files=True)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                candidates.append((layout, candidate))
            if len(candidates) != 1:
                raise ValueError(
                    f"{path}: metadata schema is not uniquely recoverable at CSV line "
                    f"{csv_line}; candidates={len(candidates)} errors={errors}"
                )
            layout, candidate = candidates[0]
            rows.append(candidate)
            if layout != stored_header:
                repaired += 1
    return rows, repaired


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with temp.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def expected_run_ids(prompts_csv: Path, expected_count: int) -> list[int]:
    selected: list[int] = []
    with prompts_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"prompt CSV has no header: {prompts_csv}")
        prompt_field = "prompt" if "prompt" in reader.fieldnames else reader.fieldnames[0]
        for row_index, row in enumerate(reader):
            if not str(row.get(prompt_field) or "").strip():
                continue
            selected.append(row_index)
            if len(selected) == expected_count:
                break
    if len(selected) != expected_count:
        raise ValueError(
            f"expected {expected_count} non-empty prompts, found {len(selected)} in {prompts_csv}"
        )
    return selected


def _same_record(left: dict[str, str], right: dict[str, str]) -> bool:
    fields = (
        "pairing_sha256",
        "base_latent_sha256",
        "clean_sha256",
        "watermarked_sha256",
        "watermark_target_sha256",
        "generation_config_sha256",
        "watermark_config_sha256",
    )
    return all(str(left.get(field, "")) == str(right.get(field, "")) for field in fields)


def collect_recorded_rows(
    method_dir: Path, num_shards: int
) -> tuple[dict[int, dict[str, str]], list[dict]]:
    sources: list[tuple[Path, int | None]] = [(method_dir / "metadata.csv", None)]
    sources.extend(
        (
            method_dir / f"metadata{shard_suffix(num_shards, index)}.csv",
            index,
        )
        for index in range(num_shards)
    )
    recorded: dict[int, dict[str, str]] = {}
    schema_repairs: list[dict] = []
    for source, shard_index in sources:
        if shard_index is None:
            rows = read_csv(source)
            repaired = 0
        else:
            rows, repaired = read_shard_csv_recover_schema(
                source, num_shards, shard_index
            )
        if repaired:
            schema_repairs.append(
                {
                    "source": str(source.resolve()),
                    "rows_reinterpreted_with_canonical_schema": repaired,
                    "reason": "stored header order differed from canonical shard row order",
                }
            )
        if not rows:
            continue
        audit_pairing_rows(rows, expected_count=len(rows), verify_files=True)
        for row in rows:
            run_id = int(row["run_id"])
            existing = recorded.get(run_id)
            if existing is not None and not _same_record(existing, row):
                raise ValueError(f"conflicting pairing provenance for run_id={run_id}")
            recorded[run_id] = row
    if recorded:
        audit_pairing_rows(
            [recorded[run_id] for run_id in sorted(recorded)],
            expected_count=len(recorded),
            verify_files=True,
        )
    return recorded, schema_repairs


def quarantine_orphans(
    root: Path,
    expected_ids: list[int],
    recorded_ids: set[int],
) -> dict:
    method_dir = root / "data" / "watermarked" / "diffusiondb" / "TR"
    clean_dir = root / "data" / "generated" / "diffusiondb"
    events: list[dict] = []
    quarantine = root / "invalid" / "orphaned_unrecorded" / stamp()
    for run_id in expected_ids:
        if run_id in recorded_ids:
            continue
        clean = clean_dir / f"{run_id:06d}.png"
        wm_dir = method_dir / f"{run_id:06d}"
        watermarked = wm_dir / "watermarked.png"
        if not clean.exists() and not wm_dir.exists():
            continue
        event = {
            "run_id": run_id,
            "reason": "image output existed without a committed pairing metadata row",
            "clean_existed": clean.is_file(),
            "watermarked_existed": watermarked.is_file(),
            "clean_sha256": sha256_path(clean) if clean.is_file() else None,
            "watermarked_sha256": sha256_path(watermarked) if watermarked.is_file() else None,
        }
        if clean.exists():
            destination = quarantine / "clean" / clean.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(clean), str(destination))
            event["quarantined_clean_path"] = str(destination.resolve())
        if wm_dir.exists():
            destination = quarantine / "watermarked" / wm_dir.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wm_dir), str(destination))
            event["quarantined_watermarked_dir"] = str(destination.resolve())
        events.append(event)
    payload = {
        "created_utc": utc_now(),
        "status": "ORPHANED_UNRECORDED_QUARANTINED" if events else "NO_ORPHANS",
        "events": events,
    }
    if events:
        write_json(quarantine / "INVALID_REASON.json", payload)
    return payload


def prepare(root: Path, prompts_csv: Path, expected_count: int, num_shards: int) -> dict:
    expected_ids = expected_run_ids(prompts_csv, expected_count)
    expected_set = set(expected_ids)
    method_dir = root / "data" / "watermarked" / "diffusiondb" / "TR"
    method_dir.mkdir(parents=True, exist_ok=True)
    recorded, schema_repairs = collect_recorded_rows(method_dir, num_shards)
    unexpected = sorted(set(recorded) - expected_set)
    if unexpected:
        raise ValueError(f"recorded run_ids are outside formal prompt selection: {unexpected[:10]}")
    quarantine = quarantine_orphans(root, expected_ids, set(recorded))
    shard_counts: dict[str, int] = {}
    for index in range(num_shards):
        rows = []
        for run_id in expected_ids:
            if run_id % num_shards != index or run_id not in recorded:
                continue
            row = order_shard_row(recorded[run_id], num_shards, index)
            rows.append(row)
        path = method_dir / f"metadata{shard_suffix(num_shards, index)}.csv"
        write_csv_atomic(path, rows)
        shard_counts[str(index)] = len(rows)
    payload = {
        "created_utc": utc_now(),
        "passed": True,
        "expected_count": expected_count,
        "num_shards": num_shards,
        "recorded_count": len(recorded),
        "shard_recorded_counts": shard_counts,
        "schema_repairs": schema_repairs,
        "quarantine": quarantine,
    }
    write_json(method_dir / "shard_prepare_audit.json", payload)
    return payload


def merge(root: Path, prompts_csv: Path, expected_count: int, num_shards: int) -> dict:
    expected_ids = expected_run_ids(prompts_csv, expected_count)
    expected_set = set(expected_ids)
    method_dir = root / "data" / "watermarked" / "diffusiondb" / "TR"
    combined: list[dict[str, str]] = []
    shard_counts: dict[str, int] = {}
    for index in range(num_shards):
        path = method_dir / f"metadata{shard_suffix(num_shards, index)}.csv"
        rows = read_csv(path)
        assigned_ids = [run_id for run_id in expected_ids if run_id % num_shards == index]
        audit_pairing_rows(rows, expected_count=len(assigned_ids), verify_files=True)
        actual_ids = {int(row["run_id"]) for row in rows}
        if actual_ids != set(assigned_ids):
            missing = sorted(set(assigned_ids) - actual_ids)
            extra = sorted(actual_ids - set(assigned_ids))
            raise ValueError(f"shard {index} assignment mismatch missing={missing[:10]} extra={extra[:10]}")
        for row in rows:
            if int(row.get("num_shards", 0)) != num_shards or int(row.get("shard_index", -1)) != index:
                raise ValueError(f"shard provenance mismatch run_id={row.get('run_id')}")
        shard_counts[str(index)] = len(rows)
        combined.extend(rows)
    combined.sort(key=lambda row: int(row["run_id"]))
    if {int(row["run_id"]) for row in combined} != expected_set:
        raise ValueError("combined shard run_ids do not equal formal prompt selection")
    audit = audit_pairing_rows(combined, expected_count=expected_count, verify_files=True)
    write_csv_atomic(method_dir / "metadata.csv", combined)
    write_json(method_dir / "pairing_audit.json", audit)
    payload = {
        "created_utc": utc_now(),
        "passed": True,
        "expected_count": expected_count,
        "num_shards": num_shards,
        "shard_counts": shard_counts,
        "pairing_audit": audit,
        "merged_metadata": str((method_dir / "metadata.csv").resolve()),
    }
    write_json(method_dir / "shard_merge_audit.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "merge"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompts-csv", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    operation = prepare if args.action == "prepare" else merge
    result = operation(
        args.root.resolve(),
        args.prompts_csv.resolve(),
        args.expected_count,
        args.num_shards,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
