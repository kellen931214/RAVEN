#!/usr/bin/env python3
"""Build a strict verification manifest from immutable formal attack records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.eval_protocol import (  # noqa: E402
    FORMAL_ATTACK_CONFIG,
    load_formal_attack_config,
    assert_formal_debug_info,
    canonical_json_hash,
    provider_config,
    provider_config_hash,
    require_uniform_provider_config,
    sha256_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--metadata", type=Path, required=True, help="Snapshot index JSONL")
    parser.add_argument("--attack-records", type=Path, required=True, help="Formal attack record JSONL")
    parser.add_argument("--snapshot-manifest", type=Path, required=True, help="Snapshot index JSONL")
    parser.add_argument("--clean-dir", type=Path, default=None, help="Optional additional clean-root constraint")
    parser.add_argument("--watermark-config", type=Path, default=None, help="Optional config whose SHA is recorded")
    parser.add_argument("--attack-config", type=Path, default=None, help="Optional immutable formal attack config JSON")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def unique_index(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in result:
            raise ValueError(f"missing/duplicate run_id={run_id!r} in {source}")
        result[run_id] = row
    return result


def load_snapshots(index_path: Path) -> tuple[list[dict[str, str]], str]:
    index_rows = load_jsonl(index_path)
    result: list[dict[str, str]] = []
    seen_batches: set[str] = set()
    for entry in index_rows:
        batch_id = str(entry.get("batch_id", ""))
        if not batch_id or batch_id in seen_batches:
            raise ValueError(f"missing/duplicate snapshot batch_id={batch_id!r}")
        seen_batches.add(batch_id)
        snapshot = Path(entry["snapshot_path"]).resolve()
        if not snapshot.is_file() or sha256_path(snapshot) != entry["snapshot_sha256"]:
            raise RuntimeError(f"snapshot file/hash mismatch: {snapshot}")
        with snapshot.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(entry["row_count"]):
            raise RuntimeError(f"snapshot row-count mismatch: {snapshot}")
        for row in rows:
            row["snapshot_sha256"] = entry["snapshot_sha256"]
            row["source_manifest_sha256"] = entry["source_metadata_sha256"]
            result.append(row)
    unique_index(result, index_path)
    return result, sha256_path(index_path)


def checked_file(row: dict[str, Any], path_field: str, hash_field: str) -> Path:
    path = Path(str(row.get(path_field, ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"run_id={row.get('run_id')}: missing {path_field}: {path}")
    actual = sha256_path(path)
    if actual != row.get(hash_field):
        raise RuntimeError(
            f"run_id={row.get('run_id')}: {hash_field} mismatch: "
            f"recorded={row.get(hash_field)!r} actual={actual}"
        )
    return path


def normalized_prompt(value: Any) -> str:
    return " ".join(str(value or "").split())


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.metadata.resolve() != args.snapshot_manifest.resolve():
        raise ValueError("--metadata and --snapshot-manifest must identify the same immutable index")
    attack_config = (
        load_formal_attack_config(args.attack_config)
        if args.attack_config is not None
        else FORMAL_ATTACK_CONFIG
    )
    snapshots, snapshot_index_sha = load_snapshots(args.snapshot_manifest)
    snapshot_by_id = unique_index(snapshots, args.snapshot_manifest)
    attacks = unique_index(load_jsonl(args.attack_records), args.attack_records)
    if set(snapshot_by_id) != set(attacks):
        raise ValueError("snapshot and formal attack run-ID sets differ")

    output_rows: list[dict[str, Any]] = []
    for run_id, source in snapshot_by_id.items():
        attack = attacks[run_id]
        for key, expected in (("dataset", args.dataset), ("method", args.method)):
            if str(source.get(key)) != expected or str(attack.get(key)) != expected:
                raise ValueError(f"run_id={run_id}: {key} mismatch")
        if normalized_prompt(source.get("prompt")) != normalized_prompt(attack.get("prompt")):
            raise ValueError(f"run_id={run_id}: prompt mismatch")
        if str(source.get("prompt_id")) != str(attack.get("prompt_id")):
            raise ValueError(f"run_id={run_id}: prompt_id mismatch")
        for field in (
            "snapshot_sha256", "source_manifest_sha256", "clean_sha256",
            "watermarked_sha256", "provider_config_hash",
        ):
            if str(source.get(field)) != str(attack.get(field)):
                raise RuntimeError(f"run_id={run_id}: {field} mismatch")
        clean = checked_file(attack, "clean_path", "clean_sha256")
        watermarked = checked_file(attack, "watermarked_path", "watermarked_sha256")
        attacked = checked_file(attack, "attacked_path", "attacked_sha256")
        debug_path = checked_file(attack, "debug_info_path", "debug_info_sha256")
        pre_color_path = checked_file(
            attack, "pre_color_attacked_path", "pre_color_attacked_sha256"
        )
        if args.clean_dir is not None and args.clean_dir.resolve() not in clean.parents:
            raise ValueError(f"run_id={run_id}: clean path is outside --clean-dir")
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        transform_hash = assert_formal_debug_info(
            debug,
            planned_flow_dx_image_px=float(attack["planned_flow_dx_image_px"]),
            planned_flow_dy_image_px=float(attack["planned_flow_dy_image_px"]),
            attack_config=attack_config,
        )
        if attack.get("transform_config_hash") != transform_hash:
            raise RuntimeError(f"run_id={run_id}: transform config hash mismatch")
        if attack.get("model_id") != attack_config["model_id"]:
            raise RuntimeError(f"run_id={run_id}: model ID mismatch")
        if attack.get("model_revision") != attack_config["model_revision"]:
            raise RuntimeError(f"run_id={run_id}: model revision mismatch")
        provider = provider_config(args.method, source)
        expected_provider_hash = provider_config_hash(args.method, source)
        if expected_provider_hash != attack.get("provider_config_hash"):
            raise RuntimeError(f"run_id={run_id}: provider config hash mismatch")
        output_rows.append({
            "dataset": args.dataset,
            "method": args.method,
            "run_id": run_id,
            "prompt_id": source["prompt_id"],
            "prompt": source["prompt"],
            "source": source.get("source", ""),
            "clean_path": str(clean),
            "clean_sha256": attack["clean_sha256"],
            "watermarked_path": str(watermarked),
            "watermarked_sha256": attack["watermarked_sha256"],
            "attacked_path": str(attacked),
            "attacked_sha256": attack["attacked_sha256"],
            "debug_info_path": str(debug_path),
            "debug_info_sha256": attack["debug_info_sha256"],
            "pre_color_attacked_path": str(pre_color_path),
            "pre_color_attacked_sha256": attack["pre_color_attacked_sha256"],
            "model_id": attack["model_id"],
            "model_revision": attack["model_revision"],
            "attack_seed": attack["attack_seed"],
            "planned_flow_dx_image_px": attack["planned_flow_dx_image_px"],
            "planned_flow_dy_image_px": attack["planned_flow_dy_image_px"],
            "effective_source_flow_dx_image_px": attack["effective_source_flow_dx_image_px"],
            "effective_source_flow_dy_image_px": attack["effective_source_flow_dy_image_px"],
            "attack_config_hash": attack["attack_config_hash"],
            "transform_config_hash": transform_hash,
            "snapshot_sha256": attack["snapshot_sha256"],
            "snapshot_index_sha256": snapshot_index_sha,
            "source_manifest_sha256": attack["source_manifest_sha256"],
            "source_code_manifest_sha256": attack["source_code_manifest_sha256"],
            "provider_config": json.dumps(provider, sort_keys=True, separators=(",", ":")),
            "provider_config_hash": expected_provider_hash,
            "target_watermark_hash": attack.get("target_watermark_hash", ""),
            "pairing_sha256": attack.get("pairing_sha256", ""),
            "base_latent_seed": attack.get("base_latent_seed", ""),
            "base_latent_sha256": attack.get("base_latent_sha256", ""),
            "watermark_target_sha256": attack.get("watermark_target_sha256", ""),
            "watermark_mask_sha256": attack.get("watermark_mask_sha256", ""),
            "generation_config_sha256": attack.get("generation_config_sha256", ""),
            "watermark_config_sha256": attack.get("watermark_config_sha256", ""),
            **provider,
        })

    require_uniform_provider_config(args.method, output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
        handle.flush()
        os.fsync(handle.fileno())
    manifest_summary = {
        "row_count": len(output_rows),
        "run_ids_hash": canonical_json_hash({"run_ids": sorted(snapshot_by_id, key=int)}),
        "snapshot_index_sha256": snapshot_index_sha,
        "attack_records_sha256": sha256_path(args.attack_records),
        "source_code_manifest_sha256": output_rows[0]["source_code_manifest_sha256"],
        "watermark_config_sha256": sha256_path(args.watermark_config) if args.watermark_config else None,
        "manifest_sha256": sha256_path(args.output),
    }
    with args.output.with_suffix(".provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
