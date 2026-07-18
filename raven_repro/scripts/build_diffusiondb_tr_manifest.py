#!/usr/bin/env python
"""Build a fail-closed P1 manifest from paired Tree-Ring provenance."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raven.pairing_provenance import PAIRING_REQUIRED_FIELDS, audit_pairing_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-metadata", type=Path)
    # Backward-compatible spelling, but the file must contain paired provenance.
    parser.add_argument("--watermarked-metadata", type=Path)
    parser.add_argument("--expected-count", type=int, default=1001)
    parser.add_argument("--attack-seed-base", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.paired_metadata or args.watermarked_metadata
    if metadata_path is None:
        parser.error("--paired-metadata is required")
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    audit = audit_pairing_rows(rows, expected_count=args.expected_count, verify_files=True)

    fields = [
        "protocol",
        "dataset",
        "run_id",
        "prompt_id",
        "prompt",
        "prompt_sha256",
        "source",
        "clean_path",
        "clean_sha256",
        "watermarked_path",
        "watermarked_sha256",
        "base_latent_seed",
        "generation_seed",
        "base_latent_sha256",
        "clean_base_latent_sha256",
        "watermarked_base_latent_sha256",
        "watermarked_latent_sha256",
        "watermark_target_sha256",
        "watermark_mask_sha256",
        "generation_config_sha256",
        "watermark_config_sha256",
        "pairing_sha256",
        "injection_only_difference_verified",
        "injection_max_abs_error",
        "attack_seed",
        "w_seed",
        "w_channel",
        "w_pattern",
        "w_mask_shape",
        "w_radius",
        "w_measurement",
        "w_injection",
        "model_id",
        "model_revision",
    ]
    missing = set(PAIRING_REQUIRED_FIELDS) - set(fields)
    if missing:
        raise AssertionError(f"manifest schema missing required fields: {sorted(missing)}")

    manifest_rows: list[dict[str, object]] = []
    for row in rows:
        item = {field: row.get(field, "") for field in fields}
        item["attack_seed"] = args.attack_seed_base + int(row["run_id"])
        manifest_rows.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    audit_path = args.output.with_suffix(".pairing_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "rows": len(manifest_rows), "audit": audit},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
