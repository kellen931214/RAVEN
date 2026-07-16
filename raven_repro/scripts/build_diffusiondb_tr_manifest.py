#!/usr/bin/env python
"""Build the RAVEN P1 manifest for existing DiffusionDB Tree-Ring pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-manifest", type=Path, default=Path("data/generated/diffusiondb/manifest.json"))
    parser.add_argument("--watermarked-metadata", type=Path, default=Path("data/watermarked/diffusiondb/TR/metadata.csv"))
    parser.add_argument("--watermarked-root", type=Path, default=Path("data/watermarked/diffusiondb/TR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated = json.loads(args.generated_manifest.read_text(encoding="utf-8"))
    with args.watermarked_metadata.open(newline="", encoding="utf-8-sig") as handle:
        metadata = {int(row["run_id"]): row for row in csv.DictReader(handle)}

    fields = [
        "dataset",
        "run_id",
        "prompt_id",
        "prompt",
        "source",
        "clean_path",
        "clean_sha256",
        "watermarked_path",
        "watermarked_sha256",
        "generation_seed",
        "attack_seed",
        "w_seed",
        "w_channel",
        "w_pattern",
        "w_mask_shape",
        "w_radius",
        "w_measurement",
        "w_injection",
    ]
    rows: list[dict[str, str]] = []
    for item in sorted(generated, key=lambda row: int(row["index"])):
        run_id = int(item["index"])
        meta = metadata[run_id]
        clean = Path(item["file"]).resolve()
        watermarked = (args.watermarked_root / f"{run_id:06d}" / "watermarked.png").resolve()
        if not clean.is_file():
            raise FileNotFoundError(clean)
        if not watermarked.is_file():
            raise FileNotFoundError(watermarked)
        rows.append(
            {
                "dataset": "diffusiondb",
                "run_id": str(run_id),
                "prompt_id": meta.get("prompt_id") or str(run_id),
                "prompt": item.get("prompt") or meta.get("prompt", ""),
                "source": meta.get("source") or "DiffusionDB",
                "clean_path": str(clean),
                "clean_sha256": sha256_path(clean),
                "watermarked_path": str(watermarked),
                "watermarked_sha256": sha256_path(watermarked),
                "generation_seed": str(42 + run_id),
                "attack_seed": str(42 + run_id),
                "w_seed": meta.get("w_seed") or "999999",
                "w_channel": meta.get("w_channel") or "3",
                "w_pattern": meta.get("w_pattern") or "ring",
                "w_mask_shape": meta.get("w_mask_shape") or "circle",
                "w_radius": meta.get("w_radius") or "10",
                "w_measurement": meta.get("w_measurement") or "l1_complex",
                "w_injection": meta.get("w_injection") or "complex",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
