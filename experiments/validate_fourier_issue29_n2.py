#!/usr/bin/env python3
"""Issue #29: canonical run roots for exactly 2 real samples (RID, HSTR, HSQR).

Each method reads from its untouched generation metadata CSV, verifies live
image SHAs, and writes a canonical ``records.jsonl`` + ``config.json`` through
the repository's serializers.
"""
from __future__ import annotations

import csv, hashlib, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))
from raven import experiment_io  # noqa: E402

CLEAN_DATA = Path("/workspace/RAVEN/data/clean/diffusiondb")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(method: str) -> int:
    m = method.lower()
    data_root = Path(f"/workspace/RAVEN/data/{m}/diffusiondb_shared_tr/{method}")
    run_root = REPO / "outputs" / "real_validation" / m / "diffusiondb" / "n2"

    with (data_root / "metadata.csv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    selected = [r for r in rows if str(r["run_id"]) in ("0", "1")]
    if len(selected) != 2:
        print(f"[{method}] error: expected 2 rows, got {len(selected)}", file=sys.stderr)
        return 1

    table: list[tuple[str, str, Path, str]] = []
    for row in selected:
        run_id = str(row["run_id"])
        wm_live = data_root / f"{int(run_id):06d}" / "watermarked.png"
        clean_live = CLEAN_DATA / f"{int(run_id):06d}.png"
        if not wm_live.is_file() or not clean_live.is_file():
            print(f"[{method}] error: image missing for run_id={run_id}", file=sys.stderr)
            return 1
        table.append((run_id, "watermarked", wm_live, row["watermarked_sha256"]))
        table.append((run_id, "clean", clean_live, row["clean_sha256"]))

    for run_id, role, path, expected in table:
        actual = sha256(path)
        if actual != expected:
            print(f"[{method}] error: SHA mismatch {role} run_id={run_id}", file=sys.stderr)
            return 1
        print(f"[{method}] {run_id} {role:12s} {path}  sha={actual}")

    run_root.mkdir(parents=True, exist_ok=True)
    experiment_io.write_config(run_root, {
        "method": method,
        "dataset": "diffusiondb_shared_tr",
        "metadata_path": str(data_root / "metadata.csv"),
    })

    for run_id, role, path, _expected in table:
        row = next(r for r in selected if str(r["run_id"]) == run_id)
        record: dict[str, object] = {
            "run_id": run_id, "role": role,
            "dataset": row.get("dataset", row.get("dataset_name", "")),
            "method": method, "input_path": str(path),
            "output_path": str(experiment_io.output_image_path(run_root, role, run_id)),
            "status": "complete",
            "prompt": row.get("prompt", ""), "prompt_id": row.get("prompt_id", ""),
            "protocol": row.get("protocol", ""),
            "model_id": row["model_id"], "model_revision": row["model_revision"],
            "scheduler": row["scheduler_target"], "resolution": row["resolution"],
            "watermark_target_sha256": row["watermark_target_sha256"],
            "watermark_mask_sha256": row["watermark_mask_sha256"],
            "clean_sha256": row["clean_sha256"], "watermarked_sha256": row["watermarked_sha256"],
            f"{m}_protocol_mode": row[f"{m}_protocol_mode"],
            f"{m}_bundle_dir": row[f"{m}_bundle_dir"],
            f"{m}_bundle_config_sha256": row[f"{m}_bundle_config_sha256"],
            f"{m}_selected_pattern_sha256": row[f"{m}_selected_pattern_sha256"],
            f"{m}_mask_sha256": row[f"{m}_mask_sha256"],
            f"{m}_key_index": row[f"{m}_key_index"],
        }
        experiment_io.write_record(run_root, role, run_id, record)

    experiment_io.rebuild_records_jsonl(run_root)
    print(f"[{method}] run root: {run_root}")
    return 0


def main() -> int:
    for method in ("RID", "HSTR", "HSQR"):
        rc = build(method)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
