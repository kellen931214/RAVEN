#!/usr/bin/env python3
"""Issue #29: canonical run root for exactly 2 real T2S samples.

T2S uses only the watermarked role; no clean cohort.
Records from untouched ``/workspace/RAVEN/data/t2s/diffusiondb_shared_tr/T2S/metadata.csv``
(rows run_id=0,1).
"""
from __future__ import annotations

import csv, hashlib, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))
from raven import experiment_io  # noqa: E402

T2S_DATA = Path("/workspace/RAVEN/data/t2s/diffusiondb_shared_tr/T2S")
RUN_ROOT = REPO / "outputs" / "real_validation" / "t2s" / "diffusiondb" / "n2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with (T2S_DATA / "metadata.csv").open(newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    selected = [r for r in rows if str(r["run_id"]) in ("0", "1")]
    if len(selected) != 2:
        print(f"error: expected 2 rows, got {len(selected)}", file=sys.stderr)
        return 1

    table: list[tuple[str, str, Path, str]] = []
    for row in selected:
        run_id = str(row["run_id"])
        wm_live = T2S_DATA / f"{int(run_id):06d}" / "watermarked.png"
        if not wm_live.is_file():
            print(f"error: wm image missing for run_id={run_id}", file=sys.stderr)
            return 1
        table.append((run_id, "watermarked", wm_live, row["watermarked_sha256"]))

    for run_id, role, path, expected in table:
        actual = sha256(path)
        if actual != expected:
            print(f"error: SHA mismatch {role} run_id={run_id}: expected={expected} actual={actual}", file=sys.stderr)
            return 1
        print(f"{run_id} {role:12s} {path}  sha={actual}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_io.write_config(RUN_ROOT, {
        "method": "T2S",
        "dataset": "diffusiondb_shared_tr",
        "metadata_path": str(T2S_DATA / "metadata.csv"),
    })

    for run_id, role, path, _expected in table:
        row = next(r for r in selected if str(r["run_id"]) == run_id)
        record: dict[str, object] = {
            "run_id": run_id, "role": role,
            "dataset": row.get("dataset", row.get("dataset_name", "")),
            "method": "T2S", "input_path": str(path),
            "output_path": str(experiment_io.output_image_path(RUN_ROOT, role, run_id)),
            "status": "complete",
            "prompt": row.get("prompt", ""), "prompt_id": row.get("prompt_id", ""),
            "protocol": row.get("protocol", ""),
            "model_id": row["model_id"], "model_revision": row["model_revision"],
            "scheduler": row["scheduler_target"], "resolution": row["resolution"],
            "watermark_target_sha256": row["watermark_target_sha256"],
            "watermark_mask_sha256": row["watermark_mask_sha256"],
            "watermarked_sha256": row["watermarked_sha256"],
            "t2s_protocol_mode": row["t2s_protocol_mode"],
            "t2s_rng_mode": row["t2s_rng_mode"],
            "t2s_inversion_mode": row["t2s_inversion_mode"],
            "t2s_num_inversion_steps": row["t2s_num_inversion_steps"],
            "t2s_watermark_id": row["t2s_watermark_id"],
            "t2s_state_path": row["t2s_state_path"],
            "t2s_provider_config_sha256": row["t2s_provider_config_sha256"],
        }
        experiment_io.write_record(RUN_ROOT, role, run_id, record)

    experiment_io.rebuild_records_jsonl(RUN_ROOT)
    print(f"run root: {RUN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
