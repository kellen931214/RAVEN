#!/usr/bin/env python3
"""Issue #28: build canonical run root for exactly 2 real TR samples.

Records come from the untouched real TR cohort metadata
(``/workspace/RAVEN/data/tr/diffusiondb/metadata.csv``), rows 0 and 1.
Only two transformations are applied:

* stale manifest paths are replaced by the verified live image paths
  (same SHA-256, files verified before writing);
* ``w_pattern_const = 0.0`` is recovered unchanged from the authoritative
  ``watermark_config.shard-001-of-002.json`` of the same formal TR run
  (required by the TR detector contract, absent from the CSV).

The run root is written with the repository's canonical serializers
(``experiment_io.write_config`` / ``write_record`` / ``rebuild_records_jsonl``).
Attacked images do not exist, so ``output.png`` is intentionally absent and
the evaluator's preflight reports ``failed_missing_image`` for the attacked
cohorts — nothing is fabricated.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven import experiment_io  # noqa: E402

TR_DATA = Path("/workspace/RAVEN/data/tr/diffusiondb")
CLEAN_DATA = Path("/workspace/RAVEN/data/clean/diffusiondb")
RUN_ROOT = REPO / "outputs" / "real_validation" / "tr" / "diffusiondb" / "n2"

WATERMARK_CONFIG = TR_DATA / "watermark_config.shard-001-of-002.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    rows = load_csv(TR_DATA / "metadata.csv")
    selected = [r for r in rows if r["run_id"] in ("0", "1")]
    if len(selected) != 2:
        print(f"error: expected 2 rows, got {len(selected)}", file=sys.stderr)
        return 1

    wpc_cfg = json.loads(WATERMARK_CONFIG.read_text(encoding="utf-8"))
    w_pattern_const = str(wpc_cfg["w_pattern_const"])

    # ---- verify live image identity against the manifest before writing ----
    import hashlib

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    table: list[tuple[str, str, Path, str]] = []
    for row in selected:
        run_id = row["run_id"]
        wm_live = TR_DATA / f"{int(run_id):06d}" / "watermarked.png"
        clean_live = CLEAN_DATA / f"{int(run_id):06d}.png"
        if not wm_live.is_file() or not clean_live.is_file():
            print(f"error: live image missing for run_id={run_id}", file=sys.stderr)
            return 1
        table.append((run_id, "watermarked", wm_live, row["watermarked_sha256"]))
        table.append((run_id, "clean", clean_live, row["clean_sha256"]))

    for run_id, role, path, expected in table:
        actual = sha256(path)
        if actual != expected:
            print(f"error: SHA mismatch {role} run_id={run_id}: "
                  f"expected={expected} actual={actual}", file=sys.stderr)
            return 1
        print(f"{run_id} {role:12s} {path}  sha={actual}")

    # ---- write config + records through the canonical serializers ----
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_io.write_config(RUN_ROOT, {
        "method": "TR",
        "dataset": "diffusiondb",
        "metadata_path": str(TR_DATA / "metadata.csv"),
    })

    for run_id, role, path, _expected in table:
        row = next(r for r in selected if r["run_id"] == run_id)
        record = {
            "run_id": run_id,
            "role": role,
            "dataset": row["dataset"],
            "method": "TR",
            "input_path": str(path),
            "output_path": str(experiment_io.output_image_path(RUN_ROOT, role, run_id)),
            "status": "complete",
            "prompt": row["prompt"],
            "prompt_id": row["prompt_id"],
            "protocol": row["protocol"],
            # Provider/identity fields, values verbatim from the manifest row
            "w_seed": row["w_seed"],
            "w_channel": row["w_channel"],
            "w_radius": row["w_radius"],
            "w_pattern": row["w_pattern"],
            "w_mask_shape": row["w_mask_shape"],
            "w_measurement": row["w_measurement"],
            "w_injection": row["w_injection"],
            "w_pattern_const": w_pattern_const,
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
            "scheduler": row["scheduler_target"],
            "steps": row["num_inference_steps_target"],
            "resolution": row["resolution"],
            "watermark_target_sha256": row["watermark_target_sha256"],
            "watermark_mask_sha256": row["watermark_mask_sha256"],
            "clean_sha256": row["clean_sha256"],
            "watermarked_sha256": row["watermarked_sha256"],
            "base_latent_sha256": row["base_latent_sha256"],
        }
        experiment_io.write_record(RUN_ROOT, role, run_id, record)

    experiment_io.rebuild_records_jsonl(RUN_ROOT)
    print(f"run root: {RUN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
