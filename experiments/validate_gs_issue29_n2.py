#!/usr/bin/env python3
"""Issue #29: canonical run root for exactly 2 real GS samples.

Records from untouched ``/workspace/RAVEN/data/gs/diffusiondb_shared_tr/GS/metadata.csv``
(rows run_id=0,1).  Live-image SHAs verified before writing.  No source data
modified; no fake metadata; no /tmp synthetic cohort.

Two derived fields needed by the unified detector contract (never in the
generation CSV):
* ``provider_config_hash`` — same formal hash ``require_uniform_provider_config``
  computes from the real CSV provider-config fields.
* ``scheduler`` — aliased from the CSV's ``scheduler_target`` column.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven import experiment_io  # noqa: E402
from raven.eval_protocol import require_uniform_provider_config  # noqa: E402

GS_DATA = Path("/workspace/RAVEN/data/gs/diffusiondb_shared_tr/GS")
CLEAN_DATA = Path("/workspace/RAVEN/data/clean/diffusiondb")
RUN_ROOT = REPO / "outputs" / "real_validation" / "gs" / "diffusiondb" / "n2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    rows = load_csv(GS_DATA / "metadata.csv")
    selected = [r for r in rows if str(r["run_id"]) in ("0", "1")]
    if len(selected) != 2:
        print(f"error: expected 2 rows, got {len(selected)}", file=sys.stderr)
        return 1

    # ---- derive provider_config_hash from real CSV fields ----
    _, provider_config_hash = require_uniform_provider_config("GS", selected)
    print(f"provider_config_hash = {provider_config_hash}")

    # ---- verify live image SHAs against the manifest ----
    table: list[tuple[str, str, Path, str]] = []
    for row in selected:
        run_id = str(row["run_id"])
        wm_live = GS_DATA / f"{int(run_id):06d}" / "watermarked.png"
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

    # ---- write config + records through canonical serializers ----
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_io.write_config(RUN_ROOT, {
        "method": "GS",
        "dataset": "diffusiondb_shared_tr",
        "metadata_path": str(GS_DATA / "metadata.csv"),
    })

    for run_id, role, path, _expected in table:
        row = next(r for r in selected if str(r["run_id"]) == run_id)
        record: dict[str, object] = {
            "run_id": run_id,
            "role": role,
            "dataset": row.get("dataset", row.get("dataset_name", "")),
            "method": "GS",
            "input_path": str(path),
            "output_path": str(experiment_io.output_image_path(RUN_ROOT, role, run_id)),
            "status": "complete",
            "prompt": row.get("prompt", ""),
            "prompt_id": row.get("prompt_id", ""),
            "protocol": row.get("protocol", ""),
            # Pipe identity — aliased from the CSV's canonical field name
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
            "scheduler": row["scheduler_target"],
            "resolution": row["resolution"],
            # Provenance digests
            "watermark_target_sha256": row["watermark_target_sha256"],
            "watermark_mask_sha256": row["watermark_mask_sha256"],
            "clean_sha256": row["clean_sha256"],
            "watermarked_sha256": row["watermarked_sha256"],
            # GS method-specific — verbatim from the manifest row
            "gs_protocol_mode": row["gs_protocol_mode"],
            "gs_secret_index": row["gs_secret_index"],
            "gs_message_sha256": row["gs_message_sha256"],
            "gs_key_sha256": row["gs_key_sha256"],
            "gs_nonce_sha256": row["gs_nonce_sha256"],
            "gs_secret_bundle_sha256": row["gs_secret_bundle_sha256"],
            "gs_detection_mode": row["gs_detection_mode"],
            # Derived from real CSV fields — same formal hash the legacy canonical
            # evaluator recorded.  No source CSV is modified.
            "provider_config_hash": provider_config_hash,
        }
        experiment_io.write_record(RUN_ROOT, role, run_id, record)

    experiment_io.rebuild_records_jsonl(RUN_ROOT)
    print(f"run root: {RUN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
