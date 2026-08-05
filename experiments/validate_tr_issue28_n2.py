#!/usr/bin/env python3
"""Issue #28: canonical run root for exactly 2 real TR samples.

Records come from the untouched real TR cohort metadata
(``/workspace/RAVEN/data/tr/diffusiondb/metadata.csv``), rows 0 and 1.
Two transformations are applied:

* stale manifest paths are replaced by the verified live image paths
  (same SHA-256, files verified before writing);
* ``w_pattern_const = 0.0`` is recovered from the authoritative
  ``watermark_config.shard-001-of-002.json`` of the same formal TR run
  (required by the TR detector contract, absent from the CSV);
* a derived ``metadata.wpc.csv`` is written alongside the run root so the
  actual metadata CSV consumed by the evaluator carries the resolved
  ``w_pattern_const`` column — never silently invented.

The run root is written with the repository's canonical serializers
(``experiment_io.write_config`` / ``write_record`` / ``rebuild_records_jsonl``).
Attacked images do not exist, so ``output.png`` is intentionally absent and
the evaluator's preflight reports ``failed_missing_image`` for the attacked
cohorts — nothing is fabricated.

Threshold provenance
--------------------
The threshold is calibrated by the evaluator from the **local original-clean**
cohort scores via ``raven.metrics.calibrate_threshold(clean_scores,
target_fpr=0.01)``.  With only 2 clean samples the max-FP budget is 0, so the
resulting threshold is a **local validation threshold** — it can validate that
the scoring pipeline is wired correctly but must NOT be interpreted as a
production or full-cohort threshold.  The same calibration helper is used for
every cohort size; the label reflects the statistical strength.

Determinism
-----------
Re-run the same root twice and compare scored rows by ``(run_id, cohort)``
key after normalising ``evaluated_utc`` (timestamp) and all absolute
run-root / image paths.
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

TR_DATA = Path("/workspace/RAVEN/data/tr/diffusiondb")
CLEAN_DATA = Path("/workspace/RAVEN/data/clean/diffusiondb")
RUN_ROOT = REPO / "outputs" / "real_validation" / "tr" / "diffusiondb" / "n2"

WATERMARK_CONFIG = TR_DATA / "watermark_config.shard-001-of-002.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_derived_metadata(
    source_meta: Path,
    wpc_config: Path,
    dest: Path,
) -> Path:
    """Write a derived CSV with ``w_pattern_const`` recovered from provenance.

    The source metadata CSV predates the issue #22 schema column.  The
    authoritative watermark config JSON of the same formal TR run pins the
    value; this function recovers it — never invents it.

    If ``w_pattern_const`` **already exists** in the source CSV the value
    is validated against the watermark config.  A mismatch fails immediately
    (provenance conflict); a match writes the column through unchanged.
    No silent overwrite of a conflicting value.
    """
    wpc = str(json.loads(wpc_config.read_text(encoding="utf-8"))["w_pattern_const"])
    rows = list(csv.DictReader(source_meta.open(newline="", encoding="utf-8")))
    fields = list(rows[0].keys())

    if "w_pattern_const" in fields:
        # column already present — validate, never silently overwrite
        existing = sorted(set(r["w_pattern_const"] for r in rows))
        if existing != [wpc]:
            raise ValueError(
                f"source CSV already carries w_pattern_const={existing} "
                f"but watermark config pins {wpc!r} — provenance conflict; "
                f"aborting rather than silently overwriting"
            )
        # match — pass through unchanged
        return source_meta

    # column absent — add it from authoritative provenance
    output_fields = fields + ["w_pattern_const"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=output_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "w_pattern_const": wpc})
    return dest


def main() -> int:
    rows = load_csv(TR_DATA / "metadata.csv")
    selected = [r for r in rows if r["run_id"] in ("0", "1")]
    if len(selected) != 2:
        print(f"error: expected 2 rows, got {len(selected)}", file=sys.stderr)
        return 1

    wpc_cfg = json.loads(WATERMARK_CONFIG.read_text(encoding="utf-8"))
    w_pattern_const = str(wpc_cfg["w_pattern_const"])

    # ---- verify live image identity against the manifest before writing ----
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

    # ---- derived metadata: w_pattern_const from authoritative provenance ----
    derived_meta = _build_derived_metadata(
        TR_DATA / "metadata.csv",
        WATERMARK_CONFIG,
        RUN_ROOT / "metadata.wpc.csv",
    )

    # ---- write config + records through the canonical serializers ----
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    experiment_io.write_config(RUN_ROOT, {
        "method": "TR",
        "dataset": "diffusiondb",
        "metadata_path": str(derived_meta),
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

    # ---- record provenance digests for determinism analysis ----
    prov = {
        "source_metadata_csv_sha256": sha256(TR_DATA / "metadata.csv"),
        "derived_metadata_csv_sha256": sha256(derived_meta),
        "watermark_config_sha256": sha256(WATERMARK_CONFIG),
        "record_count": 4,
        "scored_cohorts_expected": ["original_clean", "original_watermarked"],
        "attacked_cohorts_expected_failed": ["attacked_clean", "attacked_watermarked"],
        "threshold_type": "local_validation_threshold_2_sample",
        "threshold_provenance": "raven.metrics.calibrate_threshold(local_original_clean, target_fpr=0.01)",
        "max_fp_budget_for_n2": 0,
    }
    (RUN_ROOT / "build_provenance.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"run root: {RUN_ROOT}")
    print(f"derived metadata: {derived_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
