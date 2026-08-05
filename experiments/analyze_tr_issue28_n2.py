#!/usr/bin/env python3
"""Issue #28: analyse a completed 2-sample TR detector evaluation.

Reads ``detector_records.jsonl`` and ``result.json`` from a run root,
validates the complex-L1 score contract, and reports every scored row
with its detection decision using the canonical calibration helper.

The threshold is calibrated from the **local** original-clean canonical
scores via ``raven.metrics.calibrate_threshold(clean, target_fpr=0.01)``
at analysis time — never hardcoded.  With small sample counts this is a
**local validation threshold** only; it must not be interpreted as a
production or full-cohort threshold.  The analysis threshold is diagnostic
and is **not** written into ``result.json``, ``detector_records.jsonl``,
or the production ``detection_summary``.

Detection decision: ``canonical_score >= threshold``, which is equivalent
to ``raw_score <= -threshold`` (both inclusive on the boundary).

Determinism check
-----------------
When a second run root (``--det2``) is provided, the script compares rows
by ``(run_id, evaluation_cohort)`` key:

* Duplicate keys are rejected — two rows with the same key in one run is a
  data error and fails immediately.
* Scored rows are compared by canonical score on the same key.
* Failed rows are compared after normalising ``image_path`` (run-root prefix
  → ``<ROOT>``) and the ``error`` message (run-root prefix → ``<ROOT>``,
  so the exact output-directory name does not affect the comparison).
* Scored-row normalisation additionally strips ``evaluated_utc``.

The output explicitly distinguishes:

* ``scored-row snapshot`` — scored rows only, after normalisation.
* ``full-record snapshot`` — all rows (scored + failed), after the
  same normalisation applied per row type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.metrics import calibrate_threshold, detection_rate  # noqa: E402


def _sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Keyed collection — duplicates fail immediately, never silently overwrite
# ---------------------------------------------------------------------------
def _scored_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """{(run_id, evaluation_cohort): canonical_score} for scored rows.

    Raises ``ValueError`` on duplicate key — the same (run_id, cohort)
    appearing twice in scored rows is a data integrity error.
    """
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        if r.get("status") != "scored":
            continue
        key = (str(r["run_id"]), str(r["evaluation_cohort"]))
        if key in out:
            raise ValueError(
                f"duplicate scored key (run_id={key[0]!r}, "
                f"cohort={key[1]!r}) — data integrity violation"
            )
        out[key] = float(r["canonical_score"])
    return out


def _all_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """{(run_id, evaluation_cohort): full_row} for every row.

    Duplicate keys fail immediately.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (str(r["run_id"]), str(r["evaluation_cohort"]))
        if key in out:
            raise ValueError(
                f"duplicate key (run_id={key[0]!r}, cohort={key[1]!r}) "
                f"— data integrity violation"
            )
        out[key] = r
    return out


def _scored_cohorts(
    rows: list[dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    """Return {cohort: [(run_id, canonical_score), ...]} for scored rows."""
    out: dict[str, list[tuple[str, float]]] = {}
    scored = _scored_by_key(rows)  # validates uniqueness
    for (run_id, cohort), score in scored.items():
        out.setdefault(cohort, []).append((run_id, score))
    return out


# ---------------------------------------------------------------------------
# Provenance from the first scored row
# ---------------------------------------------------------------------------
def _prov_from_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    for r in rows:
        if r.get("status") == "scored":
            return {
                "target_sha": r.get("tr_detector_watermark_target_sha256", ""),
                "mask_sha": r.get("tr_detector_watermark_mask_sha256", ""),
                "provider_config_hash": r.get("tr_provider_config_hash", ""),
                "model_id": r.get("tr_model_id", ""),
                "model_revision": r.get("tr_model_revision", ""),
                "scheduler": r.get("tr_scheduler", ""),
                "inverse_scheduler": r.get("tr_inverse_scheduler", ""),
                "steps": r.get("tr_steps", ""),
                "resolution": r.get("tr_resolution", ""),
                "detector_dtype": r.get("tr_detector_dtype", ""),
                "vae_id": r.get("tr_vae_id", ""),
                "vae_scaling_factor": r.get("tr_vae_scaling_factor", ""),
                "w_pattern_const": r.get("tr_w_pattern_const", ""),
                "target_verified": r.get("tr_target_verified"),
                "mask_verified": r.get("tr_mask_verified"),
                "score_protocol": r.get("tr_score_protocol"),
                "score_definition": r.get("tr_score_definition"),
                "raw_score_direction": r.get("tr_raw_score_direction"),
                "canonical_score_direction": r.get("tr_canonical_score_direction"),
                "comparison_operator": r.get("tr_comparison_operator"),
            }
    return {}


# ---------------------------------------------------------------------------
# Normalisation helpers — strip timestamps and run-root absolute paths
# ---------------------------------------------------------------------------
def _normalise_path_prefix(value: str, run_root: str) -> str:
    """Replace every occurrence of *run_root* in *value* with ``<ROOT>``."""
    return value.replace(run_root, "<ROOT>")


def _normalise_row_for_snapshot(
    row: dict[str, Any],
    run_root: str,
) -> dict[str, Any]:
    """Return a copy of *row* with timestamp dropped and run-root paths
    replaced by ``<ROOT>`` in every string field.

    For ``failed_missing_image`` rows the ``image_path`` and ``error``
    fields are normalised; for scored rows the ``image_path`` and any
    other string field embedding the run root are normalised.
    """
    rn = dict(row)
    rn.pop("evaluated_utc", None)
    for key in list(rn):
        val = rn[key]
        if isinstance(val, str) and run_root in val:
            rn[key] = _normalise_path_prefix(val, run_root)
    return rn


def _scored_snapshot(rows: list[dict[str, Any]], run_root: str) -> str:
    """Deterministic canonical JSON of scored rows, normalised."""
    items = []
    for r in rows:
        if r.get("status") != "scored":
            continue
        items.append(_normalise_row_for_snapshot(r, run_root))
    items.sort(key=lambda r: (str(r.get("run_id", "")),
                               str(r.get("evaluation_cohort", ""))))
    return json.dumps(items, sort_keys=True, ensure_ascii=False)


def _full_snapshot(rows: list[dict[str, Any]], run_root: str) -> str:
    """Deterministic canonical JSON of all rows (scored + failed), normalised."""
    items = [_normalise_row_for_snapshot(r, run_root) for r in rows]
    items.sort(key=lambda r: (str(r.get("run_id", "")),
                               str(r.get("evaluation_cohort", ""))))
    return json.dumps(items, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyse(root: Path, label: str, det2_root: Path | None = None) -> int:
    records_path = root / "evaluation" / "detector_records.jsonl"
    result_path = root / "result.json"
    if not records_path.is_file():
        print(f"error: {records_path} not found", file=sys.stderr)
        return 1

    rows = _load_records(records_path)
    scored = [r for r in rows if r.get("status") == "scored"]
    failed = [r for r in rows if r.get("status") != "scored"]

    # ---- uniqueness check for all rows ----
    try:
        _all_by_key(rows)
    except ValueError as exc:
        print(f"error: duplicate (run_id, cohort) key: {exc}", file=sys.stderr)
        return 1

    print(f"=== {label} ({root}) ===")
    print(f"detector_records.jsonl sha256: {_sha256_hex(records_path.read_bytes())}")

    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        st = result.get("stages", {}).get("detector", {})
        print(f"result.json sha256: {_sha256_hex(result_path.read_bytes())}")
        print(f"overall_status: {result.get('overall_status')}  "
              f"exit_code_policy: {result.get('allowed_by_policy')}")
        print(f"stage_status: {st.get('status')}  "
              f"dominant_cause: {st.get('dominant_failure_cause')}")
        print(f"requested: {st.get('requested_count')}  "
              f"  (= {st.get('requested_count', 0) // 4} run_ids × "
              f"2 roles × 2 variants (original + attacked))")
        print(f"scored: {st.get('scored_count')}  "
              f"failed: {st.get('failed_count')}  "
              f"unscored_setup: {st.get('unscored_due_to_setup_count')}  "
              f"count_invariant: {st.get('count_invariant_satisfied')}")
        print(f"cohort_counts: {st.get('cohort_counts')}")

        # ---- metric availability (primary vs recalibrated) ----
        ma = st.get("metric_availability", {})

        ds = st.get("detection_summary")
        if ds is None:
            primary_missing = ma.get("threshold_missing") or []
            print(f"\nprimary detection_summary: ABSENT")
            if primary_missing:
                print(f"primary required cohorts missing: {primary_missing}")
            else:
                print(f"primary required cohorts missing: "
                      f"{ma.get('threshold_required_cohorts')} (all absent)")
        else:
            print(f"\nprimary detection_summary: PRESENT "
                  f"(threshold={ds.get('original_clean_threshold')!r})")

        # recalibrated — independent block, separate from primary
        recal = st.get("tr_recalibrated", {})
        recal_avail = ma.get("recalibrated_cohorts_available", False)
        if recal_avail and recal.get("recalibrated_metrics_available"):
            print(f"recalibrated report: PRESENT")
        else:
            print(f"recalibrated report: ABSENT")
            if "recalibrated_required_cohorts" in ma:
                missing_recal = sorted(
                    set(ma["recalibrated_required_cohorts"])
                    - set(ma.get("scored_cohorts", []))
                )
                print(f"recalibrated required cohorts missing: "
                      f"{missing_recal or ma['recalibrated_required_cohorts']}")
    else:
        print("result.json: not found")

    # ---- provenance ----
    prov = _prov_from_rows(rows)
    if prov:
        print(f"\nprovenance (from scored rows):")
        for k in ("target_sha", "mask_sha", "provider_config_hash",
                   "target_verified", "mask_verified",
                   "model_id", "model_revision", "scheduler",
                   "inverse_scheduler", "steps", "resolution",
                   "detector_dtype", "vae_id", "vae_scaling_factor",
                   "w_pattern_const"):
            print(f"  {k}: {prov.get(k)}")
        print(f"  score_protocol: {prov.get('score_protocol')}  "
              f"definition: {prov.get('score_definition')}")
        print(f"  raw_direction: {prov.get('raw_score_direction')}  "
              f"canonical_direction: {prov.get('canonical_score_direction')}  "
              f"operator: {prov.get('comparison_operator')}")

    # ---- per-cohort scores by (run_id, cohort) ----
    by_cohort = _scored_cohorts(rows)
    print(f"\nscored rows (by cohort):")
    for cohort in sorted(by_cohort):
        items = by_cohort[cohort]
        print(f"  {cohort}: {len(items)} row(s)")
        for run_id, score in items:
            print(f"    run_id={run_id}  canonical={score:.12f}  "
                  f"raw={-score:.12f}")

    # ---- diagnostic threshold (local, NOT production) ----
    clean = [score for run_id, score in by_cohort.get("original_clean", [])]
    wm = [score for run_id, score in by_cohort.get("original_watermarked", [])]
    if clean and wm:
        cal = calibrate_threshold(clean, target_fpr=0.01)
        print(f"\ndiagnostic threshold (LOCAL validation — {len(clean)} "
              f"clean samples, NOT a production/full-cohort threshold; NOT "
              f"written into result.json)")
        print(f"  calibrator: raven.metrics.calibrate_threshold(clean, "
              f"target_fpr=0.01)")
        print(f"  max_fp_budget: {cal.max_false_positives}")
        print(f"  clean N: {cal.num_clean}")
        print(f"  threshold (canonical, operator >=): {cal.threshold!r}")
        print(f"  threshold (raw L1,     operator <=): {-cal.threshold!r}")
        print(f"  clean FP count: {cal.false_positives}  "
              f"actual clean FPR: {cal.actual_fpr:.6f}")
        tpr = detection_rate(wm, cal.threshold)
        detected = sum(1 for s in wm if s >= cal.threshold)
        print(f"  original_watermarked detected: {detected}/{len(wm)}  "
              f"TPR: {tpr:.6f}")
        print(f"  clean detected (expected 0): "
              f"{sum(1 for s in clean if s >= cal.threshold)}")

        # ---- decision per (run_id, cohort) ----
        print(f"\ndetection decisions (canonical >= threshold, "
              f"raw <= -threshold):")
        for r in scored:
            dec = r["canonical_score"] >= cal.threshold
            print(f"  ({r['run_id']}, {r['evaluation_cohort']})  "
                  f"canonical={r['canonical_score']:.12f}  "
                  f"{'>= threshold' if dec else '<  threshold'}  "
                  f"→ detected={dec}")
    else:
        print("\ninsufficient scored cohorts for diagnostic threshold "
              "calibration")

    # ---- determinism check against second run ----
    if det2_root is not None:
        print(f"\n=== determinism vs {det2_root} ===")
        det2_path = det2_root / "evaluation" / "detector_records.jsonl"
        if not det2_path.is_file():
            print(f"  error: {det2_path} not found")
            return 0

        rows2 = _load_records(det2_path)

        # uniqueness check for second run
        try:
            _all_by_key(rows2)
        except ValueError as exc:
            print(f"error: duplicate key in run2: {exc}", file=sys.stderr)
            return 1

        keyed1 = _scored_by_key(rows)
        keyed2 = _scored_by_key(rows2)

        print(f"  run1 scored rows: {len(keyed1)}  "
              f"run2 scored rows: {len(keyed2)}")

        # ---- compare by (run_id, cohort) key ----
        mismatched = []
        for key in sorted(keyed1):
            if key not in keyed2:
                mismatched.append((key, "missing in run2"))
                continue
            if keyed1[key] != keyed2[key]:
                mismatched.append(
                    (key, f"score diff: run1={keyed1[key]:.12f} "
                          f"run2={keyed2[key]:.12f}")
                )
        for key in sorted(set(keyed2) - set(keyed1)):
            mismatched.append((key, "missing in run1"))

        if mismatched:
            print(f"  MISMATCH: {len(mismatched)} scored key(s) differ:")
            for key, reason in mismatched:
                print(f"    ({key[0]}, {key[1]}): {reason}")
        else:
            print(f"  scored rows: {len(keyed1)} (run_id, cohort) pairs "
                  f"consistent across both runs")

        # ---- normalised scored-row snapshot ----
        root_str = str(root)
        root2_str = str(det2_root)
        snap1 = _scored_snapshot(rows, root_str)
        snap2 = _scored_snapshot(rows2, root2_str)
        same_scored = snap1 == snap2
        print(f"  normalized scored-row snapshots identical: {same_scored}")

        # ---- normalised full-record snapshot (scored + failed) ----
        full1 = _full_snapshot(rows, root_str)
        full2 = _full_snapshot(rows2, root2_str)
        same_full = full1 == full2
        print(f"  normalized full-record snapshots identical: {same_full}")
        if not same_full:
            lines1 = full1.splitlines(keepends=True)
            lines2 = full2.splitlines(keepends=True)
            for i, (a, b) in enumerate(zip(lines1, lines2)):
                if a != b:
                    print(f"  first full-record diff at line {i}:")
                    print(f"    run1: {a.rstrip()}")
                    print(f"    run2: {b.rstrip()}")
                    break

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="Run root with detector_records.jsonl")
    p.add_argument("--label", default="2-sample TR")
    p.add_argument("--det2", type=Path, default=None,
                   help="Second run root for determinism check")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(analyse(args.root, args.label, args.det2))
