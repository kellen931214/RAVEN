"""Standalone GaussMarker verification, threshold calibration and paper evaluation.

This runner does CLI parsing, image enumeration, provider invocation and
serialization only. Every GaussMarker algorithm lives in
``utils/wm/gm_provider.py``; shared IO/ROC bookkeeping lives in
``utils/wm/gm_runtime.py``.

Official reference: https://github.com/SunnierLee/GaussMarker at commit
``4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155``.

Modes
-----
``verify`` (default)
    RAVEN *deployment extension*: score one or more suspect images against a
    pre-calibrated compatible ``threshold.json`` (or an explicit user-supplied
    threshold). Needs neither the original prompt nor the original image, never
    modifies the bundle and never creates watermark state. Raw detector scores
    are always emitted, even when no binary threshold is available.

``calibrate``
    Derive a threshold from a clean (negative) and a same-bundle watermarked
    (positive) cohort with the same detector path, and store it beside the
    bundle as ``threshold.json``.

``paper_eval``
    Reproduce the official paper protocol: paired positive/negative cohorts →
    same inversion/detector path → ensemble probability → ROC → TPR at the
    requested FPR and the experiment-specific threshold. The resulting threshold
    is *not* a universal detector threshold and is labelled accordingly.

Examples
--------
    python eval_bench_wm/run_verify_watermark.py \\
        --wm_type GM --gm_bundle_dir out/gm_bundle \\
        --suspect_path images/or_directory --out_dir out/gm_verification
"""

from __future__ import annotations

import argparse
import csv
import sys
import typing
from pathlib import Path

import torch

from utils.pipe import pipe_utils
from utils.wm import gm_bundle, gm_runtime
from utils.wm.gm_bundle import GmBundle, GmBundleError
from utils.wm.gm_provider import GM_SCORE_DEFINITION
from utils.wm.gm_provider import apply_arg_defaults as gm_apply_profile
from utils.wm.gm_provider import parser as gm_parser


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_COLUMNS = [
    "image_path",
    "image_sha256",
    "status",
    "error",
    "report_label",
    "raw_bit_accuracy",
    "restored_bit_accuracy",
    "raw_ring_l1",
    "ring_classifier_feature",
    "classifier_probability",
    "score",
    "score_definition",
    "score_direction",
    "threshold",
    "threshold_source",
    "comparison_operator",
    "detection_success",
    "inversion_seed",
    "inversion_steps",
    "recovered_latent_sha256",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GaussMarker standalone verification / calibration / paper evaluation",
        parents=[gm_parser],
    )
    parser.add_argument("--wm_type", type=str, default="GM", choices=["GM"])
    parser.add_argument("--mode", type=str, default="verify",
                        choices=["verify", "calibrate", "paper_eval"])
    parser.add_argument("--suspect_path", type=str, default=None,
                        help="One image file or a directory of images (verify mode).")
    parser.add_argument("--positive_path", type=str, default=None,
                        help="Watermarked cohort from the same bundle (calibrate / paper_eval).")
    parser.add_argument("--negative_path", type=str, default=None,
                        help="Non-watermarked cohort (calibrate / paper_eval).")
    parser.add_argument("--out_dir", type=str, default="out/gm_verification")
    parser.add_argument("--overwrite_threshold", action="store_true", default=False)
    parser.add_argument("--allow_unmatched_cohorts", action="store_true", default=False,
                        help="paper_eval requires matched cohort sizes unless this is set "
                             "(the run is then no longer official_paper_evaluation).")

    # model / pipeline (defaults come from the GM profile, not from generic RAVEN defaults)
    parser.add_argument("--modelid_target", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--scheduler_target", type=str, default="DPM",
                        choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)

    # Tree-Ring style ring parameters. They live on this parser (not on the shared
    # gm_parser) because run_watermark.py already inherits them from tr_parser.
    parser.add_argument("--w_seed", type=int, default=999999)
    parser.add_argument("--w_channel", type=int, default=3)
    parser.add_argument("--w_pattern", type=str, default="ring")
    parser.add_argument("--w_mask_shape", type=str, default="circle")
    parser.add_argument("--w_radius", type=int, default=4)
    parser.add_argument("--w_measurement", type=str, default="l1_complex")
    parser.add_argument("--w_injection", type=str, default="complex")
    parser.add_argument("--w_pattern_const", type=float, default=0.0)
    return parser


def _write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, typing.Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _score_cohort(provider, pipe_provider, image_paths, threshold_info, role: str):
    rows = []
    for image_path in image_paths:
        row = gm_runtime.score_image(provider, pipe_provider, image_path, threshold_info)
        row["cohort_role"] = role
        rows.append(row)
    return rows


def _require(value, message: str):
    if value is None:
        raise SystemExit(f"error: {message}")
    return value


def _resolve_inputs(args) -> typing.Dict[str, typing.Any]:
    """Enumerate and validate the input cohorts before loading any model."""
    if args.mode == "verify":
        images = gm_runtime.enumerate_images(
            _require(args.suspect_path, "--suspect_path is required in verify mode")
        )
        if not images:
            raise SystemExit(f"error: no supported images found under {args.suspect_path}")
        return {"suspects": images}

    positives = gm_runtime.enumerate_images(
        _require(args.positive_path, f"--positive_path is required in {args.mode} mode")
    )
    negatives = gm_runtime.enumerate_images(
        _require(args.negative_path, f"--negative_path is required in {args.mode} mode")
    )
    if not positives or not negatives:
        raise SystemExit("error: both cohorts must contain at least one image")

    matched = len(positives) == len(negatives)
    if args.mode == "paper_eval" and not matched and not args.allow_unmatched_cohorts:
        raise SystemExit(
            "error: the official paper protocol uses matched positive/negative cohort sizes "
            f"({len(positives)} vs {len(negatives)}); pass --allow_unmatched_cohorts to run an ablation"
        )
    return {"positives": positives, "negatives": negatives, "matched": matched}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)

    profile_info = gm_apply_profile(args, argv)
    print(f"[GM] profile: {profile_info}", flush=True)

    if not args.gm_bundle_dir:
        raise SystemExit("error: --gm_bundle_dir is required; verification loads state from a bundle")

    bundle_dir = Path(args.gm_bundle_dir)
    bundle_before = GmBundle.load(bundle_dir).artifact_mtimes()

    # Resolve and validate the cohorts/suspects *before* any model is loaded so a
    # CLI mistake fails immediately instead of after a multi-GB download.
    inputs = _resolve_inputs(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = gm_runtime.gpu_preflight(DEVICE)
    print(f"[GM] device preflight: {gpu_info}", flush=True)

    pipe_provider = gm_runtime.build_pipe_provider(args, DEVICE)
    # create_bundle=False: verification must never create watermark state and must
    # never sample a new latent.
    provider = gm_runtime.build_provider(
        args,
        latent_shape=pipe_provider.get_latent_shape(),
        device=DEVICE,
        create_bundle=False,
    )
    if provider.state_source != "bundle":
        raise GmBundleError(
            f"verification must load state from an existing bundle, got {provider.state_source!r}"
        )

    provenance = gm_runtime.run_provenance(args, provider, pipe_provider)
    provenance["entrypoint"] = "eval_bench_wm/run_verify_watermark.py"
    provenance["mode"] = args.mode

    if args.mode == "verify":
        summary = _run_verify(args, provider, pipe_provider, out_dir, provenance, inputs)
    else:
        summary = _run_cohort(args, provider, pipe_provider, out_dir, provenance, inputs)

    bundle_after = GmBundle.load(bundle_dir).artifact_mtimes()
    if args.mode == "calibrate":
        # Only threshold.json may appear/change during calibration.
        changed = {k for k in set(bundle_before) | set(bundle_after)
                   if bundle_before.get(k) != bundle_after.get(k)}
        unexpected = changed - {gm_bundle.THRESHOLD_FILENAME}
        if unexpected:
            raise GmBundleError(f"calibration modified bundle artifacts it must not touch: {sorted(unexpected)}")
    elif bundle_before != bundle_after:
        raise GmBundleError("the bundle was modified during verification; this must never happen")

    print(gm_bundle.canonical_json(summary), flush=True)
    return 0


def _run_verify(args, provider, pipe_provider, out_dir: Path, provenance, inputs) -> dict:
    images = inputs["suspects"]

    threshold_info = provider.resolve_threshold()
    if threshold_info["threshold_available"]:
        report_label = threshold_info["report_label"]
    else:
        report_label = "official_profile_raw_scores" if provider.profile_is_official else "legacy_or_ablation_mode"
        print(
            "[GM] no compatible threshold artifact and no --gm_threshold: emitting raw detector "
            "scores only; no binary decision is produced.",
            flush=True,
        )

    rows = _score_cohort(provider, pipe_provider, images, threshold_info, role="suspect")
    for row in rows:
        row["report_label"] = report_label
        row["evaluation_mode"] = "deployment_verification_extension"
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})

    gm_bundle.write_jsonl(out_dir / "results.jsonl", rows)
    _write_csv(out_dir / "results.csv", rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    decided = [row for row in ok_rows if row["detection_success"] is not None]
    summary = dict(provenance)
    summary.update(
        {
            "evaluation_mode": "deployment_verification_extension",
            "report_label": report_label,
            "image_count": len(rows),
            "ok_count": len(ok_rows),
            "error_count": len(rows) - len(ok_rows),
            "score_definition": GM_SCORE_DEFINITION,
            "score_direction": threshold_info["score_direction"],
            "comparison_operator": threshold_info["comparison_operator"],
            "threshold": threshold_info["threshold"],
            "threshold_source": threshold_info["threshold_source"],
            "threshold_target_fpr": threshold_info.get("threshold_target_fpr"),
            "threshold_empirical_fpr": threshold_info.get("threshold_empirical_fpr"),
            "detected_count": sum(1 for row in decided if row["detection_success"]),
            "decided_count": len(decided),
            "note": (
                "Deployment verification extension: a fixed-threshold single-image decision. "
                "This is NOT the official paper cohort-evaluation protocol."
            ),
        }
    )
    (out_dir / "summary.json").write_text(gm_bundle.canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def _run_cohort(args, provider, pipe_provider, out_dir: Path, provenance, inputs) -> dict:
    positives = inputs["positives"]
    negatives = inputs["negatives"]
    matched = inputs["matched"]

    # Cohort scoring uses the same provider/detector path as single-image verification.
    raw_threshold_info = {
        "threshold": None,
        "threshold_source": "cohort_pending",
        "score_direction": "higher_is_watermarked",
        "comparison_operator": ">=",
    }
    rows = _score_cohort(provider, pipe_provider, positives, raw_threshold_info, role="positive")
    rows += _score_cohort(provider, pipe_provider, negatives, raw_threshold_info, role="negative")

    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        gm_bundle.write_jsonl(out_dir / "results.jsonl", rows)
        raise GmBundleError(
            f"{len(failed)} cohort image(s) failed to score; cohort ROC evaluation fails closed. "
            f"First error: {failed[0]['error']}"
        )
    if any(row["classifier_probability"] is None for row in rows):
        gm_bundle.write_jsonl(out_dir / "results.jsonl", rows)
        raise GmBundleError(
            "cohort evaluation needs the official ensemble probability, which requires both the "
            "GNR checkpoint (--gm_gnr_path) and the classifier (--gm_classifier_path)"
        )

    positive_scores = [row["score"] for row in rows if row["cohort_role"] == "positive"]
    negative_scores = [row["score"] for row in rows if row["cohort_role"] == "negative"]
    roc = gm_runtime.official_roc(positive_scores, negative_scores, args.gm_target_fpr)

    if args.mode == "paper_eval":
        report_label = "official_paper_evaluation" if (provider.profile_is_official and matched) else "legacy_or_ablation_mode"
        evaluation_mode = "official_paper_evaluation"
    else:
        report_label = "calibrated_deployment_verification"
        evaluation_mode = "threshold_calibration"

    for row in rows:
        row["threshold"] = roc["threshold"]
        row["threshold_source"] = "cohort_roc"
        row["report_label"] = report_label
        row["evaluation_mode"] = evaluation_mode
        row["detection_success"] = provider.decide(row["score"], roc["threshold"])
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})

    gm_bundle.write_jsonl(out_dir / "results.jsonl", rows)
    _write_csv(out_dir / "results.csv", rows)

    summary = dict(provenance)
    summary.update(roc)
    summary.update(
        {
            "evaluation_mode": evaluation_mode,
            "report_label": report_label,
            "matched_cohorts": matched,
            "positive_cohort_sha256": gm_bundle.cohort_sha256(positives),
            "negative_cohort_sha256": gm_bundle.cohort_sha256(negatives),
            "note": (
                "The threshold is derived from THIS positive/negative cohort with "
                "sklearn.metrics.roc_curve. It is an experiment-specific operating point, "
                "not a universal detector threshold."
            ),
        }
    )

    if args.mode == "calibrate":
        artifact = gm_bundle.build_threshold_artifact(
            threshold=roc["threshold"],
            binding=provider.binding_config(),
            score_definition=GM_SCORE_DEFINITION,
            threshold_source="cohort_calibration",
            report_label="calibrated_deployment_verification",
            target_fpr=roc["target_fpr"],
            empirical_fpr=roc["empirical_fpr"],
            tpr_at_target_fpr=roc["tpr_at_target_fpr"],
            roc_auc=roc["roc_auc"],
            positive_count=roc["positive_count"],
            negative_count=roc["negative_count"],
            positive_cohort_sha256=summary["positive_cohort_sha256"],
            negative_cohort_sha256=summary["negative_cohort_sha256"],
        )
        path = provider.bundle.save_threshold(artifact, overwrite=args.overwrite_threshold)
        summary["threshold_artifact"] = path.as_posix()
        print(f"[GM] wrote calibrated threshold to {path}", flush=True)

    (out_dir / "summary.json").write_text(gm_bundle.canonical_json(summary) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
