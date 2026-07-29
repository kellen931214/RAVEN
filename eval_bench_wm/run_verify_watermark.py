"""Standalone GaussMarker / RingID verification, calibration and paper evaluation.

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

RingID (``--wm_type RID``) adds the official ECCV 2024 workflows on top of the
same structure; see ``rid_main`` below and ``utils/wm/ringid_provider.py``
(official reference https://github.com/showlab/RingID at commit
``45631a59aecd7d63ccdb640aaaf3e616fdb89fb9``).

Examples
--------
    python eval_bench_wm/run_verify_watermark.py \\
        --wm_type GM --gm_bundle_dir out/gm_bundle \\
        --suspect_path images/or_directory --out_dir out/gm_verification

    python eval_bench_wm/run_verify_watermark.py \\
        --mode paper_eval_verification --wm_type RID --rid_profile official_sd21 \\
        --rid_bundle_dir out/rid_bundle --clean_path data/rid/clean \\
        --watermarked_path data/rid/watermarked --target_fpr 0.01 \\
        --out_dir out/rid_verification

    python eval_bench_wm/run_verify_watermark.py \\
        --mode identify --wm_type RID --rid_bundle_dir out/rid_bundle \\
        --suspect_path images/or_directory --out_dir out/rid_identification
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
from utils.wm.ringid_provider import RID_SCORE_DEFINITION
from utils.wm.ringid_provider import apply_arg_defaults as rid_apply_profile
from utils.wm.ringid_provider import parser as rid_parser


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
    parser.add_argument("--pair_manifest", type=str, default=None,
                        help="JSON file declaring the positive/negative pairing explicitly: "
                             '{"protocol": ..., "pairs": [{"positive": ..., "negative": ...}, ...]}.')
    parser.add_argument("--allow_unmatched_cohorts", action="store_true", default=False,
                        help="paper_eval requires a verified one-to-one paired cohort unless this "
                             "is set; the run is then labelled legacy_or_ablation_mode and is NOT "
                             "an official paper evaluation.")

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

    # Equal cohort sizes are necessary but nowhere near sufficient: the official
    # paper protocol pairs each watermarked positive with the non-watermarked
    # negative produced from the same prompt/seed under the same distortion.
    pairing = gm_runtime.resolve_pairing(positives, negatives, args.pair_manifest)
    if args.mode == "paper_eval" and not pairing["paired"] and not args.allow_unmatched_cohorts:
        raise SystemExit(
            "error: the official paper protocol requires a verified one-to-one paired cohort "
            f"({pairing['reason']}). Supply --pair_manifest or per-sample metadata, or pass "
            "--allow_unmatched_cohorts to run a non-official ablation."
        )
    if not pairing["paired"]:
        print(f"[GM] cohorts are NOT verifiably paired: {pairing['reason']}", flush=True)
    return {"positives": positives, "negatives": negatives, "pairing": pairing}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    # The two methods have disjoint flag sets and different mode vocabularies, so
    # the method is resolved first and only its own parser is built.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--wm_type", type=str, default="GM")
    method = peek.parse_known_args(argv)[0].wm_type
    if method == "RID":
        return rid_main(argv)

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
    pairing = inputs["pairing"]

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
        report_label = gm_runtime.cohort_report_label(
            args.mode, provider.profile_is_official, pairing["paired"]
        )
        evaluation_mode = "official_paper_evaluation"
    else:
        report_label = gm_runtime.cohort_report_label(
            args.mode, provider.profile_is_official, pairing["paired"]
        )
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
            "paired_cohorts": pairing["paired"],
            "pairing_reason": pairing["reason"],
            "pairing_sha256": pairing["pairing_sha256"],
            "pairing_protocol": pairing["protocol"],
            "pair_manifest": pairing["pair_manifest"],
            "pair_count": len(pairing["pairs"]),
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



# ===========================================================================
# RingID (Issue #3)
# ===========================================================================

RID_MODES = (
    "verify",
    "calibrate",
    "paper_eval_verification",
    "identify",
    "paper_eval_identification",
)


def build_rid_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RingID standalone verification / calibration / identification",
        parents=[rid_parser],
    )
    parser.add_argument("--wm_type", type=str, default="RID", choices=["RID"])
    parser.add_argument("--mode", type=str, default="verify", choices=list(RID_MODES))
    parser.add_argument("--suspect_path", type=str, default=None,
                        help="One image file or a directory of images (verify / identify).")
    parser.add_argument("--watermarked_path", type=str, default=None,
                        help="Watermarked (positive) cohort from the same bundle.")
    parser.add_argument("--clean_path", type=str, default=None,
                        help="Non-watermarked (negative) cohort.")
    parser.add_argument("--out_dir", type=str, default="out/rid_verification")
    parser.add_argument("--target_fpr", type=float, default=None,
                        help="Overrides the profile's --rid_target_fpr for cohort evaluation.")
    parser.add_argument("--overwrite_threshold", action="store_true", default=False)
    parser.add_argument("--pair_manifest", type=str, default=None,
                        help="JSON file declaring the positive/negative pairing explicitly.")
    parser.add_argument("--allow_unmatched_cohorts", action="store_true", default=False,
                        help="paper_eval_verification requires a verified one-to-one paired "
                             "cohort unless this is set; the run is then labelled "
                             "legacy_or_ablation_mode and is NOT an official paper evaluation.")
    parser.add_argument("--rid_candidate_indices", type=str, default=None,
                        help="Comma-separated declared keybook (capacity study). Default: the "
                             "full official candidate keybook.")
    parser.add_argument("--true_key_index", type=int, default=None,
                        help="Known true key for every suspect image (evaluation only). Per-image "
                             "sample metadata takes precedence when present.")

    # model / pipeline (defaults come from the RingID profile, not generic RAVEN defaults)
    parser.add_argument("--modelid_target", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--scheduler_target", type=str, default="DPM",
                        choices=sorted(pipe_utils.SCHEDULER_CLASSES.keys()))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps_target", type=int, default=50)
    parser.add_argument("--guidance_scale_target", type=float, default=7.5)
    return parser


def _rid_write_csv(path: Path, rows: typing.Sequence[typing.Mapping[str, typing.Any]],
                   columns: typing.Sequence[str]) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in columns:
                value = row.get(key)
                flat[key] = _json.dumps(value) if isinstance(value, (list, dict)) else value
            writer.writerow(flat)


def _rid_candidate_indices(args) -> typing.Optional[typing.List[int]]:
    if not args.rid_candidate_indices:
        return None
    indices = [int(token) for token in str(args.rid_candidate_indices).split(",") if token.strip()]
    if not indices:
        raise SystemExit("error: --rid_candidate_indices is empty")
    return indices


def _rid_true_key(args, image_path: Path) -> typing.Optional[int]:
    """True key for one image: per-image metadata first, then the CLI default."""
    meta = gm_runtime.load_pair_metadata(image_path) or {}
    if meta.get("selected_key_index") is not None:
        return int(meta["selected_key_index"])
    return args.true_key_index


def _rid_score_cohort(args, provider, pipe_provider, image_paths, threshold_info, role: str,
                      identify: bool, candidate_indices, start_index: int = 0):
    from utils.wm import rid_runtime

    rows = []
    for offset, image_path in enumerate(image_paths):
        row = rid_runtime.score_image(
            provider,
            pipe_provider,
            image_path,
            image_index=start_index + offset,
            threshold_info=threshold_info,
            identify=identify,
            candidate_indices=candidate_indices,
            true_key_index=_rid_true_key(args, Path(image_path)) if identify else None,
        )
        row["cohort_role"] = role
        rows.append(row)
    return rows


def _rid_resolve_inputs(args) -> typing.Dict[str, typing.Any]:
    """Enumerate and validate the inputs before loading any model."""
    from utils.wm import rid_runtime

    if args.mode in ("verify", "identify", "paper_eval_identification"):
        images = rid_runtime.enumerate_images(
            _require(args.suspect_path, f"--suspect_path is required in {args.mode} mode")
        )
        if not images:
            raise SystemExit(f"error: no supported images found under {args.suspect_path}")
        return {"suspects": images}

    positives = rid_runtime.enumerate_images(
        _require(args.watermarked_path, f"--watermarked_path is required in {args.mode} mode")
    )
    negatives = rid_runtime.enumerate_images(
        _require(args.clean_path, f"--clean_path is required in {args.mode} mode")
    )
    if not positives or not negatives:
        raise SystemExit("error: both cohorts must contain at least one image")

    # Equal cohort sizes are necessary but nowhere near sufficient: the official
    # protocol pairs each watermarked positive with the non-watermarked negative
    # produced from the same prompt/seed under the same distortion.
    pairing = rid_runtime.resolve_pairing(positives, negatives, args.pair_manifest)
    if (args.mode == "paper_eval_verification" and not pairing["paired"]
            and not args.allow_unmatched_cohorts):
        raise SystemExit(
            "error: the official RingID verification protocol requires a verified one-to-one "
            f"paired cohort ({pairing['reason']}). Supply --pair_manifest or per-sample metadata, "
            "or pass --allow_unmatched_cohorts to run a non-official ablation."
        )
    if not pairing["paired"]:
        print(f"[RID] cohorts are NOT verifiably paired: {pairing['reason']}", flush=True)
    return {"positives": positives, "negatives": negatives, "pairing": pairing}


def rid_main(argv) -> int:
    from utils.wm import rid_bundle, rid_runtime
    from utils.wm.rid_bundle import RidBundle, RidBundleError

    args = build_rid_parser().parse_args(argv)
    profile_info = rid_apply_profile(args, argv)
    print(f"[RID] profile: {profile_info}", flush=True)

    if not args.rid_bundle_dir:
        raise SystemExit(
            "error: --rid_bundle_dir is required; verification and identification load the key "
            "artifact from a bundle and never reconstruct undocumented RNG state"
        )

    bundle_dir = Path(args.rid_bundle_dir)
    bundle_before = RidBundle.load(bundle_dir).artifact_mtimes()

    inputs = _rid_resolve_inputs(args)
    candidate_indices = _rid_candidate_indices(args)
    target_fpr = args.rid_target_fpr if args.target_fpr is None else float(args.target_fpr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_info = rid_runtime.gpu_preflight(DEVICE)
    print(f"[RID] device preflight: {gpu_info}", flush=True)

    pipe_provider = rid_runtime.build_pipe_provider(args, DEVICE)
    # create_bundle=False: verification/identification must never create or
    # modify watermark state.
    provider = rid_runtime.build_provider(
        args,
        latent_shape=pipe_provider.get_latent_shape(),
        device=DEVICE,
        create_bundle=False,
    )
    if provider.state_source != "bundle":
        raise RidBundleError(
            f"verification must load the key from an existing bundle, got {provider.state_source!r}"
        )

    provenance = rid_runtime.run_provenance(args, provider, pipe_provider)
    provenance["entrypoint"] = "eval_bench_wm/run_verify_watermark.py:rid_main"
    provenance["mode"] = args.mode

    if args.mode in ("verify", "identify", "paper_eval_identification"):
        summary = _rid_run_suspects(args, provider, pipe_provider, out_dir, provenance,
                                    inputs, candidate_indices)
    else:
        summary = _rid_run_cohort(args, provider, pipe_provider, out_dir, provenance,
                                  inputs, target_fpr)

    allowed = (rid_bundle.THRESHOLD_FILENAME,) if args.mode == "calibrate" else ()
    rid_runtime.assert_bundle_untouched(bundle_dir, bundle_before, allowed=allowed)

    print(rid_bundle.canonical_json(summary), flush=True)
    return 0


def _rid_run_suspects(args, provider, pipe_provider, out_dir: Path, provenance, inputs,
                      candidate_indices) -> dict:
    """Deployment verification and/or multi-key identification of suspect images."""
    from utils.wm import rid_bundle, rid_runtime

    images = inputs["suspects"]
    identify = args.mode in ("identify", "paper_eval_identification")

    threshold_info = provider.resolve_threshold()
    if threshold_info["threshold_available"]:
        report_label = threshold_info["report_label"]
    else:
        report_label = ("official_profile_raw_scores" if provider.profile_is_official
                        else "legacy_or_ablation_mode")
        if not identify:
            print(
                "[RID] no compatible threshold artifact and no --rid_threshold: emitting raw "
                "detector scores only; no binary decision is produced.",
                flush=True,
            )
    if identify:
        report_label = ("official_paper_identification" if provider.profile_is_official
                        else "legacy_or_ablation_mode")

    rows = _rid_score_cohort(args, provider, pipe_provider, images, threshold_info,
                             role="suspect", identify=identify,
                             candidate_indices=candidate_indices)
    evaluation_mode = ("official_identification" if identify
                       else "deployment_verification_extension")
    for row in rows:
        row["report_label"] = report_label
        row["evaluation_mode"] = evaluation_mode
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})

    rid_bundle.write_jsonl(out_dir / "results.jsonl", rows)
    _rid_write_csv(out_dir / "results.csv", rows, rid_runtime.RESULT_FIELDS)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    decided = [row for row in ok_rows if row["detection_success"] is not None]
    scored = [row for row in ok_rows if row["identification_correct"] is not None]

    summary = dict(provenance)
    summary.update({
        "evaluation_mode": evaluation_mode,
        "report_label": report_label,
        "image_count": len(rows),
        "ok_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "score_definition": rid_runtime.RID_SCORE_DEFINITION,
        "score_direction": "higher_is_watermarked",
        "comparison_operator": ">=",
        "threshold": threshold_info["threshold"],
        "threshold_source": threshold_info["threshold_source"],
        "threshold_target_fpr": threshold_info.get("threshold_target_fpr"),
        "threshold_empirical_fpr": threshold_info.get("threshold_empirical_fpr"),
        "detected_count": sum(1 for row in decided if row["detection_success"]),
        "decided_count": len(decided),
        "keybook_sha256": ok_rows[0]["keybook_sha256"] if (identify and ok_rows) else None,
        "candidate_count": ok_rows[0]["candidate_count"] if (identify and ok_rows) else None,
        "declared_candidate_indices": candidate_indices,
        "identification_evaluated_count": len(scored),
        "identification_accuracy": (
            None if not scored
            else float(sum(1 for row in scored if row["identification_correct"]) / len(scored))
        ),
        "note": (
            "Official RingID argmin identification over the declared candidate keybook."
            if identify else
            "Deployment verification extension: a fixed-threshold single-image decision. "
            "This is NOT the official paper cohort-evaluation protocol."
        ),
    })
    (out_dir / "summary.json").write_text(
        rid_bundle.canonical_json(summary) + "\n", encoding="utf-8"
    )
    return summary


def _rid_run_cohort(args, provider, pipe_provider, out_dir: Path, provenance, inputs,
                    target_fpr: float) -> dict:
    """Cohort-derived ROC verification (paper protocol) and threshold calibration."""
    from utils.wm import rid_bundle, rid_runtime
    from utils.wm.rid_bundle import RidBundleError

    positives, negatives = inputs["positives"], inputs["negatives"]
    pairing = inputs["pairing"]

    raw_threshold_info = {
        "threshold": None,
        "threshold_source": "cohort_pending",
        "comparison_operator": ">=",
    }
    rows = _rid_score_cohort(args, provider, pipe_provider, positives, raw_threshold_info,
                             role="positive", identify=False, candidate_indices=None)
    rows += _rid_score_cohort(args, provider, pipe_provider, negatives, raw_threshold_info,
                              role="negative", identify=False, candidate_indices=None,
                              start_index=len(positives))

    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        rid_bundle.write_jsonl(out_dir / "results.jsonl", rows)
        raise RidBundleError(
            f"{len(failed)} cohort image(s) failed to score; cohort ROC evaluation fails closed. "
            f"First error: {failed[0]['error']}"
        )

    positive_scores = [row["rid_score"] for row in rows if row["cohort_role"] == "positive"]
    negative_scores = [row["rid_score"] for row in rows if row["cohort_role"] == "negative"]
    roc = rid_runtime.official_roc(
        positive_scores, negative_scores, target_fpr,
        score_definition=rid_runtime.RID_SCORE_DEFINITION,
    )

    if args.mode == "paper_eval_verification":
        evaluation_mode = "official_paper_verification"
        report_label = ("official_paper_verification"
                        if (provider.profile_is_official and pairing["paired"])
                        else "legacy_or_ablation_mode")
    else:
        evaluation_mode = "threshold_calibration"
        report_label = "calibrated_deployment_verification"

    for row in rows:
        row["threshold"] = roc["threshold"]
        row["threshold_source"] = "cohort_roc"
        row["report_label"] = report_label
        row["evaluation_mode"] = evaluation_mode
        row["detection_success"] = provider.decide(row["rid_score"], roc["threshold"])
        row.update({k: v for k, v in provenance.items() if k != "created_utc"})

    rid_bundle.write_jsonl(out_dir / "results.jsonl", rows)
    _rid_write_csv(out_dir / "results.csv", rows, rid_runtime.RESULT_FIELDS)

    summary = dict(provenance)
    summary.update(roc)
    summary.update({
        "evaluation_mode": evaluation_mode,
        "report_label": report_label,
        "paired_cohorts": pairing["paired"],
        "pairing_reason": pairing["reason"],
        "pairing_sha256": pairing["pairing_sha256"],
        "pairing_protocol": pairing["protocol"],
        "pair_manifest": pairing["pair_manifest"],
        "pair_count": len(pairing["pairs"]),
        "positive_cohort_sha256": rid_bundle.cohort_sha256(positives),
        "negative_cohort_sha256": rid_bundle.cohort_sha256(negatives),
        "note": (
            "The threshold is derived from THIS positive/negative cohort with "
            "sklearn.metrics.roc_curve, exactly as official verify.py does. It is an "
            "experiment-specific operating point, not a universal detector threshold. "
            "The paper reports 1000 watermarked + 1000 unwatermarked images; a smaller "
            "cohort is a smoke/debug run."
        ),
    })

    if args.mode == "calibrate":
        artifact = rid_bundle.build_threshold_artifact(
            threshold=roc["threshold"],
            binding=provider.binding_config(),
            score_definition=rid_runtime.RID_SCORE_DEFINITION,
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
        print(f"[RID] wrote calibrated threshold to {path}", flush=True)

    (out_dir / "summary.json").write_text(
        rid_bundle.canonical_json(summary) + "\n", encoding="utf-8"
    )
    return summary

if __name__ == "__main__":
    raise SystemExit(main())
