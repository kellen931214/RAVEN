#!/usr/bin/env python
"""Stream detector scores for a strict clean/watermarked/attacked manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

STAGES = ("clean", "watermarked", "attacked")
PROVENANCE_FIELDS = [
    "dataset", "run_id", "method", "prompt_id", "prompt", "source",
    "model_id", "model_revision", "vae_id", "vae_scaling_factor",
    "scheduler", "inverse_scheduler", "steps", "resolution", "detector_dtype",
    "score_direction", "provider_parameters", "generation_seed", "attack_seed",
    "watermark_seed", "fix_gt", "offset", "legacy_threshold", "target_fpr",
    "provider_config_hash", "target_watermark_hash", "source_watermark_target_sha256",
    "detector_watermark_target_sha256", "source_watermark_mask_sha256",
    "detector_watermark_mask_sha256", "gs_protocol_mode", "gs_secret_index",
    "gs_message_sha256", "gs_key_sha256", "gs_nonce_sha256",
    "gs_secret_bundle_sha256", "gs_sampling_seed", "gs_sampling_uniform_sha256",
    "gs_official_tau_onebit", "gs_official_tau_bits",
]
PATH_FIELDS = [item for stage in STAGES for item in (f"{stage}_path", f"{stage}_sha256")]
SCORE_FIELDS = [
    f"{stage}_{suffix}" for stage in STAGES for suffix in (
        "raw_score", "canonical_score", "tr_log_p", "tr_sigma", "tr_lambda",
        "tr_statistic", "tr_df", "tr_p_underflow",
    )
]
GS_FIELDS = [f"{stage}_decoded_bits_sha256" for stage in STAGES]
FIELDNAMES = PROVENANCE_FIELDS + PATH_FIELDS + SCORE_FIELDS + GS_FIELDS + ["error"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--metadata", type=Path, required=True, help="Strict pairing manifest CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Append after validated completed rows in an existing output")
    parser.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    parser.add_argument("--model-id", default="RedbeardNZ/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--scheduler", choices=["DDIM"], default="DDIM")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--min-cpu-mem-gb", type=float, default=92.0)
    parser.add_argument("--warn-cpu-mem-gb", type=float, default=110.0)
    parser.add_argument("--max-process-ram-gb", type=float, default=16.0)
    return parser


def first(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return None


def integer(row: dict[str, str], names: tuple[str, ...], default: int) -> int:
    value = first(row, *names)
    return int(value) if value is not None else default


def provider_kwargs(method: str, row: dict[str, str]) -> dict:
    if method == "GS":
        secret_index = integer(row, ("gs_secret_index", "offset"), 0)
        return {
            "offset": secret_index,
            "gs_secret_index": secret_index,
            "gs_sampling_seed": integer(row, ("gs_sampling_seed",), 0),
        }
    if method == "TR":
        return {
            "w_seed": integer(row, ("w_seed", "watermark_seed"), 999999),
            "w_channel": integer(row, ("w_channel",), 3),
            "w_radius": integer(row, ("w_radius",), 10),
            "w_pattern": first(row, "w_pattern") or "ring",
            "w_mask_shape": first(row, "w_mask_shape") or "circle",
            "w_measurement": first(row, "w_measurement") or "l1_complex",
            "w_injection": first(row, "w_injection") or "complex",
        }
    if method == "RID":
        return {
            "rid_seed": integer(row, ("rid_seed", "watermark_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt",), 1),
            "time_shift": integer(row, ("time_shift",), 1),
        }
    if method == "HSTR":
        return {"hstr_seed": integer(row, ("hstr_seed", "watermark_seed"), 999999), "fix_gt": integer(row, ("fix_gt",), 1)}
    if method == "HSQR":
        return {
            "hsqr_seed": integer(row, ("hsqr_seed", "watermark_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt",), 1),
            "delta": integer(row, ("delta",), 0),
        }
    raise ValueError(method)


def provider_class(method: str):
    if method == "TR":
        from utils.wm.tr_provider import TrProvider
        return TrProvider
    if method == "GS":
        from utils.wm.gs_provider import GsProvider
        return GsProvider
    if method == "RID":
        from utils.wm.ringid_provider import RingIDProvider
        return RingIDProvider
    if method == "HSTR":
        from utils.wm.hstr_provider import HSTRProvider
        return HSTRProvider
    if method == "HSQR":
        from utils.wm.hsqr_provider import HSQRProvider
        return HSQRProvider
    raise ValueError(method)


def raw_score(method: str, result: dict) -> float:
    if method == "TR":
        return float(result["p_values"][0])
    if method in {"RID", "HSTR", "HSQR"}:
        return float(result["l1_dist"][0])
    return float(result["bit_accuracies"][0])


def canonical_score(method: str, raw: float, result: dict) -> float:
    if method == "TR":
        diagnostics = result.get("p_value_diagnostics") or []
        log_p = float(diagnostics[0].get("log_p", float("nan"))) if diagnostics else float("nan")
        return -log_p / math.log(10.0) if math.isfinite(log_p) else -math.log10(max(raw, sys.float_info.min))
    return -raw if method in {"RID", "HSTR", "HSQR"} else raw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_image(torch, provider, pipe, path: Path, steps: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    with torch.no_grad():
        inversion = provider.invert_images(image, pipe_provider_target=pipe, num_inference_steps=steps) \
            if hasattr(provider, "invert_images") else pipe.invert_images(image, num_inference_steps=steps)
        recovered = inversion["zT_torch"]
        result = provider.get_accuracies(recovered)
        if provider.get_wm_type() == "TR":
            import scipy.stats

            diagnostics = []
            recovered_fft = torch.fft.fftshift(torch.fft.fft2(recovered), dim=(-1, -2))
            mask = provider.watermarking_mask[0]
            target = provider.gt_patch[0][mask].flatten()
            target = torch.concatenate([target.real, target.imag])
            for latent_fft in recovered_fft:
                observed = latent_fft[mask].flatten()
                observed = torch.concatenate([observed.real, observed.imag])
                sigma = observed.std()
                noncentrality = (target.square() / sigma.square()).sum().item()
                statistic = (((observed - target) / sigma).square()).sum().item()
                raw_log_p = float(scipy.stats.ncx2.logcdf(statistic, df=len(target), nc=noncentrality))
                p_value = float(result["p_values"][len(diagnostics)])
                p_underflow = p_value == 0.0 or not math.isfinite(raw_log_p)
                log_p = raw_log_p if math.isfinite(raw_log_p) else math.log(sys.float_info.min)
                diagnostics.append({
                    "log_p": log_p, "sigma": float(sigma.item()), "lambda": noncentrality,
                    "statistic": statistic, "df": len(target), "p_underflow": p_underflow,
                })
            result["p_value_diagnostics"] = diagnostics
    del inversion
    return result


def add_tr_diagnostics(record: dict, stage: str, result: dict) -> None:
    diagnostics = result.get("p_value_diagnostics") or []
    if not diagnostics:
        return
    item = diagnostics[0]
    mapping = {
        "log_p": "tr_log_p", "sigma": "tr_sigma", "lambda": "tr_lambda",
        "statistic": "tr_statistic", "df": "tr_df", "p_underflow": "tr_p_underflow",
    }
    for source, destination in mapping.items():
        record[f"{stage}_{destination}"] = item.get(source, "")


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing score file: {args.output}")
    sys.path.insert(0, str(args.eval_repo.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from raven.resource_guard import CpuMemoryGuard, limit_cpu_threads
    from raven.eval_protocol import canonical_json_hash, require_uniform_provider_config
    from raven.pairing_provenance import tensor_sha256
    from utils.pipe import pipe_utils
    from utils.utils import describe_legacy_detection_threshold

    limit_cpu_threads(1)
    guard = CpuMemoryGuard(args.min_cpu_mem_gb, args.max_process_ram_gb, args.warn_cpu_mem_gb)
    guard.check("detector extraction startup")
    with args.metadata.open(newline="", encoding="utf-8-sig") as source:
        manifest_rows = list(csv.DictReader(source))
    if not manifest_rows:
        raise ValueError(f"No rows in {args.metadata}")
    method = args.method.upper()
    uniform_kwargs, uniform_hash = require_uniform_provider_config(method, manifest_rows)
    recorded_hashes = {row.get("provider_config_hash", "") for row in manifest_rows}
    if recorded_hashes not in ({""}, {uniform_hash}):
        raise ValueError(f"Manifest provider hashes do not match canonical config: {sorted(recorded_hashes)}")

    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    load_options = {"revision": args.model_revision} if args.model_revision else {}
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id, resolution=args.resolution,
        device=device, eager_loading=False, schedulers_name=args.scheduler,
        disable_tqdm=True, **load_options,
    )
    latent_shape = pipe.get_latent_shape()
    vae_scaling = float(pipe.pipe.vae.config.scaling_factor)
    inverse_scheduler = type(pipe.scheduler_inverse).__name__
    legacy = describe_legacy_detection_threshold(args.method, args.model_id)

    provider = None
    target_hash = ""
    if method != "GS":
        provider = provider_class(method)(
            latent_shape=latent_shape, dtype=pipe.get_dtype(), device=device, **uniform_kwargs
        )
        target = getattr(provider, "gt_patch", None)
        target_hash = tensor_sha256(target) if target is not None else ""
    processed, errors = 0, 0
    completed: set[str] = set()
    output_mode, write_header = "x", True
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as existing:
            existing_reader = csv.DictReader(existing)
            if existing_reader.fieldnames != FIELDNAMES:
                raise ValueError(f"Cannot resume {args.output}: output schema does not match")
            for existing_row in existing_reader:
                if existing_row.get("error"):
                    raise ValueError(f"Cannot resume output containing failed row: {existing_row.get('run_id')}")
                identifier = existing_row.get("run_id")
                if not identifier or identifier in completed:
                    raise ValueError(f"Cannot resume output with missing/duplicate run_id: {identifier!r}")
                completed.add(identifier)
        output_mode, write_header = "a", False
        print(f"resuming {args.output} after {len(completed)} completed rows", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(output_mode, newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            output.flush()
            os.fsync(output.fileno())
        for index, row in enumerate(manifest_rows):
            if args.limit is not None and processed >= args.limit:
                break
            identifier = first(row, "run_id", "sample_id", "id", "index") or str(index)
            if identifier in completed:
                continue
            manifest_model_id = first(row, "model_id")
            if manifest_model_id and manifest_model_id != args.model_id:
                raise ValueError(
                    f"run_id={identifier}: manifest model_id={manifest_model_id!r} "
                    f"does not match --model-id={args.model_id!r}"
                )
            record = {field: "" for field in FIELDNAMES}
            record.update({
                "dataset": first(row, "dataset", "dataset_name") or "unspecified",
                "run_id": identifier, "method": method, "prompt_id": row.get("prompt_id", ""),
                "prompt": first(row, "prompt", "caption", "text") or "", "source": row.get("source", ""),
                "model_id": args.model_id, "model_revision": args.model_revision or row.get("model_revision") or "unspecified",
                "vae_id": row.get("vae_id") or "checkpoint-default", "vae_scaling_factor": vae_scaling,
                "scheduler": args.scheduler, "inverse_scheduler": inverse_scheduler, "steps": args.steps,
                "resolution": args.resolution, "detector_dtype": str(pipe.get_dtype()),
                "score_direction": "lower raw p-value means watermark; canonical=-log10(p)" if method == "TR" else
                    ("lower raw L1 means watermark; canonical=-L1" if method != "GS" else "higher bit accuracy means watermark"),
                "generation_seed": row.get("generation_seed", ""), "attack_seed": row.get("attack_seed", ""),
                "legacy_threshold": row.get("legacy_threshold") or legacy["threshold"], "target_fpr": args.target_fpr,
                "provider_config_hash": uniform_hash,
                "target_watermark_hash": (
                    row.get("watermark_target_sha256", "") if method == "GS" else target_hash
                ),
            })
            try:
                kwargs = dict(uniform_kwargs)
                if method == "GS":
                    kwargs.update(provider_kwargs(method, row))
                    provider = provider_class(method)(
                        latent_shape=latent_shape,
                        dtype=pipe.get_dtype(),
                        device=device,
                        **kwargs,
                    )
                    secret = provider.secret_provenance()
                    for field in (
                        "gs_secret_index", "gs_message_sha256", "gs_key_sha256",
                        "gs_nonce_sha256", "gs_secret_bundle_sha256",
                    ):
                        source_field = field
                        expected = str(row.get(source_field, ""))
                        actual = str(
                            secret[{
                                "gs_secret_index": "secret_index",
                                "gs_message_sha256": "message_sha256",
                                "gs_key_sha256": "key_sha256",
                                "gs_nonce_sha256": "nonce_sha256",
                                "gs_secret_bundle_sha256": "secret_bundle_sha256",
                            }[field]]
                        )
                        if not expected or expected != actual:
                            raise RuntimeError(
                                f"run_id={identifier}: detector/source {field} mismatch"
                            )
                        record[field] = actual
                    record["gs_protocol_mode"] = provider.gs_protocol_mode
                    for field in ("gs_sampling_seed", "gs_sampling_uniform_sha256"):
                        if not str(row.get(field, "")):
                            raise RuntimeError(f"run_id={identifier}: missing {field}")
                        record[field] = row[field]
                    thresholds = provider.official_thresholds()
                    record["gs_official_tau_onebit"] = thresholds["tau_onebit"]
                    record["gs_official_tau_bits"] = thresholds["tau_bits"]
                    source_target_hash = str(row.get("watermark_target_sha256", ""))
                    detector_target_hash = tensor_sha256(provider.watermark_target_tensor())
                    source_mask_hash = str(row.get("watermark_mask_sha256", ""))
                    detector_mask_hash = canonical_json_hash(
                        {"method": "GS", "mask": "not_applicable", "version": 1}
                    )
                    if not source_target_hash or source_target_hash != detector_target_hash:
                        raise RuntimeError(
                            f"run_id={identifier}: detector/source target SHA mismatch"
                        )
                    if not source_mask_hash or source_mask_hash != detector_mask_hash:
                        raise RuntimeError(
                            f"run_id={identifier}: detector/source mask SHA mismatch"
                        )
                    record.update({
                        "source_watermark_target_sha256": source_target_hash,
                        "detector_watermark_target_sha256": detector_target_hash,
                        "source_watermark_mask_sha256": source_mask_hash,
                        "detector_watermark_mask_sha256": detector_mask_hash,
                    })
                record["provider_parameters"] = json.dumps(uniform_kwargs, sort_keys=True)
                record["watermark_seed"] = next((kwargs[key] for key in ("w_seed", "rid_seed", "hstr_seed", "hsqr_seed") if key in kwargs), "")
                record["fix_gt"], record["offset"] = kwargs.get("fix_gt", ""), kwargs.get("offset", "")
                stage_results = {}
                for stage in STAGES:
                    path = Path(row[f"{stage}_path"]).resolve()
                    record[f"{stage}_path"] = str(path)
                    actual_sha = sha256(path)
                    expected_sha = row.get(f"{stage}_sha256")
                    if expected_sha and expected_sha != actual_sha:
                        raise RuntimeError(f"run_id={identifier}: {stage} SHA mismatch")
                    record[f"{stage}_sha256"] = actual_sha
                    guard.check(f"{method}/{identifier}/{stage}")
                    result = evaluate_image(torch, provider, pipe, path, args.steps)
                    stage_results[stage] = result
                    raw = raw_score(method, result)
                    record[f"{stage}_raw_score"] = raw
                    record[f"{stage}_canonical_score"] = canonical_score(method, raw, result)
                    if method == "TR":
                        add_tr_diagnostics(record, stage, result)
                if method == "GS":
                    for stage in STAGES:
                        decoded = stage_results[stage]["message_bits_str_list"][0]
                        record[f"{stage}_decoded_bits_sha256"] = hashlib.sha256(
                            decoded.encode("ascii")
                        ).hexdigest()
                del stage_results
            except Exception as exc:
                errors += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
            writer.writerow(record)
            output.flush()
            os.fsync(output.fileno())
            processed += 1
            print(f"[{method}] {processed} run_id={identifier} error={record['error'] or 'none'}", flush=True)
    print(f"wrote {args.output} rows={processed} errors={errors}", flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
