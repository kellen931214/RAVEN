#!/usr/bin/env python
"""Extract auditable raw watermark scores from clean/before/after images.

This script intentionally does not apply detection thresholds.  Its CSV is the
input to ``eval_reproduction.py``, which calibrates TPR@1%FPR from clean
negatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["GS", "TR", "RID", "HSTR", "HSQR"])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-repo", type=Path, default=Path(__file__).resolve().parents[2] / "eval_bench_wm")
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def first(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_path(value: str | None, metadata: Path, workspace: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [metadata.parent / path, workspace / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(row: dict[str, str], names: tuple[str, ...], default: int) -> int:
    value = first(row, names)
    return int(value) if value is not None else default


def provider_kwargs(method: str, row: dict[str, str], run_id: int) -> dict:
    if method == "GS":
        return {"offset": integer(row, ("offset", "message_index", "run_id", "sample_id"), run_id)}
    if method == "TR":
        return {"w_seed": integer(row, ("w_seed", "watermark_seed", "wm_seed"), 999999)}
    if method == "RID":
        return {
            "rid_seed": integer(row, ("rid_seed", "watermark_seed", "wm_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt",), 1),
        }
    if method == "HSTR":
        return {
            "hstr_seed": integer(row, ("hstr_seed", "watermark_seed", "wm_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt", "sample_index", "watermark_index", "run_id"), run_id),
        }
    if method == "HSQR":
        return {
            "hsqr_seed": integer(row, ("hsqr_seed", "watermark_seed", "wm_seed"), 999999),
            "fix_gt": integer(row, ("fix_gt", "sample_index", "watermark_index", "run_id"), run_id),
        }
    raise ValueError(method)


def raw_score(method: str, result: dict) -> float:
    if method == "TR":
        return float(result["p_values"][0])
    if method in {"RID", "HSTR", "HSQR"}:
        return float(result["l1_dist"][0])
    if method == "GS":
        return float(result["bit_accuracies"][0])
    raise ValueError(method)


def bits_from_bytes(value: bytes, length: int) -> str:
    return "".join(f"{byte:08b}" for byte in value)[:length]


def evaluate_image(torch, provider, pipe, path: Path, steps: int) -> tuple[dict, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    with torch.no_grad():
        if hasattr(provider, "invert_images"):
            inversion = provider.invert_images(
                image,
                pipe_provider_target=pipe,
                num_inference_steps=steps,
            )
        else:
            inversion = pipe.invert_images(image, num_inference_steps=steps)
        result = provider.get_accuracies(inversion["zT_torch"])
    return result, inversion["zT_torch"]


def main() -> int:
    args = build_parser().parse_args()
    metadata = args.metadata.resolve()
    eval_repo = args.eval_repo.resolve()
    sys.path.insert(0, str(eval_repo))
    import torch
    from utils.pipe import pipe_utils
    from utils.wm.wm_utils import WmProviders

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    load_options = {"revision": args.model_revision} if args.model_revision else {}
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
        **load_options,
    )

    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]

    output_rows = []
    method = args.method.upper()
    for index, row in enumerate(rows):
        identifier = first(row, ("run_id", "sample_id", "id", "index")) or str(index)
        run_id = int(identifier) if identifier.isdigit() else index
        clean = resolve_path(first(row, ("clean_path", "clean_image", "original_path")), metadata, args.workspace_root)
        before = resolve_path(
            first(row, ("watermarked_path", "image_path", "before_path", "path")), metadata, args.workspace_root
        )
        after = resolve_path(
            first(row, ("attacked_path", "after_path", "raven_path", "output_path")), metadata, args.workspace_root
        )
        record = {
            "run_id": identifier,
            "method": method,
            "model_id": args.model_id,
            "model_revision": args.model_revision or "unspecified",
            "steps": args.steps,
            "clean_path": str(clean) if clean else "",
            "watermarked_path": str(before) if before else "",
            "attacked_path": str(after) if after else "",
            "prompt": first(row, ("prompt", "caption", "text")) or "",
            "seed": first(row, ("seed", "generation_seed")) or "",
            "dx": first(row, ("dx", "shift_x")) or "",
            "dy": first(row, ("dy", "shift_y")) or "",
        }
        try:
            kwargs = provider_kwargs(method, row, run_id)
            provider_cls = WmProviders[method].value
            provider = provider_cls(
                latent_shape=pipe.get_latent_shape(),
                dtype=pipe.get_dtype(),
                device=device,
                **kwargs,
            )
            stage_results = {}
            for stage, path in (("clean", clean), ("watermarked", before), ("attacked", after)):
                if path is None:
                    raise ValueError(f"missing {stage} path in metadata")
                result, _ = evaluate_image(torch, provider, pipe, path, args.steps)
                stage_results[stage] = result
                record[f"{stage}_raw_score"] = raw_score(method, result)
                record[f"{stage}_sha256"] = sha256(path)
            if method == "GS":
                offset = int(getattr(provider, "offset", 0))
                expected = int(provider.message_width_in_bits)
                record["ground_truth_bits"] = bits_from_bytes(provider.messages[offset], expected)
                record["watermarked_predicted_bits"] = stage_results["watermarked"]["message_bits_str_list"][0]
                record["attacked_predicted_bits"] = stage_results["attacked"]["message_bits_str_list"][0]
            record["error"] = ""
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        output_rows.append(record)
        print(f"[{method}] {index + 1}/{len(rows)} run_id={identifier} error={record['error'] or 'none'}")

    fieldnames = sorted({key for row in output_rows for key in row})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    errors = sum(bool(row["error"]) for row in output_rows)
    print(f"wrote {args.output} rows={len(output_rows)} errors={errors}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
