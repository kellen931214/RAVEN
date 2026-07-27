#!/usr/bin/env python3
"""Diagnose watermark Bit Accuracy without using the benchmark table formatter.

The script has two deliberately separate modes:

* ``--contract-only`` inspects the checked-in provider implementations with the
  Python AST.  It needs no ML dependencies or dataset files and exposes missing
  ``bit_accuracies`` / predicted-bit result keys without converting them to 0.
* The default runtime mode reads the benchmark metadata/images, validates the
  before/after mapping, hashes the files, uses the EvalBench inversion pipeline,
  and invokes the selected provider's real ``get_accuracies`` implementation.

Missing bit vectors are reported as missing metrics.  They are never replaced
with 0.0 or 0.5.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageOps, ImageStat


METHODS = ("GS", "TR", "RID", "HSTR", "HSQR")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RAVEN_FINAL_NAMES = ("final_color_corrected.png", "final.png", "view_guided_output.png")


@dataclass(frozen=True)
class MethodSpec:
    provider_file: str
    provider_class: str
    gt_description: str
    expected_bit_length: Optional[int]
    runtime_decoder: str


METHOD_SPECS: Dict[str, MethodSpec] = {
    "GS": MethodSpec(
        "gs_provider.py",
        "GsProvider",
        "provider.messages[offset], 32 bytes converted MSB-first to 256 bits",
        256,
        "GsProvider.get_accuracies (Gaussian recovery, ChaCha20 decrypt, majority vote)",
    ),
    "TR": MethodSpec(
        "tr_provider.py",
        "TrProvider",
        "seeded complex FFT ring template; no bit payload is exposed",
        None,
        "TrProvider.get_accuracies (noncentral-chi-square p-value detector only)",
    ),
    "RID": MethodSpec(
        "ringid_provider.py",
        "RingIDProvider",
        "one complex template from a 2048-key pattern space; selected key bits are not retained",
        None,
        "RingIDProvider.get_accuracies (matched-template L1 detector only)",
    ),
    "HSTR": MethodSpec(
        "hstr_provider.py",
        "HSTRProvider",
        "one complex template from a 2048-key pattern space; selected key index is discarded",
        None,
        "HSTRProvider.get_accuracies (matched-template L1 detector only)",
    ),
    "HSQR": MethodSpec(
        "hsqr_provider.py",
        "HSQRProvider",
        "provider.gt_patch, Boolean [1, 42, 42] QR modules",
        1764,
        "HSQRProvider.get_accuracies (matched-template L1 detector; no module decoder)",
    ),
}


@dataclass
class Sample:
    row_index: int
    identifier: str
    metadata: Dict[str, str]
    before_path: Path
    after_path: Optional[Path]
    clean_path: Optional[Path]


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_workspace = Path(os.environ.get("RAVEN_WORKSPACE_ROOT", "/workspace"))
    if not default_workspace.exists():
        default_workspace = repo_root

    parser = argparse.ArgumentParser(
        description="Inspect actual Bit Accuracy inputs/outputs without benchmark aggregation."
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", default="mscoco")
    parser.add_argument("--max-samples", type=positive_int, default=3)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--before-only", action="store_true")
    direction.add_argument("--after-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Inspect source/result contracts only; do not load images or ML dependencies.",
    )
    parser.add_argument("--workspace-root", type=Path, default=default_workspace)
    parser.add_argument("--eval-repo", type=Path, default=repo_root / "eval_bench_wm")
    parser.add_argument("--watermarked-dir", type=Path)
    parser.add_argument("--raven-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--model-id",
        default="RedbeardNZ/stable-diffusion-2-1-base",
        help="Must match watermark generation/evaluation.",
    )
    parser.add_argument("--resolution", type=positive_int, default=512)
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--derive-hsqr-bits",
        action="store_true",
        help=(
            "Additionally run a diagnostic HSQR sign decoder. This decoder is not currently "
            "called by HSQRProvider.get_accuracies and is labelled separately."
        ),
    )
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _find_class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise RuntimeError(f"Could not find {class_name}.{method_name}")


def _literal_dict_keys(node: ast.AST) -> List[str]:
    keys: List[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Return) or not isinstance(child.value, ast.Dict):
            continue
        for key in child.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
    return sorted(set(keys))


def inspect_contract(eval_repo: Path, method: str) -> Dict[str, Any]:
    spec = METHOD_SPECS[method]
    provider_path = eval_repo / "utils" / "wm" / spec.provider_file
    validate_path = eval_repo / "utils" / "imprint_utils.py"
    if not provider_path.is_file():
        raise FileNotFoundError(f"Provider source not found: {provider_path}")
    if not validate_path.is_file():
        raise FileNotFoundError(f"Validation helper not found: {validate_path}")

    provider_tree = ast.parse(provider_path.read_text(encoding="utf-8"), filename=str(provider_path))
    method_node = _find_class_method(provider_tree, spec.provider_class, "get_accuracies")
    result_keys = _literal_dict_keys(method_node)

    validate_source = validate_path.read_text(encoding="utf-8")
    missing_to_zero = (
        'accuracy_results["bit_accuracies"][0]'
        in validate_source
        and 'if "bit_accuracies" in accuracy_results else 0.0' in validate_source
    )
    return {
        "method": method,
        "provider": f"{spec.provider_class}.get_accuracies",
        "provider_path": str(provider_path),
        "provider_lines": [method_node.lineno, getattr(method_node, "end_lineno", method_node.lineno)],
        "result_keys": result_keys,
        "has_bit_accuracies": "bit_accuracies" in result_keys,
        "has_predicted_bits": bool(
            {"predicted_bits", "pred_bits", "message_bits_str_list"}.intersection(result_keys)
        ),
        "gt_description": spec.gt_description,
        "expected_bit_length": spec.expected_bit_length,
        "runtime_decoder": spec.runtime_decoder,
        "validate_path": str(validate_path),
        "validate_missing_bit_accuracy_to_zero": missing_to_zero,
    }


def print_contract(contract: Mapping[str, Any]) -> None:
    print("=== provider contract ===")
    print(json.dumps(contract, indent=2, ensure_ascii=False))
    if not contract["has_bit_accuracies"]:
        if contract["validate_missing_bit_accuracy_to_zero"]:
            print(
                "CONTRACT FAILURE: provider returns no bit_accuracies. "
                "The checked-in validate() helper converts this missing metric to literal 0.0."
            )
        else:
            print(
                "CONTRACT NOTICE: provider returns no bit_accuracies; "
                "the validation helper preserves it as an unavailable metric."
            )
    else:
        print("CONTRACT OK: provider exposes bit_accuracies.")


def _first_nonempty(row: Mapping[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _resolve_existing_path(value: str, bases: Sequence[Path]) -> Optional[Path]:
    candidate = Path(os.path.expandvars(os.path.expanduser(value)))
    if candidate.is_absolute():
        return candidate.resolve() if candidate.is_file() else None
    for base in bases:
        path = (base / candidate).resolve()
        if path.is_file():
            return path
    return None


def _image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _identifier(row: Mapping[str, str], row_index: int, fallback_path: Optional[Path] = None) -> str:
    value = _first_nonempty(
        row,
        ("run_id", "sample_id", "sample_identifier", "id", "index", "image_id", "filename", "file"),
    )
    if value:
        return Path(value).stem if Path(value).suffix else value
    if fallback_path is not None:
        return fallback_path.stem
    return str(row_index)


def _numeric_variants(identifier: str) -> List[str]:
    variants = [identifier]
    try:
        number = int(identifier)
    except ValueError:
        return variants
    for width in (6, 5, 4):
        variants.append(f"{number:0{width}d}")
    variants.append(str(number))
    return list(dict.fromkeys(variants))


def resolve_before_path(
    row: Mapping[str, str],
    identifier: str,
    watermarked_dir: Path,
    metadata_path: Optional[Path],
    workspace_root: Path,
) -> Path:
    bases = [watermarked_dir, workspace_root]
    if metadata_path is not None:
        bases.insert(0, metadata_path.parent)
    value = _first_nonempty(
        row,
        (
            "watermarked_path",
            "watermarked_image_path",
            "wm_image_path",
            "image_path",
            "output_path",
            "filename",
            "file",
        ),
    )
    if value:
        path = _resolve_existing_path(value, bases)
        if path is not None:
            return path

    matches: List[Path] = []
    for stem in _numeric_variants(identifier):
        for suffix in sorted(IMAGE_SUFFIXES):
            path = watermarked_dir / f"{stem}{suffix}"
            if path.is_file():
                matches.append(path.resolve())
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No watermarked image for sample {identifier!r} under {watermarked_dir}; "
            "provide --metadata/--watermarked-dir with the original benchmark files."
        )
    raise RuntimeError(f"Ambiguous watermarked mapping for {identifier!r}: {matches}")


def resolve_after_path(
    row: Mapping[str, str],
    identifier: str,
    before_path: Path,
    raven_dir: Path,
    workspace_root: Path,
) -> Optional[Path]:
    value = _first_nonempty(
        row,
        (
            "raven_path",
            "raven_output_path",
            "after_path",
            "attacked_image_path",
            "final_image_path",
        ),
    )
    if value:
        path = _resolve_existing_path(value, [raven_dir, workspace_root])
        if path is not None:
            return path

    base_names = list(dict.fromkeys([identifier, before_path.stem] + _numeric_variants(identifier)))
    item_names: List[str] = []
    for base_name in base_names:
        item_names.extend(
            (base_name, f"run_id={base_name}", f"run_{base_name}", f"sample_{base_name}")
        )
    item_names = list(dict.fromkeys(item_names))
    matches: List[Path] = []
    for item_name in item_names:
        item_dir = raven_dir / item_name
        for final_name in RAVEN_FINAL_NAMES:
            path = item_dir / final_name
            if path.is_file():
                matches.append(path.resolve())
                break
        for suffix in sorted(IMAGE_SUFFIXES):
            flat_path = raven_dir / f"{item_name}{suffix}"
            if flat_path.is_file():
                matches.append(flat_path.resolve())
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise RuntimeError(f"Ambiguous RAVEN mapping for {identifier!r}: {matches}")


def resolve_clean_path(
    row: Mapping[str, str], metadata_path: Optional[Path], workspace_root: Path
) -> Optional[Path]:
    value = _first_nonempty(
        row,
        ("clean_path", "clean_image_path", "original_path", "source_path", "generated_path"),
    )
    if not value:
        return None
    bases = [workspace_root]
    if metadata_path is not None:
        bases.insert(0, metadata_path.parent)
    return _resolve_existing_path(value, bases)


def load_samples(args: argparse.Namespace) -> Tuple[List[Sample], Path, Path, Optional[Path]]:
    workspace = args.workspace_root.resolve()
    watermarked_dir = (
        args.watermarked_dir
        or workspace / "data" / "watermarked" / args.dataset / args.method
    ).resolve()
    raven_dir = (
        args.raven_dir
        or workspace / "outputs" / "raven_eval" / args.dataset / args.method
    ).resolve()
    metadata_path = (args.metadata or watermarked_dir / "metadata.csv").resolve()

    rows: List[Dict[str, str]] = []
    fallback_images: List[Path] = []
    if metadata_path.is_file():
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"Metadata has no header: {metadata_path}")
            rows = [dict(row) for row in reader]
    else:
        fallback_images = _image_files(watermarked_dir)
        rows = [{} for _ in fallback_images]
        metadata_path = None

    if not rows:
        raise FileNotFoundError(
            "No benchmark samples were found. Checked "
            f"metadata={metadata_path or watermarked_dir / 'metadata.csv'} and images={watermarked_dir}."
        )

    samples: List[Sample] = []
    used_before: Dict[Path, str] = {}
    used_after: Dict[Path, str] = {}
    for row_index, row in enumerate(rows[: args.max_samples]):
        fallback = fallback_images[row_index] if fallback_images else None
        identifier = _identifier(row, row_index, fallback)
        before_path = fallback or resolve_before_path(
            row, identifier, watermarked_dir, metadata_path, workspace
        )
        after_path = resolve_after_path(row, identifier, before_path, raven_dir, workspace)
        clean_path = resolve_clean_path(row, metadata_path, workspace)

        previous = used_before.get(before_path)
        if previous is not None:
            raise RuntimeError(
                f"Duplicate before mapping: {previous!r} and {identifier!r} -> {before_path}"
            )
        used_before[before_path] = identifier
        if after_path is not None:
            previous = used_after.get(after_path)
            if previous is not None:
                raise RuntimeError(
                    f"Duplicate after mapping: {previous!r} and {identifier!r} -> {after_path}"
                )
            used_after[after_path] = identifier
        samples.append(
            Sample(row_index, identifier, row, before_path, after_path, clean_path)
        )
    return samples, watermarked_dir, raven_dir, metadata_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_stats(path: Path) -> Dict[str, Any]:
    with Image.open(path) as opened:
        source_mode = opened.mode
        image = ImageOps.exif_transpose(opened).convert("RGB")
        stat = ImageStat.Stat(image)
        extrema = image.getextrema()
        return {
            "path": str(path),
            "file_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_mode": source_mode,
            "decoded_mode": image.mode,
            "shape_hwc": [image.height, image.width, 3],
            "pixel_min_rgb": [item[0] for item in extrema],
            "pixel_max_rgb": [item[1] for item in extrema],
            "pixel_mean_rgb": [round(value, 6) for value in stat.mean],
            "pixel_std_rgb": [round(value, 6) for value in stat.stddev],
        }


def image_difference(first: Path, second: Path) -> Dict[str, Any]:
    with Image.open(first) as first_opened, Image.open(second) as second_opened:
        a = ImageOps.exif_transpose(first_opened).convert("RGB")
        b = ImageOps.exif_transpose(second_opened).convert("RGB")
        if a.size != b.size:
            return {"comparable": False, "first_size": list(a.size), "second_size": list(b.size)}
        diff = ImageChops.difference(a, b)
        stat = ImageStat.Stat(diff)
        mean_abs_rgb = stat.mean
        return {
            "comparable": True,
            "identical_pixels": diff.getbbox() is None,
            "mean_abs_difference_rgb": [round(value, 6) for value in mean_abs_rgb],
            "mean_abs_difference": round(statistics.fmean(mean_abs_rgb), 6),
        }


def print_mapping(samples: Sequence[Sample], method: str) -> None:
    print("=== path and image mapping ===")
    for sample in samples:
        payload: Dict[str, Any] = {
            "method": method,
            "sample_identifier": sample.identifier,
            "clean": image_stats(sample.clean_path) if sample.clean_path else None,
            "before": image_stats(sample.before_path),
            "after": image_stats(sample.after_path) if sample.after_path else None,
        }
        if sample.clean_path:
            payload["clean_to_before_diff"] = image_difference(sample.clean_path, sample.before_path)
        if sample.after_path:
            payload["before_to_after_diff"] = image_difference(sample.before_path, sample.after_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def _row_int(row: Mapping[str, str], keys: Sequence[str], default: int) -> Tuple[int, str]:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            parsed = str(value).strip()
            return int(parsed), f"metadata.{key}={parsed}"
    return default, f"provider default {default}"


def provider_kwargs(method: str, sample: Sample) -> Tuple[Dict[str, Any], Dict[str, str]]:
    row = sample.metadata
    kwargs: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    try:
        run_id = int(sample.identifier)
    except ValueError:
        run_id = sample.row_index

    if method == "GS":
        offset, source = _row_int(row, ("offset", "message_index", "run_id", "sample_id"), run_id)
        kwargs["offset"] = offset
        sources["offset"] = source if source.startswith("metadata") else "sample identifier/row index"
    elif method == "TR":
        kwargs["w_seed"], sources["w_seed"] = _row_int(
            row, ("w_seed", "watermark_seed", "wm_seed"), 999999
        )
    elif method == "RID":
        kwargs["rid_seed"], sources["rid_seed"] = _row_int(
            row, ("rid_seed", "watermark_seed", "wm_seed"), 999999
        )
        kwargs["fix_gt"], sources["fix_gt"] = _row_int(row, ("fix_gt",), 1)
    elif method == "HSTR":
        kwargs["hstr_seed"], sources["hstr_seed"] = _row_int(
            row, ("hstr_seed", "watermark_seed", "wm_seed"), 999999
        )
        kwargs["fix_gt"], source = _row_int(
            row, ("fix_gt", "sample_index", "watermark_index", "run_id", "sample_id"), run_id
        )
        sources["fix_gt"] = source if source.startswith("metadata") else "sample identifier/row index"
    elif method == "HSQR":
        kwargs["hsqr_seed"], sources["hsqr_seed"] = _row_int(
            row, ("hsqr_seed", "watermark_seed", "wm_seed"), 999999
        )
        kwargs["fix_gt"], source = _row_int(
            row, ("fix_gt", "sample_index", "watermark_index", "run_id", "sample_id"), run_id
        )
        sources["fix_gt"] = source if source.startswith("metadata") else "sample identifier/row index"
    return kwargs, sources


def load_runtime(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    eval_repo = args.eval_repo.resolve()
    if not eval_repo.is_dir():
        raise FileNotFoundError(f"EvalBench repository not found: {eval_repo}")
    sys.path.insert(0, str(eval_repo))
    try:
        import torch
        from utils.pipe import pipe_utils
        from utils.wm.wm_utils import WmProviders
    except ImportError as exc:
        raise RuntimeError(
            "Runtime diagnostics require the original evaluation environment (torch, numpy, scipy, "
            "diffusers, transformers, torchvision, cryptography, and provider dependencies). "
            f"Import failed: {type(exc).__name__}: {exc}"
        ) from exc

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    pipe = pipe_utils.get_pipe_provider(
        pretrained_model_name_or_path=args.model_id,
        resolution=args.resolution,
        device=device,
        eager_loading=False,
        schedulers_name="DDIM",
        disable_tqdm=True,
    )
    return torch, WmProviders, pipe


def tensor_summary(torch: Any, value: Any, first_values: int = 16) -> Dict[str, Any]:
    detached = value.detach().cpu()
    flat = detached.reshape(-1)
    summary: Dict[str, Any] = {
        "type": type(value).__name__,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }
    if value.numel() == 0:
        return summary
    if torch.is_complex(detached):
        magnitudes = detached.abs()
        summary.update(
            min_abs=float(magnitudes.min()),
            max_abs=float(magnitudes.max()),
            mean_abs=float(magnitudes.float().mean()),
            first_values=[str(item.item()) for item in flat[:first_values]],
        )
    else:
        finite = torch.isfinite(detached) if detached.dtype.is_floating_point else None
        summary.update(
            min=float(detached.min()),
            max=float(detached.max()),
            mean=float(detached.float().mean()),
            first_values=flat[:first_values].tolist(),
        )
        if finite is not None:
            summary["all_finite"] = bool(finite.all())
    try:
        unique = torch.unique(detached)
        if unique.numel() <= 16:
            summary["unique_values"] = unique.tolist()
    except (RuntimeError, TypeError):
        pass
    return summary


def summarize_raw(torch: Any, value: Any, verbose: bool) -> Any:
    if torch.is_tensor(value):
        return tensor_summary(torch, value, first_values=32 if verbose else 8)
    if isinstance(value, Mapping):
        return {str(key): summarize_raw(torch, item, verbose) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        limit = len(value) if verbose else min(len(value), 8)
        return [summarize_raw(torch, item, verbose) for item in value[:limit]] + (
            [f"... {len(value) - limit} more"] if len(value) > limit else []
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and not verbose and len(value) > 64:
            return value[:64] + "..."
        return value
    return {"type": type(value).__name__, "repr": repr(value)[:256]}


def normalize_bits(torch: Any, value: Any) -> Tuple[Optional[List[int]], Optional[str]]:
    if value is None:
        return None, "value is None"
    if isinstance(value, str):
        if value and set(value) <= {"0", "1"}:
            return [int(char) for char in value], None
        return None, "string is not a {0,1} bit string"
    if isinstance(value, (bytes, bytearray)):
        return [int(bit) for byte in value for bit in f"{byte:08b}"], None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flattened: List[Any] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                flattened.extend(item)
            else:
                flattened.append(item)
        value = flattened
    else:
        return None, f"unsupported bit container {type(value).__name__}"

    unique = set(value)
    if unique <= {False, True, 0, 1}:
        return [int(item) for item in value], None
    if unique <= {-1, 1}:
        return [1 if item > 0 else 0 for item in value], None
    return None, f"values are not Boolean/{{0,1}}/{{-1,1}}: {sorted(unique, key=str)[:16]}"


def extract_provider_bits(
    torch: Any, method: str, provider: Any, raw: Mapping[str, Any]
) -> Tuple[Optional[List[int]], Optional[List[int]], str, List[str]]:
    notes: List[str] = []
    gt_value = None
    pred_value = None
    gt_source = "not exposed"

    for key in ("ground_truth_bits", "gt_bits", "target_bits"):
        if key in raw:
            gt_value = raw[key]
            gt_source = f"provider result key {key}"
            break
    for key in ("predicted_bits", "pred_bits", "decoded_bits"):
        if key in raw:
            pred_value = raw[key]
            break

    if method == "GS":
        offset = int(getattr(provider, "offset", 0))
        gt_value = provider.messages[offset]
        gt_source = f"GsProvider.messages[{offset}] (message contents redacted to bit prefix)"
        recovered = raw.get("message_bits_str_list")
        pred_value = recovered[0] if isinstance(recovered, Sequence) and recovered else None
    elif method == "HSQR":
        gt_value = getattr(provider, "gt_patch", None)
        gt_source = "HSQRProvider.gt_patch Boolean QR modules"

    gt_bits, gt_error = normalize_bits(torch, gt_value)
    pred_bits, pred_error = normalize_bits(torch, pred_value)
    if gt_error:
        notes.append(f"ground truth bits unavailable: {gt_error}")
    if pred_error:
        notes.append(f"predicted bits unavailable: {pred_error}")
    return gt_bits, pred_bits, gt_source, notes


def summarize_provider_ground_truth(torch: Any, method: str, provider: Any) -> Dict[str, Any]:
    if method == "GS":
        offset = int(getattr(provider, "offset", 0))
        message = provider.messages[offset]
        bits = [int(bit) for byte in message for bit in f"{byte:08b}"]
        return {
            "source": f"GsProvider.messages[{offset}]",
            "raw_type": type(message).__name__,
            "raw_message_bytes_used": min(len(message), int(provider.message_width_in_bytes)),
            "bit_shape": [int(provider.message_width_in_bits)],
            "bit_dtype": "int{0,1}",
            "bit_unique_values": sorted(set(bits[: int(provider.message_width_in_bits)])),
            "first_32_bits": "".join(map(str, bits[:32])),
            "secret_key_and_nonce": "redacted",
        }
    gt_patch = getattr(provider, "gt_patch", None)
    if gt_patch is None:
        return {"source": "provider.gt_patch", "available": False}
    return {
        "source": f"{type(provider).__name__}.gt_patch",
        "available": True,
        "value": summarize_raw(torch, gt_patch, verbose=False),
        "bit_interpretation": (
            "Boolean QR modules" if method == "HSQR" else "complex detector template, not a bit vector"
        ),
    }


def derive_hsqr_bits(torch: Any, provider: Any, latents: Any) -> List[int]:
    """Diagnostic decoder mirroring HSQR's sign injection, not its current evaluator."""
    center = latents[:, :, provider.start : provider.end, provider.start : provider.end]
    coefficients = provider.rfft(center)
    channel = 3
    qr_size = int(provider.gt_patch.shape[-1])
    half = (qr_size + 1) // 2
    region = coefficients[0, channel, 1 : 1 + qr_size, 1 : 1 + half]
    predicted = torch.cat((region.real > 0, region.imag > 0), dim=-1)
    return [int(item) for item in predicted.reshape(-1).detach().cpu().tolist()]


def compare_bits(gt_bits: Optional[List[int]], pred_bits: Optional[List[int]]) -> Dict[str, Any]:
    if gt_bits is None or pred_bits is None:
        return {
            "available": False,
            "bit_accuracy": None,
            "reason": "ground-truth and predicted bit vectors are both required",
        }
    if len(gt_bits) != len(pred_bits):
        return {
            "available": False,
            "bit_accuracy": None,
            "gt_length": len(gt_bits),
            "prediction_length": len(pred_bits),
            "reason": "shape/length mismatch; comparison was not broadcast or truncated",
        }
    equal = sum(left == right for left, right in zip(gt_bits, pred_bits))
    total = len(gt_bits)
    return {
        "available": True,
        "gt_shape": [total],
        "prediction_shape": [total],
        "gt_dtype": "int{0,1}",
        "prediction_dtype": "int{0,1}",
        "gt_unique_values": sorted(set(gt_bits)),
        "prediction_unique_values": sorted(set(pred_bits)),
        "gt_first_32_bits": "".join(map(str, gt_bits[:32])),
        "prediction_first_32_bits": "".join(map(str, pred_bits[:32])),
        "equal_bits": equal,
        "total_bits": total,
        "bit_accuracy": equal / total if total else None,
    }


def evaluate_image(
    torch: Any,
    provider: Any,
    pipe: Any,
    image_path: Path,
    args: argparse.Namespace,
) -> Tuple[Mapping[str, Any], Any]:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    with torch.no_grad():
        if hasattr(provider, "invert_images"):
            inversion = provider.invert_images(
                image,
                pipe_provider_target=pipe,
                num_inference_steps=args.steps,
            )
        else:
            inversion = pipe.invert_images(image, num_inference_steps=args.steps)
        latents = inversion["zT_torch"]
        raw = provider.get_accuracies(latents)
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"{type(provider).__name__}.get_accuracies returned {type(raw).__name__}, expected mapping"
        )
    return raw, latents


def runtime_diagnostics(args: argparse.Namespace, samples: Sequence[Sample]) -> int:
    torch, providers, pipe = load_runtime(args)
    measured: Dict[str, List[float]] = {"before": [], "after": []}
    missing: Dict[str, int] = {"before": 0, "after": 0}

    for sample in samples:
        kwargs, gt_mapping_sources = provider_kwargs(args.method, sample)
        provider_cls = providers[args.method].value
        provider = provider_cls(
            latent_shape=pipe.get_latent_shape(),
            device=pipe.device,
            **kwargs,
        )
        stages: List[Tuple[str, Optional[Path]]] = []
        if not args.after_only:
            stages.append(("before", sample.before_path))
        if not args.before_only:
            stages.append(("after", sample.after_path))

        for stage, image_path in stages:
            if image_path is None:
                missing[stage] += 1
                print(
                    json.dumps(
                        {
                            "method": args.method,
                            "sample_identifier": sample.identifier,
                            "stage": stage,
                            "error": "RAVEN output image is missing",
                            "bit_accuracy": None,
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            raw, latents = evaluate_image(torch, provider, pipe, image_path, args)
            gt_bits, pred_bits, gt_source, notes = extract_provider_bits(
                torch, args.method, provider, raw
            )
            decoder_name = f"{type(provider).__name__}.get_accuracies"
            if args.method == "HSQR" and args.derive_hsqr_bits:
                pred_bits = derive_hsqr_bits(torch, provider, latents)
                decoder_name += " + diagnostic HSQR rFFT sign decoder (not benchmark code)"
                notes.append("HSQR predicted bits were derived diagnostically; current provider does not emit them")
            comparison = compare_bits(gt_bits, pred_bits)
            if comparison["available"]:
                measured[stage].append(float(comparison["bit_accuracy"]))
            else:
                missing[stage] += 1

            record = {
                "method": args.method,
                "sample_identifier": sample.identifier,
                "stage": stage,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "ground_truth_source": gt_source,
                "ground_truth_raw": summarize_provider_ground_truth(
                    torch, args.method, provider
                ),
                "ground_truth_mapping": gt_mapping_sources,
                "inversion_function": f"{type(pipe).__name__}.invert_images",
                "decoder_function": decoder_name,
                "raw_output": summarize_raw(torch, raw, args.verbose),
                "recovered_latent": tensor_summary(torch, latents, 16 if args.verbose else 8),
                "comparison": comparison,
                "notes": notes,
            }
            print(json.dumps(record, indent=2, ensure_ascii=False))

    aggregate = {
        stage: {
            "decoded_samples": len(values),
            "missing_metric_samples": missing[stage],
            "aggregate_bit_accuracy": statistics.fmean(values) if values else None,
        }
        for stage, values in measured.items()
        if (stage == "before" and not args.after_only) or (stage == "after" and not args.before_only)
    }
    print("=== aggregate Bit Accuracy ===")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if any(item["aggregate_bit_accuracy"] is None for item in aggregate.values()):
        print("MISSING METRIC: no aggregate was fabricated from absent predicted/GT bits.")
        return 2
    return 0


def main() -> int:
    args = build_parser().parse_args()
    contract = inspect_contract(args.eval_repo.resolve(), args.method)
    print_contract(contract)
    if args.contract_only:
        return 0 if contract["has_bit_accuracies"] else 2

    samples, watermarked_dir, raven_dir, metadata_path = load_samples(args)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "method": args.method,
                "watermarked_dir": str(watermarked_dir),
                "raven_dir": str(raven_dir),
                "metadata": str(metadata_path) if metadata_path else None,
                "samples_selected": len(samples),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print_mapping(samples, args.method)
    return runtime_diagnostics(args, samples)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
