"""Minimal CPU-only smoke test for the public RAVEN package.

Verifies:
- ``raven`` package import
- Detector registry has all 7 methods
- ``raven_repro/main.py`` and ``raven_repro/eval.py`` parsers are constructable
- ``normalize_config`` basic round-trip
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "raven_repro"))


def test_raven_import() -> None:
    import raven
    assert raven


def test_detector_registry_all_seven_methods() -> None:
    from raven.detectors import get_detector_module
    for method in ("TR", "GS", "GM", "T2S", "RID", "HSTR", "HSQR"):
        mod = get_detector_module(method)
        assert mod is not None, f"missing detector module: {method}"
        assert hasattr(mod, "load_state"), f"{method}: missing load_state"
        assert hasattr(mod, "score_image"), f"{method}: missing score_image"
        assert hasattr(mod, "aggregate"), f"{method}: missing aggregate"


def test_main_parser() -> None:
    from main import build_parser
    parser = build_parser()
    assert parser is not None


def test_eval_parser() -> None:
    from eval import build_parser
    parser = build_parser()
    assert parser is not None


def test_normalize_config_roundtrip() -> None:
    from raven.experiment_config import normalize_config, ALGORITHM_FIELDS
    config = normalize_config(
        diffusion_mode="ddim",
        method="TR",
        dataset="smoke",
        metadata_path="/tmp/metadata.csv",
        output_dir="/tmp/out",
    )
    for field in ALGORITHM_FIELDS:
        assert field in config, f"missing algorithm field: {field}"
    assert config["diffusion_mode"] == "ddim"
    assert config["method"] == "TR"
    assert config["dataset"] == "smoke"



def test_eval_pixel_shift_parser_and_black_fill() -> None:
    from PIL import Image
    from eval import build_parser, pixel_shift_with_black_fill

    args = build_parser().parse_args([
        "--workflow", "pixel-shift", "--output-dir", "/tmp/pixel-shift",
        "--metadata", "/tmp/metadata.csv",
    ])
    assert args.workflow == "pixel-shift"
    assert args.stages is None

    image = Image.new("RGB", (3, 2))
    image.putdata([
        (10, 0, 0), (20, 0, 0), (30, 0, 0),
        (40, 0, 0), (50, 0, 0), (60, 0, 0),
    ])
    shifted = pixel_shift_with_black_fill(image, 1, 0)
    assert list(shifted.getdata()) == [
        (0, 0, 0), (10, 0, 0), (20, 0, 0),
        (0, 0, 0), (40, 0, 0), (50, 0, 0),
    ]


def test_rebuild_pixel_shift_records_uses_canonical_clean_output(tmp_path, monkeypatch) -> None:
    from PIL import Image
    import eval as raven_eval
    from raven.experiment_io import output_image_path, read_records_jsonl

    clean = tmp_path / "clean.png"
    watermarked = tmp_path / "watermarked.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(clean)
    Image.new("RGB", (2, 2), (4, 5, 6)).save(watermarked)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "run_id,watermarked_path,clean_path,prompt\n"
        f"sample-1,{watermarked},{clean},a prompt\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    run_dir = output_root / "pixel_shift_raven_16px"
    output_image_path(run_dir, "watermarked", "sample-1").parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (7, 8, 9)).save(
        output_image_path(run_dir, "watermarked", "sample-1"),
    )
    args = raven_eval.build_parser().parse_args([
        "--workflow", "rebuild-pixel-shift", "--output-dir", str(output_root),
        "--metadata", str(metadata), "--magnitudes", "16", "--stages", "quality",
    ])
    monkeypatch.setattr(
        raven_eval, "_evaluate_pixel_shift_directory",
        lambda _args, _dir: {"stages": {}, "failed_stages": [], "skipped_stages": []},
    )

    result = raven_eval.rebuild_pixel_shift_records(args)

    records = read_records_jsonl(run_dir)
    assert result["magnitudes"]["16"]["rebuilt_watermarked_count"] == 1
    assert {record["role"] for record in records} == {"clean", "watermarked"}
    assert output_image_path(run_dir, "clean", "sample-1").is_file()
    clean_record = next(record for record in records if record["role"] == "clean")
    assert clean_record["output_path"] == str(output_image_path(run_dir, "clean", "sample-1"))
