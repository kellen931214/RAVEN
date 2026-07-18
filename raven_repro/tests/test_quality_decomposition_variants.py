from pathlib import Path

import pytest
from PIL import Image

from scripts.quality_decomposition_experiment import (
    DEFAULT_VARIANT_KEYS,
    ensure_color_outputs,
    resolve_variants,
)


def test_default_variants_remain_alignment_only():
    assert DEFAULT_VARIANT_KEYS == ("alignment_color", "blend_alignment_color")


def test_no_color_variant_can_be_selected_alone():
    variants = resolve_variants(["ddim_shift_no_color"])
    assert [variant["key"] for variant in variants] == ["ddim_shift_no_color"]
    assert variants[0]["color_transfer"] == "none"


def test_duplicate_or_unknown_variants_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        resolve_variants(["ddim_shift_no_color", "ddim_shift_no_color"])
    with pytest.raises(ValueError, match="unknown"):
        resolve_variants(["stale_legacy_variant"])


def test_no_color_reuses_verified_pre_color_outputs(tmp_path: Path):
    wm_pre = tmp_path / "wm_pre.png"
    clean_pre = tmp_path / "clean_pre.png"
    reference = tmp_path / "reference.png"
    for path, color in (
        (wm_pre, (1, 2, 3)),
        (clean_pre, (4, 5, 6)),
        (reference, (7, 8, 9)),
    ):
        Image.new("RGB", (512, 512), color).save(path)

    wm_output, clean_output = ensure_color_outputs(
        output_dir=tmp_path / "eval",
        variant_key="ddim_shift_no_color",
        color_mode="none",
        alpha=0.0,
        run_id="0",
        wm_pre_path=wm_pre,
        clean_pre_path=clean_pre,
        watermarked_path=reference,
        clean_path=reference,
        dx=24,
        dy=-24,
    )

    assert wm_output == wm_pre
    assert clean_output == clean_pre
    assert not (tmp_path / "eval" / "variant_outputs").exists()
