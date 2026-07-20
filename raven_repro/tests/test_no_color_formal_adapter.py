import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from raven.eval_protocol import canonical_json_hash, sha256_path
from experiments.run_raven_no_color_ablation_eval import (
    build_variant_records,
    configure_single_gpu,
)


def _image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (512, 512), color).save(path)
    return sha256_path(path)


def test_no_color_adapter_uses_pre_color_output_and_distinct_variant_hash(tmp_path):
    root = tmp_path / "formal"
    snapshot = root / "snapshots" / "batch.csv"
    snapshot.parent.mkdir(parents=True)
    clean = _image(tmp_path / "clean.png", (1, 2, 3))
    watermarked = _image(tmp_path / "watermarked.png", (4, 5, 6))
    snapshot.write_text(
        "run_id,clean_path,clean_sha256,watermarked_path,watermarked_sha256,prompt\n"
        f"0,{tmp_path / 'clean.png'},{clean},{tmp_path / 'watermarked.png'},{watermarked},prompt\n"
    )
    (root / "snapshots" / "snapshot_index.jsonl").write_text(
        '{"snapshot_path":"' + str(snapshot) + '","snapshot_sha256":"' + sha256_path(snapshot) + '"}\n'
    )
    config_hash = "base-config"
    (root / "run_config.json").write_text(
        '{"attack_config_hash":"base-config","source_code_manifest_sha256":"source"}'
    )
    for role in ("watermarked", "clean"):
        output = root / "attack_cache" / config_hash / "0" / role / "output"
        output.mkdir(parents=True)
        pre = _image(output / "view_guided_output.png", (7, 8, 9))
        final = _image(output / "final_color_corrected.png", (10, 11, 12))
        debug = output / "debug_info.json"
        debug.write_text("{}")
        record = {
            "run_id": "0", "attacked_path": str(output / "final_color_corrected.png"),
            "attacked_sha256": final, "debug_info_path": str(debug), "input_role": role,
            "attack_config_hash": config_hash, "formal_config_hash": config_hash,
            "watermarked_path": str(tmp_path / "watermarked.png"), "watermarked_sha256": watermarked,
            "clean_path": str(tmp_path / "clean.png"), "clean_sha256": clean,
        }
        (output.parent / "record.json").write_text(__import__("json").dumps(record))
    wm, clean_rows, variant_hash = build_variant_records(root, 1)
    assert wm[0]["attacked_path"].endswith("view_guided_output.png")
    assert clean_rows[0]["attacked_path"].endswith("view_guided_output.png")
    assert wm[0]["attacked_sha256"] != wm[0]["source_final_output_sha256"]
    assert variant_hash != config_hash
    assert wm[0]["output_color_transfer"] is False


def test_no_color_adapter_isolates_the_requested_physical_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    configure_single_gpu(8)
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "8"
