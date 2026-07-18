import csv
import hashlib
from pathlib import Path

from PIL import Image

from raven.pairing_provenance import PAIRING_PROTOCOL, build_pairing_sha256, sha256_path
from scripts.paired_generation_shards import (
    canonical_shard_fieldnames,
    merge,
    prepare,
    read_csv,
    write_csv_atomic,
)


def make_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), rgb).save(path)


def paired_row(root: Path, run_id: int, *, shard_index: int | None = None) -> dict:
    clean = root / "data" / "generated" / "diffusiondb" / f"{run_id:06d}.png"
    watermarked = (
        root
        / "data"
        / "watermarked"
        / "diffusiondb"
        / "TR"
        / f"{run_id:06d}"
        / "watermarked.png"
    )
    make_image(clean, (run_id + 1, 2, 3))
    make_image(watermarked, (run_id + 1, 2, 4))
    prompt = f"prompt {run_id}"
    row = {
        "protocol": PAIRING_PROTOCOL,
        "dataset": "diffusiondb",
        "run_id": str(run_id),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "base_latent_seed": str(42 + run_id),
        "base_latent_sha256": f"base-{run_id}",
        "clean_base_latent_sha256": f"base-{run_id}",
        "watermarked_base_latent_sha256": f"base-{run_id}",
        "watermarked_latent_sha256": f"wm-latent-{run_id}",
        "watermark_target_sha256": "target",
        "watermark_mask_sha256": "mask",
        "generation_config_sha256": "generation",
        "watermark_config_sha256": "watermark",
        "clean_path": str(clean.resolve()),
        "clean_sha256": sha256_path(clean),
        "watermarked_path": str(watermarked.resolve()),
        "watermarked_sha256": sha256_path(watermarked),
        "model_id": "model",
        "model_revision": "revision",
    }
    if shard_index is not None:
        row["num_shards"] = "2"
        row["shard_index"] = str(shard_index)
    row["pairing_sha256"] = build_pairing_sha256(row)
    return row


def write_prompts(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt"])
        writer.writeheader()
        for run_id in range(count):
            writer.writerow({"prompt": f"prompt {run_id}"})


def test_prepare_quarantines_orphan_and_merge_audits_shards(tmp_path: Path) -> None:
    root = tmp_path / "run"
    prompts = root / "inputs" / "prompts.csv"
    write_prompts(prompts, 2)
    method_dir = root / "data" / "watermarked" / "diffusiondb" / "TR"

    committed = paired_row(root, 0)
    write_csv_atomic(method_dir / "metadata.csv", [committed])
    orphan = paired_row(root, 1)
    orphan_clean_sha = orphan["clean_sha256"]
    orphan_wm_sha = orphan["watermarked_sha256"]

    result = prepare(root, prompts, expected_count=2, num_shards=2)
    assert result["recorded_count"] == 1
    assert result["quarantine"]["status"] == "ORPHANED_UNRECORDED_QUARANTINED"
    assert not Path(orphan["clean_path"]).exists()
    assert not Path(orphan["watermarked_path"]).exists()
    event = result["quarantine"]["events"][0]
    assert event["clean_sha256"] == orphan_clean_sha
    assert event["watermarked_sha256"] == orphan_wm_sha

    shard0 = method_dir / "metadata.shard-000-of-002.csv"
    shard1 = method_dir / "metadata.shard-001-of-002.csv"
    assert [row["run_id"] for row in read_csv(shard0)] == ["0"]
    assert not shard1.exists()

    regenerated = paired_row(root, 1, shard_index=1)
    write_csv_atomic(shard1, [regenerated])
    merged = merge(root, prompts, expected_count=2, num_shards=2)
    assert merged["passed"] is True
    assert merged["pairing_audit"]["unique_base_latent_hashes"] == 2
    assert [row["run_id"] for row in read_csv(method_dir / "metadata.csv")] == ["0", "1"]


def test_prepare_recovers_mixed_old_and_canonical_shard_field_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    prompts = root / "inputs" / "prompts.csv"
    write_prompts(prompts, 3)
    method_dir = root / "data" / "watermarked" / "diffusiondb" / "TR"
    method_dir.mkdir(parents=True)
    old_order_row = paired_row(root, 0, shard_index=0)
    canonical_order_row = paired_row(root, 2, shard_index=0)
    stored_fields = list(old_order_row)
    canonical_fields = canonical_shard_fieldnames(stored_fields)
    path = method_dir / "metadata.shard-000-of-002.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(stored_fields)
        writer.writerow([old_order_row[field] for field in stored_fields])
        writer.writerow([canonical_order_row[field] for field in canonical_fields])

    result = prepare(root, prompts, expected_count=3, num_shards=2)

    assert result["recorded_count"] == 2
    assert result["schema_repairs"] == [
        {
            "source": str(path.resolve()),
            "rows_reinterpreted_with_canonical_schema": 1,
            "reason": "stored header order differed from canonical shard row order",
        }
    ]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == canonical_fields
        rows = list(reader)
    assert [row["run_id"] for row in rows] == ["0", "2"]
    assert [row["base_latent_seed"] for row in rows] == ["42", "44"]
