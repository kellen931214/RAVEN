import pytest

from raven.eval_protocol import sha256_path, stage_fid_records


def test_fid_staging_is_exact_fresh_and_config_scoped(tmp_path):
    records = []
    for run_id in (1, 2):
        reference = tmp_path / f"reference-{run_id}.png"
        attacked = tmp_path / f"attacked-{run_id}.png"
        reference.write_bytes(f"reference-{run_id}".encode())
        attacked.write_bytes(f"attacked-{run_id}".encode())
        records.append({
            "run_id": str(run_id), "watermarked_path": str(reference),
            "watermarked_sha256": sha256_path(reference), "attacked_path": str(attacked),
            "attacked_sha256": sha256_path(attacked),
        })
    root, manifest = stage_fid_records(
        records, formal_output=tmp_path / "formal", quality_config_hash="config-a", expected_count=2
    )
    assert {path.name for path in (root / "reference_watermarked").iterdir()} == {"000001.png", "000002.png"}
    assert {path.name for path in (root / "attacked").iterdir()} == {"000001.png", "000002.png"}
    assert manifest["image_count"] == 2
    assert (root / "fid_manifest.sha256").read_text().split()[0] == manifest["manifest_file_sha256"]
    with pytest.raises(FileExistsError):
        stage_fid_records(records, formal_output=tmp_path / "formal", quality_config_hash="config-a", expected_count=2)
    other, _ = stage_fid_records(
        records, formal_output=tmp_path / "formal", quality_config_hash="config-b", expected_count=2
    )
    assert other != root


def test_fid_staging_rejects_source_sha_drift(tmp_path):
    source = tmp_path / "one.png"
    source.write_bytes(b"one")
    record = {"run_id": "1", "watermarked_path": str(source), "watermarked_sha256": "bad", "attacked_path": str(source), "attacked_sha256": sha256_path(source)}
    with pytest.raises(ValueError, match="SHA mismatch"):
        stage_fid_records([record], formal_output=tmp_path / "formal", quality_config_hash="x", expected_count=1)
