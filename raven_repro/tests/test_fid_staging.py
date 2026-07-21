import pytest

from raven.eval_protocol import (
    NO_COLOR_FID_ATTACKED_DEFINITION,
    bind_pre_color_attack_record,
    sha256_path,
    stage_fid_records,
    stage_no_color_fid_records,
)


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



def test_no_color_record_and_fid_definition_use_explicit_pre_color_provenance(tmp_path):
    reference = tmp_path / "wm.png"
    final = tmp_path / "final.png"
    pre_color = tmp_path / "view.png"
    reference.write_bytes(b"wm")
    final.write_bytes(b"final")
    pre_color.write_bytes(b"pre")
    base = {
        "run_id": "1",
        "watermarked_path": str(reference),
        "watermarked_sha256": sha256_path(reference),
        "attacked_path": str(final),
        "attacked_sha256": sha256_path(final),
        "pre_color_attacked_path": str(pre_color),
        "pre_color_attacked_sha256": sha256_path(pre_color),
    }
    no_color = bind_pre_color_attack_record(base)
    assert no_color["attacked_path"] == str(pre_color.resolve())
    _, manifest = stage_no_color_fid_records(
        [base],
        formal_output=tmp_path / "formal",
        quality_config_hash="no-color",
        expected_count=1,
    )
    assert manifest["attacked_definition"] == NO_COLOR_FID_ATTACKED_DEFINITION
    assert manifest["records"][0]["attacked_source_path"] == str(pre_color.resolve())
    pre_color.write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="pre-color attacked SHA mismatch"):
        bind_pre_color_attack_record(base)
