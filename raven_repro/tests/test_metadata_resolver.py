"""Tests for metadata_resolver — issue #18."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "raven_repro"))

from raven.metadata_resolver import (  # noqa: E402
    AmbiguousMetadataError,
    DuplicateMetadataError,
    MetadataConflictError,
    MetadataResolver,
    MetadataResolverError,
    load_metadata_csv,
)


# ===========================================================================
# Helpers
# ===========================================================================
def _write_csv(path: Path, rows: list[dict[str, str]]):
    if not rows:
        return
    writer = csv.DictWriter(path.open("w", newline=""), fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


# ===========================================================================
# Basic CSV loading
# ===========================================================================
class TestLoadMetadataCSV:
    def test_loads_valid_csv(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/a.png",
                         "clean_path": "/b.png", "prompt": "cat"}])
        rows = load_metadata_csv(p)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "1"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_metadata_csv("/nonexistent/path.csv")

    def test_empty_csv_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("run_id,watermarked_path\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No rows"):
            load_metadata_csv(p)


# ===========================================================================
# Single-row run_id join (acceptance criterion 1)
# ===========================================================================
class TestSingleRowJoin:
    def test_basic_run_id_lookup(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "42", "watermarked_path": "/wm.png",
                         "prompt": "a dog", "w_seed": "12345"}])
        resolver = MetadataResolver.from_path(p)
        row = resolver.resolve("42", "watermarked")
        assert row["run_id"] == "42"
        assert row["w_seed"] == "12345"
        assert row["prompt"] == "a dog"

    def test_run_id_only_fallback_when_no_role(self, tmp_path):
        """When CSV has no role column, resolve works for any role."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "clean_path": "/cl.png", "gm_bundle_dir": "/bundle"}])
        resolver = MetadataResolver.from_path(p)
        # Both roles resolve to same row
        wm = resolver.resolve("1", "watermarked")
        cl = resolver.resolve("1", "clean")
        assert wm["gm_bundle_dir"] == "/bundle"
        assert cl["gm_bundle_dir"] == "/bundle"

    def test_missing_run_id_raises(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png"}])
        resolver = MetadataResolver.from_path(p)
        with pytest.raises(MetadataResolverError):
            resolver.resolve("99", "watermarked")


# ===========================================================================
# (run_id, role) join (acceptance criterion 2)
# ===========================================================================
class TestRunIdRoleJoin:
    def test_role_specific_lookup(self, tmp_path):
        """When CSV has separate watermarked/clean rows, each resolves correctly."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [
            {"run_id": "1", "watermarked_path": "/wm1.png", "clean_path": "",
             "w_seed": "100"},
            {"run_id": "1", "watermarked_path": "", "clean_path": "/cl1.png",
             "w_seed": ""},
        ])
        resolver = MetadataResolver.from_path(p)
        wm = resolver.resolve("1", "watermarked")
        cl = resolver.resolve("1", "clean")
        assert wm["w_seed"] == "100"
        assert "w_seed" not in cl or not cl["w_seed"]

    def test_wm_and_cl_not_mismatched(self):
        """Watermarked and clean records with same run_id cannot be mismatched."""
        resolver = MetadataResolver([
            {"run_id": "1", "watermarked_path": "/wm.png", "t2s_state_path": "/state_wm.pt"},
            {"run_id": "1", "clean_path": "/cl.png", "t2s_state_path": "/state_cl.pt"},
        ])
        wm = resolver.resolve("1", "watermarked")
        cl = resolver.resolve("1", "clean")
        assert wm["t2s_state_path"] == "/state_wm.pt"
        assert cl["t2s_state_path"] == "/state_cl.pt"
        assert wm["t2s_state_path"] != cl["t2s_state_path"]


# ===========================================================================
# Duplicate/ambiguous rows (acceptance criterion 3)
# ===========================================================================
class TestDuplicateAmbiguous:
    def test_duplicate_run_id_raises(self):
        with pytest.raises(DuplicateMetadataError):
            MetadataResolver([
                {"run_id": "1", "watermarked_path": "/a.png"},
                {"run_id": "1", "watermarked_path": "/b.png"},
            ])

    def test_duplicate_runid_role_raises(self):
        with pytest.raises(DuplicateMetadataError):
            MetadataResolver([
                {"run_id": "1", "watermarked_path": "/a.png"},
                {"run_id": "1", "watermarked_path": "/b.png"},
            ])

    def test_conflicting_csv_vs_embedded_raises(self, tmp_path):
        """CSV vs embedded source_metadata conflict raises MetadataConflictError."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "999"}])
        resolver = MetadataResolver.from_path(p)
        record = {
            "run_id": "1", "role": "watermarked",
            "source_metadata": {"run_id": "1", "w_seed": "111"},
        }
        with pytest.raises(MetadataConflictError):
            resolver.enrich_record(record, csv_path=str(p))


# ===========================================================================
# New record schema excludes full source_metadata (acceptance criterion 4)
# ===========================================================================
class TestNewRecordSchema:
    def test_new_record_no_source_metadata(self):
        """New attack records do NOT contain source_metadata."""
        rec = {
            "run_id": "17", "role": "watermarked", "method": "TR",
            "attack_seed": 59, "metadata_path": "/tmp/metadata.csv",
            "input_path": "/tmp/in.png", "output_path": "/tmp/out.png",
        }
        assert "source_metadata" not in rec
        assert rec["metadata_path"] == "/tmp/metadata.csv"

    def test_metadata_path_in_record(self):
        """New records carry metadata_path for resolver."""
        rec = {
            "run_id": "1", "role": "watermarked",
            "metadata_path": "/data/metadata.csv",
        }
        assert "metadata_path" in rec


# ===========================================================================
# Backwards compatibility (acceptance criterion 5)
# ===========================================================================
class TestBackwardsCompat:
    def test_embedded_source_metadata_fallback(self, tmp_path):
        """When CSV unavailable, embedded source_metadata is used."""
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1", "w_seed": "42",
                                  "w_channel": "3", "w_radius": "10",
                                  "w_pattern": "ring", "w_mask_shape": "circle",
                                  "w_measurement": "l1_complex",
                                  "w_injection": "complex"}},
        ])
        assert resolver is not None
        row = resolver.resolve("1", "watermarked")
        assert row["w_seed"] == "42"

    def test_csv_and_embedded_consistent_is_ok(self, tmp_path):
        """CSV and embedded agree → no error, CSV wins."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "42"}])
        resolver = MetadataResolver.from_path(p)
        record = {
            "run_id": "1", "role": "watermarked",
            "source_metadata": {"run_id": "1", "w_seed": "42"},
        }
        enriched = resolver.enrich_record(record, csv_path=str(p))
        assert enriched["_metadata"]["w_seed"] == "42"

    def test_no_embedded_fallback_when_no_source_metadata(self):
        """Records without source_metadata produce None resolver."""
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked"},
        ])
        assert resolver is None

    def test_enrich_merges_metadata_to_top_level(self, tmp_path):
        """Enriched record has metadata fields at top level for detector access."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "gm_bundle_dir": "/bundle", "w_seed": "99"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked"}, csv_path=str(p))
        assert enriched["gm_bundle_dir"] == "/bundle"
        assert enriched["w_seed"] == "99"
        assert enriched["_metadata"]["gm_bundle_dir"] == "/bundle"


# ===========================================================================
# Synthetic CSV with all method fields (acceptance criterion 1 extended)
# ===========================================================================
class TestAllMethodFields:
    def test_full_csv_resolves_all_methods(self, tmp_path):
        """Synthetic CSV with GM/T2S/GS/TR/Fourier fields resolves correctly."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{
            "run_id": "1",
            "watermarked_path": "/wm.png",
            "clean_path": "/cl.png",
            "prompt": "test",
            # TR
            "w_seed": "12345", "w_channel": "3", "w_radius": "10",
            "w_pattern": "ring", "w_mask_shape": "circle",
            "w_measurement": "l1_complex", "w_injection": "complex",
            # GM
            "gm_bundle_dir": "/data/gm_bundle",
            "gm_bundle_config_sha256": "abc123",
            "gm_w1_file_sha256": "def456",
            "gm_w2_file_sha256": "ghi789",
            "gm_protocol_mode": "official",
            # T2S
            "t2s_state_path": "/data/t2s.pt",
            "t2s_state_sha256": "t2s_sha",
            "t2s_provider_config_sha256": "t2s_cfg_sha",
            "t2s_protocol_mode": "t2s_official",
            # GS
            "gs_secret_index": "5",
            "gs_secret_bundle_sha256": "gs_bundle_sha",
            "gs_protocol_mode": "official_compatible",
            # Fourier RID
            "rid_bundle_dir": "/data/rid_bundle",
            "rid_bundle_config_sha256": "rid_cfg",
            "rid_selected_pattern_sha256": "rid_pat",
            "rid_mask_sha256": "rid_mask",
            "rid_key_index": "0",
        }])
        resolver = MetadataResolver.from_path(p)
        row = resolver.resolve("1", "watermarked")
        assert row["w_seed"] == "12345"
        assert row["gm_bundle_dir"] == "/data/gm_bundle"
        assert row["t2s_state_path"] == "/data/t2s.pt"
        assert row["gs_secret_index"] == "5"
        assert row["rid_bundle_dir"] == "/data/rid_bundle"


# ===========================================================================
# Detector adapter receives resolved metadata (acceptance criterion 6)
# ===========================================================================
class TestDetectorReceivesMetadata:
    def test_enriched_record_has_metadata_key(self, tmp_path):
        """Enriched record has _metadata dict for detector access."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "42", "w_channel": "3"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked"}, csv_path=str(p))
        assert "_metadata" in enriched
        assert enriched["_metadata"]["w_seed"] == "42"

    def test_top_level_aliases_for_detector_compat(self, tmp_path):
        """Metadata fields are also accessible at top level for detector compat."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "42"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked"}, csv_path=str(p))
        # Detector can read from top level (backwards compat) OR _metadata
        assert enriched["w_seed"] == "42"
        assert enriched["_metadata"]["w_seed"] == "42"

    def test_no_overwrite_existing_top_level(self, tmp_path):
        """Metadata does NOT overwrite existing top-level attack fields."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "prompt": "csv_prompt"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked", "prompt": "record_prompt"},
            csv_path=str(p))
        # Record's own prompt takes precedence
        assert enriched["prompt"] == "record_prompt"
        # CSV prompt still accessible via _metadata
        assert enriched["_metadata"]["prompt"] == "csv_prompt"


# ===========================================================================
# Main.py no longer writes source_metadata (acceptance criterion 4)
# ===========================================================================
class TestMainSourceMetadataRemoved:
    def test_main_no_source_metadata_in_record(self):
        """main.py does not write source_metadata to record."""
        source = (REPO / "experiments" / "main.py").read_text()
        # The word "source_metadata" should not appear in record construction
        record_section = source.split("record = {")[1].split("}")[0] if "record = {" in source else ""
        assert '"source_metadata"' not in record_section, (
            "main.py must not write source_metadata to records")

    def test_main_has_metadata_path_in_record(self):
        """main.py writes metadata_path to record."""
        source = (REPO / "experiments" / "main.py").read_text()
        assert '"metadata_path"' in source


# ===========================================================================
# Eval.py uses resolver (acceptance criterion 6)
# ===========================================================================
class TestEvalUsesResolver:
    def test_eval_imports_resolver(self):
        """eval.py imports MetadataResolver."""
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "MetadataResolver" in source
        assert "metadata_resolver" in source

    def test_eval_enriches_records(self):
        """eval.py calls resolver.enrich_record."""
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "enrich_record" in source

    def test_eval_has_fallback_path(self):
        """eval.py has from_records_fallback for backwards compat."""
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "from_records_fallback" in source
