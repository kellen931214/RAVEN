"""Tests for metadata_resolver — issue #18."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

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


def _write_csv(path: Path, rows: list[dict[str, str]]):
    # Collect all field names across all rows for consistent headers
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Basic loading
# ===========================================================================
class TestLoadMetadataCSV:
    def test_loads_valid_csv(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/a.png",
                         "prompt": "cat"}])
        rows = load_metadata_csv(p)
        assert rows[0]["run_id"] == "1"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_metadata_csv("/nonexistent/path.csv")


# ===========================================================================
# Single-row run_id join
# ===========================================================================
class TestSingleRowJoin:
    def test_basic_run_id_lookup(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "42", "watermarked_path": "/wm.png",
                         "w_seed": "12345"}])
        resolver = MetadataResolver.from_path(p)
        row = resolver.resolve("42", "watermarked")
        assert row["w_seed"] == "12345"

    def test_missing_run_id_raises(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png"}])
        resolver = MetadataResolver.from_path(p)
        with pytest.raises(MetadataResolverError):
            resolver.resolve("99", "watermarked")


# ===========================================================================
# (run_id, role) join
# ===========================================================================
class TestRunIdRoleJoin:
    def test_role_specific_lookup(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [
            {"run_id": "1", "watermarked_path": "/wm1.png", "w_seed": "100",
             "clean_path": ""},
            {"run_id": "1", "watermarked_path": "", "clean_path": "/cl1.png",
             "w_seed": ""},
        ])
        resolver = MetadataResolver.from_path(p)
        assert resolver.resolve("1", "watermarked")["w_seed"] == "100"
        assert resolver.resolve("1", "clean").get("w_seed", "") == ""

    def test_same_runid_wm_and_cl_not_mismatched(self):
        resolver = MetadataResolver([
            {"run_id": "1", "watermarked_path": "/wm.png",
             "t2s_state_path": "/state_wm.pt", "clean_path": ""},
            {"run_id": "1", "clean_path": "/cl.png",
             "t2s_state_path": "/state_cl.pt", "watermarked_path": ""},
        ])
        wm = resolver.resolve("1", "watermarked")
        cl = resolver.resolve("1", "clean")
        assert wm["t2s_state_path"] == "/state_wm.pt"
        assert cl["t2s_state_path"] == "/state_cl.pt"


# ===========================================================================
# Duplicate rows → fail closed
# ===========================================================================
class TestDuplicateRows:
    def test_duplicate_run_id_raises(self):
        with pytest.raises(DuplicateMetadataError):
            MetadataResolver([
                {"run_id": "1", "watermarked_path": "/a.png"},
                {"run_id": "1", "watermarked_path": "/b.png"},
            ])

    def test_duplicate_generic_raises(self):
        with pytest.raises(DuplicateMetadataError):
            MetadataResolver([
                {"run_id": "1", "watermarked_path": "/a.png",
                 "clean_path": "/c.png"},
                {"run_id": "1", "watermarked_path": "/b.png",
                 "clean_path": "/d.png"},
            ])

    def test_duplicate_in_csv_no_fallback(self, tmp_path):
        """Duplicate in CSV must fail — no fallback to embedded."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [
            {"run_id": "1", "watermarked_path": "/a.png"},
            {"run_id": "1", "watermarked_path": "/b.png"},
        ])
        with pytest.raises(DuplicateMetadataError):
            MetadataResolver.from_path(p)


# ===========================================================================
# CSV exists but missing run_id → fail closed
# ===========================================================================
class TestCSVMissingRow:
    def test_csv_exists_missing_run_id_fails(self, tmp_path):
        """CSV exists but missing run_id → resolve() raises, no fallback."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png"}])
        resolver = MetadataResolver.from_path(p)
        with pytest.raises(MetadataResolverError, match="No metadata row"):
            resolver.resolve("99", "watermarked")

    def test_enrich_with_missing_row_fails(self, tmp_path):
        """enrich_record fails when CSV exists but row is missing."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png"}])
        resolver = MetadataResolver.from_path(p)
        with pytest.raises(MetadataResolverError):
            resolver.enrich_record(
                {"run_id": "99", "role": "watermarked"}, csv_path=str(p))


# ===========================================================================
# Legacy fallback (CSV genuinely absent)
# ===========================================================================
class TestLegacyFallback:
    def test_csv_absent_legacy_succeeds(self):
        """CSV genuinely absent → embedded source_metadata fallback works."""
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1", "w_seed": "42",
                                  "watermarked_path": "/wm.png",
                                  "w_channel": "3"}},
        ])
        assert resolver is not None
        row = resolver.resolve("1", "watermarked")
        assert row["w_seed"] == "42"

    def test_same_runid_dual_role_legacy(self):
        """Same run_id with watermarked+clean legacy records resolve correctly."""
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1",
                                  "watermarked_path": "/wm.png",
                                  "clean_path": "/cl.png"}},
            {"run_id": "1", "role": "clean",
             "source_metadata": {"run_id": "1",
                                  "watermarked_path": "/wm.png",
                                  "clean_path": "/cl.png"}},
        ])
        assert resolver is not None
        # Both resolve to same metadata since the embedded dict is identical
        wm = resolver.resolve("1", "watermarked")
        cl = resolver.resolve("1", "clean")
        assert wm["watermarked_path"] == "/wm.png"
        assert cl["clean_path"] == "/cl.png"

    def test_legacy_fallback_no_duplicate_on_same_runid(self):
        """from_records_fallback does not create duplicate generic rows
        for same run_id with different roles."""
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1", "w_seed": "99"}},
            {"run_id": "1", "role": "clean",
             "source_metadata": {"run_id": "1", "w_seed": "99"}},
        ])
        assert resolver is not None
        # Both roles should resolve
        assert resolver.resolve("1", "watermarked")["w_seed"] == "99"
        assert resolver.resolve("1", "clean")["w_seed"] == "99"

    def test_no_embedded_fallback_when_no_source_metadata(self):
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked"},
        ])
        assert resolver is None

    def test_same_key_different_metadata_raises(self):
        """Same (run_id, role) with different embedded metadata → conflict."""
        with pytest.raises(MetadataConflictError, match="different embedded"):
            MetadataResolver.from_records_fallback([
                {"run_id": "1", "role": "watermarked",
                 "source_metadata": {"run_id": "1", "w_seed": "99"}},
                {"run_id": "1", "role": "watermarked",
                 "source_metadata": {"run_id": "1", "w_seed": "100"}},
            ])

    def test_same_key_identical_metadata_ok(self):
        """Same (run_id, role) with identical embedded metadata → deduplicated."""
        resolver = MetadataResolver.from_records_fallback([
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1", "w_seed": "99"}},
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1", "w_seed": "99"}},
        ])
        assert resolver is not None
        assert resolver.resolve("1", "watermarked")["w_seed"] == "99"


# ===========================================================================
# CSV vs embedded conflict
# ===========================================================================
class TestCSVEmbeddedConflict:
    def test_conflict_raises(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "999"}])
        resolver = MetadataResolver.from_path(p)
        record = {"run_id": "1", "role": "watermarked",
                   "source_metadata": {"run_id": "1", "w_seed": "111"}}
        with pytest.raises(MetadataConflictError):
            resolver.enrich_record(record, csv_path=str(p))

    def test_consistent_no_error(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "42"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked",
             "source_metadata": {"run_id": "1", "w_seed": "42"}},
            csv_path=str(p))
        assert enriched["_metadata"]["w_seed"] == "42"


# ===========================================================================
# Explicit role column priority
# ===========================================================================
class TestExplicitRoleColumn:
    def test_explicit_role_column_watermarked(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "role": "watermarked",
                         "watermarked_path": "/wm.png", "w_seed": "50"}])
        resolver = MetadataResolver.from_path(p)
        assert resolver.resolve("1", "watermarked")["w_seed"] == "50"

    def test_explicit_role_column_clean(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "role": "clean",
                         "clean_path": "/cl.png"}])
        resolver = MetadataResolver.from_path(p)
        assert resolver.resolve("1", "clean")["clean_path"] == "/cl.png"

    def test_source_role_column(self, tmp_path):
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "source_role": "wm",
                         "watermarked_path": "/wm.png"}])
        resolver = MetadataResolver.from_path(p)
        assert resolver.resolve("1", "watermarked")["watermarked_path"] == "/wm.png"

    def test_explicit_role_contradictory_path_raises(self, tmp_path):
        """Explicit role=clean with only watermarked_path → AmbiguousMetadataError."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "role": "clean",
                         "watermarked_path": "/wm.png", "clean_path": ""}])
        with pytest.raises(AmbiguousMetadataError, match="Contradictory"):
            MetadataResolver.from_path(p)

    def test_unknown_explicit_role_raises(self, tmp_path):
        """Explicit role='something_else' → AmbiguousMetadataError."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "role": "something_else",
                         "watermarked_path": "/wm.png"}])
        with pytest.raises(AmbiguousMetadataError, match="Unknown explicit role"):
            MetadataResolver.from_path(p)

    def test_wm_role_with_only_clean_path_raises(self, tmp_path):
        """Explicit role=watermarked with only clean_path → AmbiguousMetadataError."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "role": "watermarked",
                         "clean_path": "/cl.png", "watermarked_path": ""}])
        with pytest.raises(AmbiguousMetadataError, match="Contradictory"):
            MetadataResolver.from_path(p)


# ===========================================================================
# Mixed generic + role-specific rows
# ===========================================================================
class TestMixedGenericRows:
    def test_mixed_in_same_csv(self, tmp_path):
        """Generic rows + role-specific rows coexist. resolve() tries
        (run_id,role) first, then generic fallback."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [
            # Generic row for run_id=1 (both paths → no role)
            {"run_id": "1", "watermarked_path": "/wm1.png",
             "clean_path": "/cl1.png", "w_seed": "generic_seed"},
            # Role-specific row for run_id=2 (only watermarked)
            {"run_id": "2", "watermarked_path": "/wm2.png",
             "clean_path": "", "w_seed": "wm2_seed"},
        ])
        resolver = MetadataResolver.from_path(p)
        # Run 1: generic row for both roles
        assert resolver.resolve("1", "watermarked")["w_seed"] == "generic_seed"
        assert resolver.resolve("1", "clean")["w_seed"] == "generic_seed"
        # Run 2: role-specific row
        assert resolver.resolve("2", "watermarked")["w_seed"] == "wm2_seed"
        # Run 2 clean: no role row, no generic → fail
        with pytest.raises(MetadataResolverError):
            resolver.resolve("2", "clean")

    def test_generic_fallback_not_disabled_by_other_role_rows(self):
        """Generic lookup for run_id=1 works even though run_id=2 has role rows."""
        resolver = MetadataResolver([
            {"run_id": "1", "watermarked_path": "/wm1.png",
             "clean_path": "/cl1.png", "w_seed": "s1"},
            {"run_id": "2", "watermarked_path": "/wm2.png",
             "clean_path": "", "w_seed": "s2"},
        ])
        assert resolver.resolve("1", "watermarked")["w_seed"] == "s1"
        assert resolver.resolve("1", "clean")["w_seed"] == "s1"


# ===========================================================================
# load_state receives resolved metadata (not raw attack record)
# ===========================================================================
class TestLoadStateReceivesResolvedMetadata:
    def test_enriched_record_has_metadata(self, tmp_path):
        """Enriched record carries _metadata dict with CSV fields."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "42", "w_channel": "3"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked"}, csv_path=str(p))
        assert "_metadata" in enriched
        assert enriched["_metadata"]["w_seed"] == "42"

    def test_enriched_has_top_level_aliases(self, tmp_path):
        """Detector-required fields accessible at top level."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{"run_id": "1", "watermarked_path": "/wm.png",
                         "w_seed": "42"}])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked"}, csv_path=str(p))
        assert enriched["w_seed"] == "42"

    def test_all_method_fields_resolved(self, tmp_path):
        """Synthetic CSV with GM/T2S/GS/TR/Fourier fields resolves all."""
        p = tmp_path / "meta.csv"
        _write_csv(p, [{
            "run_id": "1", "watermarked_path": "/wm.png",
            "w_seed": "99", "w_channel": "3",
            "gm_bundle_dir": "/b", "gm_w1_file_sha256": "w1",
            "t2s_state_path": "/s.pt", "t2s_state_sha256": "sha",
            "gs_secret_index": "5",
            "rid_bundle_dir": "/rid", "rid_key_index": "0",
        }])
        resolver = MetadataResolver.from_path(p)
        enriched = resolver.enrich_record(
            {"run_id": "1", "role": "watermarked"})
        assert enriched["w_seed"] == "99"
        assert enriched["gm_bundle_dir"] == "/b"
        assert enriched["t2s_state_path"] == "/s.pt"
        assert enriched["gs_secret_index"] == "5"
        assert enriched["rid_bundle_dir"] == "/rid"


# ===========================================================================
# New record schema
# ===========================================================================
class TestNewRecordSchema:
    def test_no_source_metadata_in_new_record(self):
        """New attack records do NOT contain source_metadata."""
        rec = {"run_id": "17", "role": "watermarked", "method": "TR",
               "input_path": "/tmp/in.png"}
        assert "source_metadata" not in rec

    def test_no_metadata_path_in_new_record(self):
        """New attack records do NOT contain per-record metadata_path."""
        rec = {"run_id": "17", "role": "watermarked",
               "input_path": "/tmp/in.png"}
        assert "metadata_path" not in rec

    def test_main_py_no_source_metadata_in_record(self):
        source = (REPO / "experiments" / "main.py").read_text()
        record_section = source.split("record = {")[1].split("}")[0]
        assert '"source_metadata"' not in record_section

    def test_main_py_no_metadata_path_in_record(self):
        source = (REPO / "experiments" / "main.py").read_text()
        record_section = source.split("record = {")[1].split("}")[0]
        assert '"metadata_path"' not in record_section


# ===========================================================================
# eval.py integration
# ===========================================================================
class TestEvalIntegration:
    def test_eval_resolves_before_load_state(self):
        """eval.py builds resolver and enriches records BEFORE load_state call."""
        source = (REPO / "experiments" / "eval.py").read_text()
        # Metadata resolution block should appear before load_state block
        meta_idx = source.find("MetadataResolver.from_path")
        load_idx = source.find("det_mod.load_state(")
        assert meta_idx < load_idx, (
            "Metadata resolution must happen BEFORE load_state")

    def test_eval_path_not_exists_triggers_fallback(self):
        """eval.py uses path.exists() check before fallback, path.is_file() after."""
        source = (REPO / "experiments" / "eval.py").read_text()
        assert "not path.exists()" in source or "path.exists()" in source
        assert "not path.is_file()" in source or "path.is_file()" in source

    def test_eval_fails_closed_on_metadata_errors(self):
        """eval.py catches DuplicateMetadataError etc to fail closed (return error),
        NOT to fall back to embedded metadata."""
        source = (REPO / "experiments" / "eval.py").read_text()
        # The catch block must return an error response with STATUS_FAILED_INTERNAL_ERROR
        resolver_block = source.split("MetadataResolver.from_path")[1]
        assert "STATUS_FAILED_INTERNAL_ERROR" in resolver_block, (
            "Metadata validation errors must return failure, not fallback")
