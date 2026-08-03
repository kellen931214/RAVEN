"""Canonical metadata resolution for detector evaluation.

The original metadata CSV is the single source of truth for detector state.
Attack records carry only an identifier (``run_id``) plus attack/runtime facts.
The resolver joins records to metadata by ``(run_id, role)`` when role-specific
rows exist, otherwise by ``run_id`` alone.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetadataResolverError(Exception):
    """Base for metadata resolution errors."""


class DuplicateMetadataError(MetadataResolverError):
    """Multiple metadata rows match the same (run_id, role)."""


class AmbiguousMetadataError(MetadataResolverError):
    """Metadata row matches by run_id but role is ambiguous."""


class MetadataConflictError(MetadataResolverError):
    """CSV metadata disagrees with embedded source_metadata fallback."""


def load_metadata_csv(path: str | Path) -> list[dict[str, str]]:
    """Load and return raw metadata rows from a CSV file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"No rows in metadata CSV: {path}")
    return rows


def _normalize_run_id(row: dict[str, str]) -> str:
    for col in ("run_id", "sample_id", "id"):
        val = row.get(col)
        if val is not None and str(val).strip():
            return str(val).strip()
    raise MetadataResolverError("Metadata row has no run_id/sample_id/id")


def _normalize_role(row: dict[str, str]) -> str | None:
    """Return 'watermarked' or 'clean' if the row has a detectable role."""
    has_wm = bool(row.get("watermarked_path") or row.get("watermarked_image_path"))
    has_cl = bool(row.get("clean_path") or row.get("clean_image_path"))
    if has_wm and not has_cl:
        return "watermarked"
    if has_cl and not has_wm:
        return "clean"
    return None  # ambiguous or unknown


class MetadataResolver:
    """Resolves per-sample metadata from the canonical CSV.

    Joins attack records by ``(run_id, role)`` with role-specific CSV rows,
    falling back to ``run_id``-only join when no role column exists.
    """

    def __init__(self, csv_rows: list[dict[str, str]]):
        self._by_runid_role: dict[tuple[str, str], dict[str, str]] = {}
        self._by_runid: dict[str, dict[str, str]] = {}
        self._has_role_index = False

        for row in csv_rows:
            run_id = _normalize_run_id(row)
            role = _normalize_role(row)
            if role is not None:
                self._has_role_index = True
                key = (run_id, role)
                if key in self._by_runid_role:
                    raise DuplicateMetadataError(
                        f"Duplicate metadata row for (run_id={run_id!r}, role={role!r})"
                    )
                self._by_runid_role[key] = row
            else:
                if run_id in self._by_runid:
                    raise DuplicateMetadataError(
                        f"Duplicate metadata row for run_id={run_id!r}"
                    )
                self._by_runid[run_id] = row

    @property
    def has_role_index(self) -> bool:
        return self._has_role_index

    def resolve(self, run_id: str, role: str) -> dict[str, str]:
        """Return the metadata row for a given (run_id, role).

        Prefer role-indexed lookup when available; fall back to run_id-only.
        """
        if self._has_role_index:
            key = (str(run_id), str(role))
            if key in self._by_runid_role:
                return dict(self._by_runid_role[key])
            raise MetadataResolverError(
                f"No metadata row for (run_id={run_id!r}, role={role!r})"
            )
        if str(run_id) in self._by_runid:
            return dict(self._by_runid[str(run_id)])
        raise MetadataResolverError(
            f"No metadata row for run_id={run_id!r}"
        )

    def enrich_record(
        self, record: dict[str, Any], *, csv_path: str | None = None,
    ) -> dict[str, Any]:
        """Return a copy of *record* with resolved metadata merged in.

        Backwards compatibility: if *record* contains ``source_metadata``
        and the CSV row is also available, the two are compared and conflicts
        raise ``MetadataConflictError``.  If the CSV is unavailable, the
        embedded ``source_metadata`` is used as a fallback.
        """
        enriched = dict(record)
        run_id = str(record["run_id"])
        role = str(record.get("role", "watermarked"))
        embedded = record.get("source_metadata")

        try:
            csv_row = self.resolve(run_id, role)
        except MetadataResolverError:
            csv_row = None

        if csv_row is not None and isinstance(embedded, dict) and embedded:
            # Both sources available — validate consistency on shared fields
            conflicts = []
            shared_keys = set(csv_row) & set(embedded)
            for key in sorted(shared_keys):
                csv_val = str(csv_row.get(key, ""))
                emb_val = str(embedded.get(key, ""))
                if csv_val != emb_val:
                    conflicts.append(key)
            if conflicts:
                raise MetadataConflictError(
                    f"run_id={run_id!r} role={role!r}: CSV metadata conflicts "
                    f"with embedded source_metadata on fields: {conflicts}"
                )

        if csv_row is not None:
            # Primary path: resolved CSV metadata
            enriched["_metadata"] = dict(csv_row)
        elif isinstance(embedded, dict) and embedded:
            # Fallback: embedded source_metadata from legacy record
            enriched["_metadata"] = dict(embedded)
        else:
            raise MetadataResolverError(
                f"No metadata available for run_id={run_id!r} role={role!r}. "
                f"CSV: {csv_path or 'not specified'}"
            )

        # Also merge top-level convenience aliases for common fields
        # so detector adapters still find them without code changes
        meta = enriched["_metadata"]
        for field in meta:
            if field not in enriched or enriched.get(field) in (None, ""):
                enriched[field] = meta[field]

        return enriched

    @classmethod
    def from_path(cls, csv_path: str | Path) -> MetadataResolver:
        return cls(load_metadata_csv(csv_path))

    @classmethod
    def from_records_fallback(
        cls, records: list[dict[str, Any]],
    ) -> MetadataResolver | None:
        """Build a resolver from embedded ``source_metadata`` in records.

        Returns ``None`` when no records carry embedded metadata.
        Inherits ``run_id`` from the record when missing from the embedded dict.
        """
        rows: list[dict[str, str]] = []
        for rec in records:
            embedded = rec.get("source_metadata")
            if isinstance(embedded, dict) and embedded:
                row = {str(k): str(v) for k, v in embedded.items()}
                if "run_id" not in row and "sample_id" not in row and "id" not in row:
                    row["run_id"] = str(rec.get("run_id", ""))
                # Preserve role hint from record
                if "watermarked_path" not in row and "clean_path" not in row:
                    role = rec.get("role", "watermarked")
                    if role == "watermarked":
                        row["watermarked_path"] = rec.get("input_path", "")
                    elif role == "clean":
                        row["clean_path"] = rec.get("input_path", "")
                rows.append(row)
        if not rows:
            return None
        return cls(rows)
