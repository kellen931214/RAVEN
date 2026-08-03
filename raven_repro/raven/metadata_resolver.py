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
    """Return 'watermarked' or 'clean' from explicit column or path inference.

    Priority: explicit ``role`` / ``source_role`` / ``image_role`` column,
    then path inference from ``watermarked_path`` / ``clean_path``.
    Returns ``None`` for generic rows (both paths present or no path).
    """
    # Explicit role columns take priority
    for col in ("role", "source_role", "image_role"):
        val = row.get(col)
        if val is not None and str(val).strip():
            v = str(val).strip().lower()
            if v in ("watermarked", "wm"):
                return "watermarked"
            if v in ("clean", "cl"):
                return "clean"
            # Unknown explicit value → treat as generic
            return None

    # Path-based inference
    has_wm = bool(row.get("watermarked_path") or row.get("watermarked_image_path"))
    has_cl = bool(row.get("clean_path") or row.get("clean_image_path"))
    if has_wm and not has_cl:
        return "watermarked"
    if has_cl and not has_wm:
        return "clean"
    return None  # both or neither → generic


class MetadataResolver:
    """Resolves per-sample metadata from the canonical CSV.

    Supports mixed generic rows (no role) and role-specific rows in the same
    CSV.  ``resolve(run_id, role)`` first looks for an exact ``(run_id, role)``
    match; if none exists, it falls back to a unique generic ``run_id``-only row.
    """

    def __init__(self, csv_rows: list[dict[str, str]]):
        self._by_runid_role: dict[tuple[str, str], dict[str, str]] = {}
        self._by_runid: dict[str, dict[str, str]] = {}

        for row in csv_rows:
            run_id = _normalize_run_id(row)
            role = _normalize_role(row)
            if role is not None:
                key = (run_id, role)
                if key in self._by_runid_role:
                    raise DuplicateMetadataError(
                        f"Duplicate metadata row for "
                        f"(run_id={run_id!r}, role={role!r})"
                    )
                self._by_runid_role[key] = row
            else:
                if run_id in self._by_runid:
                    raise DuplicateMetadataError(
                        f"Duplicate generic metadata row for run_id={run_id!r}"
                    )
                self._by_runid[run_id] = row

    def resolve(self, run_id: str, role: str) -> dict[str, str]:
        """Return the metadata row for a given (run_id, role).

        Tries ``(run_id, role)`` first, then generic ``run_id`` as fallback.
        Fails if neither exists.
        """
        key = (str(run_id), str(role))
        if key in self._by_runid_role:
            return dict(self._by_runid_role[key])
        if str(run_id) in self._by_runid:
            return dict(self._by_runid[str(run_id)])
        raise MetadataResolverError(
            f"No metadata row for (run_id={run_id!r}, role={role!r}) "
            f"and no generic row for run_id={run_id!r}"
        )

    def enrich_record(
        self, record: dict[str, Any], *, csv_path: str | None = None,
    ) -> dict[str, Any]:
        """Return a copy of *record* with resolved CSV metadata merged in.

        Backwards compatibility: if *record* carries embedded ``source_metadata``
        and the CSV row is also available, the two are validated for consistency
        on shared fields.  Conflicts raise ``MetadataConflictError``.
        """
        enriched = dict(record)
        run_id = str(record["run_id"])
        role = str(record.get("role", "watermarked"))
        embedded = record.get("source_metadata")

        csv_row = self.resolve(run_id, role)

        if isinstance(embedded, dict) and embedded:
            # Validate consistency on shared fields
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

        enriched["_metadata"] = dict(csv_row)

        # Top-level aliases for backwards-compatible detector access
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
        """Build a resolver from embedded ``source_metadata`` in legacy records.

        Role is taken from the record's own ``role`` field.  The same
        run_id with different roles (watermarked, clean) produces two
        role-specific rows rather than duplicate generic rows.
        """
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str | None]] = set()
        for rec in records:
            embedded = rec.get("source_metadata")
            if not isinstance(embedded, dict) or not embedded:
                continue
            row = {str(k): str(v) for k, v in embedded.items()}
            if "run_id" not in row and "sample_id" not in row and "id" not in row:
                row["run_id"] = str(rec.get("run_id", ""))

            # Use record's own role to create role-specific row
            rec_role = str(rec.get("role", "watermarked")).strip().lower()
            row["role"] = rec_role if rec_role in ("watermarked", "clean") else "watermarked"

            key = (row.get("run_id", ""), row["role"] if "role" in row else None)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        if not rows:
            return None
        return cls(rows)
