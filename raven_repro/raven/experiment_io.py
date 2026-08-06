"""Output layout, record I/O, resume/overwrite safety for the unified pipeline.

Canonical output layout::

    <output-dir>/
    ├── config.json
    ├── records.jsonl
    ├── evaluation/
    │   ├── detector_records.jsonl
    │   ├── quality_records.jsonl
    │   ├── fid_result.json
    │   └── clip_result.json
    └── samples/
        ├── watermarked/
        │   └── <run_id>/
        │       ├── output.png
        │       └── record.json
        └── clean/
            └── <run_id>/
                ├── output.png
                └── record.json

``records.jsonl`` is always rebuilt atomically from the individual
``record.json`` files so it can never disagree with them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

# Directories that must never be deleted by --overwrite.
PROTECTED_DIRS = frozenset({
    "/",
    "/workspace",
    "/workspace/RAVEN",
    "/workspace/RAVEN/data",
    "/workspace/RAVEN/outputs",
})


def _resolve_protected() -> frozenset[str]:
    """Resolve protected directories to absolute paths."""
    result = set()
    for path_str in PROTECTED_DIRS:
        p = Path(path_str)
        if p.exists():
            result.add(str(p.resolve()))
    repo = Path(__file__).resolve().parents[2]
    result.add(str(repo))
    result.add(str(repo / "data"))
    result.add(str(repo / "outputs"))
    return frozenset(result)


def sample_dir(output_dir: str | Path, role: str, run_id: str) -> Path:
    """Canonical per-sample output directory."""
    return Path(output_dir) / "samples" / role / str(run_id)


def output_image_path(output_dir: str | Path, role: str, run_id: str) -> Path:
    """Canonical ``output.png`` path for one sample."""
    return sample_dir(output_dir, role, run_id) / "output.png"


def record_path(output_dir: str | Path, role: str, run_id: str) -> Path:
    """Canonical ``record.json`` path for one sample."""
    return sample_dir(output_dir, role, run_id) / "record.json"


def config_path(output_dir: str | Path) -> Path:
    """Canonical ``config.json`` path."""
    return Path(output_dir) / "config.json"


def records_jsonl_path(output_dir: str | Path) -> Path:
    """Canonical ``records.jsonl`` path."""
    return Path(output_dir) / "records.jsonl"


def evaluation_dir(output_dir: str | Path) -> Path:
    """Canonical evaluation output directory."""
    return Path(output_dir) / "evaluation"


def detector_records_path(output_dir: str | Path) -> Path:
    """Canonical ``evaluation/detector_records.jsonl`` path."""
    return evaluation_dir(output_dir) / "detector_records.jsonl"


def is_sample_complete(output_dir: str | Path, role: str, run_id: str) -> bool:
    """A sample is complete when both ``output.png`` and ``record.json`` exist
    and the record carries ``status: "complete"``."""
    rec = record_path(output_dir, role, run_id)
    out = output_image_path(output_dir, role, run_id)
    if not rec.is_file() or not out.is_file():
        return False
    try:
        data = json.loads(rec.read_text(encoding="utf-8"))
        return data.get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


def validate_output_dir_safety(output_dir: str | Path) -> None:
    """Refuse to delete protected directories during --overwrite.

    Rejects the protected directory itself AND any child path.
    """
    resolved = str(Path(output_dir).resolve())
    for protected in _resolve_protected():
        if resolved == protected or resolved.startswith(protected + os.sep):
            raise ValueError(
                f"output-dir {resolved} is a protected directory or inside one. "
                "Choose a path outside the repository data/ and outputs/ trees."
            )


def _dir_is_empty(path: Path) -> bool:
    try:
        return not any(path.iterdir())
    except FileNotFoundError:
        return True


def prepare_output_dir(
    output_dir: str | Path,
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    """Create or validate the output directory.

    Rules
    -----
    * Directory does not exist → create.
    * Directory exists and is empty → use.
    * Directory is non-empty + ``--resume`` → validate config later, continue.
    * Directory is non-empty + ``--overwrite`` → safely delete and recreate.
    * Directory is non-empty, neither resume nor overwrite → fail.
    """
    path = Path(output_dir).resolve()

    if not path.exists():
        path.mkdir(parents=True)
        return path

    if _dir_is_empty(path):
        return path

    if overwrite:
        validate_output_dir_safety(path)
        import shutil
        shutil.rmtree(path)
        path.mkdir(parents=True)
        return path

    if resume:
        if not config_path(path).is_file():
            raise FileNotFoundError(
                f"output-dir {path} is non-empty but has no config.json — "
                "cannot resume."
            )
        return path

    raise FileExistsError(
        f"output-dir {path} is non-empty. Use --resume (to continue) "
        "or --overwrite (to start fresh)."
    )


def write_config(output_dir: str | Path, config: dict[str, Any]) -> Path:
    """Write ``config.json`` atomically."""
    target = config_path(output_dir)
    tmp = target.with_name(f".config.json.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


def read_config(output_dir: str | Path) -> dict[str, Any]:
    """Read ``config.json``.  Raises ``FileNotFoundError`` if missing."""
    return json.loads(config_path(output_dir).read_text(encoding="utf-8"))


def write_record(
    output_dir: str | Path,
    role: str,
    run_id: str,
    record: dict[str, Any],
) -> Path:
    """Write one ``record.json`` atomically.  ``status`` is forced to ``"complete"``."""
    rec = dict(record)
    rec["status"] = "complete"
    target = record_path(output_dir, role, run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".record.json.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


def read_record(output_dir: str | Path, role: str, run_id: str) -> dict[str, Any]:
    """Read one ``record.json``."""
    return json.loads(record_path(output_dir, role, run_id).read_text(encoding="utf-8"))


def rebuild_records_jsonl(output_dir: str | Path) -> Path:
    """Rebuild ``records.jsonl`` atomically from all complete ``record.json`` files.

    Reads every ``samples/*/*/record.json``, validates ``status == "complete"``,
    sorts by run_id, and writes ``records.jsonl`` in one atomic replacement.
    """
    root = Path(output_dir)
    samples_dir = root / "samples"
    records: list[dict[str, Any]] = []

    if samples_dir.is_dir():
        for role_dir in sorted(samples_dir.iterdir()):
            if not role_dir.is_dir():
                continue
            for run_dir in sorted(role_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                rec_file = run_dir / "record.json"
                if not rec_file.is_file():
                    continue
                data = json.loads(rec_file.read_text(encoding="utf-8"))
                if data.get("status") != "complete":
                    continue
                records.append(data)

    records.sort(key=lambda r: str(r.get("run_id", "")))

    target = records_jsonl_path(root)
    tmp = target.with_name(f".records.jsonl.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(
                json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n"
            )
    os.replace(tmp, target)
    return target


def read_records_jsonl(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read ``records.jsonl``.  Returns empty list if missing."""
    path = records_jsonl_path(output_dir)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_incomplete_run_ids(
    output_dir: str | Path,
    roles: Iterable[str],
    expected_run_ids: Iterable[str],
) -> list[tuple[str, str]]:
    """Return ``(role, run_id)`` pairs whose samples are not complete."""
    incomplete: list[tuple[str, str]] = []
    for role in roles:
        for run_id in expected_run_ids:
            if not is_sample_complete(output_dir, role, str(run_id)):
                incomplete.append((role, str(run_id)))
    return incomplete


def cleanup_intermediates(
    output_dir: str | Path,
    role: str,
    run_id: str,
) -> None:
    """Remove pipeline intermediate files, keeping only ``output.png``."""
    sample = sample_dir(output_dir, role, run_id)
    keep = {"output.png", "record.json"}
    for item in list(sample.iterdir()):
        if item.is_file() and item.name not in keep:
            item.unlink()
