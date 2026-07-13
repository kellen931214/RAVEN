"""Dataset manifest helpers for protocol-correct RAVEN evaluation.

The helpers in this module are intentionally lightweight and avoid importing
ML frameworks. They normalize legacy absolute paths, pair clean/watermarked/
RAVEN images, and provide a stable JSONL manifest used by the evaluation
scripts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


METHODS = ("GS", "TR", "RID", "HSTR", "HSQR")
RAVEN_FINAL_NAMES = ("final_color_corrected.png", "final.png", "view_guided_output.png")


@dataclass
class ManifestRow:
    dataset_name: str
    wm_type: str
    run_id: int
    prompt_id: str
    prompt: str
    source: str
    clean_path: str
    watermarked_path: str
    raven_path: str
    debug_info_path: str
    metadata_path: str
    result_path: str
    target_model: str = ""
    scheduler_target: str = ""
    num_inference_steps_target: str = ""
    guidance_scale_target: str = ""
    resolution: str = ""
    threshold_mode: str = ""
    threshold_source: str = ""
    detection_threshold: str = ""
    detection_metric: str = ""
    before_detection_successful: str = ""
    before_detection_metric_value: str = ""
    before_bit_accuracy: str = ""
    after_detection_successful: str = ""
    after_detection_metric_value: str = ""
    after_bit_accuracy: str = ""
    dx: Optional[int] = None
    dy: Optional[int] = None
    raven_model_id: str = ""
    raven_steps: str = ""
    raven_strength: str = ""
    raven_guidance_scale: str = ""
    shift_space: str = ""
    padding_mode: str = ""
    view_guided_attention: str = ""
    color_transfer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_path_map(items: Optional[Sequence[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"path map entries must be OLD=NEW, got {item!r}")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError(f"path map OLD prefix cannot be empty: {item!r}")
        mapping[old.rstrip("/")] = new.rstrip("/")
    return mapping


def resolve_legacy_path(value: str | Path | None, path_map: Mapping[str, str]) -> Path:
    if value in {None, ""}:
        return Path("")
    text = str(value)
    path = Path(text)
    if path.exists():
        return path
    for old, new in sorted(path_map.items(), key=lambda pair: len(pair[0]), reverse=True):
        if text == old or text.startswith(old + "/"):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
            return candidate
    return path


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_by_run_id(rows: Iterable[Mapping[str, str]]) -> Dict[int, Mapping[str, str]]:
    indexed: Dict[int, Mapping[str, str]] = {}
    for row in rows:
        try:
            indexed[int(str(row.get("run_id", "")).strip())] = row
        except Exception:
            continue
    return indexed


def find_raven_image(item_dir: Path) -> Path:
    for name in RAVEN_FINAL_NAMES:
        path = item_dir / name
        if path.exists():
            return path
    return Path("")


def read_debug_info(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
