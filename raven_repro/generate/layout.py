"""Canonical repository data/output layout for generation scripts.

These paths define where watermarked cohorts live on disk.  Attack and eval
runtime never import this module — only generation orchestrators use it.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = REPO_ROOT / "data"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
CLEAN_DATA_ROOT = DATA_ROOT / "clean"

METHOD_DATA_ROOTS: dict[str, Path] = {
    "TR": DATA_ROOT / "tr",
    "GS": DATA_ROOT / "gs",
    "GM": DATA_ROOT / "gm",
    "T2S": DATA_ROOT / "t2s",
    "RID": DATA_ROOT / "rid",
    "HSTR": DATA_ROOT / "hstr",
    "HSQR": DATA_ROOT / "hsqr",
}
METHOD_OUTPUT_ROOTS: dict[str, Path] = {
    "TR": OUTPUTS_ROOT / "tr",
    "GS": OUTPUTS_ROOT / "gs",
    "GM": OUTPUTS_ROOT / "gm",
    "T2S": OUTPUTS_ROOT / "t2s",
    "RID": OUTPUTS_ROOT / "rid",
    "HSTR": OUTPUTS_ROOT / "hstr",
    "HSQR": OUTPUTS_ROOT / "hsqr",
}

# Methods whose cohort uses the flat layout:
#   data/<method>/<dataset>/metadata.csv
#   data/<method>/<dataset>/<run_id>/watermarked.png
FLAT_COHORT_METHODS = frozenset({"TR"})


def method_data_root(method: str) -> Path:
    """Canonical watermarked-data root for ``method`` (fail closed on unknown)."""
    key = str(method).upper()
    try:
        return METHOD_DATA_ROOTS[key]
    except KeyError:
        raise ValueError(
            f"no canonical data root for method {method!r}; "
            f"known methods: {sorted(METHOD_DATA_ROOTS)}"
        ) from None


def cohort_dir(method: str, dataset: str) -> Path:
    """Directory holding one method's cohort metadata and per-run image dirs."""
    root = method_data_root(method) / dataset
    if str(method).upper() in FLAT_COHORT_METHODS:
        return root
    return root / str(method).upper()


def source_metadata_path(method: str, dataset: str) -> Path:
    """Canonical source metadata CSV for a generated cohort."""
    return cohort_dir(method, dataset) / "metadata.csv"


def watermarked_image_path(method: str, dataset: str, run_id: int | str) -> Path:
    """Canonical watermarked image path for one run of a cohort."""
    name = f"{int(run_id):06d}" if str(run_id).isdigit() else str(run_id)
    return cohort_dir(method, dataset) / name / "watermarked.png"


def clean_data_dir(dataset: str, method: str | None = None) -> Path:
    """Canonical clean-image directory for a dataset (GS cohorts nest under GS/)."""
    root = CLEAN_DATA_ROOT / dataset
    if method is not None and str(method).upper() == "GS":
        return root / "GS"
    return root
