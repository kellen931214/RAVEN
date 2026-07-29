"""Load the *real* frozen official SFWMark implementation for parity testing.

This module does not transcribe the official algorithm: it executes the actual
bytes of ``src/utils.py`` from

    https://github.com/thomas11809/SFWMark
    commit 78666128b44614a0cc471993649e3132d5dddfcb

so the fixtures in ``hsqr_official_fixtures.json`` are produced by the official
code itself rather than by a re-reading of the paper.

Why an exec loader instead of a plain import
--------------------------------------------
The official ``src/utils.py`` imports the union of every dependency the whole
official project needs (``bm3d``, ``compressai``, ``lpips``, ``pyzbar``,
``prettytable``, ``pytorch_fid`` ...), most of which are unrelated to HSQR and
are not installed here. Importing it normally therefore fails before reaching a
single line of watermark code. The loader installs inert stub modules for the
dependencies that are missing, then executes the unmodified official file.

Two properties keep this honest:

* the file is **hash-pinned** (:data:`OFFICIAL_UTILS_SHA256`). If the official
  source changes by a single byte, loading fails instead of silently producing
  different fixtures.
* only *missing* dependencies are stubbed, and none of the stubbed modules is
  reachable from the HSQR pattern / injection / distance code paths, all of
  which use nothing beyond ``torch``, ``numpy`` and ``qrcode``. The stubs cannot
  therefore change any value this module is used to produce.

The official checkout is not vendored into this repository (it carries its own
licence). Tests that need it are skipped when it is absent; the committed
fixtures are what make the parity assertions run in a normal checkout. Point
the loader at a checkout with::

    export SFWMARK_OFFICIAL_SRC=/path/to/SFWMark

or place one at ``third_party/SFWMark`` inside the repository.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path

# Imported eagerly, before any stubbing or exec, so the official file binds the
# already-initialised real packages. Importing torch for the first time from
# inside ``exec`` re-runs its C++ library registration and aborts.
import numpy  # noqa: F401
import qrcode  # noqa: F401
import torch  # noqa: F401
import torchvision.transforms  # noqa: F401

OFFICIAL_SFWMARK_REPO = "https://github.com/thomas11809/SFWMark"
OFFICIAL_SFWMARK_COMMIT = "78666128b44614a0cc471993649e3132d5dddfcb"

#: SHA-256 of ``src/utils.py`` at the frozen commit. Loading refuses to continue
#: when the on-disk file does not match, so fixtures can never be regenerated
#: from a different revision of the official code.
OFFICIAL_UTILS_SHA256 = "d3deb279006a143e2e082a1bf1195c5cc4846722bf01a702c58efbb834f6dea8"
OFFICIAL_GENERATE_SHA256 = "bf8b79543d6ddfc4b904a5026c82bd0f32d91bedf2b6a46c0e4e8d51fc40243e"
OFFICIAL_DETECT_SHA256 = "fac82635c48843fbbbb172dd918dadb13d1fc8694dbf23f8b30d310c2a70ea0c"

#: Environment variable pointing at a checkout of the official repository.
OFFICIAL_SRC_ENV = "SFWMARK_OFFICIAL_SRC"

#: Dependencies of the official ``utils.py`` that HSQR never touches. Each is
#: stubbed only if it is genuinely not importable.
_STUBBABLE = (
    "prettytable",
    "pyzbar",
    "pyzbar.pyzbar",
    "bm3d",
    "lpips",
    "compressai",
    "compressai.zoo",
    "pytorch_fid",
    "pytorch_fid.fid_score",
    "datasets",
    "matplotlib",
    "matplotlib.pyplot",
    "skimage",
    "skimage.metrics",
    "cv2",
)


class OfficialSourceUnavailable(RuntimeError):
    """Raised when the frozen official source cannot be loaded."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_roots() -> list[Path]:
    """Directories that may hold a checkout of the official repository."""
    roots: list[Path] = []
    env = os.environ.get(OFFICIAL_SRC_ENV)
    if env:
        roots.append(Path(env))
    repo_root = Path(__file__).resolve().parents[2]
    roots.append(repo_root / "third_party" / "SFWMark")
    return roots


def find_official_root() -> Path:
    """Return the official checkout, verifying its commit and file hashes."""
    tried: list[str] = []
    for root in candidate_roots():
        utils_path = root / "src" / "utils.py"
        if not utils_path.is_file():
            tried.append(f"{root} (no src/utils.py)")
            continue
        actual = _sha256_file(utils_path)
        if actual != OFFICIAL_UTILS_SHA256:
            raise OfficialSourceUnavailable(
                f"official src/utils.py at {root} has SHA-256 {actual}, expected "
                f"{OFFICIAL_UTILS_SHA256} for commit {OFFICIAL_SFWMARK_COMMIT}. "
                "Check out the frozen commit before regenerating fixtures."
            )
        _assert_commit(root)
        return root
    raise OfficialSourceUnavailable(
        "frozen official SFWMark source not found. Clone "
        f"{OFFICIAL_SFWMARK_REPO} at commit {OFFICIAL_SFWMARK_COMMIT} and set "
        f"${OFFICIAL_SRC_ENV}. Tried: " + ", ".join(tried or ["<none>"])
    )


def _assert_commit(root: Path) -> None:
    """Best-effort check that the checkout sits on the frozen commit."""
    if not (root / ".git").exists():
        return
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return
    if head != OFFICIAL_SFWMARK_COMMIT:
        raise OfficialSourceUnavailable(
            f"official checkout at {root} is on commit {head}, expected the frozen "
            f"commit {OFFICIAL_SFWMARK_COMMIT}"
        )


def _install_stubs() -> list[str]:
    """Stub the official dependencies that are missing. Returns their names."""
    stubbed: list[str] = []
    for name in _STUBBABLE:
        if name in sys.modules:
            continue
        try:
            __import__(name)
            continue
        except Exception:
            pass
        module = types.ModuleType(name)
        # ``from <stub> import *`` must not route through __getattr__.
        module.__all__ = []
        module.__dict__["__getattr__"] = lambda attr, _n=name: _StubAttr(f"{_n}.{attr}")
        if "." in name:
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is None:
                parent = types.ModuleType(parent_name)
                parent.__path__ = []  # mark as a package
                sys.modules[parent_name] = parent
                stubbed.append(parent_name)
            setattr(parent, child, module)
        else:
            module.__path__ = []
        sys.modules[name] = module
        stubbed.append(name)
    return stubbed


#: Every attribute access and call made against a stubbed dependency, in order.
#: The fixture generator asserts this stays empty across the HSQR calls, which
#: is the machine-checked form of "no stub can influence a fixture value".
STUB_USAGE: list[str] = []


def reset_stub_usage() -> None:
    """Forget recorded stub usage (call before a block that must not touch stubs)."""
    STUB_USAGE.clear()


def stub_usage() -> list[str]:
    """Stub attributes touched since the last :func:`reset_stub_usage`."""
    return list(STUB_USAGE)


class _StubAttr:
    """Permissive placeholder for an attribute of a stubbed dependency.

    The official ``utils.py`` builds unrelated image-attack models (compressai
    VAEs, LPIPS) at import time, so stubs must tolerate being called and
    chained. Every touch is recorded in :data:`STUB_USAGE` so tests can prove
    the HSQR code paths never reached one.
    """

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)
        STUB_USAGE.append(name)

    def __call__(self, *args, **kwargs):
        return _StubAttr(f"{object.__getattribute__(self, '_name')}()")

    def __getattr__(self, attr: str):
        return _StubAttr(f"{object.__getattribute__(self, '_name')}.{attr}")

    def __iter__(self):
        return iter(())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<stub {object.__getattribute__(self, '_name')}>"


_CACHE: dict[str, types.ModuleType] = {}


def load_official_utils() -> types.ModuleType:
    """Execute the frozen official ``src/utils.py`` and return it as a module."""
    if "utils" in _CACHE:
        return _CACHE["utils"]

    root = find_official_root()
    utils_path = root / "src" / "utils.py"

    _install_stubs()

    module = types.ModuleType("sfwmark_official_utils")
    module.__file__ = str(utils_path)
    source = utils_path.read_text(encoding="utf-8")
    sys.path.insert(0, str(root / "src"))
    try:
        exec(compile(source, str(utils_path), "exec"), module.__dict__)
    finally:
        try:
            sys.path.remove(str(root / "src"))
        except ValueError:  # pragma: no cover - defensive
            pass

    _CACHE["utils"] = module
    return module


def official_available() -> bool:
    """True when the frozen official source can be located and loaded."""
    try:
        load_official_utils()
        return True
    except Exception:
        return False


def official_provenance() -> dict:
    """Provenance block recorded alongside generated fixtures."""
    return {
        "official_repo": OFFICIAL_SFWMARK_REPO,
        "official_commit": OFFICIAL_SFWMARK_COMMIT,
        "official_utils_sha256": OFFICIAL_UTILS_SHA256,
        "official_generate_sha256": OFFICIAL_GENERATE_SHA256,
        "official_detect_sha256": OFFICIAL_DETECT_SHA256,
    }
