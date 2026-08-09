"""Make the GPBoost macOS wheel importable without Homebrew.

The precompiled GPBoost arm64 wheel (``lib_gpboost.so``) is dynamically linked
against Homebrew's OpenMP runtime at the absolute path
``/opt/homebrew/opt/libomp/lib/libomp.dylib``. On a machine without Homebrew
``import gpboost`` fails with a dlopen error, and the OpenMP runtimes that ship
inside other wheels (scikit-learn, R) are too old -- they lack the
``__kmpc_dispatch_deinit`` symbol this GPBoost build needs.

This script vendors a pinned, modern ``libomp.dylib`` (from conda-forge's
``llvm-openmp``) next to ``lib_gpboost.so`` and rewrites the load command to
``@loader_path/libomp.dylib`` so GPBoost loads the sibling copy. It is
idempotent (a no-op once patched) and a no-op on non-macOS / non-arm64
platforms, where the Linux/Windows wheels bundle their own OpenMP.

``make setup`` runs this automatically after ``pip install``. It can also be run
by hand: ``.venv/bin/python tools/fix_gpboost_libomp.py``.

FEEDBACK (for the GPBoost maintainer): the macOS wheels would be far easier to
install if they bundled and @rpath-linked libomp the way scikit-learn's wheels
do, instead of hard-linking a Homebrew absolute path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import platform
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Pinned conda-forge llvm-openmp (osx-arm64). Modern enough to export
# __kmpc_dispatch_deinit; verified to load the GPBoost 1.7.1.1 arm64 wheel.
LIBOMP_VERSION = "22.1.8"
LIBOMP_CONDA_URL = (
    "https://conda.anaconda.org/conda-forge/osx-arm64/"
    "llvm-openmp-22.1.8-hc7d1edf_0.conda"
)
LIBOMP_CONDA_SHA256 = (
    "ccbaad6bbc88f135ab849bc36af5fa6eda36a9ed18ce6f58e3dde3d11784c156"
)
# sha256 of libomp.dylib as extracted from the package, BEFORE we rewrite its
# install-name and ad-hoc re-sign it (both of which change the on-disk bytes).
LIBOMP_DYLIB_SHA256 = (
    "844796bbd71a7492d78776fad75a306fbb1206e521c1d4ca593e5c1b83d338f1"
)

HOMEBREW_LIBOMP = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
VENDORED_LOAD_PATH = "@loader_path/libomp.dylib"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gpboost_dir() -> Path | None:
    """Locate the installed gpboost package without importing it."""
    spec = importlib.util.find_spec("gpboost")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _current_libomp_load_path(lib_so: Path) -> str | None:
    """Return the libomp.dylib path currently baked into lib_gpboost.so."""
    out = subprocess.run(
        ["otool", "-L", str(lib_so)], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        entry = line.strip().split(" ")[0]
        if entry.endswith("libomp.dylib"):
            return entry
    return None


def _download_libomp() -> bytes:
    """Download the pinned conda package and extract lib/libomp.dylib."""
    from compression import zstd  # Python 3.14 stdlib

    print(f"  downloading llvm-openmp {LIBOMP_VERSION} from conda-forge ...")
    with urllib.request.urlopen(LIBOMP_CONDA_URL) as response:
        conda_bytes = response.read()
    got = _sha256(conda_bytes)
    if got != LIBOMP_CONDA_SHA256:
        raise RuntimeError(
            f"conda artifact sha256 mismatch: expected {LIBOMP_CONDA_SHA256}, "
            f"got {got}"
        )
    with zipfile.ZipFile(io.BytesIO(conda_bytes)) as archive:
        pkg = next(
            name
            for name in archive.namelist()
            if name.startswith("pkg-") and name.endswith(".tar.zst")
        )
        tar_bytes = zstd.decompress(archive.read(pkg))
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        member = next(
            m for m in tar.getmembers() if m.name.endswith("lib/libomp.dylib")
        )
        dylib = tar.extractfile(member).read()
    got = _sha256(dylib)
    if got != LIBOMP_DYLIB_SHA256:
        raise RuntimeError(
            f"libomp.dylib sha256 mismatch: expected {LIBOMP_DYLIB_SHA256}, "
            f"got {got}"
        )
    return dylib


def _ensure_vendored_libomp(dest: Path) -> None:
    # Always (re)create when called: the caller only reaches here when the
    # top-level idempotency check found the patch missing or broken, so a stale
    # copy must not be trusted. The freshly extracted bytes are hash-verified
    # inside _download_libomp; the install-name rewrite + re-sign below then
    # deliberately change them.
    dest.write_bytes(_download_libomp())
    dest.chmod(0o755)
    # Self-referential id + ad-hoc re-sign (arm64 requires a valid signature).
    subprocess.run(
        ["install_name_tool", "-id", VENDORED_LOAD_PATH, str(dest)], check=True
    )
    subprocess.run(["codesign", "--force", "--sign", "-", str(dest)], check=True)


def _import_ok() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import gpboost"], capture_output=True, text=True
    )
    return result.returncode == 0


def main() -> int:
    if sys.platform != "darwin":
        print("fix_gpboost_libomp: non-macOS platform, nothing to do.")
        return 0
    if platform.machine() != "arm64":
        print(
            "fix_gpboost_libomp: only Apple-silicon (arm64) is auto-vendored; "
            "on Intel macOS install libomp via `brew install libomp`."
        )
        return 0

    gpb_dir = _gpboost_dir()
    if gpb_dir is None:
        print("fix_gpboost_libomp: gpboost is not installed, nothing to do.")
        return 0

    lib_so = gpb_dir / "lib_gpboost.so"
    vendored = gpb_dir / "libomp.dylib"
    if not lib_so.exists():
        print(f"fix_gpboost_libomp: {lib_so} not found, nothing to do.")
        return 0

    current = _current_libomp_load_path(lib_so)
    if current == VENDORED_LOAD_PATH and vendored.exists() and _import_ok():
        print("fix_gpboost_libomp: already patched, import works.")
        return 0

    print("fix_gpboost_libomp: patching GPBoost to use a vendored libomp ...")
    _ensure_vendored_libomp(vendored)
    if current is not None and current != VENDORED_LOAD_PATH:
        subprocess.run(
            ["install_name_tool", "-change", current, VENDORED_LOAD_PATH, str(lib_so)],
            check=True,
        )
        subprocess.run(["codesign", "--force", "--sign", "-", str(lib_so)], check=True)

    if not _import_ok():
        print("fix_gpboost_libomp: ERROR -- import gpboost still fails.", file=sys.stderr)
        return 1
    print(f"fix_gpboost_libomp: OK, `import gpboost` works ({gpb_dir}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
