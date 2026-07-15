"""Record enough execution context to audit a generated research result."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT
from .reproducibility import RANDOM_SEED

TRACKED_PACKAGES = (
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
)


def write_run_metadata(
    destination: Path,
    *,
    workflow: str,
    scripts: list[str],
) -> None:
    """Write a JSON record of code, data, environment, and deterministic seed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "scripts": scripts,
        "random_seed": RANDOM_SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in TRACKED_PACKAGES
        },
        "git": _git_metadata(),
        "data_checksums_sha256": _data_checksums(),
        "command": (
            f"{sys.executable} -m research.{workflow.replace('/', '.')}"
        ),
    }
    destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_metadata() -> dict[str, object]:
    revision = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")
    return {
        "revision": revision or None,
        "has_uncommitted_changes": bool(status),
    }


def _run_git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def _data_checksums() -> dict[str, str]:
    manifest = PROJECT_ROOT / "data" / "checksums.sha256"
    checksums = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = digest
    return checksums
