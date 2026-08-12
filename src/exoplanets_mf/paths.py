"""Canonical project paths.

Keeping paths here makes workflows independent of the caller's working
directory. Set ``EXOPLANETS_MF_ROOT`` only when the repository is deliberately
relocated relative to the installed package.
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    default = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("EXOPLANETS_MF_ROOT", default)).expanduser().resolve()


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPLORATORY_RESULTS_DIR = RESULTS_DIR / "exploratory"
REVERSE_LABEL_RESULTS_DIR = RESULTS_DIR / "reverse_label"
MODELLING_RESULTS_DIR = RESULTS_DIR / "modelling"
LOG_MODELLING_RESULTS_DIR = RESULTS_DIR / "log_modelling"
VALIDATION_RESULTS_DIR = RESULTS_DIR / "validation"
PARAMETERS_RESULTS_DIR = RESULTS_DIR / "parameters"


def approximation_suffix(
    gp_approx: str, num_neighbors: int | None, *, default: str = "none"
) -> str:
    """Output-directory suffix identifying a GPBoost approximation.

    Exact and approximate fits of the same model are different estimators, so
    they must not overwrite each other's outputs. A run at the workflow's
    default approximation keeps the plain directory name (empty suffix);
    anything else is labelled, e.g. ``_vecchia_k30``.
    """
    if gp_approx == default:
        return ""
    if gp_approx == "none":
        return "_exact"
    suffix = f"_{gp_approx}"
    return suffix if num_neighbors is None else f"{suffix}_k{num_neighbors}"
