"""Run the sklearn/custom-kernel per-wavelength MF-GP workflows."""

from __future__ import annotations

import runpy
from pathlib import Path

from exoplanets_mf.paths import MODELLING_RESULTS_DIR
from exoplanets_mf.provenance import write_run_metadata

EXPERIMENT_RESULTS_DIR = MODELLING_RESULTS_DIR / "01_per_wavelength_ar1"


def main() -> None:
    workflow_dir = Path(__file__).resolve().parent
    workflows = [
        workflow
        for workflow in sorted(workflow_dir.glob("[0-9][0-9]_*.py"))
        if "gpboost" not in workflow.stem
    ]

    for workflow in workflows:
        print(f"\n{'=' * 72}\nRunning {workflow.name}\n{'=' * 72}", flush=True)
        runpy.run_path(str(workflow), run_name="__main__")

    write_run_metadata(
        EXPERIMENT_RESULTS_DIR / "run_metadata.json",
        workflow="modelling/01_per_wavelength_ar1",
        scripts=[workflow.name for workflow in workflows],
    )
    print(f"\nRun provenance written to {EXPERIMENT_RESULTS_DIR / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
