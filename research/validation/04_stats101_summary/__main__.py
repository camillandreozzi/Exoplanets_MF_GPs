"""Run the Statistics-101 summary of held-out CV predictions."""

from __future__ import annotations

import runpy
from pathlib import Path

from exoplanets_mf.paths import VALIDATION_RESULTS_DIR
from exoplanets_mf.provenance import write_run_metadata

EXPERIMENT_RESULTS_DIR = VALIDATION_RESULTS_DIR / "04_stats101_summary"


def main() -> None:
    workflow_dir = Path(__file__).resolve().parent
    workflows = sorted(workflow_dir.glob("[0-9][0-9]_*.py"))

    for workflow in workflows:
        print(f"\n{'=' * 72}\nRunning {workflow.name}\n{'=' * 72}", flush=True)
        runpy.run_path(str(workflow), run_name="__main__")

    write_run_metadata(
        EXPERIMENT_RESULTS_DIR / "run_metadata.json",
        workflow="validation/04_stats101_summary",
        scripts=[workflow.name for workflow in workflows],
    )
    print(f"\nRun provenance written to {EXPERIMENT_RESULTS_DIR / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
