"""Run all exploratory workflows in their numbered order."""

from __future__ import annotations

import runpy
from pathlib import Path

from exoplanets_mf.paths import EXPLORATORY_RESULTS_DIR
from exoplanets_mf.provenance import write_run_metadata


def main() -> None:
    workflow_dir = Path(__file__).resolve().parent
    workflows = sorted(workflow_dir.glob("[0-9][0-9]_*.py"))

    for workflow in workflows:
        print(f"\n{'=' * 72}\nRunning {workflow.name}\n{'=' * 72}", flush=True)
        runpy.run_path(str(workflow), run_name="__main__")

    write_run_metadata(
        EXPLORATORY_RESULTS_DIR / "run_metadata.json",
        workflow="exploratory",
        scripts=[workflow.name for workflow in workflows],
    )
    print(
        f"\nRun provenance written to "
        f"{EXPLORATORY_RESULTS_DIR / 'run_metadata.json'}"
    )


if __name__ == "__main__":
    main()
