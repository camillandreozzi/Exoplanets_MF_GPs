"""Run the GPBoost model-comparison workflow."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runpy.run_path(
        str(Path(__file__).with_name("01_compare_gpboost.py")),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
