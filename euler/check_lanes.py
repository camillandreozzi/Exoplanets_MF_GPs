"""Report which LOO lanes are still missing from a results directory.

Reads the same CV_LOO_* environment as modelling/cv/cv_loo.py, so inside a job
it reports on exactly the run that was submitted. The flags below override that
environment, for asking about a run from an ordinary shell:

    python3 euler/check_lanes.py --models 1,3 --lf 10000
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="1, 2, 3, a comma list, or all")
    parser.add_argument("--lf", help="Low-fidelity sample size, or 'none'")
    parser.add_argument("--results-dir", help="Results directory to inspect")
    parser.add_argument("--tag", help="Run tag naming the default results directory")
    parser.add_argument("--wavelengths", help="Wavelength subset size")
    parser.add_argument("--max-folds", help="Fold cap")
    return parser.parse_args()


def apply_overrides(arguments):
    """Turn the flags into CV_LOO_* variables, before cv_loo reads them."""
    if arguments.models is not None:
        models = "1,2,3" if arguments.models == "all" else arguments.models
        selected = {model.strip() for model in models.split(",")}
        for number in ("1", "2", "3"):
            os.environ[f"CV_LOO_RUN_MODEL{number}"] = (
                "1" if number in selected else "0"
            )
    if arguments.lf is not None:
        os.environ["CV_LOO_LF_SAMPLE_SIZE"] = arguments.lf
    # Mirror the default directory that submit_loo.sh derives, so that the same
    # --models/--lf that submitted a run also inspect it. Without this, --lf
    # changes the sample size but leaves this pointed at the unsuffixed
    # results/cv/loo, which is a different run entirely.
    if arguments.results_dir is not None:
        os.environ["CV_LOO_RESULTS_DIR"] = arguments.results_dir
    elif arguments.tag is not None:
        os.environ["CV_LOO_RESULTS_DIR"] = f"results/cv/loo_{arguments.tag}"
    elif arguments.lf is not None:
        os.environ["CV_LOO_RESULTS_DIR"] = f"results/cv/loo_lf{arguments.lf}"
    if arguments.wavelengths is not None:
        os.environ["CV_LOO_N_RESPONSE_SAMPLE"] = arguments.wavelengths
    if arguments.max_folds is not None:
        os.environ["CV_LOO_MAX_FOLDS"] = arguments.max_folds


def main():
    apply_overrides(parse_arguments())

    # Imported after the overrides: cv_loo reads its configuration at import.
    from modelling.cv import cv_5fold as cv_helpers
    from modelling.cv import cv_loo
    from src.data_load import load_full_data

    # all_tasks() needs the HF count to know how many Model 2 folds there are,
    # and only cv_loo.main() normally sets it.
    cv_data = cv_helpers.sample_lf_rows(
        load_full_data(),
        lf_sample_size=cv_loo.LF_SAMPLE_SIZE,
        random_state=cv_loo.LF_SAMPLE_RANDOM_STATE,
    )
    cv_loo._HF_SAMPLE_COUNT = int(cv_data["is_hf"].eq(1).sum())

    missing = defaultdict(list)
    total = defaultdict(int)
    for model_name, key in cv_loo.all_tasks():
        total[model_name] += 1
        if not cv_loo.lane_output_file(model_name, key).exists():
            missing[model_name].append(key)

    print(f"Lane status in {cv_loo.LANE_RESULTS_DIR}")
    for model_name in sorted(total):
        done = total[model_name] - len(missing[model_name])
        print(f"  {model_name}: {done}/{total[model_name]} lanes complete")
        if missing[model_name]:
            print(f"    missing: {compact(missing[model_name])}")

    if any(missing.values()):
        print(
            "\nResubmit with the same settings to fit only these; completed "
            "lanes are skipped."
        )
    else:
        print("\nAll lanes complete.")


def compact(keys):
    """Collapse consecutive keys into ranges, so 195 missing lanes stay legible."""
    keys = sorted(keys)
    ranges = []
    start = previous = keys[0]
    for key in keys[1:]:
        if key == previous + 1:
            previous = key
            continue
        ranges.append((start, previous))
        start = previous = key
    ranges.append((start, previous))

    return ", ".join(
        str(first) if first == last else f"{first}-{last}" for first, last in ranges
    )


if __name__ == "__main__":
    main()
