"""Held-out spectrum overlay: actual vs every model's out-of-fold prediction.

Plots one spectrum (default: spectrum 81, the historical holdout sample) as
predicted by Models 1A, 1B, and 2 in the fold where it was held out, using
the CV predictions saved by the Model 2 CV workflow. Top panel shows the
linear-scale fits, bottom panel the log10 fits back-transformed with
``10**prediction``; shaded bands are 95% predictive intervals (every model
is a joint MF-GP with a predictive std in the archive).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exoplanets_mf.data import load_YHF
from exoplanets_mf.paths import (
    LOG_MODELLING_RESULTS_DIR,
    MODELLING_RESULTS_DIR,
    VALIDATION_RESULTS_DIR,
)

CV_DIRS = {
    "linear": MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "cv" / "linear",
    "log10": (
        LOG_MODELLING_RESULTS_DIR / "02_augmented_wavelength" / "cv" / "log10"
    ),
}
OUTPUT_DIR = VALIDATION_RESULTS_DIR / "04_stats101_summary"

SPECTRUM_LABEL = "spectrum 81"

MODEL_SHORT_LABELS = {
    "model_1a": "Model 1A",
    "model_1b": "Model 1B",
    "model_2": "Model 2",
}
MODEL_COLORS = {
    "model_1a": "#2a78d6",
    "model_1b": "#1baf7a",
    "model_2": "#008300",
}
GP_MODELS = ("model_1a", "model_1b", "model_2")


def load_cv_predictions(scale: str) -> dict[str, np.ndarray]:
    path = CV_DIRS[scale] / "cv_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make modelling_sklearn` first to generate the "
            "out-of-fold CV predictions this plot reuses."
        )
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def plot_holdout_spectrum(
    wavelengths: np.ndarray,
    y_true: np.ndarray,
    linear_archive: dict[str, np.ndarray],
    log_archive: dict[str, np.ndarray],
    row: int,
    savepath: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)

    for ax, archive, view, backtransform in (
        (axes[0], linear_archive, "linear fits", False),
        (axes[1], log_archive, "log10 back-transformed fits", True),
    ):
        for key, label in MODEL_SHORT_LABELS.items():
            pred = archive[f"y_pred_{key}"][row]
            if backtransform:
                pred = 10**pred
            ax.plot(wavelengths, pred, lw=1.2, color=MODEL_COLORS[key], label=label)
        for key in GP_MODELS:
            mu = archive[f"y_pred_{key}"][row]
            sd = archive[f"y_std_{key}"][row]
            low, high = mu - 1.96 * sd, mu + 1.96 * sd
            if backtransform:
                low, high = 10**low, 10**high
            ax.fill_between(
                wavelengths, low, high, color=MODEL_COLORS[key], alpha=0.12, lw=0
            )
        ax.plot(wavelengths, y_true, color="black", lw=1.6, label="actual (held out)")
        ax.set_ylabel("eclipse depth")
        ax.set_title(view)
        ax.legend(ncol=5, fontsize=9, loc="upper left")

    axes[1].set_xlabel(r"wavelength $\lambda$ [$\mu$m]")
    fig.suptitle(
        f"{SPECTRUM_LABEL.capitalize()}: out-of-fold predictions vs actual "
        "(shaded: 95% intervals for GP models)"
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    YHF = load_YHF()
    row = list(YHF.index).index(SPECTRUM_LABEL)
    wavelengths = YHF.columns.to_numpy(dtype=float)
    y_true = YHF.to_numpy()[row]

    linear_archive = load_cv_predictions("linear")
    log_archive = load_cv_predictions("log10")

    savepath = OUTPUT_DIR / f"03_{SPECTRUM_LABEL.replace(' ', '')}_holdout.png"
    plot_holdout_spectrum(
        wavelengths, y_true, linear_archive, log_archive, row, savepath
    )
    print(
        f"{SPECTRUM_LABEL} (row {row}, held out in fold "
        f"{int(linear_archive['fold_of_sample'][row])}) plotted to {savepath}"
    )


if __name__ == "__main__":
    main()
