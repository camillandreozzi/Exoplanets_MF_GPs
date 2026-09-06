"""Compare Model 1 and Model 3 in-sample NRMSE per wavelength."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMPARISON_DIR = PROJECT_ROOT / "results" / "comparisons"
MODEL1_FILE = PROJECT_ROOT / "results/model1/full_fit/model1_sf_vs_mf_nrmse_per_wavelength.csv"
MODEL3_FILE = PROJECT_ROOT / "results/model3/full_fit/model3_sf_vs_mf_nrmse_per_wavelength.csv"
OUTPUT_CSV = COMPARISON_DIR / "model1_model3_sf_mf_nrmse_per_wavelength.csv"
OUTPUT_PLOT = COMPARISON_DIR / "model1_model3_sf_mf_nrmse_per_wavelength.png"
os.environ.setdefault("MPLCONFIGDIR", str(COMPARISON_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "wavelength",
    "sf_nrmse",
    "mf_nrmse",
    "delta_mf_minus_sf",
}


def main():
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    comparison = combine_model_results()
    comparison.to_csv(OUTPUT_CSV, index=False)
    plot_comparison(comparison)

    print(f"Saved Model 1 vs Model 3 NRMSE CSV to {OUTPUT_CSV}")
    print(f"Saved Model 1 vs Model 3 NRMSE plot to {OUTPUT_PLOT}")


def combine_model_results():
    model1 = read_model_result(MODEL1_FILE, "model1")
    model3 = read_model_result(MODEL3_FILE, "model3")
    combined = model1.merge(model3, on="wavelength", how="inner")

    if combined.empty:
        raise ValueError("Model 1 and Model 3 NRMSE files have no wavelengths in common.")
    if len(combined) != min(len(model1), len(model3)):
        raise ValueError("Model 1 and Model 3 NRMSE files do not align by wavelength.")

    return combined.sort_values("wavelength")


def read_model_result(path, model):
    if not path.exists():
        raise FileNotFoundError(f"Missing {model} NRMSE result file: {path}")

    data = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return data.loc[
        :,
        ["wavelength", "sf_nrmse", "mf_nrmse", "delta_mf_minus_sf"],
    ].rename(
        columns={
            "sf_nrmse": f"{model}_sf_nrmse",
            "mf_nrmse": f"{model}_mf_nrmse",
            "delta_mf_minus_sf": f"{model}_delta_mf_minus_sf",
        }
    )


def plot_comparison(comparison):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.1, 9.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    plot_nrmse_curves(axes[0], comparison)
    plot_delta_curves(axes[1], comparison)

    axes[1].set_xlabel("Wavelength")
    fig.savefig(OUTPUT_PLOT, dpi=200)
    plt.close(fig)


def plot_nrmse_curves(ax, comparison):
    for column, label, color, linestyle in [
        ("model1_sf_nrmse", "Model 1 SF", "tab:orange", "-"),
        ("model1_mf_nrmse", "Model 1 MF", "tab:blue", "--"),
        ("model3_sf_nrmse", "Model 3 SF", "tab:green", "-"),
        ("model3_mf_nrmse", "Model 3 MF", "tab:red", "--"),
    ]:
        ax.plot(
            comparison["wavelength"],
            comparison[column],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )

    ax.set_ylabel("NRMSE")
    ax.set_title("Model 1 vs Model 3 NRMSE per Wavelength (in sample): SF vs MF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)


def plot_delta_curves(ax, comparison):
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    for column, label, color in [
        ("model1_delta_mf_minus_sf", "Model 1 MF - SF", "tab:blue"),
        ("model3_delta_mf_minus_sf", "Model 3 MF - SF", "tab:red"),
    ]:
        ax.plot(
            comparison["wavelength"],
            comparison[column],
            label=label,
            color=color,
            linewidth=1.5,
        )

    ax.set_ylabel("Delta NRMSE")
    ax.set_title("MF - SF NRMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


if __name__ == "__main__":
    main()
