"""Plot HF, Model 1, Model 3 and CN LOO spectra for a zero-based HF sample.

CN has no sample IDs; its rows follow the HF order, as in the existing CN
comparison script. Defaults match the sample 81 closure test and lf10000 run.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results/.matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=81)
    parser.add_argument("--variant", choices=("mf", "sf"), default="mf")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/cv/loo_lf10000")
    args = parser.parse_args()

    hf = pd.read_csv(ROOT / "data/YHF.csv", index_col=0)
    cn = pd.read_csv(ROOT / "data/lit/YSF_LOO.csv")
    if not 0 <= args.sample < len(hf):
        raise ValueError(f"Sample must be between 0 and {len(hf) - 1}.")
    if hf.shape != cn.shape:
        raise ValueError("HF and CN table dimensions differ.")
    wavelength = hf.columns.to_numpy(dtype=float)
    np.testing.assert_allclose(cn.columns.to_numpy(dtype=float), wavelength,
                               rtol=0, atol=1e-10)
    truth = hf.iloc[args.sample].to_numpy(dtype=float)
    table = pd.DataFrame({"wavelength_um": wavelength, "hf": truth})
    predictions = pd.read_csv(args.results_dir / "loo_predictions.csv")
    selected = predictions[
        predictions.fold.eq(args.sample + 1)
        & predictions.variant.eq(args.variant)
        & predictions.model.isin(["model1", "model3"])
    ]
    if selected.held_out_source_index.nunique() != 1:
        raise ValueError("Expected a single held-out source for this fold.")
    for model in ("model1", "model3"):
        rows = selected[selected.model.eq(model)].sort_values("response_index")
        np.testing.assert_array_equal(rows.response_index, np.arange(len(wavelength)))
        np.testing.assert_allclose(rows.wavelength, wavelength, rtol=0, atol=1e-10)
        np.testing.assert_allclose(rows.y_true, truth, rtol=1e-10, atol=1e-14)
        table[f"{model}_{args.variant}"] = rows.y_pred.to_numpy()
    table["cn"] = cn.iloc[args.sample].to_numpy(dtype=float)
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Nonfinite spectrum values.")
    table = table.sort_values("wavelength_um")

    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    series = [
        ("hf", "HF values", "#171717", "-"),
        (f"model1_{args.variant}", f"Model 1 {args.variant.upper()}", "#2a78d6", "--"),
        (f"model3_{args.variant}", f"Model 3 {args.variant.upper()}", "#15956b", "-."),
        ("cn", "CN baseline", "#df682e", ":"),
    ]
    for column, label, color, style in series:
        ax.plot(table.wavelength_um, table[column] * 1e6, label=label,
                color=color, linestyle=style, linewidth=1.9,
                zorder=4 if column == "hf" else 3)
    ax.set(xlabel="Wavelength (μm)", ylabel="Eclipse depth (ppm)")
    ax.set_title(
        f"Sample {args.sample} · HF vs LOO predictions · LF 10,000\n"
        f"Zero-based HF index {args.sample} ({hf.index[args.sample]}) · LOO fold {args.sample + 1} · {len(table)} wavelengths",
        fontsize=12, pad=12,
    )
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.margins(x=0.01)
    stem = args.results_dir / f"sample{args.sample}_hf_model1_model3_{args.variant}_cn_spectrum"
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)
    table.to_csv(stem.with_suffix(".csv"), index=False)
    print(f"Saved {stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
