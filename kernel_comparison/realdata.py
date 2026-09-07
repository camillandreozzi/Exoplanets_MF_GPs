"""Real-data study: scikit-learn vs GPBoost exact vs GPBoost Vecchia.

The 97 high-fidelity atmospheres in ``data/XHF.csv`` have a matched
low-fidelity spectrum for exactly the same inputs (``data/YLF.csv`` vs
``data/YHF.csv``), which is the ideal design for a two-level AR(1) model: the
autoregressive coefficient ``rho`` is identified directly from co-located
LF/HF pairs.  ``EXTRA_LF_SAMPLES`` further low-fidelity atmospheres are drawn
from the 10k pool so the low-fidelity process is also constrained away from the
HF design.

Rather than fitting all 195 wavelengths, ten channels are used, each sitting on
the absorption band of a different atmospheric species (see ``SPECIES_BANDS``).
Each channel is an independent scalar-output multi-fidelity GP regression
problem in the 9 atmospheric inputs, and all three arms are handed identical
data and identical starting values on each.

Why three arms
--------------
    sklearn          exact likelihood, scipy L-BFGS-B
    gpboost_exact    exact likelihood, GPBoost L-BFGS
    gpboost_vecchia  Vecchia-approximated likelihood, GPBoost L-BFGS

sklearn vs gpboost_exact isolates the optimiser; gpboost_vecchia vs
gpboost_exact isolates what the Vecchia approximation costs at this sample
size.  Every arm uses GPBoost's built-in ``ar1_mf_matern`` covariance (matched
by ``Matern(nu=1.5)`` on the scikit-learn side) and is scored on the exact
likelihood with the same exact predictor.

Outputs
-------
results/kernel/<framework>/realdata_fits.csv   per-species fits, per arm
results/kernel/comparison/realdata_*.csv       paired comparison + summaries
results/kernel/comparison/realdata_*.png       figures
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "kernel"
COMPARISON_DIR = RESULTS_DIR / "comparison"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functools import partial

from kernel_comparison.gpboost_ar1 import (
    COV_FUNCTION,
    NUM_NEIGHBORS,
    fit_gpboost_ar1,
)
from kernel_comparison.sklearn_ar1 import (
    MATERIAL_TOLERANCE,
    MATERN_NU,
    PARAMETER_NAMES,
    TIE_TOLERANCE,
    AR1Params,
    ar1_predict,
    classify_winner,
    default_init,
    fit_sklearn_ar1,
    make_inits,
    prediction_metrics,
)


# ---------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------
FRAMEWORKS = (
    ("sklearn", fit_sklearn_ar1),
    ("gpboost_exact", partial(fit_gpboost_ar1, gp_approx="none")),
    ("gpboost_vecchia", partial(fit_gpboost_ar1, gp_approx="vecchia")),
)
FRAMEWORK_NAMES = tuple(name for name, _ in FRAMEWORKS)

# Ten channels, one per atmospheric species, at the band centre each species is
# usually diagnosed from in this 2.46 - 11.75 micron range.  Each is mapped to
# the nearest wavelength on the 195-point grid.
SPECIES_BANDS = {
    "H2O": 2.70,
    "HCN": 3.00,
    "CH4": 3.31,
    "H2S": 3.85,
    "CO2": 4.26,
    "CO": 4.67,
    "SO2": 7.35,
    "C2H2": 7.53,
    "PH3": 10.08,
    "NH3": 10.50,
}

N_HF_TRAIN = 67           # of the 97 matched atmospheres; the rest are held out
EXTRA_LF_SAMPLES = 150    # additional LF atmospheres from the 10k pool
N_INITS = 3               # shared multi-start budget, identical for every arm
RANDOM_STATE = 42


def main():
    for name in FRAMEWORK_NAMES:
        (RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    data = load_matched_data()
    channels = select_species_channels(data["wavelengths"])
    channels.to_csv(COMPARISON_DIR / "realdata_species_channels.csv", index=False)

    print(
        f"Real-data study: {len(channels)} species channels, "
        f"{N_HF_TRAIN} HF training atmospheres + "
        f"{len(data['x_hf']) - N_HF_TRAIN} held out, "
        f"{len(data['x_hf']) + EXTRA_LF_SAMPLES} LF atmospheres, "
        f"{data['x_hf'].shape[1]} inputs, {N_INITS} shared starting values."
    )
    print(
        f"Covariance: GPBoost built-in {COV_FUNCTION} (cov_fct_shape={MATERN_NU}), "
        f"matched by Matern(nu={MATERN_NU}) in scikit-learn; "
        f"Vecchia uses {NUM_NEIGHBORS} neighbours."
    )
    print("Arms:", ", ".join(FRAMEWORK_NAMES))
    print(channels.to_string(index=False))
    print()

    records = []
    prediction_records = []
    for row in channels.itertuples():
        dataset = build_channel_dataset(data, row.response_index)
        inits = make_inits(
            default_init(dataset["y_train"]),
            n_inits=N_INITS,
            seed=RANDOM_STATE + row.response_index,
        )
        channel_records = []
        for _, fit_function in FRAMEWORKS:
            fit = fit_function(
                dataset["X_train"], dataset["f_train"], dataset["y_train"], inits
            )
            metrics, predictions = evaluate_on_test(fit.params, dataset)
            prediction_records.append(
                pd.DataFrame(
                    {
                        "species": row.species,
                        "wavelength": row.wavelength,
                        "framework": fit.framework,
                        **predictions,
                    }
                )
            )
            channel_records.append(
                {
                    "species": row.species,
                    "wavelength": row.wavelength,
                    "response_index": row.response_index,
                    "framework": fit.framework,
                    "neg_log_likelihood": fit.neg_log_likelihood,
                    "native_neg_log_likelihood": fit.native_neg_log_likelihood,
                    # GPBoost's own objective minus the exact one at its
                    # solution: ~0 for the exact arms, the Vecchia error for
                    # the Vecchia arm.
                    "objective_approximation_gap": (
                        fit.native_neg_log_likelihood - fit.neg_log_likelihood
                    ),
                    "seconds": fit.seconds,
                    "n_inits": fit.n_inits,
                    "n_iterations": fit.n_iterations,
                    "n_params_at_bound": fit.extra["n_params_at_bound"],
                    **metrics,
                    **fit.params.to_dict(),
                }
            )
        records.extend(channel_records)
        report_channel(channel_records)

    fits = pd.DataFrame(records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    for framework, frame in fits.groupby("framework"):
        path = RESULTS_DIR / framework / "realdata_fits.csv"
        frame.to_csv(path, index=False)
        print(f"\nWrote {path}")
    for framework, frame in predictions.groupby("framework"):
        path = RESULTS_DIR / framework / "realdata_predictions.csv"
        frame.to_csv(path, index=False)
        print(f"Wrote {path}")
    predictions.to_csv(COMPARISON_DIR / "realdata_actual_vs_predicted.csv", index=False)

    paired = build_paired_table(fits)
    parameters = build_parameter_table(fits)
    summary = build_summary_table(fits, paired)

    paired.to_csv(COMPARISON_DIR / "realdata_paired_nll.csv", index=False)
    parameters.to_csv(COMPARISON_DIR / "realdata_parameters.csv", index=False)
    summary.to_csv(COMPARISON_DIR / "realdata_summary.csv", index=False)
    (COMPARISON_DIR / "realdata_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "frameworks": list(FRAMEWORK_NAMES),
                    "species_bands": SPECIES_BANDS,
                    "n_hf_train": N_HF_TRAIN,
                    "n_hf_test": int(len(data["x_hf"]) - N_HF_TRAIN),
                    "extra_lf_samples": EXTRA_LF_SAMPLES,
                    "n_inits": N_INITS,
                    "random_state": RANDOM_STATE,
                    "gpboost_cov_function": COV_FUNCTION,
                    "cov_fct_shape": MATERN_NU,
                    "gpboost_num_neighbors": NUM_NEIGHBORS,
                },
                "summary": summary.to_dict(orient="records"),
                "per_species": paired.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_objective_and_accuracy(fits, paired)
    plot_fitted_parameters(fits)
    plot_actual_vs_predicted(predictions)

    print_report(paired, parameters, summary)
    print(f"\nWrote comparison outputs to {COMPARISON_DIR}")


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def load_matched_data():
    """Load the 97 matched LF/HF atmospheres plus the extra LF pool.

    ``YLF.csv`` holds the low-fidelity spectra at the *same* 97 input rows as
    ``XHF.csv`` / ``YHF.csv``; ``XLF10k.csv`` / ``YLF10k.csv`` hold the wider
    low-fidelity pool.
    """
    x_hf = pd.read_csv(DATA_DIR / "XHF.csv", skipinitialspace=True)
    x_hf.columns = x_hf.columns.str.strip()
    input_names = list(x_hf.columns)

    y_hf = pd.read_csv(DATA_DIR / "YHF.csv", index_col=0)
    wavelengths = y_hf.columns.astype(float).to_numpy()
    y_lf_matched = pd.read_csv(DATA_DIR / "YLF.csv")

    x_lf_pool = pd.read_csv(DATA_DIR / "XLF10k.csv", skipinitialspace=True)
    x_lf_pool.columns = x_lf_pool.columns.str.strip()
    if list(x_lf_pool.columns) != input_names:
        x_lf_pool = pd.read_csv(
            DATA_DIR / "XLF10k.csv", header=None, names=input_names, skipinitialspace=True
        )
    y_lf_pool = pd.read_csv(DATA_DIR / "YLF10k.csv")

    if not (len(x_hf) == len(y_hf) == len(y_lf_matched)):
        raise ValueError("XHF / YHF / YLF must all have the same number of rows.")

    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(x_hf))
    extra_rows = rng.choice(len(x_lf_pool), size=EXTRA_LF_SAMPLES, replace=False)

    return {
        "x_hf": x_hf.to_numpy(dtype=float),
        "y_hf": y_hf.to_numpy(dtype=float),
        "y_lf_matched": y_lf_matched.to_numpy(dtype=float),
        "x_lf_extra": x_lf_pool.to_numpy(dtype=float)[extra_rows],
        "y_lf_extra": y_lf_pool.to_numpy(dtype=float)[extra_rows],
        "hf_train_rows": order[:N_HF_TRAIN],
        "hf_test_rows": order[N_HF_TRAIN:],
        "wavelengths": wavelengths,
        "input_names": input_names,
    }


def select_species_channels(wavelengths):
    """Map each species band centre to the nearest wavelength on the grid."""
    rows = []
    for species, band in SPECIES_BANDS.items():
        index = int(np.argmin(np.abs(wavelengths - band)))
        rows.append(
            {
                "species": species,
                "band_centre": band,
                "wavelength": float(wavelengths[index]),
                "response_index": index,
            }
        )
    channels = pd.DataFrame(rows)
    if channels["response_index"].duplicated().any():
        raise ValueError(
            "Two species bands map to the same wavelength channel; "
            "adjust SPECIES_BANDS."
        )
    return channels.sort_values("wavelength").reset_index(drop=True)


def build_channel_dataset(data, response_index):
    """Assemble one channel's LF/HF training set and held-out HF test set.

    Inputs are standardised (the nine atmospheric parameters differ by orders
    of magnitude, e.g. Tint ~ 300 vs Rp ~ 1, so a shared length-scale prior is
    meaningless without it).  Responses are centred per fidelity and scaled by
    a single common factor, because all arms are zero-mean; the LF and HF
    means differ, and an AR(1) model on uncentred spectra would spend ``rho``
    on the offset.  Predictions are mapped back to the original units before
    the metrics are computed.
    """
    x_lf = np.vstack([data["x_hf"], data["x_lf_extra"]])
    y_lf = np.concatenate(
        [
            data["y_lf_matched"][:, response_index],
            data["y_lf_extra"][:, response_index],
        ]
    )
    x_hf_train = data["x_hf"][data["hf_train_rows"]]
    y_hf_train = data["y_hf"][data["hf_train_rows"], response_index]
    x_hf_test = data["x_hf"][data["hf_test_rows"]]
    y_hf_test = data["y_hf"][data["hf_test_rows"], response_index]

    input_mean = x_lf.mean(axis=0)
    input_std = x_lf.std(axis=0)
    input_std[input_std == 0.0] = 1.0
    standardise = lambda values: (values - input_mean) / input_std

    low_mean = float(y_lf.mean())
    high_mean = float(y_hf_train.mean())
    scale = float(y_lf.std())
    if scale == 0.0:
        raise ValueError(f"Channel {response_index} has a constant LF response.")

    return {
        "X_train": np.vstack([standardise(x_lf), standardise(x_hf_train)]),
        "f_train": np.concatenate([np.zeros(len(x_lf)), np.ones(len(x_hf_train))]),
        "y_train": np.concatenate(
            [(y_lf - low_mean) / scale, (y_hf_train - high_mean) / scale]
        ),
        "X_test": standardise(x_hf_test),
        "f_test": np.ones(len(x_hf_test)),
        "y_test": y_hf_test,
        "hf_test_rows": data["hf_test_rows"],
        "high_mean": high_mean,
        "scale": scale,
    }


def evaluate_on_test(params: AR1Params, dataset):
    """Held-out HF metrics in the original spectral units."""
    mean, _, observation_var = ar1_predict(
        params,
        dataset["X_train"],
        dataset["f_train"],
        dataset["y_train"],
        dataset["X_test"],
        dataset["f_test"],
    )
    # Metrics on the standardised scale first (NLPD is easier to read there,
    # and it is the scale every arm actually fitted on).
    y_test_standardised = (dataset["y_test"] - dataset["high_mean"]) / dataset["scale"]
    standardised = prediction_metrics(y_test_standardised, mean, observation_var)

    mean_original = mean * dataset["scale"] + dataset["high_mean"]
    residuals = mean_original - dataset["y_test"]
    predictions = {
        "hf_row": dataset["hf_test_rows"],
        "actual": dataset["y_test"],
        "predicted": mean_original,
        "predicted_sd": np.sqrt(observation_var) * dataset["scale"],
        "residual": residuals,
    }
    metrics = {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "rmse_ppm": float(np.sqrt(np.mean(residuals**2)) * 1e6),
        "mae_ppm": float(np.mean(np.abs(residuals)) * 1e6),
        "nrmse": standardised["nrmse"],
        "nlpd_standardised": standardised["nlpd"],
        "coverage_95": standardised["coverage_95"],
    }
    return metrics, predictions


# ---------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------
def build_paired_table(fits):
    """One row per species, every arm side by side."""
    values = [
        "neg_log_likelihood",
        "seconds",
        "nrmse",
        "rmse_ppm",
        "nlpd_standardised",
        "rho",
        "n_params_at_bound",
        "objective_approximation_gap",
    ]
    wide = fits.pivot(index="species", columns="framework", values=values)

    paired = pd.DataFrame({"species": wide.index})
    for column in values:
        for framework in FRAMEWORK_NAMES:
            paired[f"{column}_{framework}"] = wide[(column, framework)].to_numpy()
    paired["wavelength"] = (
        fits.groupby("species")["wavelength"].first().reindex(wide.index).to_numpy()
    )

    nll_columns = [f"neg_log_likelihood_{name}" for name in FRAMEWORK_NAMES]
    best = paired[nll_columns].min(axis=1)
    paired["best_neg_log_likelihood"] = best
    for framework in FRAMEWORK_NAMES:
        paired[f"nll_gap_{framework}"] = (
            paired[f"neg_log_likelihood_{framework}"] - best
        )

    objectives = [
        {name: row[f"neg_log_likelihood_{name}"] for name in FRAMEWORK_NAMES}
        for _, row in paired.iterrows()
    ]
    paired["winner"] = [classify_winner(v, TIE_TOLERANCE) for v in objectives]
    paired["material_winner"] = [
        classify_winner(v, MATERIAL_TOLERANCE) for v in objectives
    ]
    # The two questions the three arms exist to separate.
    paired["optimiser_difference"] = (
        paired["neg_log_likelihood_sklearn"]
        - paired["neg_log_likelihood_gpboost_exact"]
    )
    paired["vecchia_cost"] = (
        paired["neg_log_likelihood_gpboost_vecchia"]
        - paired["neg_log_likelihood_gpboost_exact"]
    )
    # Keyed by species so it cannot drift out of step with the pivot order.
    paired["max_log_param_spread"] = paired["species"].map(parameter_spread(fits))
    return paired.sort_values("wavelength").reset_index(drop=True)


def parameter_spread(fits):
    """Largest log(max/min) across arms over the positive parameters, per species.

    Arms can reach the same likelihood at very different parameter values when
    the likelihood has a flat ridge; this quantifies that, which the NLL
    comparison alone cannot.
    """
    names = [name for name in PARAMETER_NAMES if name != "rho"]
    values = {}
    for species, frame in fits.groupby("species"):
        matrix = frame.set_index("framework").reindex(list(FRAMEWORK_NAMES))[names]
        matrix = np.clip(matrix.to_numpy(dtype=float), 1e-12, None)
        values[species] = float(
            np.max(np.log(matrix.max(axis=0) / matrix.min(axis=0)))
        )
    return pd.Series(values, name="max_log_param_spread")


def build_parameter_table(fits):
    """Long table of every fitted parameter, every arm, per species."""
    long = fits.melt(
        id_vars=["species", "wavelength", "framework"],
        value_vars=list(PARAMETER_NAMES),
        var_name="parameter",
        value_name="value",
    )
    wide = long.pivot(
        index=["species", "wavelength", "parameter"],
        columns="framework",
        values="value",
    ).reset_index()
    wide.columns.name = None

    matrix = np.clip(wide[list(FRAMEWORK_NAMES)].to_numpy(dtype=float), 1e-12, None)
    wide["log_spread"] = np.log(matrix.max(axis=1) / matrix.min(axis=1))
    # rho is signed, so a log ratio is meaningless for it.
    wide.loc[wide["parameter"] == "rho", "log_spread"] = np.nan
    return wide.sort_values(["wavelength", "parameter"]).reset_index(drop=True)


def build_summary_table(fits, paired):
    rows = []
    for framework in FRAMEWORK_NAMES:
        frame = fits[fits["framework"] == framework]
        gap = paired[f"nll_gap_{framework}"]
        rows.append(
            {
                "framework": framework,
                "median_neg_log_likelihood": float(frame["neg_log_likelihood"].median()),
                "median_nll_gap_to_best": float(gap.median()),
                "max_nll_gap_to_best": float(gap.max()),
                "n_wins": int((paired["winner"] == framework).sum()),
                "n_material_wins": int((paired["material_winner"] == framework).sum()),
                "n_ties": int((paired["winner"] == "tie").sum()),
                "n_species": int(len(paired)),
                "median_seconds": float(frame["seconds"].median()),
                "total_seconds": float(frame["seconds"].sum()),
                "median_objective_approximation_gap": float(
                    frame["objective_approximation_gap"].median()
                ),
                "median_nrmse": float(frame["nrmse"].median()),
                "mean_nrmse": float(frame["nrmse"].mean()),
                "median_rmse_ppm": float(frame["rmse_ppm"].median()),
                "median_nlpd_standardised": float(frame["nlpd_standardised"].median()),
                "median_coverage_95": float(frame["coverage_95"].median()),
                "median_rho": float(frame["rho"].median()),
                "median_params_at_bound": float(frame["n_params_at_bound"].median()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
# sklearn and gpboost_exact routinely agree to ~1e-9, so whichever is drawn
# first is hidden underneath the other.  Each arm therefore gets its own marker
# and linestyle, and is drawn thick-to-thin so coincident series stay visible.
FRAMEWORK_STYLE = {
    "sklearn": dict(
        color="tab:blue", marker="o", linestyle="-", linewidth=3.2, markersize=11
    ),
    "gpboost_exact": dict(
        color="tab:orange", marker="s", linestyle="--", linewidth=1.8, markersize=6.5
    ),
    "gpboost_vecchia": dict(
        color="tab:green", marker="^", linestyle=":", linewidth=1.4, markersize=5
    ),
}
FRAMEWORK_COLORS = {name: style["color"] for name, style in FRAMEWORK_STYLE.items()}


def framework_offsets(width=0.26):
    """Evenly spaced offsets, centred on the tick."""
    n = len(FRAMEWORK_NAMES)
    return {
        name: (index - (n - 1) / 2) * width
        for index, name in enumerate(FRAMEWORK_NAMES)
    }, width


def scatter_style(framework, shrink=1.0):
    """Marker style for a scatter, sized so overlapping arms remain visible."""
    style = FRAMEWORK_STYLE[framework]
    return dict(
        color=style["color"],
        marker=style["marker"],
        s=(style["markersize"] * shrink) ** 2,
    )


def line_style(framework):
    """Line style for a series, drawn thick-to-thin across the arms."""
    style = FRAMEWORK_STYLE[framework]
    return dict(
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        markersize=style["markersize"] * 0.7,
    )


def plot_objective_and_accuracy(fits, paired):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    species = paired["species"].to_numpy()
    positions = np.arange(len(species))
    offsets, width = framework_offsets()

    ax = axes[0, 0]
    # A bar cannot show a gap of exactly zero, which is precisely the winning
    # arm - hence stems with a marker at the tip, visible at y=0.
    for framework in FRAMEWORK_NAMES:
        x = positions + offsets[framework]
        gaps = paired[f"nll_gap_{framework}"].to_numpy()
        style = FRAMEWORK_STYLE[framework]
        ax.vlines(x, 0.0, gaps, color=style["color"], linewidth=width * 6, alpha=0.45)
        ax.plot(
            x,
            gaps,
            linestyle="none",
            marker=style["marker"],
            markersize=style["markersize"] * 0.8,
            color=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=framework,
        )
    ax.axhline(TIE_TOLERANCE, color="grey", linestyle=":", linewidth=1.0)
    ax.set_yscale("symlog", linthresh=TIE_TOLERANCE)
    ax.set_xticks(positions)
    ax.set_xticklabels(species, rotation=45, ha="right")
    ax.set_ylabel("NLL - best NLL of the three (nats)")
    ax.set_title("Objective reached per species (marker on 0 = won; symlog)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for framework in FRAMEWORK_NAMES:
        ax.bar(
            positions + offsets[framework],
            paired[f"nrmse_{framework}"],
            width=width * 0.9,
            color=FRAMEWORK_COLORS[framework],
            alpha=0.85,
            label=framework,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(species, rotation=45, ha="right")
    ax.set_ylabel("held-out HF NRMSE")
    ax.set_title("Held-out prediction per species (shared predictor)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for framework in FRAMEWORK_NAMES:
        ax.plot(
            paired["wavelength"],
            paired[f"rho_{framework}"],
            alpha=0.85,
            label=framework,
            **line_style(framework),
        )
    for wavelength, name, value in zip(
        paired["wavelength"], species, paired["rho_sklearn"]
    ):
        ax.annotate(
            name,
            (wavelength, value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("wavelength (micron)")
    ax.set_ylabel("fitted rho")
    ax.set_title("AR(1) coefficient: how far HF is a rescaled LF")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for framework in FRAMEWORK_NAMES:
        frame = fits[fits["framework"] == framework]
        ax.scatter(
            frame["seconds"],
            frame["nrmse"],
            alpha=0.85,
            label=framework,
            **scatter_style(framework, shrink=0.9),
        )
    ax.set_xscale("log")
    ax.set_xlabel(f"seconds for {N_INITS} starts (log scale)")
    ax.set_ylabel("held-out HF NRMSE")
    ax.set_title("Cost vs accuracy, one point per species")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = COMPARISON_DIR / "realdata_objective_and_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_fitted_parameters(fits):
    """The fitted covariance parameters across the spectrum, one line per arm.

    With a single shared length scale per block there is no ARD relevance map
    to draw, so what is informative instead is how the four positive parameters
    move with wavelength and whether the three arms agree on them.
    """
    panels = [
        ("low_var", "low-fidelity variance", True),
        ("discrepancy_var", "discrepancy variance", True),
        ("low_length_scale", "low-fidelity length scale", True),
        ("discrepancy_length_scale", "discrepancy length scale", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    for ax, (column, title, log_scale) in zip(axes.ravel(), panels):
        for framework in FRAMEWORK_NAMES:
            frame = fits[fits["framework"] == framework].sort_values("wavelength")
            ax.plot(
                frame["wavelength"],
                frame[column],
                alpha=0.85,
                label=framework,
                **line_style(framework),
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(column)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("wavelength (micron)")
    axes[0, 0].legend(fontsize=8)

    fig.suptitle(
        "Fitted ar1_mf_matern parameters per species channel "
        "(log scale; agreement between arms means the parameter is identified)"
    )
    fig.tight_layout()
    path = COMPARISON_DIR / "realdata_fitted_parameters.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_actual_vs_predicted(predictions):
    """Held-out HF spectra: actual vs predicted, one panel per arm.

    There is no ground truth for the covariance parameters on real data, so the
    actual-vs-predicted view here is of the response, coloured by species.
    """
    species_order = list(dict.fromkeys(predictions["species"]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(species_order), 2)))
    color_by_species = dict(zip(species_order, colors))

    fig, axes = plt.subplots(
        1, len(FRAMEWORK_NAMES), figsize=(5.2 * len(FRAMEWORK_NAMES), 5.2),
        sharex=True, sharey=True,
    )
    scale = 1e6  # transit depths are ~1e-4; plot in ppm
    limits = [
        predictions["actual"].min() * scale,
        predictions["actual"].max() * scale,
    ]
    for ax, framework in zip(np.atleast_1d(axes), FRAMEWORK_NAMES):
        frame = predictions[predictions["framework"] == framework]
        for species in species_order:
            rows = frame[frame["species"] == species]
            ax.scatter(
                rows["actual"] * scale,
                rows["predicted"] * scale,
                s=22,
                alpha=0.8,
                color=color_by_species[species],
                label=species,
            )
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.0)
        rmse = float(np.sqrt(np.mean(frame["residual"] ** 2))) * scale
        ax.set_title(f"{framework}\nRMSE {rmse:.1f} ppm")
        ax.set_xlabel("actual transit depth (ppm)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("predicted transit depth (ppm)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, ncol=2, title="species")

    fig.suptitle(
        "Held-out high-fidelity spectra: actual vs predicted "
        f"({len(species_order)} species channels)"
    )
    fig.tight_layout()
    path = COMPARISON_DIR / "realdata_actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------
def report_channel(channel_records):
    by_framework = {record["framework"]: record for record in channel_records}
    objectives = {
        name: by_framework[name]["neg_log_likelihood"] for name in FRAMEWORK_NAMES
    }
    leader = classify_winner(objectives, TIE_TOLERANCE)
    first = channel_records[0]
    nll_text = "  ".join(
        f"{name.split('_')[-1]} {objectives[name]:9.3f}" for name in FRAMEWORK_NAMES
    )
    nrmse_text = "/".join(
        f"{by_framework[name]['nrmse']:.3f}" for name in FRAMEWORK_NAMES
    )
    seconds_text = "/".join(
        f"{by_framework[name]['seconds']:.0f}s" for name in FRAMEWORK_NAMES
    )
    print(
        f"  {first['species']:<5} {first['wavelength']:6.3f} um  {nll_text}  "
        f"-> {leader:<16} | NRMSE {nrmse_text} | {seconds_text}"
    )


def print_report(paired, parameters, summary):
    print("\n" + "=" * 100)
    print("REAL-DATA SUMMARY")
    print("=" * 100)

    for label, column, tolerance in (
        ("Objective, round-off tolerance", "winner", TIE_TOLERANCE),
        ("Objective, material differences only", "material_winner", MATERIAL_TOLERANCE),
    ):
        counts = paired[column].value_counts()
        print(
            f"\n{label} (best two within {tolerance:g} nats counted as a tie):\n  "
            + ", ".join(
                f"{name}: {int(counts.get(name, 0))}/{len(paired)}"
                for name in FRAMEWORK_NAMES + ("tie",)
            )
        )

    print("\nThe two effects the three arms separate (median over species):")
    print(
        f"  optimiser  sklearn - gpboost_exact    "
        f"{paired['optimiser_difference'].median():+.6f} nats"
    )
    print(
        f"  approx.    gpboost_vecchia - exact    "
        f"{paired['vecchia_cost'].median():+.6f} nats"
    )

    print("\nPer-framework summary:")
    print(
        summary[
            [
                "framework",
                "median_neg_log_likelihood",
                "median_nll_gap_to_best",
                "max_nll_gap_to_best",
                "n_wins",
                "n_material_wins",
                "median_seconds",
                "total_seconds",
                "median_objective_approximation_gap",
                "median_nrmse",
                "median_nlpd_standardised",
                "median_rho",
                "median_params_at_bound",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )

    print("\nPer species (NLL gap to the best arm):")
    print(
        paired[
            ["species", "wavelength"]
            + [f"nll_gap_{name}" for name in FRAMEWORK_NAMES]
            + [f"nrmse_{name}" for name in FRAMEWORK_NAMES]
            + ["max_log_param_spread", "winner"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )

    print("\nFitted rho per species:")
    print(
        paired[["species", "wavelength"] + [f"rho_{name}" for name in FRAMEWORK_NAMES]]
        .to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )

    print(
        "\nParameter agreement: even where the arms reach the same likelihood, "
        "the largest log(max/min) across arms over the positive parameters is "
        f"{paired['max_log_param_spread'].median():.3g} (median) and "
        f"{paired['max_log_param_spread'].max():.3g} (worst species)."
        "\nLarge values with a tied likelihood mean the likelihood has a flat "
        "ridge, so the individual parameters are not identified even though the "
        "fit is."
    )

    largest = parameters.reindex(
        parameters["log_spread"].abs().sort_values(ascending=False).index
    ).head(8)
    print("\nLargest per-parameter disagreements:")
    print(
        largest[["species", "parameter"] + list(FRAMEWORK_NAMES) + ["log_spread"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )


if __name__ == "__main__":
    main()
