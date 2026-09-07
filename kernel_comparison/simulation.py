"""Simulation study: scikit-learn vs GPBoost exact vs GPBoost Vecchia.

Data are drawn from the AR(1) multi-fidelity model itself,

    f_H(x) = rho * f_L(x) + delta(x),
    y      = f(x) + eps,   eps ~ N(0, noise_var)

using GPBoost's built-in ``ar1_mf_matern`` covariance (matched by
``Matern(nu=1.5)`` on the scikit-learn side), ``N_LF`` low-fidelity and
``N_HF`` high-fidelity observations on a nested design.  All three arms are handed identical data and
identical starting values.

Why three arms
--------------
    sklearn          exact likelihood, scipy L-BFGS-B
    gpboost_exact    exact likelihood, GPBoost L-BFGS
    gpboost_vecchia  Vecchia-approximated likelihood, GPBoost L-BFGS

Outputs
-------
results/kernel/<framework>/simulation_fits.csv   per-replicate fits, per arm
results/kernel/comparison/simulation_*.csv       paired comparison + summaries
results/kernel/comparison/simulation_*.png       figures
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os

RESULTS_DIR = PROJECT_ROOT / "kernel_comparison" / "results_kernel"
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
    gpboost_neg_log_likelihood,
)
from kernel_comparison.sklearn_ar1 import (
    MATERIAL_TOLERANCE,
    MATERN_NU,
    PARAMETER_NAMES,
    TIE_TOLERANCE,
    AR1Params,
    ar1_covariance,
    ar1_neg_log_likelihood,
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

N_DIMS = 3
N_LF = 100
N_HF = 20
N_TEST = 300
N_REPLICATES = 25
N_INITS = 4          # shared multi-start budget, identical for every arm
RANDOM_STATE = 42
NESTED_DESIGN = True  # HF points are a subset of the LF points

# True kernel parameters.  The discrepancy is smaller in amplitude and rougher
# (shorter length scale) than the low-fidelity process, which is the regime the
# AR(1) model is meant for.
TRUE_PARAMS = AR1Params(
    noise_var=0.01,
    low_var=1.0,
    low_length_scale=0.60,
    discrepancy_var=0.15,
    discrepancy_length_scale=0.35,
    rho=0.85,
)


def main():
    for name in FRAMEWORK_NAMES:
        (RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Simulation study: {N_REPLICATES} replicates, "
        f"{N_LF} LF + {N_HF} HF points in {N_DIMS}D, "
        f"{N_INITS} shared starting values per fit."
    )
    print(
        f"Covariance: GPBoost built-in {COV_FUNCTION} (cov_fct_shape={MATERN_NU}), "
        f"matched by Matern(nu={MATERN_NU}) in scikit-learn; "
        f"Vecchia uses {NUM_NEIGHBORS} neighbours."
    )
    print("Arms:", ", ".join(FRAMEWORK_NAMES))
    print("True parameters:", format_params(TRUE_PARAMS))

    check_objectives_agree()

    records = []
    prediction_records = []
    for replicate in range(N_REPLICATES):
        dataset = simulate_dataset(seed=RANDOM_STATE + replicate)
        inits = make_inits(
            default_init(dataset["y_train"]),
            n_inits=N_INITS,
            seed=RANDOM_STATE + 1000 * replicate,
        )
        nll_truth = ar1_neg_log_likelihood(
            TRUE_PARAMS, dataset["X_train"], dataset["f_train"], dataset["y_train"]
        )

        replicate_records = []
        for _, fit_function in FRAMEWORKS:
            fit = fit_function(
                dataset["X_train"], dataset["f_train"], dataset["y_train"], inits
            )
            metrics, predictions = evaluate_on_test(fit.params, dataset)
            prediction_records.append(
                pd.DataFrame(
                    {
                        "replicate": replicate,
                        "framework": fit.framework,
                        "test_point": np.arange(len(predictions["predicted"])),
                        **predictions,
                        "residual": predictions["predicted"]
                        - predictions["actual_latent"],
                    }
                )
            )
            replicate_records.append(
                {
                    "replicate": replicate,
                    "framework": fit.framework,
                    "neg_log_likelihood": fit.neg_log_likelihood,
                    "neg_log_likelihood_at_truth": nll_truth,
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
        records.extend(replicate_records)
        report_replicate(replicate_records)

    fits = pd.DataFrame(records)
    predictions = pd.concat(prediction_records, ignore_index=True)
    for framework, frame in fits.groupby("framework"):
        path = RESULTS_DIR / framework / "simulation_fits.csv"
        frame.to_csv(path, index=False)
        print(f"Wrote {path}")
    for framework, frame in predictions.groupby("framework"):
        path = RESULTS_DIR / framework / "simulation_predictions.csv"
        frame.to_csv(path, index=False)
        print(f"Wrote {path}")

    parameter_recovery = build_actual_vs_predicted_parameters(fits)
    parameter_recovery.to_csv(
        COMPARISON_DIR / "simulation_actual_vs_predicted_parameters.csv", index=False
    )
    predictions.to_csv(
        COMPARISON_DIR / "simulation_actual_vs_predicted.csv", index=False
    )

    paired = build_paired_table(fits)
    recovery = build_recovery_table(fits)
    summary = build_summary_table(fits, paired)

    paired.to_csv(COMPARISON_DIR / "simulation_paired_nll.csv", index=False)
    recovery.to_csv(COMPARISON_DIR / "simulation_parameter_recovery.csv", index=False)
    summary.to_csv(COMPARISON_DIR / "simulation_summary.csv", index=False)
    (COMPARISON_DIR / "simulation_summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "frameworks": list(FRAMEWORK_NAMES),
                    "n_dims": N_DIMS,
                    "n_lf": N_LF,
                    "n_hf": N_HF,
                    "n_test": N_TEST,
                    "n_replicates": N_REPLICATES,
                    "n_inits": N_INITS,
                    "nested_design": NESTED_DESIGN,
                    "random_state": RANDOM_STATE,
                    "gpboost_cov_function": COV_FUNCTION,
                    "cov_fct_shape": MATERN_NU,
                    "gpboost_num_neighbors": NUM_NEIGHBORS,
                },
                "true_params": TRUE_PARAMS.to_dict(),
                "summary": summary.to_dict(orient="records"),
                "verdict": verdict(paired, summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_parameter_recovery(fits)
    plot_optimisation_quality(fits, paired)
    plot_actual_vs_predicted_parameters(parameter_recovery)
    plot_actual_vs_predicted(predictions)

    print_report(paired, recovery, summary)
    print(f"\nWrote comparison outputs to {COMPARISON_DIR}")


# ---------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------
def simulate_dataset(seed):
    """Draw one LF/HF training set and a fresh HF test set from the truth."""
    rng = np.random.default_rng(seed)

    X_lf = rng.uniform(0.0, 1.0, size=(N_LF, N_DIMS))
    if NESTED_DESIGN:
        X_hf = X_lf[rng.choice(N_LF, size=N_HF, replace=False)]
    else:
        X_hf = rng.uniform(0.0, 1.0, size=(N_HF, N_DIMS))
    X_test = rng.uniform(0.0, 1.0, size=(N_TEST, N_DIMS))

    # Draw training and test latent values jointly so the test targets come
    # from the same realisation of the process.
    X_all = np.vstack([X_lf, X_hf, X_test])
    f_all = np.concatenate([np.zeros(N_LF), np.ones(N_HF + N_TEST)])

    covariance = ar1_covariance(TRUE_PARAMS, X_all, f_all)
    covariance[np.diag_indices_from(covariance)] += 1e-10
    latent = np.linalg.cholesky(covariance) @ rng.normal(size=len(X_all))

    noise = rng.normal(0.0, np.sqrt(TRUE_PARAMS.noise_var), size=len(X_all))
    observed = latent + noise
    n_train = N_LF + N_HF

    return {
        "X_train": X_all[:n_train],
        "f_train": f_all[:n_train],
        "y_train": observed[:n_train],
        "X_test": X_all[n_train:],
        "f_test": f_all[n_train:],
        "y_test": observed[n_train:],
        "latent_test": latent[n_train:],
    }


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------
def check_objectives_agree():
    """Assert GPBoost's exact likelihood is the reference likelihood.

    This is what licenses reading a lower NLL as a better optimum.  The Vecchia
    surface is then evaluated at the same parameters so the size of the
    approximation is on the record before any fitting happens.
    """
    dataset = simulate_dataset(seed=RANDOM_STATE)
    arguments = (
        TRUE_PARAMS, dataset["X_train"], dataset["f_train"], dataset["y_train"]
    )
    reference = ar1_neg_log_likelihood(*arguments)
    gpboost_exact = gpboost_neg_log_likelihood(*arguments, gp_approx="none")
    difference = abs(reference - gpboost_exact)
    print(
        f"\nObjective check at the true parameters: reference {reference:.10f} vs "
        f"GPBoost exact {gpboost_exact:.10f} (|diff| = {difference:.2e})"
    )
    if difference > 1e-5 * max(1.0, abs(reference)):
        raise AssertionError(
            "GPBoost and the reference AR(1) likelihood disagree; the "
            "parametrisation mapping in AR1Params.to_gpboost is wrong."
        )

    approximate = gpboost_neg_log_likelihood(
        *arguments, gp_approx="vecchia", num_neighbors=NUM_NEIGHBORS
    )
    print(
        f"Vecchia ({NUM_NEIGHBORS} neighbours) puts the same parameters at "
        f"{approximate:.10f}, i.e. {approximate - reference:+.4f} nats from exact.\n"
    )


def evaluate_on_test(params, dataset):
    """Held-out HF metrics and per-point predictions, via the shared predictor."""
    mean, _, observation_var = ar1_predict(
        params,
        dataset["X_train"],
        dataset["f_train"],
        dataset["y_train"],
        dataset["X_test"],
        dataset["f_test"],
    )
    metrics = prediction_metrics(
        dataset["y_test"], mean, observation_var, y_latent=dataset["latent_test"]
    )
    predictions = {
        "actual_latent": dataset["latent_test"],
        "actual_observed": dataset["y_test"],
        "predicted": mean,
        "predicted_sd": np.sqrt(observation_var),
    }
    return metrics, predictions


def parameter_columns():
    return list(PARAMETER_NAMES)


def build_paired_table(fits):
    """One row per replicate, every arm side by side."""
    values = [
        "neg_log_likelihood",
        "seconds",
        "rmse_latent",
        "nlpd",
        "objective_approximation_gap",
    ]
    wide = fits.pivot(index="replicate", columns="framework", values=values)

    paired = pd.DataFrame({"replicate": wide.index})
    for column in values:
        for framework in FRAMEWORK_NAMES:
            paired[f"{column}_{framework}"] = wide[(column, framework)].to_numpy()
    paired["nll_at_truth"] = (
        fits.groupby("replicate")["neg_log_likelihood_at_truth"]
        .first()
        .reindex(wide.index)
        .to_numpy()
    )

    nll_columns = [f"neg_log_likelihood_{name}" for name in FRAMEWORK_NAMES]
    best = paired[nll_columns].min(axis=1)
    paired["best_neg_log_likelihood"] = best
    for framework in FRAMEWORK_NAMES:
        paired[f"nll_gap_{framework}"] = (
            paired[f"neg_log_likelihood_{framework}"] - best
        )

    objectives = [
        {
            name: row[f"neg_log_likelihood_{name}"] for name in FRAMEWORK_NAMES
        }
        for _, row in paired.iterrows()
    ]
    paired["winner"] = [
        classify_winner(values, TIE_TOLERANCE) for values in objectives
    ]
    # A second, much coarser verdict: differences below MATERIAL_TOLERANCE nats
    # do not change any conclusion drawn from the fitted model.
    paired["material_winner"] = [
        classify_winner(values, MATERIAL_TOLERANCE) for values in objectives
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
    return paired


def build_actual_vs_predicted_parameters(fits):
    """Long table of true vs fitted value for every parameter, arm and replicate."""
    truth = dict(zip(parameter_columns(), TRUE_PARAMS.to_vector()))
    long = fits.melt(
        id_vars=["replicate", "framework"],
        value_vars=list(parameter_columns()),
        var_name="parameter",
        value_name="predicted",
    )
    long["actual"] = long["parameter"].map(truth)
    long["error"] = long["predicted"] - long["actual"]
    long["relative_error"] = long["error"] / long["actual"]
    # rho is signed, so a log ratio is only meaningful for the positive ones.
    positive = long["parameter"] != "rho"
    long["log_ratio"] = np.where(
        positive & (long["predicted"] > 0),
        np.log(np.clip(long["predicted"], 1e-300, None) / long["actual"]),
        np.nan,
    )
    return long.sort_values(["parameter", "framework", "replicate"]).reset_index(
        drop=True
    )


def build_recovery_table(fits):
    """Per-parameter recovery: bias, spread and |log ratio| against the truth."""
    truth = dict(zip(parameter_columns(), TRUE_PARAMS.to_vector()))
    rows = []
    for name in parameter_columns():
        true_value = truth[name]
        for framework, frame in fits.groupby("framework"):
            estimates = frame[name].to_numpy(dtype=float)
            row = {
                "parameter": name,
                "framework": framework,
                "true_value": true_value,
                "median_estimate": float(np.median(estimates)),
                "mean_estimate": float(np.mean(estimates)),
                "std_estimate": float(np.std(estimates, ddof=1)),
                "bias": float(np.mean(estimates) - true_value),
                "rmse": float(np.sqrt(np.mean((estimates - true_value) ** 2))),
            }
            # rho may be negative, so a log ratio is only meaningful for the
            # strictly positive variance / length-scale parameters.
            if name != "rho" and true_value > 0 and np.all(estimates > 0):
                log_ratio = np.log(estimates / true_value)
                row["median_abs_log_ratio"] = float(np.median(np.abs(log_ratio)))
                row["median_log_ratio"] = float(np.median(log_ratio))
            else:
                row["median_abs_log_ratio"] = np.nan
                row["median_log_ratio"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


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
                "mean_nll_gap_to_best": float(gap.mean()),
                "max_nll_gap_to_best": float(gap.max()),
                "n_wins": int((paired["winner"] == framework).sum()),
                "n_material_wins": int((paired["material_winner"] == framework).sum()),
                "n_ties": int((paired["winner"] == "tie").sum()),
                "n_material_ties": int((paired["material_winner"] == "tie").sum()),
                "n_replicates": int(len(paired)),
                "median_seconds": float(frame["seconds"].median()),
                "median_objective_approximation_gap": float(
                    frame["objective_approximation_gap"].median()
                ),
                "median_params_at_bound": float(frame["n_params_at_bound"].median()),
                "median_rmse_latent": float(frame["rmse_latent"].median()),
                "median_nrmse": float(frame["nrmse"].median()),
                "median_nlpd": float(frame["nlpd"].median()),
                "median_coverage_95": float(frame["coverage_95"].median()),
                "median_rho": float(frame["rho"].median()),
                "mean_abs_log_ratio_all_positive_params": float(
                    mean_abs_log_ratio(frame)
                ),
            }
        )
    return pd.DataFrame(rows)


def mean_abs_log_ratio(frame):
    """Average |log(estimate / truth)| over every strictly positive parameter."""
    truth = dict(zip(parameter_columns(), TRUE_PARAMS.to_vector()))
    values = []
    for name, true_value in truth.items():
        if name == "rho" or true_value <= 0:
            continue
        estimates = frame[name].to_numpy(dtype=float)
        values.append(np.abs(np.log(np.clip(estimates, 1e-12, None) / true_value)))
    return float(np.mean(np.concatenate(values)))


def verdict(paired, summary):
    indexed = summary.set_index("framework")
    return {
        "objective_wins": {
            name: int((paired["winner"] == name).sum())
            for name in FRAMEWORK_NAMES + ("tie",)
        },
        "material_objective_wins": {
            name: int((paired["material_winner"] == name).sum())
            for name in FRAMEWORK_NAMES + ("tie",)
        },
        "lower_median_nll_gap": indexed["median_nll_gap_to_best"].idxmin(),
        "median_nll_gap": {
            name: float(indexed.loc[name, "median_nll_gap_to_best"])
            for name in FRAMEWORK_NAMES
        },
        "median_seconds": {
            name: float(indexed.loc[name, "median_seconds"])
            for name in FRAMEWORK_NAMES
        },
        "median_rmse_latent": {
            name: float(indexed.loc[name, "median_rmse_latent"])
            for name in FRAMEWORK_NAMES
        },
        # The two isolated effects, in nats (positive = first arm worse).
        "median_optimiser_difference_sklearn_minus_gpboost_exact": float(
            paired["optimiser_difference"].median()
        ),
        "median_vecchia_cost_vs_gpboost_exact": float(paired["vecchia_cost"].median()),
    }


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


def plot_gap_stems(ax, positions, paired, offsets, width):
    """Gap-to-best per arm as stems with a marker at the tip.

    A bar cannot show a gap of exactly zero, which is precisely the winning
    arm - hence markers, which stay visible at y=0.
    """
    for framework in FRAMEWORK_NAMES:
        x = positions + offsets[framework]
        gaps = paired[f"nll_gap_{framework}"].to_numpy()
        style = FRAMEWORK_STYLE[framework]
        ax.vlines(
            x, 0.0, gaps, color=style["color"], linewidth=width * 6, alpha=0.45
        )
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


def plot_parameter_recovery(fits):
    """Estimate distributions per parameter, on a log-ratio scale vs the truth."""
    names = [name for name in parameter_columns() if name != "rho"]
    truth = dict(zip(parameter_columns(), TRUE_PARAMS.to_vector()))
    offsets, width = framework_offsets()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), width_ratios=(4, 1))

    positions = np.arange(len(names))
    for framework in FRAMEWORK_NAMES:
        frame = fits[fits["framework"] == framework]
        data = [
            np.log(np.clip(frame[name].to_numpy(dtype=float), 1e-12, None) / truth[name])
            for name in names
        ]
        box = axes[0].boxplot(
            data,
            positions=positions + offsets[framework],
            widths=width * 0.85,
            patch_artist=True,
            medianprops=dict(color="black"),
            flierprops=dict(markersize=3, alpha=0.6),
        )
        for patch in box["boxes"]:
            patch.set_facecolor(FRAMEWORK_COLORS[framework])
            patch.set_alpha(0.55)

    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("log(estimate / truth)")
    axes[0].set_title(
        f"Parameter recovery over {N_REPLICATES} replicates "
        f"({N_LF} LF + {N_HF} HF points)"
    )
    axes[0].legend(
        handles=[
            plt.Line2D([], [], color=FRAMEWORK_COLORS[name], linewidth=8, alpha=0.55,
                       label=name)
            for name in FRAMEWORK_NAMES
        ]
    )

    for framework in FRAMEWORK_NAMES:
        frame = fits[fits["framework"] == framework]
        box = axes[1].boxplot(
            [frame["rho"].to_numpy(dtype=float)],
            positions=[offsets[framework]],
            widths=width * 0.85,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        box["boxes"][0].set_facecolor(FRAMEWORK_COLORS[framework])
        box["boxes"][0].set_alpha(0.55)
    axes[1].axhline(TRUE_PARAMS.rho, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["rho"])
    axes[1].set_ylabel("estimate (natural scale)")
    axes[1].set_title("AR(1) coefficient")

    fig.tight_layout()
    path = COMPARISON_DIR / "simulation_parameter_recovery.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_optimisation_quality(fits, paired):
    """Objective gaps, the two isolated effects, cost and held-out accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    offsets, width = framework_offsets()
    replicates = paired["replicate"].to_numpy()

    ax = axes[0, 0]
    plot_gap_stems(ax, replicates, paired, offsets, width)
    ax.set_yscale("symlog", linthresh=TIE_TOLERANCE)
    ax.axhline(TIE_TOLERANCE, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xlabel("replicate")
    ax.set_ylabel("NLL - best NLL of the three (nats)")
    ax.set_title("Gap to the best optimum (marker on 0 = won; symlog scale)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.boxplot(
        [
            paired["optimiser_difference"].to_numpy(),
            paired["vecchia_cost"].to_numpy(),
        ],
        tick_labels=["sklearn -\ngpboost_exact", "gpboost_vecchia -\ngpboost_exact"],
        patch_artist=True,
        medianprops=dict(color="black"),
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("difference in exact NLL (nats)")
    ax.set_title("The two effects, separated\n(>0 = first arm worse)")

    ax = axes[1, 0]
    for framework in FRAMEWORK_NAMES:
        frame = fits[fits["framework"] == framework].sort_values("replicate")
        ax.scatter(
            frame["seconds"],
            paired[f"nll_gap_{framework}"],
            alpha=0.85,
            label=framework,
            **scatter_style(framework, shrink=0.8),
        )
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=TIE_TOLERANCE)
    ax.set_xlabel(f"seconds for {N_INITS} starts (log scale)")
    ax.set_ylabel("NLL gap to best")
    ax.set_title("Cost of the optimisation vs its quality")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for framework in FRAMEWORK_NAMES:
        frame = fits[fits["framework"] == framework]
        ax.scatter(
            frame["rmse_latent"],
            frame["nlpd"],
            alpha=0.85,
            label=framework,
            **scatter_style(framework, shrink=0.8),
        )
    ax.set_xlabel(f"held-out HF RMSE ({N_TEST} points, latent truth)")
    ax.set_ylabel("negative log predictive density")
    ax.set_title("Held-out prediction (shared predictor)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = COMPARISON_DIR / "simulation_optimisation_quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_actual_vs_predicted_parameters(parameter_recovery):
    """True vs fitted parameter value: everything should sit on the diagonal."""
    positive = parameter_recovery[parameter_recovery["parameter"] != "rho"]
    rho = parameter_recovery[parameter_recovery["parameter"] == "rho"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), width_ratios=(3, 2))

    ax = axes[0]
    for framework in FRAMEWORK_NAMES:
        frame = positive[positive["framework"] == framework]
        ax.scatter(
            frame["actual"],
            np.clip(frame["predicted"], 1e-8, None),
            alpha=0.6,
            label=framework,
            **scatter_style(framework, shrink=0.75),
        )
    limits = [
        0.5 * positive["actual"].min(),
        2.0 * max(positive["actual"].max(), positive["predicted"].max()),
    ]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("actual (true parameter)")
    ax.set_ylabel("predicted (fitted parameter)")
    ax.set_title(
        "Actual vs fitted covariance parameters\n"
        f"(variances and length scales, {N_REPLICATES} replicates)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for framework in FRAMEWORK_NAMES:
        frame = rho[rho["framework"] == framework]
        ax.scatter(
            frame["actual"] + np.random.default_rng(0).normal(0, 0.004, len(frame)),
            frame["predicted"],
            alpha=0.6,
            label=framework,
            **scatter_style(framework, shrink=0.75),
        )
    ax.axvline(TRUE_PARAMS.rho, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(TRUE_PARAMS.rho, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("actual rho (jittered for visibility)")
    ax.set_ylabel("predicted rho")
    ax.set_title("Actual vs fitted rho")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = COMPARISON_DIR / "simulation_actual_vs_predicted_parameters.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_actual_vs_predicted(predictions):
    """Held-out HF response: actual (latent truth) vs predicted, one panel per arm."""
    fig, axes = plt.subplots(
        1, len(FRAMEWORK_NAMES), figsize=(5 * len(FRAMEWORK_NAMES), 5),
        sharex=True, sharey=True,
    )
    limits = [
        predictions["actual_latent"].min(),
        predictions["actual_latent"].max(),
    ]
    for ax, framework in zip(np.atleast_1d(axes), FRAMEWORK_NAMES):
        frame = predictions[predictions["framework"] == framework]
        ax.scatter(
            frame["actual_latent"],
            frame["predicted"],
            s=8,
            alpha=0.25,
            color=FRAMEWORK_COLORS[framework],
        )
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.0)
        rmse = float(np.sqrt(np.mean(frame["residual"] ** 2)))
        ax.set_title(f"{framework}\nRMSE {rmse:.4f}")
        ax.set_xlabel("actual (latent HF truth)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("predicted")

    fig.suptitle(
        f"Held-out high-fidelity prediction, {N_TEST} points x "
        f"{N_REPLICATES} replicates"
    )
    fig.tight_layout()
    path = COMPARISON_DIR / "simulation_actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------
def format_params(params):
    return ", ".join(
        f"{name}={value:.4g}"
        for name, value in zip(PARAMETER_NAMES, params.to_vector())
    )


def report_replicate(replicate_records):
    by_framework = {record["framework"]: record for record in replicate_records}
    objectives = {
        name: by_framework[name]["neg_log_likelihood"] for name in FRAMEWORK_NAMES
    }
    leader = classify_winner(objectives, TIE_TOLERANCE)
    nll_text = "  ".join(
        f"{name.split('_')[-1]} {objectives[name]:9.3f}" for name in FRAMEWORK_NAMES
    )
    seconds_text = "/".join(
        f"{by_framework[name]['seconds']:.0f}s" for name in FRAMEWORK_NAMES
    )
    print(
        f"  replicate {replicate_records[0]['replicate']:>3}  {nll_text}  "
        f"(truth {replicate_records[0]['neg_log_likelihood_at_truth']:8.3f})  "
        f"-> {leader:<16} | {seconds_text}"
    )


def print_report(paired, recovery, summary):
    print("\n" + "=" * 96)
    print("SIMULATION SUMMARY")
    print("=" * 96)

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

    print("\nThe two effects the three arms separate (median over replicates):")
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
                "median_objective_approximation_gap",
                "median_rmse_latent",
                "median_nlpd",
                "median_params_at_bound",
                "mean_abs_log_ratio_all_positive_params",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4g}")
    )

    print("\nParameter recovery (median |log(estimate/truth)|, lower is better):")
    pivot = recovery.pivot(
        index="parameter", columns="framework", values="median_abs_log_ratio"
    ).reindex(parameter_columns())
    pivot = pivot[list(FRAMEWORK_NAMES)]
    pivot["true_value"] = [
        dict(zip(parameter_columns(), TRUE_PARAMS.to_vector()))[name]
        for name in pivot.index
    ]
    print(pivot.to_string(float_format=lambda value: f"{value:.4g}"))

    print("\nrho estimates (truth {:.3f}):".format(TRUE_PARAMS.rho))
    rho_rows = recovery[recovery["parameter"] == "rho"]
    print(
        rho_rows[["framework", "median_estimate", "bias", "rmse"]].to_string(
            index=False, float_format=lambda value: f"{value:.4g}"
        )
    )


if __name__ == "__main__":
    main()
