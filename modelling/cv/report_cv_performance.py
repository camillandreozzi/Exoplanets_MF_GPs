"""Recompute manuscript statistics and LaTeX from the full 10,000-LF LOO run.

NRMSE_j = RMSE_i / std_i(y_ij, ddof=0). Table entries average NRMSE_j.
Paired fold scores use sqrt(mean_j((error_ij / std_i(y_ij))**2)), preserving
wavelength normalization across designs. These differ from spectrum NRMSE,
which divides a spectrum's RMSE by its own standard deviation.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr, ttest_rel, wilcoxon

from modelling.cv.recompute_cv_outputs import reference_predictions

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/cv/loo_lf10000"
OUTPUT = RESULTS / "manuscript"
ORDER = ["model1-sf", "model3-sf", "cn", "model1-mf", "model3-mf"]
LABELS = {"model1-sf": "Single-fidelity GP", "model3-sf": "Single-fidelity GPBoost",
          "cn": "PCA–PCE", "model1-mf": "Model 1 MFGP", "model3-mf": "Model 3 GPBoost"}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(RESULTS / "loo_predictions.csv")
    reference = reference_predictions(predictions)
    combined = pd.concat([predictions, reference], ignore_index=True)
    combined["predictor"] = np.where(combined.model.eq("cn"), "cn", combined.model + "-" + combined.variant)
    truth = pd.read_csv(ROOT / "data/YHF.csv", index_col=0)
    y = truth.to_numpy()
    wavelengths = truth.columns.astype(float).to_numpy()
    observed = pd.read_csv(ROOT / "data/Observed_Spectra.csv")
    np.testing.assert_allclose(observed.wavelength, wavelengths)
    sigma_obs = observed[["measured_err_lo", "measured_err_hi"]].mean(axis=1).to_numpy()
    sigma_bar = float(sigma_obs.mean())
    x = pd.read_csv(ROOT / "data/XHF.csv", skipinitialspace=True)
    x.columns = x.columns.str.strip()
    y_std = y.std(axis=0, ddof=0)
    if y.shape != (97, 195) or (y_std <= 0).any():
        raise ValueError("Expected 97 complete designs, 195 nonconstant wavelength channels")
    bands = {"all": np.ones(len(wavelengths), dtype=bool),
             "F322W2": wavelengths < 4,
             "F444W": (wavelengths >= 4) & (wavelengths < 5),
             "MIRI": wavelengths >= 5}
    table_rows, wavelength_rows, fold_rows = [], [], []
    errors = {}
    for name in ORDER:
        group = combined[combined.predictor.eq(name)]
        if group.duplicated(["fold", "response_index"]).any():
            raise ValueError(f"Duplicate predictions for {name}")
        target = group.pivot(index="fold", columns="response_index", values="y_true")
        np.testing.assert_allclose(target, y, atol=1e-14)
        error = group.pivot(index="fold", columns="response_index", values="residual").to_numpy()
        if not np.isfinite(error).all():
            raise ValueError(f"Incomplete predictions for {name}")
        errors[name] = error
        nrmse = np.sqrt(np.mean(error**2, axis=0)) / y_std
        mae = np.mean(np.abs(error), axis=0)
        row = dict(predictor=name, architecture=LABELS[name])
        row.update({band: float(nrmse[mask].mean()) for band, mask in bands.items()})
        row.update(mae_ppm=float(mae.mean() * 1e6), mae_over_sigma=float(mae.mean() / sigma_bar),
                   global_nrmse=float(np.sqrt(np.mean(error**2)) / y.std(ddof=0)))
        table_rows.append(row)
        wavelength_rows.append(pd.DataFrame(dict(predictor=name, wavelength=wavelengths,
            y_true_std=y_std, nrmse=nrmse, mae_ppm=mae * 1e6,
            rmse_ppm=np.sqrt(np.mean(error**2, axis=0)) * 1e6)))
        fold_rows.append(pd.DataFrame(dict(predictor=name, fold=np.arange(1, 98),
            standardized_fold_rmse=np.sqrt(np.mean((error / y_std)**2, axis=1)),
            spectrum_nrmse=np.sqrt(np.mean(error**2, axis=1)) / y.std(axis=1, ddof=0),
            relative_l2=np.linalg.norm(error, axis=1) / np.linalg.norm(y, axis=1))))
    table = pd.DataFrame(table_rows).set_index("predictor")
    wave_metrics = pd.concat(wavelength_rows, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    table.to_csv(OUTPUT / "cv_metrics.csv")
    wave_metrics.to_csv(OUTPUT / "cv_per_wavelength.csv", index=False)
    folds.to_csv(OUTPUT / "cv_per_design.csv", index=False)
    scores = folds.pivot(index="fold", columns="predictor", values="standardized_fold_rmse")
    pairs = [("model1-mf", "model3-mf"), ("cn", "model1-mf"),
             ("cn", "model3-mf"), ("model1-sf", "model3-mf")]
    paired = []
    for baseline, candidate in pairs:
        difference = scores[baseline] - scores[candidate]
        paired.append(dict(baseline=baseline, candidate=candidate,
            candidate_wins=int((difference > 0).sum()), n_designs=len(difference),
            mean_baseline_minus_candidate=float(difference.mean()),
            sd_difference=float(difference.std(ddof=1)),
            wilcoxon_p=float(wilcoxon(difference, alternative="two-sided").pvalue),
            paired_t_p=float(ttest_rel(scores[baseline], scores[candidate]).pvalue)))
    paired = pd.DataFrame(paired)
    # Holm adjustment for precisely the four stated comparisons.
    order = np.argsort(paired.wilcoxon_p.to_numpy())
    adjusted = np.minimum(1., np.maximum.accumulate(paired.wilcoxon_p.to_numpy()[order] * np.arange(4, 0, -1)))
    paired.loc[order, "wilcoxon_p_holm"] = adjusted
    paired.to_csv(OUTPUT / "paired_comparisons.csv", index=False)
    best = errors["model3-mf"]
    mae = np.abs(best).mean(axis=0)
    rmse = np.sqrt(np.mean(best**2, axis=0))
    relative_l2 = np.linalg.norm(best, axis=1) / np.linalg.norm(y, axis=1)
    correlations = pd.DataFrame([dict(input=column, rho=float(spearmanr(x[column], relative_l2).statistic),
        p=float(spearmanr(x[column], relative_l2).pvalue)) for column in x])
    correlations.to_csv(OUTPUT / "input_correlations.csv", index=False)
    squared_error = np.sum(best**2, axis=0)
    worst = np.argsort(squared_error)[::-1]
    contributions = pd.DataFrame(dict(wavelength=wavelengths[worst],
        squared_error_fraction=squared_error[worst] / squared_error.sum()))
    contributions.to_csv(OUTPUT / "squared_error_contributions.csv", index=False)
    top8 = np.argsort(relative_l2)[-8:]
    nrmse = rmse / y_std
    stats = dict(n_hf=len(y), n_wavelengths=len(wavelengths),
        normalization="Population std (ddof=0) of held-out reference values at each wavelength",
        table_aggregation="Arithmetic mean of per-wavelength NRMSE (not global pooled NRMSE)",
        fold_score="sqrt(mean_wavelength((residual / wavelength_reference_std)**2))",
        paired_test="Two-sided Wilcoxon signed-rank; Holm adjustment across the four stated comparisons",
        reference_alignment="YSF_LOO.csv row order assumed identical to YHF.csv; no reference sample IDs available",
        sigma_obs_ppm=sigma_bar * 1e6,
        sigma_obs_definition="Arithmetic mean of lower and upper error bars in each bin, then mean across bins",
        instrument_bins={k: int(v.sum()) for k, v in bands.items()},
        improvement_over_reference_percent=100 * (1 - table.loc["model3-mf", "all"] / table.loc["cn", "all"]),
        model3_mae_ppm=float(mae.mean() * 1e6),
        model3_mae_over_sigma=float(mae.mean() / sigma_bar),
        bins_mae_below_sigma=int((mae < sigma_obs).sum()),
        bins_rmse_below_sigma=int((rmse < sigma_obs).sum()),
        nircam_mean_nrmse=float(nrmse[wavelengths < 5].mean()),
        miri_mean_nrmse=float(nrmse[wavelengths >= 5].mean()),
        below3_mae_ppm=float(mae[wavelengths < 3].mean() * 1e6),
        above10_mae_ppm=float(mae[wavelengths > 10].mean() * 1e6),
        below3_hf_mean_ppm=float(y[:, wavelengths < 3].mean() * 1e6),
        above10_hf_mean_ppm=float(y[:, wavelengths > 10].mean() * 1e6),
        hf_amplitude_ratio=float(y[:, wavelengths > 10].mean() / y[:, wavelengths < 3].mean()),
        squared_error_percent={str(n): float(100 * squared_error[worst[:n]].sum() / squared_error.sum()) for n in [5, 10, 20]},
        worst5_wavelength_min=float(wavelengths[worst[:5]].min()),
        worst5_wavelength_max=float(wavelengths[worst[:5]].max()),
        peak_nrmse=float(nrmse.max()), peak_nrmse_wavelength=float(wavelengths[nrmse.argmax()]),
        relative_l2_median=float(np.median(relative_l2)), relative_l2_mean=float(relative_l2.mean()),
        relative_l2_skewness=float(skew(relative_l2, bias=True)), relative_l2_max=float(relative_l2.max()),
        oxygen_rho=float(correlations.loc[correlations.input.eq("[O/H]"), "rho"].iloc[0]),
        oxygen_p=float(correlations.loc[correlations.input.eq("[O/H]"), "p"].iloc[0]),
        worst8_oxygen_standard_deviations=float(((x["[O/H]"] - x["[O/H]"].mean()) / x["[O/H]"].std(ddof=0)).iloc[top8].mean()))
    save_json(OUTPUT / "supporting_statistics.json", stats)
    make_figure(wave_metrics)
    write_latex(table, paired, stats)
    print("\nManuscript metrics (mean wavelength NRMSE):\n" + table.to_string())
    print("\nPaired wavelength-standardized fold scores:\n" + paired.to_string(index=False))
    print("\nSupporting statistics:\n" + json.dumps(stats, indent=2))
    print(f"\nSaved manuscript outputs to {OUTPUT}")


def make_figure(metrics):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    colors = {"model1-sf": "#4477AA", "model3-sf": "#228833", "cn": "#AA3377",
              "model1-mf": "#EE7733", "model3-mf": "#007755"}
    for name in ORDER:
        series = metrics[metrics.predictor.eq(name)]
        for ax, column in zip(axes, ["nrmse", "mae_ppm"]):
            ax.plot(series.wavelength, series[column], color=colors[name],
                    linestyle="--" if name.endswith("sf") else "-", label=LABELS[name])
    axes[0].set_ylabel("NRMSE (RMSE / reference std)")
    axes[1].set_ylabel("MAE (ppm)")
    axes[1].set_xlabel("Wavelength (μm)")
    axes[0].legend(ncol=2, frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Leave-one-out performance: 97 HF designs, 195 wavelengths")
    for suffix in ["png", "pdf"]:
        fig.savefig(OUTPUT / f"cv_performance.{suffix}", dpi=200)
    plt.close(fig)


def write_latex(table, paired, stats):
    template = (Path(__file__).with_name("cv_results_template.tex")).read_text()
    values = {}
    for name, row in table.iterrows():
        for column in ["all", "F322W2", "F444W", "MIRI", "mae_ppm", "mae_over_sigma"]:
            digits = 1 if column == "mae_ppm" else 3 if column == "mae_over_sigma" else 4
            cell = f"{row[column]:.{digits}f}"
            values[f"{name}_{column}_plain"] = cell
            if row[column] == table[column].min():
                cell = r"\textbf{" + cell + "}"
            values[f"{name}_{column}"] = cell
    for key, value in stats.items():
        if isinstance(value, float):
            values[key] = f"{value:.4f}"
        elif isinstance(value, int):
            values[key] = str(value)
    formats = {"improvement_over_reference_percent": 1, "model3_mae_ppm": 1,
        "model3_mae_over_sigma": 3, "sigma_obs_ppm": 1, "below3_mae_ppm": 1,
        "above10_mae_ppm": 1, "below3_hf_mean_ppm": 1, "above10_hf_mean_ppm": 1,
        "hf_amplitude_ratio": 1, "peak_nrmse_wavelength": 4,
        "relative_l2_skewness": 2, "relative_l2_max": 3,
        "worst8_oxygen_standard_deviations": 2, "oxygen_rho": 3}
    for key, digits in formats.items():
        values[key] = f"{stats[key]:.{digits}f}"
    for n, percent in stats["squared_error_percent"].items():
        values[f"sse{n}"] = f"{percent:.1f}"
    for index, row in paired.iterrows():
        values[f"pair{index}_wins"] = str(row.candidate_wins)
        values[f"pair{index}_mean"] = f"{abs(row.mean_baseline_minus_candidate):.4f}"
        values[f"pair{index}_sd"] = f"{row.sd_difference:.4f}"
        for key in ["wilcoxon_p", "wilcoxon_p_holm"]:
            value = row[key]
            if value < .001:
                mantissa, exponent = f"{value:.2e}".split("e")
                values[f"pair{index}_{key}"] = mantissa + r"\times10^{" + str(int(exponent)) + "}"
            else:
                values[f"pair{index}_{key}"] = f"{value:.4f}"
    for key, value in values.items():
        template = template.replace("@@" + key + "@@", value)
    if "@@" in template:
        raise ValueError("Unfilled LaTeX template field")
    (OUTPUT / "cross_validated_surrogate_performance.tex").write_text(template)


if __name__ == "__main__":
    main()
