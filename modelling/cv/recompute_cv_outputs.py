"""Regenerate existing LOO metric tables and figures from saved predictions.

Run with python3 -m modelling.cv.recompute_cv_outputs. No model fitting is
needed when only the error normalization changes. Raw predictions are retained.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelling.cv import cv_5fold
from modelling.plot import plot_cv_loo_results as cv_plot
from modelling.plot import plot_loo_model1_mf_vs_cn as cn_plot
from modelling.plot import plot_model1_sf_vs_mf_nrmse as model1_plot
from modelling.plot import plot_loo_model1_model2_sf_vs_mf as models_plot
from modelling.plot import plot_loo_model1_model2_sf_vs_mf_per_sample as sample_plot
from modelling.plot import plot_model1_cn_model3_shared_wavelengths as shared_plot

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def results_directory(module, directory):
    previous = module.RESULTS_DIR
    module.RESULTS_DIR = directory
    try:
        yield
    finally:
        module.RESULTS_DIR = previous
        plt.close("all")


def reference_predictions(predictions):
    """Align the reference with HF rows; reference CSV has no sample IDs."""
    base = predictions.query("model == 'model1' and variant == 'mf'").copy()
    base = base.sort_values(["fold", "response_index"])
    truth = pd.read_csv(ROOT / "data/YHF.csv", index_col=0)
    reference = pd.read_csv(ROOT / "data/lit/YSF_LOO.csv")
    if reference.shape != truth.shape or len(base) != truth.size:
        raise ValueError("Reference and HF prediction dimensions differ")
    np.testing.assert_allclose(reference.columns.astype(float), truth.columns.astype(float))
    np.testing.assert_allclose(base.wavelength.to_numpy().reshape(truth.shape),
                               np.broadcast_to(truth.columns.astype(float), truth.shape))
    np.testing.assert_allclose(base.y_true.to_numpy().reshape(truth.shape), truth, atol=1e-14)
    base["model"], base["variant"] = "cn", "cn"
    base["y_pred"] = reference.to_numpy().ravel()
    base["residual"] = base.y_pred - base.y_true
    base["variance"] = np.nan
    return base


def per_sample_metrics(predictions):
    metrics = predictions.groupby(
        ["model", "variant", "held_out_source_index", "fold"], as_index=False
    ).agg(n_wavelengths=("wavelength", "size"),
          rmse=("residual", cv_5fold.rmse),
          mae=("residual", cv_5fold.mean_absolute),
          bias=("residual", "mean"),
          max_abs_error=("residual", cv_5fold.max_absolute),
          y_true_std=("y_true", lambda x: x.std(ddof=0)))
    metrics["nrmse"] = metrics.rmse / metrics.y_true_std.replace(0, np.nan)
    return metrics


def legacy_model1_samples(predictions, directory):
    data = predictions.query("model == 'model1'").assign(error=lambda x: x.residual)
    table = sample_plot.compute_per_sample(data).drop(columns="model")
    table = table.rename(columns={"nrmse_delta_mf_minus_sf": "delta_mf_minus_sf"})
    table["rmse_delta_mf_minus_sf"] = table.rmse_mf - table.rmse_sf
    table["mf_nrmse_percent_change_vs_sf"] = 100 * table.delta_mf_minus_sf / table.nrmse_sf
    table["winner"] = np.select([table.delta_mf_minus_sf < 0, table.delta_mf_minus_sf > 0],
                                 ["mf", "sf"], default="tie")
    output = directory / "model1_loo_sf_vs_mf_nrmse_per_sample.csv"
    table.to_csv(output, index=False)
    summary = dict(input_file=str((directory / "loo_predictions.csv").relative_to(ROOT)),
                   output_file=str(output.relative_to(ROOT)), n_samples=len(table),
                   n_complete_195_wavelength_samples=int((table.n_wavelengths_sf.eq(195) &
                                                           table.n_wavelengths_mf.eq(195)).sum()),
                   normalization="RMSE across wavelengths / std(reference spectrum, ddof=0)",
                   winner_counts=table.winner.value_counts().reindex(["mf", "sf", "tie"], fill_value=0).to_dict())
    for variant in ["sf", "mf"]:
        for statistic in ["mean", "median"]:
            summary[f"{statistic}_nrmse_{variant}"] = float(getattr(table[f"nrmse_{variant}"], statistic)())
    for statistic in ["mean", "median"]:
        summary[f"{statistic}_delta_mf_minus_sf"] = float(getattr(table.delta_mf_minus_sf, statistic)())
        summary[f"{statistic}_mf_percent_change_vs_sf"] = float(getattr(table.mf_nrmse_percent_change_vs_sf, statistic)())
    summary["best_mf_sample"] = table.loc[table.delta_mf_minus_sf.idxmin()].to_dict()
    summary["worst_mf_sample"] = table.loc[table.delta_mf_minus_sf.idxmax()].to_dict()
    output.with_name(output.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def shared_metrics(predictions, reference, directory):
    selection = predictions[predictions.model.isin(["model1", "model3"])]
    groups = [set(g.response_index) for _, g in selection.groupby(["model", "variant"])]
    indexes = set.intersection(*groups, set(reference.response_index))
    combined = pd.concat([selection, reference], ignore_index=True)
    combined = combined[combined.response_index.isin(indexes)].copy()
    combined["predictor"] = np.where(combined.model.eq("cn"), "cn", combined.model + "-" + combined.variant)
    wavelength = cv_5fold.per_wavelength_metrics(combined)
    samples = per_sample_metrics(combined)
    for table in [wavelength, samples]:
        table["predictor"] = np.where(table.model.eq("cn"), "cn", table.model + "-" + table.variant)
    prefix = f"model1_cn_model3_loo_shared{len(indexes)}wl"
    wavelength.to_csv(directory / f"{prefix}_per_wavelength.csv", index=False)
    samples.to_csv(directory / f"{prefix}_per_sample.csv", index=False)
    if len(indexes) == 28:
        with results_directory(shared_plot, directory):
            shared_plot.main()


def refresh(directory):
    predictions = pd.read_csv(directory / "loo_predictions.csv")
    keys = ["model", "variant", "fold", "response_index"]
    if predictions.duplicated(keys).any():
        raise ValueError(f"Duplicate predictions in {directory}")
    np.testing.assert_allclose(predictions.residual, predictions.y_pred - predictions.y_true, atol=1e-14)
    metrics = cv_5fold.per_wavelength_metrics(predictions)
    metrics.to_csv(directory / "loo_metrics_per_wavelength.csv", index=False)
    cv_5fold.nrmse_comparison(metrics).to_csv(directory / "loo_nrmse_per_wavelength.csv", index=False)
    for model, group in metrics.groupby("model"):
        if {"sf", "mf"}.issubset(set(group.variant)):
            cv_5fold.sf_mf_comparison(metrics, model).to_csv(
                directory / f"{model}_loo_sf_vs_mf_nrmse_per_wavelength.csv", index=False)
    summary = cv_5fold.overall_summary(metrics)
    summary.to_csv(directory / "loo_summary.csv", index=False)
    print(f"\n{directory.relative_to(ROOT)}\n{summary.to_string(index=False)}", flush=True)
    reference = reference_predictions(predictions)
    cv_5fold.per_wavelength_metrics(reference).to_csv(directory / "cn_loo_metrics_per_wavelength.csv", index=False)
    per_sample_metrics(reference).to_csv(directory / "cn_loo_per_sample_metrics.csv", index=False)
    per_sample_metrics(predictions).to_csv(directory / "loo_metrics_per_sample.csv", index=False)
    legacy_model1_samples(predictions, directory)
    for module in [cv_plot, model1_plot, cn_plot]:
        with results_directory(module, directory):
            module.main()
    if "model2" in set(predictions.model):
        for module in [models_plot, sample_plot]:
            with results_directory(module, directory):
                module.main()
    if (directory / "model1_cn_model3_loo_shared28wl_per_wavelength.csv").exists():
        shared_metrics(predictions, reference, directory)


def main():
    for prediction_file in sorted((ROOT / "results/cv").glob("*/loo_predictions.csv")):
        refresh(prediction_file.parent)


if __name__ == "__main__":
    main()
