"""Reverse the input-sensitivity labels: spectral channels Y -> parameters X.

This mirrors ``research/exploratory/03_input_sensitivity.py`` on the same
10,000-point low-fidelity design, but uses the 195 spectral values as predictors
and the nine physical parameters as outputs.

Two versions of each aggregate are reported:

* literal/raw reproduces the original variance weighting after swapping X/Y;
* scale-balanced standardizes the parameter outputs or weights them equally.

The distinction matters because the nine parameters have different units. The
spectral predictors are also strongly correlated, so the binned indices are
inverse univariate associations, not an additive Sobol decomposition.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from exoplanets_mf.data import load_XLF_10k, load_YLF_10k
from exoplanets_mf.instruments import instrument_mode_masks
from exoplanets_mf.paths import REVERSE_LABEL_RESULTS_DIR
from exoplanets_mf.reproducibility import RANDOM_SEED

RESULTS_DIR = REVERSE_LABEL_RESULTS_DIR

N_BINS = 20
N_PCA_KEEP = 9
RF_SUBSAMPLE = 4000
RF_N_ESTIMATORS = 200
TOP_N = 20


def pca(matrix: np.ndarray, *, standardize: bool = False):
    """Return eigenvalues, loadings, and scores for centered row observations."""
    centered = matrix - matrix.mean(axis=0)
    if standardize:
        scale = centered.std(axis=0, ddof=1)
        if np.any(scale == 0):
            raise ValueError("PCA cannot standardize a constant output column")
        centered = centered / scale

    U, singular_values, components = np.linalg.svd(
        centered, full_matrices=False
    )
    eigenvalues = singular_values**2 / (len(matrix) - 1)
    scores = U * singular_values
    return eigenvalues, components, scores


def main_effect_per_output(
    predictors: np.ndarray,
    outputs: np.ndarray,
    *,
    n_bins: int = N_BINS,
):
    """Estimate Var(E[output | predictor bin]) / Var(output).

    The returned array has shape ``(n_predictors, n_outputs)``. Quantile bins
    are collapsed when repeated predictor values create duplicate edges.
    """
    n_rows, n_predictors = predictors.shape
    n_outputs = outputs.shape[1]
    output_variance = outputs.var(axis=0)
    safe_variance = np.where(output_variance > 0, output_variance, np.nan)
    grand_mean = outputs.mean(axis=0)
    associations = np.full((n_predictors, n_outputs), np.nan)

    probabilities = np.linspace(0, 1, n_bins + 1)
    for predictor_index in range(n_predictors):
        edges = np.unique(
            np.quantile(predictors[:, predictor_index], probabilities)
        )
        if len(edges) < 2:
            continue

        bin_index = np.searchsorted(
            edges[1:-1],
            predictors[:, predictor_index],
            side="right",
        )
        actual_bins = len(edges) - 1
        counts = np.bincount(bin_index, minlength=actual_bins).astype(float)
        sums = np.zeros((actual_bins, n_outputs))
        np.add.at(sums, bin_index, outputs)
        bin_means = sums / counts[:, None]
        conditional_variance = (
            counts[:, None] * (bin_means - grand_mean) ** 2
        ).sum(axis=0) / n_rows
        associations[predictor_index] = conditional_variance / safe_variance

    return associations, output_variance


def variance_weighted_average(
    associations: np.ndarray,
    output_variance: np.ndarray,
) -> np.ndarray:
    """Aggregate outputs exactly as the original sensitivity workflow does."""
    return np.nansum(
        associations * output_variance[None, :], axis=1
    ) / np.nansum(output_variance)


def rf_pca_importance(
    predictors: np.ndarray,
    scores: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    n_keep: int = N_PCA_KEEP,
    seed: int = RANDOM_SEED,
):
    """Eigenvalue-weight random-forest impurity importance across target PCs."""
    keep = min(n_keep, scores.shape[1])
    rng = np.random.default_rng(seed)
    subset = rng.choice(
        len(predictors),
        size=min(RF_SUBSAMPLE, len(predictors)),
        replace=False,
    )

    per_component = np.zeros((keep, predictors.shape[1]))
    for component_index in range(keep):
        forest = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            n_jobs=-1,
            random_state=seed,
        )
        forest.fit(
            predictors[subset],
            scores[subset, component_index],
        )
        per_component[component_index] = forest.feature_importances_

    weights = eigenvalues[:keep] / eigenvalues[:keep].sum()
    return (weights[:, None] * per_component).sum(axis=0), per_component


def plot_pca(
    raw_eigenvalues: np.ndarray,
    standardized_eigenvalues: np.ndarray,
    savepath,
) -> None:
    """Compare literal raw-scale PCA with scale-balanced PCA."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    components = np.arange(1, len(raw_eigenvalues) + 1)

    for row, (label, eigenvalues) in enumerate(
        (
            ("Raw parameter units", raw_eigenvalues),
            ("Standardized parameters", standardized_eigenvalues),
        )
    ):
        fraction = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(fraction)
        axes[row, 0].bar(components, fraction, color="tab:blue")
        axes[row, 0].set_ylabel("variance fraction")
        axes[row, 0].set_title(f"{label}: scree")
        axes[row, 1].plot(components, cumulative, "o-", color="tab:blue")
        axes[row, 1].axhline(
            0.99, ls="--", color="tab:red", lw=1, label="99%"
        )
        axes[row, 1].set_ylim(0, 1.02)
        axes[row, 1].set_ylabel("cumulative variance")
        axes[row, 1].set_title(f"{label}: cumulative")
        axes[row, 1].legend()

    for axis in axes[-1]:
        axis.set_xlabel("principal component")
        axis.set_xticks(components)
    fig.suptitle("Reverse-label PCA of physical-parameter outputs XLF_10k")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_pca_loadings(
    parameter_names: list[str],
    raw_components: np.ndarray,
    standardized_components: np.ndarray,
    savepath,
) -> None:
    """Show which physical parameters define each reverse-output PC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, title, components in (
        (axes[0], "Raw parameter units", raw_components),
        (axes[1], "Standardized parameters", standardized_components),
    ):
        image = axis.imshow(
            components,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axis.set_xticks(np.arange(len(parameter_names)))
        axis.set_xticklabels(parameter_names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(parameter_names)))
        axis.set_yticklabels(
            [f"PC{index}" for index in range(1, len(parameter_names) + 1)]
        )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("PCA loadings of reverse-label parameter outputs")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def _shade_instrument_modes(axis, wavelengths: np.ndarray) -> None:
    for mode, _ in instrument_mode_masks(wavelengths):
        axis.axvspan(
            max(wavelengths.min(), mode.wavelength_min),
            min(wavelengths.max(), mode.wavelength_max),
            color=mode.color,
            alpha=0.55,
            zorder=0,
        )


def plot_sensitivity_by_wavelength(
    wavelengths: np.ndarray,
    raw_first_order: np.ndarray,
    balanced_first_order: np.ndarray,
    raw_rf: np.ndarray,
    standardized_rf: np.ndarray,
    savepath,
) -> None:
    """Plot every spectral predictor in wavelength order."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = (
        (
            axes[0, 0],
            raw_first_order,
            "First-order association: raw variance weighted",
            "tab:blue",
        ),
        (
            axes[0, 1],
            raw_rf,
            "RF importance: raw target PCA weighted",
            "tab:green",
        ),
        (
            axes[1, 0],
            balanced_first_order,
            "First-order association: equal parameter weight",
            "tab:blue",
        ),
        (
            axes[1, 1],
            standardized_rf,
            "RF importance: standardized target PCA weighted",
            "tab:green",
        ),
    )
    for axis, values, title, color in panels:
        _shade_instrument_modes(axis, wavelengths)
        axis.plot(wavelengths, values, color=color, lw=1.4)
        axis.set_title(title)
        axis.set_ylabel("aggregate value")
        axis.grid(alpha=0.2)

    for axis in axes[-1]:
        axis.set_xlabel(r"spectral predictor wavelength $\lambda$ [$\mu$m]")
    fig.suptitle("Reverse-label spectrum Y -> parameter X associations")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_rankings(
    wavelengths: np.ndarray,
    balanced_first_order: np.ndarray,
    standardized_rf: np.ndarray,
    savepath,
    *,
    top_n: int = TOP_N,
) -> None:
    """Rank the strongest scale-balanced spectral channels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    panels = (
        (
            axes[0],
            balanced_first_order,
            "Equal-parameter first-order association",
            "tab:blue",
        ),
        (
            axes[1],
            standardized_rf,
            "Standardized-target RF importance",
            "tab:green",
        ),
    )
    y_positions = np.arange(min(top_n, len(wavelengths)))
    for axis, values, title, color in panels:
        order = np.argsort(values)[::-1][:top_n]
        axis.barh(y_positions, values[order], color=color)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(
            [f"{wavelengths[index]:.4g} µm" for index in order]
        )
        axis.invert_yaxis()
        axis.set_xlabel("aggregate value")
        axis.set_title(title)

    fig.suptitle("Top reverse-label spectral predictors")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_heatmap(
    wavelengths: np.ndarray,
    parameter_names: list[str],
    associations: np.ndarray,
    savepath,
) -> None:
    """Heatmap of each wavelength's first-order association with each output."""
    finite_values = associations[np.isfinite(associations)]
    vmax = np.percentile(finite_values, 99) if len(finite_values) else 1
    wavelength_edges = np.empty(len(wavelengths) + 1)
    wavelength_edges[1:-1] = 0.5 * (wavelengths[:-1] + wavelengths[1:])
    wavelength_edges[0] = wavelengths[0] - 0.5 * (
        wavelengths[1] - wavelengths[0]
    )
    wavelength_edges[-1] = wavelengths[-1] + 0.5 * (
        wavelengths[-1] - wavelengths[-2]
    )
    fig, axis = plt.subplots(figsize=(14, 6))
    image = axis.pcolormesh(
        wavelength_edges,
        np.arange(len(parameter_names) + 1),
        associations.T,
        cmap="magma",
        vmin=0,
        vmax=vmax,
        shading="flat",
    )
    axis.set_yticks(np.arange(len(parameter_names)) + 0.5)
    axis.set_yticklabels(parameter_names)
    axis.set_xlabel(r"spectral predictor wavelength $\lambda$ [$\mu$m]")
    axis.set_title(
        "Reverse first-order association by parameter "
        r"$\mathrm{Var}(\mathbb{E}[X_j\mid Y_\lambda])/\mathrm{Var}(X_j)$"
    )
    fig.colorbar(image, ax=axis, label="first-order association")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def save_tables(
    wavelengths: np.ndarray,
    parameter_names: list[str],
    associations: np.ndarray,
    raw_first_order: np.ndarray,
    balanced_first_order: np.ndarray,
    raw_rf: np.ndarray,
    standardized_rf: np.ndarray,
    raw_eigenvalues: np.ndarray,
    standardized_eigenvalues: np.ndarray,
    raw_components: np.ndarray,
    standardized_components: np.ndarray,
) -> None:
    """Write the plotted values as reusable CSV files."""
    sensitivity = pd.DataFrame(
        {
            "wavelength_um": wavelengths,
            "first_order_raw_variance_weighted": raw_first_order,
            "first_order_equal_parameter_weighted": balanced_first_order,
            "rf_raw_target_pca_weighted": raw_rf,
            "rf_standardized_target_pca_weighted": standardized_rf,
        }
    )
    for index, parameter_name in enumerate(parameter_names):
        sensitivity[f"first_order_{parameter_name}"] = associations[:, index]
    sensitivity.to_csv(
        RESULTS_DIR / "00_sensitivity_by_wavelength.csv",
        index=False,
    )

    pca_rows = []
    for scaling, eigenvalues in (
        ("raw", raw_eigenvalues),
        ("standardized", standardized_eigenvalues),
    ):
        fraction = eigenvalues / eigenvalues.sum()
        for index, (eigenvalue, explained, cumulative) in enumerate(
            zip(eigenvalues, fraction, np.cumsum(fraction), strict=True),
            start=1,
        ):
            pca_rows.append(
                {
                    "scaling": scaling,
                    "component": index,
                    "eigenvalue": eigenvalue,
                    "explained_variance_fraction": explained,
                    "cumulative_variance_fraction": cumulative,
                }
            )
    pd.DataFrame(pca_rows).to_csv(
        RESULTS_DIR / "00_pca_explained_variance.csv",
        index=False,
    )

    loading_rows = []
    for scaling, components in (
        ("raw", raw_components),
        ("standardized", standardized_components),
    ):
        for component_index, component in enumerate(components, start=1):
            for parameter_name, loading in zip(
                parameter_names, component, strict=True
            ):
                loading_rows.append(
                    {
                        "scaling": scaling,
                        "component": component_index,
                        "parameter": parameter_name,
                        "loading": loading,
                    }
                )
    pd.DataFrame(loading_rows).to_csv(
        RESULTS_DIR / "00_pca_loadings.csv",
        index=False,
    )


def _top_wavelengths(
    wavelengths: np.ndarray,
    values: np.ndarray,
    *,
    count: int = 10,
) -> list[dict[str, float]]:
    order = np.argsort(values)[::-1][:count]
    return [
        {
            "wavelength_um": float(wavelengths[index]),
            "value": float(values[index]),
        }
        for index in order
    ]


def save_summary(
    wavelengths: np.ndarray,
    raw_eigenvalues: np.ndarray,
    standardized_eigenvalues: np.ndarray,
    raw_first_order: np.ndarray,
    balanced_first_order: np.ndarray,
    raw_rf: np.ndarray,
    standardized_rf: np.ndarray,
) -> None:
    """Write concise headline results and interpretation warnings."""
    raw_fraction = raw_eigenvalues / raw_eigenvalues.sum()
    standardized_fraction = (
        standardized_eigenvalues / standardized_eigenvalues.sum()
    )
    summary = {
        "pca": {
            "raw_pc1_fraction": float(raw_fraction[0]),
            "raw_components_for_99_percent": int(
                np.searchsorted(np.cumsum(raw_fraction), 0.99) + 1
            ),
            "standardized_pc1_fraction": float(standardized_fraction[0]),
            "standardized_components_for_99_percent": int(
                np.searchsorted(
                    np.cumsum(standardized_fraction), 0.99
                ) + 1
            ),
        },
        "top_wavelengths": {
            "first_order_raw_variance_weighted": _top_wavelengths(
                wavelengths, raw_first_order
            ),
            "first_order_equal_parameter_weighted": _top_wavelengths(
                wavelengths, balanced_first_order
            ),
            "rf_raw_target_pca_weighted": _top_wavelengths(
                wavelengths, raw_rf
            ),
            "rf_standardized_target_pca_weighted": _top_wavelengths(
                wavelengths, standardized_rf
            ),
        },
        "interpretation": [
            "Raw target PCA and variance weighting are sensitive to parameter units.",
            (
                "Spectral predictors are correlated, so reverse first-order "
                "values are not an additive Sobol decomposition."
            ),
            (
                "Random-forest impurity importance can distribute credit "
                "among correlated neighbouring wavelengths."
            ),
            (
                "Reverse associations measure inverse predictive information, "
                "not physical causation."
            ),
        ],
    }
    (RESULTS_DIR / "00_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    parameter_frame = load_XLF_10k()
    spectrum_frame = load_YLF_10k()
    parameters = parameter_frame.to_numpy(dtype=float)
    spectra = spectrum_frame.to_numpy(dtype=float)
    wavelengths = spectrum_frame.columns.to_numpy(dtype=float)
    parameter_names = list(parameter_frame.columns)

    if len(parameters) != len(spectra):
        raise ValueError("XLF_10k and YLF_10k must remain row aligned")
    print(
        f"reverse predictors Y {spectra.shape}, "
        f"reverse outputs X {parameters.shape}"
    )

    raw_eigenvalues, raw_components, raw_scores = pca(parameters)
    (
        standardized_eigenvalues,
        standardized_components,
        standardized_scores,
    ) = pca(parameters, standardize=True)

    raw_fraction = raw_eigenvalues / raw_eigenvalues.sum()
    standardized_fraction = (
        standardized_eigenvalues / standardized_eigenvalues.sum()
    )
    print(
        "\nPCA:"
        f"\n  raw PC1 = {raw_fraction[0]:.4f}; "
        f"99% needs "
        f"{np.searchsorted(np.cumsum(raw_fraction), 0.99) + 1} PC(s)"
        f"\n  standardized PC1 = {standardized_fraction[0]:.4f}; "
        f"99% needs "
        f"{np.searchsorted(np.cumsum(standardized_fraction), 0.99) + 1} PCs"
    )

    associations, parameter_variance = main_effect_per_output(
        spectra, parameters
    )
    raw_first_order = variance_weighted_average(
        associations, parameter_variance
    )
    balanced_first_order = np.nanmean(associations, axis=1)

    print("\nFitting RF importance for raw target PCs...", flush=True)
    raw_rf, _ = rf_pca_importance(
        spectra, raw_scores, raw_eigenvalues
    )
    print("Fitting RF importance for standardized target PCs...", flush=True)
    standardized_rf, _ = rf_pca_importance(
        spectra, standardized_scores, standardized_eigenvalues
    )

    plot_pca(
        raw_eigenvalues,
        standardized_eigenvalues,
        RESULTS_DIR / "00_pca_scree.png",
    )
    plot_pca_loadings(
        parameter_names,
        raw_components,
        standardized_components,
        RESULTS_DIR / "00_pca_loadings.png",
    )
    plot_sensitivity_by_wavelength(
        wavelengths,
        raw_first_order,
        balanced_first_order,
        raw_rf,
        standardized_rf,
        RESULTS_DIR / "00_sensitivity_by_wavelength.png",
    )
    plot_rankings(
        wavelengths,
        balanced_first_order,
        standardized_rf,
        RESULTS_DIR / "00_sensitivity_ranking.png",
    )
    plot_heatmap(
        wavelengths,
        parameter_names,
        associations,
        RESULTS_DIR / "00_sensitivity_heatmap.png",
    )
    save_tables(
        wavelengths,
        parameter_names,
        associations,
        raw_first_order,
        balanced_first_order,
        raw_rf,
        standardized_rf,
        raw_eigenvalues,
        standardized_eigenvalues,
        raw_components,
        standardized_components,
    )
    save_summary(
        wavelengths,
        raw_eigenvalues,
        standardized_eigenvalues,
        raw_first_order,
        balanced_first_order,
        raw_rf,
        standardized_rf,
    )

    print("\nTop scale-balanced first-order wavelengths:")
    for item in _top_wavelengths(
        wavelengths, balanced_first_order, count=10
    ):
        print(f"  {item['wavelength_um']:7.4f} um  {item['value']:.4f}")
    print("\nTop standardized-target RF wavelengths:")
    for item in _top_wavelengths(
        wavelengths, standardized_rf, count=10
    ):
        print(f"  {item['wavelength_um']:7.4f} um  {item['value']:.4f}")
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
