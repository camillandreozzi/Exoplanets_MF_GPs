"""Three interpolation diagnostics for the saved held-out-81 Model 2 fits."""

from pathlib import Path
import sys

import gpboost as gpb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_load import load_full_data
from src.model_2 import prepare_model2_data

HELD_OUT_HF_SAMPLE = 81
IN_SAMPLE_HF_SAMPLES = (0, 48, 96)
DENSE_HF_SAMPLE = IN_SAMPLE_HF_SAMPLES[0]
LF_SAMPLES = 1000
N_WAVELENGTHS = 50
RANDOM_STATE = 42
DENSE_GRID_SIZE = 500

MODEL_DIR = PROJECT_ROOT / "results" / "model2" / "leave_hf_81_out"
OUTPUT_DIR = PROJECT_ROOT / "results" / "model2" / "interpolation"
VARIANTS = {
    "SF": {"hf_only": True, "filename": "model2_SF.pkl", "color": "tab:blue"},
    "MF": {"hf_only": False, "filename": "model2_MF.pkl", "color": "tab:purple"},
}


def with_attrs(data, attrs):
    result = data.copy()
    result.attrs.update(attrs)
    return result


def load_models():
    models = {}
    for variant, settings in VARIANTS.items():
        path = MODEL_DIR / settings["filename"]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {variant} model: {path}\n"
                "Run python3 modelling/full_fit/full_fit_model2.py first."
            )
        models[variant] = gpb.GPModel(model_file=str(path))
    return models


def predict(model, data, hf_only):
    """Return GP response mean and variance for a Model 2 long-format frame."""
    x_pred = data.iloc[:, :-2].to_numpy(dtype=float)
    mf_coords = data.iloc[:, :-1].to_numpy(dtype=float)
    result = model.predict(
        gp_coords_pred=x_pred if hf_only else mf_coords,
        X_pred=x_pred,
        predict_response=True,
        predict_var=True,
    )
    mean = np.asarray(result["mu"], dtype=float)
    variance = np.maximum(np.asarray(result["var"], dtype=float), 0.0)
    return mean, variance


def predict_variants(models, data):
    return {
        variant: predict(model, data, VARIANTS[variant]["hf_only"])
        for variant, model in models.items()
    }


def reconstruct_training_masks(full_data, held_out_row):
    """Recreate the 50 wavelengths sampled for every fitted training row."""
    training_wide = with_attrs(full_data.drop(index=held_out_row), full_data.attrs)
    design = prepare_model2_data(
        training_wide,
        LF_number=LF_SAMPLES,
        n_wavelengths=N_WAVELENGTHS,
        random_state=RANDOM_STATE,
    )
    source_rows = design.attrs["source_dataframe_indices_by_row"]
    sampled = design.attrs["wavelength_indices_by_row"]
    return {
        source_row: np.asarray(sampled[row_id], dtype=int)
        for row_id, source_row in enumerate(source_rows)
    }


def load_experiment():
    full_data = load_full_data()
    wavelength_map = full_data.attrs["wavelength_map"]
    all_indices = np.asarray(sorted(wavelength_map), dtype=int)
    hf_rows = full_data.index[full_data["is_hf"].eq(1)].to_numpy()

    requested = {HELD_OUT_HF_SAMPLE, *IN_SAMPLE_HF_SAMPLES, DENSE_HF_SAMPLE}
    invalid = sorted(index for index in requested if not 0 <= index < len(hf_rows))
    if invalid:
        raise ValueError(f"Invalid HF sample positions: {invalid}")
    if HELD_OUT_HF_SAMPLE in IN_SAMPLE_HF_SAMPLES:
        raise ValueError("The held-out sample cannot be an in-sample example.")

    held_out_row = hf_rows[HELD_OUT_HF_SAMPLE]
    return {
        "full_data": full_data,
        "all_indices": all_indices,
        "hf_rows": hf_rows,
        "masks": reconstruct_training_masks(full_data, held_out_row),
    }


def spectrum_frame(experiment, hf_sample, wavelength_indices):
    source_row = experiment["hf_rows"][hf_sample]
    wide = with_attrs(
        experiment["full_data"].loc[[source_row]], experiment["full_data"].attrs
    )
    return prepare_model2_data(
        wide, LF_number=None, wavelength_indices=wavelength_indices
    )


def dense_frame(experiment, hf_sample, wavelengths):
    """Build a Model 2 query frame at arbitrary wavelengths."""
    source_row = experiment["hf_rows"][hf_sample]
    sample = experiment["full_data"].loc[source_row]
    x_columns = [
        column
        for column in experiment["full_data"].columns
        if not column.startswith("response_")
        and column not in {"is_hf", "source_index"}
    ]
    query = pd.DataFrame(
        np.repeat(
            sample[x_columns].to_numpy(dtype=float)[None, :], len(wavelengths), axis=0
        ),
        columns=x_columns,
    )
    query["wavelength"] = wavelengths
    query["is_hf"] = 1
    query["response"] = np.nan
    return query


def plot_prediction(ax, wavelengths, actual, mean, variance, color):
    """Draw the GP mean, 95% interval, truth, and predictive variance."""
    std = np.sqrt(variance)
    ax.fill_between(
        wavelengths,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color=color,
        alpha=0.2,
        label="95% predictive interval",
    )
    ax.plot(wavelengths, mean, color=color, linewidth=1.6, label="GP mean")
    if actual is not None:
        ax.plot(wavelengths, actual, color="black", linewidth=1.1, label="Truth")

    variance_ax = ax.twinx()
    variance_ax.plot(
        wavelengths,
        variance,
        color="tab:orange",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label="GP variance",
    )
    variance_ax.set_ylabel("GP variance", color="tab:orange")
    variance_ax.tick_params(axis="y", colors="tab:orange")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Eclipse depth")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    var_handles, var_labels = variance_ax.get_legend_handles_labels()
    ax.legend(handles + var_handles, labels + var_labels, fontsize=8)


def plot_held_out_full_spectrum(experiment, models):
    data = spectrum_frame(experiment, HELD_OUT_HF_SAMPLE, experiment["all_indices"])
    wavelengths = data["wavelength"].to_numpy(dtype=float)
    actual = data["response"].to_numpy(dtype=float)
    predictions = predict_variants(models, data)
    records = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, variant in zip(axes, VARIANTS):
        mean, variance = predictions[variant]
        records.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "hf_sample": HELD_OUT_HF_SAMPLE,
                    "wavelength": wavelengths,
                    "actual": actual,
                    "predicted": mean,
                    "prediction_variance": variance,
                    "prediction_std": np.sqrt(variance),
                }
            )
        )
        plot_prediction(
            ax, wavelengths, actual, mean, variance, VARIANTS[variant]["color"]
        )
        ax.set_title(f"{variant}: full spectrum for held-out HF sample 81")
    fig.savefig(OUTPUT_DIR / "held_out_81_full_spectrum.png", dpi=200)
    plt.close(fig)
    pd.concat(records, ignore_index=True).to_csv(
        OUTPUT_DIR / "held_out_81_full_spectrum.csv", index=False
    )


def plot_three_missing_spectra(experiment, models):
    records = []
    fig, axes = plt.subplots(
        3, 2, figsize=(15, 12), constrained_layout=True, squeeze=False
    )
    for row, hf_sample in enumerate(IN_SAMPLE_HF_SAMPLES):
        source_row = experiment["hf_rows"][hf_sample]
        seen_indices = experiment["masks"][source_row]
        missing_indices = np.setdiff1d(experiment["all_indices"], seen_indices)
        missing = spectrum_frame(experiment, hf_sample, missing_indices)
        seen = spectrum_frame(experiment, hf_sample, seen_indices)
        predictions = predict_variants(models, missing)

        wavelengths = missing["wavelength"].to_numpy(dtype=float)
        actual = missing["response"].to_numpy(dtype=float)
        for column, variant in enumerate(VARIANTS):
            ax = axes[row, column]
            mean, variance = predictions[variant]
            records.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "hf_sample": hf_sample,
                        "wavelength": wavelengths,
                        "actual": actual,
                        "predicted": mean,
                        "prediction_variance": variance,
                        "prediction_std": np.sqrt(variance),
                        "was_used_for_this_spectrum": False,
                    }
                )
            )
            plot_prediction(
                ax, wavelengths, actual, mean, variance, VARIANTS[variant]["color"]
            )
            ax.scatter(
                seen["wavelength"], seen["response"], s=13, color="tab:green",
                zorder=4, label="50 wavelengths used for this spectrum",
            )
            ax.set_title(f"{variant}: HF sample {hf_sample}, 145 missing wavelengths")
            ax.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / "three_in_sample_missing_wavelengths.png", dpi=200)
    plt.close(fig)
    pd.concat(records, ignore_index=True).to_csv(
        OUTPUT_DIR / "three_in_sample_missing_wavelengths.csv", index=False
    )


def plot_dense_interpolation(experiment, models):
    source_row = experiment["hf_rows"][DENSE_HF_SAMPLE]
    seen_indices = experiment["masks"][source_row]
    seen = spectrum_frame(experiment, DENSE_HF_SAMPLE, seen_indices)
    seen_wavelengths = seen["wavelength"].to_numpy(dtype=float)
    dense_wavelengths = np.linspace(
        seen_wavelengths.min(), seen_wavelengths.max(), DENSE_GRID_SIZE
    )
    dense = dense_frame(experiment, DENSE_HF_SAMPLE, dense_wavelengths)
    predictions = predict_variants(models, dense)
    records = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, variant in zip(axes, VARIANTS):
        mean, variance = predictions[variant]
        records.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "hf_sample": DENSE_HF_SAMPLE,
                    "wavelength": dense_wavelengths,
                    "predicted": mean,
                    "prediction_variance": variance,
                    "prediction_std": np.sqrt(variance),
                }
            )
        )
        plot_prediction(
            ax, dense_wavelengths, None, mean, variance, VARIANTS[variant]["color"]
        )
        ax.scatter(
            seen_wavelengths,
            seen["response"].to_numpy(dtype=float),
            s=18,
            color="black",
            zorder=4,
            label="50 seen wavelengths",
        )
        ax.set_title(f"{variant}: dense interpolation for HF sample {DENSE_HF_SAMPLE}")
        ax.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / "dense_uniform_interpolation.png", dpi=200)
    plt.close(fig)
    pd.concat(records, ignore_index=True).to_csv(
        OUTPUT_DIR / "dense_uniform_interpolation.csv", index=False
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment()
    models = load_models()
    plot_held_out_full_spectrum(experiment, models)
    plot_three_missing_spectra(experiment, models)
    plot_dense_interpolation(experiment, models)

    print("Created:")
    for filename in (
        "held_out_81_full_spectrum.png",
        "held_out_81_full_spectrum.csv",
        "three_in_sample_missing_wavelengths.png",
        "three_in_sample_missing_wavelengths.csv",
        "dense_uniform_interpolation.png",
        "dense_uniform_interpolation.csv",
    ):
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
