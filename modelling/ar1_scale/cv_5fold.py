"""Paired five-fold HF validation: fixed versus wavelength-specific AR scaling."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Adjustable experiment and model settings (no command-line arguments).
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "ar1_scale"
N_FOLDS = 5
RANDOM_SEED = 2026
LF_SAMPLE_SIZE = None  # None uses all 10,000 LF rows; exact GPs are expensive.
RESPONSE_INDEXES = None  # None uses every wavelength.
FIXED_RHO = 1.0  # Also the starting rho for Model 1B.
RHO_BOUNDS = (-5.0, 5.0)
MATERN_NU = 1.5
LENGTH_SCALE_INIT = 1.0
LENGTH_SCALE_BOUNDS = (1e-2, 1e3)
SIGNAL_VARIANCE_BOUNDS = (1e-6, 1e6)
NOISE_VARIANCE_BOUNDS = (1e-10, 1e2)
INITIAL_NOISE_VARIANCE = 0.1  # Variances are in standardized response units.
INITIAL_LF_VARIANCE = 0.5
INITIAL_DELTA_VARIANCE = 0.5
JITTER = 1e-10
OPTIMIZER_MAX_ITER = 1000
OPTIMIZER_FTOL = 1e-9
OPTIMIZER_MAX_LINE_SEARCH = 50
N_RESTARTS_OPTIMIZER = 0
PLOT_DPI = 180
MODELS = {"model1a": "Model 1A (fixed rho)", "model1b": "Model 1B (fitted rho)"}

from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel, Matern, WhiteKernel


class AR1MultiFidelityKernel(Kernel):
    """Local copy of kernel_comparison/sklearn_ar1.py, extended for fixed rho.

    Both models use this custom scikit-learn covariance. Set rho_bounds to
    "fixed" to remove rho from theta, bounds, and the likelihood gradient.

    The last column of ``X`` is the fidelity indicator (0 = low, 1 = high);
    every preceding column is an input coordinate handed to the sub-kernels.

    ``rho`` is unconstrained: unlike every other kernel hyperparameter it is
    optimised on the natural scale, not the log scale, so that it is allowed to
    be negative (as in GPBoost).
    """

    def __init__(
        self,
        low_kernel,
        discrepancy_kernel,
        rho=1.0,
        rho_bounds=RHO_BOUNDS,
    ):
        self.low_kernel = low_kernel
        self.discrepancy_kernel = discrepancy_kernel
        self.rho = rho
        self.rho_bounds = rho_bounds

    # -- scikit-learn plumbing ---------------------------------------------
    def get_params(self, deep=True):
        params = {
            "low_kernel": self.low_kernel,
            "discrepancy_kernel": self.discrepancy_kernel,
            "rho": self.rho,
            "rho_bounds": self.rho_bounds,
        }
        if deep:
            for prefix, kernel in (
                ("low_kernel", self.low_kernel),
                ("discrepancy_kernel", self.discrepancy_kernel),
            ):
                for name, value in kernel.get_params().items():
                    params[f"{prefix}__{name}"] = value
        return params

    @property
    def hyperparameter_rho(self):
        return Hyperparameter("rho", "numeric", self.rho_bounds)

    @property
    def hyperparameters(self):
        """Sub-kernel hyperparameters (prefixed) followed by ``rho``."""
        hyperparameters = []
        for prefix, kernel in (
            ("low_kernel", self.low_kernel),
            ("discrepancy_kernel", self.discrepancy_kernel),
        ):
            for hyperparameter in kernel.hyperparameters:
                hyperparameters.append(
                    Hyperparameter(
                        f"{prefix}__{hyperparameter.name}",
                        hyperparameter.value_type,
                        hyperparameter.bounds,
                        hyperparameter.n_elements,
                    )
                )
        hyperparameters.append(self.hyperparameter_rho)
        return hyperparameters

    @property
    def theta(self):
        """Sub-kernel thetas (log scale) with ``rho`` appended (natural scale)."""
        return np.concatenate(
            [
                self.low_kernel.theta,
                self.discrepancy_kernel.theta,
                np.array([] if self.hyperparameter_rho.fixed else [self.rho], dtype=float),
            ]
        )

    @theta.setter
    def theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        n_low = self.low_kernel.n_dims
        n_discrepancy = self.discrepancy_kernel.n_dims
        expected = n_low + n_discrepancy + int(not self.hyperparameter_rho.fixed)
        if theta.shape[0] != expected:
            raise ValueError(
                f"theta has {theta.shape[0]} entries, expected {expected}"
            )
        self.low_kernel.theta = theta[:n_low]
        self.discrepancy_kernel.theta = theta[n_low : n_low + n_discrepancy]
        if not self.hyperparameter_rho.fixed:
            self.rho = float(theta[-1])

    @property
    def bounds(self):
        return np.vstack(
            [
                self.low_kernel.bounds,
                self.discrepancy_kernel.bounds,
                np.empty((0, 2)) if self.hyperparameter_rho.fixed else np.atleast_2d(np.asarray(self.rho_bounds, dtype=float)),
            ]
        )

    def is_stationary(self):
        # Not stationary: the covariance depends on the fidelity level itself.
        return False

    def __repr__(self):
        return (
            f"AR1MF(low={self.low_kernel!r}, "
            f"discrepancy={self.discrepancy_kernel!r}, rho={self.rho:.3g})"
        )

    # -- the covariance itself ---------------------------------------------
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        x_train, fidelity_x = X[:, :-1], X[:, -1]
        scale_x = np.where(fidelity_x == 1.0, self.rho, 1.0)

        if Y is None:
            if eval_gradient:
                k_low, grad_low = self.low_kernel(x_train, eval_gradient=True)
                k_discrepancy, grad_discrepancy = self.discrepancy_kernel(
                    x_train, eval_gradient=True
                )
            else:
                k_low = self.low_kernel(x_train)
                k_discrepancy = self.discrepancy_kernel(x_train)
            scale_y, fidelity_y = scale_x, fidelity_x
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y = np.atleast_2d(Y)
            y_train, fidelity_y = Y[:, :-1], Y[:, -1]
            scale_y = np.where(fidelity_y == 1.0, self.rho, 1.0)
            k_low = self.low_kernel(x_train, y_train)
            k_discrepancy = self.discrepancy_kernel(x_train, y_train)

        low_mask = np.outer(scale_x, scale_y)
        discrepancy_mask = np.outer(fidelity_x, fidelity_y)
        K = low_mask * k_low + discrepancy_mask * k_discrepancy

        if not eval_gradient:
            return K

        # d/d(theta) of each block; rho enters only through the low-fidelity term.
        gradient_low = low_mask[:, :, np.newaxis] * grad_low
        gradient_discrepancy = discrepancy_mask[:, :, np.newaxis] * grad_discrepancy
        d_low_mask_d_rho = np.outer(fidelity_x, scale_y) + np.outer(scale_x, fidelity_y)
        gradient_rho = (d_low_mask_d_rho * k_low)[:, :, np.newaxis]

        gradient = np.dstack([gradient_low, gradient_discrepancy]
                             + ([] if self.hyperparameter_rho.fixed else [gradient_rho]))
        return K, gradient

    def diag(self, X):
        X = np.atleast_2d(X)
        x_train, fidelity = X[:, :-1], X[:, -1]
        scale = np.where(fidelity == 1.0, self.rho, 1.0)
        return scale**2 * self.low_kernel.diag(x_train) + fidelity * (
            self.discrepancy_kernel.diag(x_train)
        )


@dataclass
class FittedModel1:
    gp: GaussianProcessRegressor
    input_mean: np.ndarray
    input_std: np.ndarray
    response_mean: float
    response_scale: float


def _optimizer(objective, initial_theta, bounds):
    result = minimize(objective, initial_theta, jac=True, bounds=bounds,
                      method="L-BFGS-B", options={"maxiter": OPTIMIZER_MAX_ITER,
                                                "ftol": OPTIMIZER_FTOL,
                                                "maxls": OPTIMIZER_MAX_LINE_SEARCH})
    if not result.success:
        warnings.warn(f"AR1 optimizer: {result.message}", ConvergenceWarning)
    return result.x, result.fun


def _fit_model1(train_data, fixed_rho):
    x = train_data.iloc[:, :-2].to_numpy(dtype=float)
    fidelity = train_data.iloc[:, -2].to_numpy(dtype=float)
    y = train_data.iloc[:, -1].to_numpy(dtype=float)
    if not np.all(np.isfinite(np.column_stack([x, fidelity, y]))):
        raise ValueError("Model inputs and responses must be finite.")
    if set(np.unique(fidelity)) != {0., 1.}:
        raise ValueError("Training data must contain both LF (0) and HF (1).")
    mean = x[fidelity == 1].mean(axis=0)
    std = x[fidelity == 1].std(axis=0)
    std = np.where(std == 0, 1., std)
    coords = np.column_stack([(x - mean) / std, fidelity])
    # One shared response transform for LF and HF preserves their relative scale.
    response_mean = float(y.mean())
    scale = float(y.std(ddof=0))
    scale = scale if scale > 0 else 1.0
    kernel = AR1MultiFidelityKernel(
        ConstantKernel(INITIAL_LF_VARIANCE, SIGNAL_VARIANCE_BOUNDS)
        * Matern(LENGTH_SCALE_INIT, LENGTH_SCALE_BOUNDS, nu=MATERN_NU),
        ConstantKernel(INITIAL_DELTA_VARIANCE, SIGNAL_VARIANCE_BOUNDS)
        * Matern(LENGTH_SCALE_INIT, LENGTH_SCALE_BOUNDS, nu=MATERN_NU),
        rho=FIXED_RHO if fixed_rho is None else fixed_rho,
        rho_bounds=RHO_BOUNDS if fixed_rho is None else "fixed",
    ) + WhiteKernel(INITIAL_NOISE_VARIANCE, NOISE_VARIANCE_BOUNDS)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=JITTER, normalize_y=False,
        optimizer=_optimizer, n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        random_state=RANDOM_SEED)
    gp.fit(coords, (y - response_mean) / scale)
    return FittedModel1(gp, mean, std, response_mean, scale)


def fit_model1a(train_data):
    """Exact custom sklearn AR1 GP with rho held at FIXED_RHO."""
    return _fit_model1(train_data, fixed_rho=FIXED_RHO)


def fit_model1b(train_data):
    """Exact custom sklearn AR1 GP estimating rho for this wavelength/fold."""
    return _fit_model1(train_data, fixed_rho=None)


def predict_model1(model, validation_data):
    """Restore the training mean and original response units; no validation fitting."""
    x = validation_data.iloc[:, :-2].to_numpy(dtype=float)
    fidelity = validation_data.iloc[:, -2].to_numpy(dtype=float)
    coords = np.column_stack([(x - model.input_mean) / model.input_std, fidelity])
    mu, std = model.gp.predict(coords, return_std=True)
    return {"mu": model.response_mean + model.response_scale * mu,
            "var": (model.response_scale * std)**2}


def model1_cov_pars(model):
    ar1, noise = model.gp.kernel_.k1, model.gp.kernel_.k2
    scale2 = model.response_scale**2
    return np.array([noise.noise_level * scale2,
        ar1.low_kernel.k1.constant_value * scale2, ar1.low_kernel.k2.length_scale,
        ar1.discrepancy_kernel.k1.constant_value * scale2,
        ar1.discrepancy_kernel.k2.length_scale, ar1.rho], dtype=float)


def load_full_data():
    """Load the repository's wide spectra without importing shared model utilities."""
    x_hf = pd.read_csv(DATA_DIR / "XHF.csv", skipinitialspace=True)
    x_hf.columns = x_hf.columns.str.strip()
    def lf_path(prefix):
        for name in [f"{prefix}10k.csv", f"{prefix}_10k.csv"]:
            if (DATA_DIR / name).exists():
                return DATA_DIR / name
        raise FileNotFoundError(f"Missing {prefix}10k.csv or {prefix}_10k.csv")
    x_lf = pd.read_csv(lf_path("XLF"), skipinitialspace=True)
    x_lf.columns = x_lf.columns.str.strip()
    if list(x_lf.columns) != list(x_hf.columns):
        x_lf = pd.read_csv(lf_path("XLF"), header=None, names=x_hf.columns, skipinitialspace=True)
    y_hf = pd.read_csv(DATA_DIR / "YHF.csv", index_col=0)
    wavelengths = y_hf.columns.astype(float)
    y_lf = pd.read_csv(lf_path("YLF"))
    responses = [f"response_{i}" for i in range(len(wavelengths))]
    y_hf.columns = responses
    y_lf.columns = responses
    frames = []
    for x, y, fidelity in [(x_lf, y_lf, 0), (x_hf, y_hf, 1)]:
        if len(x) != len(y):
            raise ValueError("Predictor and response row counts differ.")
        frames.append(pd.concat([x.reset_index(drop=True).assign(is_hf=fidelity),
                                 y.reset_index(drop=True)], axis=1))
    data = pd.concat(frames, ignore_index=True)
    data.attrs["wavelength_map"] = dict(enumerate(wavelengths))
    return data


def iter_hf_kfold_splits(data, n_folds, random_state):
    hf_indexes = data.index[data.is_hf.eq(1)].to_numpy()
    if not 2 <= n_folds <= len(hf_indexes):
        raise ValueError("N_FOLDS must be between 2 and the number of HF samples.")
    for fold, held_out in enumerate(np.array_split(
            np.random.default_rng(random_state).permutation(hf_indexes), n_folds), 1):
        train = data.drop(index=held_out).copy()
        valid = data.loc[np.sort(held_out)].copy()
        train["source_index"] = train.index
        valid["source_index"] = valid.index
        yield {"fold": fold, "train_data": train, "validation_data": valid}


def metric_table(predictions, keys):
    """Range-normalized RMSE; a zero target range gives undefined NRMSE."""
    rows = []
    for key, group in predictions.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        residual = group.y_pred.to_numpy() - group.y_true.to_numpy()
        span = float(group.y_true.max() - group.y_true.min())
        rmse = float(np.sqrt(np.mean(residual**2)))
        rows.append(dict(zip(keys, key), n=len(group), rmse=rmse,
                         mae=float(np.mean(np.abs(residual))),
                         nrmse=rmse / span if span > 0 else np.nan,
                         y_true_range=span))
    return pd.DataFrame(rows)


def run_fold(split, response_indexes):
    train, valid = split["train_data"], split["validation_data"]
    x_columns = [c for c in train.columns if c not in {"is_hf", "source_index"}
                 and not c.startswith("response_")]
    predictions, parameters = [], []
    for position, index in enumerate(response_indexes, 1):
        columns = [*x_columns, "is_hf", f"response_{index}"]
        train_scalar, valid_scalar = train[columns].copy(), valid[columns].copy()
        wavelength = float(train.attrs["wavelength_map"][index])
        for model in MODELS:
            fitted = (fit_model1a if model == "model1a" else fit_model1b)(train_scalar)
            prediction = predict_model1(fitted, valid_scalar)
            cov = model1_cov_pars(fitted)
            if model == "model1a" and not np.isclose(cov[-1], FIXED_RHO, rtol=0, atol=1e-12):
                raise RuntimeError("Custom sklearn kernel did not preserve the fixed rho.")
            if not np.all(np.isfinite(prediction["mu"])) or not np.all(np.isfinite(cov)):
                raise RuntimeError(f"Non-finite fit: fold={split['fold']}, {model}, response={index}")
            parameters.append(dict(fold=split["fold"], model=model,
                response_index=index, wavelength=wavelength,
                **dict(zip(["noise_var", "lf_var", "lf_range", "delta_var", "delta_range", "rho"], cov))))
            for source, truth, mu, var in zip(valid.source_index,
                    valid_scalar.iloc[:, -1], prediction["mu"], prediction["var"]):
                predictions.append(dict(fold=split["fold"], model=model,
                    held_out_source_index=int(source), response_index=index,
                    wavelength=wavelength, y_true=float(truth), y_pred=float(mu),
                    variance=float(var), residual=float(mu - truth)))
        if position == 1 or position % 10 == 0 or position == len(response_indexes):
            print(f"Fold {split['fold']}/{N_FOLDS}: {position}/{len(response_indexes)} wavelengths", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(parameters)


def plot_results(predictions, metrics, parameters, output):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for model, label in MODELS.items():
        group = metrics[metrics.model.eq(model)].sort_values("wavelength")
        for ax, metric in zip(axes, ["nrmse", "mae", "rmse"]):
            ax.plot(group.wavelength, group[metric], label=label)
            ax.set_ylabel(metric.upper())
            ax.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel("Wavelength")
    fig.tight_layout()
    fig.savefig(output / "metrics_per_wavelength.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Every held-out spectrum, avoiding selection by prediction performance.
    with PdfPages(output / "predicted_spectra_and_residuals.pdf") as pdf:
        for number, (source, group) in enumerate(predictions.groupby("held_out_source_index")):
            fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
            truth = group[group.model.eq("model1a")].sort_values("wavelength")
            axes[0].plot(truth.wavelength, truth.y_true, color="black", label="Observed HF")
            for model, label in MODELS.items():
                spectrum = group[group.model.eq(model)].sort_values("wavelength")
                axes[0].plot(spectrum.wavelength, spectrum.y_pred, label=label, alpha=.85)
                axes[1].plot(spectrum.wavelength, spectrum.residual, label=label)
            axes[0].set_title(f"Held-out HF sample {source}, fold {int(group.fold.iloc[0])}")
            axes[0].set_ylabel("Response")
            axes[0].legend()
            axes[1].axhline(0, color="black", linewidth=.8)
            axes[1].set_ylabel("Predicted − observed")
            axes[1].set_xlabel("Wavelength")
            for ax in axes:
                ax.grid(alpha=.25)
            fig.tight_layout()
            pdf.savefig(fig)
            if number == 0:
                fig.savefig(output / "predicted_spectra_and_residuals.png", dpi=PLOT_DPI)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for fold, group in parameters[parameters.model.eq("model1b")].groupby("fold"):
        group = group.sort_values("wavelength")
        ax.plot(group.wavelength, group.rho, label=f"Model 1B fold {fold}", alpha=.7)
    ax.axhline(parameters.loc[parameters.model.eq("model1a"), "rho"].iloc[0],
               color="black", linestyle="--", label="Model 1A fixed rho")
    ax.set(xlabel="Wavelength", ylabel="rho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "rho_per_wavelength.png", dpi=PLOT_DPI)
    plt.close(fig)


def main():
    if not np.isfinite(FIXED_RHO) or not RHO_BOUNDS[0] <= FIXED_RHO <= RHO_BOUNDS[1]:
        raise ValueError("FIXED_RHO must be finite and inside RHO_BOUNDS for paired initialization.")
    data = load_full_data()
    lf = data[data.is_hf.eq(0)]
    if LF_SAMPLE_SIZE is not None:
        if not 1 <= LF_SAMPLE_SIZE <= len(lf):
            raise ValueError(f"LF_SAMPLE_SIZE must be between 1 and {len(lf)}")
        selected = lf.sample(n=LF_SAMPLE_SIZE, random_state=RANDOM_SEED).index
        data = data.loc[selected.union(data.index[data.is_hf.eq(1)]).sort_values()].copy()
    output = RESULTS_DIR.resolve()
    output.mkdir(parents=True, exist_ok=True)
    indexes = list(data.attrs["wavelength_map"]) if RESPONSE_INDEXES is None else list(RESPONSE_INDEXES)
    if not indexes or len(set(indexes)) != len(indexes) or any(i not in data.attrs["wavelength_map"] for i in indexes):
        raise ValueError("RESPONSE_INDEXES must contain unique valid wavelength indexes.")
    config = dict(n_folds=N_FOLDS, cv_method="hf_kfold", seed=RANDOM_SEED,
        model1a_fixed_rho=FIXED_RHO, model1b_rho="fitted per wavelength within each fold",
        n_lf=int(data.is_hf.eq(0).sum()), n_hf=int(data.is_hf.eq(1).sum()),
        response_indexes=indexes, backend="sklearn GaussianProcessRegressor",
        kernel="local AR1MultiFidelityKernel + WhiteKernel", gp_approx="exact",
        input_scaling="training HF only, GP coordinates only",
        mean="zero in standardized response space; no linear baseline",
        response_scaling="shared training LF+HF mean and standard deviation per wavelength; reversed for predictions",
        fidelity_scaling="none: categorical 0/1 kernel indicator",
        settings={name: value for name, value in globals().items()
                  if name.isupper() and isinstance(value, (int, float, bool, tuple))},
        nrmse="RMSE / (max observed - min observed) within each reported group; NaN for zero range",
        residual="predicted - observed")
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame({"source_index": data.index[data.is_hf.eq(0)]}).to_csv(
        output / "lf_sample_source_indices.csv", index=False)
    predictions, parameters = [], []
    for split in iter_hf_kfold_splits(data, n_folds=N_FOLDS, random_state=RANDOM_SEED):
        pred, pars = run_fold(split, indexes)
        pred.to_csv(output / f"fold_{split['fold']:02d}_predictions.csv", index=False)
        pars.to_csv(output / f"fold_{split['fold']:02d}_parameters.csv", index=False)
        predictions.append(pred)
        parameters.append(pars)
    predictions, parameters = pd.concat(predictions, ignore_index=True), pd.concat(parameters, ignore_index=True)
    predictions.to_csv(output / f"{N_FOLDS}fold_predictions.csv", index=False)
    parameters.to_csv(output / f"{N_FOLDS}fold_parameters.csv", index=False)
    metrics = metric_table(predictions, ["model", "response_index", "wavelength"])
    metrics.to_csv(output / f"{N_FOLDS}fold_metrics_per_wavelength.csv", index=False)
    metric_table(predictions, ["model", "fold"]).to_csv(output / f"{N_FOLDS}fold_metrics_per_fold.csv", index=False)
    metric_table(predictions, ["model", "held_out_source_index"]).to_csv(output / f"{N_FOLDS}fold_metrics_per_spectrum.csv", index=False)
    summary = metric_table(predictions, ["model"])
    means = metrics.groupby("model")[["nrmse", "mae", "rmse"]].mean().add_prefix("mean_wavelength_")
    summary = summary.join(means, on="model")
    summary.to_csv(output / f"{N_FOLDS}fold_summary.csv", index=False)
    comparison = metrics.pivot(index=["response_index", "wavelength"], columns="model", values=["nrmse", "mae", "rmse"])
    comparison.columns = [f"{model}_{metric}" for metric, model in comparison.columns]
    for metric in ["nrmse", "mae", "rmse"]:
        comparison[f"{metric}_delta_b_minus_a"] = comparison[f"model1b_{metric}"] - comparison[f"model1a_{metric}"]
    comparison.to_csv(output / f"{N_FOLDS}fold_comparison_per_wavelength.csv")
    plot_results(predictions, metrics, parameters, output)
    print(summary.to_string(index=False))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
