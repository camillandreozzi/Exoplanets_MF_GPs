"""Checks for fixed-rho estimation and the paired CV output pipeline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from modelling.ar1_scale import cv_5fold as cv
from sklearn.base import clone
from threadpoolctl import threadpool_limits


class ARScaleTests(unittest.TestCase):
    def test_fixed_rho_and_free_rho(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-3, 3, 35)
        data = pd.DataFrame({"x": np.tile(x, 2),
            "is_hf": np.repeat([0, 1], len(x)),
            "response_0": np.r_[np.sin(x), 2 * np.sin(x)] + rng.normal(0, .02, 70)})
        with patch.object(cv, "FIXED_RHO", .7), threadpool_limits(limits=1):
            fixed_model = cv.fit_model1a(data)
            free_model = cv.fit_model1b(data)
        fixed, free = map(cv.model1_cov_pars, [fixed_model, free_model])
        self.assertAlmostEqual(fixed[-1], .7, places=12)
        self.assertGreater(abs(free[-1] - .7), .1)
        self.assertEqual(fixed_model.gp.kernel_.n_dims, 5)
        self.assertEqual(free_model.gp.kernel_.n_dims, 6)
        before = cv.predict_model1(free_model, data)
        changed = data.copy()
        changed["response_0"] = 1e9
        np.testing.assert_allclose(before["mu"], cv.predict_model1(free_model, changed)["mu"])

    def test_scaling_and_original_units(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=40)
        data = pd.DataFrame({"x": x, "constant": 7.,
            "is_hf": np.repeat([0, 1], 20),
            "response_0": np.sin(x) + rng.normal(0, .02, 40)})
        transformed = data.copy()
        transformed["x"] = 300 + 20 * x
        transformed["response_0"] = 100 + 5 * data.response_0
        for fitter in [cv.fit_model1a, cv.fit_model1b]:
            with self.subTest(model=fitter.__name__), threadpool_limits(limits=1):
                original = fitter(data)
                scaled = fitter(transformed)
                np.testing.assert_allclose(scaled.gp.X_train_, original.gp.X_train_, atol=1e-13)
                np.testing.assert_allclose(scaled.gp.y_train_, original.gp.y_train_, atol=1e-13)
                hf_coords = scaled.gp.X_train_[20:, :-1]
                np.testing.assert_allclose(hf_coords.mean(axis=0), 0, atol=1e-13)
                np.testing.assert_allclose(hf_coords.std(axis=0), [1, 0], atol=1e-13)
                np.testing.assert_array_equal(scaled.gp.X_train_[:, -1], data.is_hf)
                self.assertAlmostEqual(scaled.gp.y_train_.mean(), 0)
                self.assertAlmostEqual(scaled.gp.y_train_.std(), 1)
                prediction = cv.predict_model1(scaled, transformed)
                coords = np.column_stack([
                    (transformed.iloc[:, :-2].to_numpy() - scaled.input_mean) / scaled.input_std,
                    transformed.is_hf])
                mu, std = scaled.gp.predict(coords, return_std=True)
                np.testing.assert_allclose(prediction["mu"], transformed.response_0.mean() + transformed.response_0.std(ddof=0) * mu)
                np.testing.assert_allclose(prediction["var"], transformed.response_0.var(ddof=0) * std**2)

    def test_kernel_gradient_and_reference(self):
        from kernel_comparison.sklearn_ar1 import AR1MultiFidelityKernel as Reference
        x = np.array([[0., 0.], [.3, 1.], [1., 0.], [1.2, 1.]])
        low = cv.ConstantKernel(1.2) * cv.Matern(.8, nu=1.5)
        delta = cv.ConstantKernel(.4) * cv.Matern(1.1, nu=1.5)
        for bounds in ["fixed", (-5., 5.)]:
            kernel = cv.AR1MultiFidelityKernel(clone(low), clone(delta), -.7, bounds)
            covariance, gradient = kernel(x, eval_gradient=True)
            np.testing.assert_allclose(covariance, Reference(low, delta, -.7)(x))
            np.testing.assert_allclose(kernel.diag(x), np.diag(covariance))
            for i in range(kernel.n_dims):
                offset = np.zeros(kernel.n_dims)
                offset[i] = 1e-6
                numerical = (kernel.clone_with_theta(kernel.theta + offset)(x)
                             - kernel.clone_with_theta(kernel.theta - offset)(x)) / 2e-6
                np.testing.assert_allclose(gradient[:, :, i], numerical, atol=1e-7)

    def test_metrics(self):
        data = pd.DataFrame({"model": ["a", "a", "b", "b"],
            "y_true": [1., 3., 2., 2.], "y_pred": [2., 1., 2., 3.]})
        metrics = cv.metric_table(data, ["model"]).set_index("model")
        self.assertAlmostEqual(metrics.loc["a", "rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(metrics.loc["a", "mae"], 1.5)
        self.assertAlmostEqual(metrics.loc["a", "nrmse"], np.sqrt(2.5) / 2)
        self.assertTrue(np.isnan(metrics.loc["b", "nrmse"]))

    def test_five_fold_pipeline(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(-2, 2, 127)
        fidelity = np.r_[np.zeros(30), np.ones(97)]
        data = pd.DataFrame({"x": x, "is_hf": fidelity,
            **{f"response_{i}": np.sin(x) * (1 + .4 * fidelity) + i * .01 + rng.normal(0, .03, len(x))
               for i in range(195)}})
        data.attrs["wavelength_map"] = {i: i + 1. for i in range(195)}
        original_plot = cv.plot_results
        def small_plot(predictions, metrics, parameters, output):
            # Exercise the real renderer while keeping this check to one PDF page.
            original_plot(predictions[predictions.held_out_source_index.eq(30)],
                          metrics, parameters, output)
        with tempfile.TemporaryDirectory() as directory, patch.object(cv, "load_full_data", return_value=data), \
             patch.object(cv, "RESPONSE_INDEXES", [0, 1]), patch.object(cv, "RESULTS_DIR", Path(directory)), \
             patch.object(cv, "plot_results", side_effect=small_plot), threadpool_limits(limits=1):
            cv.main()
            output = Path(directory)
            predictions = pd.read_csv(output / "5fold_predictions.csv")
            self.assertEqual(len(predictions), 97 * 2 * 2)
            self.assertEqual(predictions.fold.nunique(), 5)
            self.assertTrue(predictions.groupby("held_out_source_index").fold.nunique().eq(1).all())
            self.assertFalse(predictions.duplicated(["model", "held_out_source_index", "response_index"]).any())
            self.assertTrue(predictions.held_out_source_index.ge(30).all())
            params = pd.read_csv(output / "5fold_parameters.csv")
            np.testing.assert_allclose(params.loc[params.model.eq("model1a"), "rho"], 1.)
            for file in ["5fold_summary.csv", "5fold_comparison_per_wavelength.csv",
                         "predicted_spectra_and_residuals.pdf", "metrics_per_wavelength.png"]:
                self.assertGreater((output / file).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
