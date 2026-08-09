"""Unit tests for the GPBoost AR(1) multi-fidelity GP variants.

These skip cleanly when gpboost is unavailable, so `make verify` stays green in
environments without the optional dependency.
"""

from __future__ import annotations

import unittest

import numpy as np

from exoplanets_mf.cv import cv_metrics
from exoplanets_mf.gpboost_mf import gpboost_available

if gpboost_available():
    from exoplanets_mf.gpboost_mf import (
        cv_predict_model1_gpboost,
        fit_model1_gpboost,
        hyperparameter_table_model1_gpboost,
        predict_hf_model1_gpboost,
    )
    from exoplanets_mf.gpboost_model2 import (
        fit_model2_gpboost,
        predict_hf_model2_gpboost,
    )


def _two_fidelity_data(seed: int = 0):
    """Well-posed low-dimensional AR(1) 2-fidelity data with a known rho."""
    rng = np.random.default_rng(seed)
    p = 2
    rho_true = 0.7

    def f_low(X):
        return np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2

    def delta(X):
        return 0.3 * np.cos(4 * X[:, 1])

    X_lf = rng.uniform(0, 1, size=(200, p))
    X_hf = rng.uniform(0, 1, size=(50, p))
    Y_lf = (f_low(X_lf) + 0.01 * rng.standard_normal(200))[:, None]
    Y_hf = (rho_true * f_low(X_hf) + delta(X_hf) + 0.01 * rng.standard_normal(50))[:, None]
    return X_lf, Y_lf, X_hf, Y_hf, rho_true


@unittest.skipUnless(gpboost_available(), "gpboost not installed")
class GPBoostModel1Tests(unittest.TestCase):
    def test_recovers_known_rho(self):
        X_lf, Y_lf, X_hf, Y_hf, rho_true = _two_fidelity_data()
        layer = fit_model1_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, np.array([1.0]), seed=0, subsample_size=None
        )
        self.assertEqual(layer.rho.shape, (1,))
        self.assertAlmostEqual(float(layer.rho[0]), rho_true, delta=0.1)

    def test_predict_shapes_and_accuracy(self):
        X_lf, Y_lf, X_hf, Y_hf, _ = _two_fidelity_data()
        layer = fit_model1_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, np.array([1.0]), seed=0, subsample_size=150
        )
        means, stds = predict_hf_model1_gpboost(layer, X_hf)
        self.assertEqual(means.shape, (X_hf.shape[0], 1))
        self.assertEqual(stds.shape, (X_hf.shape[0], 1))
        self.assertTrue(np.all(stds > 0))
        rmse = float(np.sqrt(((means - Y_hf) ** 2).mean()))
        self.assertLess(rmse, 0.1)

    def test_hyperparameter_table_columns(self):
        X_lf, Y_lf, X_hf, Y_hf, _ = _two_fidelity_data()
        layer = fit_model1_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, np.array([1.0]), seed=0, subsample_size=120
        )
        table = hyperparameter_table_model1_gpboost(layer)
        for column in ("wavelength", "rho", "low_signal_variance", "error_var"):
            self.assertIn(column, table.columns)
        self.assertEqual(len(table), 1)

    def test_cv_produces_one_prediction_per_sample(self):
        X_lf, Y_lf, X_hf, Y_hf, _ = _two_fidelity_data()
        predictions = cv_predict_model1_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, np.array([1.0]),
            seed=0, subsample_size=120, n_splits=3,
        )
        self.assertEqual(predictions.y_pred.shape, Y_hf.shape)
        self.assertFalse(np.isnan(predictions.y_pred).any())
        metrics = cv_metrics(Y_hf, predictions)
        self.assertTrue(np.isfinite(metrics.rmse_pooled))
        self.assertIsNotNone(metrics.coverage_95)


@unittest.skipUnless(gpboost_available(), "gpboost not installed")
class GPBoostModel2Tests(unittest.TestCase):
    def _augmented_data(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        p = 2
        n_lf, n_hf, n_lambda = 60, 20, 12
        X_lf = rng.uniform(0, 1, size=(n_lf, p))
        X_hf = rng.uniform(0, 1, size=(n_hf, p))
        wavelengths = np.linspace(2.5, 5.0, n_lambda)

        def spectra(X, rho):
            base = np.sin(3 * X[:, [0]]) + wavelengths[None, :]
            return rho * base + 0.01 * rng.standard_normal((len(X), n_lambda))

        Y_lf = spectra(X_lf, 1.0)
        Y_hf = spectra(X_hf, 0.6)
        return X_lf, Y_lf, X_hf, Y_hf, wavelengths

    def test_fit_and_predict_all_wavelengths(self):
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = self._augmented_data()
        layer = fit_model2_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, lf_sample_size=40, lambda_stride=3,
        )
        self.assertTrue(np.isfinite(layer.rho))
        means, stds = predict_hf_model2_gpboost(layer, X_hf)
        self.assertEqual(means.shape, (X_hf.shape[0], len(wavelengths)))
        self.assertEqual(stds.shape, (X_hf.shape[0], len(wavelengths)))
        self.assertTrue(np.all(stds > 0))


if __name__ == "__main__":
    unittest.main()
