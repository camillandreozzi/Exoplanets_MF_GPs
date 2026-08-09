"""Unit tests for the high-fidelity-only (single-fidelity) GP baseline."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from exoplanets_mf.cv import cv_metrics, cv_predict_hf_only_gp
from exoplanets_mf.hf_only_gp import (
    fit_hf_only_gp,
    hyperparameter_table,
    predict_hf_only,
)
from exoplanets_mf.mf_gp import N_INPUT_DIMS


def _toy_hf_data(seed: int = 0, n_samples: int = 20, n_wavelengths: int = 3):
    rng = np.random.default_rng(seed)
    X_hf = rng.normal(size=(n_samples, N_INPUT_DIMS))
    Y_hf = np.column_stack(
        (np.sin(X_hf[:, 0]), np.cos(X_hf[:, 1]), 0.3 * X_hf[:, 2])
    ) + 1e-2 * rng.normal(size=(n_samples, n_wavelengths))
    return X_hf, Y_hf, np.array([1.0, 2.0, 3.0])


class HFOnlyGPTests(unittest.TestCase):
    def _fit_toy(self):
        X_hf, Y_hf, wl = _toy_hf_data()
        # The synthetic inputs are mostly inert, so length-scales legitimately
        # pin the (wide) upper bound; that warning is not the object of test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layer = fit_hf_only_gp(
                X_hf, Y_hf, wl, seed=0, n_restarts_optimizer=0
            )
        return layer, X_hf, Y_hf, wl

    def test_fit_and_predict_shapes(self) -> None:
        layer, X_hf, Y_hf, wl = self._fit_toy()
        self.assertEqual(len(layer.models), len(wl))
        mean, std = predict_hf_only(layer, X_hf[:5])
        self.assertEqual(mean.shape, (5, len(wl)))
        self.assertEqual(std.shape, (5, len(wl)))
        self.assertFalse(np.isnan(mean).any())
        self.assertTrue(np.all(std >= 0.0))

    def test_hyperparameter_table_has_one_row_per_wavelength(self) -> None:
        layer, _, _, wl = self._fit_toy()
        table = hyperparameter_table(layer)
        self.assertEqual(len(table), len(wl))
        for column in ("signal_variance", "noise_level", "length_scale_0"):
            self.assertIn(column, table.columns)

    def test_cv_covers_every_sample_with_std(self) -> None:
        X_hf, Y_hf, wl = _toy_hf_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = cv_predict_hf_only_gp(
                X_hf, Y_hf, wl, n_splits=5, seed=0
            )
        self.assertEqual(predictions.y_pred.shape, Y_hf.shape)
        self.assertFalse(np.isnan(predictions.y_pred).any())
        self.assertIsNotNone(predictions.y_std)
        self.assertEqual(set(predictions.fold_of_sample), set(range(5)))
        metrics = cv_metrics(Y_hf, predictions)
        self.assertEqual(metrics.rmse_per_wavelength.shape, (len(wl),))


if __name__ == "__main__":
    unittest.main()
