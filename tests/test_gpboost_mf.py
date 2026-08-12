"""Unit tests for the GPBoost AR(1) multi-fidelity GP variants.

These skip cleanly when gpboost is unavailable, so `make verify` stays green in
environments without the optional dependency.
"""

from __future__ import annotations

import unittest

import numpy as np

from exoplanets_mf.cv import cv_metrics
from exoplanets_mf.gpboost_mf import gpboost_available, heuristic_init_cov_pars
from exoplanets_mf.paths import approximation_suffix

if gpboost_available():
    from exoplanets_mf.gpboost_hf_only import (
        cv_predict_hf_only_gpboost,
        fit_hf_only_gpboost,
        hyperparameter_table_hf_only_gpboost,
        predict_hf_only_gpboost,
    )
    from exoplanets_mf.gpboost_mf import (
        cv_predict_model1_gpboost,
        cv_predict_model1a_gpboost,
        fit_model1_gpboost,
        fit_model1a_gpboost,
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


class ApproximationSuffixTests(unittest.TestCase):
    """Exact and approximate fits must not overwrite each other's outputs."""

    def test_default_approximation_keeps_the_plain_directory_name(self):
        self.assertEqual(approximation_suffix("none", None), "")
        self.assertEqual(
            approximation_suffix("vecchia", 30, default="vecchia"), ""
        )

    def test_vecchia_is_labelled_with_its_neighbour_count(self):
        self.assertEqual(approximation_suffix("vecchia", 30), "_vecchia_k30")
        self.assertEqual(approximation_suffix("vecchia", None), "_vecchia")
        self.assertEqual(
            approximation_suffix("vecchia_euclidean", 20), "_vecchia_euclidean_k20"
        )

    def test_exact_is_labelled_when_it_is_not_the_default(self):
        self.assertEqual(approximation_suffix("none", None, default="vecchia"), "_exact")

    def test_distinct_settings_never_collide(self):
        settings = [("none", None), ("vecchia", 20), ("vecchia", 30), ("fitc", None)]
        suffixes = [approximation_suffix(a, k) for a, k in settings]
        self.assertEqual(len(set(suffixes)), len(settings))


class HeuristicInitTests(unittest.TestCase):
    """The starting values that keep GPBoost out of its degenerate optimum.

    These need no gpboost install: the heuristic is pure numpy.
    """

    def _coords(self, scale: float = 1.0, n_dims: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        features = scale * rng.normal(size=(120, n_dims))
        fidelity = np.repeat([0.0, 1.0], 60)
        coords = np.column_stack([features, fidelity])
        y = rng.normal(size=len(coords))
        return coords, y

    def test_layout_matches_gpboost_parameter_order(self):
        coords, y = self._coords(n_dims=4)
        num_cov_pars = 4 + 2 * 4  # err + (var + 4 ranges) x 2 + rho
        init = heuristic_init_cov_pars(coords, y, num_cov_pars, seed=0)
        self.assertEqual(init.shape, (num_cov_pars,))
        self.assertTrue(np.all(init[:-1] > 0))
        # The two base blocks start from the same length-scales.
        np.testing.assert_allclose(init[2:6], init[7:11])
        self.assertEqual(init[-1], 1.0)  # rho

    def test_length_scales_follow_the_point_spacing(self):
        narrow, y = self._coords(scale=1.0)
        wide, _ = self._coords(scale=10.0)
        num_cov_pars = 12
        init_narrow = heuristic_init_cov_pars(narrow, y, num_cov_pars, seed=0)
        init_wide = heuristic_init_cov_pars(wide, y, num_cov_pars, seed=0)
        ratio = init_wide[2:6] / init_narrow[2:6]
        np.testing.assert_allclose(ratio, 10.0, rtol=1e-6)

    def test_length_scales_exceed_typical_point_separation(self):
        """The bug being fixed: ranges far below the spacing kill the kernel."""
        coords, y = self._coords(n_dims=9)
        init = heuristic_init_cov_pars(coords, y, 4 + 2 * 9, seed=0)
        features = coords[:, :-1]
        distances = np.sqrt(
            ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1)
        )
        nearest = np.median(np.sort(distances, axis=1)[:, 1])
        self.assertGreater(init[2:11].min(), nearest)

    def test_rejects_a_parameter_count_that_is_not_ar1_mf(self):
        coords, y = self._coords()
        with self.assertRaises(ValueError):
            heuristic_init_cov_pars(coords, y, 7, seed=0)


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
class GPBoostHFOnlyTests(unittest.TestCase):
    def test_predict_shapes_and_hyperparameters(self):
        _, _, X_hf, Y_hf, _ = _two_fidelity_data()
        wavelengths = np.array([1.0])
        layer = fit_hf_only_gpboost(X_hf, Y_hf, wavelengths, seed=0)
        means, stds = predict_hf_only_gpboost(layer, X_hf[:5])
        self.assertEqual(means.shape, (5, 1))
        self.assertEqual(stds.shape, (5, 1))
        self.assertTrue(np.all(stds > 0))

        table = hyperparameter_table_hf_only_gpboost(layer)
        self.assertEqual(len(table), 1)
        for column in ("wavelength", "signal_variance", "error_var"):
            self.assertIn(column, table.columns)

    def test_cv_produces_one_prediction_per_sample(self):
        _, _, X_hf, Y_hf, _ = _two_fidelity_data()
        predictions = cv_predict_hf_only_gpboost(
            X_hf, Y_hf, np.array([1.0]), seed=0, n_splits=3
        )
        self.assertEqual(predictions.y_pred.shape, Y_hf.shape)
        self.assertFalse(np.isnan(predictions.y_pred).any())
        metrics = cv_metrics(Y_hf, predictions)
        self.assertTrue(np.isfinite(metrics.rmse_pooled))
        self.assertIsNotNone(metrics.coverage_95)


def _shared_rho_data(seed: int = 0, n_lambda: int = 3):
    """Multi-wavelength AR(1) data where every wavelength shares one rho."""
    rng = np.random.default_rng(seed)
    p = 2
    rho_true = 0.7
    X_lf = rng.uniform(0, 1, size=(150, p))
    X_hf = rng.uniform(0, 1, size=(40, p))

    def f_low(X, j):
        return np.sin(3 * X[:, 0] + j) + 0.5 * X[:, 1] ** 2

    Y_lf = np.column_stack(
        [f_low(X_lf, j) + 0.01 * rng.standard_normal(len(X_lf)) for j in range(n_lambda)]
    )
    Y_hf = np.column_stack(
        [
            rho_true * f_low(X_hf, j)
            + 0.3 * np.cos(4 * X_hf[:, 1] + j)
            + 0.01 * rng.standard_normal(len(X_hf))
            for j in range(n_lambda)
        ]
    )
    wavelengths = np.linspace(2.5, 5.0, n_lambda)
    return X_lf, Y_lf, X_hf, Y_hf, wavelengths, rho_true


@unittest.skipUnless(gpboost_available(), "gpboost not installed")
class GPBoostModel1ATests(unittest.TestCase):
    def test_shares_one_rho_across_wavelengths(self):
        X_lf, Y_lf, X_hf, Y_hf, wavelengths, rho_true = _shared_rho_data()
        layer = fit_model1a_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, seed=0, subsample_size=100
        )
        self.assertEqual(layer.rho.shape, (len(wavelengths),))
        # Every wavelength must carry the identical shared rho.
        np.testing.assert_allclose(layer.rho, layer.rho[0], rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(layer.rho[0]), rho_true, delta=0.2)

    def test_sweeps_do_not_decrease_the_joint_likelihood(self):
        X_lf, Y_lf, X_hf, Y_hf, wavelengths, _ = _shared_rho_data(seed=1)
        layer = fit_model1a_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, seed=0, subsample_size=100
        )
        history = layer.global_rho_sweeps
        self.assertGreaterEqual(len(history), 1)
        summed = [record["summed_log_marginal_likelihood"] for record in history]
        for before, after in zip(summed, summed[1:]):
            self.assertGreaterEqual(after, before - 1e-6)
        # The sweep converged rather than exhausting its budget.
        self.assertLess(history[-1]["abs_delta_rho"], 1e-3)

    def test_warm_start_layer_must_match_the_design(self):
        X_lf, Y_lf, X_hf, Y_hf, wavelengths, _ = _shared_rho_data()
        warm = fit_model1_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, seed=0, subsample_size=100
        )
        with self.assertRaises(ValueError):
            fit_model1a_gpboost(
                X_lf, Y_lf, X_hf, Y_hf, wavelengths,
                seed=0, subsample_size=80, warm_start_layer=warm,
            )

    def test_cv_produces_one_prediction_per_sample(self):
        X_lf, Y_lf, X_hf, Y_hf, wavelengths, _ = _shared_rho_data(n_lambda=2)
        predictions = cv_predict_model1a_gpboost(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, subsample_size=80, n_splits=3,
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
