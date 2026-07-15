"""Unit tests for the exact two-stage Kronecker MF-GP (Model 2K)."""

from __future__ import annotations

import unittest

import numpy as np

from exoplanets_mf.cv import cv_predict
from exoplanets_mf.kron_gp import posterior_mean_var
from exoplanets_mf.model2_kron import (
    Model2KronLayer,
    build_stage_l_design,
    cv_predict_model2_kron,
    fit_model2_kron,
    fit_stage_l,
    predict_hf_model2_kron,
)


def _toy_mf_problem(seed: int = 0, rho_true: float = 0.7):
    """Smooth separable MF surfaces on a small complete (theta, lambda) grid.

    Same structure as the Model 2 toy (y_H = rho * y_L + smooth delta), plus
    the paired LF spectra at the HF inputs that the recursion consumes.
    """
    rng = np.random.default_rng(seed)
    n_lf, n_hf, n_lambdas = 30, 12, 12
    X_lf = rng.uniform(-1, 1, size=(n_lf, 2))
    X_hf = rng.uniform(-1, 1, size=(n_hf, 2))
    wavelengths = np.linspace(2.4, 12.0, n_lambdas)

    def lf_surface(X):
        theta_part = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1]
        lambda_part = 1.0 + 0.3 * np.cos(wavelengths / 3.0)
        return np.outer(theta_part, lambda_part) + 2.0

    def delta_surface(X):
        return 0.05 * np.outer(X[:, 0], np.sin(wavelengths / 2.0))

    Y_lf = lf_surface(X_lf)
    Y_lf_paired = lf_surface(X_hf)
    Y_hf = rho_true * Y_lf_paired + delta_surface(X_hf)
    return X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths


_FIT_KWARGS = dict(seed=0, n_restarts=2, screen_size=None)


class StageLDesignTests(unittest.TestCase):
    def test_stacked_design_is_nested(self) -> None:
        """Every HF theta (with its OBSERVED paired LF spectrum) must be in
        the stage-L design -- nestedness is what makes the recursion exact."""
        X_lf, Y_lf, X_hf, _, Y_lf_paired, _ = _toy_mf_problem()
        X, Y = build_stage_l_design(X_lf, Y_lf, X_hf, Y_lf_paired)
        self.assertEqual(X.shape, (len(X_lf) + len(X_hf), 2))
        np.testing.assert_array_equal(X[len(X_lf):], X_hf)
        np.testing.assert_array_equal(Y[len(X_lf):], Y_lf_paired)

    def test_duplicate_theta_rows_are_rejected(self) -> None:
        X_lf, Y_lf, X_hf, _, Y_lf_paired, _ = _toy_mf_problem()
        X_hf_dup = X_hf.copy()
        X_hf_dup[0] = X_lf[3]
        with self.assertRaises(ValueError):
            build_stage_l_design(X_lf, Y_lf, X_hf_dup, Y_lf_paired)

    def test_screen_then_polish_fits_full_grid(self) -> None:
        """Screening restarts run on the subset, but the returned GP must be
        fitted on ALL stacked rows."""
        X_lf, Y_lf, X_hf, _, Y_lf_paired, wavelengths = _toy_mf_problem()
        stage_l = fit_stage_l(
            X_lf, Y_lf, X_hf, Y_lf_paired, wavelengths,
            seed=0, n_restarts=2, screen_size=20,
        )
        self.assertEqual(stage_l.gp.X.shape[0], len(X_lf) + len(X_hf))
        self.assertEqual(len(stage_l.screen_records), 2)
        self.assertEqual(stage_l.screen_size, 20)


class FitModel2KronTests(unittest.TestCase):
    def test_recovers_known_rho_and_predicts_holdout(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem(
            rho_true=0.7
        )
        layer = fit_model2_kron(
            X_lf, Y_lf, X_hf[:9], Y_hf[:9], Y_lf_paired[:9], wavelengths,
            **_FIT_KWARGS,
        )
        self.assertAlmostEqual(layer.rho, 0.7, delta=0.15)

        means, stds = predict_hf_model2_kron(layer, X_hf[9:])
        self.assertEqual(means.shape, (3, len(wavelengths)))
        self.assertEqual(stds.shape, (3, len(wavelengths)))
        self.assertTrue(np.all(np.isfinite(means)))
        self.assertTrue(np.all(stds > 0))
        rmse = float(np.sqrt(((means - Y_hf[9:]) ** 2).mean()))
        spread = float(Y_hf.max() - Y_hf.min())
        self.assertLess(rmse / spread, 0.1)

    def test_prediction_composition_matches_manual_unstandardization(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem()
        layer = fit_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths, **_FIT_KWARGS
        )
        means, stds = predict_hf_model2_kron(layer, X_hf[:2])

        X_std = layer.stage_l.standardize_theta(X_hf[:2])
        mean_l, var_l = posterior_mean_var(layer.stage_l.gp, X_std)
        mean_d, var_d = posterior_mean_var(layer.stage_delta, X_std)
        mean_manual = (
            layer.rho * mean_l + mean_d
        ) * layer.stage_l.column_sd + layer.stage_l.column_mu
        std_manual = (
            np.sqrt(layer.rho**2 * var_l + var_d) * layer.stage_l.column_sd
        )
        np.testing.assert_allclose(means, mean_manual, rtol=1e-12)
        np.testing.assert_allclose(stds, std_manual, rtol=1e-12)

    def test_prefit_stage_l_is_reused_not_refit(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem()
        stage_l = fit_stage_l(
            X_lf, Y_lf, X_hf, Y_lf_paired, wavelengths,
            seed=0, n_restarts=2, screen_size=None,
        )
        layer = fit_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths,
            stage_l=stage_l, **_FIT_KWARGS,
        )
        self.assertIs(layer.stage_l, stage_l)


class CVAdapterTests(unittest.TestCase):
    def _cv_kwargs(self):
        return dict(seed=0, n_splits=3, n_restarts=2, screen_size=None)

    def test_shapes_stds_and_determinism(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem()
        predictions = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths, **self._cv_kwargs()
        )
        repeat = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths, **self._cv_kwargs()
        )
        self.assertEqual(predictions.y_pred.shape, Y_hf.shape)
        self.assertFalse(np.isnan(predictions.y_pred).any())
        self.assertIsNotNone(predictions.y_std)
        self.assertTrue(np.all(predictions.y_std > 0))
        np.testing.assert_array_equal(predictions.y_pred, repeat.y_pred)

    def test_fold_assignment_matches_plain_cv_predict(self) -> None:
        """Identical (n_samples, n_splits, seed) must give identical folds,
        otherwise paired comparisons against Models 1/2 are invalid."""
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem()
        predictions = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths, **self._cv_kwargs()
        )
        reference = cv_predict(
            fit_fn=lambda train_idx: None,
            predict_fn=lambda model, test_idx: np.zeros(
                (len(test_idx), Y_hf.shape[1])
            ),
            n_samples=Y_hf.shape[0],
            n_wavelengths=Y_hf.shape[1],
            n_splits=3,
            seed=0,
        )
        np.testing.assert_array_equal(
            predictions.fold_of_sample, reference.fold_of_sample
        )

    def test_holdout_hf_rows_never_influence_other_folds(self) -> None:
        """Corrupting one sample's HF spectrum must not move predictions of
        samples in other folds -- in particular the shared stage L must be
        HF-free (only stage delta sees Y_hf)."""
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem()
        predictions = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths, **self._cv_kwargs()
        )

        corrupted_sample = 0
        Y_hf_corrupted = Y_hf.copy()
        Y_hf_corrupted[corrupted_sample] += 1e3
        corrupted = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf_corrupted, Y_lf_paired, wavelengths,
            **self._cv_kwargs(),
        )

        same_fold = (
            predictions.fold_of_sample
            == predictions.fold_of_sample[corrupted_sample]
        )
        np.testing.assert_array_equal(
            predictions.y_pred[same_fold], corrupted.y_pred[same_fold]
        )

    def test_prefit_stage_l_gives_identical_predictions(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths = _toy_mf_problem()
        stage_l = fit_stage_l(
            X_lf, Y_lf, X_hf, Y_lf_paired, wavelengths,
            seed=0, n_restarts=2, screen_size=None,
        )
        with_prefit = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths,
            stage_l=stage_l, seed=0, n_splits=3, n_restarts=2,
        )
        internal = cv_predict_model2_kron(
            X_lf, Y_lf, X_hf, Y_hf, Y_lf_paired, wavelengths, **self._cv_kwargs()
        )
        np.testing.assert_array_equal(with_prefit.y_pred, internal.y_pred)
        np.testing.assert_array_equal(with_prefit.y_std, internal.y_std)


if __name__ == "__main__":
    unittest.main()
