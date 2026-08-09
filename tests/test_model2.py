"""Unit tests for the wavelength-augmented joint MF-GP (Model 2)."""

from __future__ import annotations

import unittest

import numpy as np

from exoplanets_mf.cv import cv_predict
from exoplanets_mf.mf_gp import (
    assert_exact_fit_fits_in_memory,
    exact_fit_memory_bytes,
    fit_joint_mf_gp,
    select_lf_subsample,
)
from exoplanets_mf.model2 import (
    N_AUGMENTED_DIMS,
    build_augmented_design,
    cv_predict_model2,
    derive_lambda_stride,
    fit_model2,
    lf_column_moments,
    model2_n_theta,
    predict_hf_model2,
    select_wavelength_subgrid,
)
from exoplanets_mf.reproducibility import LF_SUBSAMPLE_SIZE, RANDOM_SEED


def _toy_mf_problem(seed: int = 0, rho_true: float = 0.7):
    """Smooth separable-ish MF surfaces on a small (theta, lambda) grid.

    2-dim theta keeps every joint fit at a few hundred points and seconds of
    wall time; the structure (y_H = rho * y_L + smooth delta) is what
    fit_model2 assumes, so the fitted scalar rho must recover rho_true.
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
    Y_hf = rho_true * lf_surface(X_hf) + delta_surface(X_hf)
    return X_lf, Y_lf, X_hf, Y_hf, wavelengths


class WavelengthSubgridTests(unittest.TestCase):
    def test_stride_and_offset(self) -> None:
        np.testing.assert_array_equal(
            select_wavelength_subgrid(10, stride=3), [0, 3, 6, 9]
        )
        np.testing.assert_array_equal(
            select_wavelength_subgrid(10, stride=3, offset=1), [1, 4, 7]
        )

    def test_stride_one_is_the_full_grid(self) -> None:
        np.testing.assert_array_equal(
            select_wavelength_subgrid(5, stride=1), np.arange(5)
        )

    def test_invalid_stride_and_offset_raise(self) -> None:
        with self.assertRaises(ValueError):
            select_wavelength_subgrid(10, stride=0)
        with self.assertRaises(ValueError):
            select_wavelength_subgrid(10, stride=3, offset=3)
        with self.assertRaises(ValueError):
            select_wavelength_subgrid(10, stride=3, offset=-1)


class DeriveLambdaStrideTests(unittest.TestCase):
    @staticmethod
    def _n_total(n_lf, n_hf, n_w, stride):
        offset = (stride // 2) % stride
        return (
            n_lf * len(select_wavelength_subgrid(n_w, stride=stride, offset=offset))
            + n_hf * len(select_wavelength_subgrid(n_w, stride=stride))
        )

    def test_returns_smallest_stride_within_budget(self) -> None:
        n_lf, n_hf, n_w, budget = 200, 97, 195, 3400
        stride = derive_lambda_stride(n_lf, n_hf, n_w, max_points=budget)
        # within budget ...
        self.assertLessEqual(self._n_total(n_lf, n_hf, n_w, stride), budget)
        # ... and the next-finer grid (stride - 1) would exceed it
        self.assertGreater(self._n_total(n_lf, n_hf, n_w, stride - 1), budget)

    def test_raises_when_no_stride_fits(self) -> None:
        with self.assertRaises(ValueError):
            derive_lambda_stride(10_000, 10_000, 195, max_points=1)


class SharedSubsampleTests(unittest.TestCase):
    """Every model must draw the ONE canonical LF subsample."""

    def test_select_lf_subsample_is_deterministic(self) -> None:
        a = select_lf_subsample(1000, LF_SUBSAMPLE_SIZE, seed=RANDOM_SEED)
        b = select_lf_subsample(1000, LF_SUBSAMPLE_SIZE, seed=RANDOM_SEED)
        self.assertEqual(a.shape, (LF_SUBSAMPLE_SIZE,))
        np.testing.assert_array_equal(a, b)

    def test_model1_and_model2_pick_identical_lf_rows(self) -> None:
        """Given the same LF pool, size and seed, Model 1 and Model 2 select
        the identical LF rows -- the enforced shared subsample."""
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem()
        size = 16
        model1 = fit_joint_mf_gp(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=RANDOM_SEED, subsample_size=size, n_restarts_optimizer=0,
        )
        model2 = fit_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=RANDOM_SEED, lf_sample_size=size, lambda_stride=2,
            n_restarts_optimizer=0,
        )
        np.testing.assert_array_equal(
            model1.subsample_indices, model2.lf_sample_indices
        )
        np.testing.assert_array_equal(
            model1.subsample_indices,
            select_lf_subsample(X_lf.shape[0], size, seed=RANDOM_SEED),
        )


class AugmentedDesignTests(unittest.TestCase):
    def test_shapes_ordering_and_lambda_column(self) -> None:
        X = np.arange(12, dtype=float).reshape(4, 3)
        Y = np.arange(20, dtype=float).reshape(4, 5)
        wavelengths = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
        sample_indices = np.array([2, 0])
        lambda_indices = np.array([1, 3])

        Z, y = build_augmented_design(
            X, Y, wavelengths,
            sample_indices=sample_indices, lambda_indices=lambda_indices,
        )

        self.assertEqual(Z.shape, (4, 4))  # 2 samples x 2 lambdas, 3+1 columns
        # sample-major: sample 2 at lambdas (3, 7), then sample 0
        np.testing.assert_array_equal(Z[:, -1], [3.0, 7.0, 3.0, 7.0])
        np.testing.assert_array_equal(Z[0, :-1], X[2])
        np.testing.assert_array_equal(Z[2, :-1], X[0])
        np.testing.assert_array_equal(y, [Y[2, 1], Y[2, 3], Y[0, 1], Y[0, 3]])

    def test_augmented_dimension_constant_matches_data(self) -> None:
        self.assertEqual(N_AUGMENTED_DIMS, 10)
        self.assertEqual(model2_n_theta(), 25)


class ColumnMomentsTests(unittest.TestCase):
    def test_round_trip_and_lf_only(self) -> None:
        rng = np.random.default_rng(3)
        Y_lf = 2.0 + 3.0 * rng.normal(size=(40, 6))
        mu, sd = lf_column_moments(Y_lf)
        np.testing.assert_allclose(mu, Y_lf.mean(axis=0))
        np.testing.assert_allclose(sd, Y_lf.std(axis=0))
        standardized = (Y_lf - mu) / sd
        np.testing.assert_allclose(standardized * sd + mu, Y_lf, rtol=1e-12)

    def test_zero_variance_column_raises(self) -> None:
        Y_lf = np.ones((10, 3))
        Y_lf[:, :2] = np.random.default_rng(0).normal(size=(10, 2))
        with self.assertRaises(ValueError):
            lf_column_moments(Y_lf)


class FitModel2Tests(unittest.TestCase):
    def test_recovers_known_rho_and_interpolates_lambda(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem(rho_true=0.7)
        layer = fit_model2(
            X_lf, Y_lf, X_hf[:9], Y_hf[:9], wavelengths,
            seed=0, lf_sample_size=24, lambda_stride=2,
            n_restarts_optimizer=0,
        )

        self.assertAlmostEqual(layer.rho, 0.7, delta=0.15)

        # held-out samples, ALL wavelengths -- half were never trained on
        means, stds = predict_hf_model2(layer, X_hf[9:])
        self.assertEqual(means.shape, (3, len(wavelengths)))
        self.assertEqual(stds.shape, (3, len(wavelengths)))
        self.assertTrue(np.all(np.isfinite(means)))
        self.assertTrue(np.all(stds > 0))
        rmse = float(np.sqrt(((means - Y_hf[9:]) ** 2).mean()))
        spread = float(Y_hf.max() - Y_hf.min())
        self.assertLess(rmse / spread, 0.1)

    def test_design_composition_and_lambda_offset(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem()
        layer = fit_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, lf_sample_size=10, lambda_stride=4,
            n_restarts_optimizer=0,
        )

        np.testing.assert_array_equal(layer.hf_lambda_indices, [0, 4, 8])
        np.testing.assert_array_equal(layer.lf_lambda_indices, [2, 6, 10])
        n_expected = 10 * 3 + len(X_hf) * 3
        self.assertEqual(layer.model.X_train_.shape, (n_expected, 4))
        # fidelity flag is the last column: 0 for the LF block, 1 for HF
        np.testing.assert_array_equal(
            layer.model.X_train_[:, -1],
            np.concatenate((np.zeros(30), np.ones(len(X_hf) * 3))),
        )

    def test_standardization_leaves_rho_invariant(self) -> None:
        """A common per-column scale on both fidelities cancels in the AR(1)
        relation, so fits with and without per-wavelength standardization must
        agree on rho for well-scaled toy data."""
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem(rho_true=0.7)
        kwargs = dict(
            seed=0, lf_sample_size=24, lambda_stride=2, n_restarts_optimizer=0
        )
        with_std = fit_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, **kwargs
        )
        without_std = fit_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            standardize_per_wavelength=False, **kwargs,
        )
        self.assertIsNone(without_std.column_mu)
        self.assertAlmostEqual(with_std.rho, without_std.rho, delta=0.1)

    def test_memory_gate_blocks_oversized_designs(self) -> None:
        """The Model 2 hyperparameter count feeds the same gate as Model 1."""
        n_theta = model2_n_theta()
        n_points = 20_000  # the full HF grid alone already exceeds any budget
        required = exact_fit_memory_bytes(n_points, n_theta)
        with self.assertRaises(MemoryError) as context:
            assert_exact_fit_fits_in_memory(
                n_points, n_theta, available_bytes=required // 2
            )
        self.assertIn(str(n_points), str(context.exception))


class PredictModel2Tests(unittest.TestCase):
    def test_prediction_composition_matches_manual_unstandardization(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem()
        layer = fit_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, lf_sample_size=12, lambda_stride=3,
            n_restarts_optimizer=0,
        )
        means, stds = predict_hf_model2(layer, X_hf[:2])

        n_lambdas = len(wavelengths)
        Z = np.empty((2 * n_lambdas, 3))
        Z[:, :-1] = np.repeat(X_hf[:2], n_lambdas, axis=0)
        Z[:, -1] = np.tile(wavelengths, 2)
        Z_scaled = layer.scaler.transform(Z)
        Z_augmented = np.column_stack((Z_scaled, np.ones(len(Z_scaled))))
        raw_mean, raw_std = layer.model.predict(Z_augmented, return_std=True)

        np.testing.assert_allclose(
            means,
            raw_mean.reshape(2, n_lambdas) * layer.column_sd + layer.column_mu,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            stds, raw_std.reshape(2, n_lambdas) * layer.column_sd, rtol=1e-12
        )


class CVAdapterTests(unittest.TestCase):
    def _cv_kwargs(self):
        return dict(
            seed=0,
            n_splits=3,
            lf_sample_size=16,
            lambda_stride=3,
            n_restarts_optimizer=0,
        )

    def test_shapes_stds_and_determinism(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem()
        predictions = cv_predict_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, **self._cv_kwargs()
        )
        repeat = cv_predict_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, **self._cv_kwargs()
        )

        self.assertEqual(predictions.y_pred.shape, Y_hf.shape)
        self.assertFalse(np.isnan(predictions.y_pred).any())
        self.assertIsNotNone(predictions.y_std)
        self.assertTrue(np.all(predictions.y_std > 0))
        np.testing.assert_array_equal(predictions.y_pred, repeat.y_pred)

    def test_fold_assignment_matches_plain_cv_predict(self) -> None:
        """Identical (n_samples, n_splits, seed) must give identical folds,
        otherwise paired comparisons against Model 1 are invalid."""
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem()
        predictions = cv_predict_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, **self._cv_kwargs()
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
        samples in other folds (whole-spectrum holdout, no lambda leakage)."""
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = _toy_mf_problem()
        predictions = cv_predict_model2(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths, **self._cv_kwargs()
        )

        corrupted_sample = 0
        Y_hf_corrupted = Y_hf.copy()
        Y_hf_corrupted[corrupted_sample] += 1e3
        corrupted = cv_predict_model2(
            X_lf, Y_lf, X_hf, Y_hf_corrupted, wavelengths, **self._cv_kwargs()
        )

        same_fold = (
            predictions.fold_of_sample
            == predictions.fold_of_sample[corrupted_sample]
        )
        np.testing.assert_array_equal(
            predictions.y_pred[same_fold], corrupted.y_pred[same_fold]
        )


if __name__ == "__main__":
    unittest.main()
