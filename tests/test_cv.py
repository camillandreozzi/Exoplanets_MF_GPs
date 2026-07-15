"""Unit tests for the cross-validation model-comparison framework."""

from __future__ import annotations

import unittest

import numpy as np

from exoplanets_mf.cv import (
    compare_cv,
    cv_metrics,
    cv_predict,
    cv_predict_joint_mf_gp,
)
from exoplanets_mf.mf_gp import N_INPUT_DIMS


def _paired_data(seed: int = 1, n_samples: int = 60, n_wavelengths: int = 8):
    rng = np.random.default_rng(seed)
    YLF = rng.normal(size=(n_samples, n_wavelengths)) * np.arange(1, n_wavelengths + 1)
    YHF = 0.7 * YLF + rng.normal(scale=1e-2, size=YLF.shape)
    return YHF, YLF


class _TrainMeanModel:
    """Minimal leakage-detectable model: predicts the training-fold mean."""

    def __init__(self, Y_train: np.ndarray):
        self.mean = Y_train.mean(axis=0)

    def predict(self, n_test: int) -> np.ndarray:
        return np.tile(self.mean, (n_test, 1))


class CVEngineTests(unittest.TestCase):
    def test_every_sample_predicted_exactly_once(self) -> None:
        YHF, _ = _paired_data()
        fold_fit_counts = []

        def fit_fn(train_idx):
            fold_fit_counts.append(len(train_idx))
            return _TrainMeanModel(YHF[train_idx])

        predictions = cv_predict(
            fit_fn=fit_fn,
            predict_fn=lambda model, test_idx: model.predict(len(test_idx)),
            n_samples=YHF.shape[0],
            n_wavelengths=YHF.shape[1],
            n_splits=5,
            seed=0,
        )

        self.assertEqual(predictions.y_pred.shape, YHF.shape)
        self.assertFalse(np.isnan(predictions.y_pred).any())
        self.assertEqual(len(fold_fit_counts), 5)
        # Every fold trained without its held-out samples.
        self.assertTrue(all(count < YHF.shape[0] for count in fold_fit_counts))
        # Each fold index appears, and every sample belongs to exactly one fold.
        self.assertEqual(set(predictions.fold_of_sample), set(range(5)))

    def test_std_passthrough_and_none(self) -> None:
        YHF, _ = _paired_data()

        with_std = cv_predict(
            fit_fn=lambda train_idx: None,
            predict_fn=lambda model, test_idx: (
                YHF[test_idx], np.full_like(YHF[test_idx], 0.5)
            ),
            n_samples=YHF.shape[0],
            n_wavelengths=YHF.shape[1],
            n_splits=4,
            seed=0,
        )
        self.assertIsNotNone(with_std.y_std)
        np.testing.assert_allclose(with_std.y_std, 0.5)

        without_std = cv_predict(
            fit_fn=lambda train_idx: None,
            predict_fn=lambda model, test_idx: YHF[test_idx],
            n_samples=YHF.shape[0],
            n_wavelengths=YHF.shape[1],
            n_splits=4,
            seed=0,
        )
        self.assertIsNone(without_std.y_std)

    def test_predictions_can_be_revalued_without_changing_cv_assignment(self) -> None:
        YHF, _ = _paired_data()
        predictions = cv_predict(
            fit_fn=lambda train_idx: None,
            predict_fn=lambda model, test_idx: YHF[test_idx],
            n_samples=YHF.shape[0],
            n_wavelengths=YHF.shape[1],
            n_splits=5,
            seed=0,
        )

        shifted = predictions.with_values(predictions.y_pred + 1.0)
        np.testing.assert_allclose(shifted.y_pred, predictions.y_pred + 1.0)
        np.testing.assert_array_equal(
            shifted.fold_of_sample, predictions.fold_of_sample
        )
        self.assertEqual(shifted.n_splits, predictions.n_splits)
        self.assertIsNone(shifted.y_std)

        with self.assertRaises(ValueError):
            predictions.with_values(predictions.y_pred[:, :1])

    def test_deterministic_for_fixed_seed(self) -> None:
        YHF, _ = _paired_data()

        def run():
            return cv_predict(
                fit_fn=lambda train_idx: _TrainMeanModel(YHF[train_idx]),
                predict_fn=lambda model, test_idx: model.predict(len(test_idx)),
                n_samples=YHF.shape[0],
                n_wavelengths=YHF.shape[1],
                n_splits=6,
                seed=3,
            )

        first, second = run(), run()
        np.testing.assert_array_equal(first.y_pred, second.y_pred)
        np.testing.assert_array_equal(first.fold_of_sample, second.fold_of_sample)


class JointMFGPAdapterTests(unittest.TestCase):
    """cv_predict_joint_mf_gp on tiny synthetic data (both rho criteria)."""

    def _tiny_mf_data(self):
        rng = np.random.default_rng(0)
        X_lf = rng.normal(size=(14, N_INPUT_DIMS))
        X_hf = X_lf[:10]
        Y_lf = np.column_stack((np.sin(X_lf[:, 0]), np.cos(X_lf[:, 1])))
        Y_hf = 1.5 * Y_lf[:10] + rng.normal(scale=1e-2, size=(10, 2))
        return X_lf, Y_lf, X_hf, Y_hf, np.array([1.0, 2.0])

    def test_adapter_covers_every_sample_for_both_rho_criteria(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = self._tiny_mf_data()
        for per_wavelength in (True, False):
            with self.subTest(per_wavelength_rho=per_wavelength):
                predictions = cv_predict_joint_mf_gp(
                    X_lf,
                    Y_lf,
                    X_hf,
                    Y_hf,
                    wavelengths,
                    per_wavelength_rho=per_wavelength,
                    subsample_size=len(X_lf),
                    n_splits=2,
                    seed=0,
                )
                self.assertEqual(predictions.y_pred.shape, Y_hf.shape)
                self.assertFalse(np.isnan(predictions.y_pred).any())
                self.assertIsNotNone(predictions.y_std)
                self.assertEqual(set(predictions.fold_of_sample), {0, 1})


class ModelComparisonTests(unittest.TestCase):
    def test_compare_cv_paired_deltas(self) -> None:
        YHF, _ = _paired_data()

        def constant_predictions(offset: float):
            return cv_predict(
                fit_fn=lambda train_idx: None,
                predict_fn=lambda model, test_idx: YHF[test_idx] + offset,
                n_samples=YHF.shape[0],
                n_wavelengths=YHF.shape[1],
                n_splits=5,
                seed=0,
            )

        worse, better = constant_predictions(0.1), constant_predictions(0.01)
        comparison = compare_cv(YHF, worse, better, labels=("worse", "better"))
        # delta = model 2 - model 1, negative means model 2 wins.
        self.assertLess(comparison.delta_mean, 0.0)
        self.assertLess(comparison.delta_rmse_pooled, 0.0)
        self.assertEqual(comparison.fraction_samples_model2_wins, 1.0)

    def test_metrics_shapes_and_coverage(self) -> None:
        YHF, _ = _paired_data()
        predictions = cv_predict(
            fit_fn=lambda train_idx: None,
            predict_fn=lambda model, test_idx: (
                YHF[test_idx] + 0.01, np.full_like(YHF[test_idx], 1.0)
            ),
            n_samples=YHF.shape[0],
            n_wavelengths=YHF.shape[1],
            n_splits=5,
            seed=0,
        )
        metrics = cv_metrics(YHF, predictions)
        self.assertEqual(metrics.rmse_per_wavelength.shape, (YHF.shape[1],))
        self.assertEqual(metrics.rmse_per_sample.shape, (YHF.shape[0],))
        self.assertAlmostEqual(metrics.rmse_pooled, 0.01, places=10)
        # |z| = 0.01 <= 1.96 everywhere -> full coverage.
        self.assertEqual(metrics.coverage_95, 1.0)


if __name__ == "__main__":
    unittest.main()
