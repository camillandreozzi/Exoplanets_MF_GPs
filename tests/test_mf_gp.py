"""Unit tests for the joint AR(1) multi-fidelity GP layers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from exoplanets_mf.data import load_all
from exoplanets_mf.mf_gp import (
    N_INPUT_DIMS,
    assert_exact_fit_fits_in_memory,
    assert_holdout_row,
    exact_fit_memory_bytes,
    fit_joint_mf_gp,
    fit_joint_mf_gp_global_rho,
    make_joint_mf_kernel,
    predict_fidelity,
    predict_hf,
)


class JointMultiFidelityGPTests(unittest.TestCase):
    def _fit_toy_layer(self):
        """Two wavelengths and tiny joint LF/HF data for a fast fit."""
        rng = np.random.default_rng(0)
        X_lf = rng.normal(size=(16, N_INPUT_DIMS))
        X_hf = X_lf[:8]
        Y_lf = np.column_stack(
            (np.sin(X_lf[:, 0]), np.cos(X_lf[:, 1]))
        )
        Y_hf = 1.5 * Y_lf[:8] + np.column_stack(
            (0.1 * X_hf[:, 2], -0.1 * X_hf[:, 3])
        )
        layer = fit_joint_mf_gp(
            X_lf,
            Y_lf,
            X_hf,
            Y_hf,
            np.array([1.0, 2.0]),
            seed=0,
            subsample_size=len(X_lf),
            n_restarts_optimizer=0,
        )
        return layer, X_lf

    def test_kernel_has_expected_block_covariance(self) -> None:
        kernel = make_joint_mf_kernel(1)
        kernel.rho = 2.0
        X = np.array([[0.0, 0.0], [0.0, 1.0]])
        covariance = kernel(X)

        low_variance = kernel.low_kernel(np.array([[0.0]]))[0, 0]
        discrepancy_variance = kernel.discrepancy_kernel(
            np.array([[0.0]])
        )[0, 0]
        self.assertAlmostEqual(covariance[0, 1], 2.0 * low_variance)
        self.assertAlmostEqual(
            covariance[1, 1] - kernel.high_noise,
            4.0 * low_variance + discrepancy_variance,
        )

    def test_joint_fit_contains_both_fidelities(self) -> None:
        layer, _ = self._fit_toy_layer()
        self.assertEqual(layer.models[0].X_train_.shape[0], 16 + 8)
        self.assertEqual(set(layer.models[0].X_train_[:, -1]), {0.0, 1.0})

    def test_exact_fit_uses_all_lf_rows_and_records_timing(self) -> None:
        """subsample_size=None (the default) must be the exact fit."""
        rng = np.random.default_rng(1)
        X_lf = rng.normal(size=(12, N_INPUT_DIMS))
        X_hf = X_lf[:5]
        Y_lf = np.sin(X_lf[:, :1])
        Y_hf = 1.5 * Y_lf[:5] + 0.1 * X_hf[:, 2:3]

        layer = fit_joint_mf_gp(
            X_lf, Y_lf, X_hf, Y_hf, np.array([1.0]),
            seed=0, n_restarts_optimizer=0,
        )

        self.assertIsNone(layer.subsample_indices)
        self.assertEqual(layer.models[0].X_train_.shape[0], 12 + 5)
        self.assertEqual(layer.fit_seconds.shape, (1,))
        self.assertGreater(layer.fit_seconds[0], 0.0)

    def test_predict_fidelity_shapes(self) -> None:
        layer, X = self._fit_toy_layer()
        mean, std = predict_fidelity(layer, X[:5], fidelity=0)

        self.assertEqual(mean.shape, (5, 2))
        self.assertEqual(std.shape, (5, 2))
        self.assertTrue(np.all(std >= 0))

    def test_predict_hf_needs_no_lf_output(self) -> None:
        layer, X = self._fit_toy_layer()
        mean, std = predict_hf(layer, X[:4])

        self.assertEqual(mean.shape, (4, 2))
        self.assertEqual(std.shape, (4, 2))


class GlobalRhoJointFitTests(unittest.TestCase):
    """Model 1A: one shared rho against the summed marginal likelihood."""

    def _toy_data(self):
        rng = np.random.default_rng(0)
        X_lf = rng.normal(size=(16, N_INPUT_DIMS))
        X_hf = X_lf[:8]
        Y_lf = np.column_stack((np.sin(X_lf[:, 0]), np.cos(X_lf[:, 1])))
        Y_hf = 1.5 * Y_lf[:8] + np.column_stack(
            (0.1 * X_hf[:, 2], -0.1 * X_hf[:, 3])
        )
        return X_lf, Y_lf, X_hf, Y_hf, np.array([1.0, 2.0])

    def _fit_global(self, **kwargs):
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = self._toy_data()
        return fit_joint_mf_gp_global_rho(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, subsample_size=len(X_lf), n_restarts_optimizer=0,
            **kwargs,
        )

    def test_all_wavelengths_share_one_fixed_rho(self) -> None:
        layer = self._fit_global()
        rho = layer.rho
        self.assertEqual(rho.shape, (2,))
        self.assertEqual(rho[0], rho[1])
        for model in layer.models:
            self.assertEqual(model.kernel_.rho_bounds, "fixed")

    def test_sweeps_monotonically_increase_joint_likelihood(self) -> None:
        layer = self._fit_global()
        lml = [
            sweep["summed_log_marginal_likelihood"]
            for sweep in layer.global_rho_sweeps
        ]
        self.assertGreater(len(lml), 0)
        for earlier, later in zip(lml, lml[1:]):
            self.assertGreaterEqual(later, earlier - 1e-6)

    def test_warm_start_from_per_wavelength_layer(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = self._toy_data()
        warm = fit_joint_mf_gp(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, subsample_size=len(X_lf), n_restarts_optimizer=0,
        )
        layer = self._fit_global(warm_start_layer=warm)
        self.assertEqual(layer.rho[0], layer.rho[1])
        mean, std = predict_hf(layer, X_lf[:3])
        self.assertEqual(mean.shape, (3, 2))
        self.assertTrue(np.all(std >= 0))

    def test_warm_start_rejects_mismatched_subsample(self) -> None:
        X_lf, Y_lf, X_hf, Y_hf, wavelengths = self._toy_data()
        warm = fit_joint_mf_gp(
            X_lf, Y_lf, X_hf, Y_hf, wavelengths,
            seed=0, subsample_size=10, n_restarts_optimizer=0,
        )
        with self.assertRaises(ValueError):
            fit_joint_mf_gp_global_rho(
                X_lf, Y_lf, X_hf, Y_hf, wavelengths,
                seed=0, subsample_size=12, warm_start_layer=warm,
            )


class ExactFitMemoryGateTests(unittest.TestCase):
    def test_memory_estimate_grows_quadratically(self) -> None:
        small = exact_fit_memory_bytes(100, 23)
        large = exact_fit_memory_bytes(200, 23)
        self.assertGreater(small, 0)
        self.assertEqual(large, 4 * small)

    def test_gate_passes_within_budget(self) -> None:
        # 100 points is tiny; must pass against any sane budget.
        assert_exact_fit_fits_in_memory(
            100, 23, available_bytes=1_000_000_000
        )

    def test_gate_raises_with_numbers_when_over_budget(self) -> None:
        with self.assertRaises(MemoryError) as context:
            assert_exact_fit_fits_in_memory(
                10_097, 23, available_bytes=16 * 1024**3
            )
        message = str(context.exception)
        self.assertIn("10097", message)
        self.assertIn("subsample_size", message)

    def test_gate_skips_check_when_ram_unknown(self) -> None:
        assert_exact_fit_fits_in_memory(10_097, 23, available_bytes=None)


class HoldoutRowAssertionTests(unittest.TestCase):
    def _toy_frames(self):
        XHF = pd.DataFrame({"Kzz": [1.0, 2.0, 3.0]})
        YHF = pd.DataFrame({"a": [1, 2, 3]}, index=["spectrum 1", "spectrum 2", "spectrum 3"])
        return XHF, YHF

    def test_assert_holdout_row_passes_on_matching_row(self) -> None:
        XHF, YHF = self._toy_frames()
        assert_holdout_row(XHF, YHF, 1, expected_kzz=2.0, expected_label="spectrum 2")

    def test_assert_holdout_row_raises_on_kzz_mismatch(self) -> None:
        XHF, YHF = self._toy_frames()
        with self.assertRaises(ValueError):
            assert_holdout_row(XHF, YHF, 1, expected_kzz=99.0, expected_label="spectrum 2")

    def test_assert_holdout_row_raises_on_label_mismatch(self) -> None:
        XHF, YHF = self._toy_frames()
        with self.assertRaises(ValueError):
            assert_holdout_row(XHF, YHF, 1, expected_kzz=2.0, expected_label="spectrum 99")

    def test_assert_holdout_row_matches_real_project_data(self) -> None:
        """Integration-style check against the real files -- guards the exact
        row-80 / 'spectrum 81' / Kzz=8.47701413791753 alignment the leave-
        one-out holdout validation depends on."""
        data = load_all()
        assert_holdout_row(
            data["XHF"], data["YHF"], 80,
            expected_kzz=8.47701413791753, expected_label="spectrum 81",
        )


if __name__ == "__main__":
    unittest.main()
