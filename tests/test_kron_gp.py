"""Validation gates for the exact Kronecker grid GP (kron_gp).

Gate 1: LML / posterior mean / posterior variance match a dense Cholesky
reference on a small complete grid to rtol 1e-8, for both a plain stage and
a rho-profiled (stage-delta) fit. Gate 2: analytic gradients match central
finite differences. Gate 3: the row-major (i, j) -> i*m+j flattening matches
K_theta (x) K_lambda entrywise against a dense SE-ARD kernel over the
concatenated (theta, lambda) inputs.
"""

from __future__ import annotations

import pickle
import unittest

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from exoplanets_mf.kron_gp import (
    SEKernel1D,
    fit_kron_gp,
    kron_gp_at,
    kron_lml_and_grad,
    posterior_lambda_cov,
    posterior_mean_var,
    se_ard,
    se_ard_cross,
)


def _toy_grid(seed: int = 0, n: int = 20, m: int = 15, p: int = 3):
    """Random theta rows, a 1-d lambda grid, and a smooth n x m target."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n, p))
    t = np.linspace(-1.2, 1.2, m)
    theta_part = np.sin(X @ rng.uniform(0.5, 1.5, size=p))
    lambda_part = 1.0 + 0.4 * np.cos(2.0 * t)
    Y = np.outer(theta_part, lambda_part) + 0.01 * rng.normal(size=(n, m))
    return X, t, Y


def _log_eta(signal_variance, length_scales, lambda_length_scale, noise_variance):
    return np.log(
        np.concatenate(
            (
                [signal_variance],
                np.asarray(length_scales, dtype=float),
                [lambda_length_scale],
                [noise_variance],
            )
        )
    )


def _dense_covariance(X, t, log_eta):
    """sigma_f^2 kron(K_theta, K_lambda) + sigma_n^2 I from the same factors."""
    p = X.shape[1]
    signal_variance = np.exp(log_eta[0])
    length_scales = np.exp(log_eta[1 : 1 + p])
    lambda_length_scale = np.exp(log_eta[1 + p])
    noise_variance = np.exp(log_eta[-1])
    K_theta = se_ard(X, length_scales)
    K_lambda = se_ard(t.reshape(-1, 1), np.array([lambda_length_scale]))
    K = signal_variance * np.kron(K_theta, K_lambda)
    K[np.diag_indices_from(K)] += noise_variance
    return K


def _dense_lml(K, y):
    factor = cho_factor(K, lower=True)
    alpha = cho_solve(factor, y)
    logdet = 2.0 * np.log(np.diag(factor[0])).sum()
    return float(-0.5 * y @ alpha - 0.5 * logdet - 0.5 * len(y) * np.log(2 * np.pi))


class OrderingTests(unittest.TestCase):
    def test_row_major_flattening_matches_kron_entrywise(self) -> None:
        """Gate 3: SE-ARD over concatenated (theta, lambda) inputs, flattened
        row-major ((i, j) -> i*m+j), equals kron(K_theta, K_lambda)."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(5, 2))
        t = np.array([-1.0, -0.2, 0.4, 1.3])
        length_scales = np.array([0.8, 1.7])
        lambda_length_scale = 0.6

        n, m = len(X), len(t)
        Z = np.empty((n * m, 3))
        Z[:, :2] = np.repeat(X, m, axis=0)   # theta varies slowest: row i*m+j
        Z[:, 2] = np.tile(t, n)
        K_dense = se_ard(Z, np.array([*length_scales, lambda_length_scale]))

        K_kron = np.kron(
            se_ard(X, length_scales),
            se_ard(t.reshape(-1, 1), np.array([lambda_length_scale])),
        )
        np.testing.assert_allclose(K_dense, K_kron, rtol=1e-12, atol=1e-15)


class ExactnessTests(unittest.TestCase):
    """Gate 1: Kronecker path == dense Cholesky reference at FIXED hypers."""

    def test_lml_matches_dense_cholesky(self) -> None:
        X, t, Y = _toy_grid()
        settings = [
            _log_eta(1.0, [1.0, 1.0, 1.0], 1.0, 1e-4),
            _log_eta(5.0, [0.6, 2.0, 1.1], 0.4, 1e-6),
            _log_eta(0.3, [1.5, 0.8, 3.0], 2.5, 0.5),
        ]
        for log_eta in settings:
            lml, _, rho = kron_lml_and_grad(log_eta, X, t, Y, SEKernel1D())
            self.assertIsNone(rho)
            dense = _dense_lml(_dense_covariance(X, t, log_eta), Y.reshape(-1))
            np.testing.assert_allclose(lml, dense, rtol=1e-8)

    def test_profiled_rho_and_lml_match_dense_gls(self) -> None:
        rng = np.random.default_rng(2)
        X, t, Y_pair = _toy_grid(seed=2)
        Y = 0.7 * Y_pair + 0.05 * rng.normal(size=Y_pair.shape)
        log_eta = _log_eta(2.0, [1.2, 0.9, 1.6], 0.8, 1e-3)

        lml, _, rho = kron_lml_and_grad(
            log_eta, X, t, Y, SEKernel1D(), Y_pair=Y_pair
        )

        K = _dense_covariance(X, t, log_eta)
        factor = cho_factor(K, lower=True)
        y_h, y_l = Y.reshape(-1), Y_pair.reshape(-1)
        rho_dense = float(y_l @ cho_solve(factor, y_h) / (y_l @ cho_solve(factor, y_l)))
        np.testing.assert_allclose(rho, rho_dense, rtol=1e-8)
        np.testing.assert_allclose(
            lml, _dense_lml(K, y_h - rho_dense * y_l), rtol=1e-8
        )

    def _fixed_hyper_fit(self, X, t, Y, log_eta, Y_pair=None):
        return kron_gp_at(X, t, Y, log_eta, Y_pair=Y_pair)

    def test_posterior_mean_and_var_match_dense(self) -> None:
        X, t, Y = _toy_grid()
        X_new = np.random.default_rng(3).uniform(-1.5, 1.5, size=(4, X.shape[1]))
        log_eta = _log_eta(1.8, [0.9, 1.4, 1.1], 0.7, 1e-3)
        fit = self._fixed_hyper_fit(X, t, Y, log_eta)

        p = X.shape[1]
        signal_variance = np.exp(log_eta[0])
        length_scales = np.exp(log_eta[1 : 1 + p])
        lambda_length_scale = np.exp(log_eta[1 + p])
        noise_variance = np.exp(log_eta[-1])
        K = _dense_covariance(X, t, log_eta)
        K_lambda = se_ard(t.reshape(-1, 1), np.array([lambda_length_scale]))
        K_cross = signal_variance * np.kron(
            se_ard_cross(X_new, X, length_scales), K_lambda
        )
        factor = cho_factor(K, lower=True)
        mean_dense = (K_cross @ cho_solve(factor, Y.reshape(-1))).reshape(4, len(t))
        prior_diag = signal_variance * np.tile(np.diag(K_lambda), 4)
        var_dense = (
            prior_diag - np.einsum("ij,ij->i", K_cross, cho_solve(factor, K_cross.T).T)
        ).reshape(4, len(t))

        means, variances = posterior_mean_var(fit, X_new, include_noise=False)
        np.testing.assert_allclose(means, mean_dense, rtol=1e-8)
        np.testing.assert_allclose(variances, var_dense, rtol=1e-8)

        _, with_noise = posterior_mean_var(fit, X_new, include_noise=True)
        np.testing.assert_allclose(
            with_noise, var_dense + noise_variance, rtol=1e-8
        )

    def test_posterior_lambda_cov_matches_dense_block(self) -> None:
        X, t, Y = _toy_grid()
        x_new = np.array([[0.3, -0.7, 0.9]])
        log_eta = _log_eta(1.8, [0.9, 1.4, 1.1], 0.7, 1e-3)
        fit = self._fixed_hyper_fit(X, t, Y, log_eta)

        p = X.shape[1]
        signal_variance = np.exp(log_eta[0])
        length_scales = np.exp(log_eta[1 : 1 + p])
        lambda_length_scale = np.exp(log_eta[1 + p])
        K = _dense_covariance(X, t, log_eta)
        K_lambda = se_ard(t.reshape(-1, 1), np.array([lambda_length_scale]))
        K_cross = signal_variance * np.kron(
            se_ard_cross(x_new, X, length_scales), K_lambda
        )
        cov_dense = signal_variance * K_lambda - K_cross @ cho_solve(
            cho_factor(K, lower=True), K_cross.T
        )

        cov = posterior_lambda_cov(fit, x_new[0], include_noise=False)
        np.testing.assert_allclose(cov, cov_dense, rtol=1e-8, atol=1e-12)

    def test_stage_delta_posterior_uses_residual_targets(self) -> None:
        """With Y_pair, the posterior must be the GP on Y - rho* Y_pair."""
        rng = np.random.default_rng(4)
        X, t, Y_pair = _toy_grid(seed=4)
        Y = 0.6 * Y_pair + 0.05 * rng.normal(size=Y_pair.shape)
        log_eta = _log_eta(1.0, [1.0, 1.0, 1.0], 1.0, 1e-3)
        fit = self._fixed_hyper_fit(X, t, Y, log_eta, Y_pair=Y_pair)
        residual_fit = self._fixed_hyper_fit(
            X, t, Y - fit.rho * Y_pair, log_eta
        )
        X_new = rng.uniform(-1.5, 1.5, size=(3, X.shape[1]))
        means, variances = posterior_mean_var(fit, X_new)
        means_ref, variances_ref = posterior_mean_var(residual_fit, X_new)
        np.testing.assert_allclose(means, means_ref, rtol=1e-10)
        np.testing.assert_allclose(variances, variances_ref, rtol=1e-10)


class GradientTests(unittest.TestCase):
    """Gate 2: analytic gradients vs central finite differences."""

    def _check_gradients(self, Y_pair=None) -> None:
        X, t, Y = _toy_grid(seed=5, n=8, m=6, p=2)
        rng = np.random.default_rng(6)
        for _ in range(4):
            log_eta = np.concatenate(
                (
                    rng.uniform(-1.0, 1.0, size=1),         # log signal variance
                    rng.uniform(-0.7, 0.7, size=X.shape[1]),  # log length scales
                    rng.uniform(-0.7, 0.7, size=1),         # log lambda scale
                    rng.uniform(-6.0, -1.0, size=1),        # log noise variance
                )
            )
            _, grad, _ = kron_lml_and_grad(
                log_eta, X, t, Y, SEKernel1D(), Y_pair=Y_pair
            )
            fd = np.empty_like(grad)
            h = 1e-6
            for k in range(len(log_eta)):
                shift = np.zeros_like(log_eta)
                shift[k] = h
                lml_plus, _, _ = kron_lml_and_grad(
                    log_eta + shift, X, t, Y, SEKernel1D(), Y_pair=Y_pair
                )
                lml_minus, _, _ = kron_lml_and_grad(
                    log_eta - shift, X, t, Y, SEKernel1D(), Y_pair=Y_pair
                )
                fd[k] = (lml_plus - lml_minus) / (2 * h)
            np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)

    def test_plain_objective_gradients(self) -> None:
        self._check_gradients()

    def test_rho_profiled_objective_gradients(self) -> None:
        """The envelope theorem makes the partial gradient at rho* exact for
        the profiled objective -- rho* is recomputed inside every call."""
        _, _, Y_pair = _toy_grid(seed=5, n=8, m=6, p=2)
        self._check_gradients(Y_pair=0.8 * Y_pair + 0.3)


class FitTests(unittest.TestCase):
    def test_fit_improves_on_default_start_and_reports_restarts(self) -> None:
        X, t, Y = _toy_grid(seed=7, n=12, m=8, p=2)
        fit = fit_kron_gp(X, t, Y, seed=0, n_restarts=3, maxiter=50)
        default_lml, _, _ = kron_lml_and_grad(
            np.concatenate(([0.0], np.zeros(2), [0.0], [np.log(1e-4)])),
            X, t, Y, fit.lambda_kernel,
        )
        self.assertGreaterEqual(fit.log_marginal_likelihood, default_lml)
        self.assertEqual(len(fit.restart_records), 3)
        self.assertTrue(all(np.isfinite(r["lml"]) for r in fit.restart_records))

    def test_recovers_negative_rho(self) -> None:
        rng = np.random.default_rng(8)
        X, t, Y_pair = _toy_grid(seed=8, n=15, m=10, p=2)
        Y = -0.6 * Y_pair + 0.02 * rng.normal(size=Y_pair.shape)
        fit = fit_kron_gp(X, t, Y, seed=0, n_restarts=2, Y_pair=Y_pair, maxiter=60)
        self.assertLess(fit.rho, 0.0)
        self.assertAlmostEqual(fit.rho, -0.6, delta=0.1)

    def test_eigenvalue_clipping_keeps_lml_finite_near_singular(self) -> None:
        """Duplicated theta rows make K_theta numerically rank-deficient; the
        clipped eigenvalues plus the noise floor must keep everything finite."""
        X, t, Y = _toy_grid(seed=9, n=10, m=6, p=2)
        X[1] = X[0]
        Y[1] = Y[0]
        log_eta = _log_eta(1e3, [0.5, 0.5], 0.5, 1e-8)
        lml, grad, _ = kron_lml_and_grad(log_eta, X, t, Y, SEKernel1D())
        self.assertTrue(np.isfinite(lml))
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_bound_pinned_optima_are_reported(self) -> None:
        """Hyperparameters sitting ON bounds must be reported as pinned
        (exercises the reporting logic deterministically via kron_gp_at)."""
        X, t, Y = _toy_grid(seed=10, n=6, m=5, p=2)
        log_eta = _log_eta(1e3, [1e-2, 1.0], 1e2, 1e-8)
        fit = kron_gp_at(X, t, Y, log_eta)
        self.assertEqual(fit.pinned_bounds["signal_variance"], "upper")
        self.assertEqual(fit.pinned_bounds["length_scale_0"], "lower")
        self.assertEqual(fit.pinned_bounds["lambda_length_scale"], "upper")
        self.assertEqual(fit.pinned_bounds["noise_variance"], "lower")
        self.assertNotIn("length_scale_1", fit.pinned_bounds)

    def test_pickle_roundtrip_drops_cache_and_predicts_identically(self) -> None:
        """The joblib artifact path: the eigen cache is rebuilt after loading
        and predictions are unchanged."""
        X, t, Y = _toy_grid(seed=11, n=10, m=7, p=2)
        fit = fit_kron_gp(X, t, Y, seed=0, n_restarts=1, maxiter=30)
        X_new = np.random.default_rng(12).uniform(-1, 1, size=(3, 2))
        means, variances = posterior_mean_var(fit, X_new)

        restored = pickle.loads(pickle.dumps(fit))
        self.assertIsNone(restored._cache)
        means_restored, variances_restored = posterior_mean_var(restored, X_new)
        np.testing.assert_allclose(means_restored, means, rtol=1e-12)
        np.testing.assert_allclose(variances_restored, variances, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
