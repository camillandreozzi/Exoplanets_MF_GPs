"""Unit tests for the log10 output-space transforms."""

from __future__ import annotations

import unittest

import numpy as np

from exoplanets_mf.transforms import (
    inverse_log10_spectra,
    log10_observed_errors,
    log10_spectra,
)


class Log10SpectraTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        Y = 10.0 ** rng.uniform(-6, -2, size=(20, 5))
        np.testing.assert_allclose(inverse_log10_spectra(log10_spectra(Y)), Y)

    def test_known_values(self) -> None:
        np.testing.assert_allclose(
            log10_spectra(np.array([1.0, 10.0, 100.0])), [0.0, 1.0, 2.0]
        )

    def test_raises_on_zero_and_negative(self) -> None:
        with self.assertRaises(ValueError):
            log10_spectra(np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            log10_spectra(np.array([1.0, -1e-6]))


class Log10ObservedErrorsTests(unittest.TestCase):
    def test_hand_computed_asymmetry(self) -> None:
        # d=100, err_lo=10, err_hi=10: symmetric in linear space becomes
        # asymmetric in log space.
        err_lo_log, err_hi_log = log10_observed_errors(
            np.array([100.0]), np.array([10.0]), np.array([10.0])
        )
        self.assertAlmostEqual(err_lo_log[0], 2.0 - np.log10(90.0))
        self.assertAlmostEqual(err_hi_log[0], np.log10(110.0) - 2.0)
        self.assertGreater(err_lo_log[0], err_hi_log[0])

    def test_outputs_positive(self) -> None:
        rng = np.random.default_rng(1)
        depth = 10.0 ** rng.uniform(-5, -3, size=50)
        err = 0.5 * depth
        err_lo_log, err_hi_log = log10_observed_errors(depth, err, err)
        self.assertTrue((err_lo_log > 0).all())
        self.assertTrue((err_hi_log > 0).all())

    def test_raises_on_nonpositive_depth(self) -> None:
        with self.assertRaises(ValueError):
            log10_observed_errors(
                np.array([-1e-5]), np.array([1e-6]), np.array([1e-6])
            )

    def test_raises_when_lower_bar_crosses_zero(self) -> None:
        with self.assertRaises(ValueError):
            log10_observed_errors(
                np.array([1e-5]), np.array([2e-5]), np.array([1e-6])
            )


if __name__ == "__main__":
    unittest.main()
