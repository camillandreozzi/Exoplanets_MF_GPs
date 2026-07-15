"""Tests for the JWST instrument-mode wavelength partition."""

import unittest

import numpy as np

from exoplanets_mf.data import load_all
from exoplanets_mf.instruments import instrument_mode_masks


class InstrumentModeTests(unittest.TestCase):
    def test_dataset_is_partitioned_into_three_expected_modes(self) -> None:
        wavelengths = load_all()["wavelengths"]
        assignments = instrument_mode_masks(wavelengths)

        self.assertEqual(
            [mode.label for mode, _ in assignments],
            ["NIRCam F322W2", "NIRCam F444W", "MIRI LRS"],
        )
        self.assertEqual([int(mask.sum()) for _, mask in assignments], [103, 65, 27])
        np.testing.assert_array_equal(
            np.sum([mask for _, mask in assignments], axis=0),
            np.ones(wavelengths.shape, dtype=int),
        )

    def test_nominal_boundaries_are_half_open(self) -> None:
        wavelengths = np.array([2.4, 3.999, 4.0, 4.999, 5.0, 12.0])
        assignments = instrument_mode_masks(wavelengths)

        np.testing.assert_array_equal(
            [mask.tolist() for _, mask in assignments],
            [
                [True, True, False, False, False, False],
                [False, False, True, True, False, False],
                [False, False, False, False, True, True],
            ],
        )


if __name__ == "__main__":
    unittest.main()
