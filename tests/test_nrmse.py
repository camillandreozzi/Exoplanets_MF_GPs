"""NRMSE uses reference population std at each reporting scope."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kernel_comparison.sklearn_ar1 import prediction_metrics
from modelling.cv import cv_5fold, cv_learning_curve
from modelling.full_fit import summarize_full_fit_metrics as full_fit
from modelling.plot import plot_learning_curve_results as learning_plot
from modelling.plot import plot_loo_model1_mf_vs_cn as cn_plot
from modelling.plot import plot_loo_model1_model2_sf_vs_mf_per_sample as sample_plot
from modelling.plot import plot_model3_in_sample_nrmse as model3_plot


class NRMSETests(unittest.TestCase):
    def predictions(self, truth):
        return pd.DataFrame([
            dict(model="model1", variant=variant, n_hf_train=10,
                 response_index=0, wavelength=1., fold=i,
                 held_out_source_index=i, y_true=y, y_pred=y + 2.,
                 residual=2., error=2.)
            for variant in ["sf", "mf"] for i, y in enumerate(truth)
        ])

    def test_cv_and_plot_metrics(self):
        # Unequal spacing distinguishes std from range and sample std.
        for truth in [[1., 3., 7.], [2., 2., 2.], [2.]]:
            expected = 6. / np.sqrt(56.) if len(set(truth)) > 1 else np.nan
            predictions = self.predictions(truth)
            with self.subTest(truth=truth):
                for module in [cv_5fold, cv_learning_curve]:
                    result = module.per_wavelength_metrics(predictions)
                    np.testing.assert_allclose(result.nrmse, expected)
                result = cn_plot.summarise(predictions)
                np.testing.assert_allclose(result.nrmse, expected)
                with patch.object(learning_plot.pd, "read_csv", return_value=predictions):
                    result = learning_plot.summarize_checkpointed_predictions()
                np.testing.assert_allclose(result.mean_nrmse, expected)
                # Transpose the data into one spectrum per variant.
                predictions["wavelength"] = predictions["fold"]
                predictions[["fold", "held_out_source_index"]] = 0
                result = sample_plot.compute_per_sample(predictions)
                np.testing.assert_allclose(result[["nrmse_sf", "nrmse_mf"]], expected)
                result = prediction_metrics(np.array(truth), np.array(truth) + 2.,
                                            np.ones(len(truth)))
                np.testing.assert_allclose(result["nrmse"], expected)

    def test_full_fit_scopes(self):
        truth = np.array([[1., 1., 1.], [3., 7., 1.]])
        residuals = np.full_like(truth, 2.)
        by_wavelength = full_fit.per_wavelength_metrics(
            "model1", "sf", [1., 2., 3.], truth, residuals)
        np.testing.assert_allclose(by_wavelength.nrmse, [2., 2. / 3., np.nan])
        by_sample = full_fit.per_sample_metrics(
            "model1", "sf", [0, 1], truth, residuals)
        np.testing.assert_allclose(by_sample.nrmse, [np.nan, 6. / np.sqrt(56.)])
        result = full_fit.global_metrics("model1", "sf", truth, residuals)
        self.assertAlmostEqual(result["nrmse"], 3. / np.sqrt(11.))
        result = full_fit.global_metrics("model1", "sf", np.ones_like(truth), residuals)
        self.assertTrue(np.isnan(result["nrmse"]))

    def test_model3_uses_target_std(self):
        data = pd.DataFrame({"x": [0., 1., 2.], "is_hf": 1,
                             "response_0": [1., 3., 7.]})
        data.attrs["wavelength_map"] = {0: 1.}
        indexes = {"x_columns": ["x"], "hf_row_idx": data.index}
        with patch.object(model3_plot, "load_full_data", return_value=data), \
             patch.object(model3_plot, "create_data_indexes", return_value=indexes), \
             patch.object(model3_plot, "RESPONSE_DIM", 1), \
             patch.object(model3_plot, "predict_rmse", return_value=2.):
            result = model3_plot.compute_model3_nrmse()
        np.testing.assert_allclose(result[["sf_nrmse", "mf_nrmse"]], 6. / np.sqrt(56.))
        self.assertTrue(np.isnan(model3_plot.normalize_rmse(2., 0.)))


if __name__ == "__main__":
    unittest.main()
