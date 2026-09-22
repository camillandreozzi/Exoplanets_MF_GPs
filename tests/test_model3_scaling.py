"""Regression checks for residual-GP scaling and model persistence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import model_3 as m3


class Model3ScalingTests(unittest.TestCase):
    def test_scaling_and_roundtrip(self):
        rng = np.random.default_rng(8)
        x = rng.uniform(size=(48, 2)) * [300, .03] + [150, 3]
        fidelity = np.r_[np.zeros(24), np.ones(24)]
        y = np.sin(x[:, 0] / 100) + 8 * (x[:, 1] - 3) + .2 * fidelity
        data = pd.DataFrame(dict(a=x[:, 0], b=x[:, 1], constant=7.,
                                 is_hf=fidelity, response_0=y))
        train, valid = data.iloc[:-2].copy(), data.iloc[-2:].copy()
        original = train.copy(deep=True)
        tuning = {'best_params': {'min_data_in_leaf': 2, 'num_leaves': 4},
                  'best_iter': 5}
        for hf_only in (True, False):
            with self.subTest(HF_only=hf_only), tempfile.TemporaryDirectory() as directory:
                model = m3.fit_model3(train, HF_only=hf_only,
                    tuning_result=tuning, verbose_eval=False)
                hf = train.loc[train.is_hf.eq(1)].iloc[:, :-2]
                np.testing.assert_allclose(model.gp_input_mean, hf.mean())
                np.testing.assert_allclose(model.gp_input_std,
                    hf.std(ddof=0).replace(0, 1))
                pd.testing.assert_frame_equal(train, original)

                raw_coords = valid.iloc[:, :-2 if hf_only else -1].to_numpy()
                coords = m3._scale_gp_coordinates(raw_coords,
                    model.gp_input_mean, model.gp_input_std)
                if not hf_only:
                    np.testing.assert_array_equal(coords[:, -1], valid.is_hf)
                x_valid = valid.iloc[:, :-2].to_numpy()
                tree = m3._predict_tree_fixed_effect(model.booster, x_valid)
                expected = model.gp_model.predict(gp_coords_pred=coords,
                    offset_pred=tree, predict_response=True, predict_var=True)
                actual = m3.predict_model3(valid, HF_only=hf_only, model=model)
                np.testing.assert_allclose(actual['mu'], expected['mu'])
                np.testing.assert_allclose(actual['var'], expected['var'])
                latent = m3.predict_model3(valid, HF_only=hf_only,
                    model=model, pred_latent=True)
                np.testing.assert_allclose(latent['mu'], actual['mu'])

                filename = Path(directory) / 'model.txt'
                m3.save_model3(filename)
                loaded = m3.load_model3(filename)
                self.assertEqual(loaded.HF_only, hf_only)
                restored = m3.predict_model3(valid, HF_only=hf_only, model=loaded)
                np.testing.assert_allclose(restored['mu'], actual['mu'], rtol=1e-8)
                np.testing.assert_allclose(restored['var'], actual['var'], rtol=1e-8)
                # The instance-level save must also retain preprocessing.
                direct = Path(directory) / 'direct.txt'
                loaded.save_model(direct)
                reloaded = m3.load_model3(direct)
                np.testing.assert_array_equal(reloaded.gp_input_mean, model.gp_input_mean)
                np.testing.assert_array_equal(reloaded.gp_input_std, model.gp_input_std)

    def test_coordinate_scaling_does_not_mutate_inputs(self):
        coords = np.array([[100., 3., 0.], [200., 4., 1.]])
        original = coords.copy()
        scaled = m3._scale_gp_coordinates(coords, np.array([100., 3.]), np.array([100., 1.]))
        np.testing.assert_array_equal(coords, original)
        np.testing.assert_array_equal(scaled, [[0., 0., 0.], [1., 1., 1.]])


if __name__ == '__main__':
    unittest.main()
