"""Regression checks for the GPBoost fit, GP scaling, and model persistence."""
import tempfile
import unittest
from pathlib import Path

import gpboost as gpb
import numpy as np
import pandas as pd

from src import model_3 as m3


def _synthetic_frame():
    rng = np.random.default_rng(8)
    x = rng.uniform(size=(48, 2)) * [300, .03] + [150, 3]
    fidelity = np.r_[np.zeros(24), np.ones(24)]
    y = np.sin(x[:, 0] / 100) + 8 * (x[:, 1] - 3) + .2 * fidelity
    return pd.DataFrame(dict(a=x[:, 0], b=x[:, 1], constant=7.,
                             is_hf=fidelity, response_0=y))


TUNING = {'best_params': {'min_data_in_leaf': 2, 'num_leaves': 4},
          'best_iter': 5}


class Model3ScalingTests(unittest.TestCase):
    def setUp(self):
        data = _synthetic_frame()
        self.train, self.valid = data.iloc[:-2].copy(), data.iloc[-2:].copy()

    def test_scaling_and_roundtrip(self):
        train, valid = self.train, self.valid
        original = train.copy(deep=True)
        for hf_only in (True, False):
            with self.subTest(HF_only=hf_only), tempfile.TemporaryDirectory() as directory:
                model = m3.fit_model3(train, HF_only=hf_only,
                    tuning_result=TUNING, verbose_eval=False)
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
                expected = model.booster.predict(data=x_valid,
                    gp_coords_pred=coords, predict_var=True)
                actual = m3.predict_model3(valid, HF_only=hf_only, model=model)
                np.testing.assert_allclose(actual['mu'], expected['response_mean'])
                np.testing.assert_allclose(actual['var'], expected['response_var'])
                latent = m3.predict_model3(valid, HF_only=hf_only,
                    model=model, pred_latent=True)
                np.testing.assert_allclose(latent['mu'], actual['mu'])

                filename = Path(directory) / 'model.txt'
                saved = m3.save_model3(filename)
                # The booster carries the GP, so there is nothing beside it.
                self.assertIsNone(saved['gp_model_file'])
                loaded = m3.load_model3(filename)
                self.assertEqual(loaded.HF_only, hf_only)
                restored = m3.predict_model3(valid, HF_only=hf_only, model=loaded)
                np.testing.assert_allclose(restored['mu'], actual['mu'], rtol=1e-8)
                np.testing.assert_allclose(restored['var'], actual['var'], rtol=1e-8)
                np.testing.assert_array_equal(loaded.gp_input_mean, model.gp_input_mean)
                np.testing.assert_array_equal(loaded.gp_input_std, model.gp_input_std)
                # The instance-level save must also retain preprocessing.
                direct = Path(directory) / 'direct.txt'
                model.save_model(direct)
                reloaded = m3.load_model3(direct)
                np.testing.assert_array_equal(reloaded.gp_input_mean, model.gp_input_mean)
                np.testing.assert_array_equal(reloaded.gp_input_std, model.gp_input_std)

                # A booster restored from a file keeps no training data, so
                # GPBoost cannot write it out again.
                with self.assertRaises(ValueError):
                    loaded.save_model(Path(directory) / 'again.txt')

    def test_trees_and_covariance_are_estimated_jointly(self):
        """The trees must see the GP, not be boosted against plain L2 loss.

        Without the gp_model the fit still produces predictions, so nothing
        else here would notice; this is the assertion that pins the algorithm.
        """
        for hf_only in (True, False):
            with self.subTest(HF_only=hf_only):
                model = m3.fit_model3(self.train, HF_only=hf_only,
                    tuning_result=TUNING, verbose_eval=False)
                self.assertTrue(model.booster.has_gp_model)
                self.assertIs(model.gp_model, model.booster.gp_model)
                cov_pars = np.asarray(
                    model.gp_model.get_cov_pars(std_err=False)).ravel()
                self.assertTrue(np.all(np.isfinite(cov_pars)))
                self.assertTrue(np.all(cov_pars > 0))

    def test_multifidelity_ensemble_spans_every_row(self):
        """MF trains trees on all rows and takes the fidelity as a feature."""
        n_inputs = self.train.shape[1] - 2
        sf = m3.fit_model3(self.train, HF_only=True,
            tuning_result=TUNING, verbose_eval=False)
        mf = m3.fit_model3(self.train, HF_only=False,
            tuning_result=TUNING, verbose_eval=False)

        self.assertEqual(sf.n_tree_train, int(self.train.is_hf.sum()))
        self.assertEqual(mf.n_tree_train, len(self.train))
        self.assertEqual(mf.n_tree_train, mf.n_gp_train)

        self.assertEqual(sf.booster.num_feature(), n_inputs)
        self.assertEqual(mf.booster.num_feature(), n_inputs + 1)
        # Callers still pass the bare inputs; the indicator is filled in from
        # the last GP coordinate column.
        prediction = m3.predict_model3(self.valid, HF_only=False, model=mf)
        self.assertTrue(np.all(np.isfinite(prediction['mu'])))

    def test_legacy_two_stage_model_still_loads(self):
        """Runs stored before the switch keep loading and predicting."""
        train, valid = self.train, self.valid
        arrays = m3._prepare_model3_training_arrays(train, HF_only=True)
        mean, std = m3._gp_input_scaling(arrays)
        coords = m3._scale_gp_coordinates(arrays["coords_gp"], mean, std)

        booster = gpb.train(
            params={'verbose': -1, 'num_leaves': 4, 'min_data_in_leaf': 2},
            train_set=gpb.Dataset(arrays["x_tree"], arrays["y_tree"]),
            num_boost_round=5,
            verbose_eval=False,
        )
        offset = m3._predict_tree_fixed_effect(booster, data=arrays["x_tree"])
        gp_model = m3.make_model3_gp_model(coords, HF_only=True)
        gp_model.fit(y=arrays["y_gp"], offset=offset, params={'trace': False})

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'legacy.txt'
            legacy = m3.LegacyModel3Fit(booster=booster, gp_model=gp_model,
                HF_only=True, gp_input_mean=mean, gp_input_std=std)
            saved = legacy.save_model(filename)
            self.assertTrue(Path(saved['gp_model_file']).exists())

            loaded = m3.load_model3(filename, HF_only=True)
            self.assertIsInstance(loaded, m3.LegacyModel3Fit)
            self.assertFalse(loaded.booster.has_gp_model)
            prediction = m3.predict_model3(valid, HF_only=True, model=loaded)
            np.testing.assert_allclose(
                prediction['mu'],
                legacy.predict(data=valid.iloc[:, :-2].to_numpy(),
                               gp_coords_pred=valid.iloc[:, :-2].to_numpy(),
                               predict_var=False)['response_mean'],
            )

    def test_coordinate_scaling_does_not_mutate_inputs(self):
        coords = np.array([[100., 3., 0.], [200., 4., 1.]])
        original = coords.copy()
        scaled = m3._scale_gp_coordinates(coords, np.array([100., 3.]), np.array([100., 1.]))
        np.testing.assert_array_equal(coords, original)
        np.testing.assert_array_equal(scaled, [[0., 0., 0.], [1., 1., 1.]])

    def test_stratified_folds_keep_both_fidelities(self):
        fidelity = np.r_[np.zeros(40), np.ones(6)]
        folds = m3._make_fidelity_stratified_folds(fidelity, nfold=3, random_state=0)
        self.assertEqual(len(folds), 3)
        for train_idx, test_idx in folds:
            self.assertEqual(set(np.unique(fidelity[train_idx])), {0., 1.})
            self.assertEqual(set(np.unique(fidelity[test_idx])), {0., 1.})
            self.assertEqual(len(train_idx) + len(test_idx), len(fidelity))


if __name__ == '__main__':
    unittest.main()
