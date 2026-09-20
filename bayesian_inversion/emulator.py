"""Forward model: the 195 stored MF models, recalled and evaluated.

This module does exactly two things -- load the stored ``.pkl`` models and call
``.predict()`` on them. It never fits, never optimises hyperparameters, and never
factorises a covariance matrix. The hyperparameters and coefficients baked into the
stored models are applied exactly as GPBoost applies them.

Either family can stand in for the simulator. Model 3 is the default: on the
lf10000 LOO it scores mean MAE 8.69e-5 against Model 1 MF's 1.01e-4. Model 1
stays selectable because the earlier inversion runs used it.

Both families were trained on 10 000 LF + 96 HF spectra with HF design point 81
held out (``modelling/full_fit/full_fit_model1.py:23-29``,
``modelling/full_fit/full_fit_model3.py``), so predictions at sample 81 are
genuinely out of sample.

Two mechanical choices make this fast enough to sample with, neither of which
changes a single predicted number:

* **Models stay resident.** Each worker process loads its slice of the 195 models
  once at start-up and keeps them in memory, so nothing is reloaded inside the
  sampling loop.
* **Parallelise across wavelengths.** Measured on this machine, GPBoost's predict
  cost is fixed per-call overhead rather than the Vecchia solve: every
  ``vecchia_pred_type`` and ``num_neighbors_pred`` setting lands within 10% of
  baseline, and ``num_parallel_threads`` has no effect. Spreading the 195
  independent models over processes is the only lever that works, and it takes a
  full-spectrum evaluation from ~3.1 s to ~370 ms.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from concurrent.futures import ProcessPoolExecutor

N_WAVELENGTHS = 195
N_PARAMS = 9
DEFAULT_WORKERS = 8

# Both families store one MF model per wavelength, but the stored objects differ:
# Model 1 is a bare GPModel, Model 3 a boosted-tree ensemble with a GP. MF only;
# SF is never used here.
MODEL_FAMILIES = {
    "model1": {
        "dir": PROJECT_ROOT / "results/model1/full_fit",
        "stem": "model1_mf_response_{index}.pkl",
        "sidecars": (),
    },
    "model3": {
        "dir": PROJECT_ROOT / "results/model3/full_fit",
        "stem": "model3_mf_response_{index}.pkl",
        # A jointly fitted booster carries its own GP, so it needs no sidecar.
        # Runs saved before that left the GP in a .gp_model.json beside the
        # booster, and load_model3 falls back to a bare booster -- the trees
        # alone, no GP -- when such a file is missing. That would silently drop
        # half the model, so require the sidecar for exactly those.
        "sidecars": (".gp_model.json",),
        "gp_may_be_inline": True,
    },
}


def _carries_own_gp(path: Path) -> bool:
    """Whether a saved booster holds its GP inline rather than in a sidecar.

    A booster with an attached GP is written as JSON declaring "has_gp_model"
    on its second line; one without is plain LightGBM text. This is the same
    test gpboost itself applies when loading, kept to two lines so validating
    195 wavelengths does not mean deserializing 195 models.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.readline()
            return '"has_gp_model": 1' in handle.readline()
    except OSError:
        return False
DEFAULT_FAMILY = "model3"

# Populated once per worker process by ``_load_models``. Module-level because a
# ProcessPoolExecutor initialiser cannot return state any other way.
_MODELS: dict[int, object] = {}
_FAMILY: str = DEFAULT_FAMILY


def _load_models(family: str, model_dir: str, indices: list[int]) -> None:
    """Worker initialiser: load this worker's models and keep them resident."""
    global _FAMILY

    _FAMILY = family
    stem = MODEL_FAMILIES[family]["stem"]

    if family == "model1":
        import gpboost as gpb

        for index in indices:
            path = Path(model_dir) / stem.format(index=index)
            _MODELS[index] = gpb.GPModel(model_file=str(path))
        return

    from src.model_3 import load_model3

    for index in indices:
        path = Path(model_dir) / stem.format(index=index)
        # HF_only=False: these are the MF fits, whose GP uses the two-level
        # AR(1) covariance and whose coordinates carry the fidelity indicator.
        _MODELS[index] = load_model3(path, HF_only=False)


def _predict_chunk(task: tuple[list[int], np.ndarray]) -> list[np.ndarray]:
    """Predict one worker's wavelengths for the whole batch of parameter vectors."""
    indices, theta = task

    # The MF kernel's coordinates are the 9 parameters plus the fidelity indicator.
    # We append 1.0 because we want high-fidelity predictions -- the emulator is
    # standing in for the expensive simulator, not for the cheap one.
    coords = np.hstack([theta, np.ones((len(theta), 1))])

    if _FAMILY == "model1":
        return [
            _MODELS[index].predict(
                gp_coords_pred=coords,
                X_pred=theta,
                predict_response=True,
                predict_var=False,
            )["mu"]
            for index in indices
        ]

    # Model 3 names its arguments differently: the trees take the bare 9
    # parameters, the GP those plus the fidelity flag. Model3Fit.predict
    # rescales the GP coordinates with the stored training mean and sd, so it
    # gets raw values here, exactly as the LOO fed it. It returns
    # "response_mean" where GPModel returns "mu".
    return [
        _MODELS[index].predict(
            data=theta,
            gp_coords_pred=coords,
            predict_var=False,
        )["response_mean"]
        for index in indices
    ]


class MFEmulator:
    """Recall-only wrapper over the 195 stored MF models of one family.

    ``family`` selects which stored fit stands in for the simulator: ``model3``
    (the default, best on the lf10000 LOO) or ``model1``.

    Use as a context manager, or call ``close()`` when finished -- the worker pool
    holds ~195 loaded models and should not outlive the run.
    """

    def __init__(
        self,
        n_workers: int = DEFAULT_WORKERS,
        model_dir: Path | None = None,
        family: str = DEFAULT_FAMILY,
    ) -> None:
        if family not in MODEL_FAMILIES:
            raise ValueError(
                f"unknown model family {family!r}, expected one of "
                f"{sorted(MODEL_FAMILIES)}"
            )

        spec = MODEL_FAMILIES[family]
        self.family = family
        self.model_dir = Path(model_dir) if model_dir is not None else spec["dir"]
        self.wavelength_indices = list(range(N_WAVELENGTHS))

        stem, sidecars = spec["stem"], spec["sidecars"]
        gp_may_be_inline = spec.get("gp_may_be_inline", False)
        missing = []
        for index in self.wavelength_indices:
            path = self.model_dir / stem.format(index=index)
            if not path.exists():
                missing.append(index)
                continue
            if gp_may_be_inline and _carries_own_gp(path):
                continue
            if any(
                not path.with_suffix(path.suffix + suffix).exists()
                for suffix in sidecars
            ):
                missing.append(index)
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} stored {family} model(s) (or their "
                f"sidecars) in {self.model_dir}, first few: {missing[:5]}"
            )

        # Round-robin so each worker gets a similar share of the wavelengths.
        self._chunks = [
            self.wavelength_indices[offset::n_workers]
            for offset in range(n_workers)
            if self.wavelength_indices[offset::n_workers]
        ]

        self._pool = ProcessPoolExecutor(
            max_workers=len(self._chunks),
            initializer=_load_models,
            initargs=(self.family, str(self.model_dir), self.wavelength_indices),
        )

    def mean(self, theta: np.ndarray) -> np.ndarray:
        """Predicted spectra for a batch of parameter vectors.

        Parameters
        ----------
        theta : (B, 9) array in ``priors.PARAMETER_NAMES`` order.

        Returns
        -------
        (B, 195) array of predicted eclipse depths, in wavelength order.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if theta.shape[1] != N_PARAMS:
            raise ValueError(f"theta has {theta.shape[1]} columns, expected {N_PARAMS}")

        predictions = np.empty((len(theta), N_WAVELENGTHS))
        tasks = [(chunk, theta) for chunk in self._chunks]

        # Workers return results per chunk; scatter them back into wavelength order.
        for chunk, results in zip(self._chunks, self._pool.map(_predict_chunk, tasks)):
            for index, mu in zip(chunk, results):
                predictions[:, index] = mu

        return predictions

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "MFEmulator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def benchmark(emulator: MFEmulator, batch_size: int = 96) -> float:
    """Time one full-batch evaluation. Returns milliseconds per parameter vector.

    Printed at the start of every run so a misconfigured pool is obvious
    immediately rather than after hours of sampling.
    """
    from bayesian_inversion.priors import sample_prior

    theta = sample_prior(batch_size, np.random.default_rng(0))
    start = time.perf_counter()
    emulator.mean(theta)
    elapsed = time.perf_counter() - start
    return elapsed / batch_size * 1000.0


if __name__ == "__main__":
    import pandas as pd

    from bayesian_inversion.observations import HELD_OUT_HF_SAMPLE

    family = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FAMILY

    x_hf = pd.read_csv(PROJECT_ROOT / "data/XHF.csv", skipinitialspace=True)
    x_hf.columns = x_hf.columns.str.strip()
    y_hf = pd.read_csv(PROJECT_ROOT / "data/YHF.csv", index_col=0)

    with MFEmulator(family=family) as emulator:
        print(f"family    : {family}")
        print(f"throughput: {benchmark(emulator):.0f} ms per parameter vector")

        predicted = emulator.mean(x_hf.to_numpy(dtype=float))
        actual = y_hf.to_numpy(dtype=float)
        rms = np.sqrt(((predicted - actual) ** 2).mean(axis=1))

        held_out = np.zeros(len(rms), dtype=bool)
        held_out[HELD_OUT_HF_SAMPLE] = True

        print(f"RMS error, 96 training designs : {rms[~held_out].mean():.3e}")
        print(f"RMS error, held-out sample 81  : {rms[HELD_OUT_HF_SAMPLE]:.3e}")
        print(
            "sanity: held-out error should be clearly the larger -- "
            f"{'PASS' if rms[HELD_OUT_HF_SAMPLE] > rms[~held_out].mean() else 'FAIL'}"
        )
