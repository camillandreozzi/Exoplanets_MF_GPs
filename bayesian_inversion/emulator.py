"""Forward model: the 195 stored Model 1 MF GPs, recalled and evaluated.

This module does exactly two things -- load the stored ``.pkl`` models and call
``.predict()`` on them. It never fits, never optimises hyperparameters, and never
factorises a covariance matrix. The hyperparameters and coefficients baked into the
stored models are applied exactly as GPBoost applies them.

Those models were trained on 10 000 LF + 96 HF spectra with HF design point 81 held
out (``modelling/full_fit/full_fit_model1.py:23-29``), so predictions at sample 81
are genuinely out of sample.

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

MODEL_DIR = PROJECT_ROOT / "results/model1/full_fit"
MODEL_STEM = "model1_mf_response_{index}.pkl"  # MF only; SF is never used here
N_WAVELENGTHS = 195
N_PARAMS = 9
DEFAULT_WORKERS = 8

# Populated once per worker process by ``_load_models``. Module-level because a
# ProcessPoolExecutor initialiser cannot return state any other way.
_MODELS: dict[int, object] = {}


def _load_models(model_dir: str, indices: list[int]) -> None:
    """Worker initialiser: load this worker's models and keep them resident."""
    import gpboost as gpb

    for index in indices:
        path = Path(model_dir) / MODEL_STEM.format(index=index)
        _MODELS[index] = gpb.GPModel(model_file=str(path))


def _predict_chunk(task: tuple[list[int], np.ndarray]) -> list[np.ndarray]:
    """Predict one worker's wavelengths for the whole batch of parameter vectors."""
    indices, theta = task

    # The MF kernel's coordinates are the 9 parameters plus the fidelity indicator.
    # We append 1.0 because we want high-fidelity predictions -- the emulator is
    # standing in for the expensive simulator, not for the cheap one.
    coords = np.hstack([theta, np.ones((len(theta), 1))])

    return [
        _MODELS[index].predict(
            gp_coords_pred=coords,
            X_pred=theta,
            predict_response=True,
            predict_var=False,
        )["mu"]
        for index in indices
    ]


class MFEmulator:
    """Recall-only wrapper over the 195 stored Model 1 MF GPs.

    Use as a context manager, or call ``close()`` when finished -- the worker pool
    holds ~195 loaded models and should not outlive the run.
    """

    def __init__(
        self, n_workers: int = DEFAULT_WORKERS, model_dir: Path = MODEL_DIR
    ) -> None:
        self.model_dir = Path(model_dir)
        self.wavelength_indices = list(range(N_WAVELENGTHS))

        missing = [
            index
            for index in self.wavelength_indices
            if not (self.model_dir / MODEL_STEM.format(index=index)).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} stored model(s) in {self.model_dir}, "
                f"first few: {missing[:5]}"
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
            initargs=(str(self.model_dir), self.wavelength_indices),
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

    x_hf = pd.read_csv(PROJECT_ROOT / "data/XHF.csv", skipinitialspace=True)
    x_hf.columns = x_hf.columns.str.strip()
    y_hf = pd.read_csv(PROJECT_ROOT / "data/YHF.csv", index_col=0)

    with MFEmulator() as emulator:
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
