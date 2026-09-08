"""The two spectra we invert.

``sample81`` is the high-fidelity design point held out of the Model 1 fit
(``modelling/full_fit/full_fit_model1.py:23-29``). Its true parameters are known,
so inverting it is a closed-loop test: if the posterior does not cover the truth,
something is broken and we find out on a case with a known answer rather than on
the real observation.

``observed`` is the actual measurement, and is the scientific target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

HELD_OUT_HF_SAMPLE = 81  # positional index into the 97 HF rows
N_WAVELENGTHS = 195


@dataclass(frozen=True)
class Observation:
    """A spectrum to invert, with its observational error and (if known) truth."""

    name: str
    y: np.ndarray  # (195,) eclipse depth
    sigma: np.ndarray  # (195,) observational error, 1 sd
    wavelengths: np.ndarray  # (195,) micron
    truth: np.ndarray | None  # (9,) known parameters, or None

    def __post_init__(self) -> None:
        for field, expected in (("y", N_WAVELENGTHS), ("sigma", N_WAVELENGTHS)):
            got = len(getattr(self, field))
            if got != expected:
                raise ValueError(f"{self.name}: {field} has {got} entries, expected {expected}")


def _observational_error(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Wavelength grid and 1-sd error bars from the real observation.

    The published errors are asymmetric (``measured_err_lo`` / ``measured_err_hi``);
    a Gaussian likelihood needs one number, so they are averaged. The asymmetry is
    small here relative to the emulator error that dominates the budget (see
    ``noise.py``), so this is not a load-bearing approximation.
    """
    observed = pd.read_csv(data_dir / "Observed_Spectra.csv")
    sigma = 0.5 * (
        observed["measured_err_lo"].to_numpy() + observed["measured_err_hi"].to_numpy()
    )
    return observed["wavelength"].to_numpy(), sigma


def load_sample81(data_dir: Path = DATA_DIR) -> Observation:
    """The held-out HF simulation, with the real observation's error bars.

    The spectrum is noise-free simulator output; we attach the real error bars so
    the closed-loop test exercises the same likelihood the real run will use.
    """
    y_hf = pd.read_csv(data_dir / "YHF.csv", index_col=0)
    x_hf = pd.read_csv(data_dir / "XHF.csv", skipinitialspace=True)
    x_hf.columns = x_hf.columns.str.strip()

    wavelengths, sigma = _observational_error(data_dir)

    return Observation(
        name="sample81",
        y=y_hf.iloc[HELD_OUT_HF_SAMPLE].to_numpy(dtype=float),
        sigma=sigma,
        wavelengths=wavelengths,
        truth=x_hf.iloc[HELD_OUT_HF_SAMPLE].to_numpy(dtype=float),
    )


def load_observed(data_dir: Path = DATA_DIR) -> Observation:
    """The real measured spectrum."""
    observed = pd.read_csv(data_dir / "Observed_Spectra.csv")
    wavelengths, sigma = _observational_error(data_dir)

    return Observation(
        name="observed",
        y=observed["measured_eclipse_depth"].to_numpy(dtype=float),
        sigma=sigma,
        wavelengths=wavelengths,
        truth=None,
    )


LOADERS = {"sample81": load_sample81, "observed": load_observed}


def load_observation(target: str) -> Observation:
    if target not in LOADERS:
        raise ValueError(f"Unknown target {target!r}; expected one of {sorted(LOADERS)}")
    return LOADERS[target]()


if __name__ == "__main__":
    for target in LOADERS:
        obs = load_observation(target)
        print(
            f"{obs.name:<9} y in [{obs.y.min():.3e}, {obs.y.max():.3e}]  "
            f"median sigma {np.median(obs.sigma):.3e}  "
            f"truth {'known' if obs.truth is not None else 'unknown'}"
        )
