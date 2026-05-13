"""Load the exoplanet multifidelity dataset.

The /data folder contains:
    - XHF.csv             : atmospheric parameters for HF simulations
                            (header: Kzz, Rp, Tint, C, N, O, S, logg, f)
    - YHF.csv             : high-fidelity spectra.
                            First row = wavelength header, first column = "spectrum N".
    - YLF.csv             : low-fidelity spectra.
                            First row = wavelengths (no leading label), no index column.
    - Observed_Spectra.csv: measured eclipse depth with asymmetric error bars,
                            columns wavelength, measured_eclipse_depth,
                            measured_err_lo, measured_err_hi.

Both YHF and YLF live on the same 195-point wavelength grid as the observation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_XHF(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return HF input parameters as a DataFrame (rows = samples, cols = params)."""
    df = pd.read_csv(data_dir / "XHF.csv")
    df.columns = df.columns.str.strip()
    return df


def load_YHF(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return HF spectra. Index = 'spectrum N', columns = wavelengths (float)."""
    df = pd.read_csv(data_dir / "YHF.csv", index_col=0)
    df.columns = df.columns.astype(float)
    return df


def load_YLF(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return LF spectra. Columns = wavelengths (float); rows align with YHF rows."""
    df = pd.read_csv(data_dir / "YLF.csv")
    df.columns = df.columns.astype(float)
    return df


def load_observed(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return observed spectrum with asymmetric error bars."""
    return pd.read_csv(data_dir / "Observed_Spectra.csv")


def load_all(data_dir: Path = DATA_DIR) -> dict:
    """Load every file and return a dict plus the shared wavelength grid."""
    XHF = load_XHF(data_dir)
    YHF = load_YHF(data_dir)
    YLF = load_YLF(data_dir)
    obs = load_observed(data_dir)

    wavelengths = YHF.columns.to_numpy(dtype=float)
    assert np.allclose(YLF.columns.to_numpy(dtype=float), wavelengths), (
        "YHF and YLF wavelength grids disagree"
    )

    return {
        "XHF": XHF,
        "YHF": YHF,
        "YLF": YLF,
        "observed": obs,
        "wavelengths": wavelengths,
    }


if __name__ == "__main__":
    data = load_all()
    print(f"XHF      : {data['XHF'].shape}  cols={list(data['XHF'].columns)}")
    print(f"YHF      : {data['YHF'].shape}")
    print(f"YLF      : {data['YLF'].shape}")
    print(f"observed : {data['observed'].shape}")
    print(f"lambdas  : {data['wavelengths'].shape}, "
          f"{data['wavelengths'].min()}–{data['wavelengths'].max()} um")
