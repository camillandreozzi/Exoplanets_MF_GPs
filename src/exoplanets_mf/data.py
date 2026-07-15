"""Load the exoplanet multifidelity dataset.

The /data folder contains:
    - XHF.csv             : atmospheric parameters for HF simulations
                            (header: Kzz, Rp, Tint, C, N, O, S, logg, f)
    - YHF.csv             : high-fidelity spectra.
                            First row = wavelength header, first column = "spectrum N".
    - YLF.csv             : low-fidelity spectra evaluated at the *same* inputs as
                            XHF/YHF (97 rows, input-matched / paired with HF).
                            First row = wavelengths (no leading label), no index column.
    - XLF_10k.csv         : atmospheric parameters for the 10,000-point LF design.
                            No header; columns match XHF (Kzz, Rp, Tint, C, N, O, S,
                            logg, f). Independent (decoupled) from XHF.
    - YLF_10k.csv         : 10,000 low-fidelity spectra evaluated at XLF_10k.
                            First row = wavelengths, no index column.
    - Observed_Spectra.csv: measured eclipse depth with asymmetric error bars,
                            columns wavelength, measured_eclipse_depth,
                            measured_err_lo, measured_err_hi.

All spectra (YHF, YLF, YLF_10k) live on the same 195-point wavelength grid as the
observation. Note the two LF sets serve different roles: YLF (97) is input-matched
to HF and is the only set valid for row-paired HF-vs-LF correlation; (XLF_10k,
YLF_10k) is a large decoupled design used for distributional comparisons.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_DIR


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
    """Return paired LF spectra (97 rows, input-matched to YHF).

    Columns = wavelengths (float); row i corresponds to XHF/YHF row i.
    """
    df = pd.read_csv(data_dir / "YLF.csv")
    df.columns = df.columns.astype(float)
    return df


def load_XLF_10k(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return the 10k LF input parameters.

    XLF_10k.csv has no header; its columns match XHF (Kzz, Rp, Tint, C, N, O, S,
    logg, f). This design is independent of XHF.
    """
    cols = load_XHF(data_dir).columns
    df = pd.read_csv(data_dir / "XLF_10k.csv", header=None, names=cols)
    return df


def load_YLF_10k(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return the 10k LF spectra (rows align with XLF_10k rows).

    Columns = wavelengths (float).
    """
    df = pd.read_csv(data_dir / "YLF_10k.csv")
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
    XLF_10k = load_XLF_10k(data_dir)
    YLF_10k = load_YLF_10k(data_dir)
    obs = load_observed(data_dir)

    wavelengths = YHF.columns.to_numpy(dtype=float)
    _require_same_length(XHF, YHF, "XHF", "YHF")
    _require_same_length(XHF, YLF, "XHF", "YLF")
    _require_same_length(XLF_10k, YLF_10k, "XLF_10k", "YLF_10k")
    _require_same_grid(
        YLF.columns.to_numpy(dtype=float), wavelengths, "YLF", "YHF"
    )
    _require_same_grid(
        YLF_10k.columns.to_numpy(dtype=float), wavelengths, "YLF_10k", "YHF"
    )
    _require_same_grid(
        obs["wavelength"].to_numpy(dtype=float), wavelengths, "observed", "YHF"
    )

    return {
        "XHF": XHF,
        "YHF": YHF,
        "YLF": YLF,              # 97 rows, input-matched to HF (for correlation)
        "XLF_10k": XLF_10k,      # 10k decoupled LF inputs
        "YLF_10k": YLF_10k,      # 10k decoupled LF spectra (for distributions)
        "observed": obs,
        "wavelengths": wavelengths,
    }


def _require_same_length(left, right, left_name: str, right_name: str) -> None:
    """Raise a clear error when two row-aligned datasets have different sizes."""
    if len(left) != len(right):
        raise ValueError(
            f"{left_name} and {right_name} must be row-aligned; "
            f"found {len(left)} and {len(right)} rows"
        )


def _require_same_grid(
    candidate: np.ndarray,
    reference: np.ndarray,
    candidate_name: str,
    reference_name: str,
) -> None:
    """Raise a clear error when two wavelength grids differ."""
    if candidate.shape != reference.shape or not np.allclose(candidate, reference):
        raise ValueError(
            f"{candidate_name} and {reference_name} wavelength grids disagree"
        )


if __name__ == "__main__":
    data = load_all()
    print(f"XHF      : {data['XHF'].shape}  cols={list(data['XHF'].columns)}")
    print(f"YHF      : {data['YHF'].shape}")
    print(f"YLF      : {data['YLF'].shape}  (paired with HF)")
    print(f"XLF_10k  : {data['XLF_10k'].shape}")
    print(f"YLF_10k  : {data['YLF_10k'].shape}")
    print(f"observed : {data['observed'].shape}")
    print(f"lambdas  : {data['wavelengths'].shape}, "
          f"{data['wavelengths'].min()}–{data['wavelengths'].max()} um")
