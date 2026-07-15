# Exploratory analysis

The numbered scripts form the current exploratory workflow:

1. `00_spectra_exploration.py` compares high- and low-fidelity spectra.
2. `01_residual_slope_3d.py` studies wavelength-wise HF residual vs standardized-LF scaling.
3. `02_input_exploration.py` diagnoses the input designs.
4. `03_input_sensitivity.py` estimates input-to-spectrum sensitivity.
5. `04_wavelength_in_design.py` tests wavelength as an input dimension.
6. `05_log_spectra_exploration.py` summarizes the spectra on log10 scale
   (skewness before/after, HF/LF correlation linear vs log10) and places the
   hf_lf_observed panel side by side with its log10 counterpart — the
   motivation for `research/log_modelling/`. Writes `05_log_stats.json` and
   two figures; non-positive *observed* depths are dropped from the log panel
   (the simulated spectra are strictly positive).

Whenever results are split by observing mode, the shared definitions in
`exoplanets_mf.instruments` assign the spectrum to exactly three nominal bands:
NIRCam F322W2 on `[2.4, 4)` um, NIRCam F444W on `[4, 5)` um, and MIRI LRS on
`[5, 12]` um. These boundaries are explicit because a wavelength-step detector
cannot distinguish the two NIRCam modes, which use the same bin spacing.

Run all analyses from the repository root with:

```bash
make exploratory
```

Generated figures are written to `results/exploratory/`.
