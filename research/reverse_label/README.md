# Reverse-label sensitivity analysis

This macro reverses the labels used by
`research/exploratory/03_input_sensitivity.py`:

```text
original: physical parameters X  -> spectrum Y
reverse:  spectral channels Y    -> physical parameters X
```

It uses the row-aligned 10,000-point low-fidelity design. The workflow
reproduces the original PCA, binned first-order analysis, random-forest
importance, rankings, and heatmap with the two matrices switched.

Run it from the repository root:

```bash
make reverse-label
```

Artifacts are written to `results/reverse_label/`:

- `00_pca_scree.png` compares raw and standardized parameter-target PCA;
- `00_pca_loadings.png` shows the corresponding component loadings;
- `00_sensitivity_by_wavelength.png` shows all four aggregate reverse metrics;
- `00_sensitivity_ranking.png` ranks the strongest scale-balanced channels;
- `00_sensitivity_heatmap.png` preserves every wavelength-by-parameter
  first-order association;
- `01_strongest_association_spectra.png` highlights the peak for `C`, `f`, `O`,
  and `N` across the full wavelength range;
- `01_strongest_association_relationships.png` directly plots each selected
  spectral value against its parameter and overlays the binned conditional
  mean;
- CSV and JSON files contain the plotted values and headline results.

## Interpretation

The raw reversal is retained literally, but its PCA and aggregate sensitivity
are dominated by parameter units: `Tint` varies over hundreds while `logg`
varies over hundredths. The standardized/equal-parameter versions are the
appropriate scale-balanced comparison.

The spectral channels are also strongly correlated. Consequently, the reverse
first-order values are univariate explained-variance associations rather than
an additive Sobol decomposition, and random-forest impurity can split credit
arbitrarily among neighbouring wavelengths. These diagnostics measure inverse
predictive information; they do not imply that spectra physically cause the
atmospheric parameters.
