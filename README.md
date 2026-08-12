# Exoplanet multifidelity Gaussian processes

Reproducible research code for studying high- and low-fidelity exoplanet
spectra and developing multifidelity Gaussian-process models.

## Project layout

```text
.
├── data/                    # immutable input CSVs and checksums
├── research/
│   ├── exploratory/         # numbered exploratory workflows
│   ├── reverse_label/       # inverse spectrum-to-parameter diagnostics
│   ├── modelling/           # model experiments (linear + log10 spectra)
│   ├── parameters/          # fitted-hyperparameter visualisations
│   └── validation/          # predictive-performance validation studies
├── src/exoplanets_mf/       # reusable loaders, paths, and shared code
├── results/
│   ├── exploratory/         # generated exploratory artifacts
│   ├── reverse_label/       # generated inverse-diagnostic artifacts
│   ├── modelling/           # generated model artifacts (linear spectra)
│   ├── log_modelling/       # generated model artifacts (log10 spectra)
│   ├── parameters/          # generated hyperparameter figures/tables
│   └── validation/          # generated validation artifacts
├── tests/                   # data and code integrity checks
├── pyproject.toml           # installable project metadata
└── requirements.lock        # pinned Python environment
```

The split is deliberate: `research/` answers scientific questions, while
`src/exoplanets_mf/` holds reusable and testable implementation. New research
stages can be added as sibling folders without turning `src/` into a collection
of one-off scripts.

## Reproduce outputs

Python 3.14.3 and all runtime packages are pinned. From the repository root,
create the isolated environment first:

```bash
make setup
```

`make setup` creates `.venv`, installs `requirements.lock`, and vendors the
OpenMP runtime needed by the GPBoost wheel on macOS. It does not modify the
system Python environment.

Use the canonical pipeline targets to produce outputs:

```bash
make check
make diagnostics
make models
make validation
make parameters
make reports
```

The same routine paper and diagnostic outputs can be produced in one command:

```bash
make reproduce
```

`make reproduce` verifies the input checksums and tests, writes exploratory and
reverse-label diagnostics, fits the linear and log10 model families, runs the
matched GPBoost comparison, produces validation summaries, and generates fitted
hyperparameter reports.

The slower max-data GPBoost comparison is intentionally outside the default
pipeline. Produce it together with the routine outputs when needed:

```bash
make reproduce-heavy
```

The main result sets are written to:

1. `results/exploratory/` and `results/reverse_label/` for diagnostics.
2. `results/modelling/02_augmented_wavelength/cv/` and
   `results/log_modelling/02_augmented_wavelength/cv/` for sklearn/custom-kernel
   model comparisons.
3. `results/modelling/03_gpboost_comparison/matched_subsample/` for the matched
   GPBoost comparison.
4. `results/validation/` for holdout, full-CV, and Stats101 summaries.
5. `results/parameters/` for fitted-hyperparameter figures and tables.
6. `results/modelling/03_gpboost_comparison/max_data/` for the optional heavy
   GPBoost comparison.

To inspect the available commands:

```bash
make help
```

Or run one workflow directly:

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python \
  research/exploratory/00_spectra_exploration.py
```

All generated exploratory figures go to `results/exploratory/`, independent of
the directory from which a script is launched. Random sampling and estimators
use the shared seed in `src/exoplanets_mf/reproducibility.py`. A complete run
also writes `run_metadata.json` with the Git revision, data checksums,
environment versions, platform, and seed used for that result.

## Add a new analysis or model

- Put one-off scientific orchestration in the appropriate `research/` folder.
- Put code reused by multiple workflows in `src/exoplanets_mf/`.
- Read inputs from `data/`; never overwrite them.
- Write generated files to the matching `results/` folder.
- Record parameters, seed, evaluation protocol, and expected outputs in the
  workflow or experiment README.
- Add tests for shared logic and run `make check` before committing.
