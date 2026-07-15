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
│   ├── modelling/           # model experiments and evaluations
│   └── validation/          # predictive-performance validation studies
├── src/exoplanets_mf/       # reusable loaders, paths, and shared code
├── results/
│   ├── exploratory/         # generated exploratory artifacts
│   ├── reverse_label/       # generated inverse-diagnostic artifacts
│   ├── modelling/           # generated model artifacts
│   └── validation/          # generated validation artifacts
├── tests/                   # data and code integrity checks
├── pyproject.toml           # installable project metadata
└── requirements.lock        # pinned Python environment
```

The split is deliberate: `research/` answers scientific questions, while
`src/exoplanets_mf/` holds reusable and testable implementation. New research
stages can be added as sibling folders without turning `src/` into a collection
of one-off scripts.

## Reproduce the environment

Python 3.14.3 and all runtime packages are pinned. From the repository root:

```bash
make setup
make data-check
make verify
```

`make setup` creates an isolated `.venv`; it does not modify the system Python
environment. The checksum step verifies that the input bytes match the dataset
used for the analyses.

## Run the research workflows

Run the complete exploratory sequence:

```bash
make exploratory
```

Run the reverse-label sensitivity diagnostic, which treats spectral channels as
predictors and physical parameters as outputs:

```bash
make reverse-label
```

Run the model-fitting pipelines (linear and log10-transformed spectra) and
the validation studies with:

```bash
make modelling      # cost benchmark, joint MF-GP fit, model 1A vs 1B CV
make model2         # wavelength-augmented joint MF-GP: benchmark, fits, CV vs Model 1
make log-modelling  # same pipeline on log10-transformed spectra
make validation     # sample-81-only holdout study
make rho-loo-validation  # all-97 LOO for Model 1A/1B, log, and log-vs-linear
make full-cv        # 5-fold CV of the joint MF-GP over all 97 HF samples
```

Or run one workflow:

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python \
  research/exploratory/00_spectra_exploration.py
```

All generated exploratory figures go to `results/exploratory/`, independent of
the directory from which a script is launched. Random sampling and estimators
use the shared seed in `src/exoplanets_mf/reproducibility.py`. A complete run
also writes `run_metadata.json` with the Git revision, data checksums,
environment versions, platform, and seed used for that result.

The existing top-level `figures/` directory contains historical outputs from
before this structure was introduced. New runs do not modify it.

## Add a new analysis or model

- Put one-off scientific orchestration in the appropriate `research/` folder.
- Put code reused by multiple workflows in `src/exoplanets_mf/`.
- Read inputs from `data/`; never overwrite them.
- Write generated files to the matching `results/` folder.
- Record parameters, seed, evaluation protocol, and expected outputs in the
  workflow or experiment README.
- Add tests for shared logic and run `make verify` before committing.
