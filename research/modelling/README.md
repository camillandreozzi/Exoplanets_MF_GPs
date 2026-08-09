# Modelling

Place model experiments here once they are introduced. Use one numbered folder
per experiment when it has several files, for example:

```text
modelling/
└── 01_baseline_gp/
    ├── README.md
    ├── train.py
    └── evaluate.py
```

An experiment README should record its question, input data, configuration,
random seed, evaluation protocol, and expected outputs. Reusable estimators and
metrics belong in `src/exoplanets_mf/`, while generated models and tables belong
in `results/modelling/`.

Current paper-facing comparison sets:

- `make modelling_sklearn`: sklearn/mf_gp custom-kernel comparison across
  Model 1 and Model 2; outputs to
  `results/modelling/02_augmented_wavelength/cv/`.
- `make modelling_gpboost_comparison`: GPBoost comparison on the same canonical
  subsample and fold protocol; outputs to
  `results/modelling/03_gpboost_comparison/matched_subsample/`.
- `make modelling_gpboost`: larger configurable GPBoost/Vecchia comparison;
  outputs to `results/modelling/03_gpboost_comparison/max_data/`.
