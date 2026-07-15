# Validation

Predictive-performance / validation studies for the models defined in
`research/modelling/`. Kept as a separate top-level sibling to
`research/modelling/` so validation work never collides with that folder's
model-numbering sequence (`01_...`, `02_...`, ... reserved for models).

Use one numbered folder per validation study, following the same convention
as `research/modelling/`:

```text
validation/
├── 01_loo_holdout_validation/
│   ├── README.md
│   ├── 01_fit_excluding_holdout.py
│   └── 02_predict_and_evaluate.py
├── 02_full_cv/
│   ├── README.md
│   └── 01_cv_joint_mf_gp.py
└── 03_rho_loo_model_comparison/
    ├── README.md
    └── 01_compare_rho_loo.py
```

A study's README should record its question, input data, configuration,
random seed, evaluation protocol, and expected outputs. Reusable estimators
and metrics belong in `src/exoplanets_mf/`, while generated artifacts belong
in `results/validation/`.
