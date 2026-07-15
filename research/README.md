# Research workflows

This directory contains runnable research code, grouped by scientific stage:

- `exploratory/`: diagnostics and analyses used to understand the data.
- `reverse_label/`: inverse spectrum-to-parameter association diagnostics.
- `modelling/`: model fitting, evaluation, and comparison experiments.
- `validation/`: predictive-performance validation studies.

Add future stages as sibling directories. Keep reusable loaders, metrics, and
model components in `src/exoplanets_mf/`; workflow files should mainly
coordinate those components and write results.

Each workflow must:

1. read inputs from `data/` without modifying them;
2. use the shared random seed, or record an explicitly chosen alternative;
3. write generated artifacts under its matching `results/` directory;
4. document any parameters that materially affect its scientific result.
