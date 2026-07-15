# Input data

These CSV files are the immutable inputs to the research workflows. Do not
overwrite them with processed data or model predictions.

`checksums.sha256` records the exact byte-level dataset version used by this
repository. Run `make data-check` before reproducing results. If an intentional
data update occurs, document its provenance and regenerate the checksums in the
same commit.

Dataset roles and alignment assumptions are documented in
`src/exoplanets_mf/data.py` and validated whenever `load_all()` is called.
