SHELL := /bin/sh
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
SYSTEM_PYTHON ?= python3
PYTHON_VERSION := 3.14.3
RUN_ENV := PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib

.PHONY: help .python-version setup check diagnostics models validation reports reproduce reproduce-heavy
.PHONY: .stage-linear-family .stage-log-family .stage-augmented-family .stage-gpb-fits .stage-gpb-matched .stage-gpb-max
.PHONY: .stage-holdout .stage-cv97 .stage-validation-summary .stage-parameter-reports

help:
	@printf '%s\n' \
		'Canonical pipeline:' \
		'  make setup             Create the pinned Python environment' \
		'  make check             Verify data checksums and run unit tests' \
		'  make diagnostics       Produce exploratory and reverse-label outputs' \
		'  make models            Produce model fits and matched comparisons' \
		'  make validation        Produce holdout, full-CV, and Stats101 outputs' \
		'  make reports           Produce fitted-parameter figures and tables' \
		'  make reproduce         Produce the routine paper and diagnostic outputs' \
		'  make reproduce-heavy   Also produce max-data GPBoost outputs'

.python-version:
	@$(SYSTEM_PYTHON) -c "import platform; actual = platform.python_version(); expected = '$(PYTHON_VERSION)'; assert actual == expected, f'Expected Python {expected}, found {actual}'"

setup: .python-version
	$(SYSTEM_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install --requirement requirements.lock
	.venv/bin/python tools/fix_gpboost_libomp.py

check:
	shasum -a 256 -c data/checksums.sha256
	$(RUN_ENV) $(PYTHON) -m unittest discover -s tests -v

diagnostics:
	$(RUN_ENV) $(PYTHON) -m research.exploratory
	$(RUN_ENV) $(PYTHON) -m research.reverse_label

.stage-linear-family:
	$(RUN_ENV) $(PYTHON) -m research.modelling.01_per_wavelength_ar1

.stage-log-family: .stage-linear-family
	$(RUN_ENV) $(PYTHON) -m research.log_modelling.01_per_wavelength_ar1

.stage-augmented-family: .stage-log-family
	$(RUN_ENV) $(PYTHON) -m research.modelling.02_augmented_wavelength

.stage-gpb-fits: .stage-augmented-family
	$(RUN_ENV) $(PYTHON) research/modelling/01_per_wavelength_ar1/04_fit_model1_gpboost.py
	$(RUN_ENV) $(PYTHON) research/modelling/02_augmented_wavelength/03_fit_model2_gpboost.py

.stage-gpb-matched: .stage-gpb-fits
	$(RUN_ENV) $(PYTHON) -m research.modelling.03_gpboost_comparison --mode matched

.stage-gpb-max: .stage-gpb-matched
	$(RUN_ENV) $(PYTHON) -m research.modelling.03_gpboost_comparison --mode max-data

models: .stage-gpb-matched

.stage-holdout:
	$(RUN_ENV) $(PYTHON) -m research.validation.01_loo_holdout_validation

.stage-cv97: .stage-holdout
	$(RUN_ENV) $(PYTHON) -m research.validation.02_full_cv

.stage-validation-summary: models .stage-cv97
	$(RUN_ENV) $(PYTHON) -m research.validation.04_stats101_summary

validation: .stage-validation-summary

.stage-parameter-reports: models .stage-holdout
	$(RUN_ENV) $(PYTHON) -m research.parameters

reports: .stage-parameter-reports

reproduce: check diagnostics models validation reports

reproduce-heavy: reproduce .stage-gpb-max
