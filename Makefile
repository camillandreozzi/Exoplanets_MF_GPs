SHELL := /bin/sh

PYTHON ?= .venv/bin/python
SYSTEM_PYTHON ?= python3
PYTHON_VERSION := 3.14.3
RUN_ENV := PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib

.PHONY: check-python setup verify data-check exploratory reverse-label modelling model2 model2k log-modelling validation sample-81-validation rho-loo-validation full-cv stats101

check-python:
	@$(SYSTEM_PYTHON) -c "import platform; actual = platform.python_version(); expected = '$(PYTHON_VERSION)'; assert actual == expected, f'Expected Python {expected}, found {actual}'"

setup: check-python
	$(SYSTEM_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install --requirement requirements.lock

verify:
	$(RUN_ENV) $(PYTHON) -m unittest discover -s tests -v

data-check:
	shasum -a 256 -c data/checksums.sha256

exploratory:
	$(RUN_ENV) $(PYTHON) -m research.exploratory

reverse-label:
	$(RUN_ENV) $(PYTHON) -m research.reverse_label

modelling:
	$(RUN_ENV) $(PYTHON) -m research.modelling.01_per_wavelength_ar1

model2:
	$(RUN_ENV) $(PYTHON) -m research.modelling.02_augmented_wavelength

model2k:
	$(RUN_ENV) $(PYTHON) -m research.modelling.02b_kronecker

log-modelling:
	$(RUN_ENV) $(PYTHON) -m research.log_modelling.01_per_wavelength_ar1

validation:
	$(RUN_ENV) $(PYTHON) -m research.validation.01_loo_holdout_validation

sample-81-validation:
	$(RUN_ENV) $(PYTHON) -m research.validation.01_loo_holdout_validation

rho-loo-validation:
	$(RUN_ENV) $(PYTHON) -m research.validation.03_rho_loo_model_comparison

stats101:
	$(RUN_ENV) $(PYTHON) -m research.validation.04_stats101_summary

full-cv:
	$(RUN_ENV) $(PYTHON) -m research.validation.02_full_cv
