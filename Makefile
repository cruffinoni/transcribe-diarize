SHELL := /bin/bash

SYS_PYTHON ?= python3
VENV := .venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
VENV_SENTINEL := $(VENV)/.installed

.PHONY: help install deps test smoke run clean clean-venv

help:
	@echo "Targets:"
	@echo "  install     Create .venv (if needed) and install package + pytest"
	@echo "  deps        Install pinned requirements.txt into .venv"
	@echo "  test        Run full pytest suite"
	@echo "  smoke       Run CLI smoke test (needs HF_TOKEN and ffmpeg)"
	@echo "  clean       Remove caches and build artifacts (keeps .venv)"
	@echo "  clean-venv  Remove .venv"

$(VENV_SENTINEL):
	test -d $(VENV) || $(SYS_PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e .
	$(VENV_PIP) install pytest
	touch $(VENV_SENTINEL)

install: $(VENV_SENTINEL)
	@echo "Virtual env ready at $(VENV)"

deps: $(VENV_SENTINEL)
	$(VENV_PIP) install -r requirements.txt

test: $(VENV_SENTINEL)
	$(VENV_PYTHON) -m pytest -q

smoke: $(VENV_SENTINEL)
	$(VENV_PYTHON) -m pytest -q tests/test_smoke_cli.py

clean:
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf out

clean-venv:
	rm -rf $(VENV)
