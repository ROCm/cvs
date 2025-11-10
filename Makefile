# Makefile for testing cluster_validator package build and install

VENV_DIR = test_venv
PYTHON = python
PIP = $(VENV_DIR)/bin/pip
CVS = $(VENV_DIR)/bin/cvs

.PHONY: help venv build install test clean all clean_venv clean_build

all: build venv install test

help:
	@echo "Available targets:"
	@echo "  build    - Build source distribution"
	@echo "  venv     - Create virtual environment"
	@echo "  install  - Install from built distribution"
	@echo "  test     - Test cvs list and cvs generate commands"
	@echo "  all      - Run build, venv, install, and test"
	@echo "  clean    - Remove virtual environment, build artifacts, and Python cache files"

build: clean_build
	@echo "Building source distribution..."
	$(PYTHON) setup.py sdist

venv: clean_venv
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

install: venv build
	@echo "Installing from built distribution..."
	$(PIP) install dist/*.tar.gz

test: install
	@echo "Testing cvs commands..."
	CVS="$(CVS)" ./test_cli.sh

clean_venv:
	@echo "Removing virtual environment..."
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$$(pwd)/$(VENV_DIR)" ]; then \
		echo "ERROR: You are currently in the venv. Please run 'deactivate' first."; \
		exit 1; \
	fi
	rm -rf $(VENV_DIR)

clean_build:
	@echo "Removing build artifacts..."
	rm -rf dist/ *.egg-info/ src/*.egg-info/

clean_pycache:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

clean: clean_venv clean_build clean_pycache
