VENV_DIR = test_venv
RUFF_VENV_DIR = ruff_venv
RUFF_VERSION = 0.14.8
PYTHON := $(shell command -v python3 || command -v python)
PIP = $(VENV_DIR)/bin/pip
RUFF_PIP = $(RUFF_VENV_DIR)/bin/pip
RUFF = $(RUFF_VENV_DIR)/bin/ruff
CVS = $(VENV_DIR)/bin/cvs

.PHONY: all help build venv install ut test clean_venv clean_build clean_pycache clean

all: build venv install test

help:
	@echo "Available targets:"
	@echo "  build      - Build source distribution"
	@echo "  venv       - Create virtual environment"
	@echo "  ruff-venv  - Create ruff virtual environment"
	@echo "  install    - Install from built distribution"
	@echo "  ut         - Execute all Unittests"
	@echo "  test       - Execute all UTs and cvs cli tests"
	@echo "  lint       - Run ruff linter"
	@echo "  format     - Run ruff formatter"
	@echo "  lint-fix   - Run ruff linter with auto-fix"
	@echo "  all        - Run build, venv, install, and test"
	@echo "  clean      - Remove virtual environment, build artifacts, and Python cache files"

build: clean_build lint
	@echo "Building source distribution..."
	$(PYTHON) setup.py sdist

venv: clean_venv
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

ruff-venv:
	@if [ ! -d $(RUFF_VENV_DIR) ]; then \
		echo "Creating ruff virtual environment..."; \
		$(PYTHON) -m venv $(RUFF_VENV_DIR); \
		echo "Installing ruff..."; \
		$(RUFF_PIP) install ruff==$(RUFF_VERSION); \
	fi

install: venv build
	@echo "Installing from built distribution..."
	$(PIP) install dist/*.tar.gz

ut: install
	@echo "Unit Testing cvs..."
	$(VENV_DIR)/bin/python run_all_unittests.py

test: ut
	@echo "Testing cvs commands..."
	CVS="$(CVS)" ./test_cli.sh

lint: ruff-venv
	@echo "Running ruff linter..."
	@if ! $(RUFF) check . ; then \
		echo "\n\nLinting failed. Run 'make lint-fix' to auto-fix issues.\n"; exit 1; \
	fi
	@if ! $(RUFF) format --check . ; then \
		echo "\n\nFormatting failed. Run 'make format' to auto-fix issues.\n"; exit 1; \
	fi

format: ruff-venv
	@echo "Running ruff formatter..."
	$(RUFF) format .

lint-fix: ruff-venv
	@echo "Running ruff linter with auto-fix..."
	$(RUFF) check . --fix
	$(RUFF) format .

clean_venv:
	@echo "Removing virtual environment..."
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$$(pwd)/$(VENV_DIR)" ]; then \
		echo "ERROR: You are currently in the venv. Please run 'deactivate' first."; \
		exit 1; \
	fi
	rm -rf $(VENV_DIR)

clean_ruff_venv:
	@echo "Removing ruff virtual environment..."
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$$(pwd)/$(RUFF_VENV_DIR)" ]; then \
		echo "ERROR: You are currently in the ruff venv. Please run 'deactivate' first."; \
		exit 1; \
	fi
	rm -rf $(RUFF_VENV_DIR)

clean_build:
	@echo "Removing build artifacts..."
	rm -rf dist/ *.egg-info/ src/*.egg-info/

clean_pycache:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

clean: clean_venv clean_ruff_venv clean_build clean_pycache
