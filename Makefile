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
	@echo "  lint       - Run ruff linter (checks code quality, not formatting)"
	@echo "  fmt        - Run ruff formatter"
	@echo "  fmt-check  - Check ruff formatting without modifying files"
	@echo "  lint-fix   - Run ruff linter with auto-fix (fixes code quality issues, not formatting)"
	@echo "  unsafe-lint-fix - Interactive unsafe lint fixes"
	@echo "  all        - Run build, venv, install, and test"
	@echo "  clean      - Remove virtual environment, build artifacts, and Python cache files"

build: clean_build fmt-check lint
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
	@if ! $(RUFF) check . --unsafe-fixes ; then \
		echo "\n\nLinting failed. Run 'make lint-fix' to auto-fix issues.\n"; exit 1; \
	fi

fmt: ruff-venv
	@echo "Running ruff formatter..."
	$(RUFF) format .

fmt-check: ruff-venv
	@echo "Checking ruff formatting..."
	@if ! $(RUFF) format --check . ; then \
		echo "\n\nFormatting check failed. Run 'make fmt' to auto-fix formatting issues.\n"; exit 1; \
	fi

lint-fix: ruff-venv
	@echo "Running ruff linter with auto-fix..."
	$(RUFF) check . --fix

unsafe-lint-fix: ruff-venv
	@echo ""
	@echo "WARNING: This will apply unsafe fixes that may remove unused variables or make other potentially breaking changes."
	@echo "You can fix these issues manually after careful review, or proceed with per-file confirmation."
	@echo ""
	@echo "Getting list of files with unsafe fixes..."
	@files=$$($(RUFF) check . --unsafe-fixes | awk '/ --> / {split($$2, a, ":"); print a[1]}' | sort | uniq); \
	echo "Files with unsafe fixes:"; \
	for file in $$files; do \
		echo "  - $$file"; \
	done; \
	echo ""; \
	if [ -z "$$files" ]; then \
		echo "No unsafe fixes needed."; \
		exit 0; \
	fi; \
	for file in $$files; do \
		echo "File: $$file has unsafe fixes."; \
		echo "=== DIFF for $$file ==="; \
		diff_output=$$($(RUFF) check $$file --unsafe-fixes --diff); \
		if [ -n "$$diff_output" ]; then \
			echo "$$diff_output"; \
			echo "=== END DIFF ==="; \
			echo "Apply fixes to this file? (y/N)"; \
			read -p "" confirm; \
			if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
				echo "Applying unsafe fixes to $$file..."; \
				$(RUFF) check $$file --fix --unsafe-fixes; \
			else \
				echo "Skipping $$file."; \
			fi; \
		else \
			echo "No unsafe fixes available for this file."; \
			echo "If you want to fix issues manually, run: $(RUFF) check $$file --unsafe-fixes"; \
			echo "=== END DIFF ==="; \
		fi; \
	done
	@echo "Running formatter..."
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
