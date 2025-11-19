VENV_DIR = test_venv
PYTHON = python
PIP = $(VENV_DIR)/bin/pip

.PHONY: help venv build install test clean all clean_venv clean_build

all: build venv install test

help:
	@echo "Available targets:"
	@echo "  venv     - Create virtual environment"
	@echo "  install  - Install requirements"
	@echo "  test     - Test cvs list and cvs generate commands"
	@echo "  all      - Run venv, install, and test"
	@echo "  clean    - Remove virtual environment and Python cache files"

venv: clean_venv
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

install: venv build
	@echo "Installing from built distribution..."
	$(PIP) install -r requirements.txt

test: install
	@echo "Unit Testing cvs..."
	$(VENV_DIR)/bin/python run_all_unittests.py

clean_venv:
	@echo "Removing virtual environment..."
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$$(pwd)/$(VENV_DIR)" ]; then \
		echo "ERROR: You are currently in the venv. Please run 'deactivate' first."; \
		exit 1; \
	fi
	rm -rf $(VENV_DIR)

clean_pycache:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

clean: clean_venv clean_pycache
