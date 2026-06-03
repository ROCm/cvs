VENV = .ut_venv
PYTHON := $(shell command -v python3 || command -v python)
IMAGE = cvs-cluster-monitor

.PHONY: ut docker-build clean help

help:
	@echo "Available targets:"
	@echo "  ut           - Create test venv, install backend deps, run unit tests"
	@echo "  docker-build - Build the cluster-mon Docker image"
	@echo "  clean        - Remove the test venv"

ut:
	@echo "Setting up cluster-mon test venv..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -r backend/requirements.txt
	@echo "Running cluster-mon unit tests..."
	cd backend && ../$(VENV)/bin/python run_all_unittests.py

docker-build:
	docker build -f Dockerfile -t $(IMAGE) .

clean:
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$$(pwd)/$(VENV)" ]; then \
		echo "ERROR: You are currently in the venv. Please run 'deactivate' first."; \
		exit 1; \
	fi
	rm -rf $(VENV)
