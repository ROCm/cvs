VENV = .ut_venv
PYTHON := $(shell command -v python3 || command -v python)
IMAGE = cvs-cluster-monitor
# Build context is the repo root so the Dockerfile can build the cvs package.
CVS_REPO_ROOT = ../../..

.PHONY: ut docker-build clean help

help:
	@echo "Available targets:"
	@echo "  ut           - Create test venv, install backend deps, run unit tests"
	@echo "  docker-build - Build the cluster-mon Docker image (self-contained)"
	@echo "  clean        - Remove the test venv"

ut:
	@echo "Setting up cluster-mon test venv..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -r backend/requirements.txt
	# cluster-mon now imports cvs.lib.parallel.multiprocess_pssh via the SSH adapter.
	# Install the cvs package with --no-deps so it does NOT pull the repo-root
	# requirements (which pin a different parallel-ssh); the SSH libs the adapter
	# needs are already provided by backend/requirements.txt above.
	$(VENV)/bin/pip install -q --no-deps -e ../../..
	@echo "Running cluster-mon unit tests..."
	cd backend && ../$(VENV)/bin/python run_all_unittests.py

# Build the image. The Dockerfile builds the cvs package itself (from the
# repo-root build context), so a plain `docker build` / `docker compose build`
# is fully self-contained - no pre-step required.
# If the Docker daemon isn't reachable as the current user (permission denied on
# the socket), automatically retry with sudo - which prompts for a password in
# an interactive terminal.
docker-build:
	@if docker info >/dev/null 2>&1; then \
		echo "Building image as current user..."; \
		docker build -f Dockerfile -t $(IMAGE) $(CVS_REPO_ROOT) ; \
	else \
		echo "Docker daemon not accessible without sudo; retrying with sudo (may prompt for password)..."; \
		sudo docker build -f Dockerfile -t $(IMAGE) $(CVS_REPO_ROOT) ; \
	fi

clean:
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$$(pwd)/$(VENV)" ]; then \
		echo "ERROR: You are currently in the venv. Please run 'deactivate' first."; \
		exit 1; \
	fi
	@echo "Cleaning test venv, build artifacts, and caches..."
	rm -rf $(VENV) dist build *.egg-info
	# Stray artifacts from any host-side cvs build at the repo root.
	rm -rf $(CVS_REPO_ROOT)/build $(CVS_REPO_ROOT)/cvs.egg-info
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
