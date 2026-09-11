#!/usr/bin/env bash
# Cloud Agent bootstrap for the Cluster Validation Suite (CVS).
#
# CVS is a head-node Python CLI. This script prepares a ready-to-use checkout:
# it installs the `cvs` CLI into .cvs_venv (the README quickstart layout) and
# pre-creates the lint/format toolchain so quality gates run without a
# first-use delay. It is idempotent and safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."

# The base image ships Python 3 and build tools but not the venv/ensurepip
# package that `python -m venv` needs to bootstrap pip. Install it only when
# missing so re-runs and build snapshots stay fast.
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
        py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)')"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "python3.${py_minor}-venv" \
            || sudo apt-get install -y -qq python3-venv
    else
        echo "ERROR: python3 venv/ensurepip is unavailable and sudo is not present to install it." >&2
        exit 1
    fi
fi

# Build the source distribution and install the `cvs` CLI into .cvs_venv.
# `make install` recreates .cvs_venv each run, so this converges cleanly.
make install

# Pre-build the ruff/pylint venv so `make lint` and `make fmt-check` are ready.
make ruff-venv

echo
echo "CVS environment ready. Activate the CLI with: source .cvs_venv/bin/activate"
