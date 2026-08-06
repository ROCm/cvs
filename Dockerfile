# syntax=docker/dockerfile:1
#
# CVS is a head-node CLI: it connects to the cluster over SSH and runs the
# requested test suite remotely. The image intentionally does not include
# ROCm or the workload binaries; those belong on the cluster nodes (or in the
# workload images used by CVS's container backend).

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

WORKDIR /build

# Wheels cover the normal installation path, while these packages provide a
# portable fallback for dependencies with native extensions (for example
# parallel-ssh/libssh2 and cryptography).
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        libffi-dev \
        libssl-dev \
        libssh2-1-dev \
    && rm -rf /var/lib/apt/lists/*

# Keep dependency installation cacheable when only CVS source changes.
COPY requirements.txt setup.py MANIFEST.in README.md LICENSE pytest.ini version.txt ./
COPY cvs ./cvs

# pytest.ini is not package data, but cvs run needs it while pytest discovers
# installed test modules and their CVS-specific CLI options.
RUN python -m venv /opt/cvs-venv \
    && /opt/cvs-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/cvs-venv/bin/pip install --no-cache-dir . \
    && site_packages=$(/opt/cvs-venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && install --mode=0644 pytest.ini "$site_packages/cvs/pytest.ini"

FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Cluster Validation Suite" \
      org.opencontainers.image.description="Head-node CLI for validating AMD AI clusters" \
      org.opencontainers.image.source="https://github.com/ROCm/cvs" \
      org.opencontainers.image.licenses="MIT"

# CVS uses Python SSH libraries, and the CLI utilities are retained for
# operator workflows and diagnostic commands launched from the image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        iproute2 \
        libffi8 \
        libssh2-1 \
        libssl3 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/cvs-venv /opt/cvs-venv

ENV VIRTUAL_ENV=/opt/cvs-venv \
    PATH="/opt/cvs-venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# /workspace receives user-provided cluster and suite configuration; /results
# is a convenient bind-mount target for reports and logs. The existing CVS
# defaults remain available for callers that use /tmp/cvs or /var/www/html/cvs.
RUN mkdir -p /workspace /results /tmp/cvs /var/www/html/cvs

WORKDIR /workspace

ENTRYPOINT ["cvs"]
CMD ["--help"]
