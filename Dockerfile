# syntax=docker/dockerfile:1
#
# CVS Unified Platform daemon - single image serving all three tiles.
#
# Multi-stage build (mirrors the pattern of the existing cluster-mon and
# fleet-manager Dockerfiles, swapping the Python backend for a Go binary):
#   1. node stage builds the React UI bundle
#   2. go stage compiles the daemon with the UI embedded via go:embed
#   3. minimal runtime stage ships just the static binary
#
# The runtime stage ships the Go daemon together with the CVS Python CLI so the
# Test Execution tile can shell out to `cvs list --json` (the authoritative
# source of test suites). openssh-client is included for later SSH slices.

# ---- Stage 1: build the React UI ----
FROM node:20-alpine AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: build the Go daemon with the embedded UI ----
FROM golang:1.25-alpine AS go-builder
WORKDIR /src
COPY api/go.mod api/go.sum ./
RUN go mod download
COPY api/ ./
# Replace the committed placeholder bundle with the freshly built UI.
RUN rm -rf internal/webui/dist
COPY --from=web-builder /web/dist ./internal/webui/dist
ARG VERSION=0.0.0-dev
ARG COMMIT=unknown
ARG BUILD_TIME=unknown
RUN CGO_ENABLED=0 go build -trimpath \
    -ldflags "-s -w \
      -X github.com/ROCm/cvs/api/internal/version.Version=${VERSION} \
      -X github.com/ROCm/cvs/api/internal/version.Commit=${COMMIT} \
      -X github.com/ROCm/cvs/api/internal/version.BuildTime=${BUILD_TIME}" \
    -o /out/cvs-server ./cmd/server

# ---- Stage 3: runtime (Go daemon + CVS CLI) ----
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install the CVS CLI so the daemon can shell out to `cvs list` / `cvs run`.
#
# Use an editable install and ship the repo-root pytest.ini so the in-container
# layout matches a normal venv install: the source tree stays at /opt/cvs/cvs
# and pytest.ini sits above it. This lets `cvs run` resolve pytest's rootdir to
# /opt/cvs and discover cvs/conftest.py (which registers --cluster_file /
# --config_file). A plain `pip install .` copies the package into site-packages
# without pytest.ini, so rootdir collapses to the test dir and those options are
# rejected.
WORKDIR /opt/cvs
COPY requirements.txt setup.py version.txt MANIFEST.in pytest.ini ./
COPY cvs/ ./cvs/
RUN pip install --no-cache-dir -e .

WORKDIR /app
COPY --from=go-builder /out/cvs-server /app/cvs-server
# Persistent data dir for FileStores + uploaded SSH keys. Must be writable by the
# non-root runtime user; a named volume is mounted here in docker-compose so the
# inventory and keys survive container restarts.
RUN useradd -u 10001 -m cvs \
    && mkdir -p /app/data/keys \
    && chown -R cvs:cvs /app/data
USER cvs
EXPOSE 8080
ENV CVS_LISTEN_ADDR=:8080 \
    CVS_BIN=cvs \
    CVS_CONFIG_DIR=/opt/cvs/cvs/input/config_file \
    CVS_DATA_DIR=/app/data \
    CVS_SSH_KEY_DIR=/app/data/keys
VOLUME ["/app/data"]
ENTRYPOINT ["/app/cvs-server"]
