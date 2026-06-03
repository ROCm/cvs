# Multi-stage build for CVS Cluster Monitor
#
# NOTE: the build context is the REPO ROOT (see docker-compose.yml build.context),
# not this directory, so the image can build/install the repo-root `cvs` package
# (cvs.lib.parallel) that the SSH adapter imports. All COPY paths below are
# therefore relative to the repo root.

# Stage 1: Build React frontend
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies (copy lockfiles first for layer caching)
COPY cvs/monitors/cluster-mon/frontend/package*.json ./
RUN npm ci

# Build frontend for production
COPY cvs/monitors/cluster-mon/frontend/ ./
RUN npm run build


# Stage 2: Build the cvs wheel from the repo-root source
FROM python:3.10-slim AS cvs-builder

WORKDIR /src

# setup.py reads requirements.txt and version.txt at build time.
COPY setup.py requirements.txt version.txt ./
COPY cvs/ ./cvs/

# Build only the cvs wheel (no runtime deps; those come from backend reqs below).
RUN pip wheel --no-deps -w /wheels .


# Stage 3: Main application image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV CLUSTER_MONITOR_HOME=/app
ENV PYTHONUNBUFFERED=1
ENV HOME=/root

# Backend Python dependencies (provides the SSH stack: parallel-ssh/paramiko/scp)
COPY cvs/monitors/cluster-mon/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the cvs package (provides cvs.lib.parallel used by the SSH adapter).
# --no-deps so it does NOT override the backend's pinned SSH stack installed
# above; the libs the adapter needs come from backend/requirements.txt.
COPY --from=cvs-builder /wheels/*.whl /tmp/cvs-wheels/
RUN pip install --no-cache-dir --no-deps /tmp/cvs-wheels/*.whl

# Copy backend code
COPY cvs/monitors/cluster-mon/backend/app/ ./app/

# Copy frontend build from builder stage
COPY --from=frontend-builder /app/frontend/dist ./static

# Create config directory
RUN mkdir -p /app/config

# Expose port
EXPOSE 8001

# Health check - disabled temporarily for debugging
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:8001/health', timeout=5)" || exit 1

# Run FastAPI (serves both API and static frontend)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
