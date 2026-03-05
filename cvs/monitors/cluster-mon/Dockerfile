# Multi-stage build for CVS Cluster Monitor
# Stage 1: Build React frontend
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Build frontend for production
RUN npm run build


# Stage 2: Main application image
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

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app/ ./app/

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
