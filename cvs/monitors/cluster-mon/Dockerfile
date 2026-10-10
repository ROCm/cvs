# Multi-stage build for CVS Cluster Monitor (Go backend + React UI)
FROM golang:1.25-alpine AS go-builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ ./cmd/
COPY internal/ ./internal/
RUN CGO_ENABLED=0 GOOS=linux go build -o /cluster-mon ./cmd/cluster-mon

FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata \
    && mkdir -p /app/config /root/.ssh \
    && chmod 700 /root/.ssh

WORKDIR /app
ENV CLUSTER_MONITOR_HOME=/app
ENV HOME=/root
ENV LISTEN_ADDR=:8001
ENV STATIC_DIR=/app/static

COPY --from=go-builder /cluster-mon /usr/local/bin/cluster-mon
COPY --from=frontend-builder /app/frontend/dist /app/static

EXPOSE 8001

CMD ["/usr/local/bin/cluster-mon", "-listen", ":8001", "-config-dir", "/app/config", "-static", "/app/static"]
