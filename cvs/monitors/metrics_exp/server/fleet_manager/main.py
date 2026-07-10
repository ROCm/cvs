"""AMD GPU Fleet Manager - Main FastAPI Application."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import init_db
from .api.routes import (
    nodegroups_router,
    nodes_router,
    metrics_router,
    dashboards_router,
    monitoring_router,
    monitoring_servers_router,
    metric_groups_router,
    control_nodegroups_router,
)
from .api.schemas import HealthResponse, ServiceHealth, FleetStats
from .services import PrometheusConfigManager, GrafanaProvisioner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce httpx logging noise from health checks
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting AMD GPU Fleet Manager...")
    init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down AMD GPU Fleet Manager...")


app = FastAPI(
    title="AMD GPU Fleet Manager",
    description="Fleet monitoring and management for AMD GPU clusters",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(nodegroups_router, prefix="/api/v1")
app.include_router(nodes_router, prefix="/api/v1")
app.include_router(metrics_router, prefix="/api/v1")
app.include_router(dashboards_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(monitoring_servers_router, prefix="/api/v1")
app.include_router(metric_groups_router, prefix="/api/v1")
app.include_router(control_nodegroups_router, prefix="/api/v1")


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Serve the UI or API info."""
    static_path = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "name": "AMD GPU Fleet Manager",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api", tags=["Root"])
async def api_info():
    """API info endpoint."""
    return {
        "name": "AMD GPU Fleet Manager",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check health of all services."""
    services = []

    # Check Prometheus
    prometheus = PrometheusConfigManager()
    prom_healthy = await prometheus.check_prometheus_health()
    services.append(
        ServiceHealth(
            name="prometheus",
            status="healthy" if prom_healthy else "unhealthy",
        )
    )

    # Check Grafana
    grafana = GrafanaProvisioner()
    grafana_healthy = await grafana.check_health()
    services.append(
        ServiceHealth(
            name="grafana",
            status="healthy" if grafana_healthy else "unhealthy",
        )
    )

    # Check database
    try:
        from sqlalchemy import text
        from .models import SessionLocal

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        services.append(ServiceHealth(name="database", status="healthy"))
    except Exception as e:
        services.append(ServiceHealth(name="database", status="unhealthy", message=str(e)))

    overall_status = "healthy" if all(s.status == "healthy" for s in services) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        services=services,
    )


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check_api():
    """Check health of all services (API endpoint)."""
    return await health_check()


@app.get("/api/v1/stats", response_model=FleetStats, tags=["Statistics"])
async def get_fleet_stats():
    """Get fleet-wide statistics."""
    from .models import SessionLocal, NodeGroup, Node
    from .models.database import NodeStatus

    db = SessionLocal()
    try:
        total_groups = db.query(NodeGroup).count()
        total_nodes = db.query(Node).count()
        active_nodes = db.query(Node).filter(Node.status == NodeStatus.ACTIVE.value).count()
        pending_nodes = db.query(Node).filter(Node.status == NodeStatus.PENDING.value).count()
        error_nodes = db.query(Node).filter(Node.status == NodeStatus.ERROR.value).count()

        # Sum GPU counts from active nodes
        from sqlalchemy import func

        total_gpus = (
            db.query(func.sum(Node.gpu_count))
            .filter(
                Node.status == NodeStatus.ACTIVE.value,
                Node.gpu_count.isnot(None),
            )
            .scalar()
            or 0
        )

        return FleetStats(
            total_node_groups=total_groups,
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            pending_nodes=pending_nodes,
            error_nodes=error_nodes,
            total_gpus=total_gpus,
        )
    finally:
        db.close()


# Serve static files for UI if present
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    assets_path = os.path.join(static_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


# SPA catch-all route - must be defined after all other routes
@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard/{full_path:path}", include_in_schema=False)
@app.get("/nodegroups", include_in_schema=False)
@app.get("/nodegroups/{full_path:path}", include_in_schema=False)
@app.get("/control-nodegroups", include_in_schema=False)
@app.get("/control-nodegroups/{full_path:path}", include_in_schema=False)
@app.get("/monitoring", include_in_schema=False)
@app.get("/monitoring/{full_path:path}", include_in_schema=False)
@app.get("/metrics", include_in_schema=False)
@app.get("/metrics/{full_path:path}", include_in_schema=False)
@app.get("/settings", include_in_schema=False)
@app.get("/settings/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str = ""):
    """Serve the SPA for client-side routes."""
    static_path = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="UI not found", status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
