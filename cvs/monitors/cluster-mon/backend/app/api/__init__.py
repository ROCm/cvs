"""
API router for CVS Cluster Monitor.
"""

from fastapi import APIRouter
from app.api import cluster, nodes, metrics, config, software, restart, packages, logs, ssh_keys, collectors

router = APIRouter()

# Include sub-routers
router.include_router(cluster.router, prefix="/cluster", tags=["cluster"])
router.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
router.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
router.include_router(config.router, prefix="/config", tags=["config"])
router.include_router(software.router, prefix="/software", tags=["software"])
router.include_router(restart.router, prefix="/backend", tags=["backend"])
router.include_router(packages.router, prefix="/packages", tags=["packages"])
router.include_router(logs.router, prefix="/logs", tags=["logs"])
router.include_router(ssh_keys.router, prefix="/ssh-keys", tags=["ssh-keys"])
router.include_router(collectors.router, prefix="/collectors", tags=["collectors"])
