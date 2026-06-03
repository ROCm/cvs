"""API routes for Fleet Manager."""

from .nodegroups import router as nodegroups_router
from .nodes import router as nodes_router
from .metrics import router as metrics_router
from .dashboards import router as dashboards_router
from .monitoring import router as monitoring_router
from .monitoring_servers import router as monitoring_servers_router
from .metric_groups import router as metric_groups_router

__all__ = [
    "nodegroups_router",
    "nodes_router",
    "metrics_router",
    "dashboards_router",
    "monitoring_router",
    "monitoring_servers_router",
    "metric_groups_router",
]
