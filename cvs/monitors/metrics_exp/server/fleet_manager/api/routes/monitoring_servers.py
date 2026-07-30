"""API routes for monitoring server management."""

import logging
import os
import asyncio
import base64
from typing import Optional, Dict, List
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models import get_db, MonitoringServer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring-servers", tags=["Monitoring Servers"])

SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "/app/ssh_keys")

# In-memory storage for installation jobs
_installation_jobs: Dict[str, Dict] = {}


# Pydantic schemas
class MonitoringServerCreate(BaseModel):
    name: str
    description: Optional[str] = None
    server_ip: Optional[str] = None
    server_hostname: Optional[str] = None
    prometheus_port: int = 30090
    loki_port: int = 30100
    grafana_port: int = 30030
    prometheus_retention_time: str = "15d"
    prometheus_retention_size: str = "50GB"
    prometheus_scrape_interval: str = "15s"
    prometheus_storage_path: Optional[str] = None
    loki_retention_days: int = 7
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "admin"
    setup_monitoring_stack: bool = False
    ssh_user: Optional[str] = None
    ssh_port: int = 22
    ssh_auth_type: str = "password"
    ssh_password: Optional[str] = None
    use_jump_host: bool = False
    jump_host: Optional[str] = None
    jump_port: int = 22
    jump_user: Optional[str] = None
    jump_auth_type: str = "key"
    jump_password: Optional[str] = None
    remote_auth_type: str = "key"
    remote_key_path: Optional[str] = None
    remote_password: Optional[str] = None
    use_push_gateway: bool = False
    push_gateway_port: int = 9091


class MonitoringServerUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    server_ip: Optional[str] = None
    server_hostname: Optional[str] = None
    prometheus_port: Optional[int] = None
    loki_port: Optional[int] = None
    grafana_port: Optional[int] = None
    prometheus_retention_time: Optional[str] = None
    prometheus_retention_size: Optional[str] = None
    prometheus_scrape_interval: Optional[str] = None
    prometheus_storage_path: Optional[str] = None
    loki_retention_days: Optional[int] = None
    grafana_admin_user: Optional[str] = None
    grafana_admin_password: Optional[str] = None
    setup_monitoring_stack: Optional[bool] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_auth_type: Optional[str] = None
    ssh_password: Optional[str] = None
    use_jump_host: Optional[bool] = None
    jump_host: Optional[str] = None
    jump_port: Optional[int] = None
    jump_user: Optional[str] = None
    jump_auth_type: Optional[str] = None
    jump_password: Optional[str] = None
    remote_auth_type: Optional[str] = None
    remote_key_path: Optional[str] = None
    remote_password: Optional[str] = None
    use_push_gateway: Optional[bool] = None
    push_gateway_port: Optional[int] = None


class MonitoringServerResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    server_ip: Optional[str]
    server_hostname: Optional[str]
    prometheus_port: int
    loki_port: int
    grafana_port: int
    prometheus_retention_time: str
    prometheus_retention_size: str
    prometheus_scrape_interval: str
    prometheus_storage_path: Optional[str]
    loki_retention_days: int
    grafana_admin_user: str
    setup_monitoring_stack: bool
    ssh_user: Optional[str]
    ssh_port: int
    ssh_auth_type: str
    has_ssh_key: bool
    has_ssh_password: bool
    use_jump_host: bool
    jump_host: Optional[str]
    jump_port: int
    jump_user: Optional[str]
    jump_auth_type: str
    has_jump_key: bool
    has_jump_password: bool
    remote_auth_type: str
    remote_key_path: Optional[str]
    has_remote_password: bool
    use_push_gateway: bool
    push_gateway_port: int
    stack_installed: bool
    last_install_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    node_group_count: int

    class Config:
        from_attributes = True


def server_to_response(server: MonitoringServer) -> MonitoringServerResponse:
    """Convert DB model to response schema."""
    return MonitoringServerResponse(
        id=server.id,
        name=server.name,
        description=server.description,
        server_ip=server.server_ip,
        server_hostname=server.server_hostname,
        prometheus_port=server.prometheus_port,
        loki_port=server.loki_port,
        grafana_port=server.grafana_port,
        prometheus_retention_time=server.prometheus_retention_time,
        prometheus_retention_size=server.prometheus_retention_size,
        prometheus_scrape_interval=server.prometheus_scrape_interval,
        prometheus_storage_path=server.prometheus_storage_path,
        loki_retention_days=server.loki_retention_days,
        grafana_admin_user=server.grafana_admin_user or "admin",
        setup_monitoring_stack=server.setup_monitoring_stack,
        ssh_user=server.ssh_user,
        ssh_port=server.ssh_port or 22,
        ssh_auth_type=server.ssh_auth_type or "password",
        has_ssh_key=bool(server.ssh_key_path),
        has_ssh_password=bool(server.ssh_password),
        use_jump_host=server.use_jump_host or False,
        jump_host=server.jump_host,
        jump_port=server.jump_port or 22,
        jump_user=server.jump_user,
        jump_auth_type=server.jump_auth_type or "key",
        has_jump_key=bool(server.jump_key_path),
        has_jump_password=bool(server.jump_password),
        remote_auth_type=server.remote_auth_type or "key",
        remote_key_path=server.remote_key_path,
        has_remote_password=bool(server.remote_password),
        use_push_gateway=server.use_push_gateway or False,
        push_gateway_port=server.push_gateway_port or 9091,
        stack_installed=server.stack_installed or False,
        last_install_at=server.last_install_at,
        created_at=server.created_at,
        updated_at=server.updated_at,
        node_group_count=len(server.node_groups) if server.node_groups else 0,
    )


@router.get("", response_model=List[MonitoringServerResponse])
def list_monitoring_servers(db: Session = Depends(get_db)):
    """List all monitoring servers."""
    servers = db.query(MonitoringServer).all()
    return [server_to_response(s) for s in servers]


@router.post("", response_model=MonitoringServerResponse)
def create_monitoring_server(
    server: MonitoringServerCreate,
    db: Session = Depends(get_db),
):
    """Create a new monitoring server configuration."""
    # Check for duplicate name
    existing = db.query(MonitoringServer).filter(MonitoringServer.name == server.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Monitoring server '{server.name}' already exists")

    db_server = MonitoringServer(
        name=server.name,
        description=server.description,
        server_ip=server.server_ip,
        server_hostname=server.server_hostname,
        prometheus_port=server.prometheus_port,
        loki_port=server.loki_port,
        grafana_port=server.grafana_port,
        prometheus_retention_time=server.prometheus_retention_time,
        prometheus_retention_size=server.prometheus_retention_size,
        prometheus_scrape_interval=server.prometheus_scrape_interval,
        prometheus_storage_path=server.prometheus_storage_path,
        loki_retention_days=server.loki_retention_days,
        grafana_admin_user=server.grafana_admin_user,
        grafana_admin_password=server.grafana_admin_password,
        setup_monitoring_stack=server.setup_monitoring_stack,
        ssh_user=server.ssh_user,
        ssh_port=server.ssh_port,
        ssh_auth_type=server.ssh_auth_type,
        ssh_password=server.ssh_password,
        use_jump_host=server.use_jump_host,
        jump_host=server.jump_host,
        jump_port=server.jump_port,
        jump_user=server.jump_user,
        jump_auth_type=server.jump_auth_type,
        jump_password=server.jump_password,
        remote_auth_type=server.remote_auth_type,
        remote_key_path=server.remote_key_path,
        remote_password=server.remote_password,
        use_push_gateway=server.use_push_gateway,
        push_gateway_port=server.push_gateway_port,
    )
    db.add(db_server)
    db.commit()
    db.refresh(db_server)

    logger.info(f"Created monitoring server: {server.name}")
    return server_to_response(db_server)


@router.get("/{server_id}", response_model=MonitoringServerResponse)
def get_monitoring_server(server_id: int, db: Session = Depends(get_db)):
    """Get a monitoring server by ID."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")
    return server_to_response(server)


@router.put("/{server_id}", response_model=MonitoringServerResponse)
def update_monitoring_server(
    server_id: int,
    update: MonitoringServerUpdate,
    db: Session = Depends(get_db),
):
    """Update a monitoring server configuration."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(server, field):
            setattr(server, field, value)

    db.commit()
    db.refresh(server)

    logger.info(f"Updated monitoring server: {server.name}")
    return server_to_response(server)


@router.delete("/{server_id}")
def delete_monitoring_server(server_id: int, db: Session = Depends(get_db)):
    """Delete a monitoring server configuration."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    if server.node_groups:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete monitoring server '{server.name}' - it has {len(server.node_groups)} node groups associated",
        )

    db.delete(server)
    db.commit()

    logger.info(f"Deleted monitoring server: {server.name}")
    return {"message": f"Monitoring server '{server.name}' deleted"}


@router.post("/{server_id}/ssh-key")
async def upload_ssh_key(
    server_id: int,
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH key for accessing the monitoring server."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    os.makedirs(SSH_KEY_PATH, exist_ok=True)
    key_path = os.path.join(SSH_KEY_PATH, f"monitoring_server_{server_id}_key")

    content = await key_file.read()
    with open(key_path, "wb") as f:
        f.write(content)
    os.chmod(key_path, 0o600)

    server.ssh_key_path = key_path
    db.commit()

    logger.info(f"Uploaded SSH key for monitoring server: {server.name}")
    return {"message": "SSH key uploaded successfully", "key_path": key_path}


@router.post("/{server_id}/jump-key")
async def upload_jump_key(
    server_id: int,
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH key for accessing the jump host."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    os.makedirs(SSH_KEY_PATH, exist_ok=True)
    key_path = os.path.join(SSH_KEY_PATH, f"monitoring_server_{server_id}_jump_key")

    content = await key_file.read()
    with open(key_path, "wb") as f:
        f.write(content)
    os.chmod(key_path, 0o600)

    server.jump_key_path = key_path
    db.commit()

    logger.info(f"Uploaded jump host SSH key for monitoring server: {server.name}")
    return {"message": "Jump host SSH key uploaded successfully", "key_path": key_path}


def _create_ssh_manager(server: MonitoringServer):
    """Create an SSHManager for the monitoring server."""
    from ...services.ssh_manager import SSHManager, JumpHostConfig

    jump_host = None
    if server.use_jump_host and server.jump_host:
        jump_host = JumpHostConfig(
            host=server.jump_host,
            port=server.jump_port or 22,
            username=server.jump_user or "root",
            auth_type=server.jump_auth_type or "key",
            key_path=server.jump_key_path,
            password=server.jump_password,
            remote_auth_type=server.remote_auth_type or "key",
            remote_key_path=server.remote_key_path,
            remote_password=server.remote_password,
        )

    auth_type = server.ssh_auth_type or "password"

    return SSHManager(
        host=server.server_ip,
        username=server.ssh_user,
        port=server.ssh_port or 22,
        auth_type=auth_type,
        key_path=server.ssh_key_path if auth_type == "key" else None,
        password=server.ssh_password if auth_type == "password" else None,
        jump_host=jump_host,
    )


def _validate_ssh_config(server: MonitoringServer):
    """Validate SSH configuration for monitoring server."""
    if not server.server_ip:
        raise HTTPException(status_code=400, detail="Server IP not configured")

    if not server.setup_monitoring_stack:
        raise HTTPException(status_code=400, detail="Remote monitoring setup not enabled")

    if not server.ssh_user:
        raise HTTPException(status_code=400, detail="SSH user not configured")

    if server.use_jump_host:
        if not server.jump_host:
            raise HTTPException(status_code=400, detail="Jump host enabled but no jump host address configured")
        if not server.jump_user:
            raise HTTPException(status_code=400, detail="Jump host enabled but no jump host user configured")

        jump_auth = server.jump_auth_type or "key"
        if jump_auth == "key" and not server.jump_key_path:
            raise HTTPException(status_code=400, detail="Jump host uses key auth but no key uploaded")
        if jump_auth == "password" and not server.jump_password:
            raise HTTPException(status_code=400, detail="Jump host uses password auth but no password configured")

        remote_auth = server.remote_auth_type or "key"
        if remote_auth == "key" and not server.remote_key_path:
            raise HTTPException(status_code=400, detail="Remote auth uses key but no key path configured")
        if remote_auth == "password" and not server.remote_password:
            raise HTTPException(status_code=400, detail="Remote auth uses password but no password configured")
    else:
        auth_type = server.ssh_auth_type or "password"
        if auth_type == "key" and not server.ssh_key_path:
            raise HTTPException(status_code=400, detail="SSH key authentication selected but no key uploaded")
        if auth_type == "password" and not server.ssh_password:
            raise HTTPException(
                status_code=400, detail="SSH password authentication selected but no password configured"
            )


@router.post("/{server_id}/test-connection")
async def test_connection(server_id: int, db: Session = Depends(get_db)):
    """Test connectivity to the monitoring server."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    if not server.server_ip:
        raise HTTPException(status_code=400, detail="Server IP not configured")

    results = {}

    if server.use_jump_host and server.jump_host:
        try:
            _validate_ssh_config(server)
        except HTTPException as e:
            return {
                "overall_status": "error",
                "error": e.detail,
                "services": {},
                "connection_method": "ssh_tunnel",
            }

        ssh = _create_ssh_manager(server)

        try:
            async with ssh:
                result = await ssh.execute(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 http://localhost:{server.prometheus_port}/-/healthy",
                    timeout=15,
                )
                if result.success and result.stdout.strip() == "200":
                    results["prometheus"] = {
                        "status": "healthy",
                        "url": f"http://{server.server_ip}:{server.prometheus_port}",
                    }
                else:
                    results["prometheus"] = {"status": "unreachable", "error": result.stderr or "Connection failed"}

                result = await ssh.execute(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 http://localhost:{server.loki_port}/ready",
                    timeout=15,
                )
                if result.success and result.stdout.strip() == "200":
                    results["loki"] = {"status": "healthy", "url": f"http://{server.server_ip}:{server.loki_port}"}
                else:
                    results["loki"] = {"status": "unreachable", "error": result.stderr or "Connection failed"}

                result = await ssh.execute(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 http://localhost:{server.grafana_port}/api/health",
                    timeout=15,
                )
                if result.success and result.stdout.strip() == "200":
                    results["grafana"] = {
                        "status": "healthy",
                        "url": f"http://{server.server_ip}:{server.grafana_port}",
                    }
                else:
                    results["grafana"] = {"status": "unreachable", "error": result.stderr or "Connection failed"}

        except Exception as e:
            logger.exception(f"SSH connection failed: {e}")
            return {
                "overall_status": "error",
                "error": f"SSH connection failed: {str(e)}",
                "services": {},
                "connection_method": "ssh_tunnel",
            }

        all_healthy = all(r.get("status") == "healthy" for r in results.values())
        return {
            "overall_status": "healthy" if all_healthy else "degraded",
            "services": results,
            "connection_method": "ssh_tunnel",
        }

    # Direct HTTP connectivity test
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{server.server_ip}:{server.prometheus_port}/-/healthy")
            results["prometheus"] = {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "url": f"http://{server.server_ip}:{server.prometheus_port}",
            }
    except Exception as e:
        results["prometheus"] = {"status": "unreachable", "error": str(e)}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{server.server_ip}:{server.loki_port}/ready")
            results["loki"] = {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "url": f"http://{server.server_ip}:{server.loki_port}",
            }
    except Exception as e:
        results["loki"] = {"status": "unreachable", "error": str(e)}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{server.server_ip}:{server.grafana_port}/api/health")
            results["grafana"] = {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "url": f"http://{server.server_ip}:{server.grafana_port}",
            }
    except Exception as e:
        results["grafana"] = {"status": "unreachable", "error": str(e)}

    all_healthy = all(r.get("status") == "healthy" for r in results.values())

    return {
        "overall_status": "healthy" if all_healthy else "degraded",
        "services": results,
        "connection_method": "direct",
    }


@router.post("/{server_id}/check-services")
async def check_services(server_id: int, db: Session = Depends(get_db)):
    """Check what services are installed on the monitoring server."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    _validate_ssh_config(server)

    ssh = _create_ssh_manager(server)
    services = {}

    try:
        async with ssh:
            result = await ssh.execute("which docker && docker --version")
            services["docker"] = {
                "installed": result.success,
                "version": result.stdout.strip().split('\n')[-1] if result.success else None,
            }

            result = await ssh.execute(
                "sg docker -c 'docker compose version' 2>/dev/null || docker compose version 2>/dev/null || echo 'not found'"
            )
            has_compose = result.success and "not found" not in result.stdout
            services["docker_compose"] = {
                "installed": has_compose,
                "version": result.stdout.strip() if has_compose else None,
            }

            result = await ssh.execute(
                "sg docker -c 'docker ps' >/dev/null 2>&1 && echo 'ok' || (docker ps >/dev/null 2>&1 && echo 'ok' || echo 'no_access')"
            )
            docker_accessible = result.success and "ok" in result.stdout
            services["docker_accessible"] = {
                "accessible": docker_accessible,
                "message": "User can run docker commands" if docker_accessible else "User not in docker group",
            }

            docker_cmd = "sg docker -c"

            result = await ssh.execute(
                f"{docker_cmd} \"docker ps --filter 'name=prometheus' --format '{{{{.Names}}}} {{{{.Status}}}}'\""
            )
            services["prometheus"] = {
                "running": "prometheus" in result.stdout.lower() if result.success else False,
                "status": result.stdout.strip() if result.success and result.stdout.strip() else "not running",
            }

            result = await ssh.execute(
                f"{docker_cmd} \"docker ps --filter 'name=loki' --format '{{{{.Names}}}} {{{{.Status}}}}'\""
            )
            services["loki"] = {
                "running": "loki" in result.stdout.lower() if result.success else False,
                "status": result.stdout.strip() if result.success and result.stdout.strip() else "not running",
            }

            result = await ssh.execute(
                f"{docker_cmd} \"docker ps --filter 'name=grafana' --format '{{{{.Names}}}} {{{{.Status}}}}'\""
            )
            services["grafana"] = {
                "running": "grafana" in result.stdout.lower() if result.success else False,
                "status": result.stdout.strip() if result.success and result.stdout.strip() else "not running",
            }

            result = await ssh.execute("df -h / | tail -1 | awk '{print $4}'")
            services["disk_space"] = {
                "available": result.stdout.strip() if result.success else "unknown",
            }

        return {
            "server": server.server_ip,
            "ssh_connected": True,
            "services": services,
            "ready_for_install": (
                services["docker"]["installed"] and services["docker_compose"]["installed"] and docker_accessible
            ),
        }

    except Exception as e:
        logger.exception(f"Failed to check services: {e}")
        return {
            "server": server.server_ip,
            "ssh_connected": False,
            "error": str(e),
            "services": {},
            "ready_for_install": False,
        }


@router.post("/{server_id}/install-stack")
async def install_stack(
    server_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Install Prometheus, Grafana, and Loki on the monitoring server."""
    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    _validate_ssh_config(server)

    job_id = str(uuid4())

    _installation_jobs[job_id] = {
        "status": "starting",
        "started_at": datetime.utcnow().isoformat(),
        "server": server.server_ip,
        "logs": [],
        "current_step": "Initializing...",
        "completed": False,
        "error": None,
    }

    background_tasks.add_task(
        run_stack_installation,
        job_id=job_id,
        server_id=server.id,
    )

    return {
        "job_id": job_id,
        "message": "Monitoring stack installation started",
        "server": server.server_ip,
    }


@router.get("/{server_id}/install-status/{job_id}")
async def get_install_status(server_id: int, job_id: str):
    """Get the status of an installation job."""
    if job_id not in _installation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _installation_jobs[job_id]


@router.post("/{server_id}/sync-targets")
async def sync_targets(server_id: int, db: Session = Depends(get_db)):
    """Sync Prometheus targets for node groups using this monitoring server."""
    from ...models.database import Node, NodeGroup, NodeStatus

    server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Monitoring server not found")

    if not server.server_ip:
        raise HTTPException(status_code=400, detail="Server IP not configured")

    if not server.setup_monitoring_stack:
        raise HTTPException(status_code=400, detail="Remote monitoring setup not enabled")

    # Get all active nodes from node groups using this monitoring server
    nodes = (
        db.query(Node)
        .join(NodeGroup)
        .filter(NodeGroup.monitoring_server_id == server_id)
        .filter(Node.status == NodeStatus.ACTIVE.value)
        .all()
    )

    if not nodes:
        return {"message": "No active nodes to sync", "targets_count": 0}

    import json

    # Group nodes by node_group for better organization
    node_groups_map = {}
    for node in nodes:
        node_group = db.query(NodeGroup).filter(NodeGroup.id == node.node_group_id).first()
        ng_name = node_group.name if node_group else "unknown"
        if ng_name not in node_groups_map:
            node_groups_map[ng_name] = []
        node_groups_map[ng_name].append(node)

    _validate_ssh_config(server)
    ssh = _create_ssh_manager(server)

    try:
        async with ssh:
            install_dir = f"/home/{server.ssh_user}/fleet-monitoring"
            targets_dir = f"{install_dir}/prometheus/targets"

            # Ensure targets directory exists
            await ssh.execute(f"mkdir -p {targets_dir}")

            total_targets = 0

            # Create separate target files for each node group
            for ng_name, ng_nodes in node_groups_map.items():
                # GPU exporter targets - job label matches prometheus.yml job_name
                gpu_targets = []
                for node in ng_nodes:
                    gpu_targets.append(
                        {
                            "targets": [f"{node.ip_address}:{node.gpu_exporter_port}"],
                            "labels": {
                                "job": "amd_gpu_metrics",
                                "node_group": ng_name,
                                "hostname": node.hostname or node.ip_address,
                                "gpu_model": node.gpu_model or "N/A",
                            },
                        }
                    )

                # Node exporter targets
                node_targets = []
                for node in ng_nodes:
                    node_targets.append(
                        {
                            "targets": [f"{node.ip_address}:{node.node_exporter_port}"],
                            "labels": {
                                "job": "node_exporter",
                                "node_group": ng_name,
                                "hostname": node.hostname or node.ip_address,
                            },
                        }
                    )

                # RDMA exporter targets (port 9417)
                rdma_targets = []
                for node in ng_nodes:
                    rdma_targets.append(
                        {
                            "targets": [f"{node.ip_address}:9417"],
                            "labels": {
                                "job": "rdma_metrics",
                                "node_group": ng_name,
                                "hostname": node.hostname or node.ip_address,
                            },
                        }
                    )

                # Upload GPU targets - use heredoc to avoid shell escaping issues
                gpu_json = json.dumps(gpu_targets, indent=2)
                gpu_b64 = base64.b64encode(gpu_json.encode()).decode()
                # Write base64 to temp file, then decode (avoids shell escaping issues)
                result = await ssh.execute(
                    f"cat << 'EOFB64' | base64 -d > {targets_dir}/gpu_{ng_name}.json\n{gpu_b64}\nEOFB64"
                )
                if not result.success:
                    logger.error(f"Failed to upload GPU targets for {ng_name}: {result.stderr}")

                # Upload Node exporter targets - use heredoc to avoid shell escaping issues
                node_json = json.dumps(node_targets, indent=2)
                node_b64 = base64.b64encode(node_json.encode()).decode()
                result = await ssh.execute(
                    f"cat << 'EOFB64' | base64 -d > {targets_dir}/node_{ng_name}.json\n{node_b64}\nEOFB64"
                )
                if not result.success:
                    logger.error(f"Failed to upload node targets for {ng_name}: {result.stderr}")

                # Upload RDMA exporter targets
                rdma_json = json.dumps(rdma_targets, indent=2)
                rdma_b64 = base64.b64encode(rdma_json.encode()).decode()
                result = await ssh.execute(
                    f"cat << 'EOFB64' | base64 -d > {targets_dir}/rdma_{ng_name}.json\n{rdma_b64}\nEOFB64"
                )
                if not result.success:
                    logger.error(f"Failed to upload RDMA targets for {ng_name}: {result.stderr}")

                total_targets += len(ng_nodes)
                logger.info(f"Synced {len(ng_nodes)} nodes for group '{ng_name}'")

            # Reload Prometheus configuration
            prom_container = _get_container_prefix(server) + "-prometheus"
            reload_result = await ssh.execute(f"docker exec {prom_container} kill -HUP 1 2>/dev/null || true")
            logger.info(f"Prometheus reload: {reload_result.stdout}")

            return {
                "message": f"Synced {total_targets} active nodes ({len(node_groups_map)} node groups) to Prometheus",
                "targets_count": total_targets,
                "node_groups": list(node_groups_map.keys()),
                "targets_dir": targets_dir,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to sync targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _add_install_log(job_id: str, message: str, level: str = "info"):
    """Add a log entry to the installation job."""
    if job_id in _installation_jobs:
        _installation_jobs[job_id]["logs"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
            }
        )
        if level == "error":
            logger.error(f"[{job_id}] {message}")
        else:
            logger.info(f"[{job_id}] {message}")


def _update_install_step(job_id: str, step: str):
    """Update the current step of the installation."""
    if job_id in _installation_jobs:
        _installation_jobs[job_id]["current_step"] = step
        _add_install_log(job_id, step)


async def run_stack_installation(job_id: str, server_id: int):
    """Background task to install the monitoring stack."""
    from ...models.database import SessionLocal, MonitoringServer

    db = SessionLocal()
    try:
        server = db.query(MonitoringServer).filter(MonitoringServer.id == server_id).first()
        if not server:
            _add_install_log(job_id, f"Monitoring server {server_id} not found", "error")
            _installation_jobs[job_id]["status"] = "failed"
            _installation_jobs[job_id]["error"] = "Server not found"
            _installation_jobs[job_id]["completed"] = True
            return

        _installation_jobs[job_id]["status"] = "running"
        _update_install_step(job_id, f"Connecting to {server.server_ip}...")

        ssh = _create_ssh_manager(server)

        async with ssh:
            _add_install_log(job_id, "SSH connection established")

            # Check if Docker is installed
            _update_install_step(job_id, "Checking if Docker is installed...")
            result = await ssh.execute("which docker")
            docker_installed = result.success and result.stdout.strip()

            if not docker_installed:
                _add_install_log(job_id, "Docker not found, installing Docker...")
                _update_install_step(job_id, "Installing Docker...")

                result = await ssh.execute("curl -fsSL https://get.docker.com -o /tmp/get-docker.sh", timeout=60)
                if not result.success:
                    _add_install_log(job_id, f"Failed to download Docker install script: {result.stderr}", "error")
                    _installation_jobs[job_id]["status"] = "failed"
                    _installation_jobs[job_id]["error"] = "Failed to download Docker install script"
                    _installation_jobs[job_id]["completed"] = True
                    return

                result = await ssh.execute("sudo sh /tmp/get-docker.sh", timeout=300)
                if not result.success:
                    _add_install_log(job_id, f"Docker installation failed: {result.stderr}", "error")
                    _installation_jobs[job_id]["status"] = "failed"
                    _installation_jobs[job_id]["error"] = "Docker installation failed"
                    _installation_jobs[job_id]["completed"] = True
                    return

                _add_install_log(job_id, "Docker installed successfully")
                await ssh.execute("sudo systemctl start docker")
                await ssh.execute("sudo systemctl enable docker")

            # Check docker access - rely on exit code only; stdout capture is unreliable
            # over non-interactive SSH sessions for compound shell commands.
            _update_install_step(job_id, "Checking Docker access...")
            result = await ssh.execute("docker ps > /dev/null 2>&1")
            docker_accessible = result.success

            if not docker_accessible:
                # Ensure the docker group exists.
                result = await ssh.execute("getent group docker")
                if not result.success:
                    await ssh.execute("sudo -n groupadd docker 2>/dev/null || true")
                    if server.ssh_password:
                        await ssh.execute_with_input(
                            "sudo -S groupadd docker 2>/dev/null || true",
                            server.ssh_password + "\n",
                        )

                # Try to add the user to the docker group.
                # Attempt 1: passwordless sudo (-n exits immediately if password needed).
                _update_install_step(job_id, "Adding user to docker group...")
                result = await ssh.execute(f"sudo -n usermod -aG docker {server.ssh_user}")

                if not result.success and server.ssh_password:
                    # Attempt 2: pipe the SSH password to sudo -S (works when the
                    # SSH login password is the same as the sudo password, which is
                    # the common case on standard Linux installs).
                    _add_install_log(job_id, "Passwordless sudo unavailable, retrying with SSH password...")
                    result = await ssh.execute_with_input(
                        f"sudo -S usermod -aG docker {server.ssh_user}",
                        server.ssh_password + "\n",
                    )

                if result.success:
                    _add_install_log(job_id, f"User '{server.ssh_user}' added to docker group")
                else:
                    _add_install_log(
                        job_id,
                        f"Could not add user to docker group: {result.stderr.strip()}. "
                        "Will attempt installation using 'sudo docker' as a fallback.",
                        "warning",
                    )

            # Create installation directory
            _update_install_step(job_id, "Creating installation directory...")
            install_dir = "~/fleet-monitoring"
            result = await ssh.execute(f"mkdir -p {install_dir}")
            if not result.success:
                _add_install_log(job_id, f"Failed to create installation directory: {result.stderr}", "error")
                _installation_jobs[job_id]["status"] = "failed"
                _installation_jobs[job_id]["error"] = "Failed to create installation directory"
                _installation_jobs[job_id]["completed"] = True
                return

            result = await ssh.execute(f"cd {install_dir} && pwd")
            if result.success:
                install_dir = result.stdout.strip()
                _add_install_log(job_id, f"Installation directory: {install_dir}")

            # Create docker-compose.yml - use heredoc to avoid shell escaping issues
            _update_install_step(job_id, "Creating docker-compose.yml...")
            compose_content = _get_monitoring_compose(server)
            compose_b64 = base64.b64encode(compose_content.encode()).decode()
            result = await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/docker-compose.yml\n{compose_b64}\nEOFB64"
            )
            if not result.success:
                _add_install_log(job_id, f"Failed to create docker-compose.yml: {result.stderr}", "error")
                _installation_jobs[job_id]["status"] = "failed"
                _installation_jobs[job_id]["error"] = "Failed to create docker-compose.yml"
                _installation_jobs[job_id]["completed"] = True
                return

            # Create Prometheus config - use heredoc to avoid shell escaping issues
            _update_install_step(job_id, "Creating Prometheus configuration...")
            prom_config = _get_prometheus_config(server)
            prom_b64 = base64.b64encode(prom_config.encode()).decode()
            await ssh.execute(f"mkdir -p {install_dir}/prometheus {install_dir}/prometheus/targets")
            await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/prometheus/prometheus.yml\n{prom_b64}\nEOFB64"
            )

            # Create Loki config - use heredoc to avoid shell escaping issues
            _update_install_step(job_id, "Creating Loki configuration...")
            loki_config = _get_loki_config(server)
            loki_b64 = base64.b64encode(loki_config.encode()).decode()
            await ssh.execute(f"mkdir -p {install_dir}/loki")
            await ssh.execute(f"cat << 'EOFB64' | base64 -d > {install_dir}/loki/loki-config.yml\n{loki_b64}\nEOFB64")

            # Create Grafana config - use heredoc to avoid shell escaping issues
            _update_install_step(job_id, "Creating Grafana configuration...")
            await ssh.execute(f"mkdir -p {install_dir}/grafana/provisioning/datasources")
            await ssh.execute(f"mkdir -p {install_dir}/grafana/provisioning/dashboards")
            await ssh.execute(f"mkdir -p {install_dir}/grafana/dashboards")

            datasource_config = _get_grafana_datasources(server)
            ds_b64 = base64.b64encode(datasource_config.encode()).decode()
            await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/grafana/provisioning/datasources/datasources.yml\n{ds_b64}\nEOFB64"
            )

            dashboard_provisioning = _get_dashboard_provisioning()
            dp_b64 = base64.b64encode(dashboard_provisioning.encode()).decode()
            await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/grafana/provisioning/dashboards/dashboards.yml\n{dp_b64}\nEOFB64"
            )

            # Clean up old dashboard files to avoid UID conflicts
            _update_install_step(job_id, "Cleaning up old dashboards...")
            await ssh.execute(f"rm -f {install_dir}/grafana/dashboards/*.json 2>/dev/null || true")
            _add_install_log(job_id, "Cleaned up old dashboard files")

            # Upload all dashboards
            dashboards = _get_all_dashboards()
            for dash_name, dash_json in dashboards.items():
                dash_b64 = base64.b64encode(dash_json.encode()).decode()
                await ssh.execute(
                    f"cat << 'EOFB64' | base64 -d > {install_dir}/grafana/dashboards/{dash_name}.json\n{dash_b64}\nEOFB64"
                )
            _add_install_log(job_id, f"Uploaded {len(dashboards)} Grafana dashboards")

            # Create data directories
            _update_install_step(job_id, "Setting up data directories...")
            prometheus_data_path = server.prometheus_storage_path or f"{install_dir}/data/prometheus"
            await ssh.execute(f"mkdir -p {prometheus_data_path} {install_dir}/data/loki {install_dir}/data/grafana")
            await ssh.execute(
                f"sudo chown -R 65534:65534 {prometheus_data_path} 2>/dev/null || chmod -R 777 {prometheus_data_path}"
            )
            await ssh.execute(
                f"sudo chown -R 10001:10001 {install_dir}/data/loki 2>/dev/null || chmod -R 777 {install_dir}/data/loki"
            )
            await ssh.execute(
                f"sudo chown -R 472:472 {install_dir}/data/grafana 2>/dev/null || chmod -R 777 {install_dir}/data/grafana"
            )

            # Choose the docker invocation prefix for this session:
            #   - direct:        user was already in docker group before install
            #   - sg docker -c:  user was just added to the group via usermod
            #   - sudo docker:   usermod failed; fall back to sudo
            if docker_accessible:
                docker_cmd = ""
            elif result.success:  # usermod succeeded
                docker_cmd = "sg docker -c"
            elif server.ssh_password:
                # usermod failed but we have a password — use sudo -S docker
                docker_cmd = f"echo {server.ssh_password!r} | sudo -S"
            else:
                docker_cmd = "sudo -n"

            # Pull images - allow up to 10 minutes for large images on slow links
            _update_install_step(job_id, "Pulling Docker images (this may take several minutes)...")
            if docker_cmd:
                result = await ssh.execute(f"cd {install_dir} && {docker_cmd} 'docker compose pull'", timeout=600)
            else:
                result = await ssh.execute(f"cd {install_dir} && docker compose pull", timeout=600)
            if not result.success:
                _add_install_log(job_id, f"Image pull warning (will retry on up): {result.stderr}", "warning")
            else:
                _add_install_log(job_id, "Docker images pulled successfully")

            # Stop and remove any containers left over from a previous install
            # attempt before starting fresh. This is idempotent — safe to run
            # even when nothing is running.
            _update_install_step(job_id, "Stopping any existing stack containers...")
            if docker_cmd:
                await ssh.execute(
                    f"cd {install_dir} && {docker_cmd} 'docker compose down --remove-orphans'", timeout=60
                )
            else:
                await ssh.execute(f"cd {install_dir} && docker compose down --remove-orphans", timeout=60)
            _add_install_log(job_id, "Existing containers removed (if any)")

            # Start the stack - allow up to 5 minutes; compose up -d pulls any
            # missing images and waits for containers to start.
            _update_install_step(job_id, "Starting monitoring stack...")
            if docker_cmd:
                result = await ssh.execute(f"cd {install_dir} && {docker_cmd} 'docker compose up -d'", timeout=300)
            else:
                result = await ssh.execute(f"cd {install_dir} && docker compose up -d", timeout=300)
            if not result.success:
                _add_install_log(job_id, f"Failed to start monitoring stack: {result.stderr}", "error")
                _installation_jobs[job_id]["status"] = "failed"
                _installation_jobs[job_id]["error"] = "Failed to start Docker containers"
                _installation_jobs[job_id]["completed"] = True
                return

            _add_install_log(job_id, "Docker containers started")

            # Wait and verify
            _update_install_step(job_id, "Verifying services...")
            await asyncio.sleep(15)

            if docker_cmd:
                result = await ssh.execute(
                    f"{docker_cmd} \"docker ps --format '{{{{.Names}}}}: {{{{.Status}}}}'\"", timeout=30
                )
            else:
                result = await ssh.execute("docker ps --format '{{.Names}}: {{.Status}}'", timeout=30)

            _add_install_log(job_id, f"Running containers:\n{result.stdout}")

            # Update server status
            server.stack_installed = True
            server.last_install_at = datetime.utcnow()
            db.commit()

            _installation_jobs[job_id]["status"] = "completed"
            _installation_jobs[job_id]["completed"] = True
            _installation_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
            _update_install_step(job_id, "Installation completed successfully!")

    except Exception as e:
        _add_install_log(job_id, f"Installation failed: {str(e)}", "error")
        _installation_jobs[job_id]["status"] = "failed"
        _installation_jobs[job_id]["error"] = str(e)
        _installation_jobs[job_id]["completed"] = True
        logger.exception(f"Failed to install monitoring stack: {e}")
    finally:
        db.close()


def _get_container_prefix(server) -> str:
    """Return a safe container name prefix derived from the monitoring server name.

    Using a per-server prefix ensures container names are unique even when the
    monitoring stack is deployed on the same host as the fleet-manager management
    stack (which uses the 'fleet-' prefix internally).
    """
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in server.name.lower()).strip("-")
    return safe or f"mon-{server.id}"


def _get_monitoring_compose(server) -> str:
    """Generate docker-compose.yml for monitoring stack.

    All services use network_mode: host for simplicity and to avoid Docker
    networking issues with GPU node metrics collection.
    """
    prometheus_data_path = server.prometheus_storage_path or "./data/prometheus"
    prefix = _get_container_prefix(server)
    return f"""services:
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: {prefix}-prometheus
    restart: unless-stopped
    network_mode: host
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time={server.prometheus_retention_time}'
      - '--storage.tsdb.retention.size={server.prometheus_retention_size}'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--web.listen-address=:{server.prometheus_port}'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/targets:/etc/prometheus/targets:ro
      - {prometheus_data_path}:/prometheus

  loki:
    image: grafana/loki:2.9.5
    container_name: {prefix}-loki
    restart: unless-stopped
    network_mode: host
    command:
      - '-config.file=/etc/loki/loki-config.yml'
      - '-server.http-listen-port={server.loki_port}'
    volumes:
      - ./loki/loki-config.yml:/etc/loki/loki-config.yml:ro
      - ./data/loki:/loki

  grafana:
    image: grafana/grafana:10.4.1
    container_name: {prefix}-grafana
    restart: unless-stopped
    network_mode: host
    environment:
      GF_SECURITY_ADMIN_USER: {server.grafana_admin_user or 'admin'}
      GF_SECURITY_ADMIN_PASSWORD: {server.grafana_admin_password or 'admin'}
      GF_USERS_ALLOW_SIGN_UP: 'false'
      GF_INSTALL_PLUGINS: grafana-piechart-panel
      GF_SERVER_HTTP_PORT: {server.grafana_port}
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./data/grafana:/var/lib/grafana
"""


def _get_prometheus_config(server) -> str:
    """Generate prometheus.yml configuration.

    Note: scrape_timeout must be less than scrape_interval.
    We set scrape_timeout to 50% of scrape_interval for safety.
    """
    # Parse scrape_interval to calculate safe timeout
    interval = server.prometheus_scrape_interval or "30s"
    # Extract number and ensure timeout is less than interval
    interval_seconds = int(interval.rstrip('sm')) if interval.endswith('s') else int(interval.rstrip('sm')) * 60
    timeout_seconds = max(10, interval_seconds // 2)  # At least 10s, at most half of interval
    scrape_timeout = f"{timeout_seconds}s"

    return f"""global:
  scrape_interval: {interval}
  evaluation_interval: {interval}
  scrape_timeout: {scrape_timeout}

alerting:
  alertmanagers: []

rule_files: []

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:{server.prometheus_port}']

  - job_name: 'amd_gpu_metrics'
    scrape_interval: 30s
    scrape_timeout: 25s
    file_sd_configs:
      - files:
          - '/etc/prometheus/targets/gpu_*.json'
        refresh_interval: 30s
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - source_labels: [node_group]
        target_label: nodegroup
      - source_labels: [hostname]
        target_label: host

  - job_name: 'node_exporter'
    file_sd_configs:
      - files:
          - '/etc/prometheus/targets/node_*.json'
        refresh_interval: 30s
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - source_labels: [node_group]
        target_label: nodegroup
      - source_labels: [hostname]
        target_label: host

  - job_name: 'rdma_metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
    file_sd_configs:
      - files:
          - '/etc/prometheus/targets/rdma_*.json'
        refresh_interval: 30s
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - source_labels: [node_group]
        target_label: nodegroup
      - source_labels: [hostname]
        target_label: host
"""


def _get_loki_config(server) -> str:
    """Generate Loki configuration."""
    return f"""auth_enabled: false

server:
  http_listen_port: {server.loki_port}
  grpc_listen_port: 9096

common:
  instance_addr: 127.0.0.1
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

query_range:
  results_cache:
    cache:
      embedded_cache:
        enabled: true
        max_size_mb: 100

schema_config:
  configs:
    - from: 2020-10-24
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093

limits_config:
  retention_period: {server.loki_retention_days * 24}h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 24

compactor:
  working_directory: /loki/compactor
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
"""


def _get_grafana_datasources(server) -> str:
    """Generate Grafana datasources provisioning config.

    Note: Prometheus uses network_mode: host, so we connect via localhost.
    Loki uses the Docker network, so we connect via container name.
    """
    # Since Prometheus runs with network_mode: host, use localhost
    prometheus_url = f"http://localhost:{server.prometheus_port}"
    loki_url = f"http://localhost:{server.loki_port}"
    return f"""apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: {prometheus_url}
    isDefault: true
    editable: false
    uid: prometheus
    jsonData:
      timeInterval: "30s"
      queryTimeout: "60s"
      httpMethod: "POST"

  - name: Loki
    type: loki
    access: proxy
    url: {loki_url}
    editable: false
    uid: loki
    jsonData:
      maxLines: 1000
"""


def _get_dashboard_provisioning() -> str:
    """Generate Grafana dashboard provisioning config."""
    return """apiVersion: 1

providers:
  - name: 'GPU Fleet Dashboards'
    orgId: 1
    folder: 'GPU Fleet'
    folderUid: 'gpu-fleet'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""


def _get_all_dashboards() -> Dict[str, str]:
    """Load all dashboard JSON files from config directory.

    Ensures no duplicate UIDs are loaded to prevent Grafana provisioning issues.
    """
    import json

    dashboards = {}
    seen_uids = {}  # Track UIDs to prevent duplicates

    # Path to dashboard config files - check both Docker path and local dev path
    possible_paths = [
        '/app/config/grafana/dashboards',  # Docker container path
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'grafana', 'dashboards'),  # Local dev path
    ]

    config_dir = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            config_dir = abs_path
            break

    if not config_dir:
        logger.warning("Dashboard config directory not found")
        dashboards['fleet-overview'] = _get_fleet_overview_dashboard()
        return dashboards

    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(config_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Validate it's valid JSON and check for duplicate UIDs
                    dashboard_data = json.loads(content)
                    uid = dashboard_data.get('uid', filename.replace('.json', ''))

                    # Check for duplicate UID
                    if uid in seen_uids:
                        logger.warning(f"Skipping {filename} - duplicate UID '{uid}' (already in {seen_uids[uid]})")
                        continue

                    seen_uids[uid] = filename
                    dash_name = filename.replace('.json', '')
                    dashboards[dash_name] = content
                    logger.info(f"Loaded dashboard: {filename} (uid: {uid})")
            except Exception as e:
                logger.warning(f"Failed to load dashboard {filename}: {e}")

    # If no config files found, fall back to embedded dashboard
    if not dashboards:
        logger.info("No dashboard config files found, using embedded dashboard")
        dashboards['fleet-overview'] = _get_fleet_overview_dashboard()

    logger.info(f"Total dashboards loaded: {len(dashboards)}")
    return dashboards


def _get_fleet_overview_dashboard() -> str:
    """Generate GPU Fleet Overview dashboard JSON (fallback)."""
    import json

    dashboard = {
        "uid": "fleet-overview",
        "title": "GPU Fleet Overview",
        "tags": ["gpu", "fleet", "amd"],
        "timezone": "browser",
        "refresh": "30s",
        "schemaVersion": 39,
        "version": 1,
        "templating": {
            "list": [
                {
                    "name": "node_group",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "query": "label_values(amd_gpu_utilization_percent, node_group)",
                    "refresh": 2,
                    "includeAll": True,
                    "multi": True,
                    "current": {"selected": True, "text": "All", "value": "$__all"},
                },
                {
                    "name": "instance",
                    "type": "query",
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "query": "label_values(amd_gpu_utilization_percent{node_group=~\"$node_group\"}, instance)",
                    "refresh": 2,
                    "includeAll": True,
                    "multi": True,
                    "current": {"selected": True, "text": "All", "value": "$__all"},
                },
            ]
        },
        "panels": [
            {"id": 1, "title": "Fleet Summary", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}},
            {
                "id": 2,
                "title": "Total GPUs",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 4, "x": 0, "y": 1},
                "targets": [
                    {
                        "expr": "count(amd_gpu_utilization_percent{node_group=~\"$node_group\", instance=~\"$instance\"})",
                        "legendFormat": "GPUs",
                    }
                ],
                "fieldConfig": {
                    "defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}
                },
                "options": {"colorMode": "value", "graphMode": "none", "justifyMode": "center"},
            },
            {
                "id": 3,
                "title": "Active Nodes",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 4, "x": 4, "y": 1},
                "targets": [
                    {
                        "expr": "count(count by (instance) (amd_gpu_utilization_percent{node_group=~\"$node_group\", instance=~\"$instance\"}))",
                        "legendFormat": "Nodes",
                    }
                ],
                "fieldConfig": {
                    "defaults": {"thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}
                },
                "options": {"colorMode": "value", "graphMode": "none", "justifyMode": "center"},
            },
            {
                "id": 4,
                "title": "Avg GPU Utilization",
                "type": "gauge",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 4, "x": 8, "y": 1},
                "targets": [
                    {
                        "expr": "avg(amd_gpu_utilization_percent{node_group=~\"$node_group\", instance=~\"$instance\"})",
                        "legendFormat": "Utilization",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 100,
                        "unit": "percent",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90},
                            ],
                        },
                    }
                },
            },
            {
                "id": 5,
                "title": "Avg Memory Used",
                "type": "gauge",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 4, "x": 12, "y": 1},
                "targets": [
                    {
                        "expr": "avg(amd_gpu_memory_utilization_percent{node_group=~\"$node_group\", instance=~\"$instance\"})",
                        "legendFormat": "Memory",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 100,
                        "unit": "percent",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90},
                            ],
                        },
                    }
                },
            },
            {
                "id": 6,
                "title": "Max Temperature",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 4, "x": 16, "y": 1},
                "targets": [
                    {
                        "expr": "max(amd_gpu_temperature_junction_celsius{node_group=~\"$node_group\", instance=~\"$instance\"})",
                        "legendFormat": "Max Temp",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "celsius",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "orange", "value": 80},
                                {"color": "red", "value": 90},
                            ],
                        },
                    }
                },
                "options": {"colorMode": "value", "graphMode": "area"},
            },
            {
                "id": 7,
                "title": "Total Power",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 4, "x": 20, "y": 1},
                "targets": [
                    {
                        "expr": "sum(amd_gpu_power_watts{node_group=~\"$node_group\", instance=~\"$instance\"})",
                        "legendFormat": "Power",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "watt",
                        "thresholds": {"mode": "absolute", "steps": [{"color": "blue", "value": None}]},
                    }
                },
                "options": {"colorMode": "value", "graphMode": "area"},
            },
            {"id": 10, "title": "GPU Utilization", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5}},
            {
                "id": 11,
                "title": "GPU Utilization Over Time",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 6},
                "targets": [
                    {
                        "expr": "amd_gpu_utilization_percent{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} - GPU{{gpu_id}}",
                    }
                ],
                "fieldConfig": {
                    "defaults": {"unit": "percent", "min": 0, "max": 100, "custom": {"fillOpacity": 10, "lineWidth": 1}}
                },
                "options": {"legend": {"displayMode": "table", "placement": "right"}},
            },
            {
                "id": 12,
                "title": "Memory Utilization Over Time",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 6},
                "targets": [
                    {
                        "expr": "amd_gpu_memory_utilization_percent{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} - GPU{{gpu_id}}",
                    }
                ],
                "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "custom": {"fillOpacity": 10}}},
                "options": {"legend": {"displayMode": "table", "placement": "right"}},
            },
            {"id": 20, "title": "Thermal & Power", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 14}},
            {
                "id": 21,
                "title": "GPU Temperature",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 15},
                "targets": [
                    {
                        "expr": "amd_gpu_temperature_junction_celsius{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}}",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "celsius",
                        "custom": {"fillOpacity": 10},
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 90},
                            ],
                        },
                    }
                },
            },
            {
                "id": 22,
                "title": "GPU Power Usage",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 15},
                "targets": [
                    {
                        "expr": "amd_gpu_power_watts{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}}",
                    }
                ],
                "fieldConfig": {"defaults": {"unit": "watt", "custom": {"fillOpacity": 10}}},
            },
            {
                "id": 23,
                "title": "Fan Speed",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 8, "x": 16, "y": 15},
                "targets": [
                    {
                        "expr": "amd_gpu_fan_speed_percent{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}}",
                    }
                ],
                "fieldConfig": {"defaults": {"unit": "percent", "min": 0, "max": 100, "custom": {"fillOpacity": 10}}},
            },
        ],
    }
    return json.dumps(dashboard, indent=2)
