"""API routes for monitoring configuration."""

import logging
import os
from typing import Dict
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from ...models import get_db, MonitoringConfig
from ..schemas import (
    MonitoringConfigCreate,
    MonitoringConfigResponse,
    MonitoringConfigUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring Configuration"])

SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "/app/ssh_keys")

# In-memory storage for installation jobs (in production, use Redis or DB)
_installation_jobs: Dict[str, Dict] = {}


def get_or_create_config(db: Session, auto_commit: bool = True) -> MonitoringConfig:
    """Get the monitoring config, creating default if not exists."""
    config = db.query(MonitoringConfig).first()
    if not config:
        config = MonitoringConfig()
        db.add(config)
        if auto_commit:
            db.commit()
            db.refresh(config)
        else:
            db.flush()  # Get the ID without committing
    return config


def config_to_response(config: MonitoringConfig) -> MonitoringConfigResponse:
    """Convert DB model to response schema."""
    return MonitoringConfigResponse(
        id=config.id,
        monitoring_server_ip=config.monitoring_server_ip,
        monitoring_server_hostname=config.monitoring_server_hostname,
        prometheus_port=config.prometheus_port,
        loki_port=config.loki_port,
        grafana_port=config.grafana_port,
        grafana_admin_user=config.grafana_admin_user or "admin",
        prometheus_retention_time=config.prometheus_retention_time,
        prometheus_retention_size=config.prometheus_retention_size,
        prometheus_scrape_interval=config.prometheus_scrape_interval,
        loki_retention_days=config.loki_retention_days,
        use_push_gateway=config.use_push_gateway,
        push_gateway_port=config.push_gateway_port,
        setup_monitoring_stack=config.setup_monitoring_stack,
        monitoring_ssh_user=config.monitoring_ssh_user,
        monitoring_ssh_port=config.monitoring_ssh_port,
        monitoring_ssh_auth_type=config.monitoring_ssh_auth_type or "password",
        has_monitoring_ssh_key=bool(config.monitoring_ssh_key_path),
        has_monitoring_ssh_password=bool(config.monitoring_ssh_password),
        # Jump host fields
        monitoring_use_jump_host=config.monitoring_use_jump_host or False,
        monitoring_jump_host=config.monitoring_jump_host,
        monitoring_jump_port=config.monitoring_jump_port or 22,
        monitoring_jump_user=config.monitoring_jump_user,
        monitoring_jump_auth_type=config.monitoring_jump_auth_type or "key",
        has_monitoring_jump_key=bool(config.monitoring_jump_key_path),
        has_monitoring_jump_password=bool(config.monitoring_jump_password),
        monitoring_remote_auth_type=config.monitoring_remote_auth_type or "key",
        monitoring_remote_key_path=config.monitoring_remote_key_path,
        has_monitoring_remote_password=bool(config.monitoring_remote_password),
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@router.get("/config", response_model=MonitoringConfigResponse)
def get_monitoring_config(db: Session = Depends(get_db)):
    """Get the current monitoring configuration."""
    config = get_or_create_config(db)
    return config_to_response(config)


@router.put("/config", response_model=MonitoringConfigResponse)
def update_monitoring_config(
    config_update: MonitoringConfigUpdate,
    db: Session = Depends(get_db),
):
    """Update monitoring configuration."""
    # Don't auto-commit when getting/creating, so we can update in the same transaction
    config = get_or_create_config(db, auto_commit=False)

    update_data = config_update.model_dump(exclude_unset=True)
    logger.info(f"Updating monitoring config fields: {list(update_data.keys())}")

    for field, value in update_data.items():
        if hasattr(config, field):
            setattr(config, field, value)
            logger.debug(f"Set {field} = {value}")

    db.commit()
    db.refresh(config)

    # Verify the update
    logger.info(f"Config after commit: monitoring_server_ip={config.monitoring_server_ip}")

    logger.info(f"Updated monitoring config: {update_data}")
    return config_to_response(config)


@router.post("/config", response_model=MonitoringConfigResponse)
def create_monitoring_config(
    config_create: MonitoringConfigCreate,
    db: Session = Depends(get_db),
):
    """Create or replace monitoring configuration."""
    # Delete existing config if any
    existing = db.query(MonitoringConfig).first()
    if existing:
        db.delete(existing)
        db.commit()

    config = MonitoringConfig(
        monitoring_server_ip=config_create.monitoring_server_ip,
        monitoring_server_hostname=config_create.monitoring_server_hostname,
        prometheus_port=config_create.prometheus_port,
        loki_port=config_create.loki_port,
        grafana_port=config_create.grafana_port,
        prometheus_retention_time=config_create.prometheus_retention_time,
        prometheus_retention_size=config_create.prometheus_retention_size,
        prometheus_scrape_interval=config_create.prometheus_scrape_interval,
        loki_retention_days=config_create.loki_retention_days,
        use_push_gateway=config_create.use_push_gateway,
        push_gateway_port=config_create.push_gateway_port,
        setup_monitoring_stack=config_create.setup_monitoring_stack,
        monitoring_ssh_user=config_create.monitoring_ssh_user,
        monitoring_ssh_port=config_create.monitoring_ssh_port,
        monitoring_ssh_auth_type=config_create.monitoring_ssh_auth_type,
        monitoring_ssh_password=config_create.monitoring_ssh_password,
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    logger.info(f"Created monitoring config with server: {config.monitoring_server_ip}")
    return config_to_response(config)


@router.delete("/config")
def delete_monitoring_config(db: Session = Depends(get_db)):
    """Delete/reset monitoring configuration."""
    config = db.query(MonitoringConfig).first()
    if config:
        db.delete(config)
        db.commit()
        logger.info("Deleted monitoring configuration")
        return {"message": "Monitoring configuration deleted"}
    return {"message": "No configuration to delete"}


@router.post("/config/ssh-key")
async def upload_monitoring_ssh_key(
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH key for accessing the monitoring server."""
    config = get_or_create_config(db)

    os.makedirs(SSH_KEY_PATH, exist_ok=True)
    key_path = os.path.join(SSH_KEY_PATH, "monitoring_server_key")

    content = await key_file.read()
    with open(key_path, "wb") as f:
        f.write(content)
    os.chmod(key_path, 0o600)

    config.monitoring_ssh_key_path = key_path
    db.commit()

    logger.info("Uploaded SSH key for monitoring server")
    return {"message": "SSH key uploaded successfully", "key_path": key_path}


@router.post("/config/jump-key")
async def upload_monitoring_jump_key(
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH key for accessing the jump host."""
    config = get_or_create_config(db)

    os.makedirs(SSH_KEY_PATH, exist_ok=True)
    key_path = os.path.join(SSH_KEY_PATH, "monitoring_jump_key")

    content = await key_file.read()
    with open(key_path, "wb") as f:
        f.write(content)
    os.chmod(key_path, 0o600)

    config.monitoring_jump_key_path = key_path
    db.commit()

    logger.info("Uploaded SSH key for monitoring jump host")
    return {"message": "Jump host SSH key uploaded successfully", "key_path": key_path}


@router.get("/endpoints")
def get_monitoring_endpoints(db: Session = Depends(get_db)):
    """Get the configured monitoring endpoints for display."""
    config = get_or_create_config(db)

    if not config.monitoring_server_ip:
        return {"configured": False, "message": "Monitoring server IP not configured", "endpoints": {}}

    server = config.monitoring_server_ip
    return {
        "configured": True,
        "server": server,
        "endpoints": {
            "prometheus": f"http://{server}:{config.prometheus_port}",
            "loki": f"http://{server}:{config.loki_port}",
            "grafana": f"http://{server}:{config.grafana_port}",
            "push_gateway": f"http://{server}:{config.push_gateway_port}" if config.use_push_gateway else None,
        },
        "retention": {
            "prometheus_time": config.prometheus_retention_time,
            "prometheus_size": config.prometheus_retention_size,
            "loki_days": config.loki_retention_days,
        },
        "scrape_interval": config.prometheus_scrape_interval,
    }


@router.post("/apply-config")
async def apply_monitoring_config(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Apply monitoring configuration changes to running services."""
    config = get_or_create_config(db)

    if not config.monitoring_server_ip:
        raise HTTPException(status_code=400, detail="Monitoring server IP must be configured first")

    # Update Prometheus config with retention settings

    # prometheus = PrometheusConfigManager()

    # The prometheus config is managed via docker-compose environment variables
    # For now, just return what would need to be updated
    return {
        "message": "Configuration noted. To apply retention changes, update docker-compose.yml and restart services.",
        "prometheus_retention_time": config.prometheus_retention_time,
        "prometheus_retention_size": config.prometheus_retention_size,
        "prometheus_scrape_interval": config.prometheus_scrape_interval,
        "note": "GPU node promtail configs will use the new monitoring server IP on next installation.",
    }


@router.post("/test-connection")
async def test_monitoring_connection(db: Session = Depends(get_db)):
    """Test connectivity to the configured monitoring server.

    If jump host is configured, uses SSH tunneling to test HTTP connectivity.
    Otherwise, tests direct HTTP connectivity.
    """
    config = get_or_create_config(db)

    if not config.monitoring_server_ip:
        raise HTTPException(status_code=400, detail="Monitoring server IP not configured")

    server = config.monitoring_server_ip
    results = {}

    # If jump host is configured, use SSH to test connectivity
    if config.monitoring_use_jump_host and config.monitoring_jump_host:
        try:
            _validate_monitoring_ssh_config(config)
        except HTTPException as e:
            return {
                "overall_status": "error",
                "error": e.detail,
                "services": {},
                "connection_method": "ssh_tunnel",
            }

        ssh = _create_monitoring_ssh_manager(config)

        try:
            async with ssh:
                # Test Prometheus via SSH using curl
                result = await ssh.execute(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 http://localhost:{config.prometheus_port}/-/healthy",
                    timeout=15,
                )
                if result.success and result.stdout.strip() == "200":
                    results["prometheus"] = {
                        "status": "healthy",
                        "url": f"http://{server}:{config.prometheus_port}",
                    }
                else:
                    results["prometheus"] = {
                        "status": "unreachable",
                        "error": result.stderr or f"HTTP {result.stdout.strip()}"
                        if result.stdout.strip()
                        else "Connection failed",
                    }

                # Test Loki via SSH
                result = await ssh.execute(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 http://localhost:{config.loki_port}/ready",
                    timeout=15,
                )
                if result.success and result.stdout.strip() == "200":
                    results["loki"] = {
                        "status": "healthy",
                        "url": f"http://{server}:{config.loki_port}",
                    }
                else:
                    results["loki"] = {
                        "status": "unreachable",
                        "error": result.stderr or f"HTTP {result.stdout.strip()}"
                        if result.stdout.strip()
                        else "Connection failed",
                    }

                # Test Grafana via SSH
                result = await ssh.execute(
                    f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 http://localhost:{config.grafana_port}/api/health",
                    timeout=15,
                )
                if result.success and result.stdout.strip() == "200":
                    results["grafana"] = {
                        "status": "healthy",
                        "url": f"http://{server}:{config.grafana_port}",
                    }
                else:
                    results["grafana"] = {
                        "status": "unreachable",
                        "error": result.stderr or f"HTTP {result.stdout.strip()}"
                        if result.stdout.strip()
                        else "Connection failed",
                    }

        except Exception as e:
            logger.exception(f"SSH connection failed for test-connection: {e}")
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

    # Direct HTTP connectivity test (no jump host)
    import httpx

    # Test Prometheus
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{server}:{config.prometheus_port}/-/healthy")
            results["prometheus"] = {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "url": f"http://{server}:{config.prometheus_port}",
            }
    except Exception as e:
        results["prometheus"] = {"status": "unreachable", "error": str(e)}

    # Test Loki
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{server}:{config.loki_port}/ready")
            results["loki"] = {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "url": f"http://{server}:{config.loki_port}",
            }
    except Exception as e:
        results["loki"] = {"status": "unreachable", "error": str(e)}

    # Test Grafana
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{server}:{config.grafana_port}/api/health")
            results["grafana"] = {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "url": f"http://{server}:{config.grafana_port}",
            }
    except Exception as e:
        results["grafana"] = {"status": "unreachable", "error": str(e)}

    all_healthy = all(r.get("status") == "healthy" for r in results.values())

    return {
        "overall_status": "healthy" if all_healthy else "degraded",
        "services": results,
        "connection_method": "direct",
    }


def _create_monitoring_ssh_manager(config: MonitoringConfig):
    """Create an SSHManager for the monitoring server with optional jump host support."""
    from ...services.ssh_manager import SSHManager, JumpHostConfig

    jump_host = None
    if config.monitoring_use_jump_host and config.monitoring_jump_host:
        jump_host = JumpHostConfig(
            host=config.monitoring_jump_host,
            port=config.monitoring_jump_port or 22,
            username=config.monitoring_jump_user or "root",
            auth_type=config.monitoring_jump_auth_type or "key",
            key_path=config.monitoring_jump_key_path,
            password=config.monitoring_jump_password,
            remote_auth_type=config.monitoring_remote_auth_type or "key",
            remote_key_path=config.monitoring_remote_key_path,
            remote_password=config.monitoring_remote_password,
        )

    auth_type = config.monitoring_ssh_auth_type or "password"

    return SSHManager(
        host=config.monitoring_server_ip,
        username=config.monitoring_ssh_user,
        port=config.monitoring_ssh_port or 22,
        auth_type=auth_type,
        key_path=config.monitoring_ssh_key_path if auth_type == "key" else None,
        password=config.monitoring_ssh_password if auth_type == "password" else None,
        jump_host=jump_host,
    )


def _validate_monitoring_ssh_config(config: MonitoringConfig):
    """Validate SSH configuration for monitoring server."""
    if not config.monitoring_server_ip:
        raise HTTPException(status_code=400, detail="Monitoring server IP not configured")

    if not config.setup_monitoring_stack:
        raise HTTPException(
            status_code=400,
            detail="Remote monitoring setup not enabled. Enable 'Setup Monitoring Stack' in configuration.",
        )

    if not config.monitoring_ssh_user:
        raise HTTPException(status_code=400, detail="SSH user for monitoring server not configured")

    # Check credentials based on whether using jump host
    if config.monitoring_use_jump_host:
        if not config.monitoring_jump_host:
            raise HTTPException(status_code=400, detail="Jump host enabled but no jump host address configured")
        if not config.monitoring_jump_user:
            raise HTTPException(status_code=400, detail="Jump host enabled but no jump host user configured")

        # Check jump host credentials
        jump_auth = config.monitoring_jump_auth_type or "key"
        if jump_auth == "key" and not config.monitoring_jump_key_path:
            raise HTTPException(status_code=400, detail="Jump host uses key auth but no key uploaded")
        if jump_auth == "password" and not config.monitoring_jump_password:
            raise HTTPException(status_code=400, detail="Jump host uses password auth but no password configured")

        # Check remote (monitoring server) credentials via jump host
        remote_auth = config.monitoring_remote_auth_type or "key"
        if remote_auth == "key" and not config.monitoring_remote_key_path:
            raise HTTPException(status_code=400, detail="Remote auth uses key but no key path on jump host configured")
        if remote_auth == "password" and not config.monitoring_remote_password:
            raise HTTPException(status_code=400, detail="Remote auth uses password but no password configured")
    else:
        # Direct connection - check credentials
        auth_type = config.monitoring_ssh_auth_type or "password"
        if auth_type == "key" and not config.monitoring_ssh_key_path:
            raise HTTPException(status_code=400, detail="SSH key authentication selected but no key uploaded")
        if auth_type == "password" and not config.monitoring_ssh_password:
            raise HTTPException(
                status_code=400, detail="SSH password authentication selected but no password configured"
            )


@router.post("/check-services")
async def check_monitoring_services(db: Session = Depends(get_db)):
    """Check what monitoring services are installed on the remote server via SSH."""
    config = get_or_create_config(db)

    _validate_monitoring_ssh_config(config)

    ssh = _create_monitoring_ssh_manager(config)

    services = {}

    try:
        async with ssh:
            # Check Docker (user should be in docker group)
            result = await ssh.execute("which docker && docker --version")
            services["docker"] = {
                "installed": result.success,
                "version": result.stdout.strip().split('\n')[-1] if result.success else None,
            }

            # Check Docker Compose
            result = await ssh.execute(
                "sg docker -c 'docker compose version' 2>/dev/null || docker compose version 2>/dev/null || echo 'not found'"
            )
            has_compose = result.success and "not found" not in result.stdout
            services["docker_compose"] = {
                "installed": has_compose,
                "version": result.stdout.strip() if has_compose else None,
            }

            # Check if user can access docker (is in docker group or via sg)
            result = await ssh.execute(
                "sg docker -c 'docker ps' >/dev/null 2>&1 && echo 'ok' || (docker ps >/dev/null 2>&1 && echo 'ok' || echo 'no_access')"
            )
            docker_accessible = result.success and "ok" in result.stdout
            services["docker_accessible"] = {
                "accessible": docker_accessible,
                "message": "User can run docker commands"
                if docker_accessible
                else "User not in docker group. Run: sudo usermod -aG docker $USER",
            }

            # Use sg docker -c for docker commands to handle group membership without re-login
            docker_cmd = "sg docker -c"

            # Check if Prometheus container is running
            result = await ssh.execute(
                f"{docker_cmd} \"docker ps --filter 'name=prometheus' --format '{{{{.Names}}}} {{{{.Status}}}}'\""
            )
            services["prometheus"] = {
                "running": "prometheus" in result.stdout.lower() if result.success else False,
                "status": result.stdout.strip() if result.success and result.stdout.strip() else "not running",
            }

            # Check if Loki container is running
            result = await ssh.execute(
                f"{docker_cmd} \"docker ps --filter 'name=loki' --format '{{{{.Names}}}} {{{{.Status}}}}'\""
            )
            services["loki"] = {
                "running": "loki" in result.stdout.lower() if result.success else False,
                "status": result.stdout.strip() if result.success and result.stdout.strip() else "not running",
            }

            # Check if Grafana container is running
            result = await ssh.execute(
                f"{docker_cmd} \"docker ps --filter 'name=grafana' --format '{{{{.Names}}}} {{{{.Status}}}}'\""
            )
            services["grafana"] = {
                "running": "grafana" in result.stdout.lower() if result.success else False,
                "status": result.stdout.strip() if result.success and result.stdout.strip() else "not running",
            }

            # Check available disk space
            result = await ssh.execute("df -h / | tail -1 | awk '{print $4}'")
            services["disk_space"] = {
                "available": result.stdout.strip() if result.success else "unknown",
            }

        return {
            "server": config.monitoring_server_ip,
            "ssh_connected": True,
            "services": services,
            "ready_for_install": (
                services["docker"]["installed"] and services["docker_compose"]["installed"] and docker_accessible
            ),
        }

    except Exception as e:
        logger.exception(f"Failed to check services on monitoring server: {e}")
        return {
            "server": config.monitoring_server_ip,
            "ssh_connected": False,
            "error": str(e),
            "services": {},
            "ready_for_install": False,
        }


@router.post("/install-stack")
async def install_monitoring_stack(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Install Prometheus, Grafana, and Loki on the remote monitoring server."""
    config = get_or_create_config(db)

    _validate_monitoring_ssh_config(config)

    from uuid import uuid4

    job_id = str(uuid4())

    # Initialize job tracking
    _installation_jobs[job_id] = {
        "status": "starting",
        "started_at": datetime.utcnow().isoformat(),
        "server": config.monitoring_server_ip,
        "logs": [],
        "current_step": "Initializing...",
        "completed": False,
        "error": None,
    }

    background_tasks.add_task(
        run_stack_installation,
        job_id=job_id,
        config_id=config.id,
    )

    return {
        "job_id": job_id,
        "message": "Monitoring stack installation started",
        "server": config.monitoring_server_ip,
    }


@router.get("/install-status/{job_id}")
async def get_installation_status(job_id: str):
    """Get the status of an installation job."""
    if job_id not in _installation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _installation_jobs[job_id]


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
        # Also log to the regular logger
        if level == "error":
            logger.error(f"[{job_id}] {message}")
        else:
            logger.info(f"[{job_id}] {message}")


def _update_install_step(job_id: str, step: str):
    """Update the current step of the installation."""
    if job_id in _installation_jobs:
        _installation_jobs[job_id]["current_step"] = step
        _add_install_log(job_id, step)


async def run_stack_installation(job_id: str, config_id: int):
    """Background task to install the monitoring stack."""
    from ...models.database import SessionLocal, MonitoringConfig

    db = SessionLocal()
    try:
        config = db.query(MonitoringConfig).filter(MonitoringConfig.id == config_id).first()
        if not config:
            _add_install_log(job_id, f"Monitoring config {config_id} not found", "error")
            _installation_jobs[job_id]["status"] = "failed"
            _installation_jobs[job_id]["error"] = "Configuration not found"
            _installation_jobs[job_id]["completed"] = True
            return

        _installation_jobs[job_id]["status"] = "running"
        _update_install_step(job_id, f"Connecting to {config.monitoring_server_ip}...")

        ssh = _create_monitoring_ssh_manager(config)

        async with ssh:
            _add_install_log(job_id, "SSH connection established")

            # Check if Docker is installed
            _update_install_step(job_id, "Checking if Docker is installed...")
            result = await ssh.execute("which docker")
            docker_installed = result.success and result.stdout.strip()

            if not docker_installed:
                _add_install_log(job_id, "Docker not found, installing Docker...")
                _update_install_step(job_id, "Installing Docker (this may take a few minutes)...")

                # Install Docker using the official convenience script
                # This works on most Linux distributions
                result = await ssh.execute("curl -fsSL https://get.docker.com -o /tmp/get-docker.sh", timeout=60)
                if not result.success:
                    _add_install_log(job_id, f"Failed to download Docker install script: {result.stderr}", "error")
                    _installation_jobs[job_id]["status"] = "failed"
                    _installation_jobs[job_id]["error"] = "Failed to download Docker install script"
                    _installation_jobs[job_id]["completed"] = True
                    return

                _add_install_log(job_id, "Running Docker installation script...")
                result = await ssh.execute("sudo sh /tmp/get-docker.sh", timeout=300)
                if not result.success:
                    _add_install_log(job_id, f"Docker installation failed: {result.stderr}", "error")
                    _installation_jobs[job_id]["status"] = "failed"
                    _installation_jobs[job_id]["error"] = "Docker installation failed"
                    _installation_jobs[job_id]["completed"] = True
                    return

                _add_install_log(job_id, "Docker installed successfully")

                # Start Docker service
                _add_install_log(job_id, "Starting Docker service...")
                await ssh.execute("sudo systemctl start docker")
                await ssh.execute("sudo systemctl enable docker")

            # Check if user can already access docker
            _update_install_step(job_id, "Checking Docker access...")
            result = await ssh.execute("docker ps >/dev/null 2>&1 && echo 'ok' || echo 'no_access'")
            docker_accessible = result.success and "ok" in result.stdout

            if not docker_accessible:
                # Check if docker group exists
                result = await ssh.execute("getent group docker")
                docker_group_exists = result.success

                if not docker_group_exists:
                    _add_install_log(job_id, "Creating docker group...")
                    await ssh.execute("sudo groupadd docker")

                # Try to add user to docker group
                _update_install_step(job_id, "Adding user to docker group...")
                _add_install_log(job_id, f"Adding {config.monitoring_ssh_user} to docker group...")
                result = await ssh.execute(f"sudo usermod -aG docker {config.monitoring_ssh_user}")
                if not result.success:
                    _add_install_log(job_id, f"Failed to add user to docker group: {result.stderr}", "error")
                    _add_install_log(
                        job_id, "Please run manually on the monitoring server: sudo usermod -aG docker $USER", "error"
                    )
                    _installation_jobs[job_id]["status"] = "failed"
                    _installation_jobs[job_id]["error"] = "Failed to add user to docker group"
                    _installation_jobs[job_id]["completed"] = True
                    return
                _add_install_log(job_id, f"Added {config.monitoring_ssh_user} to docker group")

            # Create installation directory in user's home (avoids sudo issues)
            _update_install_step(job_id, "Creating installation directory...")
            install_dir = "~/fleet-monitoring"
            result = await ssh.execute(f"mkdir -p {install_dir}")
            if not result.success:
                _add_install_log(job_id, f"Failed to create installation directory: {result.stderr}", "error")
                _installation_jobs[job_id]["status"] = "failed"
                _installation_jobs[job_id]["error"] = "Failed to create installation directory"
                _installation_jobs[job_id]["completed"] = True
                return

            # Get the absolute path for later use
            result = await ssh.execute(f"cd {install_dir} && pwd")
            if result.success:
                install_dir = result.stdout.strip()
                _add_install_log(job_id, f"Installation directory: {install_dir}")

            # Create docker-compose.yml for monitoring stack
            _update_install_step(job_id, "Creating docker-compose.yml...")
            # Use heredoc for base64 transfers to avoid shell escaping issues
            compose_content = _get_monitoring_compose(config)
            import base64

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
            prom_config = _get_prometheus_config(config)
            prom_b64 = base64.b64encode(prom_config.encode()).decode()
            await ssh.execute(f"mkdir -p {install_dir}/prometheus {install_dir}/prometheus/targets")
            result = await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/prometheus/prometheus.yml\n{prom_b64}\nEOFB64"
            )
            _add_install_log(job_id, "Prometheus configuration created")

            # Create Loki config - use heredoc to avoid shell escaping issues
            _update_install_step(job_id, "Creating Loki configuration...")
            loki_config = _get_loki_config(config)
            loki_b64 = base64.b64encode(loki_config.encode()).decode()
            await ssh.execute(f"mkdir -p {install_dir}/loki")
            result = await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/loki/loki-config.yml\n{loki_b64}\nEOFB64"
            )
            _add_install_log(job_id, "Loki configuration created")

            # Create Grafana provisioning directories
            _update_install_step(job_id, "Creating Grafana configuration...")
            await ssh.execute(f"mkdir -p {install_dir}/grafana/provisioning/datasources")
            await ssh.execute(f"mkdir -p {install_dir}/grafana/provisioning/dashboards")
            await ssh.execute(f"mkdir -p {install_dir}/grafana/dashboards")

            # Create Grafana datasource config - use heredoc to avoid shell escaping issues
            datasource_config = _get_grafana_datasources(config)
            ds_b64 = base64.b64encode(datasource_config.encode()).decode()
            result = await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/grafana/provisioning/datasources/datasources.yml\n{ds_b64}\nEOFB64"
            )

            # Create Grafana dashboard provisioning config - use heredoc to avoid shell escaping issues
            dashboard_provisioning = _get_dashboard_provisioning()
            dp_b64 = base64.b64encode(dashboard_provisioning.encode()).decode()
            result = await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/grafana/provisioning/dashboards/dashboards.yml\n{dp_b64}\nEOFB64"
            )

            # Upload all dashboards - use heredoc to avoid shell escaping issues
            dashboards = _get_all_dashboards()
            for dash_name, dash_json in dashboards.items():
                dash_b64 = base64.b64encode(dash_json.encode()).decode()
                await ssh.execute(
                    f"cat << 'EOFB64' | base64 -d > {install_dir}/grafana/dashboards/{dash_name}.json\n{dash_b64}\nEOFB64"
                )
            _add_install_log(job_id, f"Grafana configuration and {len(dashboards)} dashboards created")

            # Create data directories with proper permissions
            _update_install_step(job_id, "Setting up data directories...")
            await ssh.execute(
                f"mkdir -p {install_dir}/data/prometheus {install_dir}/data/loki {install_dir}/data/grafana"
            )
            # Try to set permissions with sudo, but don't fail if sudo isn't available
            # Docker containers will run as root and can write to these directories
            result = await ssh.execute(
                f"sudo chown -R 65534:65534 {install_dir}/data/prometheus 2>/dev/null || chmod -R 777 {install_dir}/data/prometheus"
            )
            result = await ssh.execute(
                f"sudo chown -R 10001:10001 {install_dir}/data/loki 2>/dev/null || chmod -R 777 {install_dir}/data/loki"
            )
            result = await ssh.execute(
                f"sudo chown -R 472:472 {install_dir}/data/grafana 2>/dev/null || chmod -R 777 {install_dir}/data/grafana"
            )
            _add_install_log(job_id, "Data directories created")

            # Use sg docker -c to run docker commands in docker group context
            # This works even if user was just added to the group (no logout needed)
            docker_cmd = "sg docker -c" if not docker_accessible else ""

            # Pull images first
            _update_install_step(job_id, "Pulling Docker images (this may take a few minutes)...")
            _add_install_log(job_id, "Pulling Docker images: prom/prometheus, grafana/loki, grafana/grafana")
            if docker_cmd:
                result = await ssh.execute(f"cd {install_dir} && {docker_cmd} 'docker compose pull'", timeout=300)
            else:
                result = await ssh.execute(f"cd {install_dir} && docker compose pull", timeout=300)
            if not result.success:
                _add_install_log(job_id, f"Image pull warning: {result.stderr}", "warning")
            else:
                _add_install_log(job_id, "Docker images pulled successfully")

            # Start the stack
            _update_install_step(job_id, "Starting monitoring stack...")
            if docker_cmd:
                result = await ssh.execute(f"cd {install_dir} && {docker_cmd} 'docker compose up -d'", timeout=120)
            else:
                result = await ssh.execute(f"cd {install_dir} && docker compose up -d", timeout=120)
            if not result.success:
                _add_install_log(job_id, f"Failed to start monitoring stack: {result.stderr}", "error")
                _installation_jobs[job_id]["status"] = "failed"
                _installation_jobs[job_id]["error"] = "Failed to start Docker containers"
                _installation_jobs[job_id]["completed"] = True
                return

            _add_install_log(job_id, "Docker containers started")

            # Wait for services to be healthy
            _update_install_step(job_id, "Waiting for services to start...")
            import asyncio

            await asyncio.sleep(10)

            # Check if services are running
            _update_install_step(job_id, "Verifying services...")
            if docker_cmd:
                result = await ssh.execute(f"{docker_cmd} \"docker ps --format '{{{{.Names}}}}: {{{{.Status}}}}'\"")
            else:
                result = await ssh.execute("docker ps --format '{{.Names}}: {{.Status}}'")

            _add_install_log(job_id, f"Running containers:\n{result.stdout}")

            # Mark as completed successfully
            _installation_jobs[job_id]["status"] = "completed"
            _installation_jobs[job_id]["completed"] = True
            _installation_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
            _update_install_step(job_id, "Installation completed successfully!")
            _add_install_log(job_id, f"Monitoring stack installation completed on {config.monitoring_server_ip}")

    except Exception as e:
        _add_install_log(job_id, f"Installation failed: {str(e)}", "error")
        _installation_jobs[job_id]["status"] = "failed"
        _installation_jobs[job_id]["error"] = str(e)
        _installation_jobs[job_id]["completed"] = True
        logger.exception(f"Failed to install monitoring stack: {e}")
    finally:
        db.close()


def _get_monitoring_compose(config) -> str:
    """Generate docker-compose.yml for monitoring stack."""
    return f"""services:
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: fleet-prometheus
    restart: unless-stopped
    network_mode: host
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time={config.prometheus_retention_time}'
      - '--storage.tsdb.retention.size={config.prometheus_retention_size}'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--web.listen-address=:{config.prometheus_port}'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/targets:/etc/prometheus/targets:ro
      - ./data/prometheus:/prometheus

  loki:
    image: grafana/loki:2.9.5
    container_name: fleet-loki
    restart: unless-stopped
    command: -config.file=/etc/loki/loki-config.yml
    volumes:
      - ./loki/loki-config.yml:/etc/loki/loki-config.yml:ro
      - ./data/loki:/loki
    ports:
      - "{config.loki_port}:3100"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.4.1
    container_name: fleet-grafana
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_USER: {config.grafana_admin_user or 'admin'}
      GF_SECURITY_ADMIN_PASSWORD: {config.grafana_admin_password or 'admin'}
      GF_USERS_ALLOW_SIGN_UP: 'false'
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./data/grafana:/var/lib/grafana
    ports:
      - "{config.grafana_port}:3000"
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge
"""


def _get_prometheus_config(config) -> str:
    """Generate prometheus.yml configuration.

    Note: scrape_timeout must be less than scrape_interval.
    We set scrape_timeout to 50% of scrape_interval for safety.
    """
    # Parse scrape_interval to calculate safe timeout
    interval = config.prometheus_scrape_interval or "30s"
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
      - targets: ['localhost:{config.prometheus_port}']

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
"""


def _get_loki_config(config) -> str:
    """Generate Loki configuration."""
    return f"""auth_enabled: false

server:
  http_listen_port: 3100
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
  retention_period: {config.loki_retention_days * 24}h
  ingestion_rate_mb: 16
  ingestion_burst_size_mb: 24

compactor:
  working_directory: /loki/compactor
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
"""


def _get_grafana_datasources(config) -> str:
    """Generate Grafana datasources provisioning config.

    Since Prometheus uses host networking, Grafana (in bridge network) must
    connect to it via the host IP address, not via Docker internal DNS.
    Loki still uses bridge networking so it can use the container name.
    """
    # Use the monitoring server IP for Prometheus since it's on host network
    prometheus_url = f"http://{config.monitoring_server_ip}:{config.prometheus_port}"

    return f"""apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: {prometheus_url}
    isDefault: true
    editable: false
    uid: prometheus

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false
    uid: loki
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
    """Load all dashboard JSON files from config directory."""
    import json

    dashboards = {}

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
        dashboards['gpu-fleet-overview'] = _get_fleet_overview_dashboard()
        return dashboards

    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(config_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Validate it's valid JSON
                    json.loads(content)
                    dash_name = filename.replace('.json', '')
                    dashboards[dash_name] = content
                    logger.info(f"Loaded dashboard: {filename}")
            except Exception as e:
                logger.warning(f"Failed to load dashboard {filename}: {e}")

    # If no config files found, fall back to embedded dashboard
    if not dashboards:
        logger.info("No dashboard config files found, using embedded dashboard")
        dashboards['gpu-fleet-overview'] = _get_fleet_overview_dashboard()

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
            # Row: Fleet Summary
            {"id": 1, "title": "Fleet Summary", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}},
            # Total GPUs
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
            # Active Nodes
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
            # Avg GPU Utilization Gauge
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
            # Avg Memory Gauge
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
            # Max Temperature
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
            # Total Power
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
            # Row: GPU Utilization
            {"id": 10, "title": "GPU Utilization", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 5}},
            # GPU Utilization Over Time
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
            # Memory Utilization Over Time
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
            # Row: Thermal & Power
            {"id": 20, "title": "Thermal & Power", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 14}},
            # Temperature
            {
                "id": 21,
                "title": "GPU Temperature",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 15},
                "targets": [
                    {
                        "expr": "amd_gpu_temperature_junction_celsius{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} Edge",
                    },
                    {
                        "expr": "amd_gpu_temperature_junction_celsius{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} Junction",
                    },
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
            # Power
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
            # Fan Speed
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
            # Row: Errors & Health
            {"id": 30, "title": "Errors & Health", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 23}},
            # ECC Correctable
            {
                "id": 31,
                "title": "ECC Correctable Errors",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 24},
                "targets": [
                    {
                        "expr": "sum(amd_gpu_ecc_errors_corrected_total{node_group=~\"$node_group\", instance=~\"$instance\"}) or vector(0)",
                        "legendFormat": "Correctable",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 1},
                                {"color": "orange", "value": 10},
                            ],
                        }
                    }
                },
                "options": {"colorMode": "value", "graphMode": "area"},
            },
            # ECC Uncorrectable
            {
                "id": 32,
                "title": "ECC Uncorrectable Errors",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 24},
                "targets": [
                    {
                        "expr": "sum(amd_gpu_ecc_errors_uncorrected_total{node_group=~\"$node_group\", instance=~\"$instance\"}) or vector(0)",
                        "legendFormat": "Uncorrectable",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}],
                        }
                    }
                },
                "options": {"colorMode": "value", "graphMode": "area"},
            },
            # PCIe Errors
            {
                "id": 33,
                "title": "PCIe Replay Errors",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 24},
                "targets": [
                    {
                        "expr": "sum(amd_gpu_pcie_bandwidth_mbps{node_group=~\"$node_group\", instance=~\"$instance\"}) or vector(0)",
                        "legendFormat": "PCIe Replay",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 100},
                                {"color": "red", "value": 1000},
                            ],
                        }
                    }
                },
                "options": {"colorMode": "value", "graphMode": "area"},
            },
            # Throttle Status
            {
                "id": 34,
                "title": "Throttle Events",
                "type": "stat",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 24},
                "targets": [
                    {
                        "expr": "count(amd_gpu_up{node_group=~\"$node_group\", instance=~\"$instance\"} > 0) or vector(0)",
                        "legendFormat": "Throttled GPUs",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}, {"color": "red", "value": 1}],
                        }
                    }
                },
                "options": {"colorMode": "value", "graphMode": "area"},
            },
            # ECC Errors Over Time
            {
                "id": 35,
                "title": "ECC Errors Over Time",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 6, "w": 24, "x": 0, "y": 28},
                "targets": [
                    {
                        "expr": "amd_gpu_ecc_errors_corrected_total{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} Correctable",
                    },
                    {
                        "expr": "amd_gpu_ecc_errors_uncorrected_total{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} Uncorrectable",
                    },
                ],
                "fieldConfig": {"defaults": {"custom": {"fillOpacity": 10}}},
            },
            # Row: PCIe & Bandwidth
            {"id": 40, "title": "PCIe & Bandwidth", "type": "row", "gridPos": {"h": 1, "w": 24, "x": 0, "y": 34}},
            # PCIe Bandwidth
            {
                "id": 41,
                "title": "PCIe Bandwidth",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 35},
                "targets": [
                    {
                        "expr": "rate(amd_gpu_pcie_bandwidth_mbps{node_group=~\"$node_group\", instance=~\"$instance\"}[1m])",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} TX",
                    },
                    {
                        "expr": "rate(amd_gpu_pcie_bandwidth_mbps{node_group=~\"$node_group\", instance=~\"$instance\"}[1m])",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} RX",
                    },
                ],
                "fieldConfig": {"defaults": {"unit": "Bps", "custom": {"fillOpacity": 10}}},
            },
            # Clock Speeds
            {
                "id": 42,
                "title": "GPU Clock Speeds",
                "type": "timeseries",
                "datasource": {"type": "prometheus", "uid": "prometheus"},
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 35},
                "targets": [
                    {
                        "expr": "amd_gpu_sclk_mhz{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} SCLK",
                    },
                    {
                        "expr": "amd_gpu_mclk_mhz{node_group=~\"$node_group\", instance=~\"$instance\"}",
                        "legendFormat": "{{instance}} GPU{{gpu_id}} MCLK",
                    },
                ],
                "fieldConfig": {"defaults": {"unit": "rotmhz", "custom": {"fillOpacity": 10}}},
            },
        ],
    }
    return json.dumps(dashboard, indent=2)


@router.post("/sync-targets")
async def sync_prometheus_targets(db: Session = Depends(get_db)):
    """
    Sync all active GPU nodes as Prometheus scrape targets on the monitoring server.
    This updates the target files so Prometheus knows where to scrape metrics from.
    """
    from ...models.database import Node, NodeGroup, NodeStatus

    config = get_or_create_config(db)

    if not config.monitoring_server_ip:
        raise HTTPException(status_code=400, detail="Monitoring server IP not configured")

    if not config.setup_monitoring_stack:
        raise HTTPException(status_code=400, detail="Remote monitoring setup not enabled")

    # Get all active nodes
    nodes = db.query(Node).filter(Node.status == NodeStatus.ACTIVE.value).all()

    if not nodes:
        return {"message": "No active nodes to sync", "targets_count": 0}

    # Build targets JSON for Prometheus file_sd_configs
    import json

    targets = []
    for node in nodes:
        node_group = db.query(NodeGroup).filter(NodeGroup.id == node.node_group_id).first()
        targets.append(
            {
                "targets": [f"{node.ip_address}:{node.gpu_exporter_port}"],
                "labels": {
                    "job": "gpu-exporter",
                    "node_group": node_group.name if node_group else "unknown",
                    "hostname": node.hostname or node.ip_address,
                    "gpu_model": node.gpu_model or "unknown",
                },
            }
        )

    targets_json = json.dumps(targets, indent=2)

    # Validate SSH config and create manager
    _validate_monitoring_ssh_config(config)

    import base64

    ssh = _create_monitoring_ssh_manager(config)

    try:
        async with ssh:
            # Upload targets file - use heredoc to avoid shell escaping issues
            install_dir = f"/home/{config.monitoring_ssh_user}/fleet-monitoring"
            targets_b64 = base64.b64encode(targets_json.encode()).decode()
            result = await ssh.execute(
                f"cat << 'EOFB64' | base64 -d > {install_dir}/prometheus/targets/gpu-nodes.json\n{targets_b64}\nEOFB64"
            )

            if not result.success:
                raise HTTPException(status_code=500, detail=f"Failed to upload targets: {result.stderr}")

            # Reload Prometheus config (if running)
            await ssh.execute("docker exec fleet-prometheus kill -HUP 1 2>/dev/null || true")

            return {
                "message": f"Synced {len(nodes)} active nodes to Prometheus",
                "targets_count": len(nodes),
                "targets_file": f"{install_dir}/prometheus/targets/gpu-nodes.json",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to sync targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
