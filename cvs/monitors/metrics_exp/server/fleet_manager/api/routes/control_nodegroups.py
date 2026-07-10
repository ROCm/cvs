"""API routes for Control Node Groups (Slurm head nodes and Kubernetes control plane)."""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ...models.database import (
    ControlNode,
    ControlNodeGroup,
    MonitoringServer,
    NodeStatus,
    get_db,
)
from ...api.schemas import (
    ControlNodeBulkCreate,
    ControlNodeCreate,
    ControlNodeGroupCreate,
    ControlNodeGroupCreateWithNodes,
    ControlNodeGroupDetail,
    ControlNodeGroupResponse,
    ControlNodeGroupUpdate,
    ControlNodeResponse,
    InstallationRequest,
    InstallationResponse,
    InstallationStatus,
    SSHKeyUpload,
)
from ...services.ssh_manager import JumpHostConfig, SSHManager
from ...services.installer import ControlNodeInstaller
from ...services.prometheus_config import PrometheusConfigManager
from ...services.grafana_provisioner import GrafanaProvisioner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/control-nodegroups", tags=["Control Node Groups"])

# SSH key storage directory
SSH_KEYS_PATH = os.environ.get("SSH_KEYS_PATH", "/app/ssh_keys")
os.makedirs(SSH_KEYS_PATH, exist_ok=True)

# Loki URL for Promtail config
LOKI_URL = os.environ.get("LOKI_URL", "http://loki:3100")

# Max parallel SSH operations
DEFAULT_PARALLEL_LIMIT = 10

# In-memory job tracking
_jobs: dict = {}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _group_to_response(group: ControlNodeGroup, db: Session) -> ControlNodeGroupResponse:
    """Convert a ControlNodeGroup ORM object to its response schema."""
    node_count = len(group.nodes)
    active_count = sum(1 for n in group.nodes if n.status == NodeStatus.ACTIVE.value)
    return ControlNodeGroupResponse(
        id=group.id,
        name=group.name,
        description=group.description or "",
        control_type=group.control_type,
        custom_exporter_port=group.custom_exporter_port or 0,
        ssh_user=group.ssh_user,
        ssh_port=group.ssh_port,
        ssh_auth_type=group.ssh_auth_type,
        ssh_key_path=group.ssh_key_path,
        has_ssh_password=bool(group.ssh_password),
        use_jump_host=group.use_jump_host or False,
        jump_host=group.jump_host,
        jump_port=group.jump_port or 22,
        jump_user=group.jump_user,
        jump_auth_type=group.jump_auth_type or "key",
        jump_key_path=group.jump_key_path,
        has_jump_password=bool(group.jump_password),
        remote_auth_type=group.remote_auth_type or "key",
        remote_key_path=group.remote_key_path,
        has_remote_password=bool(group.remote_password),
        monitoring_server_id=group.monitoring_server_id,
        node_count=node_count,
        active_node_count=active_count,
        created_at=group.created_at,
        updated_at=group.updated_at,
        kubeconfig_source=group.kubeconfig_source or "auto",
        kubeconfig_remote_path=group.kubeconfig_remote_path,
        has_kubeconfig_upload=bool(group.kubeconfig_local_path),
    )


def _node_to_response(node: ControlNode) -> ControlNodeResponse:
    return ControlNodeResponse(
        id=node.id,
        control_node_group_id=node.control_node_group_id,
        ip_address=node.ip_address,
        hostname=node.hostname,
        status=NodeStatus(node.status),
        status_message=node.status_message,
        last_seen=node.last_seen,
        role_info=node.role_info,
        created_at=node.created_at,
        updated_at=node.updated_at,
    )


def _build_ssh_manager(group: ControlNodeGroup, target_ip: str) -> SSHManager:
    """Construct an SSHManager for a control node, respecting jump host config."""
    jump_host_cfg = None
    if group.use_jump_host and group.jump_host:
        jump_host_cfg = JumpHostConfig(
            host=group.jump_host,
            port=group.jump_port or 22,
            username=group.jump_user or "root",
            auth_type=group.jump_auth_type or "key",
            key_path=group.jump_key_path,
            password=group.jump_password,
            remote_auth_type=group.remote_auth_type or "key",
            remote_key_path=group.remote_key_path,
            remote_password=group.remote_password,
        )

    # For direct connections the SSH key/password refers to the control node itself
    key_path = group.ssh_key_path if not group.use_jump_host else None
    password = group.ssh_password if not group.use_jump_host else None

    return SSHManager(
        host=target_ip,
        username=group.ssh_user or "root",
        auth_type=group.ssh_auth_type or "key",
        key_path=key_path,
        password=password,
        port=group.ssh_port or 22,
        jump_host=jump_host_cfg,
    )


def _has_credentials(group: ControlNodeGroup) -> bool:
    """Return True if the group has enough credentials configured to attempt SSH."""
    if group.use_jump_host:
        jump_ok = (group.jump_auth_type == "password" and bool(group.jump_password)) or bool(group.jump_key_path)
        remote_ok = (group.remote_auth_type == "password" and bool(group.remote_password)) or bool(
            group.remote_key_path
        )
        return jump_ok and remote_ok
    else:
        return (group.ssh_auth_type == "password" and bool(group.ssh_password)) or bool(group.ssh_key_path)


def _get_loki_url(group: ControlNodeGroup, db: Session) -> str:
    """Return Loki URL for a control node group's monitoring server, or the default."""
    if group.monitoring_server_id:
        server = db.query(MonitoringServer).filter(MonitoringServer.id == group.monitoring_server_id).first()
        if server and server.server_ip:
            return f"http://{server.server_ip}:{server.loki_port}"
    return LOKI_URL


def _get_exporter_port(group: ControlNodeGroup) -> int:
    """Return the effective exporter port for this group."""
    if group.custom_exporter_port and group.custom_exporter_port > 0:
        return group.custom_exporter_port
    return 9418 if group.control_type == "slurm" else 9419


# -----------------------------------------------------------------------
# Background task helpers
# -----------------------------------------------------------------------


async def verify_single_node(
    node_id: int,
    node_ip: str,
    group: ControlNodeGroup,
    semaphore: asyncio.Semaphore,
    db_factory,
) -> dict:
    """Verify SSH connectivity for a single control node."""
    async with semaphore:
        db = db_factory()
        try:
            node = db.query(ControlNode).filter(ControlNode.id == node_id).first()
            if not node:
                return {"node_id": node_id, "success": False, "error": "Node not found"}

            node.status = NodeStatus.PENDING.value
            db.commit()

            ssh = _build_ssh_manager(group, node_ip)
            try:
                connected, hostname = await ssh.check_connection()
                if connected:
                    node.status = NodeStatus.CONNECTED.value
                    node.hostname = hostname or node_ip
                    node.last_seen = datetime.utcnow()
                    node.status_message = "SSH connection verified"
                else:
                    node.status = NodeStatus.UNREACHABLE.value
                    node.status_message = "SSH connection failed"
                db.commit()
                return {"node_id": node_id, "success": connected}
            except Exception as e:
                node.status = NodeStatus.UNREACHABLE.value
                node.status_message = str(e)[:500]
                db.commit()
                return {"node_id": node_id, "success": False, "error": str(e)}
            finally:
                await ssh.disconnect()
        finally:
            db.close()


async def run_verification(
    job_id: str,
    group_id: int,
    node_ids: List[int],
    parallel_limit: int,
    db_factory,
):
    """Background task: verify SSH for all nodes in a control node group."""
    _jobs[job_id] = {"status": "running", "results": []}

    db = db_factory()
    try:
        group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
        if not group:
            _jobs[job_id]["status"] = "error"
            return

        nodes = (
            db.query(ControlNode)
            .filter(
                ControlNode.id.in_(node_ids),
                ControlNode.control_node_group_id == group_id,
            )
            .all()
        )
        # Detach for use in sub-tasks
        node_data = [(n.id, n.ip_address) for n in nodes]
        group_snapshot = group
    finally:
        db.close()

    semaphore = asyncio.Semaphore(parallel_limit)
    from ...models.database import SessionLocal

    tasks = [verify_single_node(nid, nip, group_snapshot, semaphore, SessionLocal) for nid, nip in node_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _jobs[job_id]["status"] = "completed"
    _jobs[job_id]["results"] = results


async def install_single_node(
    node_id: int,
    node_ip: str,
    group: ControlNodeGroup,
    semaphore: asyncio.Semaphore,
    db_factory,
    loki_url: str,
    exporter_port: int,
) -> dict:
    """Install control plane exporters on a single node."""
    async with semaphore:
        db = db_factory()
        try:
            node = db.query(ControlNode).filter(ControlNode.id == node_id).first()
            if not node:
                return {"node_id": node_id, "success": False}

            node.status = NodeStatus.INSTALLING.value
            db.commit()

            ssh = _build_ssh_manager(group, node_ip)
            try:
                connected = await ssh.connect()
                if not connected:
                    node.status = NodeStatus.UNREACHABLE.value
                    node.status_message = "SSH connection failed during install"
                    db.commit()
                    return {"node_id": node_id, "success": False}

                hostname = node.hostname or node_ip
                installer = ControlNodeInstaller(
                    ssh_manager=ssh,
                    loki_url=loki_url,
                    node_group_name=group.name,
                    hostname=hostname,
                    control_type=group.control_type,
                    kubeconfig_source=group.kubeconfig_source or "auto",
                    kubeconfig_remote_path=group.kubeconfig_remote_path,
                    kubeconfig_local_path=group.kubeconfig_local_path,
                )

                results = await installer.install_all(
                    node_port=9100,
                    custom_exporter_port=exporter_port,
                )

                node_exporter_ok = results.get("node_exporter", False)
                promtail_ok = results.get("promtail", False)
                cp_exporter_key = "slurm_exporter" if group.control_type == "slurm" else "k8s_exporter"
                cp_exporter_ok = results.get(cp_exporter_key, False)

                if node_exporter_ok and promtail_ok:
                    node.status = NodeStatus.ACTIVE.value
                    node.status_message = (
                        f"Installed: node_exporter={node_exporter_ok}, "
                        f"promtail={promtail_ok}, "
                        f"{cp_exporter_key}={cp_exporter_ok}"
                    )
                else:
                    node.status = NodeStatus.ERROR.value
                    node.status_message = f"Install incomplete: {results}"

                node.last_seen = datetime.utcnow()
                db.commit()

                return {"node_id": node_id, "success": node.status == NodeStatus.ACTIVE.value, "results": results}

            except Exception as e:
                node.status = NodeStatus.ERROR.value
                node.status_message = str(e)[:500]
                db.commit()
                logger.error(f"Install failed on {node_ip}: {e}")
                return {"node_id": node_id, "success": False, "error": str(e)}
            finally:
                await ssh.disconnect()
        finally:
            db.close()


async def run_installation(
    job_id: str,
    group_id: int,
    node_ids: List[int],
    parallel_limit: int,
    db_factory,
):
    """Background task: install control plane exporters on multiple nodes."""
    _jobs[job_id] = {"status": "running", "results": []}

    db = db_factory()
    try:
        group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
        if not group:
            _jobs[job_id]["status"] = "error"
            return

        loki_url = _get_loki_url(group, db)
        exporter_port = _get_exporter_port(group)
        nodes = (
            db.query(ControlNode)
            .filter(
                ControlNode.id.in_(node_ids),
                ControlNode.control_node_group_id == group_id,
            )
            .all()
        )
        node_data = [(n.id, n.ip_address) for n in nodes]
        group_snapshot = group
        group_name = group.name
        control_type = group.control_type
        monitoring_server_id = group.monitoring_server_id
    finally:
        db.close()

    semaphore = asyncio.Semaphore(parallel_limit)
    from ...models.database import SessionLocal

    tasks = [
        install_single_node(nid, nip, group_snapshot, semaphore, SessionLocal, loki_url, exporter_port)
        for nid, nip in node_data
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _jobs[job_id]["results"] = results

    # Update Prometheus targets and Grafana dashboard
    try:
        db = SessionLocal()
        nodes_all = db.query(ControlNode).filter(ControlNode.control_node_group_id == group_id).all()
        nodes_for_targets = [
            {
                "ip": n.ip_address,
                "hostname": n.hostname or n.ip_address,
                "node_port": 9100,
                "exporter_port": exporter_port,
                "status": n.status,
            }
            for n in nodes_all
        ]
        db.close()

        prom = PrometheusConfigManager()
        prom.update_control_node_group_targets(group_name, control_type, nodes_for_targets)
        await prom.reload_prometheus()

        # Provision Grafana dashboard if monitoring server is set
        if monitoring_server_id:
            grafana = GrafanaProvisioner()
            await grafana.provision_control_node_group_dashboard(group_name, control_type)
    except Exception as e:
        logger.error(f"Post-install target/dashboard update failed: {e}")

    _jobs[job_id]["status"] = "completed"


# -----------------------------------------------------------------------
# CRUD Routes
# -----------------------------------------------------------------------


@router.get("", response_model=List[ControlNodeGroupResponse])
async def list_control_node_groups(db: Session = Depends(get_db)):
    """List all control node groups."""
    groups = db.query(ControlNodeGroup).all()
    return [_group_to_response(g, db) for g in groups]


@router.post("", response_model=ControlNodeGroupResponse, status_code=201)
async def create_control_node_group(
    group_in: ControlNodeGroupCreate,
    db: Session = Depends(get_db),
):
    """Create a new control node group."""
    # Validate unique name
    existing = db.query(ControlNodeGroup).filter(ControlNodeGroup.name == group_in.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Control node group '{group_in.name}' already exists")

    # Validate monitoring server if provided
    if group_in.monitoring_server_id:
        server = db.query(MonitoringServer).filter(MonitoringServer.id == group_in.monitoring_server_id).first()
        if not server:
            raise HTTPException(status_code=400, detail="Monitoring server not found")

    # Validate control_type
    if group_in.control_type not in ("slurm", "kubernetes"):
        raise HTTPException(status_code=400, detail="control_type must be 'slurm' or 'kubernetes'")

    group = ControlNodeGroup(
        name=group_in.name,
        description=group_in.description or "",
        control_type=group_in.control_type,
        custom_exporter_port=group_in.custom_exporter_port or 0,
        ssh_user=group_in.ssh_user,
        ssh_port=group_in.ssh_port,
        ssh_auth_type=group_in.ssh_auth_type,
        ssh_password=group_in.ssh_password,
        use_jump_host=group_in.use_jump_host,
        jump_host=group_in.jump_host,
        jump_port=group_in.jump_port,
        jump_user=group_in.jump_user,
        jump_auth_type=group_in.jump_auth_type,
        jump_password=group_in.jump_password,
        remote_auth_type=group_in.remote_auth_type,
        remote_key_path=group_in.remote_key_path,
        remote_password=group_in.remote_password,
        monitoring_server_id=group_in.monitoring_server_id,
        kubeconfig_source=group_in.kubeconfig_source or "auto",
        kubeconfig_remote_path=group_in.kubeconfig_remote_path,
    )
    db.add(group)
    db.commit()
    db.refresh(group)
    logger.info(f"Created control node group: {group.name} (type={group.control_type})")
    return _group_to_response(group, db)


@router.post("/with-nodes", response_model=ControlNodeGroupDetail, status_code=201)
async def create_control_node_group_with_nodes(
    group_in: ControlNodeGroupCreateWithNodes,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a control node group with initial nodes and optionally trigger verification."""
    # Validate unique name
    existing = db.query(ControlNodeGroup).filter(ControlNodeGroup.name == group_in.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Control node group '{group_in.name}' already exists")

    if group_in.monitoring_server_id:
        server = db.query(MonitoringServer).filter(MonitoringServer.id == group_in.monitoring_server_id).first()
        if not server:
            raise HTTPException(status_code=400, detail="Monitoring server not found")

    if group_in.control_type not in ("slurm", "kubernetes"):
        raise HTTPException(status_code=400, detail="control_type must be 'slurm' or 'kubernetes'")

    group = ControlNodeGroup(
        name=group_in.name,
        description=group_in.description or "",
        control_type=group_in.control_type,
        custom_exporter_port=group_in.custom_exporter_port or 0,
        ssh_user=group_in.ssh_user,
        ssh_port=group_in.ssh_port,
        ssh_auth_type=group_in.ssh_auth_type,
        ssh_password=group_in.ssh_password,
        use_jump_host=group_in.use_jump_host,
        jump_host=group_in.jump_host,
        jump_port=group_in.jump_port,
        jump_user=group_in.jump_user,
        jump_auth_type=group_in.jump_auth_type,
        jump_password=group_in.jump_password,
        remote_auth_type=group_in.remote_auth_type,
        remote_key_path=group_in.remote_key_path,
        remote_password=group_in.remote_password,
        monitoring_server_id=group_in.monitoring_server_id,
        kubeconfig_source=group_in.kubeconfig_source or "auto",
        kubeconfig_remote_path=group_in.kubeconfig_remote_path,
    )
    db.add(group)
    db.commit()
    db.refresh(group)

    # Add initial nodes
    added_nodes = []
    for ip in group_in.ip_addresses:
        ip = ip.strip()
        if not ip:
            continue
        # Skip duplicates
        if (
            db.query(ControlNode)
            .filter(
                ControlNode.control_node_group_id == group.id,
                ControlNode.ip_address == ip,
            )
            .first()
        ):
            continue
        node = ControlNode(
            control_node_group_id=group.id,
            ip_address=ip,
            status=NodeStatus.PENDING.value,
        )
        db.add(node)
        added_nodes.append(node)

    db.commit()
    db.refresh(group)

    # Auto-verify if credentials are configured
    if added_nodes and _has_credentials(group):
        node_ids = [n.id for n in added_nodes]
        job_id = str(uuid.uuid4())
        from ...models.database import SessionLocal

        background_tasks.add_task(run_verification, job_id, group.id, node_ids, DEFAULT_PARALLEL_LIMIT, SessionLocal)
        logger.info(f"Auto-triggered verification for {len(node_ids)} nodes in '{group.name}'")

    return ControlNodeGroupDetail(
        **_group_to_response(group, db).model_dump(),
        nodes=[_node_to_response(n) for n in group.nodes],
    )


@router.get("/{group_id}", response_model=ControlNodeGroupDetail)
async def get_control_node_group(group_id: int, db: Session = Depends(get_db)):
    """Get control node group detail with all nodes."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")
    return ControlNodeGroupDetail(
        **_group_to_response(group, db).model_dump(),
        nodes=[_node_to_response(n) for n in group.nodes],
    )


@router.patch("/{group_id}", response_model=ControlNodeGroupResponse)
async def update_control_node_group(
    group_id: int,
    update: ControlNodeGroupUpdate,
    db: Session = Depends(get_db),
):
    """Update a control node group."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(group, field, value)

    group.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(group)
    return _group_to_response(group, db)


@router.delete("/{group_id}", status_code=204)
async def delete_control_node_group(group_id: int, db: Session = Depends(get_db)):
    """Delete a control node group and all its nodes, Prometheus targets, and Grafana dashboard."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    group_name = group.name
    control_type = group.control_type

    # Remove Prometheus targets
    try:
        prom = PrometheusConfigManager()
        prom.remove_control_node_group_targets(group_name, control_type)
        await prom.reload_prometheus()
    except Exception as e:
        logger.warning(f"Could not remove Prometheus targets for '{group_name}': {e}")

    # Remove Grafana dashboard
    try:
        grafana = GrafanaProvisioner()
        await grafana.remove_control_node_group_dashboard(group_name)
    except Exception as e:
        logger.warning(f"Could not remove Grafana dashboard for '{group_name}': {e}")

    # Remove SSH key files
    safe_name = "".join(c if c.isalnum() else "_" for c in group_name)
    for key_file in [
        os.path.join(SSH_KEYS_PATH, f"cng_{safe_name}_id_rsa"),
        os.path.join(SSH_KEYS_PATH, f"cng_{safe_name}_jump_id_rsa"),
    ]:
        try:
            if os.path.exists(key_file):
                os.remove(key_file)
        except Exception as e:
            logger.warning(f"Could not remove key file {key_file}: {e}")

    db.delete(group)
    db.commit()
    logger.info(f"Deleted control node group: {group_name}")


# -----------------------------------------------------------------------
# SSH Key Upload
# -----------------------------------------------------------------------


@router.post("/{group_id}/ssh-key", response_model=SSHKeyUpload)
async def upload_ssh_key(
    group_id: int,
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH key for direct connection to control nodes."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    content = await key_file.read()
    content_str = content.decode("utf-8", errors="ignore")
    if "PRIVATE KEY" not in content_str:
        raise HTTPException(status_code=400, detail="File does not appear to be a private key")

    safe_name = "".join(c if c.isalnum() else "_" for c in group.name)
    key_path = os.path.join(SSH_KEYS_PATH, f"cng_{safe_name}_id_rsa")

    with open(key_path, "wb") as f:
        f.write(content)
    os.chmod(key_path, 0o600)

    group.ssh_key_path = key_path
    group.updated_at = datetime.utcnow()
    db.commit()

    return SSHKeyUpload(key_path=key_path, message="SSH key uploaded successfully")


@router.post("/{group_id}/jump-key", response_model=SSHKeyUpload)
async def upload_jump_key(
    group_id: int,
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH key for jump host connection."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    content = await key_file.read()
    content_str = content.decode("utf-8", errors="ignore")
    if "PRIVATE KEY" not in content_str:
        raise HTTPException(status_code=400, detail="File does not appear to be a private key")

    safe_name = "".join(c if c.isalnum() else "_" for c in group.name)
    key_path = os.path.join(SSH_KEYS_PATH, f"cng_{safe_name}_jump_id_rsa")

    with open(key_path, "wb") as f:
        f.write(content)
    os.chmod(key_path, 0o600)

    group.jump_key_path = key_path
    group.updated_at = datetime.utcnow()
    db.commit()

    return SSHKeyUpload(key_path=key_path, message="Jump host SSH key uploaded successfully")


@router.post("/{group_id}/kubeconfig", response_model=SSHKeyUpload)
async def upload_kubeconfig(
    group_id: int,
    kubeconfig_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a kubeconfig file for Kubernetes control plane groups.

    The file is stored securely on the Fleet Manager server.
    On the next Install/Force Reinstall it is pushed to the K8s node at
    /etc/k8s-cp-exporter/kubeconfig and the exporter is configured to use it.
    Only valid for groups with control_type='kubernetes'.
    """
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")
    if group.control_type != "kubernetes":
        raise HTTPException(
            status_code=400, detail="Kubeconfig upload is only applicable to Kubernetes control node groups"
        )

    content = await kubeconfig_file.read()
    # Basic sanity check — kubeconfig files contain these keys
    content_str = content.decode("utf-8", errors="ignore")
    if "apiVersion" not in content_str or "clusters" not in content_str:
        raise HTTPException(
            status_code=400, detail="File does not appear to be a valid kubeconfig (missing apiVersion or clusters)"
        )

    safe_name = "".join(c if c.isalnum() else "_" for c in group.name)
    kubeconfig_path = os.path.join(SSH_KEYS_PATH, f"cng_{safe_name}_kubeconfig")

    with open(kubeconfig_path, "wb") as f:
        f.write(content)
    os.chmod(kubeconfig_path, 0o600)

    group.kubeconfig_local_path = kubeconfig_path
    group.kubeconfig_source = "upload"
    group.updated_at = datetime.utcnow()
    db.commit()

    logger.info(f"Kubeconfig uploaded for control node group '{group.name}'")
    return SSHKeyUpload(
        key_path=kubeconfig_path,
        message="Kubeconfig uploaded successfully. Force Reinstall to deploy it to the K8s node.",
    )


# -----------------------------------------------------------------------
# Verify Connectivity
# -----------------------------------------------------------------------


@router.post("/{group_id}/verify")
async def verify_connectivity(
    group_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Verify SSH connectivity to all nodes in this control node group."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    if not _has_credentials(group):
        raise HTTPException(
            status_code=400,
            detail="SSH credentials not configured. Upload an SSH key or set a password first.",
        )

    nodes = db.query(ControlNode).filter(ControlNode.control_node_group_id == group_id).all()
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes in this group")

    node_ids = [n.id for n in nodes]
    job_id = str(uuid.uuid4())

    from ...models.database import SessionLocal

    background_tasks.add_task(run_verification, job_id, group_id, node_ids, DEFAULT_PARALLEL_LIMIT, SessionLocal)

    return {
        "job_id": job_id,
        "message": f"Verification started for {len(nodes)} node(s)",
        "total_nodes": len(nodes),
    }


# -----------------------------------------------------------------------
# Install Exporters
# -----------------------------------------------------------------------


@router.post("/{group_id}/install", response_model=InstallationResponse)
async def install_exporters(
    group_id: int,
    request: InstallationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Install control plane exporters on connected nodes."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    if not _has_credentials(group):
        raise HTTPException(status_code=400, detail="SSH credentials not configured")

    # Select target nodes
    query = db.query(ControlNode).filter(ControlNode.control_node_group_id == group_id)
    if request.node_ids:
        query = query.filter(ControlNode.id.in_(request.node_ids))

    if request.force:
        # Install on connected, active, and error nodes
        target_nodes = query.filter(
            ControlNode.status.in_(
                [
                    NodeStatus.CONNECTED.value,
                    NodeStatus.ACTIVE.value,
                    NodeStatus.ERROR.value,
                    NodeStatus.PENDING.value,
                ]
            )
        ).all()
    else:
        # Install only on connected nodes
        target_nodes = query.filter(ControlNode.status == NodeStatus.CONNECTED.value).all()

    if not target_nodes:
        raise HTTPException(
            status_code=400,
            detail="No eligible nodes found. Verify connectivity first.",
        )

    job_id = str(uuid.uuid4())
    node_ids = [n.id for n in target_nodes]
    parallel_limit = min(request.parallel_limit, 50)

    statuses = [
        InstallationStatus(
            node_id=n.id,
            ip_address=n.ip_address,
            status="queued",
            message="Queued for installation",
        )
        for n in target_nodes
    ]

    from ...models.database import SessionLocal

    background_tasks.add_task(run_installation, job_id, group_id, node_ids, parallel_limit, SessionLocal)

    return InstallationResponse(
        job_id=job_id,
        total_nodes=len(target_nodes),
        statuses=statuses,
    )


# -----------------------------------------------------------------------
# Refresh Prometheus Targets
# -----------------------------------------------------------------------


@router.post("/{group_id}/refresh-targets")
async def refresh_targets(group_id: int, db: Session = Depends(get_db)):
    """Refresh Prometheus target files for this control node group."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    exporter_port = _get_exporter_port(group)
    nodes = db.query(ControlNode).filter(ControlNode.control_node_group_id == group_id).all()

    nodes_for_targets = [
        {
            "ip": n.ip_address,
            "hostname": n.hostname or n.ip_address,
            "node_port": 9100,
            "exporter_port": exporter_port,
            "status": n.status,
        }
        for n in nodes
    ]

    prom = PrometheusConfigManager()
    success = prom.update_control_node_group_targets(group.name, group.control_type, nodes_for_targets)
    if success:
        await prom.reload_prometheus()

    return {"success": success, "active_nodes": sum(1 for n in nodes if n.status == "active")}


# -----------------------------------------------------------------------
# Node sub-routes
# -----------------------------------------------------------------------


@router.get("/{group_id}/nodes", response_model=List[ControlNodeResponse])
async def list_nodes(
    group_id: int,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all nodes in a control node group."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    query = db.query(ControlNode).filter(ControlNode.control_node_group_id == group_id)
    if status:
        query = query.filter(ControlNode.status == status)
    return [_node_to_response(n) for n in query.all()]


@router.post("/{group_id}/nodes", response_model=ControlNodeResponse, status_code=201)
async def add_node(
    group_id: int,
    node_in: ControlNodeCreate,
    db: Session = Depends(get_db),
):
    """Add a single node to a control node group."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    existing = (
        db.query(ControlNode)
        .filter(
            ControlNode.control_node_group_id == group_id,
            ControlNode.ip_address == node_in.ip_address,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail=f"Node {node_in.ip_address} already exists in this group")

    node = ControlNode(
        control_node_group_id=group_id,
        ip_address=node_in.ip_address,
        hostname=node_in.hostname,
        status=NodeStatus.PENDING.value,
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return _node_to_response(node)


@router.post("/{group_id}/nodes/bulk", response_model=List[ControlNodeResponse], status_code=201)
async def bulk_add_nodes(
    group_id: int,
    bulk: ControlNodeBulkCreate,
    db: Session = Depends(get_db),
):
    """Bulk-add nodes to a control node group (skips duplicates)."""
    group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Control node group not found")

    added = []
    for ip in bulk.ip_addresses:
        ip = ip.strip()
        if not ip:
            continue
        if (
            db.query(ControlNode)
            .filter(
                ControlNode.control_node_group_id == group_id,
                ControlNode.ip_address == ip,
            )
            .first()
        ):
            continue
        node = ControlNode(
            control_node_group_id=group_id,
            ip_address=ip,
            status=NodeStatus.PENDING.value,
        )
        db.add(node)
        added.append(node)

    db.commit()
    for n in added:
        db.refresh(n)
    return [_node_to_response(n) for n in added]


@router.delete("/{group_id}/nodes/{node_id}", status_code=204)
async def delete_node(
    group_id: int,
    node_id: int,
    db: Session = Depends(get_db),
):
    """Remove a node from a control node group."""
    node = (
        db.query(ControlNode)
        .filter(
            ControlNode.id == node_id,
            ControlNode.control_node_group_id == group_id,
        )
        .first()
    )
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    db.delete(node)
    db.commit()

    # Refresh Prometheus targets
    try:
        group = db.query(ControlNodeGroup).filter(ControlNodeGroup.id == group_id).first()
        if group:
            exporter_port = _get_exporter_port(group)
            remaining = db.query(ControlNode).filter(ControlNode.control_node_group_id == group_id).all()
            nodes_for_targets = [
                {
                    "ip": n.ip_address,
                    "hostname": n.hostname or n.ip_address,
                    "node_port": 9100,
                    "exporter_port": exporter_port,
                    "status": n.status,
                }
                for n in remaining
            ]
            prom = PrometheusConfigManager()
            prom.update_control_node_group_targets(group.name, group.control_type, nodes_for_targets)
            await prom.reload_prometheus()
    except Exception as e:
        logger.warning(f"Could not refresh targets after node deletion: {e}")
