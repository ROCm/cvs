"""Node group management endpoints."""

import logging
import asyncio
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from ..schemas import (
    NodeGroupCreate,
    NodeGroupCreateWithNodes,
    NodeGroupResponse,
    NodeGroupDetail,
    NodeGroupUpdate,
    SSHKeyUpload,
    InstallationRequest,
    InstallationResponse,
    InstallationStatus,
)
from ...models import get_db, NodeGroup, Node, MetricConfig
from ...models.database import MonitoringServer, MetricGroup
from ...models.database import NodeStatus as DBNodeStatus
from ...services import (
    CredentialStore,
    NodeInstaller,
    PrometheusConfigManager,
    GrafanaProvisioner,
)
from ...services.ssh_manager import SSHManager, JumpHostConfig

router = APIRouter(prefix="/nodegroups", tags=["Node Groups"])
logger = logging.getLogger(__name__)

# Service instances
credential_store = CredentialStore()
prometheus_config = PrometheusConfigManager()
grafana = GrafanaProvisioner()


def get_node_group_or_404(db: Session, node_group_id: int) -> NodeGroup:
    """Get node group by ID or raise 404."""
    node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
    if not node_group:
        raise HTTPException(status_code=404, detail="Node group not found")
    return node_group


def build_node_group_response(ng: NodeGroup, include_nodes: bool = False) -> NodeGroupResponse:
    """Build a NodeGroupResponse from a NodeGroup model."""
    active_nodes = sum(1 for n in ng.nodes if n.status == DBNodeStatus.ACTIVE.value)
    return NodeGroupResponse(
        id=ng.id,
        name=ng.name,
        description=ng.description,
        ssh_user=ng.ssh_user,
        ssh_port=ng.ssh_port,
        ssh_auth_type=ng.ssh_auth_type or "key",
        ssh_key_path=ng.ssh_key_path,
        use_jump_host=ng.use_jump_host or False,
        jump_host=ng.jump_host,
        jump_port=ng.jump_port or 22,
        jump_user=ng.jump_user,
        jump_auth_type=ng.jump_auth_type or "key",
        jump_key_path=ng.jump_key_path,
        remote_auth_type=ng.remote_auth_type or "key",
        remote_key_path=ng.remote_key_path,
        metric_config_id=ng.metric_config_id,
        monitoring_server_id=ng.monitoring_server_id,
        metric_group_id=ng.metric_group_id,
        node_count=len(ng.nodes),
        active_nodes=active_nodes,
        has_ssh_password=bool(ng.ssh_password),
        has_jump_password=bool(ng.jump_password),
        has_remote_password=bool(ng.remote_password),
        created_at=ng.created_at,
        updated_at=ng.updated_at,
    )


@router.get("", response_model=List[NodeGroupResponse])
def list_node_groups(db: Session = Depends(get_db)):
    """List all node groups."""
    node_groups = db.query(NodeGroup).all()
    return [build_node_group_response(ng) for ng in node_groups]


@router.post("", response_model=NodeGroupResponse, status_code=201)
def create_node_group(
    node_group: NodeGroupCreate,
    db: Session = Depends(get_db),
):
    """Create a new node group."""
    # Check for duplicate name
    existing = db.query(NodeGroup).filter(NodeGroup.name == node_group.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Node group with this name already exists")

    # Verify metric config exists if provided
    if node_group.metric_config_id:
        config = db.query(MetricConfig).filter(MetricConfig.id == node_group.metric_config_id).first()
        if not config:
            raise HTTPException(status_code=400, detail="Metric config not found")

    # Verify monitoring server exists if provided
    if node_group.monitoring_server_id:
        server = db.query(MonitoringServer).filter(MonitoringServer.id == node_group.monitoring_server_id).first()
        if not server:
            raise HTTPException(status_code=400, detail="Monitoring server not found")

    # Verify metric group exists if provided
    if node_group.metric_group_id:
        group = db.query(MetricGroup).filter(MetricGroup.id == node_group.metric_group_id).first()
        if not group:
            raise HTTPException(status_code=400, detail="Metric group not found")

    db_node_group = NodeGroup(
        name=node_group.name,
        description=node_group.description,
        ssh_user=node_group.ssh_user,
        ssh_port=node_group.ssh_port,
        ssh_auth_type=node_group.ssh_auth_type,
        ssh_password=node_group.ssh_password,
        use_jump_host=node_group.use_jump_host,
        jump_host=node_group.jump_host,
        jump_port=node_group.jump_port,
        jump_user=node_group.jump_user,
        jump_auth_type=node_group.jump_auth_type,
        jump_password=node_group.jump_password,
        remote_auth_type=node_group.remote_auth_type,
        remote_key_path=node_group.remote_key_path,
        remote_password=node_group.remote_password,
        metric_config_id=node_group.metric_config_id,
        monitoring_server_id=node_group.monitoring_server_id,
        metric_group_id=node_group.metric_group_id,
    )
    db.add(db_node_group)
    db.commit()
    db.refresh(db_node_group)

    return build_node_group_response(db_node_group)


@router.post("/with-nodes", response_model=NodeGroupDetail, status_code=201)
async def create_node_group_with_nodes(
    data: NodeGroupCreateWithNodes,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a node group with initial nodes and automatically verify connectivity."""
    try:
        # Check for duplicate name
        existing = db.query(NodeGroup).filter(NodeGroup.name == data.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Node group with this name already exists")

        # Verify monitoring server exists if provided
        if data.monitoring_server_id:
            server = db.query(MonitoringServer).filter(MonitoringServer.id == data.monitoring_server_id).first()
            if not server:
                raise HTTPException(status_code=400, detail="Monitoring server not found")

        # Verify metric group exists if provided
        if data.metric_group_id:
            group = db.query(MetricGroup).filter(MetricGroup.id == data.metric_group_id).first()
            if not group:
                raise HTTPException(status_code=400, detail="Metric group not found")

        # Create node group
        db_node_group = NodeGroup(
            name=data.name,
            description=data.description,
            ssh_user=data.ssh_user,
            ssh_port=data.ssh_port,
            ssh_auth_type=data.ssh_auth_type,
            ssh_password=data.ssh_password,
            use_jump_host=data.use_jump_host,
            jump_host=data.jump_host,
            jump_port=data.jump_port,
            jump_user=data.jump_user,
            jump_auth_type=data.jump_auth_type,
            jump_password=data.jump_password,
            remote_auth_type=data.remote_auth_type,
            remote_key_path=data.remote_key_path,
            remote_password=data.remote_password,
            metric_config_id=data.metric_config_id,
            monitoring_server_id=data.monitoring_server_id,
            metric_group_id=data.metric_group_id,
        )
        db.add(db_node_group)
        db.commit()
        db.refresh(db_node_group)

        # Add nodes
        nodes = []
        for ip in data.ip_addresses:
            node = Node(
                node_group_id=db_node_group.id,
                ip_address=ip.strip(),
                status=DBNodeStatus.PENDING,
            )
            db.add(node)
            nodes.append(node)

        db.commit()
        db.refresh(db_node_group)

        # Automatically start connectivity verification if credentials are configured
        node_ids = [n.id for n in db_node_group.nodes]
        has_credentials = False

        if db_node_group.use_jump_host:
            has_jump_creds = (
                db_node_group.jump_auth_type == "password" and db_node_group.jump_password
            ) or db_node_group.jump_key_path
            has_remote_creds = (
                db_node_group.remote_auth_type == "password" and db_node_group.remote_password
            ) or db_node_group.remote_key_path
            has_credentials = has_jump_creds and has_remote_creds
        else:
            has_credentials = (
                db_node_group.ssh_auth_type == "password" and db_node_group.ssh_password
            ) or db_node_group.ssh_key_path

        if has_credentials and node_ids:
            job_id = str(uuid4())
            logger.info(f"Auto-starting connectivity verification for node group {db_node_group.name} (job {job_id})")
            background_tasks.add_task(
                run_verification,
                job_id=job_id,
                node_group_id=db_node_group.id,
                node_ids=node_ids,
            )

        response = build_node_group_response(db_node_group)
        return NodeGroupDetail(
            **response.model_dump(),
            nodes=db_node_group.nodes,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating node group: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create node group: {str(e)}")


@router.get("/{node_group_id}", response_model=NodeGroupDetail)
def get_node_group(node_group_id: int, db: Session = Depends(get_db)):
    """Get detailed info for a node group."""
    ng = get_node_group_or_404(db, node_group_id)
    response = build_node_group_response(ng)
    return NodeGroupDetail(
        **response.model_dump(),
        nodes=ng.nodes,
        metric_config=ng.metric_config,
    )


@router.patch("/{node_group_id}", response_model=NodeGroupResponse)
def update_node_group(
    node_group_id: int,
    update: NodeGroupUpdate,
    db: Session = Depends(get_db),
):
    """Update a node group."""
    ng = get_node_group_or_404(db, node_group_id)

    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(ng, key, value)

    db.commit()
    db.refresh(ng)

    return build_node_group_response(ng)


@router.delete("/{node_group_id}", status_code=204)
async def delete_node_group(node_group_id: int, db: Session = Depends(get_db)):
    """Delete a node group and all its nodes."""
    ng = get_node_group_or_404(db, node_group_id)

    # Remove Prometheus targets
    prometheus_config.remove_node_group_targets(ng.name)

    # Remove Grafana dashboard
    await grafana.remove_node_group_dashboard(ng.name)

    # Remove SSH key
    credential_store.delete_ssh_key(ng.name)

    # Delete from database (cascades to nodes)
    db.delete(ng)
    db.commit()


@router.post("/{node_group_id}/ssh-key", response_model=SSHKeyUpload)
async def upload_ssh_key(
    node_group_id: int,
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH private key for a node group (direct connection to GPU nodes)."""
    ng = get_node_group_or_404(db, node_group_id)

    # Read key content
    content = await key_file.read()
    key_content = content.decode("utf-8")

    # Validate it looks like an SSH key
    if "PRIVATE KEY" not in key_content:
        raise HTTPException(status_code=400, detail="Invalid SSH private key format")

    # Store the key
    key_path = credential_store.store_ssh_key(ng.name, key_content)

    # Update node group with key path
    ng.ssh_key_path = key_path
    db.commit()

    return SSHKeyUpload(
        key_path=key_path,
        message="SSH key uploaded successfully",
    )


@router.post("/{node_group_id}/jump-key", response_model=SSHKeyUpload)
async def upload_jump_host_key(
    node_group_id: int,
    key_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload SSH private key for accessing the jump host."""
    ng = get_node_group_or_404(db, node_group_id)

    # Read key content
    content = await key_file.read()
    key_content = content.decode("utf-8")

    # Validate it looks like an SSH key
    if "PRIVATE KEY" not in key_content:
        raise HTTPException(status_code=400, detail="Invalid SSH private key format")

    # Store the key with a different name
    key_path = credential_store.store_ssh_key(f"{ng.name}_jump", key_content)

    # Update node group with jump key path
    ng.jump_key_path = key_path
    db.commit()

    return SSHKeyUpload(
        key_path=key_path,
        message="Jump host SSH key uploaded successfully",
    )


@router.post("/{node_group_id}/verify", response_model=InstallationResponse)
async def verify_connectivity(
    node_group_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Verify SSH connectivity to nodes in the node group.

    Uses parallel SSH connections to verify multiple nodes simultaneously
    (default concurrency: 10).

    This will test SSH connectivity to each node and update their status to
    CONNECTED (if reachable) or UNREACHABLE (if not). Nodes must be CONNECTED
    before installation can proceed.
    """
    ng = get_node_group_or_404(db, node_group_id)

    # Check that we have credentials configured
    if ng.use_jump_host:
        has_jump_creds = (ng.jump_auth_type == "password" and ng.jump_password) or ng.jump_key_path
        has_remote_creds = (ng.remote_auth_type == "password" and ng.remote_password) or ng.remote_key_path
        if not has_jump_creds:
            raise HTTPException(status_code=400, detail="Jump host credentials not configured")
        if not has_remote_creds:
            raise HTTPException(status_code=400, detail="GPU node credentials (from jump host) not configured")
    else:
        has_direct_creds = (ng.ssh_auth_type == "password" and ng.ssh_password) or ng.ssh_key_path
        if not has_direct_creds:
            raise HTTPException(status_code=400, detail="SSH credentials not configured for this node group")

    nodes = ng.nodes
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes in this node group")

    job_id = str(uuid4())
    node_ids = [n.id for n in nodes]

    # Start background verification
    background_tasks.add_task(
        run_verification,
        job_id=job_id,
        node_group_id=ng.id,
        node_ids=node_ids,
    )

    statuses = [
        InstallationStatus(
            node_id=n.id,
            ip_address=n.ip_address,
            status="verifying",
            message="Connectivity verification in progress",
        )
        for n in nodes
    ]

    return InstallationResponse(
        job_id=job_id,
        total_nodes=len(nodes),
        statuses=statuses,
    )


# Default concurrency for parallel SSH operations
DEFAULT_PARALLEL_LIMIT = 10


async def verify_single_node(
    node_id: int,
    node_ip: str,
    node_group_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Verify connectivity to a single node. Returns result dict."""
    from ...models.database import SessionLocal

    async with semaphore:
        db = SessionLocal()
        try:
            node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
            node = db.query(Node).filter(Node.id == node_id).first()

            if not node_group or not node:
                return {"node_id": node_id, "success": False, "error": "Not found"}

            logger.info(f"[Parallel] Verifying connectivity to {node_ip}")

            # Build SSH manager
            jump_config = None
            if node_group.use_jump_host:
                jump_config = JumpHostConfig(
                    host=node_group.jump_host,
                    port=node_group.jump_port or 22,
                    username=node_group.jump_user or "root",
                    auth_type=node_group.jump_auth_type or "key",
                    key_path=node_group.jump_key_path,
                    password=node_group.jump_password,
                    remote_auth_type=node_group.remote_auth_type or "key",
                    remote_key_path=node_group.remote_key_path,
                    remote_password=node_group.remote_password,
                )

            ssh = SSHManager(
                host=node_ip,
                username=node_group.ssh_user,
                auth_type=node_group.ssh_auth_type,
                key_path=node_group.ssh_key_path,
                password=node_group.ssh_password,
                port=node_group.ssh_port,
                jump_host=jump_config,
            )

            try:
                connected = await ssh.connect()
                if connected:
                    result = await ssh.execute("hostname")
                    if result.success:
                        hostname = result.stdout.strip()
                        node.hostname = hostname
                        node.status = DBNodeStatus.CONNECTED.value
                        node.status_message = f"Connected - hostname: {hostname}"
                        db.commit()
                        logger.info(f"[Parallel] Node {node_ip} connected: {hostname}")
                        return {"node_id": node_id, "success": True, "hostname": hostname}
                    else:
                        node.status = DBNodeStatus.UNREACHABLE.value
                        node.status_message = f"SSH connected but command failed: {result.stderr}"
                        db.commit()
                        return {"node_id": node_id, "success": False, "error": result.stderr}
                else:
                    node.status = DBNodeStatus.UNREACHABLE.value
                    node.status_message = "Failed to establish SSH connection"
                    db.commit()
                    return {"node_id": node_id, "success": False, "error": "Connection failed"}
            finally:
                await ssh.disconnect()

        except Exception as e:
            logger.exception(f"[Parallel] Verification failed for {node_ip}: {e}")
            try:
                node = db.query(Node).filter(Node.id == node_id).first()
                if node:
                    node.status = DBNodeStatus.UNREACHABLE.value
                    node.status_message = str(e)[:500]
                    db.commit()
            except Exception:
                db.rollback()
            return {"node_id": node_id, "success": False, "error": str(e)[:500]}
        finally:
            db.close()


async def run_verification(job_id: str, node_group_id: int, node_ids: List[int]):
    """Background task to verify SSH connectivity to nodes in parallel."""
    from ...models.database import SessionLocal

    db = SessionLocal()
    try:
        node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
        if not node_group:
            logger.error(f"Node group {node_group_id} not found for verification job {job_id}")
            return

        nodes = db.query(Node).filter(Node.id.in_(node_ids)).all()
        if not nodes:
            logger.error(f"No nodes found for verification job {job_id}")
            return

        logger.info(
            f"[Parallel] Starting verification for {len(nodes)} nodes with concurrency={DEFAULT_PARALLEL_LIMIT}"
        )

        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(DEFAULT_PARALLEL_LIMIT)

        # Create tasks for parallel verification
        tasks = [
            verify_single_node(
                node_id=node.id,
                node_ip=node.ip_address,
                node_group_id=node_group_id,
                semaphore=semaphore,
            )
            for node in nodes
        ]

        # Run all verifications in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        connected_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        logger.info(f"[Parallel] Verification job {job_id} completed: {connected_count}/{len(nodes)} nodes connected")

    except Exception as e:
        logger.exception(f"Verification job {job_id} failed: {e}")
    finally:
        db.close()


@router.post("/{node_group_id}/install", response_model=InstallationResponse)
async def install_exporters(
    node_group_id: int,
    request: InstallationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Install monitoring exporters on nodes in the node group.

    Uses parallel SSH connections to install on multiple nodes simultaneously.
    The parallel_limit parameter controls how many concurrent connections are made
    (default: 10, max: 50).
    """
    ng = get_node_group_or_404(db, node_group_id)

    # Check that we have credentials configured
    if ng.use_jump_host:
        # For jump host: need jump host credentials and remote credentials
        has_jump_creds = (ng.jump_auth_type == "password" and ng.jump_password) or ng.jump_key_path
        has_remote_creds = (ng.remote_auth_type == "password" and ng.remote_password) or ng.remote_key_path
        if not has_jump_creds:
            raise HTTPException(status_code=400, detail="Jump host credentials not configured")
        if not has_remote_creds:
            raise HTTPException(status_code=400, detail="GPU node credentials (from jump host) not configured")
    else:
        # For direct connection: need SSH key or password
        has_direct_creds = (ng.ssh_auth_type == "password" and ng.ssh_password) or ng.ssh_key_path
        if not has_direct_creds:
            raise HTTPException(status_code=400, detail="SSH credentials not configured for this node group")

    # Get nodes to install on
    if request.node_ids:
        nodes = [n for n in ng.nodes if n.id in request.node_ids]
    else:
        # Install on connected nodes by default (or all connected if force)
        if request.force:
            # Force mode: install on all connected/pending/error nodes (not unreachable)
            nodes = [n for n in ng.nodes if n.status != DBNodeStatus.UNREACHABLE]
        else:
            # Normal mode: only install on CONNECTED nodes
            nodes = [n for n in ng.nodes if n.status == DBNodeStatus.CONNECTED.value]

    if not nodes:
        # Check why no nodes are available
        all_statuses = {n.status for n in ng.nodes}
        if DBNodeStatus.PENDING.value in all_statuses:
            raise HTTPException(
                status_code=400, detail="No connected nodes. Please verify connectivity first using POST /verify"
            )
        elif DBNodeStatus.UNREACHABLE.value in all_statuses:
            raise HTTPException(status_code=400, detail="All nodes are unreachable. Check SSH credentials and network.")
        else:
            raise HTTPException(status_code=400, detail="No nodes available for installation")

    job_id = str(uuid4())

    # Collect node IDs and node_group_id for the background task
    # (don't pass ORM objects across task boundaries)
    node_ids = [n.id for n in nodes]
    node_group_id_for_task = ng.id

    # Get parallelism level from request
    parallel_limit = request.parallel_limit if request.parallel_limit else DEFAULT_PARALLEL_LIMIT

    # Start background installation
    background_tasks.add_task(
        run_installation,
        job_id=job_id,
        node_group_id=node_group_id_for_task,
        node_ids=node_ids,
        parallel_limit=parallel_limit,
    )

    statuses = [
        InstallationStatus(
            node_id=n.id,
            ip_address=n.ip_address,
            status="queued",
            message="Installation queued",
        )
        for n in nodes
    ]

    return InstallationResponse(
        job_id=job_id,
        total_nodes=len(nodes),
        statuses=statuses,
    )


async def install_single_node(
    node_id: int,
    node_ip: str,
    node_hostname: str,
    gpu_exporter_port: int,
    node_exporter_port: int,
    node_group_id: int,
    node_group_name: str,
    loki_url: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Install monitoring components on a single node. Returns result dict."""
    from ...models.database import SessionLocal

    async with semaphore:
        db = SessionLocal()
        try:
            node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
            node = db.query(Node).filter(Node.id == node_id).first()

            if not node_group or not node:
                return {"node_id": node_id, "success": False, "error": "Not found"}

            # Update status to installing
            logger.info(f"[Parallel] Starting installation for node {node_ip}")
            node.status = DBNodeStatus.INSTALLING.value
            node.status_message = "Installation in progress"
            db.commit()

            # Build jump host config if needed
            jump_config = None
            if node_group.use_jump_host and node_group.jump_host:
                jump_config = JumpHostConfig(
                    host=node_group.jump_host,
                    port=node_group.jump_port or 22,
                    username=node_group.jump_user or "root",
                    auth_type=node_group.jump_auth_type or "key",
                    key_path=node_group.jump_key_path,
                    password=node_group.jump_password,
                    remote_auth_type=node_group.remote_auth_type or "key",
                    remote_key_path=node_group.remote_key_path,
                    remote_password=node_group.remote_password,
                )

            # Connect and install
            ssh = SSHManager(
                host=node_ip,
                username=node_group.ssh_user,
                auth_type=node_group.ssh_auth_type or "key",
                key_path=node_group.ssh_key_path if not node_group.use_jump_host else None,
                password=node_group.ssh_password if not node_group.use_jump_host else None,
                port=node_group.ssh_port,
                jump_host=jump_config,
            )

            async with ssh:
                installer = NodeInstaller(
                    ssh_manager=ssh,
                    loki_url=loki_url,
                    node_group_name=node_group_name,
                    hostname=node_hostname or node_ip,
                )

                # Get hostname if not set
                if not node.hostname:
                    success, hostname = await ssh.check_connection()
                    if success:
                        node.hostname = hostname

                # Get GPU info
                gpu_count, gpu_model = await ssh.get_gpu_info()
                node.gpu_count = gpu_count
                node.gpu_model = gpu_model

                # Install all components
                results = await installer.install_all(
                    gpu_port=gpu_exporter_port,
                    node_port=node_exporter_port,
                )

                # Check results - amd_exporter is optional since public images may not exist
                essential_ok = results.get("node_exporter", False) and results.get("promtail", False)
                failed = [k for k, v in results.items() if not v]

                if all(results.values()):
                    node.status = DBNodeStatus.ACTIVE.value
                    node.status_message = "All components installed successfully"
                    logger.info(f"[Parallel] Node {node_ip} installation completed successfully")
                    db.commit()
                    return {"node_id": node_id, "success": True}
                elif essential_ok:
                    node.status = DBNodeStatus.ACTIVE.value
                    node.status_message = f"Essential components installed (optional failed: {', '.join(failed)})"
                    logger.info(f"[Parallel] Node {node_ip} essential components installed, optional failed: {failed}")
                    db.commit()
                    return {"node_id": node_id, "success": True, "warnings": failed}
                else:
                    node.status = DBNodeStatus.ERROR.value
                    node.status_message = f"Failed to install: {', '.join(failed)}"
                    logger.warning(f"[Parallel] Node {node_ip} installation failed: {failed}")
                    db.commit()
                    return {"node_id": node_id, "success": False, "error": f"Failed: {', '.join(failed)}"}

        except Exception as e:
            logger.exception(f"[Parallel] Installation failed for {node_ip}: {e}")
            try:
                node = db.query(Node).filter(Node.id == node_id).first()
                if node:
                    node.status = DBNodeStatus.ERROR.value
                    node.status_message = str(e)[:500]
                    db.commit()
            except Exception:
                db.rollback()
            return {"node_id": node_id, "success": False, "error": str(e)[:500]}
        finally:
            db.close()


async def run_installation(
    job_id: str, node_group_id: int, node_ids: List[int], parallel_limit: int = DEFAULT_PARALLEL_LIMIT
):
    """Background task to run installation on nodes in parallel."""
    import os
    from ...models.database import SessionLocal

    # Create a new database session for this background task
    db = SessionLocal()

    try:
        # Fetch fresh copies of node_group and nodes
        node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
        if not node_group:
            logger.error(f"Node group {node_group_id} not found for installation job {job_id}")
            return

        nodes = db.query(Node).filter(Node.id.in_(node_ids)).all()
        if not nodes:
            logger.error(f"No nodes found for installation job {job_id}")
            return

        node_group_name = node_group.name

        # Get Loki URL from the associated monitoring server, fall back to environment variable
        loki_url = os.environ.get("LOKI_URL", "http://loki:3100")
        if node_group.monitoring_server_id:
            monitoring_server = (
                db.query(MonitoringServer).filter(MonitoringServer.id == node_group.monitoring_server_id).first()
            )
            if monitoring_server and monitoring_server.server_ip:
                loki_url = f"http://{monitoring_server.server_ip}:{monitoring_server.loki_port}"
                logger.info(f"Using monitoring server Loki URL: {loki_url}")
            else:
                logger.info(f"Monitoring server has no IP configured, using default Loki URL: {loki_url}")
        else:
            logger.info(f"No monitoring server associated, using default Loki URL: {loki_url}")
        logger.info(f"[Parallel] Starting installation for {len(nodes)} nodes with concurrency={parallel_limit}")

        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(parallel_limit)

        # Create tasks for parallel installation
        tasks = [
            install_single_node(
                node_id=node.id,
                node_ip=node.ip_address,
                node_hostname=node.hostname,
                gpu_exporter_port=node.gpu_exporter_port,
                node_exporter_port=node.node_exporter_port,
                node_group_id=node_group_id,
                node_group_name=node_group_name,
                loki_url=loki_url,
                semaphore=semaphore,
            )
            for node in nodes
        ]

        # Run all installations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        logger.info(f"[Parallel] Installation batch completed: {success_count}/{len(nodes)} nodes successful")

        # Re-query nodes to get fresh status (each install_single_node uses its own session)
        db.expire_all()
        fresh_nodes = db.query(Node).filter(Node.node_group_id == node_group_id).all()

        # Update Prometheus targets with fresh node data
        active_nodes = [
            {
                "ip": n.ip_address,
                "hostname": n.hostname or n.ip_address,
                "gpu_port": n.gpu_exporter_port,
                "node_port": n.node_exporter_port,
                "status": n.status,
            }
            for n in fresh_nodes
        ]
        logger.info(f"Fresh node statuses: {[(n.ip_address, n.status) for n in fresh_nodes]}")
        prometheus_config.update_node_group_targets(node_group_name, active_nodes)

        # Reload Prometheus to pick up new targets
        await prometheus_config.reload_prometheus()

        # Create Grafana dashboard using the monitoring server's Grafana URL
        grafana_provisioner = grafana  # Use default
        if node_group.monitoring_server_id:
            monitoring_server = (
                db.query(MonitoringServer).filter(MonitoringServer.id == node_group.monitoring_server_id).first()
            )
            if monitoring_server and monitoring_server.server_ip:
                grafana_url = f"http://{monitoring_server.server_ip}:{monitoring_server.grafana_port}"
                grafana_provisioner = GrafanaProvisioner(base_url=grafana_url)
                logger.info(f"Using monitoring server Grafana URL: {grafana_url}")

        await grafana_provisioner.provision_node_group_dashboard(node_group_name)

        logger.info(f"[Parallel] Installation job {job_id} completed for node group {node_group_name}")

    except Exception as e:
        logger.exception(f"Installation job {job_id} failed: {e}")
    finally:
        db.close()


@router.post("/{node_group_id}/refresh-targets")
async def refresh_targets(node_group_id: int, db: Session = Depends(get_db)):
    """Manually refresh Prometheus targets for a node group."""
    ng = get_node_group_or_404(db, node_group_id)

    nodes = [
        {
            "ip": n.ip_address,
            "hostname": n.hostname or n.ip_address,
            "gpu_port": n.gpu_exporter_port,
            "node_port": n.node_exporter_port,
            "status": n.status,
        }
        for n in ng.nodes
    ]

    success = prometheus_config.update_node_group_targets(ng.name, nodes)
    await prometheus_config.reload_prometheus()

    return {"success": success, "node_count": len(nodes)}


@router.post("/{node_group_id}/refresh-gpu-info")
async def refresh_all_gpu_info(
    node_group_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Refresh GPU information for all active nodes in the group."""
    ng = get_node_group_or_404(db, node_group_id)

    # Get active/connected nodes
    active_nodes = [n for n in ng.nodes if n.status in [DBNodeStatus.ACTIVE.value, DBNodeStatus.CONNECTED.value]]

    if not active_nodes:
        raise HTTPException(status_code=400, detail="No active nodes to refresh")

    from uuid import uuid4

    job_id = str(uuid4())

    background_tasks.add_task(
        run_gpu_info_refresh,
        job_id=job_id,
        node_group_id=node_group_id,
        node_ids=[n.id for n in active_nodes],
    )

    return {
        "job_id": job_id,
        "message": f"GPU info refresh started for {len(active_nodes)} nodes",
        "node_count": len(active_nodes),
    }


async def refresh_gpu_info_single_node(
    node_id: int,
    node_ip: str,
    node_group_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Refresh GPU info for a single node. Returns result dict."""
    from ...models.database import SessionLocal

    async with semaphore:
        db = SessionLocal()
        try:
            node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
            node = db.query(Node).filter(Node.id == node_id).first()

            if not node_group or not node:
                return {"node_id": node_id, "success": False, "error": "Not found"}

            # Build jump host config if needed
            jump_config = None
            if node_group.use_jump_host and node_group.jump_host:
                jump_config = JumpHostConfig(
                    host=node_group.jump_host,
                    port=node_group.jump_port or 22,
                    username=node_group.jump_user or "root",
                    auth_type=node_group.jump_auth_type or "key",
                    key_path=node_group.jump_key_path,
                    password=node_group.jump_password,
                    remote_auth_type=node_group.remote_auth_type or "key",
                    remote_key_path=node_group.remote_key_path,
                    remote_password=node_group.remote_password,
                )

            ssh = SSHManager(
                host=node_ip,
                username=node_group.ssh_user,
                auth_type=node_group.ssh_auth_type or "key",
                key_path=node_group.ssh_key_path if not node_group.use_jump_host else None,
                password=node_group.ssh_password if not node_group.use_jump_host else None,
                port=node_group.ssh_port,
                jump_host=jump_config,
            )

            async with ssh:
                gpu_count, gpu_model = await ssh.get_gpu_info()
                node.gpu_count = gpu_count
                node.gpu_model = gpu_model
                db.commit()
                logger.info(f"[Parallel] Updated GPU info for {node_ip}: {gpu_count} x {gpu_model}")
                return {"node_id": node_id, "success": True, "gpu_count": gpu_count, "gpu_model": gpu_model}

        except Exception as e:
            logger.exception(f"[Parallel] Failed to refresh GPU info for {node_ip}: {e}")
            return {"node_id": node_id, "success": False, "error": str(e)[:500]}
        finally:
            db.close()


async def run_gpu_info_refresh(job_id: str, node_group_id: int, node_ids: List[int]):
    """Background task to refresh GPU info on multiple nodes in parallel."""
    from ...models.database import SessionLocal

    db = SessionLocal()
    try:
        node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
        if not node_group:
            logger.error(f"Node group {node_group_id} not found")
            return

        nodes = db.query(Node).filter(Node.id.in_(node_ids)).all()
        if not nodes:
            logger.error(f"No nodes found for GPU info refresh job {job_id}")
            return

        logger.info(f"[Parallel] Starting GPU info refresh for {len(nodes)} nodes")

        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(DEFAULT_PARALLEL_LIMIT)

        # Create tasks for parallel refresh
        tasks = [
            refresh_gpu_info_single_node(
                node_id=node.id,
                node_ip=node.ip_address,
                node_group_id=node_group_id,
                semaphore=semaphore,
            )
            for node in nodes
        ]

        # Run all refreshes in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        logger.info(f"[Parallel] GPU info refresh job {job_id} completed: {success_count}/{len(nodes)} successful")

    except Exception as e:
        logger.exception(f"GPU info refresh job {job_id} failed: {e}")
    finally:
        db.close()
