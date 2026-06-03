"""Node management endpoints."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..schemas import (
    NodeCreate,
    NodeBulkCreate,
    NodeResponse,
    NodeStatusUpdate,
    NodeStatus,
)
from ...models import get_db, NodeGroup, Node
from ...models.database import NodeStatus as DBNodeStatus
from ...services import PrometheusConfigManager

router = APIRouter(prefix="/nodegroups/{node_group_id}/nodes", tags=["Nodes"])
logger = logging.getLogger(__name__)

prometheus_config = PrometheusConfigManager()


def get_node_group_or_404(db: Session, node_group_id: int) -> NodeGroup:
    """Get node group by ID or raise 404."""
    node_group = db.query(NodeGroup).filter(NodeGroup.id == node_group_id).first()
    if not node_group:
        raise HTTPException(status_code=404, detail="Node group not found")
    return node_group


def get_node_or_404(db: Session, node_group_id: int, node_id: int) -> Node:
    """Get node by ID within a node group or raise 404."""
    node = (
        db.query(Node)
        .filter(
            Node.id == node_id,
            Node.node_group_id == node_group_id,
        )
        .first()
    )
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@router.get("", response_model=List[NodeResponse])
def list_nodes(
    node_group_id: int,
    status: NodeStatus = None,
    db: Session = Depends(get_db),
):
    """List all nodes in a node group."""
    get_node_group_or_404(db, node_group_id)

    query = db.query(Node).filter(Node.node_group_id == node_group_id)
    if status:
        query = query.filter(Node.status == DBNodeStatus(status.value))

    return query.all()


@router.post("", response_model=NodeResponse, status_code=201)
def add_node(
    node_group_id: int,
    node: NodeCreate,
    db: Session = Depends(get_db),
):
    """Add a single node to a node group."""
    # ng = get_node_group_or_404(db, node_group_id)

    # Check for duplicate IP in this node group
    existing = (
        db.query(Node)
        .filter(
            Node.node_group_id == node_group_id,
            Node.ip_address == node.ip_address,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Node with this IP already exists in the group")

    db_node = Node(
        node_group_id=node_group_id,
        ip_address=node.ip_address,
        hostname=node.hostname,
        gpu_exporter_port=node.gpu_exporter_port,
        node_exporter_port=node.node_exporter_port,
        status=DBNodeStatus.PENDING,
    )
    db.add(db_node)
    db.commit()
    db.refresh(db_node)

    return db_node


@router.post("/bulk", response_model=List[NodeResponse], status_code=201)
def add_nodes_bulk(
    node_group_id: int,
    data: NodeBulkCreate,
    db: Session = Depends(get_db),
):
    """Add multiple nodes to a node group."""
    # ng = get_node_group_or_404(db, node_group_id)

    # Get existing IPs
    existing_ips = {n.ip_address for n in db.query(Node.ip_address).filter(Node.node_group_id == node_group_id).all()}

    nodes = []
    skipped = []

    for ip in data.ip_addresses:
        ip = ip.strip()
        if not ip:
            continue

        if ip in existing_ips:
            skipped.append(ip)
            continue

        node = Node(
            node_group_id=node_group_id,
            ip_address=ip,
            gpu_exporter_port=data.gpu_exporter_port,
            node_exporter_port=data.node_exporter_port,
            status=DBNodeStatus.PENDING,
        )
        db.add(node)
        nodes.append(node)
        existing_ips.add(ip)

    db.commit()

    for node in nodes:
        db.refresh(node)

    if skipped:
        logger.info(f"Skipped {len(skipped)} duplicate IPs: {skipped[:5]}...")

    return nodes


@router.get("/{node_id}", response_model=NodeResponse)
def get_node(
    node_group_id: int,
    node_id: int,
    db: Session = Depends(get_db),
):
    """Get details for a specific node."""
    return get_node_or_404(db, node_group_id, node_id)


@router.patch("/{node_id}", response_model=NodeResponse)
def update_node_status(
    node_group_id: int,
    node_id: int,
    update: NodeStatusUpdate,
    db: Session = Depends(get_db),
):
    """Update a node's status."""
    node = get_node_or_404(db, node_group_id, node_id)

    node.status = update.status.value
    if update.status_message:
        node.status_message = update.status_message

    db.commit()
    db.refresh(node)

    # Update Prometheus targets
    ng = get_node_group_or_404(db, node_group_id)
    nodes_data = [
        {
            "ip": n.ip_address,
            "hostname": n.hostname or n.ip_address,
            "gpu_port": n.gpu_exporter_port,
            "node_port": n.node_exporter_port,
            "status": n.status,
        }
        for n in ng.nodes
    ]
    prometheus_config.update_node_group_targets(ng.name, nodes_data)

    return node


@router.post("/{node_id}/refresh-gpu-info")
async def refresh_gpu_info(
    node_group_id: int,
    node_id: int,
    db: Session = Depends(get_db),
):
    """Refresh GPU information for a node via SSH."""
    from ...services.ssh_manager import SSHManager, JumpHostConfig

    node = get_node_or_404(db, node_group_id, node_id)
    ng = get_node_group_or_404(db, node_group_id)

    # Build SSH connection
    jump_config = None
    if ng.use_jump_host and ng.jump_host:
        jump_config = JumpHostConfig(
            host=ng.jump_host,
            port=ng.jump_port or 22,
            username=ng.jump_user or "root",
            auth_type=ng.jump_auth_type or "key",
            key_path=ng.jump_key_path,
            password=ng.jump_password,
            remote_auth_type=ng.remote_auth_type or "key",
            remote_key_path=ng.remote_key_path,
            remote_password=ng.remote_password,
        )

    ssh = SSHManager(
        host=node.ip_address,
        username=ng.ssh_user,
        auth_type=ng.ssh_auth_type or "key",
        key_path=ng.ssh_key_path if not ng.use_jump_host else None,
        password=ng.ssh_password if not ng.use_jump_host else None,
        port=ng.ssh_port,
        jump_host=jump_config,
    )

    try:
        async with ssh:
            gpu_count, gpu_model = await ssh.get_gpu_info()
            node.gpu_count = gpu_count
            node.gpu_model = gpu_model
            db.commit()
            db.refresh(node)

            return {
                "node_id": node.id,
                "ip_address": node.ip_address,
                "gpu_count": gpu_count,
                "gpu_model": gpu_model,
                "message": "GPU info refreshed successfully",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh GPU info: {str(e)}")


@router.delete("/{node_id}", status_code=204)
def delete_node(
    node_group_id: int,
    node_id: int,
    db: Session = Depends(get_db),
):
    """Remove a node from a node group."""
    node = get_node_or_404(db, node_group_id, node_id)

    db.delete(node)
    db.commit()

    # Update Prometheus targets
    ng = get_node_group_or_404(db, node_group_id)
    nodes_data = [
        {
            "ip": n.ip_address,
            "hostname": n.hostname or n.ip_address,
            "gpu_port": n.gpu_exporter_port,
            "node_port": n.node_exporter_port,
            "status": n.status,
        }
        for n in ng.nodes
    ]
    prometheus_config.update_node_group_targets(ng.name, nodes_data)


@router.post("/import", response_model=List[NodeResponse])
async def import_nodes_from_file(
    node_group_id: int,
    db: Session = Depends(get_db),
):
    """
    Import nodes from uploaded file.
    Note: Use /nodegroups/{id}/nodes/bulk endpoint with parsed IP list.
    This endpoint is for documentation purposes.
    """
    raise HTTPException(
        status_code=400, detail="Use POST /nodegroups/{id}/nodes/bulk with JSON body containing ip_addresses array"
    )
