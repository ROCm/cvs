"""
API router for unified node group configuration.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.node_groups import (
    NodeGroupsSettings,
    GroupSSHConfig,
    GroupJumpHost,
    NodeGroup,
    load_node_groups,
    save_node_groups,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class GroupSSHRequest(BaseModel):
    username: str = "root"
    auth_method: str = "key"
    key_file: str = "~/.ssh/id_rsa"
    timeout: int = 30
    password: Optional[str] = None


class GroupJumpHostRequest(BaseModel):
    enabled: bool = False
    host: str = ""
    username: str = "root"
    auth_method: str = "key"
    key_file: str = "~/.ssh/id_rsa"
    password: Optional[str] = None


class NodeGroupRequest(BaseModel):
    hosts: List[str] = []
    ssh: GroupSSHRequest = GroupSSHRequest()
    jump_host: GroupJumpHostRequest = GroupJumpHostRequest()


class NodeGroupsConfigRequest(BaseModel):
    gpu_nodes: NodeGroupRequest = NodeGroupRequest()
    scale_up_switches: NodeGroupRequest = NodeGroupRequest()
    scale_out_switches: NodeGroupRequest = NodeGroupRequest()
    poll_interval: int = 300


def _get_app_state(request: Request) -> Any:
    from app.main import app_state

    return app_state


def _ensure_passwords(state: Any) -> None:
    if not hasattr(state, "node_groups_passwords"):
        state.node_groups_passwords = {}
    if not hasattr(state, "node_groups_jump_passwords"):
        state.node_groups_jump_passwords = {}


def _build_node_group(b: NodeGroupRequest) -> NodeGroup:
    return NodeGroup(
        hosts=b.hosts,
        ssh=GroupSSHConfig(
            username=b.ssh.username,
            auth_method=b.ssh.auth_method if b.ssh.auth_method in ("key", "password") else "key",
            key_file=b.ssh.key_file or "~/.ssh/id_rsa",
            timeout=b.ssh.timeout,
        ),
        jump_host=GroupJumpHost(
            enabled=b.jump_host.enabled,
            host=b.jump_host.host or "",
            username=b.jump_host.username or "root",
            auth_method=b.jump_host.auth_method if b.jump_host.auth_method in ("key", "password") else "key",
            key_file=b.jump_host.key_file or "~/.ssh/id_rsa",
        ),
    )


async def _register_with_daemon(
    group_name: str,
    hosts: List[str],
    username: str,
    auth_method: str,
    key_file: str,
    ssh_password: Optional[str],
    jump_enabled: bool,
    jump_host: str,
    jump_username: str,
    jump_auth_method: str,
    jump_key_file: str,
    jump_password: Optional[str],
    socket_path: Optional[str] = None,
) -> None:
    if not hosts:
        return
    try:
        from app.core import go_collector

        target_socket = socket_path or go_collector.CLUSTER_SOCKET
        key_path = os.path.expanduser(key_file) if auth_method == "key" and key_file else ""
        password = ssh_password if auth_method == "password" else None

        kwargs: dict = {"socket_path": target_socket}
        if jump_enabled and jump_host:
            kwargs["jump_host"] = jump_host
            kwargs["jump_user"] = jump_username
            kwargs["jump_password"] = jump_password if jump_auth_method == "password" else None
            kwargs["jump_key"] = (
                os.path.expanduser(jump_key_file) if jump_auth_method == "key" and jump_key_file else ""
            )

        await asyncio.to_thread(
            go_collector._refresh_nodes_in_daemon,
            hosts,
            username,
            key_path,
            None,
            password,
            group_name,
            **kwargs,
        )
        logger.info(
            f"Registered '{group_name}' with Go daemon ({target_socket}): "
            f"{len(hosts)} hosts, auth={auth_method}, "
            f"password={'yes' if password else 'no'}"
        )
    except Exception as exc:
        logger.warning(f"Could not register '{group_name}' with daemon: {exc}")


@router.get("/config")
async def get_config(request: Request):
    try:
        state = _get_app_state(request)
        settings: NodeGroupsSettings = getattr(state, "node_groups", None) or load_node_groups()
        return {
            "gpu_nodes": {
                "hosts": settings.gpu_nodes.hosts,
                "ssh": settings.gpu_nodes.ssh.model_dump(),
                "jump_host": settings.gpu_nodes.jump_host.model_dump(),
            },
            "scale_up_switches": {
                "hosts": settings.scale_up_switches.hosts,
                "ssh": settings.scale_up_switches.ssh.model_dump(),
                "jump_host": settings.scale_up_switches.jump_host.model_dump(),
            },
            "scale_out_switches": {
                "hosts": settings.scale_out_switches.hosts,
                "ssh": settings.scale_out_switches.ssh.model_dump(),
                "jump_host": settings.scale_out_switches.jump_host.model_dump(),
            },
            "poll_interval": settings.poll_interval,
        }
    except Exception as exc:
        logger.error(f"get_config error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/config")
async def save_config(request: Request, body: NodeGroupsConfigRequest):
    try:
        state = _get_app_state(request)
        _ensure_passwords(state)

        # Load existing settings to use as fallback for empty/null fields.
        # This prevents a race where the frontend saves before its useEffect
        # fetch completes, accidentally clearing switch hosts or passwords.
        existing: NodeGroupsSettings = getattr(state, "node_groups", None) or load_node_groups()

        def _merge_group(incoming: "NodeGroupRequest", existing_group) -> "NodeGroupRequest":
            """If incoming hosts list is empty, keep the existing hosts."""
            if not incoming.hosts and existing_group and existing_group.hosts:
                incoming.hosts = list(existing_group.hosts)
            return incoming

        body.gpu_nodes = _merge_group(body.gpu_nodes, existing.gpu_nodes)
        body.scale_up_switches = _merge_group(body.scale_up_switches, existing.scale_up_switches)
        body.scale_out_switches = _merge_group(body.scale_out_switches, existing.scale_out_switches)

        new_settings = NodeGroupsSettings(
            gpu_nodes=_build_node_group(body.gpu_nodes),
            scale_up_switches=_build_node_group(body.scale_up_switches),
            scale_out_switches=_build_node_group(body.scale_out_switches),
            poll_interval=body.poll_interval,
        )

        save_node_groups(new_settings)
        state.node_groups = new_settings

        for group_name, grp in [
            ("gpu_nodes", body.gpu_nodes),
            ("scale_up_switches", body.scale_up_switches),
            ("scale_out_switches", body.scale_out_switches),
        ]:
            # Only update in-memory password if a non-empty value was sent.
            # null/empty means "unchanged" — keep whatever was saved before.
            if grp.ssh.password:
                state.node_groups_passwords[group_name] = grp.ssh.password
            if grp.jump_host.password:
                state.node_groups_jump_passwords[group_name] = grp.jump_host.password

        if body.gpu_nodes.hosts:
            try:
                config_dir = Path("/app/config") if Path("/app/config").exists() else Path("config")
                nodes_file = config_dir / "nodes.txt"
                with open(nodes_file, "w") as fh:
                    fh.write("# GPU Cluster Nodes — auto-synced from Node Groups\n")
                    for h in body.gpu_nodes.hosts:
                        fh.write(f"{h}\n")
                logger.info(f"Synced {len(body.gpu_nodes.hosts)} GPU nodes to nodes.txt")
            except Exception as exc:
                logger.warning(f"Could not sync GPU nodes to nodes.txt: {exc}")

        # Clear stale cached data so the UI won't show old-host results
        # until the collectors run a fresh collection with the new host list.
        if hasattr(state, "latest_storage_data"):
            state.latest_storage_data = None
        if hasattr(state, "latest_cpu_data"):
            state.latest_cpu_data = None
        if hasattr(state, "latest_rack_data"):
            state.latest_rack_data = None
        if hasattr(state, "latest_metrics"):
            state.latest_metrics = {}

        return {
            "success": True,
            "gpu_nodes_count": len(new_settings.gpu_nodes.hosts),
            "scale_up_count": len(new_settings.scale_up_switches.hosts),
            "scale_out_count": len(new_settings.scale_out_switches.hosts),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"save_config error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Save failed: {exc}")


@router.post("/config/reload")
async def reload_config(request: Request):
    try:
        state = _get_app_state(request)
        _ensure_passwords(state)

        settings: NodeGroupsSettings = getattr(state, "node_groups", None) or load_node_groups()
        state.node_groups = settings

        # Register GPU nodes with the CLUSTER daemon (main socket).
        if settings.gpu_nodes.hosts:
            ssh_pw = state.node_groups_passwords.get("gpu_nodes")
            jump_pw = state.node_groups_jump_passwords.get("gpu_nodes")
            grp = settings.gpu_nodes
            await _register_with_daemon(
                group_name="cluster",
                hosts=grp.hosts,
                username=grp.ssh.username,
                auth_method=grp.ssh.auth_method,
                key_file=grp.ssh.key_file,
                ssh_password=ssh_pw,
                jump_enabled=grp.jump_host.enabled,
                jump_host=grp.jump_host.host,
                jump_username=grp.jump_host.username,
                jump_auth_method=grp.jump_host.auth_method,
                jump_key_file=grp.jump_host.key_file,
                jump_password=jump_pw,
            )
            logger.info(
                f"Registered {len(grp.hosts)} GPU nodes with cluster Go daemon "
                f"(auth={grp.ssh.auth_method}, user={grp.ssh.username}, "
                f"password={'yes' if ssh_pw else 'no'})"
            )

        # Register switch trays with the SWITCH daemon (separate socket, separate pool).
        from app.core.go_collector import SWITCH_SOCKET

        switch_hosts = list(settings.scale_up_switches.hosts) + list(settings.scale_out_switches.hosts)
        if switch_hosts:
            # Use scale_up_switches credentials (same assumed for all switch trays)
            sw = settings.scale_up_switches
            sw_pw = state.node_groups_passwords.get("scale_up_switches")
            sw_jpw = state.node_groups_jump_passwords.get("scale_up_switches")
            await _register_with_daemon(
                group_name="switches",
                hosts=switch_hosts,
                username=sw.ssh.username,
                auth_method=sw.ssh.auth_method,
                key_file=sw.ssh.key_file,
                ssh_password=sw_pw,
                jump_enabled=sw.jump_host.enabled,
                jump_host=sw.jump_host.host,
                jump_username=sw.jump_host.username,
                jump_auth_method=sw.jump_host.auth_method,
                jump_key_file=sw.jump_host.key_file,
                jump_password=sw_jpw,
                socket_path=SWITCH_SOCKET,
            )
            logger.info(
                f"Registered {len(switch_hosts)} switch trays with switch Go daemon "
                f"(auth={sw.ssh.auth_method}, user={sw.ssh.username}, "
                f"password={'yes' if sw_pw else 'no'})"
            )

        # --- Update ssh_manager host list and node health tracking ---
        new_gpu_hosts = list(settings.gpu_nodes.hosts)
        if new_gpu_hosts and state.ssh_manager is not None:
            state.ssh_manager.host_list = new_gpu_hosts
            logger.info(f"Updated ssh_manager.host_list with {len(new_gpu_hosts)} GPU nodes")

        # Purge stale hosts and seed new ones in health tracking
        if new_gpu_hosts:
            old_hosts = set(state.node_health_status.keys())
            new_hosts = set(new_gpu_hosts)
            for h in old_hosts - new_hosts:
                state.node_health_status.pop(h, None)
                state.node_failure_count.pop(h, None)
            for h in new_hosts - old_hosts:
                state.node_health_status[h] = "healthy"
                state.node_failure_count[h] = 0
            logger.info(
                f"Node health tracking: removed {len(old_hosts - new_hosts)} stale hosts, "
                f"added {len(new_hosts - old_hosts)} new hosts"
            )

        # --- Restart all affected collectors ---
        # Restart RackCollector (switch/IFoE) plus GPU-node collectors
        # (CPU, Storage, GPU metrics, NIC) so they pick up the new host list.
        collectors_to_restart = ["rack", "cpu_info", "storage", "gpu_metrics", "nic_metrics"]

        has_hosts = any(
            [
                settings.gpu_nodes.hosts,
                settings.scale_up_switches.hosts,
                settings.scale_out_switches.hosts,
            ]
        )

        if has_hosts:
            try:
                from app.main import _start_collector_task, REGISTERED_COLLECTORS
                from app.collectors.rack_collector import RackCollector

                restarted = []
                for name in collectors_to_restart:
                    existing = state.collector_tasks.get(name)
                    if existing and not existing.done():
                        existing.cancel()
                        try:
                            await asyncio.wait_for(existing, timeout=3.0)
                        except (asyncio.CancelledError, Exception):
                            pass

                    # Find the registered class by name and reinstantiate
                    cls = next((c for c in REGISTERED_COLLECTORS if c.name == name), None)
                    if cls is None and name == "rack":
                        cls = RackCollector
                    if cls is not None:
                        collector = cls()
                        state.collectors[name] = collector
                        state.collector_tasks[name] = _start_collector_task(collector)
                        restarted.append(name)

                logger.info(f"Restarted collectors: {restarted}")
            except Exception as exc:
                logger.error(f"Collector restart failed: {exc}", exc_info=True)
                return {"success": True, "message": f"Saved but collector restart failed: {exc}"}
            return {"success": True, "message": f"Node groups applied, collectors restarted: {restarted}"}

        return {"success": True, "message": "Settings saved — no hosts configured yet"}

    except HTTPException:
        raise
    except asyncio.CancelledError:
        logger.warning("reload_config: CancelledError escaped")
        return {"success": True, "message": "Settings saved; collector restart may be in progress"}
    except Exception as exc:
        logger.error(f"reload_config error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
