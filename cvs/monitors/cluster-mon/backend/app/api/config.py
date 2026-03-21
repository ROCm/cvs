"""
Configuration management API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import yaml
import os
from pathlib import Path

router = APIRouter()


def normalize_ssh_key_path(path: str) -> str:
    """
    Normalize SSH key paths for Docker container environment.
    Converts host paths to container paths.

    - /home/<user>/.ssh/xxx → /root/.ssh/xxx (for jump host connection from container)
    - ~/xxx → /root/.ssh/xxx (for jump host connection from container)
    - Paths starting with /home/<user>@<domain>/.ssh/ are left unchanged (these are ON jump host)
    """
    if not path:
        return path

    # Check if this is a path ON the jump host (contains @domain)
    # e.g., /home/user@domain.com/.ssh/cluster_key
    if "@" in path and "/home/" in path:
        # This is a key ON the jump host - don't modify
        return path

    # Normalize paths for container environment
    # /home/<user>/.ssh/xxx → /root/.ssh/xxx
    if path.startswith("/home/") and "/.ssh/" in path:
        # Extract just the filename
        filename = path.split("/.ssh/")[-1]
        return f"/root/.ssh/{filename}"

    # ~/.ssh/xxx → /root/.ssh/xxx
    if path.startswith("~/"):
        return path.replace("~", "/root")

    # Already correct or relative path
    return path


class JumpHostConfig(BaseModel):
    host: str
    username: str
    auth_method: str  # 'key' or 'password'
    password: Optional[str] = None
    key_file_path: Optional[str] = None  # SSH key to connect to jump host
    node_username: Optional[str] = None  # Username for nodes from jump host
    node_key_file: Optional[str] = None  # SSH key ON jump host to connect to nodes


class ClusterConfigUpdate(BaseModel):
    nodes: List[str]
    username: str
    auth_method: str  # 'key' or 'password'
    password: Optional[str] = None
    key_file_path: Optional[str] = None
    use_jump_host: bool = False
    jump_host: Optional[JumpHostConfig] = None


@router.post("/update")
async def update_configuration(config: ClusterConfigUpdate) -> Dict[str, Any]:
    """
    Update cluster configuration.

    Saves nodes to config/nodes.txt and updates cluster.yaml
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("CONFIGURATION SAVE REQUEST RECEIVED")
    logger.info("=" * 80)
    logger.info(f"Nodes count: {len(config.nodes)}")
    logger.info(f"Nodes list: {config.nodes}")
    logger.info(f"Username: {config.username}")
    logger.info(f"Auth method: {config.auth_method}")
    logger.info(f"Key file path: {config.key_file_path}")
    logger.info(f"Use jump host: {config.use_jump_host}")
    if config.use_jump_host and config.jump_host:
        logger.info(f"Jump host IP: {config.jump_host.host}")
        logger.info(f"Jump host username: {config.jump_host.username}")
        logger.info(f"Jump host key file: {config.jump_host.key_file_path}")
        logger.info(f"Node username via jump: {config.jump_host.node_username}")
        logger.info(f"Node key file on jumphost: {config.jump_host.node_key_file}")
    logger.info("=" * 80)

    try:
        # Validate
        if not config.nodes:
            raise HTTPException(status_code=400, detail="No nodes provided")

        if config.use_jump_host and not config.jump_host:
            raise HTTPException(status_code=400, detail="Jump host enabled but no jump host config provided")

        if config.use_jump_host and config.jump_host and not config.jump_host.host:
            raise HTTPException(status_code=400, detail="Jump host IP/hostname is required")

        # Save nodes to nodes.txt
        # Use environment variable or default path that works in both dev and Docker
        config_dir = Path(os.getenv("CLUSTER_MONITOR_HOME", ".")).parent / "config"
        if not config_dir.exists():
            # Fallback for Docker environment
            config_dir = Path("/app/config")
        if not config_dir.exists():
            # Fallback for development
            config_dir = Path("../config")

        nodes_file = config_dir / "nodes.txt"
        with open(nodes_file, 'w') as f:
            f.write("# GPU Cluster Nodes\n")
            f.write("# Auto-generated from web UI\n")
            f.write("# One IP address or hostname per line\n\n")
            for node in config.nodes:
                f.write(f"{node}\n")

        # Update cluster.yaml
        yaml_file = config_dir / "cluster.yaml"

        # Read existing config
        if yaml_file.exists():
            with open(yaml_file) as f:
                cluster_config = yaml.safe_load(f) or {}
        else:
            cluster_config = {"cluster": {}}

        # Update SSH configuration
        if "cluster" not in cluster_config:
            cluster_config["cluster"] = {}

        # Normalize SSH key path for Docker environment
        ssh_key_file = normalize_ssh_key_path(config.key_file_path or "~/.ssh/id_rsa")

        cluster_config["cluster"]["ssh"] = {
            "username": config.username,
            "key_file": ssh_key_file,
            "timeout": 30,
        }

        # SECURITY: Store password in memory only, never persist to disk
        from app.main import app_state

        if config.auth_method == "password" and config.password:
            app_state.ssh_password = config.password
        else:
            app_state.ssh_password = None

        # Remove password key from YAML if it exists
        if "password" in cluster_config["cluster"]["ssh"]:
            del cluster_config["cluster"]["ssh"]["password"]

        # Update jump host configuration
        if config.use_jump_host and config.jump_host:
            # Normalize jump host key path (for container)
            jump_key_file = normalize_ssh_key_path(config.jump_host.key_file_path or "~/.ssh/id_rsa")

            # Node key file is ALWAYS on the jump host - NEVER normalize it
            # Use default based on node username if not provided
            if config.jump_host.node_key_file:
                node_key_file = config.jump_host.node_key_file
            else:
                # Default: /home/{username}/.ssh/id_rsa on jump host
                node_user = config.jump_host.node_username or config.username
                node_key_file = f"/home/{node_user}/.ssh/id_rsa"

            cluster_config["cluster"]["ssh"]["jump_host"] = {
                "enabled": True,
                "host": config.jump_host.host,
                "username": config.jump_host.username,
                "key_file": jump_key_file,
                "node_username": config.jump_host.node_username or config.username,
                "node_key_file": node_key_file,
            }

            # Handle jump host password
            if config.jump_host.auth_method == "password" and config.jump_host.password:
                # Store in memory
                app_state.jump_host_password = config.jump_host.password
                # Also save to YAML for development/testing (WARNING: Not secure for production)
                cluster_config["cluster"]["ssh"]["jump_host"]["password"] = config.jump_host.password
                logger.warning("⚠️ Jump host password saved to cluster.yaml - NOT RECOMMENDED FOR PRODUCTION")
            else:
                app_state.jump_host_password = None
                # Remove password from YAML if using key-based auth
                if "password" in cluster_config["cluster"]["ssh"].get("jump_host", {}):
                    del cluster_config["cluster"]["ssh"]["jump_host"]["password"]
        else:
            # Disable jump host - ensure the section exists and set enabled to false
            if "jump_host" not in cluster_config["cluster"]["ssh"]:
                cluster_config["cluster"]["ssh"]["jump_host"] = {}
            cluster_config["cluster"]["ssh"]["jump_host"]["enabled"] = False

        # Save YAML
        with open(yaml_file, 'w') as f:
            yaml.dump(cluster_config, f, default_flow_style=False, sort_keys=False)

        logger.info("=" * 80)
        logger.info("CONFIGURATION SAVED TO FILES")
        logger.info("=" * 80)
        logger.info(f"nodes.txt: {nodes_file}")
        logger.info(f"cluster.yaml: {yaml_file}")
        logger.info("")
        logger.info("Saved cluster.yaml content:")
        logger.info(yaml.dump(cluster_config, default_flow_style=False, sort_keys=False))
        logger.info("=" * 80)

        password_note = ""
        if config.auth_method == "password" or (
            config.use_jump_host and config.jump_host and config.jump_host.auth_method == "password"
        ):
            password_note = " Passwords are stored in memory only."

        return {
            "success": True,
            "message": f"Configuration saved successfully! {len(config.nodes)} node(s) configured.{password_note}",
            "nodes_saved": len(config.nodes),
            "jump_host_enabled": config.use_jump_host,
            "security_note": "Passwords are never persisted to disk - stored in memory only" if password_note else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@router.post("/reload")
async def reload_configuration() -> Dict[str, Any]:
    """
    Reload configuration without restarting the backend process.
    Stops metrics collection, closes SSH connections, reloads config, and restarts collection.
    """
    from app.main import reload_configuration

    result = await reload_configuration()

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error during reload"))

    return result


@router.get("/current")
async def get_current_configuration() -> Dict[str, Any]:
    """
    Get current configuration including all SSH and jump host settings.
    """
    from app.core.config import settings

    nodes = settings.load_nodes_from_file()

    return {
        "nodes": nodes,
        "username": settings.ssh_username,
        "auth_method": "password" if settings.ssh_password else "key",
        "key_file": settings.ssh_key_file,
        "jump_host_enabled": settings.jump_host_enabled,
        "jump_host": settings.jump_host if settings.jump_host_enabled else None,
        "jump_host_username": settings.jump_host_username if settings.jump_host_enabled else None,
        "jump_host_key_file": settings.jump_host_key_file if settings.jump_host_enabled else None,
        "node_username_via_jump": settings.node_username_via_jumphost if settings.jump_host_enabled else None,
        "node_key_file_on_jumphost": settings.node_key_file_on_jumphost if settings.jump_host_enabled else None,
    }
