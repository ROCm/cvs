"""
Package installation API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class PackageInstallRequest(BaseModel):
    package: str  # 'lldp', etc.
    nodes: Optional[List[str]] = None  # If None, install on all nodes


@router.post("/install")
def install_package(request: PackageInstallRequest) -> Dict[str, Any]:
    """
    Install a package on cluster nodes.

    Supported packages:
    - lldp: LLDP daemon for network topology discovery
    """
    try:
        from app.main import app_state

        # Get SSH manager from app state
        ssh_manager = app_state.ssh_manager

        if not ssh_manager:
            raise HTTPException(status_code=503, detail="SSH manager not initialized")

        logger.info(f"Package installation requested: {request.package}")

        # Get the appropriate installer
        if request.package.lower() == 'lldp':
            from app.installers.lldp_installer import LLDPInstaller

            installer = LLDPInstaller(ssh_manager)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported package: {request.package}. Supported: lldp")

        # Run installation (synchronous)
        result = installer.install_package()

        return {
            "success": True,
            "message": f"Installation complete: {result['successful']} successful, {result['failed']} failed",
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Package installation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Package installation failed: {str(e)}")


@router.get("/status/{package}")
def get_package_status(package: str) -> Dict[str, Any]:
    """
    Check installation status of a package across all nodes.

    Args:
        package: Package name ('lldp', etc.)
    """
    try:
        from app.main import app_state

        # Get SSH manager
        ssh_manager = app_state.ssh_manager

        if not ssh_manager:
            # Return empty status instead of error
            return {
                "package": package,
                "total_nodes": 0,
                "installed_count": 0,
                "not_installed_count": 0,
                "installed_nodes": [],
                "not_installed_nodes": [],
                "status_by_node": {},
                "note": "SSH not configured yet. Save configuration first.",
            }

        logger.info(f"Checking package status: {package}")

        # Get the appropriate installer
        if package.lower() == 'lldp':
            from app.installers.lldp_installer import LLDPInstaller

            installer = LLDPInstaller(ssh_manager)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported package: {package}. Supported: lldp")

        # Check installation status (synchronous)
        installed_map = installer.check_installation()

        installed_nodes = [node for node, installed in installed_map.items() if installed]
        not_installed_nodes = [node for node, installed in installed_map.items() if not installed]

        return {
            "package": package,
            "total_nodes": len(installed_map),
            "installed_count": len(installed_nodes),
            "not_installed_count": len(not_installed_nodes),
            "installed_nodes": installed_nodes,
            "not_installed_nodes": not_installed_nodes,
            "status_by_node": installed_map,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check package status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check package status: {str(e)}")


@router.get("/list")
async def list_supported_packages() -> Dict[str, Any]:
    """
    List all supported packages that can be installed.
    """
    return {
        "packages": [
            {
                "id": "lldp",
                "name": "LLDP Daemon",
                "description": "Link Layer Discovery Protocol daemon for network topology discovery",
                "package_name": "lldpd",
                "check_command": "lldpcli",
            },
            # Add more packages here as we implement them
            # {
            #     "id": "nvtop",
            #     "name": "NVTOP",
            #     "description": "GPU monitoring tool",
            #     "package_name": "nvtop",
            #     "check_command": "nvtop"
            # },
        ]
    }
