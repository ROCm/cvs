"""
Software information API endpoints for GPU and NIC.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time

router = APIRouter()


@router.get("/gpu")
async def get_gpu_software_info() -> Dict[str, Any]:
    """
    Get GPU software information (ROCM, firmware, drivers, libraries).

    Returns cached data (TTL: 180 seconds) since software rarely changes.
    Returns immediately with cached data while refreshing in background if needed.
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    try:
        current_time = time.time()
        cache_age = current_time - app_state.gpu_software_cache_time

        # If cache is valid (less than 180 seconds old), return it immediately
        if cache_age < app_state.software_cache_ttl and app_state.cached_gpu_software:
            return app_state.cached_gpu_software

        # If cache is stale or empty, collect new data
        from app.collectors.gpu_software_collector import GPUSoftwareCollector

        collector = GPUSoftwareCollector()
        software_info = await collector.collect_all_software_info(app_state.ssh_manager)

        # Update cache
        app_state.cached_gpu_software = software_info
        app_state.gpu_software_cache_time = current_time

        return software_info

    except Exception as e:
        # If collection fails but we have cached data, return it
        if app_state.cached_gpu_software:
            return app_state.cached_gpu_software
        raise HTTPException(status_code=500, detail=f"Failed to collect GPU software info: {str(e)}")


@router.get("/nic")
async def get_nic_software_info() -> Dict[str, Any]:
    """
    Get NIC software information (firmware, drivers, statistics).

    Returns cached data (TTL: 180 seconds) since software rarely changes.
    Returns immediately with cached data while refreshing in background if needed.
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    try:
        current_time = time.time()
        cache_age = current_time - app_state.nic_software_cache_time

        # If cache is valid (less than 180 seconds old), return it immediately
        if cache_age < app_state.software_cache_ttl and app_state.cached_nic_software:
            return app_state.cached_nic_software

        # If cache is stale or empty, collect new data
        from app.collectors.nic_software_collector import NICSoftwareCollector

        collector = NICSoftwareCollector()
        software_info = await collector.collect_all_software_info(app_state.ssh_manager)

        # Update cache
        app_state.cached_nic_software = software_info
        app_state.nic_software_cache_time = current_time

        return software_info

    except Exception as e:
        # If collection fails but we have cached data, return it
        if app_state.cached_nic_software:
            return app_state.cached_nic_software
        raise HTTPException(status_code=500, detail=f"Failed to collect NIC software info: {str(e)}")


@router.get("/nic/advanced")
async def get_nic_advanced_info() -> Dict[str, Any]:
    """
    Get advanced NIC information (PCIe, congestion control).

    Returns cached data (TTL: 180 seconds) for instant display.
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    try:
        current_time = time.time()
        cache_age = current_time - app_state.nic_advanced_cache_time

        # If cache is valid (less than 180 seconds old), return it immediately
        if cache_age < app_state.software_cache_ttl and app_state.cached_nic_advanced:
            return app_state.cached_nic_advanced

        # If cache is stale or empty, collect new data
        from app.collectors.nic_advanced_collector import NICAdvancedCollector

        collector = NICAdvancedCollector()
        advanced_info = await collector.collect_all_nic_advanced_info(app_state.ssh_manager)

        # Update cache
        app_state.cached_nic_advanced = advanced_info
        app_state.nic_advanced_cache_time = current_time

        return advanced_info

    except Exception as e:
        # If collection fails but we have cached data, return it
        if app_state.cached_nic_advanced:
            return app_state.cached_nic_advanced
        raise HTTPException(status_code=500, detail=f"Failed to collect NIC advanced info: {str(e)}")


@router.get("/nic/devlink")
async def get_nic_devlink_info() -> Dict[str, Any]:
    """
    Get NIC device information from devlink.

    Returns firmware versions, drivers, serial numbers for all NICs.
    Supports AMD AINIC, NVIDIA CX7, Broadcom Thor2.
    """
    from app.main import app_state

    if not app_state.ssh_manager:
        raise HTTPException(status_code=503, detail="SSH manager not initialized")

    try:
        from app.collectors.nic_devlink_collector import NICDevlinkCollector

        collector = NICDevlinkCollector()
        devlink_info = await collector.collect_all_devlink_info(app_state.ssh_manager)

        return devlink_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to collect NIC devlink info: {str(e)}")
