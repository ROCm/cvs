"""
Backend restart API endpoint.
"""

from fastapi import APIRouter
from typing import Dict, Any
import os
import sys
import asyncio

router = APIRouter()


@router.post("/restart")
async def restart_backend() -> Dict[str, Any]:
    """
    Restart the backend server.

    This triggers a graceful shutdown and restart by executing
    the backend in a new process.
    """

    async def do_restart():
        await asyncio.sleep(1)
        # Restart using os.execv to replace current process
        python = sys.executable
        os.execv(python, [python] + sys.argv)

    asyncio.create_task(do_restart())

    return {
        "success": True,
        "message": "Backend is restarting... Please wait 10 seconds and refresh the page.",
    }
