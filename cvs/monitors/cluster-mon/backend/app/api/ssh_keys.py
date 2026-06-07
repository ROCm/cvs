"""
SSH key management API endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_ssh_key(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload SSH private key to the container.
    Saves to /root/.ssh/ with proper permissions.
    """
    try:
        # Validate file
        if not file.filename:
            logger.error("SSH key upload rejected: no filename provided")
            raise HTTPException(status_code=400, detail="No filename provided")

        # Only allow safe SSH key filenames (alphanumeric, underscore, hyphen, dot)
        import re

        if not re.match(r'^[a-zA-Z0-9._-]+$', file.filename) or '/' in file.filename or '..' in file.filename:
            logger.error(
                f"SSH key upload rejected: invalid filename '{file.filename}'. Use only alphanumeric characters, underscores, hyphens, and dots."
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid key filename. Use only alphanumeric characters, underscores, hyphens, and dots.",
            )

        # Create .ssh directory if it doesn't exist
        ssh_dir = Path("/root/.ssh")
        ssh_dir.mkdir(mode=0o700, exist_ok=True)

        # Ensure directory has correct ownership
        import os

        try:
            os.chown(ssh_dir, 0, 0)  # root:root
            ssh_dir.chmod(0o700)
        except Exception as e:
            logger.warning(f"Could not set .ssh directory ownership: {e}")

        # Save file
        key_path = ssh_dir / file.filename
        content = await file.read()

        with open(key_path, 'wb') as f:
            f.write(content)

        # Set proper permissions and ownership
        import os

        if file.filename in ["known_hosts", "config"]:
            key_path.chmod(0o644)
        else:
            key_path.chmod(0o600)

        # Fix ownership to root (container runs as root)
        try:
            os.chown(key_path, 0, 0)  # uid=0 (root), gid=0 (root)
        except Exception as e:
            logger.warning(f"Could not change ownership: {e}")

        # Verify final permissions
        stat_info = key_path.stat()
        logger.info(f"Key file saved: {key_path}")
        logger.info(f"  Permissions: {oct(stat_info.st_mode)[-3:]}")
        logger.info(f"  Owner: {stat_info.st_uid}:{stat_info.st_gid}")

        logger.info(f"SSH key uploaded: {file.filename} ({len(content)} bytes)")

        return {
            "success": True,
            "message": f"SSH key '{file.filename}' uploaded successfully",
            "filename": file.filename,
            "size": len(content),
            "path": str(key_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload SSH key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload SSH key: {str(e)}")


@router.get("/list")
async def list_ssh_keys() -> Dict[str, Any]:
    """
    List SSH keys currently in the container.
    """
    try:
        ssh_dir = Path("/root/.ssh")

        if not ssh_dir.exists():
            return {"keys": [], "message": "No SSH keys directory found"}

        keys = []
        for key_file in ssh_dir.iterdir():
            if key_file.is_file():
                stat = key_file.stat()
                keys.append(
                    {
                        "filename": key_file.name,
                        "size": stat.st_size,
                        "permissions": oct(stat.st_mode)[-3:],
                    }
                )

        return {"keys": keys, "count": len(keys)}

    except Exception as e:
        logger.error(f"Failed to list SSH keys: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list SSH keys: {str(e)}")


@router.delete("/{filename}")
async def delete_ssh_key(filename: str) -> Dict[str, Any]:
    """
    Delete an SSH key from the container.
    """
    try:
        # Security: only allow deleting SSH key files with safe filenames
        import re

        if not re.match(r'^[a-zA-Z0-9._-]+$', filename) or '/' in filename or '..' in filename:
            logger.error(f"SSH key delete rejected: invalid filename '{filename}'")
            raise HTTPException(status_code=400, detail="Invalid key filename")

        key_path = Path(f"/root/.ssh/{filename}")

        if not key_path.exists():
            raise HTTPException(status_code=404, detail=f"Key '{filename}' not found")

        key_path.unlink()
        logger.info(f"SSH key deleted: {filename}")

        return {"success": True, "message": f"SSH key '{filename}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete SSH key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete SSH key: {str(e)}")
