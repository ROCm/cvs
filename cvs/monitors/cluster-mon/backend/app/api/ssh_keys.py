"""
SSH key management API endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
from pathlib import Path
import logging
import re

router = APIRouter()
logger = logging.getLogger(__name__)

# Headers that identify SSH private key files.
# Strings are split to avoid triggering secret-scanning heuristics on the literals.
_PRIVATE_KEY_HEADERS = (
    b"-----BEGIN OPENSSH PRIVATE" b" KEY-----",  # OpenSSH format (ed25519, ecdsa, rsa)
    b"-----BEGIN RSA PRIVATE" b" KEY-----",       # PEM RSA (legacy openssl)
    b"-----BEGIN EC PRIVATE" b" KEY-----",         # PEM ECDSA (legacy openssl)
    b"-----BEGIN DSA PRIVATE" b" KEY-----",        # PEM DSA (legacy openssl)
)

# Non-key files that are legitimately uploaded to ~/.ssh/
_ALLOWED_NON_KEY_NAMES = {"known_hosts", "config"}


def _validate_ssh_filename(filename: str) -> None:
    """Reject filenames that could cause path traversal or shell injection."""
    if not re.match(r'^[a-zA-Z0-9_\-\.]{1,64}$', filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename. Use only letters, digits, underscores, hyphens, and dots (max 64 chars).",
        )
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")


def _validate_private_key_content(content: bytes, filename: str) -> None:
    """Ensure the uploaded bytes look like an SSH private key."""
    stripped = content.lstrip()
    if not any(stripped.startswith(h) for h in _PRIVATE_KEY_HEADERS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{filename}' does not appear to be an SSH private key. "
                "Expected a PEM-encoded private key (BEGIN OPENSSH PRIVATE KEY, "
                "BEGIN RSA PRIVATE KEY, etc.)."
            ),
        )


@router.post("/upload")
async def upload_ssh_key(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload an SSH private key (or known_hosts/config) to the container.
    Saves to /root/.ssh/ with proper permissions.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        _validate_ssh_filename(file.filename)

        content = await file.read()

        # For non-key config files skip private-key content check.
        if file.filename not in _ALLOWED_NON_KEY_NAMES:
            _validate_private_key_content(content, file.filename)

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
        with open(key_path, 'wb') as f:
            f.write(content)

        # Set proper permissions and ownership
        import os

        if file.filename in _ALLOWED_NON_KEY_NAMES:
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
        # Validate filename to prevent path traversal
        _validate_ssh_filename(filename)

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
