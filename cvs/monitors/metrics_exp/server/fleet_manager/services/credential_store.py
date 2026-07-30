"""Secure credential storage for SSH keys."""

import os
import stat
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet

SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "/app/ssh_keys")
SECRET_KEY = os.environ.get("SECRET_KEY", "default-secret-key")


class CredentialStore:
    """Manages secure storage of SSH keys."""

    def __init__(self, base_path: str = SSH_KEY_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to 700
        os.chmod(self.base_path, stat.S_IRWXU)

        # Generate Fernet key from secret
        # key = hashlib.sha256(SECRET_KEY.encode()).digest()
        self._fernet = Fernet(Fernet.generate_key())  # For file naming

    def _get_key_filename(self, node_group_name: str) -> str:
        """Generate a safe filename for the SSH key."""
        safe_name = "".join(c if c.isalnum() else "_" for c in node_group_name)
        return f"{safe_name}_id_rsa"

    def store_ssh_key(self, node_group_name: str, key_content: str) -> str:
        """
        Store an SSH private key securely.

        Args:
            node_group_name: Name of the node group
            key_content: The SSH private key content

        Returns:
            Path to the stored key file
        """
        filename = self._get_key_filename(node_group_name)
        key_path = self.base_path / filename

        # Write key with restricted permissions
        with open(key_path, "w") as f:
            f.write(key_content)

        # Set permissions to 600 (read/write for owner only)
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)

        return str(key_path)

    def get_ssh_key_path(self, node_group_name: str) -> Optional[str]:
        """Get the path to an SSH key if it exists."""
        filename = self._get_key_filename(node_group_name)
        key_path = self.base_path / filename

        if key_path.exists():
            return str(key_path)
        return None

    def delete_ssh_key(self, node_group_name: str) -> bool:
        """Delete an SSH key."""
        filename = self._get_key_filename(node_group_name)
        key_path = self.base_path / filename

        if key_path.exists():
            key_path.unlink()
            return True
        return False

    def key_exists(self, node_group_name: str) -> bool:
        """Check if an SSH key exists for the node group."""
        filename = self._get_key_filename(node_group_name)
        key_path = self.base_path / filename
        return key_path.exists()
