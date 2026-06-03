"""SSH connection management for remote node operations."""

import asyncio
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import asyncssh

logger = logging.getLogger(__name__)


@dataclass
class SSHResult:
    """Result of an SSH command execution."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class JumpHostConfig:
    """Configuration for SSH jump host."""

    host: str
    port: int = 22
    username: str = "root"
    auth_type: str = "key"  # "key" or "password"
    key_path: Optional[str] = None  # Key to access jump host (on monitoring server)
    password: Optional[str] = None  # Password for jump host
    # For GPU nodes accessed via jump host
    remote_auth_type: str = "key"  # "key" or "password"
    remote_key_path: Optional[str] = None  # Key path ON the jump host for GPU nodes
    remote_password: Optional[str] = None  # Password for GPU nodes


class SSHManager:
    """Manages SSH connections to GPU nodes, with optional jump host support."""

    def __init__(
        self,
        host: str,
        username: str = "root",
        auth_type: str = "key",  # "key" or "password"
        key_path: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 22,
        timeout: int = 30,
        jump_host: Optional[JumpHostConfig] = None,
    ):
        self.host = host
        self.username = username
        self.auth_type = auth_type
        self.key_path = key_path
        self.password = password
        self.port = port
        self.timeout = timeout
        self.jump_host = jump_host
        self._conn: Optional[asyncssh.SSHClientConnection] = None
        self._jump_conn: Optional[asyncssh.SSHClientConnection] = None
        # For executing commands via jump host when key is on jump host
        self._use_jump_exec = False
        self._remote_ssh_cmd: Optional[str] = None

    async def connect(self) -> bool:
        """Establish SSH connection, optionally through jump host."""
        try:
            if self.jump_host:
                return await self._connect_via_jump_host()
            else:
                return await self._connect_direct()

        except asyncssh.Error as e:
            logger.error(f"SSH connection failed to {self.host}: {e}")
            return False
        except asyncio.TimeoutError:
            logger.error(f"SSH connection timed out to {self.host}")
            return False
        except OSError as e:
            logger.error(f"Network error connecting to {self.host}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to {self.host}: {type(e).__name__}: {e}")
            return False

    async def _connect_direct(self) -> bool:
        """Direct SSH connection without jump host."""
        connect_kwargs = {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "known_hosts": None,
            "connect_timeout": self.timeout,
        }

        # Set authentication method
        if self.auth_type == "password" and self.password:
            connect_kwargs["password"] = self.password
        elif self.key_path:
            connect_kwargs["client_keys"] = [self.key_path]

        self._conn = await asyncssh.connect(**connect_kwargs)
        logger.info(f"Connected directly to {self.host}")
        return True

    async def _connect_via_jump_host(self) -> bool:
        """Connect to target host via jump host."""
        jump = self.jump_host

        # First, connect to jump host
        jump_kwargs = {
            "host": jump.host,
            "port": jump.port,
            "username": jump.username,
            "known_hosts": None,
            "connect_timeout": self.timeout,
        }

        # Set authentication for jump host (Step 1)
        if jump.auth_type == "password" and jump.password:
            jump_kwargs["password"] = jump.password
        elif jump.key_path:
            jump_kwargs["client_keys"] = [jump.key_path]

        logger.info(f"Connecting to jump host {jump.host}...")
        self._jump_conn = await asyncssh.connect(**jump_kwargs)
        logger.info(f"Connected to jump host {jump.host}")

        # Connect to GPU node from jump host (Step 2)
        # When using a key that exists ON the jump host, we can't use connect_ssh()
        # directly because it looks for keys locally. Instead, we'll execute commands
        # on the target by running ssh commands through the jump host.

        if jump.remote_auth_type == "password" and jump.remote_password:
            # Use password authentication for GPU node via SSH tunnel
            logger.info(f"Connecting to {self.host} via jump host using password auth")
            tunnel = await self._jump_conn.forward_local_port('', 0, self.host, self.port)
            self._conn = await asyncssh.connect(
                host='localhost',
                port=tunnel.get_port(),
                username=self.username,
                known_hosts=None,
                password=jump.remote_password,
                connect_timeout=self.timeout,
            )
        elif jump.remote_key_path:
            # The key is on the jump host, so we execute commands via the jump host
            # We set _conn to None and use _execute_via_jump for running commands
            logger.info(f"Using jump host to execute commands on {self.host} with key {jump.remote_key_path}")
            self._conn = None
            self._use_jump_exec = True
            self._remote_ssh_cmd = (
                f"ssh -i {jump.remote_key_path} "
                f"-o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null "
                f"-o BatchMode=yes "
                f"-p {self.port} "
                f"{self.username}@{self.host}"
            )
        else:
            # Forward connection through jump host, using local key
            logger.info(f"Connecting to {self.host} via jump host tunnel...")
            tunnel = await self._jump_conn.forward_local_port('', 0, self.host, self.port)

            connect_kwargs = {
                "host": 'localhost',
                "port": tunnel.get_port(),
                "username": self.username,
                "known_hosts": None,
                "connect_timeout": self.timeout,
            }

            if self.key_path:
                connect_kwargs["client_keys"] = [self.key_path]

            self._conn = await asyncssh.connect(**connect_kwargs)

        logger.info(f"Connected to {self.host} via jump host {jump.host}")
        return True

    async def disconnect(self):
        """Close SSH connections."""
        if self._conn:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
            logger.debug(f"Disconnected from {self.host}")

        if self._jump_conn:
            self._jump_conn.close()
            await self._jump_conn.wait_closed()
            self._jump_conn = None
            logger.debug("Disconnected from jump host")

    async def execute(self, command: str, timeout: int = 60) -> SSHResult:
        """Execute a command on the remote host."""
        # Check if we need to connect first
        if not self._conn and not self._use_jump_exec:
            connected = await self.connect()
            if not connected:
                return SSHResult(
                    success=False,
                    stdout="",
                    stderr="Connection failed",
                    exit_code=-1,
                )

        try:
            # If using jump host execution mode (key is on jump host)
            if self._use_jump_exec and self._jump_conn and self._remote_ssh_cmd:
                # Execute command via jump host: ssh target "command"
                # Escape the command for shell
                escaped_cmd = command.replace("'", "'\\''")
                full_cmd = f"{self._remote_ssh_cmd} '{escaped_cmd}'"

                result = await asyncio.wait_for(
                    self._jump_conn.run(full_cmd, check=False),
                    timeout=timeout,
                )

                return SSHResult(
                    success=result.exit_status == 0,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    exit_code=result.exit_status or 0,
                )

            # Normal direct connection
            result = await asyncio.wait_for(
                self._conn.run(command, check=False),
                timeout=timeout,
            )

            return SSHResult(
                success=result.exit_status == 0,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.exit_status or 0,
            )

        except asyncio.TimeoutError:
            logger.error(f"Command timed out on {self.host}: {command[:50]}...")
            return SSHResult(
                success=False,
                stdout="",
                stderr="Command timed out",
                exit_code=-2,
            )
        except Exception as e:
            logger.error(f"Command failed on {self.host}: {e}")
            return SSHResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )

    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to the remote host."""
        if self._use_jump_exec:
            # For jump exec mode, read file and use upload_content
            try:
                with open(local_path, 'r') as f:
                    content = f.read()
                return await self.upload_content(content, remote_path)
            except Exception as e:
                logger.error(f"Failed to read local file {local_path}: {e}")
                return False

        if not self._conn:
            connected = await self.connect()
            if not connected:
                return False

        try:
            await asyncssh.scp(local_path, (self._conn, remote_path))
            logger.debug(f"Uploaded {local_path} to {self.host}:{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to {self.host}: {e}")
            return False

    async def upload_content(self, content: str, remote_path: str, mode: str = "644") -> bool:
        """Upload content as a file to the remote host."""
        if not self._conn and not self._use_jump_exec:
            connected = await self.connect()
            if not connected:
                return False

        try:
            # Use cat with heredoc to write content - this avoids shell escaping issues
            import base64

            encoded = base64.b64encode(content.encode()).decode()
            # Use heredoc with base64 to reliably transfer content without shell escaping issues
            command = f"cat << 'EOFB64' | base64 -d > {remote_path}\n{encoded}\nEOFB64"

            result = await self.execute(command)

            if result.success:
                # Set permissions
                await self.execute(f"chmod {mode} {remote_path}")

            return result.success
        except Exception as e:
            logger.error(f"Failed to upload content to {self.host}: {e}")
            return False

    async def check_connection(self) -> Tuple[bool, str]:
        """Check if we can connect and get basic host info."""
        if not self._conn and not self._use_jump_exec:
            connected = await self.connect()
            if not connected:
                return False, "Connection failed"

        result = await self.execute("hostname")
        if result.success:
            return True, result.stdout.strip()
        return False, result.stderr

    async def get_gpu_info(self) -> Tuple[int, str]:
        """Get GPU count and model from the node."""
        gpu_count = 0
        gpu_model = "Unknown"

        # Try amd-smi list --json first (most accurate for newer AMD GPUs)
        result = await self.execute("amd-smi list --json 2>/dev/null")
        if result.success and result.stdout.strip():
            try:
                import json

                data = json.loads(result.stdout.strip())
                # amd-smi list --json returns a list of GPU objects
                if isinstance(data, list):
                    gpu_count = len(data)
                    if gpu_count > 0 and 'asic' in data[0]:
                        # Get model from first GPU's asic info
                        asic_info = data[0].get('asic', {})
                        market_name = asic_info.get('market_name', '')
                        if market_name:
                            gpu_model = market_name
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.debug(f"Failed to parse amd-smi JSON: {e}")

        # Fallback to rocm-smi if amd-smi didn't work
        if gpu_count == 0:
            result = await self.execute("rocm-smi -i 2>/dev/null | grep -cE '^GPU\\[' || echo 0")
            try:
                gpu_count = int(result.stdout.strip())
            except ValueError:
                pass

        # If still 0, try counting from /sys/class/drm
        if gpu_count == 0:
            result = await self.execute(
                "ls -d /sys/class/drm/card*/device/vendor 2>/dev/null | xargs -I{} cat {} 2>/dev/null | grep -c '0x1002' || echo 0"
            )
            try:
                gpu_count = int(result.stdout.strip())
            except ValueError:
                pass

        # If model still unknown, try rocm-smi
        if gpu_model == "Unknown":
            result = await self.execute(
                "rocm-smi --showproductname 2>/dev/null | grep -i 'card series' | head -1 | sed 's/.*: //' || echo ''"
            )
            if result.success and result.stdout.strip():
                gpu_model = result.stdout.strip()

        return gpu_count, gpu_model

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


async def batch_execute(
    hosts: List[dict],
    command: str,
    max_concurrent: int = 50,
) -> dict:
    """
    Execute a command on multiple hosts concurrently.

    Args:
        hosts: List of dicts with 'ip', 'username', 'key_path', 'port', and optional jump_host config
        command: Command to execute
        max_concurrent: Maximum concurrent connections

    Returns:
        Dict mapping IP to SSHResult
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def execute_on_host(host_info: dict) -> Tuple[str, SSHResult]:
        async with semaphore:
            jump_config = None
            if host_info.get("jump_host"):
                jh = host_info["jump_host"]
                jump_config = JumpHostConfig(
                    host=jh["host"],
                    port=jh.get("port", 22),
                    username=jh.get("username", "root"),
                    key_path=jh.get("key_path"),
                    remote_key_path=jh.get("remote_key_path"),
                )

            ssh = SSHManager(
                host=host_info["ip"],
                username=host_info.get("username", "root"),
                key_path=host_info.get("key_path"),
                port=host_info.get("port", 22),
                jump_host=jump_config,
            )
            try:
                result = await ssh.execute(command)
                return host_info["ip"], result
            finally:
                await ssh.disconnect()

    tasks = [execute_on_host(h) for h in hosts]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed:
        if isinstance(result, Exception):
            logger.error(f"Batch execution error: {result}")
        else:
            ip, ssh_result = result
            results[ip] = ssh_result

    return results
