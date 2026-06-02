'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import paramiko
from typing import Callable

from cvs.lib import globals

logger = globals.log


class JumpHostManager:
    """Manages jump host connections and proxy socket creation for ParallelSSHClient."""

    def __init__(
        self, jump_host: str, jump_user: str, jump_password: str = None, jump_pkey: str = None, jump_port: int = 22
    ):
        self.jump_host = jump_host
        self.jump_user = jump_user
        self.jump_password = jump_password
        self.jump_pkey = jump_pkey
        self.jump_port = jump_port
        self._jump_client = None
        self._jump_transport = None

    def connect(self):
        """Establish connection to jump host."""
        logger.info(f"Connecting to jump host: {self.jump_host}")

        self._jump_client = paramiko.SSHClient()
        self._jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            if self.jump_password:
                logger.info("Using password authentication for jump host")
                self._jump_client.connect(
                    hostname=self.jump_host,
                    port=self.jump_port,
                    username=self.jump_user,
                    password=self.jump_password,
                    timeout=30,
                    banner_timeout=60,
                )
            else:
                logger.info(f"Using key authentication for jump host: {self.jump_pkey}")
                self._jump_client.connect(
                    hostname=self.jump_host,
                    port=self.jump_port,
                    username=self.jump_user,
                    key_filename=self.jump_pkey,
                    timeout=30,
                    banner_timeout=60,
                )

            self._jump_transport = self._jump_client.get_transport()
            logger.info(f"Connected to jump host: {self.jump_host}")

        except Exception as e:
            logger.error(f"Failed to connect to jump host: {e}")
            raise

    def create_proxy_func(self) -> Callable:
        """
        Create proxy function for ParallelSSHClient.

        Returns a function that ParallelSSHClient can use to create proxy sockets.
        """
        if not self._jump_transport:
            raise RuntimeError("Must call connect() first")

        def proxy_func(host, port):
            """Proxy function that creates tunneled connection through jump host."""
            logger.debug(f"Creating proxy socket for {host}:{port}")
            try:
                return self._jump_transport.open_channel(
                    "direct-tcpip",
                    (host, port),
                    ("", 0),
                )
            except Exception as e:
                logger.error(f"Failed to create proxy socket for {host}:{port}: {e}")
                raise

        return proxy_func

    def is_connected(self) -> bool:
        """Check if jump host connection is active."""
        if not self._jump_transport:
            return False
        return self._jump_transport.is_active()

    def reconnect(self):
        """Reconnect to jump host if connection is lost."""
        logger.warning("Jump host connection lost - reconnecting...")
        self.disconnect()
        self.connect()

    def ensure_connected(self):
        """Ensure jump host connection is active, reconnect if needed."""
        if not self.is_connected():
            self.reconnect()

    def disconnect(self):
        """Clean up jump host connection."""
        if self._jump_client:
            try:
                self._jump_client.close()
                logger.info("Jump host connection closed")
            except:
                pass
        self._jump_client = None
        self._jump_transport = None
