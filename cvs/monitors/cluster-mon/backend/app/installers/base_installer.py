"""
Base installer class for package installation on cluster nodes.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseInstaller(ABC):
    """Base class for package installers."""

    def __init__(self, ssh_manager):
        """
        Initialize the installer.

        Args:
            ssh_manager: SSH manager instance for executing commands on nodes
        """
        self.ssh_manager = ssh_manager

    @abstractmethod
    def get_package_name(self) -> str:
        """Return the package name to install."""
        pass

    @abstractmethod
    def get_check_command(self) -> str:
        """Return command to check if package is already installed."""
        pass

    def detect_os(self) -> Dict[str, str]:
        """
        Detect OS type on each node.

        Returns:
            Dict mapping node -> os_type ('ubuntu', 'debian', 'rhel', 'centos', 'rocky', 'unknown')
        """
        nodes = self.ssh_manager.get_reachable_hosts()
        logger.info(f"Detecting OS on {len(nodes)} nodes...")

        # Command to detect OS
        detect_cmd = """
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "$ID"
elif [ -f /etc/redhat-release ]; then
    echo "rhel"
else
    echo "unknown"
fi
"""

        results = self.ssh_manager.exec(detect_cmd)

        os_map = {}
        for node in nodes:
            if node in results and results[node]:
                os_type = results[node].strip().lower()
                os_map[node] = os_type
                logger.info(f"Node {node}: detected OS = {os_type}")
            else:
                os_map[node] = 'unknown'
                logger.warning(f"Node {node}: could not detect OS")

        return os_map

    def check_installation(self) -> Dict[str, bool]:
        """
        Check if package is already installed on nodes.

        Returns:
            Dict mapping node -> is_installed (True/False)
        """
        nodes = self.ssh_manager.get_reachable_hosts()
        logger.info(f"Checking if {self.get_package_name()} is installed on {len(nodes)} nodes...")

        check_cmd = self.get_check_command()
        results = self.ssh_manager.exec(check_cmd)

        installed_map = {}
        for node in nodes:
            # If command succeeds (node in results and got output), package is installed
            installed = node in results and results[node] and results[node].strip() != ""
            installed_map[node] = installed

            if installed:
                logger.info(f"Node {node}: {self.get_package_name()} is already installed")
            else:
                logger.info(f"Node {node}: {self.get_package_name()} is NOT installed")

        return installed_map

    def get_install_command(self, os_type: str) -> Optional[str]:
        """
        Get the installation command for a specific OS.

        Args:
            os_type: OS type (ubuntu, debian, rhel, centos, rocky, etc.)

        Returns:
            Installation command string or None if OS not supported
        """
        package_name = self.get_package_name()

        # Debian-based systems (Ubuntu, Debian)
        if os_type in ['ubuntu', 'debian']:
            return f"sudo apt-get update && sudo apt-get install -y {package_name}"

        # RHEL-based systems (RHEL, CentOS, Rocky Linux, AlmaLinux)
        elif os_type in ['rhel', 'centos', 'rocky', 'almalinux']:
            return f"sudo yum install -y {package_name}"

        # Fedora (uses dnf)
        elif os_type == 'fedora':
            return f"sudo dnf install -y {package_name}"

        else:
            logger.warning(f"Unsupported OS type: {os_type}")
            return None

    def install_package(self) -> Dict[str, Any]:
        """
        Install package on all reachable nodes.

        Returns:
            Dict with installation results
        """
        nodes = self.ssh_manager.get_reachable_hosts()
        logger.info(f"Starting installation of {self.get_package_name()} on {len(nodes)} nodes...")

        # Step 1: Detect OS on all nodes
        os_map = self.detect_os()

        # Step 2: Check which nodes already have the package
        installed_map = self.check_installation()

        # Step 3: Group nodes by OS type and filter out already installed
        nodes_by_os = {}
        already_installed = []
        unsupported_os = []

        for node in nodes:
            os_type = os_map.get(node, 'unknown')

            # Skip if already installed
            if installed_map.get(node, False):
                already_installed.append(node)
                continue

            # Check if OS is supported
            install_cmd = self.get_install_command(os_type)
            if install_cmd is None:
                unsupported_os.append({"node": node, "os": os_type})
                continue

            # Group by OS type
            if os_type not in nodes_by_os:
                nodes_by_os[os_type] = []
            nodes_by_os[os_type].append(node)

        # Step 4: Install on each group
        installation_results = {}

        for os_type, os_nodes in nodes_by_os.items():
            install_cmd = self.get_install_command(os_type)
            logger.info(f"Installing on {len(os_nodes)} {os_type} nodes: {install_cmd}")

            # Create temporary SSH manager for just these nodes
            from app.core.parallel_ssh import ParallelSSHManager

            temp_manager = ParallelSSHManager(
                host_list=os_nodes,
                user=self.ssh_manager.pssh.user,
                password=getattr(self.ssh_manager.pssh, 'password', None),
                pkey=getattr(self.ssh_manager.pssh, 'pkey', '~/.ssh/id_rsa'),
                timeout=300,  # 5 minute timeout for installation
                stop_on_errors=False,
            )

            results = temp_manager.exec(install_cmd, timeout=300)

            for node in os_nodes:
                if node in results:
                    installation_results[node] = {"success": True, "os_type": os_type, "output": results[node]}
                else:
                    installation_results[node] = {
                        "success": False,
                        "os_type": os_type,
                        "error": "Installation command failed or timed out",
                    }

        # Add already installed nodes to results
        for node in already_installed:
            installation_results[node] = {
                "success": True,
                "os_type": os_map.get(node, 'unknown'),
                "already_installed": True,
                "message": "Package already installed",
            }

        # Add unsupported OS nodes to results
        for item in unsupported_os:
            installation_results[item["node"]] = {
                "success": False,
                "os_type": item["os"],
                "error": f"Unsupported OS: {item['os']}",
            }

        # Summary
        successful = sum(1 for r in installation_results.values() if r['success'])
        failed = len(installation_results) - successful

        logger.info(f"Installation complete: {successful} successful, {failed} failed")

        return {
            "package": self.get_package_name(),
            "total_nodes": len(nodes),
            "successful": successful,
            "failed": failed,
            "already_installed": len(already_installed),
            "unsupported_os": len(unsupported_os),
            "results": installation_results,
        }
