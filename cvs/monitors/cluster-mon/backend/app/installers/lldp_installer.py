"""
LLDP (Link Layer Discovery Protocol) package installer.
"""

import logging
from .base_installer import BaseInstaller

logger = logging.getLogger(__name__)


class LLDPInstaller(BaseInstaller):
    """Installer for LLDP daemon (lldpd)."""

    def get_package_name(self) -> str:
        """Return the LLDP package name."""
        return "lldpd"

    def get_check_command(self) -> str:
        """
        Check if lldpd is installed by checking for lldpcli command.
        """
        return "which lldpcli"

    def get_install_command(self, os_type: str) -> str:
        """
        Get the installation command for LLDP based on OS.

        For RHEL/CentOS systems, we also need to enable and start the service.
        """
        package_name = self.get_package_name()

        # Debian-based systems (Ubuntu, Debian)
        if os_type in ['ubuntu', 'debian']:
            # Install and enable service
            return f"""
            sudo apt-get update && \
            sudo apt-get install -y {package_name} && \
            sudo systemctl enable lldpd && \
            sudo systemctl start lldpd
            """.strip()

        # RHEL-based systems (RHEL, CentOS, Rocky Linux, AlmaLinux)
        elif os_type in ['rhel', 'centos', 'rocky', 'almalinux']:
            return f"""
            sudo yum install -y {package_name} && \
            sudo systemctl enable lldpd && \
            sudo systemctl start lldpd
            """.strip()

        # Fedora (uses dnf)
        elif os_type == 'fedora':
            return f"""
            sudo dnf install -y {package_name} && \
            sudo systemctl enable lldpd && \
            sudo systemctl start lldpd
            """.strip()

        else:
            return None
