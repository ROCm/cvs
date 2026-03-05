# Package Installers

This directory contains package installers for cluster nodes.

## Architecture

### Base Installer (`base_installer.py`)
- Abstract base class for all installers
- Handles OS detection (Ubuntu, Debian, RHEL, CentOS, Rocky, Fedora, etc.)
- Checks if package is already installed
- Groups nodes by OS type
- Executes appropriate install commands for each OS
- Returns detailed installation results

### LLDP Installer (`lldp_installer.py`)
- Installs `lldpd` package for Link Layer Discovery Protocol
- Enables and starts the lldpd service
- Checks for `lldpcli` command to verify installation

## How It Works

1. **OS Detection**: Reads `/etc/os-release` to determine OS type
2. **Installation Check**: Runs `which <command>` to see if already installed
3. **Package Installation**:
   - Debian/Ubuntu: `apt-get install -y <package>`
   - RHEL/CentOS/Rocky: `yum install -y <package>`
   - Fedora: `dnf install -y <package>`
4. **Service Management**: Enables and starts the service (for LLDP)

## API Endpoints

### POST `/api/packages/install`
Install a package on all reachable nodes.

Request body:
```json
{
  "package": "lldp"
}
```

Response:
```json
{
  "success": true,
  "message": "Installation complete: 10 successful, 0 failed",
  "package": "lldpd",
  "total_nodes": 10,
  "successful": 10,
  "failed": 0,
  "already_installed": 0,
  "unsupported_os": 0,
  "results": {
    "node1": {
      "success": true,
      "os_type": "ubuntu",
      "output": "..."
    }
  }
}
```

### GET `/api/packages/status/{package}`
Check installation status of a package.

Example: `/api/packages/status/lldp`

Response:
```json
{
  "package": "lldp",
  "total_nodes": 10,
  "installed_count": 8,
  "not_installed_count": 2,
  "installed_nodes": ["node1", "node2", ...],
  "not_installed_nodes": ["node3", "node4"],
  "status_by_node": {
    "node1": true,
    "node2": true,
    "node3": false
  }
}
```

### GET `/api/packages/list`
List all supported packages.

Response:
```json
{
  "packages": [
    {
      "id": "lldp",
      "name": "LLDP Daemon",
      "description": "Link Layer Discovery Protocol daemon for network topology discovery",
      "package_name": "lldpd",
      "check_command": "lldpcli"
    }
  ]
}
```

## Adding New Installers

To add a new package installer:

1. Create a new file (e.g., `nvtop_installer.py`)
2. Extend `BaseInstaller` class
3. Implement required methods:
   - `get_package_name()`: Return package name
   - `get_check_command()`: Return command to check if installed
   - (Optional) Override `get_install_command()` for custom installation logic

Example:
```python
from .base_installer import BaseInstaller

class NVTOPInstaller(BaseInstaller):
    def get_package_name(self) -> str:
        return "nvtop"

    def get_check_command(self) -> str:
        return "which nvtop"
```

4. Register in `packages.py` API:
```python
elif request.package.lower() == 'nvtop':
    from app.installers.nvtop_installer import NVTOPInstaller
    installer = NVTOPInstaller(ssh_manager)
```

## Design Notes

- **No asyncio**: Uses synchronous SSH execution to avoid asyncio issues
- **Parallel execution**: Uses CVS parallel SSH library for concurrent installation
- **Error handling**: Gracefully handles unreachable nodes, unsupported OS, and installation failures
- **Idempotent**: Checks if package is already installed before attempting installation
- **Detailed logging**: Provides comprehensive logs for debugging
