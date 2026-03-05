"""
GPU software information collector.
Collects ROCM version, GPU firmware, driver version, and library versions.
"""

import json
import re
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GPUSoftwareCollector:
    """Collects GPU software and firmware information."""

    @staticmethod
    def parse_json_output(output_dict: Dict[str, str]) -> Dict[str, Any]:
        """Parse JSON output from command execution."""
        parsed = {}
        for host, output in output_dict.items():
            try:
                if output and not output.startswith("ERROR") and not output.startswith("ABORT"):
                    output = output.strip()
                    parsed[host] = json.loads(output)
                else:
                    parsed[host] = {"error": output}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {host}: {e}")
                parsed[host] = {"error": f"JSON parse error: {str(e)}"}
        return parsed

    async def collect_rocm_version(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect ROCM version information.

        Commands:
        - cat /opt/rocm*/.info/version (actual ROCm version)
        - rocm-smi --showdriverversion (driver version)
        """
        logger.info("Collecting ROCM version")

        # Get actual ROCm version from file

        rocm_ver_output = await ssh_manager.exec_async(
            "cat /opt/rocm*/.info/version 2>/dev/null | head -1 || echo 'N/A'"
        )
        driver_output = await ssh_manager.exec_async("rocm-smi --showdriverversion 2>/dev/null || echo 'Not available'")

        rocm_version = {}
        for host in rocm_ver_output.keys():
            rocm_ver = rocm_ver_output.get(host, 'N/A').strip()
            driver_out = driver_output.get(host, '')

            info = {}

            # Add actual ROCm version
            if rocm_ver and rocm_ver != 'N/A' and not rocm_ver.startswith('ERROR'):
                info['rocm_version'] = rocm_ver
            else:
                info['rocm_version'] = 'N/A'

            # Parse driver version
            if "Not available" not in driver_out and not driver_out.startswith("ERROR"):
                version_match = re.search(r"Driver version:\s*(\S+)", driver_out)
                if version_match:
                    info['driver_version'] = version_match.group(1)
                else:
                    info['driver_version'] = driver_out.strip()[:50]
            else:
                info['driver_version'] = 'N/A'

            rocm_version[host] = info

        return rocm_version

    async def collect_gpu_firmware(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect GPU firmware information.

        Command: amd-smi firmware --json
        """
        logger.info("Collecting GPU firmware information")
        output = await ssh_manager.exec_async("amd-smi firmware --json")
        return self.parse_json_output(output)

    async def collect_amd_smi_version(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect AMD SMI version.

        Command: amd-smi version --json
        """
        logger.info("Collecting AMD SMI version")
        output = await ssh_manager.exec_async("amd-smi version --json 2>/dev/null || amd-smi --version")
        return self.parse_json_output(output)

    async def collect_rocm_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect ROCM installation info.

        Command: /opt/rocm/bin/rocminfo
        """
        logger.info("Collecting ROCM info")
        output = await ssh_manager.exec_async("/opt/rocm/bin/rocminfo 2>/dev/null | head -50")

        rocm_info = {}
        for host, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                rocm_info[host] = {"error": out_str}
                continue

            info = {}
            # Parse key information from rocminfo output
            for line in out_str.split("\n"):
                if "Runtime Version" in line:
                    info["runtime_version"] = line.split(":")[-1].strip()
                elif "ROCk module version" in line:
                    info["rock_version"] = line.split(":")[-1].strip()

            rocm_info[host] = info if info else {"raw": "Unable to parse"}

        return rocm_info

    async def collect_driver_version(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect AMD GPU driver version from modinfo.

        Command: modinfo amdgpu | grep version
        """
        logger.info("Collecting GPU driver version")
        output = await ssh_manager.exec_async("modinfo amdgpu 2>/dev/null | grep version | head -5")

        driver_info = {}
        for host, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT"):
                driver_info[host] = {"error": out_str}
                continue

            versions = {}
            for line in out_str.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    versions[key.strip()] = value.strip()

            driver_info[host] = versions

        return driver_info

    async def collect_rocm_libraries(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect ROCM library versions.

        Commands: Check key ROCm libraries
        """
        logger.info("Collecting ROCM library versions")

        # Check for key ROCM libraries
        commands = [
            "dpkg -l | grep rocm | awk '{print $2, $3}' 2>/dev/null || rpm -qa | grep rocm 2>/dev/null || echo 'N/A'",
        ]

        output = await ssh_manager.exec_async(commands[0])

        lib_info = {}
        for host, out_str in output.items():
            if out_str.startswith("ERROR") or out_str.startswith("ABORT") or out_str.strip() == "N/A":
                lib_info[host] = {"libraries": "Not available"}
                continue

            libraries = []
            for line in out_str.split("\n")[:20]:  # Limit to first 20 libraries
                if line.strip():
                    libraries.append(line.strip())

            lib_info[host] = {"libraries": libraries}

        return lib_info

    async def collect_all_software_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all GPU software information.

        OPTIMIZATION: Use minimal commands:
        - amd-smi version --json (for ROCm, AMDSMI, and amdgpu driver versions)
        - amd-smi firmware --json (for firmware versions per GPU)

        amd-smi version --json output format:
        [{
            "tool": "AMDSMI Tool",
            "version": "26.2.0+021c61fc",
            "amdsmi_library_version": "26.2.0",
            "rocm_version": "7.0.2",
            "amdgpu_version": "6.16.6",
            "amd_hsmp_driver_version": "N/A"
        }]

        Returns consolidated software info for all nodes.
        """
        import asyncio

        logger.info("Collecting all GPU software information (optimized)")

        # OPTIMIZATION: amd-smi version --json gives ALL version info!
        results = await asyncio.gather(
            ssh_manager.exec_async("amd-smi version --json"),
            ssh_manager.exec_async("amd-smi firmware --json"),
            return_exceptions=True,
        )

        version_output = results[0] if not isinstance(results[0], Exception) else {}
        firmware_output = results[1] if not isinstance(results[1], Exception) else {}

        # Parse amd-smi version --json output
        rocm_version_info = {}
        for host, out_str in (version_output if isinstance(version_output, dict) else {}).items():
            if not out_str.startswith("ERROR") and not out_str.startswith("ABORT"):
                try:
                    version_data = json.loads(out_str.strip())
                    if isinstance(version_data, list) and len(version_data) > 0:
                        ver_obj = version_data[0]
                        rocm_version_info[host] = {
                            'amdsmi_tool': ver_obj.get('version', 'N/A'),
                            'amdsmi_library': ver_obj.get('amdsmi_library_version', 'N/A'),
                            'rocm_version': ver_obj.get('rocm_version', 'N/A'),
                            'amdgpu_version': ver_obj.get('amdgpu_version', 'N/A'),
                            'amd_hsmp_version': ver_obj.get('amd_hsmp_driver_version', 'N/A'),
                        }
                    else:
                        rocm_version_info[host] = {'rocm_version': 'N/A', 'amdgpu_version': 'N/A'}
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse version JSON from {host}: {e}")
                    rocm_version_info[host] = {'rocm_version': 'N/A', 'amdgpu_version': 'N/A'}
            else:
                rocm_version_info[host] = {'rocm_version': 'N/A', 'amdgpu_version': 'N/A'}

        # Parse firmware
        gpu_firmware = self.parse_json_output(firmware_output) if isinstance(firmware_output, dict) else {}

        software_info = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rocm_version": rocm_version_info,
            "gpu_firmware": gpu_firmware,
        }

        return software_info
