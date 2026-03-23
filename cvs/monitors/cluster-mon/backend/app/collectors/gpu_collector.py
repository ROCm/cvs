"""
GPU metrics collector using ROCm/AMD SMI commands.
Adapted from CVS rocm_plib.py
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GPUMetricsCollector:
    """Collects GPU metrics via rocm-smi and amd-smi commands."""

    @staticmethod
    def parse_json_output(output_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse JSON output from command execution.

        Args:
            output_dict: Dictionary mapping host -> command output string

        Returns:
            Dictionary mapping host -> parsed JSON data
        """
        parsed = {}
        # logger.info('#========== before parse ===========#')
        # logger.info(output_dict)
        for host, output in output_dict.items():
            try:
                if output and not output.startswith("ERROR") and not output.startswith("ABORT"):
                    # Clean up output (remove any extra whitespace/newlines)
                    output = output.strip()

                    # Remove WARNING lines (common with amd-smi when user not in render/video groups)
                    if output.startswith("WARNING:"):
                        # Find the start of JSON (first '[' or '{')
                        json_start = min(
                            (output.find('[') if '[' in output else len(output)),
                            (output.find('{') if '{' in output else len(output)),
                        )
                        if json_start < len(output):
                            output = output[json_start:].strip()
                            logger.debug(f"Stripped WARNING from {host} output")

                    # Log raw output for debugging
                    logger.debug(f"Raw output from {host} (length={len(output)}): {output[:200]}")
                    if not output:
                        logger.warning(f"Empty output from {host}")
                        parsed[host] = {"error": "Empty output from command"}
                    else:
                        parsed[host] = json.loads(output)
                else:
                    logger.warning(f"Error output from {host}: {output[:200]}")
                    parsed[host] = {"error": output}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from {host}: {e}")
                logger.error(f"Output was: {output[:500]}")
                parsed[host] = {"error": f"JSON parse error: {str(e)}"}

        # logger.info('#=========== parsed value ===============#')
        # logger.info(parsed)
        return parsed

    async def collect_gpu_utilization(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect GPU utilization metrics using amd-smi.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "utilization": {"gfx": ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting GPU utilization")
        # Use amd-smi metric which provides comprehensive GPU metrics
        output = ssh_manager.exec("amd-smi metric --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_gpu_memory(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect GPU memory usage metrics using amd-smi.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "vram": {"total": ..., "used": ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting GPU memory usage")
        output = ssh_manager.exec("amd-smi metric --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_gpu_temperature(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect GPU temperature metrics using amd-smi.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "temperature": {"edge": ..., "hotspot": ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting GPU temperature")
        # amd-smi metric provides temperature in the main metric output
        output = ssh_manager.exec("amd-smi metric --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_gpu_power(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect GPU power consumption metrics using amd-smi.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "power": {"socket_power": "250.0 W", ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting GPU power metrics")
        output = ssh_manager.exec("amd-smi metric --power --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_gpu_metrics(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect comprehensive GPU metrics using amd-smi.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "utilization": {...}, "power": {...}, ...},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting comprehensive GPU metrics")
        output = ssh_manager.exec("amd-smi metric --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_pcie_metrics(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect PCIe metrics and errors.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "pcie": {"bandwidth": {...}, "replay_count": 0, ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting PCIe metrics")
        output = ssh_manager.exec("amd-smi metric --pcie --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_xgmi_metrics(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect XGMI (GPU-to-GPU interconnect) metrics.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "xgmi": {"bandwidth": {...}, "error_count": 0, ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting XGMI metrics")
        output = ssh_manager.exec("amd-smi metric --xgmi-err --json", timeout=120)
        logger.info('%%%%%%%%%%%')
        logger.info('parsed value of xgmi')
        logger.info(output)
        logger.info(self.parse_json_output(output))
        return self.parse_json_output(output)

    async def collect_ras_errors(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect RAS (Reliability, Availability, Serviceability) error metrics.

        Returns:
            {
                "node1": {
                    "gpu_data": [
                        {"gpu": 0, "ecc": {"total_correctable": 0, "total_uncorrectable": 0, ...}},
                        ...
                    ]
                },
                ...
            }
        """
        logger.info("Collecting RAS error metrics")
        output = ssh_manager.exec("amd-smi metric --ecc --json", timeout=120)
        logger.info('%%%%%%%%%%')
        logger.info('Output of ecc')
        logger.info(output)
        return self.parse_json_output(output)

    async def collect_gpu_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect static GPU information (model, firmware, etc.).

        Returns:
            {
                "node1": {
                    "card0": {"Card Model": "AMD Instinct MI300X", ...},
                    ...
                },
                ...
            }
        """
        logger.info("Collecting GPU info")
        output = ssh_manager.exec("rocm-smi --loglevel error --showproductname --json", timeout=120)
        return self.parse_json_output(output)

    async def collect_pcie_info(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect PCIe link information using lspci.
        Gets actual link width and speed for each GPU.

        Returns:
            {
                "node1": {
                    "card0": {"width": "x16", "speed": "32GT/s", "bdf": "0000:05:00.0"},
                    ...
                },
                ...
            }
        """
        logger.info("Collecting PCIe link info via lspci")

        # First get BDF (Bus/Device/Function) addresses from amd-smi
        static_output = ssh_manager.exec("amd-smi static --json", timeout=120)
        static_data = self.parse_json_output(static_output)

        # OPTIMIZATION: Run lspci once per node instead of once per GPU
        # This reduces 288 commands (36 nodes * 8 GPUs) to just 36 commands!
        logger.info("Running lspci once per node (optimized)")
        lspci_output = ssh_manager.exec("bash -c 'sudo lspci -vvv 2>/dev/null'", timeout=120)

        pcie_info = {}
        import re

        for node, data in static_data.items():
            if isinstance(data, dict) and 'gpu_data' in data:
                pcie_info[node] = {}

                # Get the full lspci output for this node
                node_lspci = lspci_output.get(node, '')

                for gpu in data['gpu_data']:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    bus_info = gpu.get('bus', {})
                    bdf = bus_info.get('bdf', '')

                    width = '-'
                    speed = '-'

                    if bdf and node_lspci:
                        # Parse the BDF from full lspci output
                        # Look for section starting with BDF and find LnkSta line within that section
                        lines = node_lspci.split('\n')
                        for i, line in enumerate(lines):
                            if bdf in line:
                                # Look for LnkSta in next ~50 lines (within same device section)
                                for j in range(i + 1, min(i + 50, len(lines))):
                                    if 'LnkSta:' in lines[j]:
                                        width_match = re.search(r'Width (x\d+)', lines[j])
                                        speed_match = re.search(r'Speed ([0-9.]+GT/s)', lines[j])
                                        if width_match:
                                            width = width_match.group(1)
                                        if speed_match:
                                            speed = speed_match.group(1)
                                        break
                                break

                    pcie_info[node][gpu_id] = {
                        'bdf': bdf,
                        'width': width,
                        'speed': speed,
                    }

        return pcie_info

    async def collect_all_metrics(self, ssh_manager) -> Dict[str, Any]:
        """
        Collect all GPU metrics.
        Optimized to call amd-smi metric --json once and parse all data from it.

        Returns:
            {
                "timestamp": "2025-02-11T12:00:00Z",
                "utilization": {...},
                "memory": {...},
                "temperature": {...},
                "power": {...},
                "pcie": {...},
                "xgmi": {...},
                "ras_errors": {...},
                "info": {...}
            }
        """
        import asyncio

        logger.info("Collecting all GPU metrics")

        # OPTIMIZATION: Call amd-smi metric --json ONCE to get ALL data
        # This single command includes: utilization, memory, temperature, PCIe, XGMI, and ECC metrics
        logger.info("Calling amd-smi metric --json for comprehensive GPU data")
        amd_smi_output = await asyncio.to_thread(ssh_manager.exec, "amd-smi metric --json")
        amd_smi_data = self.parse_json_output(amd_smi_output)

        # Parse all metrics from single amd-smi output
        utilization = self._parse_utilization_from_amd_smi(amd_smi_data)
        memory = self._parse_memory_from_amd_smi(amd_smi_data)
        temperature = self._parse_temperature_from_amd_smi(amd_smi_data)

        # Call dedicated commands for PCIe and ECC for cleaner data
        logger.info("Collecting PCIe metrics with dedicated command")
        pcie_output = await asyncio.to_thread(ssh_manager.exec, "amd-smi metric --pcie --json")
        pcie_data = self.parse_json_output(pcie_output)

        logger.info("Collecting XGMI metrics with dedicated command")
        xgmi_output = await asyncio.to_thread(ssh_manager.exec, "amd-smi metric --xgmi-err --json")
        xgmi_data = self.parse_json_output(xgmi_output)

        logger.info("Collecting ECC/RAS metrics with dedicated command")
        ecc_output = await asyncio.to_thread(ssh_manager.exec, "amd-smi metric --ecc --json")
        ecc_data = self.parse_json_output(ecc_output)

        # Parse for frontend display
        pcie_info = self._parse_pcie_metrics_from_amd_smi(pcie_data)

        logger.info(f"Parsed PCIE data: {len(pcie_info)} nodes")
        logger.info(f"ECC data (raw): {len(ecc_data)} nodes")

        # Package results
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "utilization": utilization,
            "memory": memory,
            "temperature": temperature,
            "power": amd_smi_data,  # Power is in the main amd-smi output
            "pcie": pcie_info,  # Parsed PCIE data
            "xgmi": xgmi_data,
            "ras_errors": ecc_data,  # Raw ECC data from dedicated command
            "pcie_link_status": pcie_info,  # For backward compatibility
            "info": amd_smi_data,  # GPU info also in amd-smi output
        }

        return metrics

    def _parse_utilization_from_amd_smi(self, amd_smi_data: Dict) -> Dict:
        """Parse utilization from amd-smi metric output."""
        util_data = {}
        for node, data in amd_smi_data.items():
            # Handle both list and dict formats
            gpu_list = None
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict) and 'gpu_data' in data:
                gpu_list = data['gpu_data']

            if gpu_list:
                util_data[node] = {}
                for gpu in gpu_list:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    # amd-smi uses "usage" field with nested structure
                    usage = gpu.get('usage', {})

                    # Parse gfx_activity value
                    gfx_val = 0.0
                    umc_val = 0.0

                    # Handle case where usage might be a string (like "N/A") or dict
                    if isinstance(usage, dict):
                        # Extract gfx_activity value (the actual GPU utilization)
                        gfx_activity = usage.get('gfx_activity', 'N/A')
                        umc_activity = usage.get('umc_activity', 'N/A')

                        if isinstance(gfx_activity, dict):
                            gfx_val = float(gfx_activity.get('value', 0))
                        elif gfx_activity != 'N/A':
                            try:
                                gfx_val = float(gfx_activity)
                            except (ValueError, TypeError):
                                gfx_val = 0.0

                        if isinstance(umc_activity, dict):
                            umc_val = float(umc_activity.get('value', 0))
                        elif umc_activity != 'N/A':
                            try:
                                umc_val = float(umc_activity)
                            except (ValueError, TypeError):
                                umc_val = 0.0
                    elif usage != 'N/A':
                        # Handle case where usage is a direct numeric value
                        try:
                            gfx_val = float(usage)
                        except (ValueError, TypeError):
                            gfx_val = 0.0

                    util_data[node][gpu_id] = {
                        'GPU use (%)': str(gfx_val),
                        'GFX Activity': str(gfx_val),
                        'UMC Activity': str(umc_val),
                    }
            else:
                util_data[node] = data
        return util_data

    def _parse_memory_from_amd_smi(self, amd_smi_data: Dict) -> Dict:
        """Parse memory from amd-smi metric output."""
        mem_data = {}
        for node, data in amd_smi_data.items():
            # Handle both list and dict formats
            gpu_list = None
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict) and 'gpu_data' in data:
                gpu_list = data['gpu_data']

            if gpu_list:
                mem_data[node] = {}
                for gpu in gpu_list:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    mem_usage = gpu.get('mem_usage', {})

                    # Extract total and used VRAM (in MB, convert to bytes)
                    total_vram_obj = mem_usage.get('total_vram', {})
                    used_vram_obj = mem_usage.get('used_vram', {})

                    total_mb = total_vram_obj.get('value', 0) if isinstance(total_vram_obj, dict) else 0
                    used_mb = used_vram_obj.get('value', 0) if isinstance(used_vram_obj, dict) else 0

                    total_bytes = total_mb * 1024 * 1024 if total_mb else 0
                    used_bytes = used_mb * 1024 * 1024 if used_mb else 0

                    # Calculate percentage
                    used_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0

                    mem_data[node][gpu_id] = {
                        'VRAM Total Memory (B)': total_bytes,
                        'VRAM Total Used Memory (B)': used_bytes,
                        'GPU Memory Allocated (VRAM%)': str(used_percent),
                    }
            else:
                mem_data[node] = data
        return mem_data

    def _parse_temperature_from_amd_smi(self, amd_smi_data: Dict) -> Dict:
        """Parse temperature from amd-smi metric output."""
        temp_data = {}
        for node, data in amd_smi_data.items():
            # Handle both list and dict formats
            gpu_list = None
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict) and 'gpu_data' in data:
                gpu_list = data['gpu_data']

            if gpu_list:
                temp_data[node] = {}
                for gpu in gpu_list:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    temp = gpu.get('temperature', {})

                    # Extract temperature values (they come as dicts with value/unit)
                    edge_obj = temp.get('edge', {})
                    hotspot_obj = temp.get('hotspot', {})
                    mem_obj = temp.get('mem', {})

                    edge_val = edge_obj.get('value', 0) if isinstance(edge_obj, dict) else 0
                    hotspot_val = hotspot_obj.get('value', 0) if isinstance(hotspot_obj, dict) else 0
                    mem_val = mem_obj.get('value', 0) if isinstance(mem_obj, dict) else 0

                    # Handle "N/A" strings
                    if edge_obj == "N/A":
                        edge_val = 0
                    if hotspot_obj == "N/A":
                        hotspot_val = 0
                    if mem_obj == "N/A":
                        mem_val = 0

                    temp_data[node][gpu_id] = {
                        'Temperature (Sensor edge) (C)': str(edge_val),
                        'Temperature (Sensor junction) (C)': str(hotspot_val),
                        'Temperature (Sensor memory) (C)': str(mem_val),
                    }
            else:
                temp_data[node] = data
        return temp_data

    def _parse_pcie_metrics_from_amd_smi(self, amd_smi_data: Dict) -> Dict:
        """
        Parse PCIe link info from amd-smi metric output.

        OPTIMIZATION: PCIe data is already in amd-smi metric --json output:
        "pcie": {
            "width": 16,
            "speed": {"value": 32, "unit": "GT/s"},
            "bandwidth": {"value": 32, "unit": "Mb/s"},
            "replay_count": 0,
            "nak_sent_count": 0,
            ...
        }

        No need for lspci commands!
        """
        pcie_info = {}
        for node, data in amd_smi_data.items():
            # Handle error case
            if isinstance(data, dict) and 'error' in data:
                pcie_info[node] = data
                continue

            # Get GPU list from data
            gpu_list = None
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict) and 'gpu_data' in data:
                gpu_list = data['gpu_data']

            if gpu_list:
                pcie_info[node] = {}
                for gpu in gpu_list:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    pcie_data = gpu.get('pcie', {})

                    if pcie_data:
                        # Flatten nested values for frontend display
                        width = pcie_data.get('width', '-')
                        if width != '-' and width != 'N/A':
                            width = f"x{width}"

                        # Flatten speed object
                        speed_obj = pcie_data.get('speed', {})
                        if isinstance(speed_obj, dict):
                            speed_val = speed_obj.get('value', '-')
                            speed_unit = speed_obj.get('unit', 'GT/s')
                            speed = f"{speed_val} {speed_unit}" if speed_val != '-' else '-'
                        else:
                            speed = str(speed_obj) if speed_obj and speed_obj != 'N/A' else '-'

                        # Flatten bandwidth object
                        bw_obj = pcie_data.get('bandwidth', {})
                        if isinstance(bw_obj, dict):
                            bw_val = bw_obj.get('value', '-')
                            bw_unit = bw_obj.get('unit', 'Mb/s')
                            bandwidth = f"{bw_val} {bw_unit}" if bw_val != '-' else '-'
                        else:
                            bandwidth = str(bw_obj) if bw_obj and bw_obj != 'N/A' else '-'

                        pcie_info[node][gpu_id] = {
                            'width': width,
                            'speed': speed,
                            'bandwidth': bandwidth,
                            'replay_count': pcie_data.get('replay_count', 0),
                            'l0_to_recovery_count': pcie_data.get('l0_to_recovery_count', 0),
                            'nak_sent_count': pcie_data.get('nak_sent_count', 0),
                            'nak_received_count': pcie_data.get('nak_received_count', 0),
                        }

            elif isinstance(data, list):
                # Direct list format - also flatten
                pcie_info[node] = {}
                for gpu in data:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    pcie_data = gpu.get('pcie', {})
                    if pcie_data:
                        # Same flattening as above
                        width = pcie_data.get('width', '-')
                        if width != '-' and width != 'N/A':
                            width = f"x{width}"

                        speed_obj = pcie_data.get('speed', {})
                        if isinstance(speed_obj, dict):
                            speed_val = speed_obj.get('value', '-')
                            speed_unit = speed_obj.get('unit', 'GT/s')
                            speed = f"{speed_val} {speed_unit}" if speed_val != '-' else '-'
                        else:
                            speed = str(speed_obj) if speed_obj and speed_obj != 'N/A' else '-'

                        bw_obj = pcie_data.get('bandwidth', {})
                        if isinstance(bw_obj, dict):
                            bw_val = bw_obj.get('value', '-')
                            bw_unit = bw_obj.get('unit', 'Mb/s')
                            bandwidth = f"{bw_val} {bw_unit}" if bw_val != '-' else '-'
                        else:
                            bandwidth = str(bw_obj) if bw_obj and bw_obj != 'N/A' else '-'

                        pcie_info[node][gpu_id] = {
                            'width': width,
                            'speed': speed,
                            'bandwidth': bandwidth,
                            'replay_count': pcie_data.get('replay_count', 0),
                            'l0_to_recovery_count': pcie_data.get('l0_to_recovery_count', 0),
                            'nak_sent_count': pcie_data.get('nak_sent_count', 0),
                            'nak_received_count': pcie_data.get('nak_received_count', 0),
                        }

        return pcie_info

    def _parse_pcie_metrics_from_amd_smi_OLD(self, amd_smi_data: Dict) -> Dict:
        """OLD VERSION - keeping for reference"""
        pcie_info = {}
        for node, data in amd_smi_data.items():
            if isinstance(data, dict) and 'gpu_data' in data:
                pcie_info[node] = {}
                for gpu in data['gpu_data']:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    pcie_data = gpu.get('pcie', {})

                    # Parse width
                    width = pcie_data.get('width', '-')
                    if width != '-':
                        width = f"x{width}"  # Format as x16, x8, etc.

                    # Parse speed
                    speed_data = pcie_data.get('speed', {})
                    if isinstance(speed_data, dict):
                        speed_value = speed_data.get('value', '-')
                        speed_unit = speed_data.get('unit', 'GT/s')
                        speed = f"{speed_value}{speed_unit}" if speed_value != '-' else '-'
                    else:
                        speed = str(speed_data) if speed_data else '-'

                    # Parse bandwidth
                    bandwidth_data = pcie_data.get('bandwidth', {})
                    if isinstance(bandwidth_data, dict):
                        bw_value = bandwidth_data.get('value', '-')
                        bw_unit = bandwidth_data.get('unit', 'Mb/s')
                        bandwidth = f"{bw_value} {bw_unit}" if bw_value != '-' else '-'
                    else:
                        bandwidth = str(bandwidth_data) if bandwidth_data else '-'

                    pcie_info[node][gpu_id] = {
                        'width': width,
                        'speed': speed,
                        'bandwidth': bandwidth,
                        'replay_count': pcie_data.get('replay_count', 0),
                        'l0_to_recovery_count': pcie_data.get('l0_to_recovery_count', 0),
                        'nak_sent_count': pcie_data.get('nak_sent_count', 0),
                        'nak_received_count': pcie_data.get('nak_received_count', 0),
                    }
        return pcie_info

    def _parse_xgmi_metrics_from_amd_smi(self, amd_smi_data: Dict) -> Dict:
        """
        Parse XGMI metrics from amd-smi output.
        Handles multiple ROCM versions with different formats.
        """
        xgmi_info = {}
        for node, data in amd_smi_data.items():
            # Handle error case
            if isinstance(data, dict) and 'error' in data:
                xgmi_info[node] = data
                continue

            # Handle both list and dict formats
            gpu_list = None
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict) and 'gpu_data' in data:
                gpu_list = data['gpu_data']

            if gpu_list:
                xgmi_info[node] = {}
                for gpu in gpu_list:
                    gpu_id = f"card{gpu.get('gpu', 0)}"
                    # Try different field names: xgmi, xgmi_err, xgmi_error
                    xgmi_data = gpu.get('xgmi', gpu.get('xgmi_err', gpu.get('xgmi_error', {})))

                    # Skip if data is just "N/A" string
                    if xgmi_data and xgmi_data != "N/A" and isinstance(xgmi_data, dict):
                        xgmi_info[node][gpu_id] = xgmi_data
                    else:
                        # No XGMI data available or N/A
                        xgmi_info[node][gpu_id] = {"status": "N/A"}
        return xgmi_info

    def _parse_ras_metrics_from_amd_smi(self, amd_smi_data: Dict) -> Dict:
        """
        Parse RAS/ECC error metrics from amd-smi output.
        Handles multiple ROCM versions with different formats.
        """
        ras_info = {}
        for node, data in amd_smi_data.items():
            # Handle error case
            if isinstance(data, dict) and 'error' in data:
                ras_info[node] = data
                continue

            # Handle both list and dict formats
            gpu_list = None
            if isinstance(data, list):
                gpu_list = data
            elif isinstance(data, dict) and 'gpu_data' in data:
                gpu_list = data['gpu_data']

            if gpu_list:
                ras_info[node] = {}
                for gpu in gpu_list:
                    gpu_id = f"card{gpu.get('gpu', 0)}"

                    # Try different field names across ROCM versions: ecc, ras, ras_errors, ecc_blocks
                    ras_data = gpu.get('ecc', gpu.get('ras', gpu.get('ras_errors', gpu.get('ecc_blocks', {}))))

                    if ras_data:
                        ras_info[node][gpu_id] = ras_data
                    else:
                        # No RAS data available (may not be supported)
                        ras_info[node][gpu_id] = {}
        return ras_info

    def normalize_metrics(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metrics into a consistent format for storage/display.

        Converts from rocm-smi format to a normalized structure:
        {
            "node1": {
                "gpus": [
                    {
                        "id": 0,
                        "utilization": 85.5,
                        "memory_used_mb": 16384,
                        "memory_total_mb": 32768,
                        "temperature_c": 65,
                        "power_w": 250.5,
                        ...
                    },
                    ...
                ]
            },
            ...
        }
        """
        normalized = {}

        # Extract utilization data
        util_data = raw_metrics.get("utilization", {})
        mem_data = raw_metrics.get("memory", {})
        temp_data = raw_metrics.get("temperature", {})

        for node in util_data.keys():
            if node.startswith("ERROR") or node.startswith("ABORT"):
                continue

            normalized[node] = {"gpus": []}

            node_util = util_data.get(node, {})
            node_mem = mem_data.get(node, {})
            node_temp = temp_data.get(node, {})

            # Get list of GPUs (cards)
            gpu_ids = list(node_util.keys()) if isinstance(node_util, dict) else []

            for gpu_id in gpu_ids:
                gpu_metrics = {
                    "id": gpu_id,
                    "utilization": None,
                    "memory_used_mb": None,
                    "memory_total_mb": None,
                    "temperature_c": None,
                }

                # Extract utilization
                if gpu_id in node_util:
                    util_val = node_util[gpu_id].get("GPU use (%)", 0)
                    gpu_metrics["utilization"] = float(util_val) if util_val else 0

                # Extract memory
                if gpu_id in node_mem:
                    mem_used = node_mem[gpu_id].get("VRAM Total Used Memory (B)", 0)
                    mem_total = node_mem[gpu_id].get("VRAM Total Memory (B)", 0)
                    gpu_metrics["memory_used_mb"] = int(mem_used) // (1024 * 1024) if mem_used else 0
                    gpu_metrics["memory_total_mb"] = int(mem_total) // (1024 * 1024) if mem_total else 0

                # Extract temperature (prefer junction/hotspot, fallback to edge or mem)
                if gpu_id in node_temp:
                    # Try junction (hotspot) first, then edge, then mem
                    temp_val = node_temp[gpu_id].get(
                        "Temperature (Sensor junction) (C)",
                        node_temp[gpu_id].get(
                            "Temperature (Sensor edge) (C)", node_temp[gpu_id].get("Temperature (Sensor memory) (C)", 0)
                        ),
                    )
                    try:
                        gpu_metrics["temperature_c"] = float(temp_val) if temp_val and temp_val != 'N/A' else 0.0
                    except (ValueError, TypeError):
                        gpu_metrics["temperature_c"] = 0.0

                normalized[node]["gpus"].append(gpu_metrics)

        return normalized
