'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Normalize facts collected by the host configuration validation suite.
'''

import re


FACT_LABELS = {
    "os_release": "OS release",
    "kernel": "Kernel",
    "bios": "BIOS",
    "rocm": "ROCm",
    "gpu_count": "GPU count",
    "online_memory": "Online memory",
    "pci_realloc": "PCI realloc",
    "iommu": "IOMMU",
    "numa_balancing": "NUMA balancing",
    "gpu_pcie": "GPU PCIe",
    "nic_pcie": "NIC PCIe",
    "pci_acs": "PCIe ACS",
    "firmware": "GPU firmware",
}


_UNAVAILABLE_MARKERS = (
    "permission denied",
    "password is required",
    "connection timed out",
    "command not found",
    "no such file",
    "name or service not known",
    "could not resolve hostname",
)


def normalize_output(output, max_len=80):
    value = str(output or "").strip()
    if not value:
        return "\u2014"
    lowered = value.lower()
    if any(marker in lowered for marker in _UNAVAILABLE_MARKERS) or value.count("\n") > 2:
        return "unavailable"
    if len(value) > max_len:
        return value[: max_len - 1] + "\u2026"
    return value


def parse_os_release(output):
    text = str(output or "")
    for key in ("PRETTY_NAME", "VERSION"):
        match = re.search(rf'^{key}=(?:"([^"]*)"|([^\n]*))$', text, re.MULTILINE)
        if match:
            return (match.group(1) or match.group(2)).strip()
    return normalize_output(text)


def parse_kernel_version(output):
    text = normalize_output(output)
    parts = text.split()
    if len(parts) >= 3 and parts[0] == "Linux":
        return parts[2]
    return text


def parse_rocm_version(output):
    text = str(output or "")
    match = re.search(r"ROCm version:\s*([^\s]+)", text, re.I)
    return match.group(1) if match else normalize_output(text)


def parse_online_memory(output):
    text = str(output or "")
    match = re.search(r"Total online memory:\s*([0-9.A-Za-z]+)", text, re.I)
    return match.group(1) if match else normalize_output(text)


def parse_gpu_count(output):
    return str(len(re.findall(r"accelerators:\s+Advanced", str(output or ""), re.I)))


def parse_pci_realloc(output):
    match = re.search(r"(?:^|\s)pci=realloc=([^\s]+)", str(output or ""), re.I)
    return match.group(1) if match else "not set"


def parse_iommu(output):
    return "pt" if re.search(r"(?:^|\s)iommu=pt(?:\s|$)", str(output or ""), re.I) else "not set"


def parse_numa_balancing(output):
    match = re.search(r"=\s*([^\s]+)", str(output or ""))
    return match.group(1) if match else normalize_output(output)


def parse_pcie_link(output):
    text = str(output or "")
    speed = re.search(r"Speed\s+([^,\s]+)", text, re.I)
    width = re.search(r"Width\s+(x[0-9]+)", text, re.I)
    if speed and width:
        return f"{speed.group(1)} \u00b7 {width.group(1)}"
    return normalize_output(text)


def parse_pci_acs(output):
    return "enabled" if re.search(r"ACSCtl:", str(output or ""), re.I) else "disabled"


def record_node_facts(store, key, output_by_node, parser=normalize_output):
    nodes = store.setdefault("nodes", {})
    for node, output in output_by_node.items():
        nodes.setdefault(str(node), {})[key] = parser(output)


def record_indexed_node_facts(store, key, index, output_by_node, parser=normalize_output):
    nodes = store.setdefault("nodes", {})
    for node, output in output_by_node.items():
        indexed = nodes.setdefault(str(node), {}).setdefault(key, {})
        indexed[str(index)] = parser(output)


def record_gpu_firmware(store, firmware_by_node):
    firmware = None
    for node, gpu_entries in firmware_by_node.items():
        if not isinstance(gpu_entries, list):
            continue
        node_firmware = None
        for gpu_entry in gpu_entries:
            if not isinstance(gpu_entry, dict):
                continue
            gpu = str(gpu_entry.get("gpu", "\u2014"))
            gpu_firmware = None
            for entry in gpu_entry.get("fw_list", []):
                if not isinstance(entry, dict) or "fw_id" not in entry:
                    continue
                if firmware is None:
                    firmware = store.setdefault("firmware", {})
                if node_firmware is None:
                    node_firmware = firmware.setdefault(str(node), {})
                if gpu_firmware is None:
                    gpu_firmware = node_firmware.setdefault(gpu, {})
                gpu_firmware[str(entry["fw_id"])] = normalize_output(entry.get("fw_version"))


def collected_fact_keys(results):
    keys = set()
    for facts in (results.get("nodes") or {}).values():
        if isinstance(facts, dict):
            keys.update(facts)
    if results.get("firmware"):
        keys.add("firmware")
    return sorted(keys)
