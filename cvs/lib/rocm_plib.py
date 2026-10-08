'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.lib.utils_lib import *


def _amd_smi_json_command(args: str, use_sudo=True) -> str:
    """Build a portable amd-smi JSON command for nodes with different install paths.

    use_sudo defaults to True so existing callers keep ``sudo bash -lc``. Host checks
    on a scheduler job pass False: bare ``sudo`` waits for a password on the HTTP
    agent, and firmware JSON is readable without root. A missing binary prints
    ``[]`` and exits 0, the same fallback as ``_rocm_smi_showbus_cmd``, so the
    JSON parser gets an empty payload instead of a failed command.
    """
    privilege = "sudo " if use_sudo else ""
    return (
        f"{privilege}bash -lc '"
        "if command -v amd-smi >/dev/null 2>&1; then AMD_SMI=$(command -v amd-smi); "
        "elif [ -x /opt/rocm/bin/amd-smi ]; then AMD_SMI=/opt/rocm/bin/amd-smi; "
        "else echo \"[]\"; exit 0; fi; "
        "\"${AMD_SMI}\" "
        f"{args} --json'"
    )


def get_rocm_smi_dict(phdl):
    rocm_smi_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi -a --json'))
    return rocm_smi_dict


def get_gpu_partition_dict(phdl):
    amd_part_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('partition')))
    return amd_part_dict


def get_gpu_process_dict(phdl):
    amd_proc_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('process')))
    return amd_proc_dict


def get_amd_smi_metric_dict(phdl):
    amd_metric_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('metric')))
    return amd_metric_dict


def get_amd_smi_fw_dict(phdl, use_sudo=True):
    firmware_dict = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('firmware', use_sudo=use_sudo)))
    return firmware_dict


def get_amd_smi_ras_metrics_dict(phdl):
    ras_dict = {}
    ras_dict_t = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('metric --ecc')))
    log.info("%s", ras_dict_t)
    for node in ras_dict_t.keys():
        ras_dict[node] = {}
        log.info('^^^^^')
        log.info("%s", ras_dict_t[node])
        if isinstance(ras_dict_t[node], dict):
            if 'gpu_data' in ras_dict_t[node].keys():
                for gpu_dict in list(ras_dict_t[node]['gpu_data']):
                    ras_dict[node][gpu_dict['gpu']] = gpu_dict['ecc']
        elif isinstance(ras_dict_t[node], list):
            for gpu_dict in ras_dict_t[node]:
                ras_dict[node][gpu_dict['gpu']] = gpu_dict['ecc']

    return ras_dict


def get_amd_smi_pcie_metrics_dict(phdl):
    pcie_dict = {}
    pcie_dict_t = convert_phdl_json_to_dict(phdl.exec(_amd_smi_json_command('metric --pcie')))
    for node in pcie_dict_t.keys():
        pcie_dict[node] = {}
        if isinstance(pcie_dict_t[node], dict):
            if 'gpu_data' in pcie_dict_t[node].keys():
                for gpu_dict in list(pcie_dict_t[node]['gpu_data']):
                    pcie_dict[node][gpu_dict['gpu']] = gpu_dict['pcie']
        elif isinstance(pcie_dict_t[node], list):
            for gpu_dict in pcie_dict_t[node]:
                pcie_dict[node][gpu_dict['gpu']] = gpu_dict['pcie']
    return pcie_dict


def get_gpu_mem_use_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showmemuse --json'))
    return d_dict


def get_gpu_use_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showuse --json'))
    return d_dict


def get_gpu_metrics_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showmetric --json'))
    return d_dict


def get_gpu_fw_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showfwinfo --json'))
    return d_dict


def _rocm_smi_showbus_cmd(use_sudo=True):
    """rocm-smi bus map. Unprivileged form resolves the binary without a login sudo.

    A missing binary prints ``[]`` and exits 0, matching ``_amd_smi_json_command``.
    Exit 127 would make ``convert_phdl_json_to_dict`` store ``{}`` and the GPU PCIe
    check would then KeyError that node while walking every card.
    """
    if use_sudo:
        return 'sudo rocm-smi --loglevel error --showbus --json'
    return (
        "bash -lc '"
        "if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --loglevel error --showbus --json; "
        "elif [ -x /opt/rocm/bin/rocm-smi ]; then /opt/rocm/bin/rocm-smi --loglevel error --showbus --json; "
        "else echo \"[]\"; exit 0; fi'"
    )


def get_gpu_pcie_bus_dict(phdl, use_sudo=True):
    d_dict = convert_phdl_json_to_dict(phdl.exec(_rocm_smi_showbus_cmd(use_sudo)))
    for node, cards in d_dict.items():
        if not isinstance(cards, dict):
            d_dict[node] = {}
    return d_dict


def get_gpu_model_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showproductname --json'))
    return d_dict


def get_gpu_temp_dict(phdl):
    d_dict = convert_phdl_json_to_dict(phdl.exec('sudo rocm-smi --loglevel error --showtemp --json'))
    return d_dict


def get_gpu_fabric_info_dict(phdl, use_sudo=True, amd_smi_path='amd-smi'):
    """Return ``amd-smi fabric --topology --json`` output per cluster node.

    Parsed via ``convert_phdl_json_to_dict``. The structure is amd-smi's JSON
    topology payload (GPU records with fabric port and pod-membership fields).
    Used by the AIMVT-181 IFoE TransferBench preflight check to detect pPod
    (physical pod) and vPod (virtual / logical pod) membership before invoking
    the TransferBench smoketest preset.

    Args:
        phdl: Parallel SSH handle for cluster nodes.
        use_sudo: When True (default), prefix the command with ``sudo``.
        amd_smi_path: Override for the ``amd-smi`` binary (e.g. an absolute
            path). Defaults to PATH-resolved ``amd-smi``.

    Returns:
        dict[str, Any]: ``{node: parsed_amd_smi_topology_json | str}``.
        When ``amd-smi`` output is not valid JSON on a node, the raw string is
        returned for that node so callers can degrade gracefully.
    """
    cmd = f'{amd_smi_path} fabric --topology --json'
    if use_sudo:
        cmd = 'sudo ' + cmd
    return convert_phdl_json_to_dict(phdl.exec(cmd))
