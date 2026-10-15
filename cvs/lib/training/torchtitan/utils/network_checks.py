'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import shlex

from cvs.lib import globals, linux_utils


_MISSING_SPEED = "__CVS_NIC_SPEED_MISSING__"
log = globals.log


def backend_nic_link_speed_violations(orch, expected_speed=400000):
    """Return backend NIC link-speed violations found on the suite hosts.

    ``expected_speed`` is the sysfs value from ``/sys/class/net/<iface>/speed``
    (megabits per second). 400000 is a 400 Gbps link. The read uses the host
    SSH handle, so it does not depend on the training container.
    """
    host_handle = orch.all
    backend_nics = linux_utils.get_backend_rdma_nic_dict(host_handle)
    if not backend_nics:
        return [f"no backend RDMA NICs discovered; expected link speed {expected_speed}"]

    node_list = list(backend_nics.keys())
    eth_dev_dict = {}
    violations = []

    for node in node_list:
        eth_dev_dict[node] = []
        for rdma_dev, nic_info in backend_nics[node].items():
            eth_device = nic_info.get("eth_device")
            if eth_device:
                eth_dev_dict[node].append((rdma_dev, eth_device))
        if not eth_dev_dict[node]:
            violations.append(f"{node}: no backend RDMA NICs discovered; expected link speed {expected_speed}")

    no_of_nics = max((len(eth_dev_dict[node]) for node in node_list), default=0)
    for i in range(no_of_nics):
        cmd_list = []
        for node in node_list:
            if i < len(eth_dev_dict[node]):
                _rdma_dev, eth_device = eth_dev_dict[node][i]
                iface = shlex.quote(eth_device)
                cmd_list.append(f'cat /sys/class/net/{iface}/speed 2>/dev/null || echo {_MISSING_SPEED}')
            else:
                cmd_list.append("true")

        out_dict = host_handle.exec_cmd_list(cmd_list)
        for node in node_list:
            if i >= len(eth_dev_dict[node]):
                continue
            rdma_dev, eth_device = eth_dev_dict[node][i]
            raw_speed = str((out_dict or {}).get(node, "")).strip()
            if not raw_speed or raw_speed == _MISSING_SPEED:
                log.info(
                    "backend NIC link speed | node=%s rdma=%s iface=%s speed=unreadable expected=%s",
                    node,
                    rdma_dev,
                    eth_device,
                    expected_speed,
                )
                violations.append(f"{node}: backend NIC {eth_device} ({rdma_dev}) has no readable link speed")
                continue
            try:
                speed = int(raw_speed.splitlines()[-1].strip())
            except ValueError:
                log.info(
                    "backend NIC link speed | node=%s rdma=%s iface=%s speed=%s expected=%s",
                    node,
                    rdma_dev,
                    eth_device,
                    raw_speed,
                    expected_speed,
                )
                violations.append(
                    f"{node}: backend NIC {eth_device} ({rdma_dev}) reported non-numeric link speed {raw_speed!r}"
                )
                continue
            log.info(
                "backend NIC link speed | node=%s rdma=%s iface=%s speed=%s expected=%s",
                node,
                rdma_dev,
                eth_device,
                speed,
                expected_speed,
            )
            if speed < expected_speed:
                violations.append(
                    f"{node}: backend NIC {eth_device} ({rdma_dev}) link speed {speed} < expected {expected_speed}"
                )

    return violations
