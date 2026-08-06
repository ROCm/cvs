"""
Per-Vendor NIC Firmware Checking Module

Ports the ``ansible/ainicfwcheck/fwcheck.yml`` playbook (AINIC), and
generalizes it to Broadcom and Mellanox NICs. Each vendor check detects
hardware presence per node (via ``lsmod``) before evaluating device count
and firmware/host-software version, so a node lacking the vendor's hardware
reports SKIPPED rather than a false FAIL.

NIC-count mismatches on a node with the vendor's hardware present are treated
as FAIL (mirrors the original playbook's blocking ``fail:`` task), while
firmware/host-software version mismatches are treated as WARNING (mirrors
the original ``failed_when: false`` tasks).

``NicFirmwareCheck`` is the public dispatcher: it takes the configured
``nic_type`` vendor list and per-vendor golden-value configs, runs only the
selected vendor checks, and merges their per-node results (FAIL > WARNING >
all-SKIPPED > PASS).
"""

from cvs.lib.preflight.base import PreflightCheck


class _VendorFirmwareCheck(PreflightCheck):
    """Shared per-vendor firmware check: one ``phdl.exec`` + per-node parse/evaluate."""

    def run(self):
        self.results = {}
        out_dict = self.phdl.exec(self._build_command())
        for node, output in out_dict.items():
            self.results[node] = self._parse_and_evaluate(output)
        return self.results

    def _build_command(self):
        raise NotImplementedError

    def _parse_and_evaluate(self, output):
        raise NotImplementedError


class AinicFirmwareCheck(_VendorFirmwareCheck):
    """Validate AINIC NIC count and (non-blocking) firmware/host-software versions."""

    def __init__(
        self,
        phdl,
        expected_nic_count=8,
        expected_fw_version="1.117.5-a-56",
        expected_host_version="1.117.5-a-56",
        use_sudo=True,
        config_dict=None,
    ):
        """
        Initialize the AINIC firmware check.

        Args:
            phdl: Parallel SSH handle for cluster nodes.
            expected_nic_count: Expected count of ionic/AINIC RDMA devices per node.
            expected_fw_version: Expected AINIC firmware version (Uboot-A and Firmware-A).
            expected_host_version: Expected AINIC host-software (nicctl, IPC driver) version.
            use_sudo: Whether to prefix ``nicctl`` invocations with ``sudo``.
            config_dict: Optional configuration dictionary.
        """
        super().__init__(phdl, config_dict)
        self.expected_nic_count = expected_nic_count
        self.expected_fw_version = expected_fw_version
        self.expected_host_version = expected_host_version
        self.use_sudo = use_sudo

    def _build_command(self):
        sudo = "sudo " if self.use_sudo else ""
        return f"""
        if ! lsmod 2>/dev/null | grep -Eq '^ionic(_rdma)?'; then
            echo "VENDOR:NOT_AINIC"
            exit 0
        fi
        echo "VENDOR:AINIC"

        NIC_COUNT=$(ibv_devices 2>/dev/null | awk '/ionic_[0-9]+/ {{c++}} END{{print c+0}}')
        echo "NIC_COUNT:$NIC_COUNT"

        FW_OUTPUT=$({sudo}nicctl show version firmware 2>/dev/null)
        echo "$FW_OUTPUT" | awk '
            /^NIC/{{nic=$3}}
            /Uboot-A/{{uboot=$NF}}
            /Firmware-A/{{fw=$NF; print "FW:"nic":"uboot":"fw}}
        '

        HOST_OUTPUT=$({sudo}nicctl show version host-software 2>/dev/null)
        echo "$HOST_OUTPUT" | awk '
            /nicctl/{{n=$NF}}
            /IPC driver/{{i=$NF}}
            END{{print "HOST:"n":"i}}
        '
        """

    @staticmethod
    def _normalize_version(value):
        return (value or '').replace('.', '').replace('-', '')

    def _parse_and_evaluate(self, output):
        lines = (output or '').strip().split('\n')
        if lines and lines[0].strip() == 'VENDOR:NOT_AINIC':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_AINIC',
                'nic_count': 0,
                'expected_nic_count': self.expected_nic_count,
                'fw_entries': [],
                'host_entry': None,
                'errors': [],
                'warnings': [],
            }

        nic_count = 0
        fw_entries = []
        host_entry = None
        for line in lines:
            if line == 'VENDOR:AINIC':
                continue
            if line.startswith('NIC_COUNT:'):
                try:
                    nic_count = int(line.split(':', 1)[1])
                except (ValueError, IndexError):
                    nic_count = 0
            elif line.startswith('FW:'):
                parts = line.split(':')
                if len(parts) >= 4:
                    fw_entries.append({'nic': parts[1], 'uboot': parts[2], 'firmware': parts[3]})
            elif line.startswith('HOST:'):
                parts = line.split(':')
                if len(parts) >= 3:
                    host_entry = {'nicctl_version': parts[1], 'ipc_driver_version': parts[2]}

        errors = []
        warnings = []

        if nic_count != self.expected_nic_count:
            errors.append(f"Expected {self.expected_nic_count} AINIC device(s), found {nic_count}")

        for entry in fw_entries:
            if entry['uboot'] != self.expected_fw_version or entry['firmware'] != self.expected_fw_version:
                warnings.append(
                    f"NIC {entry['nic']}: uboot={entry['uboot']} firmware={entry['firmware']} "
                    f"(expected {self.expected_fw_version})"
                )

        if not fw_entries:
            warnings.append("Unable to parse firmware version output from 'nicctl show version firmware'")

        if host_entry:
            normalized_expected = self._normalize_version(self.expected_host_version)
            for label, key in (("nicctl", "nicctl_version"), ("IPC driver", "ipc_driver_version")):
                detected = host_entry.get(key)
                if self._normalize_version(detected) != normalized_expected:
                    warnings.append(
                        f"{label} host-software version {detected!r} does not match expected "
                        f"{self.expected_host_version!r}"
                    )
        else:
            warnings.append("Unable to parse host-software version output from 'nicctl show version host-software'")

        status = 'FAIL' if errors else ('WARNING' if warnings else 'PASS')
        return {
            'status': status,
            'vendor': 'AINIC',
            'nic_count': nic_count,
            'expected_nic_count': self.expected_nic_count,
            'fw_entries': fw_entries,
            'host_entry': host_entry,
            'errors': errors,
            'warnings': warnings,
        }


class BroadcomFirmwareCheck(_VendorFirmwareCheck):
    """Validate Broadcom bnxt RDMA device count and (non-blocking) firmware version."""

    def __init__(
        self,
        phdl,
        expected_nic_count=2,
        expected_fw_version="<changeme>",
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_nic_count = expected_nic_count
        self.expected_fw_version = expected_fw_version

    @staticmethod
    def _build_command():
        return """
        if ! lsmod 2>/dev/null | grep -q '^bnxt_re'; then
            echo "VENDOR:NOT_BROADCOM"
            exit 0
        fi
        echo "VENDOR:BROADCOM"

        NIC_COUNT=$(ibv_devices 2>/dev/null | awk '/bnxt_re[0-9]+/ {c++} END{print c+0}')
        echo "NIC_COUNT:$NIC_COUNT"

        for iface in $(ls /sys/class/net 2>/dev/null); do
            driver=$(basename "$(readlink -f /sys/class/net/$iface/device/driver 2>/dev/null)" 2>/dev/null)
            if [ "$driver" = "bnxt_en" ]; then
                fw=$(ethtool -i "$iface" 2>/dev/null | awk -F': ' '/firmware-version/{print $2}')
                echo "FW:$iface:$fw"
            fi
        done
        """

    def _parse_and_evaluate(self, output):
        lines = (output or '').strip().split('\n')
        if lines and lines[0].strip() == 'VENDOR:NOT_BROADCOM':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_BROADCOM',
                'nic_count': 0,
                'expected_nic_count': self.expected_nic_count,
                'fw_entries': [],
                'errors': [],
                'warnings': [],
            }

        nic_count = 0
        fw_entries = []
        for line in lines:
            if line == 'VENDOR:BROADCOM':
                continue
            if line.startswith('NIC_COUNT:'):
                try:
                    nic_count = int(line.split(':', 1)[1])
                except (ValueError, IndexError):
                    nic_count = 0
            elif line.startswith('FW:'):
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    fw_entries.append({'iface': parts[1], 'firmware': parts[2]})

        errors = []
        warnings = []

        if nic_count != self.expected_nic_count:
            errors.append(f"Expected {self.expected_nic_count} Broadcom bnxt RDMA device(s), found {nic_count}")

        for entry in fw_entries:
            if entry['firmware'] != self.expected_fw_version:
                warnings.append(
                    f"Interface {entry['iface']}: firmware={entry['firmware']} (expected {self.expected_fw_version})"
                )

        if not fw_entries:
            warnings.append("Unable to parse firmware version output from 'ethtool -i'")

        status = 'FAIL' if errors else ('WARNING' if warnings else 'PASS')
        return {
            'status': status,
            'vendor': 'BROADCOM',
            'nic_count': nic_count,
            'expected_nic_count': self.expected_nic_count,
            'fw_entries': fw_entries,
            'errors': errors,
            'warnings': warnings,
        }


class MellanoxFirmwareCheck(_VendorFirmwareCheck):
    """Validate Mellanox mlx5 RDMA device count and (non-blocking) firmware version.

    New, unvalidated against real Mellanox hardware -- no bash/ansible source
    material exists to port from; designed by analogy to the AINIC/Broadcom
    checks above.
    """

    def __init__(
        self,
        phdl,
        expected_nic_count=8,
        expected_fw_version="<changeme>",
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_nic_count = expected_nic_count
        self.expected_fw_version = expected_fw_version

    @staticmethod
    def _build_command():
        return """
        if ! lsmod 2>/dev/null | grep -q '^mlx5_core'; then
            echo "VENDOR:NOT_MELLANOX"
            exit 0
        fi
        echo "VENDOR:MELLANOX"

        NIC_COUNT=$(ibv_devices 2>/dev/null | awk '/mlx5_[0-9]+/ {c++} END{print c+0}')
        echo "NIC_COUNT:$NIC_COUNT"

        for iface in $(ls /sys/class/net 2>/dev/null); do
            driver=$(basename "$(readlink -f /sys/class/net/$iface/device/driver 2>/dev/null)" 2>/dev/null)
            if [ "$driver" = "mlx5_core" ]; then
                fw=$(ethtool -i "$iface" 2>/dev/null | awk -F': ' '/firmware-version/{print $2}')
                echo "FW:$iface:$fw"
            fi
        done
        """

    def _parse_and_evaluate(self, output):
        lines = (output or '').strip().split('\n')
        if lines and lines[0].strip() == 'VENDOR:NOT_MELLANOX':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_MELLANOX',
                'nic_count': 0,
                'expected_nic_count': self.expected_nic_count,
                'fw_entries': [],
                'errors': [],
                'warnings': [],
            }

        nic_count = 0
        fw_entries = []
        for line in lines:
            if line == 'VENDOR:MELLANOX':
                continue
            if line.startswith('NIC_COUNT:'):
                try:
                    nic_count = int(line.split(':', 1)[1])
                except (ValueError, IndexError):
                    nic_count = 0
            elif line.startswith('FW:'):
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    fw_entries.append({'iface': parts[1], 'firmware': parts[2]})

        errors = []
        warnings = []

        if nic_count != self.expected_nic_count:
            errors.append(f"Expected {self.expected_nic_count} Mellanox mlx5 RDMA device(s), found {nic_count}")

        for entry in fw_entries:
            if entry['firmware'] != self.expected_fw_version:
                warnings.append(
                    f"Interface {entry['iface']}: firmware={entry['firmware']} (expected {self.expected_fw_version})"
                )

        if not fw_entries:
            warnings.append("Unable to parse firmware version output from 'ethtool -i'")

        status = 'FAIL' if errors else ('WARNING' if warnings else 'PASS')
        return {
            'status': status,
            'vendor': 'MELLANOX',
            'nic_count': nic_count,
            'expected_nic_count': self.expected_nic_count,
            'fw_entries': fw_entries,
            'errors': errors,
            'warnings': warnings,
        }


_VENDOR_CHECK_CLASSES = {
    'ainic': AinicFirmwareCheck,
    'broadcom': BroadcomFirmwareCheck,
    'mellanox': MellanoxFirmwareCheck,
}


class NicFirmwareCheck(PreflightCheck):
    """Dispatch the configured vendor firmware check(s) and merge per-node results."""

    def __init__(self, phdl, nic_types, vendor_configs=None, config_dict=None):
        """
        Args:
            phdl: Parallel SSH handle for cluster nodes.
            nic_types: List of vendor names to activate (subset of ``ainic``,
                ``broadcom``, ``mellanox``).
            vendor_configs: Optional ``{vendor: {kwarg: value}}`` overrides
                passed to each vendor check's constructor.
            config_dict: Optional configuration dictionary.
        """
        super().__init__(phdl, config_dict)
        self.nic_types = list(nic_types)
        self.vendor_configs = vendor_configs or {}

    def run(self):
        """
        Execute every configured vendor's firmware check and merge results.

        Returns:
            dict: Per-node results ``{node: {status, errors, <vendor>: {...}, ...}}``,
                one sub-key per configured vendor's per-node result.
        """
        self.results = {}
        vendor_results = {}
        for vendor in self.nic_types:
            checker_cls = _VENDOR_CHECK_CLASSES[vendor]
            checker = checker_cls(self.phdl, config_dict=self.config_dict, **self.vendor_configs.get(vendor, {}))
            vendor_results[vendor] = checker.run()

        nodes = set()
        for results in vendor_results.values():
            nodes.update(results.keys())

        for node in nodes:
            per_vendor = {vendor: results.get(node) for vendor, results in vendor_results.items()}
            statuses = [result['status'] for result in per_vendor.values() if result is not None]
            errors = [
                error for result in per_vendor.values() if result is not None for error in result.get('errors', [])
            ]

            if 'FAIL' in statuses:
                status = 'FAIL'
            elif 'WARNING' in statuses:
                status = 'WARNING'
            elif statuses and all(s == 'SKIPPED' for s in statuses):
                status = 'SKIPPED'
            else:
                status = 'PASS'

            self.results[node] = {'status': status, 'errors': errors, **per_vendor}
        return self.results
