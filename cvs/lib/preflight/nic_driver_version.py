"""
Per-Vendor NIC Driver Version Checking Module

Ports the ``pdsh/clschkbrcm`` script (Broadcom, via ``niccli``) and the
``ansible/ainicfwcheck/fwcheck.yml`` playbook (AINIC, via ``nicctl``), and
generalizes to Mellanox (``mlx5_core``/OFED, via ``modinfo``/``ofed_info``).
Since fleets may mix vendors, each vendor check detects hardware presence per
node and reports SKIPPED (not FAIL) on nodes without that vendor's hardware,
so the dispatcher can be safely left enabled cluster-wide on mixed fleets.

``NicDriverVersionCheck`` is the public dispatcher: it takes the configured
``nic_type`` vendor list and per-vendor golden-value configs, runs only the
selected vendor checks, and merges their per-node results (FAIL > WARNING >
all-SKIPPED > PASS).
"""

from cvs.lib.preflight.base import PreflightCheck


class _VendorDriverVersionCheck(PreflightCheck):
    """Shared per-vendor driver-version check: one ``phdl.exec`` + per-node parse/evaluate."""

    def run(self):
        self.results = {}
        out_dict = self.phdl.exec(self._build_command())
        for node, output in out_dict.items():
            self.results[node] = self._parse_and_evaluate(output)
        return self.results

    @staticmethod
    def _build_command():
        raise NotImplementedError

    def _parse_and_evaluate(self, output):
        raise NotImplementedError


class BroadcomDriverVersionCheck(_VendorDriverVersionCheck):
    """Validate the Broadcom NIC package version via ``niccli``, the CLI tool installed
    alongside the Broadcom driver package (not to be confused with the kernel module
    version reported by ``modinfo`` -- ``niccli`` does not report that at all).
    ``niccli`` requires root, so it is run with ``sudo`` by default
    (``use_sudo=False`` to disable).
    """

    def __init__(
        self,
        phdl,
        expected_package_version="<changeme>",
        use_sudo=True,
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_package_version = expected_package_version
        self.use_sudo = use_sudo

    def _build_command(self):
        sudo = "sudo " if self.use_sudo else ""
        return f"""
        if ! lsmod 2>/dev/null | grep -q '^bnxt_re'; then
            echo "VENDOR:NOT_BROADCOM"
            exit 0
        fi
        echo "VENDOR:BROADCOM"
        if ! command -v niccli >/dev/null 2>&1; then
            echo "NICCLI:MISSING"
            exit 0
        fi
        for idx in $({sudo}niccli --list 2>/dev/null | awk '$1 ~ /^[0-9]+\\)/{{gsub(/\\)/,"",$1); print $1}}'); do
            PKG_VER=$({sudo}niccli -i "$idx" show --pkg_ver 2>/dev/null | grep -i "Active Package Version" | awk -F':' '{{print $2}}' | xargs)
            echo "PKG:$idx:$PKG_VER"
        done
        """

    def _parse_and_evaluate(self, output):
        fields = {}
        packages = {}
        for line in (output or '').strip().split('\n'):
            if not line or ':' not in line:
                continue
            if line.startswith('PKG:'):
                _, idx, version = line.split(':', 2)
                packages[idx] = version.strip()
            else:
                key, _, value = line.partition(':')
                fields[key] = value

        if fields.get('VENDOR') != 'BROADCOM':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_BROADCOM',
                'packages': {},
                'errors': [],
            }

        if fields.get('NICCLI') == 'MISSING' or not packages:
            return {
                'status': 'WARNING',
                'vendor': 'BROADCOM',
                'packages': {},
                'errors': ["niccli not found or reported no NIC(s) (expected installed alongside the Broadcom driver)"],
            }

        errors = []
        for idx, version in sorted(packages.items()):
            if version != self.expected_package_version:
                errors.append(
                    f"NIC {idx} package version={version or 'UNKNOWN'} (expected {self.expected_package_version})"
                )

        return {
            'status': 'PASS' if not errors else 'WARNING',
            'vendor': 'BROADCOM',
            'packages': packages,
            'errors': errors,
        }


class AinicDriverVersionCheck(_VendorDriverVersionCheck):
    """Validate AINIC per-NIC firmware version via ``nicctl show version firmware`` (the CLI
    tool installed alongside the AINIC driver), checking both the ``Uboot-A`` and
    ``Firmware-A`` fields for every NIC on the node.
    """

    def __init__(
        self,
        phdl,
        expected_fw_version="1.117.5-a-56",
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_fw_version = expected_fw_version

    @staticmethod
    def _build_command():
        return """
        if ! lsmod 2>/dev/null | grep -Eq '^ionic(_rdma)?'; then
            echo "VENDOR:NOT_AINIC"
            exit 0
        fi
        echo "VENDOR:AINIC"
        if ! command -v nicctl >/dev/null 2>&1; then
            echo "NICCTL:MISSING"
            exit 0
        fi
        nicctl show version firmware 2>/dev/null | awk '
            /^NIC/{nic=$3}
            /Uboot-A/{uboot=$NF}
            /Firmware-A/{fw=$NF; print "FW:"nic":"uboot":"fw}
        '
        """

    def _parse_and_evaluate(self, output):
        fields = {}
        fw_entries = []
        for line in (output or '').strip().split('\n'):
            if not line or ':' not in line:
                continue
            if line.startswith('FW:'):
                parts = line.split(':')
                if len(parts) >= 4:
                    fw_entries.append({'nic': parts[1], 'uboot': parts[2], 'firmware': parts[3]})
            else:
                key, _, value = line.partition(':')
                fields[key] = value

        if fields.get('VENDOR') != 'AINIC':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_AINIC',
                'fw_entries': [],
                'errors': [],
            }

        if fields.get('NICCTL') == 'MISSING' or not fw_entries:
            return {
                'status': 'WARNING',
                'vendor': 'AINIC',
                'fw_entries': [],
                'errors': ["nicctl not found or reported no NIC(s) (expected installed alongside the AINIC driver)"],
            }

        errors = []
        for entry in fw_entries:
            if entry['uboot'] != self.expected_fw_version or entry['firmware'] != self.expected_fw_version:
                errors.append(
                    f"NIC {entry['nic']}: uboot={entry['uboot']} firmware={entry['firmware']} "
                    f"(expected {self.expected_fw_version})"
                )

        return {
            'status': 'PASS' if not errors else 'WARNING',
            'vendor': 'AINIC',
            'fw_entries': fw_entries,
            'errors': errors,
        }


class MellanoxDriverVersionCheck(_VendorDriverVersionCheck):
    """Validate Mellanox mlx5_core driver version and MLNX_OFED stack version.

    New, unvalidated against real Mellanox hardware -- no bash/ansible source
    material exists to port from; designed by analogy to the Broadcom/AINIC
    checks above.
    """

    def __init__(
        self,
        phdl,
        expected_mlx5_core_version="<changeme>",
        expected_ofed_version="<changeme>",
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_mlx5_core_version = expected_mlx5_core_version
        self.expected_ofed_version = expected_ofed_version

    @staticmethod
    def _build_command():
        return """
        if ! lsmod 2>/dev/null | grep -q '^mlx5_core'; then
            echo "VENDOR:NOT_MELLANOX"
            exit 0
        fi
        echo "VENDOR:MELLANOX"
        MLX5_VER=$(modinfo -F version mlx5_core 2>/dev/null)
        OFED_VER=$(ofed_info -s 2>/dev/null)
        echo "MLX5_CORE_VERSION:$MLX5_VER"
        echo "OFED_VERSION:$OFED_VER"
        """

    def _parse_and_evaluate(self, output):
        fields = {}
        for line in (output or '').strip().split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                fields[key] = value

        if fields.get('VENDOR') != 'MELLANOX':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_MELLANOX',
                'mlx5_core': {},
                'ofed': {},
                'errors': [],
            }

        mlx5_version = fields.get('MLX5_CORE_VERSION', '').strip()
        ofed_version = fields.get('OFED_VERSION', '').strip()

        mlx5_ok = mlx5_version == self.expected_mlx5_core_version
        ofed_ok = ofed_version == self.expected_ofed_version

        errors = []
        if not mlx5_ok:
            errors.append(
                f"mlx5_core version={mlx5_version or 'UNKNOWN'} (expected version={self.expected_mlx5_core_version})"
            )
        if not ofed_ok:
            errors.append(f"OFED version={ofed_version or 'UNKNOWN'} (expected version={self.expected_ofed_version})")

        return {
            'status': 'PASS' if mlx5_ok and ofed_ok else 'WARNING',
            'vendor': 'MELLANOX',
            'mlx5_core': {'version': mlx5_version},
            'ofed': {'version': ofed_version},
            'errors': errors,
        }


_VENDOR_CHECK_CLASSES = {
    'ainic': AinicDriverVersionCheck,
    'broadcom': BroadcomDriverVersionCheck,
    'mellanox': MellanoxDriverVersionCheck,
}


class NicDriverVersionCheck(PreflightCheck):
    """Dispatch the configured vendor driver-version check(s) and merge per-node results."""

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
        Execute every configured vendor's driver-version check and merge results.

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
