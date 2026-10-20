"""Broadcom Thor NIC RDMA QoS validation.

Validates per-device QoS/ETS settings and NVM options (RDMA support,
performance profile, PCIe relaxed ordering, optionally SR-IOV) on Broadcom
Thor NICs using ``niccli``. Assumes a homogeneous cluster where every node
carries identical NIC configuration.

Commands used per PCI device:
  sudo niccli --dev <pci> qos --ets --show
  sudo niccli --dev <pci> nvm --getoption support_rdma --scope 0
  sudo niccli --dev <pci> nvm --getoption performance_profile
  sudo niccli --dev <pci> nvm --getoption pcie_relaxed_ordering
  sudo niccli --dev <pci> nvm --getoption enable_sriov  (optional)
"""

from cvs.lib.preflight.base import PreflightCheck


class BroadcomRdmaQosCheck(PreflightCheck):
    """Validate Broadcom Thor NIC RDMA QoS settings per PCI device."""

    def __init__(
        self,
        orch,
        expected_prio_map="0:0 1:0 2:0 3:1 4:0 5:0 6:0 7:2",
        expected_tc_bandwidth="50% 50% 0%",
        expected_tsa_map="0:ets 1:ets 2:strict",
        expected_pfc_enabled="3",
        expected_dscp_app_table=None,
        expected_support_rdma=True,
        expected_performance_profile="RoCE",
        expected_pcie_relaxed_ordering="Enabled",
        expected_enable_sriov=None,
        use_sudo=True,
        config_dict=None,
    ):
        super().__init__(orch, config_dict)
        self.expected_prio_map = expected_prio_map
        self.expected_tc_bandwidth = expected_tc_bandwidth
        self.expected_tsa_map = expected_tsa_map
        self.expected_pfc_enabled = str(expected_pfc_enabled)
        self.expected_dscp_app_table = expected_dscp_app_table or []
        self.expected_support_rdma = expected_support_rdma
        self.expected_performance_profile = expected_performance_profile
        self.expected_pcie_relaxed_ordering = expected_pcie_relaxed_ordering
        self.expected_enable_sriov = expected_enable_sriov
        self.use_sudo = use_sudo

    def _build_command(self):
        sudo = "sudo -n " if self.use_sudo else ""
        check_sriov = "true" if self.expected_enable_sriov is not None else "false"
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
        DEVS=$({sudo}niccli --list 2>/dev/null | awk '$1 ~ /^[[:space:]]*[0-9]+\\)/ {{
            for (i=1;i<=NF;i++) if ($i ~ /^[0-9a-fA-F]{{4}}:[0-9a-fA-F]{{2}}:[0-9a-fA-F]{{2}}\\.[0-9a-fA-F]$/) print $i
        }}')
        if [ -z "$DEVS" ]; then
            echo "NICCLI:NO_DEVICES"
            exit 0
        fi
        for DEV in $DEVS; do
            echo "DEV_BEGIN:$DEV"

            {sudo}niccli --dev "$DEV" qos --ets --show 2>/dev/null | awk '
                /PRIO_MAP:/ {{ sub(/.*PRIO_MAP:[[:space:]]*/,""); print "PRIO_MAP:" $0 }}
                /TC Bandwidth:/ {{ sub(/.*TC Bandwidth:[[:space:]]*/,""); print "TC_BW:" $0 }}
                /TSA_MAP:/ {{ sub(/.*TSA_MAP:[[:space:]]*/,""); print "TSA_MAP:" $0 }}
                /PFC enabled:/ {{ print "PFC_ENABLED:" $NF }}
                /APP#[0-9]/ {{ split($0,a,"#"); app_num=a[2]+0; prio=""; sel="" }}
                /[[:space:]]Priority:/ {{ prio=$NF }}
                /[[:space:]]Sel:/ {{ sel=$NF }}
                /[[:space:]]DSCP:/ {{ if (prio!="" && sel!="") print "APP_DSCP:" prio ":" sel ":" $NF }}
                /[[:space:]]UDP or DCCP:/ {{ if (prio!="" && sel!="") print "APP_UDP:" prio ":" sel ":" $NF }}
            '

            RDMA=$({sudo}niccli --dev "$DEV" nvm --getoption support_rdma --scope 0 2>/dev/null | awk -F'=[[:space:]]*' '/support_rdma/{{print $2}}')
            echo "SUPPORT_RDMA:${{RDMA:-UNKNOWN}}"

            PERF=$({sudo}niccli --dev "$DEV" nvm --getoption performance_profile 2>/dev/null | awk -F'=[[:space:]]*' '/performance_profile/{{print $2}}')
            echo "PERF_PROFILE:${{PERF:-UNKNOWN}}"

            RO=$({sudo}niccli --dev "$DEV" nvm --getoption pcie_relaxed_ordering 2>/dev/null | awk -F'=[[:space:]]*' '/pcie_relaxed_ordering/{{print $2}}')
            echo "PCIE_RO:${{RO:-UNKNOWN}}"

            if [ "{check_sriov}" = "true" ]; then
                SRIOV=$({sudo}niccli --dev "$DEV" nvm --getoption enable_sriov 2>/dev/null | awk -F'=[[:space:]]*' '/enable_sriov/{{print $2}}')
                echo "SRIOV:${{SRIOV:-UNKNOWN}}"
            fi

            echo "DEV_END:$DEV"
        done
        """

    @staticmethod
    def _parse_device_block(lines):
        """Parse one DEV_BEGIN..DEV_END block into a structured dict."""
        result = {
            'prio_map': None,
            'tc_bandwidth': None,
            'tsa_map': None,
            'pfc_enabled': None,
            'dscp_app_table': [],
            'support_rdma': None,
            'performance_profile': None,
            'pcie_relaxed_ordering': None,
            'enable_sriov': None,
        }
        for line in lines:
            line = line.strip()
            if line.startswith('PRIO_MAP:'):
                result['prio_map'] = line[len('PRIO_MAP:') :].strip()
            elif line.startswith('TC_BW:'):
                result['tc_bandwidth'] = line[len('TC_BW:') :].strip()
            elif line.startswith('TSA_MAP:'):
                result['tsa_map'] = line[len('TSA_MAP:') :].strip()
            elif line.startswith('PFC_ENABLED:'):
                result['pfc_enabled'] = line[len('PFC_ENABLED:') :].strip()
            elif line.startswith('APP_DSCP:'):
                parts = line[len('APP_DSCP:') :].split(':')
                if len(parts) == 3:
                    result['dscp_app_table'].append(
                        {'priority': int(parts[0]), 'sel': int(parts[1]), 'dscp': int(parts[2])}
                    )
            elif line.startswith('APP_UDP:'):
                parts = line[len('APP_UDP:') :].split(':')
                if len(parts) == 3:
                    result['dscp_app_table'].append(
                        {'priority': int(parts[0]), 'sel': int(parts[1]), 'udp_dccp': int(parts[2])}
                    )
            elif line.startswith('SUPPORT_RDMA:'):
                result['support_rdma'] = line[len('SUPPORT_RDMA:') :].strip()
            elif line.startswith('PERF_PROFILE:'):
                result['performance_profile'] = line[len('PERF_PROFILE:') :].strip()
            elif line.startswith('PCIE_RO:'):
                result['pcie_relaxed_ordering'] = line[len('PCIE_RO:') :].strip()
            elif line.startswith('SRIOV:'):
                result['enable_sriov'] = line[len('SRIOV:') :].strip()
        return result

    def _validate_device(self, pci, parsed):
        """Validate one device's parsed values against golden config. Returns (status, errors)."""
        errors = []

        def check(field, actual, expected, label):
            if actual is None:
                errors.append(f"{label}: could not read value")
            elif str(actual).strip() != str(expected).strip():
                errors.append(f"{label}: got '{actual}' expected '{expected}'")

        check('prio_map', parsed['prio_map'], self.expected_prio_map, 'PRIO_MAP')
        check('tc_bandwidth', parsed['tc_bandwidth'], self.expected_tc_bandwidth, 'TC Bandwidth')
        check('tsa_map', parsed['tsa_map'], self.expected_tsa_map, 'TSA_MAP')
        check('pfc_enabled', parsed['pfc_enabled'], self.expected_pfc_enabled, 'PFC enabled')

        # APP table: check that every expected entry is present (order-independent)
        actual_apps = parsed['dscp_app_table']
        for expected_entry in self.expected_dscp_app_table:
            if expected_entry not in actual_apps:
                errors.append(f"APP table entry missing or mismatched: {expected_entry}")
        extra = [e for e in actual_apps if e not in self.expected_dscp_app_table]
        for e in extra:
            errors.append(f"APP table unexpected entry (WARNING): {e}")

        # NVM options
        actual_rdma = parsed['support_rdma']
        exp_rdma = str(self.expected_support_rdma)
        if actual_rdma and actual_rdma.lower() not in (exp_rdma.lower(), 'true', 'false'):
            errors.append(f"support_rdma: got '{actual_rdma}' expected '{exp_rdma}'")
        elif actual_rdma and (actual_rdma.lower() == 'true') != self.expected_support_rdma:
            errors.append(f"support_rdma: got '{actual_rdma}' expected '{exp_rdma}'")

        check(
            'performance_profile',
            parsed['performance_profile'],
            self.expected_performance_profile,
            'performance_profile',
        )
        check(
            'pcie_relaxed_ordering',
            parsed['pcie_relaxed_ordering'],
            self.expected_pcie_relaxed_ordering,
            'pcie_relaxed_ordering',
        )

        if self.expected_enable_sriov is not None:
            check('enable_sriov', parsed['enable_sriov'], self.expected_enable_sriov, 'enable_sriov')

        status = 'FAIL' if errors else 'PASS'
        return status, errors

    def run(self):
        self.results = {}
        out_dict = self.orch.exec(self._build_command())
        for node, output in out_dict.items():
            self.results[node] = self._parse_node_output(node, output)
        return self.results

    def _parse_node_output(self, node, output):
        raw = (output or '').strip()
        lines = raw.splitlines()

        if not lines:
            return {'status': 'FAIL', 'vendor': 'UNKNOWN', 'devices': {}, 'errors': ['Empty output']}

        stripped_lines = [line2.strip() for line2 in lines if line2.strip()]
        first_line = stripped_lines[0] if stripped_lines else ''
        if first_line == 'VENDOR:NOT_BROADCOM':
            return {'status': 'SKIPPED', 'vendor': 'NOT_BROADCOM', 'devices': {}, 'errors': []}
        if 'NICCLI:MISSING' in stripped_lines:
            return {'status': 'WARNING', 'vendor': 'BROADCOM', 'devices': {}, 'errors': ['niccli not found in PATH']}
        if 'NICCLI:NO_DEVICES' in stripped_lines:
            return {
                'status': 'WARNING',
                'vendor': 'BROADCOM',
                'devices': {},
                'errors': ['niccli found no Broadcom PCI devices'],
            }

        # Split output into per-device blocks
        device_blocks = {}
        current_dev = None
        current_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('DEV_BEGIN:'):
                current_dev = stripped[len('DEV_BEGIN:') :]
                current_lines = []
            elif stripped.startswith('DEV_END:') and current_dev:
                device_blocks[current_dev] = list(current_lines)
                current_dev = None
                current_lines = []
            elif current_dev:
                current_lines.append(stripped)

        if not device_blocks:
            return {
                'status': 'FAIL',
                'vendor': 'BROADCOM',
                'devices': {},
                'errors': ['No device blocks found in niccli output'],
            }

        all_errors = []
        device_results = {}
        for pci, block_lines in device_blocks.items():
            parsed = self._parse_device_block(block_lines)
            status, errors = self._validate_device(pci, parsed)
            device_results[pci] = {'status': status, 'parsed': parsed, 'errors': errors}
            all_errors.extend([f"{pci}: {e}" for e in errors])

        overall = 'FAIL' if any(r['status'] == 'FAIL' for r in device_results.values()) else 'PASS'
        return {
            'status': overall,
            'vendor': 'BROADCOM',
            'devices': device_results,
            'errors': all_errors,
        }
