"""
Per-Vendor NIC Driver Version Checking Module

Ports the ``pdsh/clschkbrcm`` script (Broadcom ``bnxt_re``/``bnxt_en``), and
generalizes it to AINIC (``ionic``/``ionic_rdma``) and Mellanox
(``mlx5_core``/OFED) NICs. Since fleets may mix vendors, each vendor check
detects hardware presence per node and reports SKIPPED (not FAIL) on nodes
without that vendor's hardware, so the dispatcher can be safely left enabled
cluster-wide on mixed fleets.

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
    """Validate Broadcom bnxt_re/bnxt_en driver version and DKMS provenance."""

    def __init__(
        self,
        phdl,
        expected_bnxt_re_version="236.1.155.0",
        expected_bnxt_en_version="1.10.3-236.1.155.0",
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_bnxt_re_version = expected_bnxt_re_version
        self.expected_bnxt_en_version = expected_bnxt_en_version

    @staticmethod
    def _build_command():
        return """
        if ! lsmod 2>/dev/null | grep -q '^bnxt_re'; then
            echo "VENDOR:NOT_BROADCOM"
            exit 0
        fi
        echo "VENDOR:BROADCOM"
        RE_VER=$(modinfo -F version bnxt_re 2>/dev/null)
        RE_FILE=$(modinfo -F filename bnxt_re 2>/dev/null)
        EN_VER=$(modinfo -F version bnxt_en 2>/dev/null)
        EN_FILE=$(modinfo -F filename bnxt_en 2>/dev/null)
        echo "RE_VERSION:$RE_VER"
        echo "RE_FILE:$RE_FILE"
        echo "EN_VERSION:$EN_VER"
        echo "EN_FILE:$EN_FILE"
        """

    def _parse_and_evaluate(self, output):
        fields = {}
        for line in (output or '').strip().split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                fields[key] = value

        if fields.get('VENDOR') != 'BROADCOM':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_BROADCOM',
                'bnxt_re': {},
                'bnxt_en': {},
                'errors': [],
            }

        re_version = fields.get('RE_VERSION', '').strip()
        re_file = fields.get('RE_FILE', '').strip()
        en_version = fields.get('EN_VERSION', '').strip()
        en_file = fields.get('EN_FILE', '').strip()

        re_dkms = 'dkms' in re_file.lower()
        en_dkms = 'dkms' in en_file.lower()
        re_ok = re_version == self.expected_bnxt_re_version and re_dkms
        en_ok = en_version == self.expected_bnxt_en_version and en_dkms

        errors = []
        if not re_ok:
            errors.append(
                f"bnxt_re version={re_version or 'UNKNOWN'} dkms={re_dkms} "
                f"(expected version={self.expected_bnxt_re_version}, dkms=True)"
            )
        if not en_ok:
            errors.append(
                f"bnxt_en version={en_version or 'UNKNOWN'} dkms={en_dkms} "
                f"(expected version={self.expected_bnxt_en_version}, dkms=True)"
            )

        return {
            'status': 'PASS' if re_ok and en_ok else 'WARNING',
            'vendor': 'BROADCOM',
            'bnxt_re': {'version': re_version, 'dkms': re_dkms},
            'bnxt_en': {'version': en_version, 'dkms': en_dkms},
            'errors': errors,
        }


class AinicDriverVersionCheck(_VendorDriverVersionCheck):
    """Validate AINIC ionic/ionic_rdma driver version."""

    def __init__(
        self,
        phdl,
        expected_ionic_driver_version="1.117.5-a-56",
        expected_ionic_rdma_driver_version="1.117.5-a-56",
        config_dict=None,
    ):
        super().__init__(phdl, config_dict)
        self.expected_ionic_driver_version = expected_ionic_driver_version
        self.expected_ionic_rdma_driver_version = expected_ionic_rdma_driver_version

    @staticmethod
    def _build_command():
        return """
        if ! lsmod 2>/dev/null | grep -Eq '^ionic(_rdma)?'; then
            echo "VENDOR:NOT_AINIC"
            exit 0
        fi
        echo "VENDOR:AINIC"
        IONIC_VER=$(modinfo -F version ionic 2>/dev/null)
        IONIC_RDMA_VER=$(modinfo -F version ionic_rdma 2>/dev/null)
        echo "IONIC_VERSION:$IONIC_VER"
        echo "IONIC_RDMA_VERSION:$IONIC_RDMA_VER"
        """

    def _parse_and_evaluate(self, output):
        fields = {}
        for line in (output or '').strip().split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                fields[key] = value

        if fields.get('VENDOR') != 'AINIC':
            return {
                'status': 'SKIPPED',
                'vendor': 'NOT_AINIC',
                'ionic': {},
                'ionic_rdma': {},
                'errors': [],
            }

        ionic_version = fields.get('IONIC_VERSION', '').strip()
        ionic_rdma_version = fields.get('IONIC_RDMA_VERSION', '').strip()

        ionic_ok = ionic_version == self.expected_ionic_driver_version
        ionic_rdma_ok = ionic_rdma_version == self.expected_ionic_rdma_driver_version

        errors = []
        if not ionic_ok:
            errors.append(
                f"ionic version={ionic_version or 'UNKNOWN'} (expected version={self.expected_ionic_driver_version})"
            )
        if not ionic_rdma_ok:
            errors.append(
                f"ionic_rdma version={ionic_rdma_version or 'UNKNOWN'} "
                f"(expected version={self.expected_ionic_rdma_driver_version})"
            )

        return {
            'status': 'PASS' if ionic_ok and ionic_rdma_ok else 'WARNING',
            'vendor': 'AINIC',
            'ionic': {'version': ionic_version},
            'ionic_rdma': {'version': ionic_rdma_version},
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
