'''
RDMA Connectivity Testing Module

This module provides functions for testing RDMA connectivity across cluster nodes
using parallel group-based algorithms.
'''

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals
from collections import defaultdict

from cvs.lib.preflight.base import PreflightCheck, partition_nodes_into_groups
import random
import re
import shlex
import time
# Import execute_round_with_script_coordination locally to avoid circular imports

log = globals.log


# generate_node_pairs moved to RdmaConnectivityCheck._generate_node_pairs method


# _run_ibv_rc_pingpong_batch moved to RdmaConnectivityCheck._run_batch method


# _analyze_ibv_rc_pingpong_output moved to RdmaConnectivityCheck._analyze_output method


# _extract_ibv_rc_pingpong_errors moved to RdmaConnectivityCheck._extract_errors method


# generate_intergroup_round_pairs moved to RdmaConnectivityCheck._generate_intergroup_pairs method


# execute_intragroup_testing moved to RdmaConnectivityCheck._execute_intragroup method


# execute_intergroup_testing moved to RdmaConnectivityCheck._execute_intergroup method


# execute_parallel_full_mesh_connectivity moved to RdmaConnectivityCheck._execute_full_mesh method


# _calculate_test_port_assignments moved to RdmaConnectivityCheck._calculate_port_assignments method


class RdmaConnectivityCheck(PreflightCheck):
    """Check RDMA connectivity across cluster nodes with multiple testing modes."""

    def _scriptlet_debug_enabled(self) -> bool:
        """
        True when preflight config requests scriptlet debug (preserve scripts/logs and,
        for connectivity, attach strace to generated server ibv_rc_pingpong lines).

        Accepts bool or common string forms from JSON ("true", "1", etc.).
        """
        if not self.config_dict:
            return False
        v = self.config_dict.get('scriptlet_debug', False)
        if isinstance(v, str):
            return v.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(v)

    def _port_listen_retry_max(self) -> int:
        """Max extra ScriptLet batches after initial run for PORT_LISTEN_FAILED (default 3)."""
        cfg = self.config_dict or {}
        raw = cfg.get('rdma_port_listen_retry_max', 3)
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 3
        return max(0, min(n, 10))

    def _port_listen_retry_gap(self) -> int:
        """Port offset step when remapping retry batches (reduces collision with ephemerals)."""
        cfg = self.config_dict or {}
        raw = cfg.get('rdma_port_listen_retry_port_gap', 1000)
        try:
            g = int(raw)
        except (TypeError, ValueError):
            g = 1000
        return max(1, min(g, 65535))

    @staticmethod
    def _pair_key_from_assignment(a):
        """Stable result key for one directed iface test."""
        return f"{a['server_node']} ↔ {a['client_node']} ({a['server_iface']}→{a['client_iface']})"

    @staticmethod
    def _result_is_port_listen_failed(result: dict) -> bool:
        if not isinstance(result, dict) or result.get('status') != 'FAIL':
            return False
        for line in result.get('error_details') or []:
            if 'PORT_LISTEN_FAILED' in str(line):
                return True
        return False

    def _report_output_dir(self) -> str:
        cfg = self.config_dict or {}
        return (cfg.get('report_output_dir') or '/tmp/preflight_reports').rstrip('/')

    def _remote_rdma_workspace_root(self) -> str:
        """
        Remote tree root for ScriptLet logs and scripts.

        Lives under ``report_output_dir`` so operators can point NFS-mounted preflight_reports here.
        """
        return f'{self._report_output_dir()}/rdma_connectivity_workspace'

    def _begin_full_mesh_rdma_artifacts(self) -> None:
        """
        Once per ``full_mesh`` run: ``rm -rf`` remote workspace and create a new session id.

        Clears only ``<report_output_dir>/rdma_connectivity_workspace``, not sibling HTML files.
        """
        if getattr(self, '_full_mesh_rdma_artifact_wipe_done', False):
            return
        root = self._remote_rdma_workspace_root()
        self._rdma_session_id = time.strftime('%Y%m%d_%H%M%S')
        q = shlex.quote
        self.phdl.exec(f"rm -rf {q(root)} && mkdir -p {q(root)}", timeout=180)
        self._full_mesh_rdma_artifact_wipe_done = True
        log.info(
            "RDMA full_mesh: cleared remote workspace %s once (HTML in %s preserved); session %s — "
            "per-round logs under %s/<round>/",
            root,
            self._report_output_dir(),
            self._rdma_session_id,
            f"{root}/{self._rdma_session_id}",
        )

    def _artifact_temp_dir(self, work_segment: str) -> str:
        """Remote directory for one coordination round or inter-group wave."""
        sess = getattr(self, '_rdma_session_id', None)
        if not sess:
            self._rdma_session_id = time.strftime('%Y%m%d_%H%M%S')
            sess = self._rdma_session_id
        safe = re.sub(r'[^a-zA-Z0-9_.-]+', '_', work_segment.strip()) or 'round'
        return f"{self._remote_rdma_workspace_root()}/{sess}/{safe}"

    def __init__(
        self,
        phdl,
        node_list,
        mode="basic",
        port_range="9000-9999",
        timeout=10,
        expected_interfaces=None,
        gid_index="3",
        parallel_group_size=128,
        config_dict=None,
    ):
        """
        Initialize RDMA connectivity check.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            node_list: List of cluster nodes
            mode: "basic", "full_mesh", "sample", or "skip"
            port_range: Port range for ibv_rc_pingpong tests (e.g., "9000-9999")
            timeout: Test timeout in seconds
            expected_interfaces: List of RDMA interfaces
            gid_index: GID index to use
            parallel_group_size: Group size for parallel testing
            config_dict: Optional configuration dictionary
        """
        super().__init__(phdl, config_dict)
        self.node_list = node_list
        self.mode = mode
        self.port_range = port_range
        self.timeout = timeout
        self.expected_interfaces = expected_interfaces
        self.gid_index = gid_index
        self.parallel_group_size = parallel_group_size

    def run(self):
        """
        Execute RDMA connectivity testing with multiple mode support.

        Returns:
            dict: Comprehensive connectivity test results
        """
        log.info(f"Checking RDMA connectivity using ibv_rc_pingpong (mode: {self.mode}, GID index: {self.gid_index})")

        # Handle skip mode
        if self.mode == "skip":
            log.info("RDMA connectivity test skipped by configuration")
            return {
                'mode': 'skip',
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'pair_results': {},
                'node_status': {},
                'skipped': True,
                'pruned_nodes_after_intra': [],
                'partition_groups': {},
                'inter_groups': {},
                'inter_group_mode': '',
                'inter_group_waves': [],
                'inter_group_wave_chunk': None,
            }

        # Parse port range
        port_start, port_end = map(int, self.port_range.split('-'))

        results = {
            'mode': self.mode,
            'total_pairs': 0,
            'successful_pairs': 0,
            'failed_pairs': 0,
            'pair_results': {},
            'node_status': {},
            'gid_index': self.gid_index,
            'port_range': self.port_range,
            'timeout': self.timeout,
            'pruned_nodes_after_intra': [],
            'partition_groups': {},
            'inter_groups': {},
            'inter_group_mode': '',
            'inter_group_waves': [],
            'inter_group_wave_chunk': None,
        }

        # Initialize node status
        for node in self.node_list:
            results['node_status'][node] = {
                'server_tests': 0,
                'client_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
            }

        if self.mode == "full_mesh":
            # Use new parallel group-based algorithm
            all_results, pruned_after_intra, mesh_meta = self._execute_full_mesh(port_start)
            results['pruned_nodes_after_intra'] = pruned_after_intra
            results['partition_groups'] = mesh_meta.get('partition_groups', {})
            results['inter_groups'] = mesh_meta.get('inter_groups', {})
            results['inter_group_mode'] = mesh_meta.get('inter_group_mode', 'multi_wave')
            results['inter_group_waves'] = mesh_meta.get('inter_group_waves', [])
            results['inter_group_wave_chunk'] = mesh_meta.get('inter_group_wave_chunk')

            # Process results from parallel algorithm
            for pair_key, pair_result in all_results.items():
                results['pair_results'][pair_key] = pair_result

                if pair_result['status'] == 'PASS':
                    results['successful_pairs'] += 1
                else:
                    results['failed_pairs'] += 1

                # Update node statistics
                if '(' in pair_key:
                    base_pair = pair_key.split(' (')[0]
                    server_node, client_node = base_pair.split(' ↔ ')
                else:
                    server_node, client_node = pair_key.split(' ↔ ')

                results['node_status'][server_node]['server_tests'] += 1
                results['node_status'][client_node]['client_tests'] += 1

                if pair_result['status'] == 'PASS':
                    results['node_status'][server_node]['successful_tests'] += 1
                    results['node_status'][client_node]['successful_tests'] += 1
                else:
                    results['node_status'][server_node]['failed_tests'] += 1
                    results['node_status'][client_node]['failed_tests'] += 1

            # Total tests calculated in execute_parallel_full_mesh_connectivity
            results['total_pairs'] = len(all_results)

        else:
            # Single batch for basic or sample mode
            pairs = self._generate_node_pairs()
            batch_results = self._run_batch(pairs, port_start)

            # Calculate total tests (pairs × interface_combinations)
            # Cross-interface testing: each node pair tests all server_iface → client_iface combinations
            num_interfaces = len(self.expected_interfaces) if self.expected_interfaces else 1
            results['total_pairs'] = len(pairs) * num_interfaces * num_interfaces
            results['pair_results'] = batch_results

            for pair_key, pair_result in batch_results.items():
                if pair_result['status'] == 'PASS':
                    results['successful_pairs'] += 1
                else:
                    results['failed_pairs'] += 1

                # Update node statistics
                # Handle new format: "node1 ↔ node2 (interface)" or old format: "node1 ↔ node2"
                if '(' in pair_key:
                    base_pair = pair_key.split(' (')[0]
                    server_node, client_node = base_pair.split(' ↔ ')
                else:
                    server_node, client_node = pair_key.split(' ↔ ')
                results['node_status'][server_node]['server_tests'] += 1
                results['node_status'][client_node]['client_tests'] += 1

                if pair_result['status'] == 'PASS':
                    results['node_status'][server_node]['successful_tests'] += 1
                    results['node_status'][client_node]['successful_tests'] += 1
                else:
                    results['node_status'][server_node]['failed_tests'] += 1
                    results['node_status'][client_node]['failed_tests'] += 1

        return results

    def _generate_node_pairs(self):
        """
        Generate node pairs for connectivity testing based on self.mode.

        Returns:
            list: List of tuples representing node pairs
        """
        if self.mode == "skip":
            return []
        elif self.mode == "basic":
            # Adjacent pairs like current IB tests
            pairs = []
            for i in range(0, len(self.node_list) - 1, 2):
                if i + 1 < len(self.node_list):
                    pairs.append((self.node_list[i], self.node_list[i + 1]))
            return pairs

        elif self.mode == "full_mesh":
            # All possible pairs
            pairs = []
            for i in range(len(self.node_list)):
                for j in range(i + 1, len(self.node_list)):
                    pairs.append((self.node_list[i], self.node_list[j]))
            return pairs

        elif self.mode == "sample":
            # Random 20% of all possible pairs
            all_pairs = []
            for i in range(len(self.node_list)):
                for j in range(i + 1, len(self.node_list)):
                    all_pairs.append((self.node_list[i], self.node_list[j]))

            sample_size = max(1, len(all_pairs) // 5)  # 20%
            random.seed(42)  # For reproducible results
            return random.sample(all_pairs, sample_size)

        else:
            raise ValueError(f"Unknown connectivity mode: {self.mode}")

    def _run_batch(self, pairs, base_port):
        """
        Run ibv_rc_pingpong batch test using class attributes.

        Args:
            pairs: List of node pairs to test
            base_port: Base port number for tests

        Returns:
            dict: Test results keyed by pair identifier
        """
        if not pairs:
            return {}

        if self.expected_interfaces is None:
            expected_interfaces = ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
        else:
            expected_interfaces = self.expected_interfaces

        results = {}
        current_port = base_port

        # Test each node pair across all interface combinations
        for server_node, client_node in pairs:
            for server_iface in expected_interfaces:
                for client_iface in expected_interfaces:
                    combo_name = f"{server_iface}→{client_iface}" if server_iface != client_iface else server_iface
                    pair_key = f"{server_node} ↔ {client_node} ({combo_name})"

                    # Run the test
                    server_cmd = f"timeout {self.timeout + 5} ibv_rc_pingpong -d {server_iface} -g {self.gid_index} -p {current_port}"
                    client_cmd = f"timeout {self.timeout} ibv_rc_pingpong -d {client_iface} -g {self.gid_index} -p {current_port} {server_node}"

                    # Start server first, then client
                    server_output = self.phdl.exec_cmd_on_node(server_node, f"{server_cmd} > /dev/null 2>&1 & echo $!")
                    time.sleep(0.5)  # Brief pause for server to bind
                    client_output = self.phdl.exec_cmd_on_node(client_node, client_cmd)

                    # Analyze output using class method
                    success = self._analyze_output(client_output, server_output)

                    results[pair_key] = {
                        'status': 'PASS' if success else 'FAIL',
                        'server_node': server_node,
                        'client_node': client_node,
                        'server_iface': server_iface,
                        'client_iface': client_iface,
                        'interface': combo_name,
                        'port': current_port,
                        'client_output': client_output,
                        'server_output': server_output,
                        'gid_index': self.gid_index,
                        'server_cmd': server_cmd,
                        'client_cmd': client_cmd,
                    }

                    current_port += 1

        return results

    def _execute_full_mesh(self, port_start):
        """
        Execute parallel full mesh connectivity using class attributes.

        Args:
            port_start: Starting port number

        Returns:
            dict: Full mesh test results
        """
        log.info(f"Starting parallel full mesh connectivity testing with group size {self.parallel_group_size}")
        self._begin_full_mesh_rdma_artifacts()

        # Partition nodes into groups (use only reachable nodes)
        reachable_node_list = list(self.phdl.reachable_hosts)
        groups = partition_nodes_into_groups(reachable_node_list, self.parallel_group_size)
        num_groups = len(groups)

        log.info(f"Partitioned {len(reachable_node_list)} reachable nodes into {num_groups} groups")
        for group_id, group_nodes in groups.items():
            log.info(f"{group_id}: {len(group_nodes)} nodes")

        intra_results = self._execute_intragroup(groups, port_start)
        for v in intra_results.values():
            if isinstance(v, dict):
                v['round'] = 'intra_group'

        pruned_records = []
        groups_inter = groups
        cfg = self.config_dict or {}
        if cfg.get('rdma_prune_intra_failed_nodes', True):
            _pruned_set, pruned_records, groups_inter = self._apply_intra_prune(intra_results, groups)
            if pruned_records:
                log.warning(
                    f"Round 1 pruning: excluding {len(pruned_records)} node(s) from inter-group tests: "
                    f"{', '.join(p['node'] for p in pruned_records)}"
                )
            else:
                log.info(
                    "Round 1 pruning: no nodes excluded from inter-group — "
                    "no node met the peer-failure fraction threshold for its group."
                )
        else:
            log.info(
                "Round 1 pruning: disabled (config rdma_prune_intra_failed_nodes=false); "
                "inter-group uses the same node set as after partitioning."
            )

        all_results = dict(intra_results)

        intra_assignments = self._calculate_port_assignments(groups, 'intra_group', port_start)
        if intra_assignments:
            port_cursor = max(a['port'] for a in intra_assignments) + 1
        else:
            port_cursor = port_start

        inter_results, inter_meta = self._execute_intergroup_waves(groups_inter, port_cursor, cfg)
        all_results.update(inter_results)

        mesh_meta = {
            'partition_groups': {k: list(v) for k, v in groups.items()},
            'inter_groups': {k: list(v) for k, v in groups_inter.items()},
            'inter_group_mode': inter_meta.get('mode', 'multi_wave'),
            'inter_group_waves': inter_meta.get('waves', []),
            'inter_group_wave_chunk': inter_meta.get('wave_group_pairs_chunk'),
        }

        log.info(f"Parallel full mesh connectivity testing completed: {len(all_results)} total tests")
        return all_results, pruned_records, mesh_meta

    def _execute_intergroup_waves(self, groups_inter, port_start, cfg):
        """
        Run inter-group tests either in one coordination round (single_shot) or in multiple
        waves (multi_wave) by chunking ordered group-pairs to reduce peak load.

        Config (``cfg``):
            rdma_inter_group_mode: ``single_shot`` | ``multi_wave`` (default ``multi_wave``).
            rdma_inter_group_wave_group_pairs: max ordered group-pairs (Gi→Gj keys) per wave when
                multi_wave. If omitted, empty, or ``auto``, uses **max(1, Ng−1)** where Ng is the
                number of inter-group partition groups.

        Returns:
            tuple: (merged_results_dict, meta_dict with keys ``mode``, ``waves``)
        """
        full = self._all_intergroup_ordered_pairs(groups_inter)
        if not full:
            return {}, {'mode': 'none', 'waves': [], 'wave_group_pairs_chunk': None}

        mode = (cfg.get('rdma_inter_group_mode') or 'multi_wave').lower()
        if mode not in ('single_shot', 'multi_wave'):
            log.warning(f"Unknown rdma_inter_group_mode {mode!r}; using multi_wave")
            mode = 'multi_wave'

        if mode == 'single_shot':
            log.info(f"=== Round 2: Inter-group testing (single shot: {len(full)} ordered group pairs) ===")
            wave_res = self._execute_round_with_coordination(
                full, 'inter_group', port_start, work_segment='inter_group_wave01_single'
            )
            for v in wave_res.values():
                if isinstance(v, dict):
                    v['round'] = 'inter_group'
                    v['inter_wave'] = 1
                    v['inter_waves_total'] = 1
            waves_meta = [
                {
                    'wave': 1,
                    'group_pair_keys': list(full.keys()),
                    'num_group_pairs': len(full),
                }
            ]
            return wave_res, {'mode': 'single_shot', 'waves': waves_meta, 'wave_group_pairs_chunk': None}

        ng = len(groups_inter)
        default_chunk = max(1, ng - 1)
        raw = cfg.get('rdma_inter_group_wave_group_pairs')
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ('', 'auto')):
            chunk_sz = default_chunk
        else:
            try:
                chunk_sz = max(1, int(raw))
            except (TypeError, ValueError):
                chunk_sz = default_chunk

        keys = list(full.keys())
        chunks = [keys[i : i + chunk_sz] for i in range(0, len(keys), chunk_sz)]
        n_waves = len(chunks)
        log.info(
            f"=== Round 2: Inter-group testing (multi-wave: {n_waves} wave(s), "
            f"up to {chunk_sz} ordered group-pair(s) per wave, {len(full)} total group-pairs) ==="
        )

        merged = {}
        waves_meta = []
        port_cursor = port_start

        for wi, key_chunk in enumerate(chunks, start=1):
            sub = {k: full[k] for k in key_chunk}
            log.info(
                f"Inter-group wave {wi}/{n_waves}: {len(sub)} ordered group-pair(s) "
                f"({', '.join(key_chunk[:5])}{'...' if len(key_chunk) > 5 else ''})"
            )
            wave_res = self._execute_round_with_coordination(
                sub, 'inter_group', port_cursor, work_segment=f'inter_group_wave_{wi:02d}'
            )
            for v in wave_res.values():
                if isinstance(v, dict):
                    v['round'] = 'inter_group'
                    v['inter_wave'] = wi
                    v['inter_waves_total'] = n_waves
            merged.update(wave_res)
            sub_assign = self._calculate_port_assignments(sub, 'inter_group', port_cursor)
            if sub_assign:
                port_cursor = max(a['port'] for a in sub_assign) + 1
            waves_meta.append(
                {
                    'wave': wi,
                    'group_pair_keys': list(key_chunk),
                    'num_group_pairs': len(key_chunk),
                }
            )

        return merged, {'mode': 'multi_wave', 'waves': waves_meta, 'wave_group_pairs_chunk': chunk_sz}

    def _parse_peer_failure_prune_threshold(self):
        """
        Fraction in (0, 1]. Nodes with ``failed_peers / (n-1) >= threshold`` are pruned before inter-group.

        Config: ``rdma_prune_peer_failure_threshold`` (default ``0.5``). Invalid or out-of-range values fall back to 0.5.
        """
        cfg = self.config_dict or {}
        raw = cfg.get('rdma_prune_peer_failure_threshold', 0.5)
        try:
            t = float(raw)
        except (TypeError, ValueError):
            log.warning(f"Invalid rdma_prune_peer_failure_threshold {raw!r}; using 0.5")
            t = 0.5
        if t <= 0 or t > 1:
            log.warning(f"rdma_prune_peer_failure_threshold {t} must be in (0, 1]; using 0.5")
            t = 0.5
        return t

    def _failed_peers_from_intra(self, intra_results, groups):
        """
        For each node, set of distinct peers in the same partition group with at least one **FAIL** intra test.
        """
        failed_peers = defaultdict(set)
        for pr in intra_results.values():
            if not isinstance(pr, dict) or pr.get('status') != 'FAIL':
                continue
            sn = pr.get('server_node')
            cn = pr.get('client_node')
            if not sn or not cn or sn == cn:
                continue
            for _gid, nodes in groups.items():
                node_set = set(nodes)
                if sn in node_set and cn in node_set:
                    failed_peers[sn].add(cn)
                    failed_peers[cn].add(sn)
                    break
        return failed_peers

    def _apply_intra_prune(self, intra_results, groups):
        """
        Remove nodes whose intra-group **peer failure fraction** is at or above the configured threshold.

        For a node in a group of ``n`` nodes, fraction = ``(peers with ≥1 FAIL vs that node) / (n-1)``.
        A peer counts if **any** intra test between the two nodes failed (any interface / role combo).

        Config:
            ``rdma_prune_peer_failure_threshold`` (default ``0.5``): prune when fraction >= threshold.

        Returns:
            tuple: (pruned_node_set, pruned_records, groups_for_inter)
        """
        threshold = self._parse_peer_failure_prune_threshold()
        log.info(
            f"Round 1 pruning: peer-failure fraction threshold {threshold:.0%} "
            f"(rdma_prune_peer_failure_threshold); prune when fraction ≥ threshold per node."
        )
        failed_peers = self._failed_peers_from_intra(intra_results, groups)
        pruned_records = []
        pruned_set = set()
        for gid, nodes in groups.items():
            if len(nodes) <= 1:
                continue
            denom = len(nodes) - 1
            for node in nodes:
                cnt = len(failed_peers.get(node, set()))
                fraction = cnt / denom
                if fraction >= threshold:
                    pruned_set.add(node)
                    pruned_records.append(
                        {
                            'node': node,
                            'group_id': gid,
                            'reason': (
                                f'Intra-group peer failure fraction {fraction:.0%} ({cnt}/{denom} peers with ≥1 FAIL) '
                                f'≥ threshold {threshold:.0%} (rdma_prune_peer_failure_threshold).'
                            ),
                        }
                    )

        new_groups = {}
        for gid, nodes in groups.items():
            kept = [x for x in nodes if x not in pruned_set]
            if kept:
                new_groups[gid] = kept

        return pruned_set, pruned_records, new_groups

    def _execute_intragroup(self, groups, port_start):
        """
        Execute intragroup connectivity testing using class attributes.

        Args:
            groups: Dictionary of group_id -> node_list
            port_start: Starting port number

        Returns:
            dict: Intragroup test results
        """
        log.info("=== Round 1: Intra-group parallel testing ===")
        return self._execute_round_with_coordination(groups, "intra_group", port_start, work_segment='intra_group')

    def _all_intergroup_ordered_pairs(self, groups):
        """
        Build every ordered pair of distinct groups (servers in Gi, clients in Gj).

        Required for full mesh: each directed node pair (u, v) with u in Gi and v in Gj
        must appear for both role orders when i != j.
        """
        group_ids = list(groups.keys())
        inter_group_tests = {}
        for gi in group_ids:
            for gj in group_ids:
                if gi == gj:
                    continue
                inter_group_tests[f"{gi}_to_{gj}"] = {
                    'group1': groups[gi],
                    'group2': groups[gj],
                }
        return inter_group_tests

    def _execute_intergroup(self, groups, port_start):
        """
        Execute intergroup connectivity testing using class attributes.

        Args:
            groups: Dictionary of group_id -> node_list
            port_start: Starting port number

        Returns:
            dict: Intergroup test results
        """
        log.info("=== Round 2: Inter-group testing (single coordination, legacy helper) ===")

        inter_group_tests = self._all_intergroup_ordered_pairs(groups)
        if not inter_group_tests:
            return {}

        log.info(f"Inter-group: {len(inter_group_tests)} ordered group pairs (full mesh between groups)")

        res = self._execute_round_with_coordination(
            inter_group_tests, "inter_group", port_start, work_segment='inter_group_legacy'
        )
        for v in res.values():
            if isinstance(v, dict):
                v['round'] = 'inter_group'
                v['inter_wave'] = 1
                v['inter_waves_total'] = 1
        return res

    def _analyze_output(self, client_output, server_output):
        """
        Analyze ibv_rc_pingpong output for success/failure.

        Args:
            client_output: Output from client side
            server_output: Output from server side

        Returns:
            bool: True if successful, False otherwise
        """
        if not client_output:
            return False

        # Success patterns: bandwidth/latency measurements or successful connection info
        success_patterns = [
            r'\d+\s+bytes\s+in\s+[\d.]+\s+seconds',  # Bandwidth results
            r'local\s+address:\s+LID.*GID.*remote\s+address:\s+LID.*GID',  # Connection info
            r'\d+\s+\d+\s+[\d.]+\s+[\d.]+',  # Latency results table
        ]

        for pattern in success_patterns:
            if re.search(pattern, client_output, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    def _extract_errors(self, client_output, server_output):
        """
        Extract error messages from ibv_rc_pingpong output.

        Args:
            client_output: Output from client side
            server_output: Output from server side

        Returns:
            list: List of error messages found
        """
        errors = []

        def extract_connection_details(output, role):
            """Extract connection-related error details."""
            details = []
            if "Failed to modify QP" in output:
                if "to RTR" in output:
                    details.append(f"{role}: Queue Pair failed to transition to Ready-To-Receive (RTR)")
                elif "to RTS" in output:
                    details.append(f"{role}: Queue Pair failed to transition to Ready-To-Send (RTS)")

            if "Couldn't connect to" in output or "Unable to Connect" in output:
                details.append(f"{role}: Connection establishment failed")

            if "Failed status transport retry counter exceeded" in output:
                details.append(f"{role}: Transport retry limit exceeded")

            if "parse WC failed" in output:
                details.append(f"{role}: Work Completion parsing failed")

            return details

        # Check client output
        if client_output:
            client_errors = extract_connection_details(client_output, "Client")
            errors.extend(client_errors)

            # Additional client-specific patterns
            if not client_errors:
                if "timeout" in client_output.lower():
                    errors.append("Client: Connection timeout")
                elif not client_output.strip():
                    errors.append("Client: No output received")

        # Check server output
        if server_output:
            server_errors = extract_connection_details(server_output, "Server")
            errors.extend(server_errors)

        # If we found specific errors, format them nicely
        if not errors:
            errors.append("Unknown error - ibv_rc_pingpong failed to establish connection")

        return errors

    def _generate_intergroup_pairs(self, groups, round_num):
        """
        Generate inter-group pairs for a specific round using round-robin algorithm.

        Args:
            groups: Dictionary of group_id -> [nodes]
            round_num: Round number for pair generation

        Returns:
            dict: Inter-group test pairs for this round
        """
        group_ids = list(groups.keys())
        num_groups = len(groups)
        inter_group_tests = {}

        for i, group1_id in enumerate(group_ids):
            group2_idx = (i + round_num - 2) % num_groups
            if group2_idx != i:  # Don't test group with itself
                group2_id = group_ids[group2_idx]
                inter_group_tests[f"{group1_id}_to_{group2_id}"] = {
                    'group1': groups[group1_id],
                    'group2': groups[group2_id],
                }

        return inter_group_tests

    def _calculate_port_assignments(self, test_groups, round_type, port_start):
        """
        Calculate port assignments for all tests using class attributes.

        Args:
            test_groups: Test group configuration
            round_type: "intra_group" or "inter_group"
            port_start: Starting port number

        Returns:
            list: List of dicts with port assignments
        """
        assignments = []
        if self.expected_interfaces is None:
            expected_interfaces = ["rocep28s0", "rocep62s0", "rocep79s0", "rocep96s0"]
        else:
            expected_interfaces = self.expected_interfaces

        # Track port usage per server node (each node can reuse the same port range)
        node_port_counters = {}

        if round_type == "intra_group":
            for group_id, group_nodes in test_groups.items():
                for server_node in group_nodes:
                    # Initialize port counter for this server node
                    if server_node not in node_port_counters:
                        node_port_counters[server_node] = 0

                    for client_node in group_nodes:
                        if server_node != client_node:
                            for server_iface in expected_interfaces:
                                for client_iface in expected_interfaces:
                                    # Assign sequential port for this specific server node
                                    port = port_start + node_port_counters[server_node]
                                    assignments.append(
                                        {
                                            'server_node': server_node,
                                            'client_node': client_node,
                                            'server_iface': server_iface,
                                            'client_iface': client_iface,
                                            'port': port,
                                        }
                                    )
                                    # Increment port counter only for this server node
                                    node_port_counters[server_node] += 1

        elif round_type == "inter_group":
            for test_name, test_config in test_groups.items():
                group1_nodes = test_config['group1']
                group2_nodes = test_config['group2']

                for server_node in group1_nodes:
                    # Initialize port counter for this server node
                    if server_node not in node_port_counters:
                        node_port_counters[server_node] = 0

                    for client_node in group2_nodes:
                        for server_iface in expected_interfaces:
                            for client_iface in expected_interfaces:
                                # Assign sequential port for this specific server node
                                port = port_start + node_port_counters[server_node]
                                assignments.append(
                                    {
                                        'server_node': server_node,
                                        'client_node': client_node,
                                        'server_iface': server_iface,
                                        'client_iface': client_iface,
                                        'port': port,
                                    }
                                )
                                # Increment port counter only for this server node
                                node_port_counters[server_node] += 1

        return assignments

    def _execute_round_with_coordination(self, test_groups, round_type, port_start, *, work_segment='round'):
        """
        Execute testing round with script coordination using class attributes.

        Retries subsets that fail with ``PORT_LISTEN_FAILED`` up to ``rdma_port_listen_retry_max`` times
        (default 3), using new TCP ports between attempts.

        Args:
            test_groups: Test group configuration
            round_type: "intra_group" or "inter_group"
            port_start: Starting port number
            work_segment: Path segment for this round's remote work dir (under
                ``report_output_dir/rdma_connectivity_workspace/<session>/``).

        Returns:
            dict: Test results for the round
        """
        assignments = self._calculate_port_assignments(test_groups, round_type, port_start)
        if not assignments:
            return {}

        max_pl_retries = self._port_listen_retry_max()
        gap = self._port_listen_retry_gap()
        merged: dict = {}
        next_port_cursor = max(a['port'] for a in assignments) + gap

        pending: list = [dict(a) for a in assignments]
        attempt = 0

        while pending:
            seg = work_segment if attempt == 0 else f"{work_segment}_plretry{attempt}"
            batch = self._execute_assignments_scriptlet_round(pending, round_type, seg)
            merged.update(batch)

            pl_failed_keys = {
                pk for pk, res in batch.items() if isinstance(res, dict) and self._result_is_port_listen_failed(res)
            }
            if not pl_failed_keys:
                break
            if attempt >= max_pl_retries:
                log.warning(
                    "PORT_LISTEN_FAILED persists for %d pair(s) after %d retries; keeping last results.",
                    len(pl_failed_keys),
                    max_pl_retries,
                )
                break

            pk_to_assignment = {self._pair_key_from_assignment(a): a for a in pending}
            pending = []
            for pk in sorted(pl_failed_keys):
                if pk not in pk_to_assignment:
                    continue
                base = dict(pk_to_assignment[pk])
                base['port'] = next_port_cursor
                next_port_cursor += 1
                pending.append(base)

            log.info(
                "RDMA %s: retry batch %d for %d pair(s) with PORT_LISTEN_FAILED (new ports %s..)",
                round_type,
                attempt + 1,
                len(pending),
                pending[0]['port'] if pending else 'n/a',
            )
            attempt += 1

        log.info(
            "ScriptLet-based %s round completed with %d pair result(s) (%d PORT_LISTEN retry wave(s) used)",
            round_type,
            len(merged),
            attempt,
        )
        return merged

    def _execute_assignments_scriptlet_round(self, assignments, round_type, work_segment):
        """
        Run one ScriptLet cycle (servers → clients → collect) for an explicit assignment list.
        """
        from cvs.lib.scriptlet import ScriptLet

        scriptlet_debug = self._scriptlet_debug_enabled()
        temp_dir = self._artifact_temp_dir(work_segment)

        log.info(
            "ScriptLet batch (%s): %d assignment(s), dir %s",
            round_type,
            len(assignments),
            temp_dir,
        )
        if scriptlet_debug:
            log.info(
                "scriptlet_debug: strace logs under %s/strace_server_<iface>_<port>.log",
                temp_dir,
            )

        with ScriptLet(
            self.phdl,
            debug=scriptlet_debug,
            temp_dir=temp_dir,
            cleanup_on_init=True,
            preserve_temp_dir_on_exit=True,
        ) as scriptlet:
            log.info("Phase 1: Starting ibv_rc_pingpong servers")

            for host in self.phdl.reachable_hosts:
                server_commands = self._generate_server_commands_from_assignments(host, assignments, temp_dir)

                if server_commands:
                    script_content = f"""#!/bin/bash
# Start all ibv_rc_pingpong servers in background
{chr(10).join(server_commands)}

sleep 1

echo "All servers started on {host}"
exit 0
"""
                    script_id = f"servers_{host}_{round_type}"
                    scriptlet.create_script(script_id, script_content)

            server_script_mapping = {
                host: f"servers_{host}_{round_type}"
                for host in self.phdl.reachable_hosts
                if f"servers_{host}_{round_type}" in scriptlet.local_scripts
            }

            if server_script_mapping:
                scriptlet.copy_script_list(server_script_mapping)

                log.info(f"Starting {len(server_script_mapping)} server scripts in parallel")
                server_results = scriptlet.run_parallel_group(server_script_mapping, timeout=90)

                failed_servers = []
                for node, output in server_results.items():
                    if "All servers started" not in output:
                        failed_servers.append(f"{node}: {output}")

                if failed_servers:
                    log.error(f"Server startup failed on {len(failed_servers)} nodes:")
                    for failure in failed_servers:
                        log.error(f"  {failure}")
                    raise RuntimeError(f"Server startup failures: {failed_servers}")

                log.info(f"All servers started successfully on {len(server_script_mapping)} nodes")

            log.info("Phase 2: Starting ibv_rc_pingpong clients")

            for host in self.phdl.reachable_hosts:
                client_commands = self._generate_client_commands_from_assignments(host, assignments, temp_dir)

                if client_commands:
                    script_content = f"""#!/bin/bash
# Run all ibv_rc_pingpong clients in parallel
{chr(10).join(client_commands)}

wait

echo "All clients completed on {host}"
exit 0
"""
                    script_id = f"clients_{host}_{round_type}"
                    scriptlet.create_script(script_id, script_content)

            client_script_mapping = {
                host: f"clients_{host}_{round_type}"
                for host in self.phdl.reachable_hosts
                if f"clients_{host}_{round_type}" in scriptlet.local_scripts
            }

            if client_script_mapping:
                scriptlet.copy_script_list(client_script_mapping)

                log.info(f"Starting {len(client_script_mapping)} client scripts in parallel")
                client_execution_results = scriptlet.run_parallel_group(
                    client_script_mapping, timeout=self.timeout + 90
                )

                failed_clients = []
                for node, output in client_execution_results.items():
                    if "All clients completed" not in output:
                        failed_clients.append(f"{node}: {output}")

                if failed_clients:
                    log.warning(f"Client execution issues on {len(failed_clients)} nodes:")
                    for failure in failed_clients[:5]:
                        log.warning(f"  {failure}")

                log.info(f"Client execution completed on {len(client_script_mapping)} nodes")

            log.info("Phase 3: Collecting and analyzing test results")

            test_metadata = []
            for assignment in assignments:
                server_node = assignment['server_node']
                client_node = assignment['client_node']
                server_iface = assignment['server_iface']
                client_iface = assignment['client_iface']
                port = assignment['port']

                test_metadata.append(
                    {
                        'server_node': server_node,
                        'client_node': client_node,
                        'server_iface': server_iface,
                        'client_iface': client_iface,
                        'port': port,
                        'client_log_path': f"{temp_dir}/client_{client_iface}_{port}.log",
                        'server_log_path': f"{temp_dir}/server_{server_iface}_{port}.log",
                    }
                )

            return self._collect_results_with_scriptlet(test_metadata, temp_dir)

    def _generate_server_commands_from_assignments(self, host, assignments, temp_dir):
        commands = []
        for assignment in assignments:
            if assignment['server_node'] == host:
                server_iface = assignment['server_iface']
                port = assignment['port']
                srv_log = f"{temp_dir}/server_{server_iface}_{port}.log"
                if self._scriptlet_debug_enabled():
                    trace_log = f"{temp_dir}/strace_server_{server_iface}_{port}.log"
                    cmd = (
                        f"timeout 120 strace -f -tt "
                        f"-e trace=bind,socket,setsockopt,listen,accept "
                        f"-o {trace_log} "
                        f"ibv_rc_pingpong -d {server_iface} -g {self.gid_index} -p {port} "
                        f"> {srv_log} 2>&1 &"
                    )
                else:
                    cmd = (
                        f"timeout 120 ibv_rc_pingpong -d {server_iface} -g {self.gid_index} -p {port} "
                        f"> {srv_log} 2>&1 &"
                    )
                commands.append(cmd)
        return commands

    def _generate_client_commands_from_assignments(self, host, assignments, temp_dir):
        commands = []
        for assignment in assignments:
            if assignment['client_node'] == host:
                client_iface = assignment['client_iface']
                server_node = assignment['server_node']
                port = assignment['port']
                cmd = (
                    f"timeout 30 ibv_rc_pingpong -d {client_iface} -g {self.gid_index} -p {port} "
                    f"{server_node} > {temp_dir}/client_{client_iface}_{port}.log 2>&1 &"
                )
                commands.append(cmd)
        return commands

    def _generate_server_commands(self, host, test_groups, round_type, port_start, temp_dir="/tmp/preflight"):
        """
        Generate server commands for a round using class attributes.

        Args:
            host: Host to generate commands for
            test_groups: Test group configuration
            round_type: "intra_group" or "inter_group"
            port_start: Starting port number
            temp_dir: Temporary directory for logs

        Returns:
            list: Server commands for this host
        """
        assignments = self._calculate_port_assignments(test_groups, round_type, port_start)
        return self._generate_server_commands_from_assignments(host, assignments, temp_dir)

    def _generate_client_commands(self, host, test_groups, round_type, port_start, temp_dir="/tmp/preflight"):
        """
        Generate client commands for a round using class attributes.

        Args:
            host: Host to generate commands for
            test_groups: Test group configuration
            round_type: "intra_group" or "inter_group"
            port_start: Starting port number
            temp_dir: Temporary directory for logs

        Returns:
            list: Client commands for this host
        """
        assignments = self._calculate_port_assignments(test_groups, round_type, port_start)
        return self._generate_client_commands_from_assignments(host, assignments, temp_dir)

    def _display_ibv_commands(self, server_iface, client_iface, server_node, port):
        """
        Return the ibv_rc_pingpong command lines shown in reports (match ScriptLet / batch timeouts).

        Omits shell redirections so operators can copy-paste after ssh; logs use /tmp/preflight/ when run via scripts.
        """
        server_cmd = f"timeout 120 ibv_rc_pingpong -d {server_iface} -g {self.gid_index} -p {port}"
        client_cmd = f"timeout 30 ibv_rc_pingpong -d {client_iface} -g {self.gid_index} -p {port} {server_node}"
        return server_cmd, client_cmd

    def _create_collection_script(self, test_metadata_for_node, temp_dir):
        """
        Create ibv result collection script.

        Args:
            test_metadata_for_node: Test metadata for a specific node
            temp_dir: Remote directory where client/server logs for this round live

        Returns:
            str: Collection script content
        """
        script_lines = [
            "#!/bin/bash",
            "# Optimized ibv_rc_pingpong result collection script",
            "# Only reports failed tests in key=value format",
            "",
            "# Function to analyze ibv_rc_pingpong output for success/failure",
            "analyze_ibv_output() {",
            "    local log_file=\"$1\"",
            "    local test_key=\"$2\"",
            "    ",
            "    if [[ ! -f \"$log_file\" ]]; then",
            "        echo \"${test_key}=LOG_MISSING\"",
            "        return",
            "    fi",
            "    ",
            "    local content=$(cat \"$log_file\" 2>/dev/null)",
            "    ",
            "    # Check for success patterns",
            "    if echo \"$content\" | grep -qE '[0-9]+ bytes in .* seconds|local address:.*GID.*remote address:.*GID'; then",
            "        # Success - don't report (minimal output)",
            "        return",
            "    fi",
            "    ",
            "    # Check for specific failure patterns",
            "    if echo \"$content\" | grep -qiE 'Failed to modify QP.*to RTR'; then",
            "        echo \"${test_key}=QP_RTR_FAILED\"",
            "    elif echo \"$content\" | grep -qiE 'Failed to modify QP.*to RTS'; then",
            "        echo \"${test_key}=QP_RTS_FAILED\"",
            "    elif echo \"$content\" | grep -qiE \"Couldn.*t connect to|Unable to Connect\"; then",
            "        echo \"${test_key}=CONNECTION_FAILED\"",
            "    elif echo \"$content\" | grep -qiE 'Failed status transport retry counter exceeded'; then",
            "        echo \"${test_key}=TRANSPORT_RETRY_EXCEEDED\"",
            "    elif echo \"$content\" | grep -qiE 'parse WC failed'; then",
            "        echo \"${test_key}=WC_PARSE_FAILED\"",
            "    elif echo \"$content\" | grep -qiE 'No space left on device'; then",
            "        echo \"${test_key}=NO_SPACE_LEFT\"",
            "    elif echo \"$content\" | grep -qiE \"Couldn.*t listen|listen to port|Address already in use|bind: Address already in use\"; then",
            "        echo \"${test_key}=PORT_LISTEN_FAILED\"",
            "    elif [[ -z \"$content\" ]]; then",
            "        echo \"${test_key}=EMPTY_LOG\"",
            "    else",
            "        echo \"${test_key}=UNKNOWN_FAILURE\"",
            "    fi",
            "}",
            "",
            "# Analyze test results",
        ]

        # Add analysis calls for each test involving this node
        for test in test_metadata_for_node:
            # Create a compact test identifier
            test_key = f"{test['server_node']}-{test['client_node']}-{test['server_iface']}-{test['client_iface']}-{test['port']}"

            # Check if this node has client logs to analyze
            if 'client_log_path' in test and test.get('node_role') == 'client':
                client_log = test['client_log_path']
                script_lines.append(f"analyze_ibv_output \"{client_log}\" \"CLIENT_{test_key}\"")

            # Check if this node has server logs to analyze
            if 'server_log_path' in test and test.get('node_role') == 'server':
                server_log = test['server_log_path']
                script_lines.append(f"analyze_ibv_output \"{server_log}\" \"SERVER_{test_key}\"")

        script_lines.extend(
            [
                "",
                f"# Logs and artifacts remain under {temp_dir} on each node",
                "",
                "exit 0",
            ]
        )

        return "\n".join(script_lines)

    def _collect_results_with_scriptlet(self, test_metadata, temp_dir):
        """
        Collect test results using scriptlet with class attributes.

        Args:
            test_metadata: Test metadata list
            temp_dir: Remote directory for this coordination round (same as server/client logs)

        Returns:
            dict: Collected test results
        """
        from cvs.lib.scriptlet import ScriptLet

        log.info(f"Starting ScriptLet-based result collection for {len(test_metadata)} tests")

        # Phase 1: Group test metadata by node (both client and server roles)
        node_tests = {}

        for test in test_metadata:
            client_node = test['client_node']
            server_node = test['server_node']

            # Add test to client node's list
            if client_node not in node_tests:
                node_tests[client_node] = []
            # Mark this test as involving this node as client
            test_copy = test.copy()
            test_copy['node_role'] = 'client'
            node_tests[client_node].append(test_copy)

            # Add test to server node's list (if different from client)
            if server_node != client_node:
                if server_node not in node_tests:
                    node_tests[server_node] = []
                # Mark this test as involving this node as server
                test_copy = test.copy()
                test_copy['node_role'] = 'server'
                node_tests[server_node].append(test_copy)

        log.info(f"Distributing result collection across {len(node_tests)} nodes")

        scriptlet_debug = self._scriptlet_debug_enabled()

        # Phase 2: Generate and execute collection scripts in parallel
        results = {}

        with ScriptLet(
            self.phdl,
            debug=scriptlet_debug,
            temp_dir=temp_dir,
            cleanup_on_init=False,
            preserve_temp_dir_on_exit=True,
        ) as scriptlet:
            collect_mapping = {}
            for node, node_test_list in node_tests.items():
                if node in self.phdl.reachable_hosts:
                    script_id = f"collect_{node}"
                    script_content = self._create_collection_script(node_test_list, temp_dir)
                    scriptlet.create_script(script_id, script_content)
                    collect_mapping[node] = script_id

            if collect_mapping:
                scriptlet.copy_script_list(collect_mapping)
                exec_results = scriptlet.run_parallel_group(collect_mapping, timeout=60)
                for node, script_output in exec_results.items():
                    if script_output:
                        for line in script_output.strip().split('\n'):
                            if '=' in line and line.strip():
                                key, value = line.split('=', 1)
                                results[key] = {'status': 'FAIL', 'error': value}

        # Phase 3: Fill in successful tests (those not reported as failures)
        for test in test_metadata:
            test_key = f"{test['server_node']}-{test['client_node']}-{test['server_iface']}-{test['client_iface']}-{test['port']}"
            client_key = f"CLIENT_{test_key}"
            server_key = f"SERVER_{test_key}"

            # If neither client nor server reported failure, mark as success
            if client_key not in results and server_key not in results:
                pair_key = (
                    f"{test['server_node']} ↔ {test['client_node']} ({test['server_iface']}→{test['client_iface']})"
                )
                server_cmd, client_cmd = self._display_ibv_commands(
                    test['server_iface'],
                    test['client_iface'],
                    test['server_node'],
                    test['port'],
                )
                results[pair_key] = {
                    'status': 'PASS',
                    'server_node': test['server_node'],
                    'client_node': test['client_node'],
                    'server_iface': test['server_iface'],
                    'client_iface': test['client_iface'],
                    'port': test['port'],
                    'server_cmd': server_cmd,
                    'client_cmd': client_cmd,
                }
            else:
                # Convert failure keys to pair format
                pair_key = (
                    f"{test['server_node']} ↔ {test['client_node']} ({test['server_iface']}→{test['client_iface']})"
                )
                if client_key in results or server_key in results:
                    error_details = []
                    if client_key in results:
                        error_details.append(f"Client: {results[client_key]['error']}")
                    if server_key in results:
                        error_details.append(f"Server: {results[server_key]['error']}")

                    server_cmd, client_cmd = self._display_ibv_commands(
                        test['server_iface'],
                        test['client_iface'],
                        test['server_node'],
                        test['port'],
                    )
                    results[pair_key] = {
                        'status': 'FAIL',
                        'server_node': test['server_node'],
                        'client_node': test['client_node'],
                        'server_iface': test['server_iface'],
                        'client_iface': test['client_iface'],
                        'port': test['port'],
                        'error_details': error_details,
                        'server_cmd': server_cmd,
                        'client_cmd': client_cmd,
                    }

        # Phase 2 stores raw log keys (CLIENT_*/SERVER_*); Phase 3 adds display pair_keys.
        # Drop intermediates so callers only see "node ↔ node (iface→iface)" keys.
        for k in list(results.keys()):
            if k.startswith(('CLIENT_', 'SERVER_')):
                del results[k]

        return results

    # check_rdma_connectivity removed - use RdmaConnectivityCheck class directly

    # execute_round_with_script_coordination moved to RdmaConnectivityCheck._execute_round_with_coordination method

    # generate_server_commands_for_round moved to RdmaConnectivityCheck._generate_server_commands method

    # generate_client_commands_for_round moved to RdmaConnectivityCheck._generate_client_commands method

    # create_ibv_result_collection_script moved to RdmaConnectivityCheck._create_collection_script method


# collect_test_results_with_scriptlet moved to RdmaConnectivityCheck._collect_results_with_scriptlet method
