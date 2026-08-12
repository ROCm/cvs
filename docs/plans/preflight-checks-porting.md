# Plan: Preflight Checks Porting (bash/ansible -> CVS pytest)

## Goal

Port the remaining checks from the out-of-repo pdsh/Ansible pre-flight collection
(`ValidationAutomation/scripts/MultiNodeValidation/precheck/{pdsh,ansible}`) into
CVS's pytest-based preflight suite, following the conventions already established
by the preflight work merged to `main` (`cvs/lib/preflight/*`,
`cvs/tests/preflight/preflight_checks.py`). Read-only, node-health-style
validation is in scope; remediation/"push"/install scripts are explicitly
out of scope (see Open Questions).

## Context (file:line citations)

- Root worktree instructions: `CLAUDE.md` (repo root), "Active Worktree:
  wt_preflight_checks_extension" section — defines the split
  (`cvs/lib/preflight` for libs, `cvs/tests/preflight` for tests, unit tests
  under `cvs/lib/preflight/unittests`) and names the source directories.
- Source runner/order reference:
  `/home/ahskabir/multinode_team/ValidationAutomation/scripts/MultiNodeValidation/precheck/run_iren_precheck`
  lines 44-96 — invocation order: `clschkping` → `clschkuptime` →
  `clschkamdgpu` → `clschkrdmalink` (pdsh) → `check_hosts.yml` →
  `check_ssh.yml` → `fwcheck.yml` → `readlimits.yml` →
  `check_rdma_mapping.yml` → (commented out) `applyainicqos.yml` →
  `validate_all.yml` (ansible) → existing `cvs run ... host_configs_cvs`.
  No explicit fail-fast between steps; each tool owns its own pass/fail
  semantics.
- Existing base class:
  `cvs/lib/preflight/base.py:11-79` — `PreflightCheck(ABC)` with
  `__init__(self, phdl, config_dict=None)`, abstract `run()`, `log_info/
  log_error/log_warning`, plus static helpers `partition_nodes_into_groups`
  and `find_host_group` used for full-mesh sharding.
- Generic "run command across nodes" mechanism already exists:
  `cvs/lib/preflight/gid_consistency.py:104-116` and
  `cvs/lib/preflight/version_check.py:49-84` both call
  `self.phdl.exec(cmd)` → returns `{node: stdout}` dict, iterated per node.
  `phdl` is a `MultiProcessPssh` (`cvs/lib/parallel/multiprocess_pssh.py`),
  built once per test-module run in the `phdl` fixture
  (`cvs/tests/preflight/preflight_checks.py:294-339`). It also exposes
  `phdl.reachable_hosts` and `phdl.prune_nodes(hosts)`
  (`cvs/tests/preflight/preflight_checks.py:177-192`), used to drop
  failed nodes from later checks.
- Existing RDMA NIC parsing helper (already generalizes `clschkrdmalink` /
  `check_rdma_mapping.yml`'s ACTIVE/LINK_UP check):
  `cvs/lib/linux_utils.py:190-229` `get_rdma_nic_dict(phdl)` runs
  `rdma link` and parses `device_status`/`link_status`/`eth_device` per
  device; consumed by
  `cvs/lib/preflight/interface_consistency.py:108-124`.
- Config schema in use today:
  `cvs/input/config_file/preflight/preflight_config.json` — top-level
  `preflight.node_check`, `preflight.connectivity_check.{rdma,ifoe}`
  (with `ifoe.l2ping`, `ifoe.transferbench`), `preflight.reporting`,
  `preflight.debug`. Validated by a pydantic model
  (`cvs/parsers/schemas.py`, referenced from
  `cvs/tests/preflight/preflight_checks.py:25,253,282`).
- Test module structure and result aggregation:
  `cvs/tests/preflight/preflight_checks.py:208-339` (fixtures
  `cluster_file`/`config_file`/`cluster_dict`/`config_dict`/`phdl`/`shdl`),
  `:368-1069` (sequential `test_*` functions writing into module-global
  `preflight_results` dict), `:1226-1334` (`test_generate_preflight_report`
  builds `PreflightReportGenerator` and writes the HTML/summary).
- Report generator conventions: `cvs/lib/preflight/report.py:60-171`
  (`_generate_preflight_summary` keyed by check name) and
  `:981-1039` (per-check HTML section pattern, only rendered for
  failures/non-skips).
- `cvs/tests/preflight/README.md:1-493` documents the currently-shipped
  check list (node health, GID consistency, RDMA interface presence, ROCm
  version, IFoE L2 connectivity, IFoE TransferBench smoketest, RDMA
  connectivity) and the config schema/examples.
- All 18 pdsh scripts and ~15 Ansible playbooks/scripts under
  `ValidationAutomation/scripts/MultiNodeValidation/precheck/{pdsh,ansible}`
  were read in full (see Inventory below for per-script disposition).
- No nested `CLAUDE.md` exists yet under `cvs/`, `cvs/lib/`,
  `cvs/lib/preflight/`, `cvs/tests/`, or `cvs/tests/preflight/` — this is a
  documentation gap called out by the root `CLAUDE.md` but out of scope
  for this plan (separate follow-up task).

## Inventory: source scripts vs port status

| Source script/role | What it checks | Ported? | Existing target file | Proposed target location |
|---|---|---|---|---|
| `pdsh/clschkping` | ICMP `ping -c4 -W1` reachability | No | — | `cvs/lib/preflight/node_reachability.py` (new `PingReachabilityCheck`), test in `preflight_checks.py` |
| `pdsh/clschkuptime` | `uptime` per node (informational) | No | — | fold into new `node_reachability.py` or a small `node_diagnostics.py` |
| `pdsh/clschkssh` | SSH connectivity with detailed failure classification (no route, timeout, refused, perm denied, hostkey, DNS, net unreachable) | Partial — `test_node_reachability` (`preflight_checks.py:368-415`) only checks `SSH_OK` echo, no failure-reason classification, no full mesh | Partially superseded | Extend `node_reachability.py`; optional full-mesh variant separate (see `check_ssh.yml` below) |
| `pdsh/clschkamdgpu` | `lsmod \| grep amdgpu` | Superseded (broader) | `cvs/lib/preflight/scaleup_fabric.py` (`NodeHealthCheck`) already validates AMDGPU/KFD + kernel health via `amd-smi`/dmesg patterns | No action needed |
| `pdsh/clschkrdmalink` | `rdma link show`, 8 named ionic NICs ACTIVE/LINK_UP | Superseded (generalized) | `cvs/lib/preflight/interface_consistency.py` + `cvs/lib/linux_utils.py:190` | No action needed |
| `pdsh/clschkbrcm` | bnxt_re/bnxt_en driver version + DKMS status | No | — | In scope (confirmed: Broadcom NICs remain relevant alongside AINIC): `cvs/lib/preflight/nic_driver_version.py` |
| `pdsh/clschkompi` | `/usr/bin/mpirun` executable exists | No | — | Confirmed out of scope: drop, no action (CVS ships its own MPI orchestration) |
| `pdsh/clscheckmunge`, `clsstopmunge` | munge (SLURM auth) service status/stop | No — and likely out of scope | — | Out of scope: SLURM infra, not GPU cluster health; flag as open question |
| `pdsh/clsinstamdgpu`, `clsinstlibpcidev`, `clsinstompi`, `clsinstperftest`, `clsinstrccltests`, `clspushhostkey`, `clsreboot` | Installation/remediation actions | N/A | — | Out of scope by design (preflight checks must be read-only validation, not remediation) |
| `ansible/checketchosts/check_hosts.yml` | `/etc/hosts` has ~128 expected entries between markers | No | — | `cvs/lib/preflight/etc_hosts_consistency.py` (new), but must replace the hardcoded IREN 128-entry list with entries derived from `cluster_dict['node_dict']` (config-driven, not hardcoded) |
| `ansible/checketchosts/push_hosts.yml` | Writes/repairs `/etc/hosts` | N/A (remediation) | — | Out of scope |
| `ansible/mutualssh/check_ssh.yml` | Full node×node passwordless SSH mesh | No | — | `cvs/lib/preflight/ssh_mesh_connectivity.py` (new); complements the existing head-only `test_node_reachability` |
| `ansible/checkrdmanetdev/check_rdma_mapping.yml` | Exact ordered netdev↔PCI RDMA mapping (8 hardcoded lines) | Partial | `interface_consistency.py` checks ACTIVE/LINK_UP presence but not exact PCI/netdev pairing | Extend `InterfaceConsistencyCheck` (or new `rdma_netdev_mapping.py`) with an optional, config-supplied `expected_mapping: {iface: pci_bdf}` |
| `ansible/ainicfwcheck/fwcheck.yml` | Ionic NIC count==8 (blocking), FW version vs golden (non-blocking), host-SW version (non-blocking) | No | — | `cvs/lib/preflight/nic_firmware_check.py` (new); config-driven `expected_fw_version`, `expected_nic_count` |
| `ansible/readlimits/readlimits.yml` | `/etc/security/limits.conf` has 8 required lines (blocking `fail:`) | No | — | `cvs/lib/preflight/limits_conf_check.py` (new) |
| `ansible/updatelimits/updatelimits.yml` | Writes limits.conf | N/A (remediation) | — | Out of scope |
| `ansible/knownhosts/known_hosts.yml` | `ssh-keyscan` population | N/A (infra setup) | — | Out of scope |
| `ansible/pushsshkeys/push_ssh_keys.yml` | Distributes shared SSH keypair | N/A (remediation; also ships a private key file in source repo) | — | Out of scope; flag security note in Open Questions |
| `ansible/pushsudoers/push_sudoers.yml` | Pushes sudoers file | N/A (remediation) | — | Out of scope |
| `ansible/ainicfwupdate/upgrade_nic_fw_async.yml` | Async NIC FW upgrade | N/A (remediation) | — | Out of scope |
| `ansible/applyqos/applyainicqos.yml` | Applies DSCP/PFC/DCQCN config | N/A (remediation; also commented out in `run_iren_precheck`) | — | Out of scope |
| `ansible/ainicvalidation/scripts/validate_pfc.sh` | PFC pause-type golden value via `nicctl show port -c <id>` | No | — | `cvs/lib/preflight/ainic_pfc_qos_dcqcn.py` (new `PfcValidationCheck`) |
| `ansible/ainicvalidation/scripts/validate_qos.sh` | 8 QoS golden values (DSCP, no-drop bitmap, sched priority) via `nicctl show qos -c <id>` | No | — | same module, `QosValidationCheck` |
| `ansible/ainicvalidation/scripts/validate_dcqcn.sh` | 12 DCQCN golden values via `nicctl show dcqcn -r <dev> -i 1` | No | — | same module, `DcqcnValidationCheck` |
| `ansible/ainicvalidation/playbooks/validate_all.yml` | Orchestrates the 3 scripts above per node + cluster summary | No | — | New pytest test functions in `preflight_checks.py` orchestrating the 3 new check classes, mirroring the existing IFoE/TransferBench pattern (opt-in config, mandatory-once-enabled, non-pruning) |

Already-ported/superseded checks not derived from this source material at all
(net-new CVS work, keep as-is): `GidConsistencyCheck`, `RocmVersionCheck`,
`RdmaConnectivityCheck` (`ibv_rc_pingpong`), `IfoeL2ConnectivityCheck`
(`afmctl test ping`), `TransferBenchSmokeCheck`, `NodeHealthCheck`
(MI4XX/AFM/vPOD admission).

## Existing conventions to reuse

Confirmed: a generic "run command across nodes, collect pass/fail" helper
**does** exist — `phdl.exec(cmd)` (backed by `MultiProcessPssh`,
`cvs/lib/parallel/multiprocess_pssh.py`) returns `{node: stdout}`; every
existing check class parses that dict per node
(`cvs/lib/preflight/gid_consistency.py:104-116`,
`cvs/lib/preflight/version_check.py:49-84`,
`cvs/lib/preflight/interface_consistency.py:108-124` via
`linux_utils.get_rdma_nic_dict`). New ports should reuse this same
`phdl.exec()` pattern rather than introducing pdsh/ssh-loop equivalents.

Other conventions to follow:

- Subclass `PreflightCheck` (`cvs/lib/preflight/base.py`), implement only
  `run()`, store per-node dict in `self.results`, use `self.log_info/
  log_error/log_warning`.
- Config lives under `preflight.<new_section>` in
  `cvs/input/config_file/preflight/preflight_config.json`, read via the
  local `get_nested_config(config_dict, section, key, default)` helper
  duplicated in both `preflight_checks.py:32-60` and `report.py:18-46`
  (consider not re-duplicating a third time — pass config sections through
  rather than re-deriving).
  New pydantic sub-models should be added to `cvs/parsers/schemas.py`'s
  `PreflightConfigFile` alongside `node_check`/`connectivity_check`.
- Each check is opt-in via an `enabled` flag normalized with
  `_config_flag_enabled()` (`preflight_checks.py:63-71`), defaulting to
  `False` for new/experimental checks (matching `l2ping`/`transferbench`)
  or `True` only if it should run out-of-the-box like `node_check`.
- Result dict shape per check follows one of two shapes already in use:
  per-node dict (`{node: {'status': 'PASS'|'FAIL', 'errors': [...], ...}}`,
  used by GID/ROCm/interface checks) or an aggregate dict with
  `node_results`, `status`, `coverage`, `failed_nodes` (used by
  `NodeHealthCheck`, `IfoeL2ConnectivityCheck`, `TransferBenchSmokeCheck`).
  New checks that are per-node/stateless should use the first (simpler)
  shape; only checks needing mesh/coverage bookkeeping need the second.
- Add a `_summarize_<check>_results()` + `_generate_<check>_html()` pair
  to `cvs/lib/preflight/report.py` for every new check, and wire it into
  `_generate_preflight_summary()`'s `checks` dict and
  `_generate_html_content()`'s section list.
- Failed nodes are pruned from `phdl` only when a later check's validity
  depends on it (`_prune_nodes_from_phdl`, `preflight_checks.py:177-192`);
  purely informational checks (ROCm version mismatch, uptime) must not
  prune.
- Unit tests: `unittest.TestCase` with `MagicMock()` standing in for
  `phdl`, `sys.path.insert(...)` shim to reach the `cvs` package root
  (`cvs/lib/preflight/unittests/test_rdma_connectivity.py:1-42`); one test
  file per lib module under `cvs/lib/preflight/unittests/`.

## Approach for remaining ports

For each new lib module below: subclass `PreflightCheck`, use
`phdl.exec()`, return the per-node result dict, add a `test_*` function in
`preflight_checks.py` that stores into `preflight_results`, and add
report summarize/HTML methods.

1. **`node_reachability.py`** (`PingReachabilityCheck`, optionally
   `UptimeCheck`) — ports `clschkping` + `clschkuptime`. Command:
   `ping -c 4 -W 1 <target-ip>` run from the CVS driver host (not via
   `phdl.exec`, since ping is driver→node, not node→node — may need a
   lightweight local-subprocess helper rather than `phdl.exec`, since
   `phdl` targets remote nodes over SSH and ping here precedes SSH
   viability). Config: `preflight.node_check.ping_check.enabled` (default
   `false`), `timeout_sec`, `count`. Unit tests mock `subprocess.run`.
   Optionally extend `test_node_reachability` to layer ping *before* the
   SSH echo check, and prune nodes that fail ping before attempting SSH.

2. **`ssh_mesh_connectivity.py`** (`SshMeshConnectivityCheck`) — ports
   `ansible/mutualssh/check_ssh.yml`. Each reachable node SSHes to every
   other reachable node (`ssh -o BatchMode=yes <peer> true`) via
   `phdl.exec()` with a per-node command that loops over a peer list
   templated into the command string (same pattern
   `RdmaConnectivityCheck` uses for pairwise tests — reuse
   `partition_nodes_into_groups`/`find_host_group` from `base.py` for
   large clusters). Config:
   `preflight.connectivity_check.ssh_mesh.enabled` (default `false`).
   Non-fatal/WARNING by default since CVS's own driver→node SSH (`phdl`)
   is what actually matters; full mesh is a diagnostic extra.

3. **`etc_hosts_consistency.py`** (`EtcHostsConsistencyCheck`) — ports
   `check_hosts.yml`. Do **not** hardcode the 128-entry IREN list; derive
   expected `{hostname: ip}` pairs from `cluster_dict['node_dict']` (already
   available as a fixture) plus an optional config override list for
   entries outside the cluster (e.g. management/head node aliases).
   Command: `cat /etc/hosts`, parse and diff. Config:
   `preflight.node_check.etc_hosts.enabled` (default `false`),
   `extra_entries: [{hostname, ip}]`.

4. **`limits_conf_check.py`** (`LimitsConfCheck`) — ports
   `ansible/readlimits/readlimits.yml`. Command: `cat
   /etc/security/limits.conf`, check for N required lines (config-driven
   list, not hardcoded — pull the 8 IREN-specific lines into
   `preflight_config.json` as a default list that operators can override).
   Config: `preflight.node_check.limits_conf.enabled` (default `false`),
   `required_lines: [...]`. Mirrors source's blocking semantics: FAIL gate
   when enabled (source used an explicit Ansible `fail:` on `n_failed>0`).

5. **`nic_firmware_check.py`** (`NicFirmwareCheck`) — ports
   `ansible/ainicfwcheck/fwcheck.yml`. Uses `nicctl show port`/`nicctl
   show device` (AINIC/Pollara equivalent of the original bnxt/ionic
   commands) via `phdl.exec()`. Config:
   `preflight.connectivity_check.ifoe.nic_firmware.enabled` (default
   `false`), `expected_fw_version`, `expected_nic_count` (default 8).
   Preserve source's split severity: NIC-count mismatch is FAIL; FW/host-SW
   version mismatch is WARNING (non-blocking), matching
   `ansible/ainicfwcheck/fwcheck.yml`'s `failed_when: false` on the
   version comparisons.

6. **`ainic_pfc_qos_dcqcn.py`** (`PfcValidationCheck`, `QosValidationCheck`,
   `DcqcnValidationCheck`) — ports the `ainicvalidation` trio. Each class
   runs the equivalent `nicctl show {port,qos,dcqcn}` commands per card
   via `phdl.exec()` and compares against a config-supplied golden-value
   dict rather than the hardcoded constants in `validate_pfc.sh`/
   `validate_qos.sh`/`validate_dcqcn.sh`, since golden values are
   deployment-specific (DSCP maps, scheduling priorities, DCQCN tuning).
   Config: `preflight.connectivity_check.ifoe.pfc_qos_dcqcn.enabled`
   (default `false`), with `pfc`, `qos`, `dcqcn` sub-blocks each carrying
   the golden-value table. Confirmed: ship a built-in default golden-value
   table (the same values validated in the source `validate_*.sh`
   scripts) as the out-of-the-box default for AINIC deployments generally
   — do not brand or document these as deployment-specific in code/docs,
   just as sensible AINIC defaults — with every field overridable per
   deployment via config. This is the single largest remaining port — it
   duplicates `validate_all.yml`'s three-play structure (deploy → run →
   per-node PASS/FAIL → cluster summary) but the "deploy scripts to
   /tmp then run" step is unnecessary since CVS runs the equivalent
   command directly via `phdl.exec()` instead of copying a shell script.
   Unit tests: golden-value comparison logic covered exhaustively with
   synthetic `nicctl` output fixtures (valid, one-field-off, missing
   card, malformed output).

7. **`nic_driver_version.py`** (`NicDriverVersionCheck`) — ports
   `clschkbrcm`. Confirmed in scope: the fleet has a mix of Broadcom
   (bnxt_re/bnxt_en) and AINIC/Pollara NICs, so this check must not assume
   one vendor. Detect NIC vendor per node first (e.g. `lspci` grep or
   `ls /sys/class/infiniband`), then only validate bnxt_re/bnxt_en driver
   version + DKMS status (`modinfo bnxt_re`/`dkms status`) on nodes where
   Broadcom hardware is actually present; nodes with only AINIC hardware
   are skipped (not failed) for this check. Config:
   `preflight.node_check.nic_driver_version.enabled` (default `false`),
   `expected_bnxt_re_version`, `expected_bnxt_en_version`. This runs
   independently of and in addition to `nic_firmware_check.py` (AINIC),
   not as a replacement.

## Suggested grouping/ordering

Mirror `run_iren_precheck`'s "eliminate unreachable nodes first, then
validate static config, then validate functional/data-plane behavior"
structure, merged into the existing `preflight_checks.py` sequence:

1. **Elimination tier (WARNING, prune)** — `test_node_ping_reachability`
   (new, ICMP) → `test_node_reachability` (existing SSH echo) →
   `test_ssh_mesh_connectivity` (new, full mesh, WARNING-only, no prune
   since it's diagnostic) → `test_node_uptime` (new, informational only).
2. **Static config validation tier (mostly WARNING, some FAIL)** —
   `test_etc_hosts_consistency` (new) → `test_limits_conf` (new, FAIL
   when enabled) → `test_nic_firmware` (new; NIC-count sub-check FAIL,
   version sub-checks WARNING) → `test_nic_driver_version` (new,
   Broadcom-only nodes, WARNING on version mismatch, SKIP on non-Broadcom
   nodes) → `test_node_health` (existing, mandatory FAIL gate, unchanged
   position) → `test_rocm_version_consistency` (existing).
3. **IFoE functional tier (FAIL gates once enabled)** —
   `test_ifoe_l2_connectivity` (existing) → **new**
   `test_ainic_pfc_qos_dcqcn` (new, placed here because it validates the
   same AINIC control-plane subsystem as L2 connectivity, before the
   TransferBench data-path test which depends on a correctly configured
   fabric) → `test_ifoe_transferbench_smoke` (existing).
4. **RDMA tier (unchanged)** — `test_interface_name_consistency` →
   `test_gid_consistency` → `test_rdma_connectivity`.
5. **Reporting** — `test_generate_preflight_report` (existing; extend
   `required_checks` list and summary/HTML wiring for every new check).

All new checks default to `enabled: false` in
`preflight_config.json` (consistent with how `l2ping`/`transferbench`
were introduced) so existing customer configs are unaffected until they
opt in.

## Testing & validation plan

- **Unit tests**: one `unittest.TestCase` file per new lib module under
  `cvs/lib/preflight/unittests/`, following
  `test_rdma_connectivity.py`'s pattern (MagicMock `phdl`, `sys.path`
  shim). Cover: PASS path, each documented FAIL path from the source
  script/playbook (e.g. missing NIC, wrong FW version, empty
  `/etc/hosts`, one limits.conf line missing, one PFC/QoS/DCQCN field
  off), and malformed/empty command output.
- **Config schema tests**: extend `cvs/parsers/schemas.py` and its
  existing unit tests for each new `preflight.*` sub-section, including
  default `enabled=false` behavior and unknown-key rejection (mirroring
  `_node_check_config`/`_ifoe_config`'s `unknown` key validation in
  `preflight_checks.py:86-118`).
- **Workflow**: `make fmt-check` → `make lint` → `make test` after every
  module is added (per `CONTRIBUTORS.md`'s documented workflow); do not
  skip `make fmt` before committing.
- **Report/HTML smoke test**: run the full `preflight_checks.py` suite
  against a small synthetic/mocked cluster (or the existing unit-test
  mocks) to confirm the HTML report renders sections for every newly
  wired check without exceptions when results are SKIPPED (default
  `enabled=false` state) — this catches missing `_summarize_*`/
  `_generate_*_html` wiring before real hardware.
- **Real-hardware smoke test** (adapted from root `CLAUDE.md`'s
  `CVS_features` harness flow):
  1. Rebuild the sdist and reinstall into `.test_venv` (`make test` from
     repo root — editable install is NOT used, per `CLAUDE.md`).
  2. Update `cvs/input/config_file/preflight/preflight_config.json` to
     enable exactly one new check at a time (e.g.
     `node_check.limits_conf.enabled: true`), keeping all others at
     their existing settings; confirm no `<changeme>` placeholders
     remain, `gid_index` is `3`, `expected_rocm_version` is `7.2.0`.
  3. Copy the updated config to
     `/home/ahskabir/multinode_team/DEV/CVS_features/config/preflight_config.json`.
  4. Run
     `bash /home/ahskabir/multinode_team/DEV/CVS_features/runpreflighttest`
     and inspect `output/log_preflight.txt` plus the generated HTML
     report for the new check's section.
  5. Repeat incrementally for each newly enabled check before enabling
     multiple simultaneously, to isolate failures to a single new module.

## Decisions (previously open questions — all resolved by repo owner)

1. **Remediation/"push" scripts scope** — CONFIRMED out of scope.
   `push_hosts.yml`, `updatelimits.yml`, `push_ssh_keys.yml`,
   `push_sudoers.yml`, `known_hosts.yml`, `upgrade_nic_fw_async.yml`,
   `applyainicqos.yml`, and all `clsinst*`/`clspushhostkey`/`clsreboot`
   pdsh scripts stay unported. `cvs/tests/preflight` remains a read-only
   validation suite; no separate remediation suite is planned.
2. **munge (SLURM auth)** — CONFIRMED out of scope entirely. Do not port
   `clscheckmunge`/`clsstopmunge`.
3. **Broadcom NIC support** — CONFIRMED in scope. Both Broadcom
   (bnxt_re/bnxt_en) and AINIC/Pollara NICs are present across the fleet;
   provision for both. See `nic_driver_version.py` (item 7 above) and
   `nic_firmware_check.py` (item 5) as independent, vendor-specific
   checks rather than one superseding the other.
4. **Golden-value defaults for PFC/QoS/DCQCN** — CONFIRMED: ship a
   built-in default golden-value table (derived from the source
   `validate_*.sh` scripts) as the out-of-the-box default for AINIC
   deployments generally. Do not brand these as deployment-specific
   (e.g. no "IREN" naming/comments in code or docs) — present them as
   the standard AINIC defaults, fully overridable per deployment via
   config.
5. **Ansible inventory group naming (`rccl_nodes`/`all`/`core42`)** —
   CONFIRMED not needed. Ported checks use CVS's own `cluster_dict`
   (flat `node_dict`) exclusively; no need to reproduce Ansible
   inventory group semantics.
6. **`clschkompi` (mpirun presence)** — CONFIRMED drop, no port.

## Incremental change: per-vendor NIC config (nic_driver_version, nic_firmware)

### Goal

Restructure the two single-vendor NIC checks - `nic_driver_version`
(currently Broadcom-only) and `nic_firmware` (currently AINIC-only) - into
config blocks with one sub-block per NIC vendor (`ainic`, `broadcom`,
`mellanox`), activated by an outer `nic_type` selector list, so a mixed-vendor
fleet can validate whichever vendors are actually present without needing
separate config files or separate check names. This is a request from the
repo owner, given verbatim:

> "the nic_driver_version should have sub-blocks for AINIC, broadcom nics,
> mellanox nic. and the user should be able to activate the sub-block by
> choosing the nic type at the outer block. similarly for nic_firmware
> version."

### Context (file:line citations)

Current flat, single-vendor shapes:

- `cvs/lib/preflight/nic_driver_version.py:14-113` - `NicDriverVersionCheck`
  hardcodes Broadcom `bnxt_re`/`bnxt_en` `modinfo`/DKMS parsing; non-Broadcom
  nodes are detected per-node (`lsmod | grep '^bnxt_re'`, `nic_driver_version.py:40-43`)
  and reported `SKIPPED`, not FAILed - this per-node vendor-presence
  detection is exactly the mixed-fleet safety net the new dispatcher design
  (below) generalizes to AINIC/Mellanox.
- `cvs/lib/preflight/nic_firmware_check.py:16-143` - `NicFirmwareCheck`
  hardcodes AINIC `ibv_devices`/`nicctl show version {firmware,host-software}`
  parsing; no per-node vendor detection today (a node reporting 0 `ionic_*`
  devices is unconditionally FAILed, `nic_firmware_check.py:108-109`) because
  the original `ansible/ainicfwcheck/fwcheck.yml` assumed an AINIC-only
  fleet.
- `cvs/parsers/schemas.py:1020-1030` - `PreflightNicDriverVersionConfig`
  (flat: `enabled`, `expected_bnxt_re_version`, `expected_bnxt_en_version`),
  nested under `PreflightNodeCheckConfig.nic_driver_version`
  (`schemas.py:1053-1055`).
- `cvs/parsers/schemas.py:1207-1221` - `PreflightNicFirmwareConfig` (flat:
  `enabled`, `expected_nic_count`, `expected_fw_version`,
  `expected_host_version`), nested under
  `PreflightIfoeConfig.nic_firmware` (`schemas.py:1306-1309`).
- `cvs/input/config_file/preflight/preflight_config.json:67-78` -
  `preflight.node_check.nic_driver_version` flat Broadcom fields.
- `cvs/input/config_file/preflight/preflight_config.json:187-201` -
  `preflight.connectivity_check.ifoe.nic_firmware` flat AINIC fields.
- `cvs/tests/preflight/preflight_checks.py:320-336` (`_nic_driver_version_config`/
  `_nic_driver_version_enabled`) and `:221-237` (`_nic_firmware_config`/
  `_nic_firmware_enabled`) - the unknown-key allowlists that must be updated
  to the new nested keys, plus `:866-905` (`test_nic_driver_version`) and
  `:819-864` (`test_nic_firmware`) - the instantiation call sites.
- `cvs/lib/preflight/report.py:708-718` (`_summarize_nic_firmware_results`/
  `_summarize_nic_driver_version_results`, both thin wrappers around
  `_summarize_simple_check_results`, `report.py:618-673`) and
  `report.py:2423-2427` (`_generate_nic_firmware_html`/
  `_generate_nic_driver_version_html`, wrappers around
  `_generate_simple_check_html`, `report.py:2371-2409`) - both generic
  helpers only inspect each node's top-level `status`/`errors` keys and
  ignore any other keys in the per-node dict; **already proven** by the
  precedent at `preflight_checks.py:1049-1178` (`test_ainic_pfc_qos_dcqcn`),
  which runs three independent lib classes (`PfcValidationCheck`,
  `QosValidationCheck`, `DcqcnValidationCheck`) and merges their per-node
  results into one dict shaped `{status, pfc, qos, dcqcn, errors}` - extra
  nested keys pass through the generic report helpers untouched. This is
  the direct template for the per-vendor merge design below (though
  `pfc_qos_dcqcn` uses its own custom summarizer for its `BLOCKED`/admission
  semantics, `report.py:720-728`, which do not apply here).
- `cvs/lib/preflight/unittests/test_nic_driver_version.py:1-122` and
  `cvs/lib/preflight/unittests/test_nic_firmware_check.py:1-112` - existing
  unit tests, all against the current flat single-class shape; need
  rewriting against the new per-vendor classes + dispatcher.
- `cvs/parsers/schemas.py:875-916` (`normalize_legacy_preflight_rdma_config`)
  - the one precedent for a config-migration shim in this codebase. It
  exists to move exactly two deprecated, already-shipped keys
  (RDMA-related aliases under `node_check`) to their canonical location
  with a `FutureWarning`, because RDMA config predates the current
  `node_check`/`connectivity_check` split and had real deployed configs to
  protect. It is a targeted, one-time compatibility shim for a specific
  historical key, **not** a "preflight always migrates legacy shapes
  forever" project convention - there is no generic migration framework,
  and no other flat-to-nested rename in this codebase (e.g. the
  `ssh_mesh`, `pfc_qos_dcqcn`, `limits_conf` additions) came with a
  migration shim; they simply shipped as new, opt-in, default-`false`
  blocks. Given nic_driver_version/nic_firmware are already opt-in and
  default-`false`, and per root `CLAUDE.md`'s "Active Worktree" section
  this whole suite is still being actively built out (not GA'd to
  external customers), the same no-shim precedent applies here.
- Searched `/home/ahskabir/multinode_team/ValidationAutomation/scripts/MultiNodeValidation/precheck/{pdsh,ansible}`
  case-insensitively for `mellanox`, `mlx5`, `mlx4`, `mlx`, `ofed`: **zero
  matches**, and no file/directory name contains `mellanox`/`mlx` either
  (confirmed via `grep -rli` and `find -iname`). The only vendor-named
  material in that tree is Broadcom (`pdsh/clschkbrcm`) and AINIC/ionic
  (`ansible/ainicfwcheck/`, `ansible/ainicvalidation/`). **There is no
  Mellanox source material to port.** The Mellanox sub-blocks below are new,
  designed by analogy to the Broadcom/AINIC fields, and are explicitly
  unvalidated against real Mellanox hardware.

### New config shape

```json
"node_check": {
  "nic_driver_version": {
    "_comment": "Per-vendor NIC driver version + provenance validation. Activate one or more vendor sub-blocks via nic_type. Opt-in, disabled by default.",

    "enabled": false,
    "_comment_enabled": "Enable NIC driver version validation.",

    "nic_type": ["broadcom"],
    "_comment_nic_type": "One or more of: ainic, broadcom, mellanox. Selects which vendor sub-block(s) below actually run. A node lacking the selected vendor's hardware is reported SKIPPED, not FAILed.",

    "ainic": {
      "expected_ionic_driver_version": "1.117.5-a-56",
      "_comment_expected_ionic_driver_version": "Expected 'ionic' kernel module version (modinfo -F version ionic).",

      "expected_ionic_rdma_driver_version": "1.117.5-a-56",
      "_comment_expected_ionic_rdma_driver_version": "Expected 'ionic_rdma' kernel module version (modinfo -F version ionic_rdma)."
    },

    "broadcom": {
      "expected_bnxt_re_version": "236.1.155.0",
      "_comment_expected_bnxt_re_version": "Expected bnxt_re kernel module version.",

      "expected_bnxt_en_version": "1.10.3-236.1.155.0",
      "_comment_expected_bnxt_en_version": "Expected bnxt_en kernel module version."
    },

    "mellanox": {
      "_comment": "NEW, UNVALIDATED against real Mellanox hardware -- no bash/ansible source material exists to port from; designed by analogy to the broadcom/ainic sub-blocks.",

      "expected_mlx5_core_version": "<changeme>",
      "_comment_expected_mlx5_core_version": "Expected mlx5_core kernel module version (modinfo -F version mlx5_core).",

      "expected_ofed_version": "<changeme>",
      "_comment_expected_ofed_version": "Expected MLNX_OFED stack version (ofed_info -s)."
    }
  }
},

"connectivity_check": {
  "ifoe": {
    "nic_firmware": {
      "_comment": "Per-vendor NIC firmware/host-software validation. Activate one or more vendor sub-blocks via nic_type. Opt-in, disabled by default.",

      "enabled": false,
      "_comment_enabled": "Enable NIC firmware/host-software validation.",

      "nic_type": ["ainic"],
      "_comment_nic_type": "One or more of: ainic, broadcom, mellanox. Selects which vendor sub-block(s) below actually run.",

      "ainic": {
        "expected_nic_count": 8,
        "_comment_expected_nic_count": "Exact number of AINIC RDMA devices expected on every node (FAIL on mismatch). Unchanged from today's shipped behavior: no per-node vendor-presence SKIP for AINIC -- see Design decisions.",

        "expected_fw_version": "1.117.5-a-56",
        "_comment_expected_fw_version": "Expected AINIC firmware version (WARNING on mismatch).",

        "expected_host_version": "1.117.5-a-56",
        "_comment_expected_host_version": "Expected AINIC host-software version (WARNING on mismatch)."
      },

      "broadcom": {
        "_comment": "NEW: not present in the original ansible/ainicfwcheck/fwcheck.yml (AINIC-only). Adds symmetric firmware validation for Broadcom NICs, detected per-node so mixed fleets don't false-FAIL on non-Broadcom nodes.",

        "expected_nic_count": 2,
        "_comment_expected_nic_count": "Exact number of Broadcom bnxt RDMA devices expected on every node with Broadcom hardware present (FAIL on mismatch).",

        "expected_fw_version": "<changeme>",
        "_comment_expected_fw_version": "Expected Broadcom NIC firmware version, e.g. from 'ethtool -i <bnxt_en iface>' firmware-version field (WARNING on mismatch)."
      },

      "mellanox": {
        "_comment": "NEW, UNVALIDATED against real Mellanox hardware -- no bash/ansible source material exists to port from; designed by analogy to the ainic/broadcom sub-blocks.",

        "expected_nic_count": 8,
        "_comment_expected_nic_count": "Exact number of Mellanox mlx5 RDMA devices expected on every node with Mellanox hardware present (FAIL on mismatch).",

        "expected_fw_version": "<changeme>",
        "_comment_expected_fw_version": "Expected Mellanox NIC firmware version, e.g. from 'mlxfwmanager --query' or 'ethtool -i <mlx5 iface>' firmware-version field (WARNING on mismatch)."
      }
    }
  }
}
```

### Design decisions

1. **`nic_type` shape: list of strings, not a single string.** A cluster
   could in principle be single-vendor today, but
   `nic_driver_version.py`'s existing per-node SKIP-on-non-Broadcom path
   (`nic_driver_version.py:74-82`) already assumes a fleet *can* be mixed
   (e.g. some racks Broadcom, some AINIC/Pollara). A list lets one config
   validate against several vendors in one pass (`["broadcom", "mellanox"]`)
   without needing multiple config files or multiple check names. Values
   are validated against the closed set `{ainic, broadcom, mellanox}`;
   duplicates and unknown values raise `ValueError` at config load,
   matching this codebase's existing "reject unknown option" style
   (`preflight_checks.py:114-115`, `:130-131`). Default is a single-element
   list matching each check's current shipped vendor
   (`nic_driver_version.nic_type` defaults to `["broadcom"]`,
   `nic_firmware.nic_type` defaults to `["ainic"]`) so that if an existing
   deployment flips `enabled: true` without otherwise touching the config,
   behavior is unchanged from today.

2. **Module structure: decompose into per-vendor `PreflightCheck` subclasses
   sharing a small base, plus a thin dispatcher class that merges results.**
   This mirrors the already-proven `ainic_pfc_qos_dcqcn.py` /
   `test_ainic_pfc_qos_dcqcn` pattern (three independent lib classes, merged
   per-node by the caller into one result dict - see Context above) rather
   than inventing a new "one big class with an if/elif per vendor inside
   `run()`" shape. Concretely, in `nic_driver_version.py`:
   - `_VendorDriverVersionCheck(PreflightCheck)` - shared abstract base:
     `run()` calls `self.phdl.exec(self._build_command())` once and, per
     node, calls `self._parse_and_evaluate(output)` (subclass-provided) to
     get `(status, detail_dict, errors)`; stores
     `{'status', 'vendor', 'detail', 'errors'}` per node. This deduplicates
     the `phdl.exec` + per-node-loop boilerplate that today is copy-pasted
     across `nic_driver_version.py` and `nic_firmware_check.py`.
   - `BroadcomDriverVersionCheck`, `AinicDriverVersionCheck` (new),
     `MellanoxDriverVersionCheck` (new) - each subclasses the base, keeps
     its own per-node vendor-presence detection (e.g. `lsmod | grep
     '^bnxt_re'` / `'^ionic_rdma'` / `'^mlx5_core'`) so a node without that
     vendor's hardware reports `SKIPPED`, matching today's Broadcom
     behavior.
   - `NicDriverVersionCheck(PreflightCheck)` - the dispatcher kept as the
     public/imported name (no call-site rename needed beyond constructor
     args): takes `nic_types: list[str]` and `vendor_configs: dict[str,
     dict]`, instantiates only the configured vendor classes, runs each,
     and merges into `self.results[node] = {'status': ..., 'errors': [...],
     **{vendor: per_vendor_result_for_node}}` - `status` is `FAIL` if any
     configured vendor sub-check is `FAIL` (not expected for driver-version
     checks today, but keeps the aggregation rule symmetric with
     `nic_firmware`), else `WARNING` if any is `WARNING`, else `SKIPPED` if
     *all* configured vendors are `SKIPPED` on that node, else `PASS`.
     `errors` is the concatenation of every configured vendor's `errors`
     list. This keeps the per-node dict's top-level shape
     (`status`/`errors`) exactly what `_summarize_simple_check_results`/
     `_generate_simple_check_html` already expect (see Report.py wiring
     impact below).
   - Same structure in `nic_firmware_check.py`:
     `_VendorFirmwareCheck` base, `AinicFirmwareCheck` (existing logic,
     unchanged - see decision 3 below), `BroadcomFirmwareCheck` (new),
     `MellanoxFirmwareCheck` (new), and a `NicFirmwareCheck` dispatcher with
     the same merge rule (`FAIL` > `WARNING` > all-`SKIPPED` > `PASS`).
   - Reasoning for this shape over a single monolithic dispatch-inside-`run()`
     class: each vendor's command string, parsing, and golden-value
     comparison is independently unit-testable (mirrors
     `ainic_pfc_qos_dcqcn.py`'s three classes each getting their own test
     coverage), and adding a fourth vendor later means adding one class, not
     editing a growing `if vendor == ...` chain inside one `run()`.

3. **Per-node vendor detection vs. trusting the config's declared
   `nic_type` list.** Decision: **do both**, in a layered way - `nic_type`
   at the outer block decides which vendor sub-checks the dispatcher even
   attempts to run (config-declared, not autodetected - if `nic_type` is
   `["broadcom"]`, `AinicDriverVersionCheck`/`MellanoxDriverVersionCheck`
   never run at all, so a purely-Broadcom fleet incurs zero AINIC/Mellanox
   probe commands); but *within* a run of a given vendor's check, per-node
   hardware-presence detection (the existing `lsmod | grep '^bnxt_re'`
   pattern) still decides `SKIPPED` vs. evaluated `PASS`/`WARNING`/`FAIL`
   per node. This preserves today's mixed-fleet safety net (a node that
   happens to lack the declared vendor's hardware is `SKIPPED`, not
   spuriously `FAIL`ed or crashed on empty `modinfo` output) while letting
   the config scope which vendors are worth probing at all - e.g. a
   pure-AINIC deployment never has to see Broadcom `modinfo` commands show
   up in its logs. `AinicFirmwareCheck` also gains this same per-node
   hardware-presence detection (see Open question 2's resolution below) so
   the FAIL-vs-SKIPPED rule is now consistent across all six vendor
   classes.

4. **Migration/backward compatibility: clean break, no legacy-shape shim.**
   `normalize_legacy_preflight_rdma_config` (`schemas.py:875-916`) is the
   only migration precedent in this codebase, and it is a narrow, one-time
   shim for two specific deprecated RDMA keys that predate the
   `node_check`/`connectivity_check` split and had real config files to
   protect. It is not evidence of a general "preflight always migrates"
   convention - every other nested-block addition in this suite
   (`ssh_mesh`, `pfc_qos_dcqcn`, `limits_conf`, `etc_hosts`, ...) shipped as
   a brand-new opt-in block with no shim for anything, because there was no
   previous shape to be compatible with. `nic_driver_version` and
   `nic_firmware` are in the same position here: both are already
   `enabled: false` by default, and per root `CLAUDE.md`'s "Active
   Worktree" section this whole suite is still under active construction,
   not GA'd externally. Decision: **no migration shim** - old flat keys
   (`expected_bnxt_re_version` directly under `nic_driver_version`, etc.)
   simply become unsupported/rejected (`extra="forbid"` at the pydantic
   level, plus updated allowlists in `preflight_checks.py`'s
   `_nic_driver_version_config`/`_nic_firmware_config`). The repo's own
   default config (`cvs/input/config_file/preflight/preflight_config.json`)
   and the out-of-repo `CVS_features/config/preflight_config.json` copy
   (per root `CLAUDE.md`'s own instructions for keeping that copy in sync)
   are the only two configs known to exist today and both get updated as
   part of this change's Steps below.

5. **Mellanox field design (new, unvalidated).** No bash/ansible source
   material exists (confirmed above), so Mellanox fields are designed
   purely by analogy:
   - Driver version (`node_check.nic_driver_version.mellanox`):
     `expected_mlx5_core_version` (the `mlx5_core` kernel module version,
     checked the same way as `bnxt_re`/`ionic` via `modinfo -F version
     mlx5_core`) and `expected_ofed_version` (the MLNX_OFED stack version
     via `ofed_info -s`, since Mellanox NICs are conventionally paired with
     the vendor's own OFED driver stack rather than only an in-kernel
     module - there is no Broadcom/AINIC equivalent of this field, it is
     Mellanox-specific).
   - Firmware (`connectivity_check.ifoe.nic_firmware.mellanox`):
     `expected_nic_count` and `expected_fw_version`, mirroring the
     AINIC/Broadcom firmware shape; firmware version would be read via
     `mlxfwmanager --query` or `ethtool -i <mlx5 iface>`'s
     `firmware-version` field. Every default value in this sub-block is a
     `<changeme>` placeholder rather than a guessed real version string,
     to avoid silently shipping a plausible-looking but fabricated
     "expected" firmware version that could mask a real mismatch. This
     sub-block must be explicitly called out in the README and code
     comments as new/unvalidated (already reflected in the JSON shape
     above) - unlike the AINIC PFC/QoS/DCQCN golden-value defaults (Decision
     4 in the existing Decisions section above), which *were* derived from
     real source scripts and so were shipped as unbranded, sensible
     defaults, Mellanox has no such source to derive real defaults from.

6. **Report.py wiring impact: none required.** Confirmed via the
   `test_ainic_pfc_qos_dcqcn` precedent (Context above) that
   `_summarize_simple_check_results`/`_generate_simple_check_html`
   (`report.py:618-673`, `:2371-2409`) only read each node's top-level
   `status` and `errors` keys and are indifferent to any other keys present
   in that node's dict. Because the `NicDriverVersionCheck`/
   `NicFirmwareCheck` dispatchers (decision 2) are designed to always
   produce `{node: {'status': ..., 'errors': [...], <vendor>: {...}, ...}}`
   - i.e. they preserve the existing per-node dict contract and merely add
   per-vendor sub-keys - `_summarize_nic_firmware_results`/
   `_summarize_nic_driver_version_results` and
   `_generate_nic_firmware_html`/`_generate_nic_driver_version_html`
   (`report.py:708-718`, `:2423-2427`) need **no code changes**. This is a
   deliberate design constraint on the dispatcher merge logic (decision 2),
   not an accident - any alternative merge shape that dropped or renamed
   the top-level `status`/`errors` keys would require new report.py
   wrappers.

### Steps

1. **`cvs/parsers/schemas.py`** - Split `PreflightNicDriverVersionConfig`
   into: `PreflightAinicDriverVersionConfig` (new;
   `expected_ionic_driver_version`, `expected_ionic_rdma_driver_version`),
   `PreflightBroadcomDriverVersionConfig` (existing two fields, renamed from
   `PreflightNicDriverVersionConfig`), `PreflightMellanoxDriverVersionConfig`
   (new; `expected_mlx5_core_version`, `expected_ofed_version`, both
   `<changeme>` defaults), and an outer `PreflightNicDriverVersionConfig`
   with `enabled: bool`, `nic_type: List[str]` (default `["broadcom"]`) plus
   a `field_validator`/`model_validator` rejecting unknown vendor names,
   duplicates, and (when `enabled=True`) an empty list, and the three
   `ainic`/`broadcom`/`mellanox` sub-model fields. Mirror the same
   restructuring for `PreflightNicFirmwareConfig` ->
   `PreflightAinicFirmwareConfig` (existing 3 fields),
   `PreflightBroadcomFirmwareConfig` (new; `expected_nic_count`,
   `expected_fw_version`), `PreflightMellanoxFirmwareConfig` (new, same
   shape, `<changeme>` defaults), outer `PreflightNicFirmwareConfig` with
   `enabled`, `nic_type` (default `["ainic"]`), and the three sub-models.
   All new/renamed sub-models keep `model_config = ConfigDict(extra="forbid")`.
2. **`cvs/lib/preflight/nic_driver_version.py`** - Introduce
   `_VendorDriverVersionCheck` base (shared `run()` calling
   `self.phdl.exec` + per-node parse/evaluate hook), rename the current
   Broadcom logic into `BroadcomDriverVersionCheck(_VendorDriverVersionCheck)`
   with no behavior change, add `AinicDriverVersionCheck` and
   `MellanoxDriverVersionCheck` (new), and replace the current
   `NicDriverVersionCheck` with a dispatcher class of the same name per
   decision 2 (`nic_types`, `vendor_configs` constructor args; merges
   per-vendor per-node results).
3. **`cvs/lib/preflight/nic_firmware_check.py`** - Same restructuring:
   `_VendorFirmwareCheck` base, `AinicFirmwareCheck` (existing
   `ibv_devices`/`nicctl` parsing, PLUS new per-node presence detection per
   Open question 2's resolution - `lsmod | grep -E '^ionic(_rdma)?'` before
   evaluating device count/firmware, so a node with no `ionic*` module
   reports `SKIPPED` instead of FAIL-on-zero-devices), `BroadcomFirmwareCheck`
   and `MellanoxFirmwareCheck` (new, with the same per-node vendor-presence
   detection pattern), dispatcher `NicFirmwareCheck` merging per-vendor
   results.
4. **`cvs/input/config_file/preflight/preflight_config.json`** - Replace
   both flat blocks (lines 67-78 and 187-201) with the nested shape shown
   above, including `_comment*` keys consistent with this file's existing
   documentation-via-comments style.
5. **`cvs/tests/preflight/preflight_checks.py`** - Update
   `_nic_driver_version_config` (`:320-332`) and `_nic_firmware_config`
   (`:221-233`) unknown-key allowlists to `{'enabled', 'nic_type', 'ainic',
   'broadcom', 'mellanox'}`; update `test_nic_driver_version` (`:866-905`)
   and `test_nic_firmware` (`:819-864`) to read `nic_type` plus the three
   vendor sub-dicts from config and construct the dispatcher classes with
   `nic_types=`/`vendor_configs=` instead of the old flat kwargs. No change
   needed to `preflight_results` keying, tiering, or pruning behavior - both
   checks stay opt-in (`enabled: false` default) and non-pruning.
6. **`cvs/lib/preflight/report.py`** - No functional change required (see
   design decision 6). Optionally reword the two summary label strings
   (`'AINIC NIC firmware/count validation'` / `'NIC driver version
   validation'` in `_summarize_nic_firmware_results`/
   `_summarize_nic_driver_version_results`, `report.py:708-718`) to drop the
   now-inaccurate `AINIC`-only phrasing, e.g. `'NIC firmware/count
   validation'` (driver-version label is already vendor-neutral).
7. **`cvs/lib/preflight/unittests/test_nic_driver_version.py`** - Rewrite
   against the new classes: one test class per vendor checker
   (`BroadcomDriverVersionCheck`, `AinicDriverVersionCheck`,
   `MellanoxDriverVersionCheck`) covering PASS/WARNING/SKIPPED/malformed
   output per vendor (porting the existing Broadcom cases verbatim onto
   `BroadcomDriverVersionCheck`), plus a new test class for the
   `NicDriverVersionCheck` dispatcher covering: single-vendor `nic_type`
   (behavior-identical to today), multi-vendor `nic_type` merge (FAIL/WARNING/
   SKIPPED aggregation rules from decision 2), and a vendor absent on a
   given node (per-node SKIPPED still surfaces through the merge).
8. **`cvs/lib/preflight/unittests/test_nic_firmware_check.py`** - Same
   restructuring: per-vendor test classes (porting existing AINIC cases
   verbatim onto `AinicFirmwareCheck`, new cases for
   `BroadcomFirmwareCheck`/`MellanoxFirmwareCheck`), plus dispatcher-merge
   tests for `NicFirmwareCheck`.
9. **`cvs/tests/preflight/README.md`** - Update the two check sections
   (lines ~14-15, 87-90, 112, 142, 153-156, 191-196, 220, 225, 533-537) to
   document the `nic_type` selector, the three vendor sub-blocks per check,
   and explicitly flag the Mellanox sub-blocks as new/unvalidated per
   decision 5.
10. **Workflow** - `make fmt` -> `make lint` -> `make test` before
    committing, per root `CLAUDE.md`/`CONTRIBUTORS.md` (not run by this
    planning pass; left for the implementation step).

### Files touched

- `cvs/parsers/schemas.py` - split `PreflightNicDriverVersionConfig`/
  `PreflightNicFirmwareConfig` into outer selector + `ainic`/`broadcom`/
  `mellanox` sub-models, each with `enabled`-independent `nic_type` gating.
- `cvs/lib/preflight/nic_driver_version.py` - add `_VendorDriverVersionCheck`
  base, `BroadcomDriverVersionCheck` (renamed from current
  `NicDriverVersionCheck`), new `AinicDriverVersionCheck`/
  `MellanoxDriverVersionCheck`, new dispatcher `NicDriverVersionCheck`.
- `cvs/lib/preflight/nic_firmware_check.py` - add `_VendorFirmwareCheck`
  base, rename current logic to `AinicFirmwareCheck`, new
  `BroadcomFirmwareCheck`/`MellanoxFirmwareCheck`, new dispatcher
  `NicFirmwareCheck`.
- `cvs/input/config_file/preflight/preflight_config.json` - nested
  `nic_type` + per-vendor shape for both blocks.
- `cvs/tests/preflight/preflight_checks.py` - updated config-key allowlists
  and dispatcher instantiation in `_nic_driver_version_config`,
  `_nic_firmware_config`, `test_nic_driver_version`, `test_nic_firmware`.
- `cvs/lib/preflight/report.py` - optional label-text touch-up only; no
  structural change.
- `cvs/lib/preflight/unittests/test_nic_driver_version.py` - rewritten for
  per-vendor classes + dispatcher.
- `cvs/lib/preflight/unittests/test_nic_firmware_check.py` - rewritten for
  per-vendor classes + dispatcher.
- `cvs/tests/preflight/README.md` - documents `nic_type` selector and the
  three vendor sub-blocks per check, flags Mellanox as new/unvalidated.
- (Out of repo, at real-hardware-smoke-test time only, per root
  `CLAUDE.md`'s existing instruction to keep it in sync)
  `/home/ahskabir/multinode_team/DEV/CVS_features/config/preflight_config.json`
  - copy of the updated default config.

### Testing

- **Unit tests**: extend/rewrite the two files under
  `cvs/lib/preflight/unittests/` as described in Steps 7-8. Cover, for each
  of the six new/renamed vendor classes: hardware-present PASS, version
  mismatch WARNING (or count mismatch FAIL for firmware), hardware-absent
  SKIPPED, and malformed/empty command output. Cover the two dispatcher
  classes' merge logic directly (not just through the vendor classes):
  single-vendor `nic_type` (must reproduce today's exact result shape for a
  Broadcom-only / AINIC-only config, since decision 1's defaults are chosen
  specifically to make this the no-op case), multi-vendor `nic_type` with
  one vendor WARNING/FAIL and another PASS/SKIPPED (verify the FAIL >
  WARNING > all-SKIPPED > PASS precedence), and an empty reachable-hosts
  dict.
- **Config schema tests**: add/extend pydantic-level tests for
  `PreflightNicDriverVersionConfig`/`PreflightNicFirmwareConfig` covering:
  default `nic_type` values, rejection of an unknown vendor name in
  `nic_type`, rejection of duplicate entries, rejection of an empty
  `nic_type` list when `enabled=True`, and rejection of unknown keys inside
  each vendor sub-block (`extra="forbid"`).
- **`make fmt-check` -> `make lint` -> `make test`** after implementation,
  per `CONTRIBUTORS.md`'s documented workflow.
- **Report/HTML smoke test**: run `preflight_checks.py`'s existing
  mocked/unit-test coverage (or a small synthetic cluster) with both checks
  enabled and a multi-vendor `nic_type` list to confirm the HTML report
  renders correctly with no exceptions - this is expected to need **no**
  report.py changes per decision 6, so this step is primarily a regression
  check that the merge shape assumption holds in practice, not a design
  validation.
- **Real-hardware smoke-test caveats**: only the `broadcom`/`ainic` sub-blocks
  can be validated against real hardware today, using the existing
  `CVS_features` harness flow (root `CLAUDE.md`'s "Running the pairwise
  RCCL test against real hardware" section) - enable one vendor at a time
  in `nic_type`, confirm no `<changeme>` placeholders remain for the
  vendors actually enabled. The `mellanox` sub-block **cannot** be
  real-hardware validated in this environment (no Mellanox nodes in the
  known cluster inventory, no source scripts to cross-check parsing
  against) - it should ship reachable only via explicit, documented opt-in
  (`nic_type: ["mellanox", ...]`) and stay flagged new/unvalidated in the
  README until someone with Mellanox hardware access confirms the
  `modinfo`/`ofed_info`/`mlxfwmanager` command output format assumptions.

### Open questions

1. **Mellanox command/field format is unverified.** `expected_mlx5_core_version`/
   `expected_ofed_version`/`expected_fw_version`'s exact source commands
   (`modinfo -F version mlx5_core`, `ofed_info -s`, `mlxfwmanager --query`
   or `ethtool -i`) are a reasonable best guess by analogy to the
   Broadcom/AINIC patterns already in this codebase, but have not been
   run against real Mellanox hardware or cross-checked against any
   Mellanox-specific source script (none exists). Needs sign-off from
   someone with Mellanox/ConnectX lab access before the `mellanox`
   sub-block is exercised for real.
2. **Resolved (repo owner, 2026-08-05): `AinicFirmwareCheck` gains per-node
   vendor-presence detection too, for symmetry.** Overriding decision 3/4's
   original "leave AINIC firmware unchanged" call: the entire point of the
   `nic_type` selector is to let mixed-vendor fleets validate more than one
   vendor in a single run, and leaving `AinicFirmwareCheck`'s NIC-count-0 as
   an unconditional FAIL would produce a false FAIL on every non-AINIC node
   the moment an operator actually sets `nic_type: ["ainic", "broadcom"]` -
   exactly the scenario this change is meant to support. Fix: `ibv_devices`
   returning 0 AINIC devices is only a FAIL when independent evidence of
   AINIC hardware presence exists on that node (mirror the
   `bnxt_re`/`ionic_rdma` `lsmod`-presence pattern used elsewhere - e.g. an
   `ionic`/`ionic_rdma` kernel module present but `ibv_devices` reporting
   zero RDMA devices is a real FAIL; no `ionic*` module at all is `SKIPPED`).
   This is a pure widening of when `SKIPPED` applies (a node with zero
   AINIC-driver evidence today already has 0 `ibv_devices` output, so a
   single-vendor `nic_type: ["ainic"]` deployment across an already-uniform
   AINIC fleet sees no behavior change - see Steps below for the exact
   implementation).

### Incremental fix: validate vendor sub-block types in nic_type config

### Goal

A second evaluator pass, run after the per-vendor NIC config change above
had already landed (including its first round of evaluator-found-bug
fixes), found two further confirmed bugs in the same area: a config where a
selected vendor's sub-block (`ainic`/`broadcom`/`mellanox`) is not a JSON
object - e.g. a stray integer or a typo'd string where an object was
intended - either crashes with a raw, unhelpful `TypeError` deep inside the
check function, or, worse, is silently swallowed with the operator's
intended override discarded and no error or warning at all. Both were
reproduced directly against the working tree (not just theorized). Fix:
extend the existing `nic_type` validation so malformed vendor sub-blocks are
rejected at config-load time with a clear `ValueError`, consistent with
every other config-shape check in this file.

### Context (file:line citations)

- `cvs/tests/preflight/preflight_checks.py:53-64` - `_validate_nic_type(config,
  config_path, default)`, added by the per-vendor NIC config change above.
  Validates only the `nic_type` selector list itself (must be a list,
  vendor names must be in `_VALID_NIC_VENDORS`, no duplicates, non-empty
  when `enabled` is true). It has no knowledge of, and never inspects, the
  actual per-vendor sub-block values (`config.get('ainic')`,
  `config.get('broadcom')`, `config.get('mellanox')`) even though it
  already holds both the full `config` dict and the vendor-name universe
  needed to check them.
- `cvs/tests/preflight/preflight_checks.py:249-260` (`_nic_firmware_config`)
  and `:347-358` (`_nic_driver_version_config`) - both call
  `_validate_nic_type(config, <path>, <default>)` as their last step, after
  their own unknown-top-level-key allowlist check
  (`{'enabled', 'nic_type', 'ainic', 'broadcom', 'mellanox'}`). Neither
  function itself checks that `config.get('ainic')`/`'broadcom'`/`'mellanox'`
  is a `dict` - the allowlist check only confirms the *key name* is
  recognized, not that its *value* has the right shape. A config like
  `{"enabled": true, "nic_type": ["broadcom"], "broadcom": 999}` or
  `{"enabled": true, "nic_type": ["broadcom"], "broadcom": "oops"}` passes
  both functions cleanly today.
- `cvs/tests/preflight/preflight_checks.py:845-891` (`test_nic_firmware`)
  and `:894-937` (`test_nic_driver_version`) - both build a `vendor_configs`
  dict via the same shape of comprehension, e.g.
  (`test_nic_firmware`, `:865-873`):
  ```python
  vendor_configs = {
      vendor: {
          kwarg: nic_firmware_config.get(vendor, {}).get(kwarg)
          for kwarg in kwargs
          if kwarg in nic_firmware_config.get(vendor, {})
      }
      for vendor, kwargs in _NIC_FIRMWARE_VENDOR_KWARGS.items()
      if vendor in nic_types
  }
  ```
  (`test_nic_driver_version`, `:916-924`, is byte-for-byte the same pattern
  against `_NIC_DRIVER_VERSION_VENDOR_KWARGS`/`nic_driver_version_config`.)
  Two distinct failure modes come out of this same comprehension depending
  on the malformed value's type:
  - **Bug 1 (crash).** `nic_firmware_config.get(vendor, {})` only falls back
    to `{}` when the key is *absent*; if `vendor` is present with value
    `999`, `.get(vendor, {})` returns `999` itself. The comprehension's `if
    kwarg in nic_firmware_config.get(vendor, {})` clause then evaluates
    `kwarg in 999`, and Python's dict/list-comprehension semantics evaluate
    the `if` clause before the value expression for each loop item - so this
    raises `TypeError: argument of type 'int' is not iterable` on the very
    first `kwarg`, as an uncaught exception deep inside `test_nic_firmware`/
    `test_nic_driver_version` rather than a clean, actionable config error.
  - **Bug 2 (silent data loss).** If the sub-block is a *string* instead
    (e.g. `"broadcom": "oops"`), `kwarg in "oops"` is valid Python (a
    substring test) and evaluates to `False` for every real kwarg name (none
    of `expected_nic_count`/`expected_fw_version`/etc. are substrings of
    `"oops"`). No exception is raised at all; the comprehension silently
    produces `vendor_configs['broadcom'] = {}`. The operator's intended
    override is discarded with zero errors or warnings, and
    `NicFirmwareCheck`/`NicDriverVersionCheck` quietly fall back to their
    own hardcoded defaults - a silent misconfiguration that could mask a
    real firmware/driver-version mismatch on every node.
  - Both reproduced directly by the evaluator against the current working
    tree (not merely reasoned about from reading the code).
- `_limits_conf_config` (`preflight_checks.py:234-242`) is this file's
  existing precedent for the "raise `ValueError` with a dotted config path
  in the message" convention this fix follows (e.g. `_nic_firmware_config`'s
  own `"preflight.connectivity_check.ifoe.nic_firmware must be an object"`
  for the block-level shape check it already does at `:252-253`) - this fix
  is the same style of check, just one level deeper (per-vendor sub-block
  shape rather than block shape).

### Approach / design decision

**Extend `_validate_nic_type` itself to also validate vendor sub-block
types**, rather than adding a separate helper or duplicating the check
inside `_nic_firmware_config`/`_nic_driver_version_config` individually.
`_validate_nic_type` already receives the full `config` dict and already
owns `_VALID_NIC_VENDORS`, so it is the single natural place to add "and
each vendor key actually present in this dict must map to an object" -
adding it here means both call sites (`_nic_firmware_config`,
`_nic_driver_version_config`) get the fix for free with no change to either
function's own body, exactly mirroring how they already get the `nic_type`
list checks for free today.

Concretely, append to the end of `_validate_nic_type`:

```python
for vendor in sorted(_VALID_NIC_VENDORS & set(config)):
    vendor_block = config[vendor]
    if not isinstance(vendor_block, dict):
        raise ValueError(f"{config_path}.{vendor} must be an object")
```

This raises e.g. `"preflight.connectivity_check.ifoe.nic_firmware.broadcom
must be an object"` / `"preflight.node_check.nic_driver_version.broadcom
must be an object"` - dotted-path style consistent with every other
`ValueError` in this file - for both the `int` case (bug 1, now a clean
`ValueError` instead of an uncaught `TypeError`) and the `str` case (bug 2,
now a clean `ValueError` instead of a silently-empty override). Because
`_validate_nic_type` runs unconditionally inside `_nic_firmware_config`/
`_nic_driver_version_config` (not gated on `enabled` or on which vendors are
in `nic_type`), this also fails config loading fast, before `phdl.exec` or
any node is touched, matching this suite's existing "validation failures
raise before any node is touched" convention (`cvs/CLAUDE.md`, "How a test
run flows", step 4).

**Scope decision: validate every vendor sub-block key present in `config`,
not only the ones currently selected via `nic_type`.** I.e. the loop above
is `_VALID_NIC_VENDORS & set(config)` (any of `ainic`/`broadcom`/`mellanox`
that appear as a key at all), not `set(nic_type) & set(config)` (only the
active selection). Reasoning:
- This file already validates the *whole* config shape eagerly and
  unconditionally elsewhere - e.g. `_node_check_config`'s and
  `_ifoe_config`'s unknown-key checks run regardless of whether the
  corresponding check is `enabled`, and `_validate_nic_type`'s own
  empty-`nic_type`-when-enabled check aside, the vendor-name/duplicate
  checks on `nic_type` itself already run unconditionally too. Scoping the
  new check to "only vendors currently selected" would be the first
  vendor-shape check in this block to depend on `nic_type`'s contents
  rather than just the block's own top-level keys.
- Without this, a malformed-but-currently-inactive sub-block (e.g.
  `nic_type: ["broadcom"]` today with a broken `"mellanox": "oops"` sitting
  unused elsewhere in the same block) would pass validation silently and
  only blow up (or silently misbehave) later, at the moment an operator
  flips `nic_type` to include `mellanox` - the worst possible time to
  discover a typo, mid-fleet-onboarding rather than at config-review time.
  Failing fast on the full block's shape, independent of current selection,
  catches this at the point the config file is authored/edited instead.
- **This is flagged as an open question below** rather than treated as a
  closed call, because "validate everything in the block eagerly" vs.
  "validate lazily, only what's about to be used" is a genuine, arguable
  product decision, and the prompt that requested this fix explicitly asked
  for it to be surfaced rather than silently decided one way.

### Steps

1. **`cvs/tests/preflight/preflight_checks.py`** - extend `_validate_nic_type`
   (`:53-64`) with the vendor-sub-block-shape loop shown above, appended
   after the existing `nic_type`-empty-when-enabled check. No change needed
   to `_nic_firmware_config` (`:249-260`) or `_nic_driver_version_config`
   (`:347-358`) themselves, nor to `test_nic_firmware` (`:845-891`) or
   `test_nic_driver_version` (`:894-937`) - the fix is fully contained in
   the shared helper both config functions already call.
2. **Unit tests** - see Testing below; add regression coverage for both
   failure modes before/alongside the fix.
3. **Workflow** - `make fmt` -> `make lint` -> `make test` before
   committing, per root `CLAUDE.md`/`CONTRIBUTORS.md` (not run by this
   planning pass; left for the implementation step).

### Files touched

- `cvs/tests/preflight/preflight_checks.py` - extend `_validate_nic_type`
  with the vendor-sub-block dict-type check.
- Unit test file(s) under `cvs/lib/preflight/unittests/` - new regression
  tests for both bugs (see Testing).

### Testing

- **No existing unit tests exercise `_validate_nic_type`, `_nic_firmware_config`,
  or `_nic_driver_version_config` directly today** - confirmed by grepping
  `cvs/lib/preflight/unittests/` and the rest of the repo for these three
  function names: they are plain module-level functions in
  `cvs/tests/preflight/preflight_checks.py` (a pytest test module, not a
  pydantic schema class in `cvs/parsers/schemas.py`), and the only existing
  coverage that touches this module at all
  (`test_nic_firmware_check.py`, `test_nic_driver_version.py`,
  `test_rdma_connectivity.py`, `test_ifoe_l2_connectivity.py`,
  `test_scaleup_fabric.py`, `test_transferbench_smoke.py` under
  `cvs/lib/preflight/unittests/`) does so end-to-end, by `from
  cvs.tests.preflight import preflight_checks` and calling a public
  `test_*` function (e.g. `preflight_checks.test_nic_firmware(phdl,
  config_dict)`) with a mocked `phdl` and patched lib check classes,
  asserting on `preflight_checks.preflight_results[...]` afterward. New
  tests for this fix should follow that same established pattern rather
  than inventing a new one, and land in the existing
  `cvs/lib/preflight/unittests/test_nic_firmware_check.py` and
  `test_nic_driver_version.py` (both already slated for rewriting in Steps
  7-8 of the per-vendor NIC config change above, so this fix's tests can be
  added alongside that rewrite rather than as a separate pass) - no new test
  module is needed.
- **Required new cases** (in each of the two files, mirroring the
  firmware/driver-version split):
  - **Bug 1 regression**: a config with a selected vendor's sub-block set to
    a non-dict, non-string value (e.g. `"broadcom": 999`) must now raise
    `ValueError` (via `_validate_nic_type`/`_nic_firmware_config`, or
    end-to-end via `preflight_checks.test_nic_firmware(phdl, config_dict)`
    - use `pytest.raises(ValueError)` / `self.assertRaises(ValueError)`
    around the call), where today it raises an uncaught `TypeError`.
  - **Bug 2 regression**: the same, but with the sub-block set to a string
    (e.g. `"broadcom": "oops"`) - must now raise `ValueError` instead of
    silently succeeding with an empty override dict. This case is the more
    important of the two to get right in the test, since before this fix it
    produced no exception and no wrong-looking output at the call site
    itself - the test needs to assert the `ValueError` is now raised, not
    merely that the old code "didn't crash."
  - A **positive/regression-safety case**: a well-formed config (all
    present vendor sub-blocks are `dict`s) must continue to pass validation
    unchanged, so the new check doesn't accidentally reject legitimate
    configs (e.g. re-run the existing PASS-path tests for both checks after
    the fix lands).
  - If the "validate every vendor key present, not just selected ones" scope
    decision above is confirmed, add one more case: a malformed sub-block
    for a vendor *not* in the active `nic_type` list must still raise
    `ValueError` (proves the check isn't accidentally scoped to only the
    active selection).
- **`make fmt-check` -> `make lint` -> `make test`** after implementation,
  per `CONTRIBUTORS.md`'s documented workflow.

### Open questions

1. ~~Validate all present vendor sub-blocks, or only the ones currently
   selected in `nic_type`?~~ **Resolved (repo owner, 2026-08-05): validate
   all present vendor sub-blocks, eagerly, regardless of `nic_type`
   selection** - i.e. keep the `_VALID_NIC_VENDORS & set(config)` scope from
   the Approach section as-is, not `set(nic_type) & set(config)`. This
   matches every other config-shape check in this file (`_node_check_config`,
   `_ifoe_config`, `_validate_nic_type`'s own vendor-name/duplicate checks on
   `nic_type` itself) already running unconditionally regardless of whether
   the corresponding check is enabled or selected, and avoids the "typo only
   surfaces the moment you flip `nic_type`" surprise for a
   currently-inactive vendor block. Executor: implement exactly as specified
   in the Approach section above, including the "malformed sub-block for a
   non-selected vendor must still raise" test case from Testing.

