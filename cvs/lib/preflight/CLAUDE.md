# cvs/lib/preflight/ — preflight check library conventions

Helper classes for `cvs/tests/preflight/preflight_checks.py`. See
`cvs/tests/preflight/README.md` for the full check-by-check reference (what
each check validates, config keys, troubleshooting) and
`docs/plans/preflight-checks-porting.md` for the bash/ansible → CVS porting
history and status. This file is about the *shared conventions* new checks
must follow.

## The `PreflightCheck` base class (`base.py`)

```python
class PreflightCheck(ABC):
    def __init__(self, phdl, config_dict=None):
        self.phdl = phdl
        self.config_dict = config_dict or {}
        self.results = {}

    @abstractmethod
    def run(self): ...          # returns self.results

    def get_results(self): ...
    def log_info/log_error/log_warning(self, message): ...
```

Every check subclasses `PreflightCheck`, implements only `run()`, stores its
per-node dict into `self.results`, and uses `self.log_info/log_error/
log_warning` for logging (prefixes messages with the class name). Static
helpers also live here and are reused for mesh/sharding checks:
`partition_nodes_into_groups(node_list, group_size)`,
`calculate_resource_requirements(group_size, num_interfaces)`,
`find_host_group(host, groups)` (module-level aliases of the same names are
exported for `cvs.lib.preflight.base.partition_nodes_into_groups`-style
imports).

## The `phdl.exec()` remote-execution convention

`phdl` is a `MultiProcessPssh` (`cvs/lib/parallel/multiprocess_pssh.py`),
built once per test-module run by the `phdl` fixture in
`preflight_checks.py`. `phdl.exec(cmd)` runs `cmd` on every currently
reachable node and returns `{node: stdout}`; every node-targeted check
builds a shell command string and iterates that dict to build its per-node
result (see `gid_consistency.py::_build_gid_check_command`/`run()`,
`version_check.py`, `interface_consistency.py` via
`cvs/lib/linux_utils.py::get_rdma_nic_dict(phdl)`). Reuse this pattern for
any new node-targeted check rather than introducing a pdsh/ssh-loop
equivalent.

**Exception — `node_reachability.py::PingReachabilityCheck`:** ICMP ping is
a *driver-host → node* operation, not node → node, so it cannot go through
`phdl.exec` (which targets remote nodes over SSH and presupposes SSH
already works — ping has to run *before* that assumption holds). This check
instead shells out locally via `subprocess.run(["ping", "-c", ..., "-W", ...,
target])` per node and catches `subprocess.TimeoutExpired`/`OSError` as a
FAIL rather than letting them raise. `UptimeCheck` in the same module is a
normal `phdl.exec("uptime")` broadcast — it only differs from other checks
in that it never produces a FAIL status. When adding a check that must run
before or independent of node SSH reachability, this is the pattern to
follow instead of `phdl.exec`.

## Result-dict shape conventions

Two shapes are in use; pick the simplest one that fits:

1. **Per-node dict** — `{node: {'status': 'PASS'|'FAIL'|'SKIPPED', 'errors':
   [...], ...extra fields}}`. Used by simple, stateless, one-command-per-node
   checks: `gid_consistency.py`, `version_check.py`,
   `interface_consistency.py`, `node_reachability.py`,
   `ssh_mesh_connectivity.py`, `etc_hosts_consistency.py`,
   `limits_conf_check.py`, `nic_driver_version.py`. Default to this shape for
   new checks.
2. **Aggregate dict with coverage** — `{'status': ..., 'node_results': {...},
   'coverage': ..., 'failed_nodes': [...], ...}`. Used only where mesh/pair
   coverage bookkeeping matters: `scaleup_fabric.py::NodeHealthCheck`,
   `ifoe_l2_connectivity.py::IfoeL2ConnectivityCheck`,
   `transferbench_smoke.py::TransferBenchSmokeCheck`,
   `ainic_pfc_qos_dcqcn.py`, `rdma_connectivity.py`. Only reach for this shape
   if the check inherently needs to reason about pairs/groups, not per-node
   in isolation.

## Config-driven `enabled` flag pattern

Config lives at `cvs/input/config_file/preflight/preflight_config.json`
under top-level `preflight.node_check.*` and
`preflight.connectivity_check.{rdma,ifoe,ssh_mesh}.*`, validated by
pydantic models in `cvs/parsers/schemas.py` (`PreflightConfigFile` at the
top, with nested models per check — e.g. `PreflightPingCheckConfig`,
`PreflightEtcHostsConfig`, `PreflightLimitsConfConfig`,
`PreflightNicDriverVersionConfig`, `PreflightNicFirmwareConfig`,
`PreflightPfcConfig`/`PreflightQosConfig`/`PreflightDcqcnConfig`/
`PreflightPfcQosDcqcnConfig`, `PreflightSshMeshConfig`). A flat legacy key
under `node_check`/top-level (e.g. bare `node_health`, `l2ping`,
`transferbench`) is explicitly rejected by
`preflight_checks.py::_reject_flat_preflight_checks` — new config must nest
under `node_check` or `connectivity_check.ifoe`.

Each check reads its `enabled` flag through a small `_<check>_config()` +
`_<check>_enabled()` pair of helpers in `preflight_checks.py`, which call the
shared `get_nested_config(config_dict, section, key, default)` (duplicated
verbatim in both `preflight_checks.py` and `report.py` — reuse one of those
two, do not add a third copy) and normalize bool/string flags via
`_config_flag_enabled(value, default=True)`. New/experimental checks default
`enabled=False` (matching `ping_check`, `etc_hosts`, `limits_conf`,
`nic_driver_version`, `ssh_mesh`, `nic_firmware`, `pfc_qos_dcqcn`); only
flip the default to `True` if the check should run out-of-the-box like
`node_check.enabled` itself.

Failed nodes are pruned from `phdl` (`_prune_nodes_from_phdl`,
`preflight_checks.py`) only when a later check's validity actually depends
on the earlier failure (e.g. node-health admission gating IFoE/RDMA tiers).
Purely diagnostic/informational checks (ping, uptime, SSH mesh, ROCm version
mismatch) must never prune.

## Report wiring (`report.py`, `PreflightReportGenerator`)

Every check needs a `_summarize_<check>_results(self, results)` method (feeds
`_generate_preflight_summary()`'s `checks` dict) and a
`_generate_<check>_html(self, results)` method (called from
`_generate_html_content()`'s section list, only rendered when the check
actually ran/failed — skipped checks don't get a section). For the common
"simple per-node dict, no mesh bookkeeping" shape, don't write a bespoke
pair — reuse the generic
`_summarize_simple_check_results(self, results, label, skip_message,
demote_fail_to_warning=False)` /
`_generate_simple_check_html(self, results, title)` helpers and add a
one-line named wrapper calling them (see `_summarize_ping_reachability_results`,
`_summarize_ssh_mesh_results`, `_summarize_etc_hosts_results`,
`_summarize_limits_conf_results`, `_summarize_nic_firmware_results`,
`_summarize_nic_driver_version_results`, `_summarize_pfc_qos_dcqcn_results`
and their `_generate_*_html` counterparts for the pattern to copy).
`demote_fail_to_warning=True` is how diagnostic-only checks (e.g. ping) avoid
gating overall preflight status while still surfacing per-node failures.

## Unit-test convention (`unittests/`)

One `unittest.TestCase` file per lib module (`test_<module>.py`). Pattern
(see `test_node_reachability.py`, `test_rdma_connectivity.py`):

```python
import os, sys, unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from cvs.lib.preflight.<module> import <CheckClass>
```

`phdl = MagicMock()`; for `phdl.exec()`-based checks, set
`phdl.exec.return_value = {...}`; for `node_reachability.py`'s ping path,
`patch('subprocess.run', ...)` instead. Cover: the PASS path, every
documented FAIL path from the source script/playbook, and malformed/empty
command output.

## Checklist: adding a new preflight check

Touch all of these, in this order, or the check will silently not appear in
config validation / the report / CI:

1. **Lib module** — `cvs/lib/preflight/<new_module>.py`, subclass
   `PreflightCheck`, implement `run()` per the conventions above.
2. **Unit tests** — `cvs/lib/preflight/unittests/test_<new_module>.py`.
3. **Config schema** — add a pydantic sub-model to
   `cvs/parsers/schemas.py` (nested under `PreflightNodeCheckConfig` or
   `PreflightIfoeConfig`/`PreflightConnectivityCheckConfig` as appropriate),
   defaulting `enabled=False` unless it should ship on by default.
4. **Default config** — add the corresponding default block to
   `cvs/input/config_file/preflight/preflight_config.json`.
5. **Test wiring** — add `_<check>_config()`/`_<check>_enabled()` helpers and
   a `test_<check>(phdl, config_dict, ...)` function to
   `cvs/tests/preflight/preflight_checks.py` that stores its result into the
   module-global `preflight_results` dict; slot it into the correct tier
   (see `cvs/tests/preflight/CLAUDE.md` for ordering).
6. **Report wiring** — add `_summarize_<check>_results`/
   `_generate_<check>_html` to `report.py` (reusing the simple-check helpers
   where the shape allows) and register the check name in
   `_generate_preflight_summary()`'s `checks` dict and
   `_generate_html_content()`'s section list.
7. **README** — document the new check in
   `cvs/tests/preflight/README.md` (overview list, config keys, example
   block, troubleshooting section).
8. **Workflow** — `make fmt` → `make lint` → `make test` before committing
   (per root `CLAUDE.md`/`CONTRIBUTORS.md`).

## Existing check modules (one line each)

- `base.py` — `PreflightCheck` ABC and shared static helpers (see above).
- `node_reachability.py` — `PingReachabilityCheck` (local `ping` subprocess,
  driver→node), `UptimeCheck` (informational `uptime` via `phdl.exec`).
- `ssh_mesh_connectivity.py` — `SshMeshConnectivityCheck`: all-pairs
  passwordless SSH reachability across reachable nodes; diagnostic, never
  prunes.
- `etc_hosts_consistency.py` — `EtcHostsConsistencyCheck`: validates
  `/etc/hosts` entries against the cluster inventory plus config-supplied
  `extra_entries`.
- `limits_conf_check.py` — `LimitsConfCheck`: validates required lines are
  present in `/etc/security/limits.conf`; blocking FAIL when enabled.
- `nic_firmware_check.py` — `NicFirmwareCheck`: dispatcher over per-vendor
  `AinicFirmwareCheck`/`BroadcomFirmwareCheck`/`MellanoxFirmwareCheck`
  (selected via config `nic_type`), each validating device count (FAIL on
  mismatch) and firmware/host-software version (WARNING on mismatch). AINIC
  uses `ibv_devices` (count) + `nicctl show version {firmware,host-software}`;
  Broadcom uses `niccli --list` (count + `FwVersion` column, in one command);
  Mellanox uses `ibv_devices` (count) + `ethtool -i` (firmware version).
  SKIPPED (not FAIL) on nodes without that vendor's hardware. Merges
  per-vendor per-node results FAIL > WARNING > all-SKIPPED > PASS.
- `nic_driver_version.py` — `NicDriverVersionCheck`: dispatcher over
  per-vendor `BroadcomDriverVersionCheck`/`AinicDriverVersionCheck`/
  `MellanoxDriverVersionCheck` (selected via config `nic_type`). AINIC
  validates per-NIC firmware version via `nicctl show version firmware`
  (`Uboot-A`/`Firmware-A` fields, installed alongside the AINIC driver);
  Broadcom validates NIC package version via `niccli` (installed alongside
  the Broadcom driver; does not report kernel module/`modinfo` version at
  all); Mellanox validates driver module version via `modinfo`. SKIPPED
  (not FAIL) on nodes without that vendor's hardware. Merges per-vendor
  per-node results FAIL > WARNING > all-SKIPPED > PASS.
- `scaleup_fabric.py` — `NodeHealthCheck`: AMDGPU/KFD + kernel-health
  validation with optional MI4XX AFM/vPOD scale-up fabric admission; state
  validation only, never mutates driver/fabric state.
- `gid_consistency.py` — `GidConsistencyCheck`: validates a configured GID
  index has a non-zero value on every RDMA interface.
- `interface_consistency.py` — `InterfaceConsistencyCheck`: validates
  expected RDMA interface names are present, ACTIVE, and LINK_UP (via
  `cvs/lib/linux_utils.py::get_rdma_nic_dict`).
- `version_check.py` — `RocmVersionCheck`: validates consistent ROCm version
  across all nodes.
- `ifoe_l2_connectivity.py` — `IfoeL2ConnectivityCheck` (+
  `AfmctlPortParser`/`parse_afmctl_show_device_json` helpers reused by
  `scaleup_fabric.py`): L2 reachability via `afmctl test ping`, zero-loss
  policy, full pair coverage required.
- `ainic_pfc_qos_dcqcn.py` — `PfcValidationCheck`, `QosValidationCheck`,
  `DcqcnValidationCheck`: AINIC control-plane config (PFC pause type, QoS
  DSCP/priority maps, DCQCN tuning) via `nicctl show {port,qos,dcqcn}`
  against built-in generic AINIC golden-value defaults, fully overridable.
- `transferbench_smoke.py` — `TransferBenchSmokeCheck`: IFoE scale-up
  data-path smoketest via the TransferBench `smoketest` preset, `node` or
  `cluster` scope.
- `rdma_connectivity.py` — `RdmaConnectivityCheck`: node-to-node RDMA
  connectivity via `ibv_rc_pingpong`, basic/full-mesh/skip modes.
- `report.py` — `PreflightReportGenerator` (subclasses `PreflightCheck`):
  builds the summary dict and HTML report; the `_summarize_*`/`_generate_*_html`
  method pairs described above.
