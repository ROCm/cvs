# cvs/tests/preflight/ — preflight test suite conventions

For the full check-by-check reference (what each check validates, every
config key, troubleshooting commands, example config blocks) see
`README.md` in this same directory — do not duplicate that detail here.
For the bash/ansible source material this suite ports from, its inventory
and porting status, see `docs/plans/preflight-checks-porting.md`. For the
lib-side conventions (base class, result shapes, report wiring checklist),
see `cvs/lib/preflight/CLAUDE.md`.

This directory currently has a single test module,
`preflight_checks.py` (~2000 lines), run as `cvs run preflight_checks
--cluster_file <f> --config_file <f>`.

## The `preflight_results` accumulation pattern

`preflight_results = {}` is a **module-global dict**, not a fixture. Every
`test_*` function is sequential (pytest runs them in file order within a
module) and writes its outcome into `preflight_results[<check_name>]` before
returning — there is no `return`/assertion-based signaling between checks;
downstream checks and the final report both read prior results back out of
this same global. This is why:

- Adding a new check means adding both a `test_*` function *and* wiring the
  key name into `test_generate_preflight_report`'s `required_checks` list
  (`preflight_checks.py`) — otherwise the report logs "missing results" and
  synthesizes a `SKIPPED` placeholder for it.
- Downstream checks that need to know whether an earlier gate passed read
  `preflight_results.get('<earlier_check>')` directly (e.g.
  `_node_health_admitted_port_ids()` reads `preflight_results['node_health']`
  to build the IFoE port-admission map) rather than receiving it as a
  fixture/parameter.
- Tests are not independent/parallelizable — do not add `pytest-xdist`-style
  parallelism to this module or reorder `test_*` functions without
  understanding what state each one depends on.

## Ordering / tiering convention

Functions run in this order today (mirrors `run_iren_precheck`'s "eliminate
unreachable nodes, then validate static config, then functional/data-plane,
then RDMA" structure — see the porting plan for the historical rationale):

1. **Elimination tier** — `test_node_ping_reachability` (ICMP, diagnostic
   only, never gates/prunes) → `test_node_reachability` (SSH echo; **prunes**
   failed nodes) → `test_ssh_mesh_connectivity` (opt-in full mesh,
   diagnostic, never prunes) → `test_node_uptime` (informational only).
2. **Static config tier** — `test_etc_hosts_consistency` →
   `test_limits_conf` (blocking FAIL when enabled) → `test_nic_firmware` →
   `test_nic_driver_version` (SKIPPED, not FAILed, on non-Broadcom nodes) →
   `test_node_health` (mandatory GPU/fabric admission gate) →
   `test_rocm_version_consistency`.
3. **IFoE/functional tier** — `test_ifoe_l2_connectivity` →
   `test_ainic_pfc_qos_dcqcn` → `test_ifoe_transferbench_smoke`. All three
   check `_node_health_admission_failed(config_dict)` first and, if the
   node-health gate failed, short-circuit their result to `BLOCKED`
   (`_blocked_by_node_health(check_name)`) instead of running — they depend
   on admitted GPU/fabric state but do not themselves prune nodes.
4. **RDMA tier** — `test_interface_name_consistency` (**prunes**) →
   `test_gid_consistency` (**prunes**) → `test_rdma_connectivity` (also
   `BLOCKED`-gated by node-health admission, like tier 3).
5. **Reporting** — `test_generate_preflight_report`: validates every key in
   `required_checks` is present in `preflight_results`, builds
   `PreflightReportGenerator(phdl, preflight_results,
   config_dict).run()`, logs the summary, writes/links the HTML report, and
   *only then* asserts — a mandatory `node_health` admission failure fails
   the test here (deliberately deferred past report-writing so the failure
   shows up in report artifacts rather than aborting the module early); a
   generic overall `FAIL` status otherwise only warns, it does not fail the
   pytest run (preflight is designed as a reporting tool, not a hard gate,
   except for the node-health admission case).

## Node pruning: when a check should/shouldn't prune

`_prune_nodes_from_phdl(phdl, failed_nodes, reason)` removes hosts from
`phdl.reachable_hosts` and recreates the parallel client, so every later
`phdl.exec()` call in the module only targets nodes that passed. In the
current code, pruning happens at exactly three points:
`test_node_reachability` (SSH), `test_interface_name_consistency`, and
`test_gid_consistency`. The rule for a new check:

- **Prune** only if a later check would produce a false/misleading
  result (or crash) by running against a node that already failed this
  check — i.e. this check establishes a precondition later checks assume
  (SSH reachability, interface presence, GID validity).
- **Do not prune** for diagnostic/informational checks (ping, uptime, SSH
  mesh) or for checks whose failure should be visible/blocking rather than
  silently excluding the node (node-health admission uses the `BLOCKED`
  short-circuit pattern above instead of pruning, precisely so a failed
  node still shows up — as `BLOCKED`, not silently dropped — in every
  downstream check's results and the final report).

## Fixtures

`cluster_file`/`config_file` (read `pytestconfig` options) →
`cluster_dict`/`config_dict` (parse + validate/normalize JSON, including the
legacy-RDMA-path migration via
`cvs.parsers.schemas.normalize_legacy_preflight_rdma_config`) → `phdl`
(module-scoped `MultiProcessPssh` built from `cluster_dict`+`config_dict`) →
`shdl` (a single-host `Pssh` handle scoped to the head node only, for
head-node-specific operations rather than cluster-wide broadcast).
All fixtures are `scope="module"`, consistent with the single shared
`preflight_results` global.

## Where to look next

- Full check list, config schema, and troubleshooting: `README.md` (this
  directory).
- What's ported from bash/ansible vs. still open, and why each design
  decision was made: `docs/plans/preflight-checks-porting.md`.
- Lib-side conventions and the "adding a new check" checklist:
  `cvs/lib/preflight/CLAUDE.md`.
