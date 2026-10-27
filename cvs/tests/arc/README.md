# AMD Rack Check (ARC) suites

ARC validates rack topology and runs flows and items from **one controller per rack**. CVS builds each rack's ARC rack file from the cluster file, installs ARC, and collects its logs through the controllers.

## Quick start

```bash
cvs config copy cluster_rack.json --output ~/cluster_rack.json
# Set node_dict, rack_id per node (in tray order), and racks.<rack_id>.platform.
cvs config copy arc/arc_config.json --output ~/arc_config.json
# Replace arc_release_url; leave rack_config empty to generate rack files.
cvs list arc_test_diag
cvs run arc_installation --cluster_file ~/cluster_rack.json --config_file ~/arc_config.json
cvs run arc_test_diag test_PingTest --cluster_file ~/cluster_rack.json --config_file ~/arc_config.json
```

## Prerequisites and controller model

ARC runs on baremetal only. The runner needs passwordless SSH to every controller. ARC itself needs passwordless sudo for tools that require escalation. Controllers need Python 3.9 or newer.

Each rack in the cluster file's `racks` block gets one controller: `arc_controller`, or the first `node_dict` entry with that `rack_id`. CVS fills the ARC template for the rack's `platform` with that rack's compute nodes, plus any switch, BMC, and RMC addresses you list, and validates it on the controller. List each rack's compute nodes in tray order (Helios: U9 first, U33 last), because ARC derives scale-up MACs and switch ports from the tray number. Without switch or BMC access, leave those lists out and set `arc.spec_mode` to `partial`. Setting `arc.rack_config` skips generation and uses a rack JSON you staged at that path on every controller. `arc.controller_nodes` is deprecated and applies only when the cluster file has no `racks` block.

Non-controller nodes are never contacted by these suites.

## Installation and configuration

`arc.arc_release_url` points directly to a `.deb`, `.rpm`, or `.tar.gz` artifact. `arc.arc_version` sets the minimum installed version. Tar packages honor `ARC_INSTALL_PATH`; deb and rpm packages install under `/opt/amdtools/arc`. Downloads are staged in a private directory and cleaned up. ARC runs as the SSH user and handles its own tool-specific sudo calls.

`arc.mock` uses the `MOCK=1` environment variable. The release build rejects the development-only `--mock` argument. Release packages may not include mock data for real diagnostics; the `Mock_*` example content is the integration self-test. `arc.spec_mode` is `full` or `partial`. `arc.continue_on_failure` passes ARC's continuation flag. `arc.print_all_to_console` controls live output. See [the sample configuration](../../input/config_file/arc/arc_config.json) for every field.

Both console modes print a liveness heartbeat every 60 s while ARC runs. `arc.inactivity_timeout` is the SSH channel safety net (default 900 s, minimum 120 s). `arc.run_timeout` limits one run to 14400 s by default; `0` disables it. When the limit passes or CVS stops listening, the controller shell sends TERM to ARC's process group, waits up to 30 s, then sends KILL. Processes ARC starts through `sudo`, in another session, or on trays over its own SSH may survive and need manual cleanup.

## Suites

| Suite | Content |
| --- | --- |
| `arc_installation` | Install, version check, and rack JSON validation |
| `arc_test_examples` | `example_pass`, `example_failure` integration self-tests |
| `arc_test_flows` | Diagnostic flows, including `diags_lvl1`, `diags_lvl2`, `wld2_verify` |
| `arc_test_diag` | Rack diagnostics, including `PingTest` |
| `arc_test_telemetry` | AMC/BMC and firmware telemetry |
| `arc_test_anc` | ANC CPU and GPU tests driven through ARC |
| `arc_test_workload` | Front-end and scale-up workload checks |
| `arc_test_disruptive_flows` | State-changing flows; opt-in |
| `arc_test_disruptive` | State-changing items; opt-in |

`diags_lvl1` and `diags_lvl2` clear AMC log stores as PRE/POST conditions. Names missing from a platform's installed ARC content report n/a and fail the test. Classify new content before adding it to a suite.

### ⚠ Disruptive content

These suites skip unless `arc.allow_disruptive` is `"True"`; mock mode does not bypass this gate. The gate runs before installation or ARC execution.

| Names | Reason |
| --- | --- |
| `heliosm_mfg_l11`, `heliosp_mfg_l11`, `heliosr_mfg_l11`, `reboot_all` | Reboot or power-cycle rack hardware |
| `scale_up_config`, `scale_up_verify` | Reconfigure EAM, AFM, or switches |
| `ComputeDCCycle`, `RebootComputeTest`, `RebootRmcTest`, `RebootSwitchTest` | Power-cycle or reboot components |
| `PortToggle`, `ScaleOutConnectivityTestID`, `ScaleOutConnectivityTestL2`, `ScaleOutConnectivityTestL3` | Change ports, namespaces, or routes |
| `ScaleUpConnectivityTest`, `ScaleupEAMConfig`, `ScaleupSwitchConfig` | Reconfigure scale-up fabric |
| `PRBSCheck`, `PRBSCheckEAMFirst`, `SBRTest` | Run PRBS or SBR diagnostics |
| `SwitchInternalLoopback` | Reconfigure switch loopback |

## Verdicts and artifacts

A controller passes only when ARC exits 0 and its collected `ARC/summary.json` reports `SUCCESS` or `WARNING`. Any mismatch or missing summary fails. `example_failure` deliberately fails, exercising the failure report and Run Deck. A missing flow or item reports n/a in the deck and fails the test.

CVS pulls the whole ARC session under `<run_dir>/arc_logs/<node>/<test_name>/<timestamp>/<session>/ARC/`, including `summary.json`, `errors.json`, `journal.log`, and `output.log` when ARC writes them. The HTML report links the summary and errors files. Log tarballs are attached on failure, or on passing runs when `ADD_ARC_LOGS_TO_HTML_REPORTS` is true. The `arc_test_*` suites also emit a Run Deck with the ARC version. Controller-side `$HOME/arc_logs` is retained for review and is not cleaned automatically. The rack JSON itself is not collected.
