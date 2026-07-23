# ANC (AMD Node Check) CVS Tests

This directory packages the ANC validation suite as CVS tests.

> **ANC requires root permissions to execute.** Every ANC group is invoked as
> `sudo ./anc.py`, and artifact collection archives ANC's root-owned log
> directory with `sudo tar`. The runner must therefore have **passwordless
> `sudo`** on each target node (see [Prerequisites](#1-prerequisites)). Without
> root, ANC will not run and log collection will fail.

**Installation:**

| File | `cvs run` command | Purpose |
| --- | --- | --- |
| `anc_installation.py` | `cvs run anc_installation` | Download and install the ANC tool on every node. |

**Group suites** — there are exactly two group suites, `anc_test_cpu` and
`anc_test_gpu` (in the `cpu/` and `gpu/` subfolders). Each holds one
`test_<group>` **function** per ANC group. Running the whole suite runs every
group in that set; naming a function runs just that group. Each function ensures
ANC is installed and ROCm ldconfig is fixed first (session-cached via
`anc_lib.ensure_anc_ready`, so a full-suite run pays the setup cost once). The
two files are **generated** from the single source `anc_lib.CPU_GROUPS` /
`GPU_GROUPS`.

> Generated files — do NOT hand-edit. To add/remove a group, edit the lists in
> `cvs/lib/anc_lib.py` and run `make gen-anc-suites` (wraps
> `build_tools/gen_anc_suites.py`), which rewrites the two suite files and
> prunes stale ones. Then reinstall (`make install` / `pip install .`).

```bash
# run every CPU group (each group its own test + log dir)
cvs run anc_test_cpu                  --cluster_file <c.json> --config_file <cfg.json>
# run a single group by its function name
cvs run anc_test_cpu test_cpu_all     --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_gpu test_hbm_lvl1    --cluster_file <c.json> --config_file <cfg.json>
# list the per-group functions in a suite
cvs list anc_test_cpu
```

- CPU (`anc_test_cpu`): `test_ampttk_full`, `test_cachewalker_full`,
  `test_cpu_all`, `test_cpu_content_check`, `test_cpu_mfg_l10`,
  `test_cpu_sanity`, `test_difect_full`, `test_fpdeluge_full`,
  `test_hdrt_full`, `test_maxcorestim_full`, `test_memtest_full`,
  `test_miidct_full`, `test_mithac_full`, `test_weighted_sanity`
- GPU (`anc_test_gpu`): `test_gpu_content_check`, `test_gpu_mfg_l10`,
  `test_hbm_lvl1` … `test_hbm_lvl5`

Shared logic — package install by archive flavour (deb/rpm/tar), version check,
the session-cached setup guard `ensure_anc_ready`, ldconfig fix, group
execution, and artifact collection — lives in `cvs/lib/anc_lib.py` (group sets
are `CPU_GROUPS` / `GPU_GROUPS`). Shared pytest fixtures live in this directory's
`conftest.py` and apply to the `cpu/` and `gpu/` subfolders too. ANC is invoked
from its installed location `/opt/amdtools/anc/anc.py`.

**Logs & console:** each group's ANC log directory is copied to
`log_folder_path` (default
`{runner_log_folder}/anc_logs/<node>/<test_name>/<timestamp>`, where `<node>` is
that node's `<ip>_<hostname>` label — so multi-node runs group every
test/timestamp under each node's own folder); the resolved pattern is printed
before the run. Set `print_all_to_console` to `False` in config to suppress the
ANC group output on the console (install/ldconfig diagnostics still print). Pass
or fail is read from each run's `console.log` final `ANC_SUCCESS [0]`.

---

## 1. Prerequisites

- The `cvs-internal` package is built and installed, and its virtual environment
  is active. From the repository root:

  ```bash
  make install
  source .cvs_venv/bin/activate
  cvs --version          # sanity check
  cvs list               # anc_installation, anc_test_cpu, anc_test_gpu
  ```

  See the repository root `README.md` for full build/setup details.

- Passwordless SSH from the runner to every node (key-based), and passwordless
  `sudo` (root) on each node. **ANC requires root to execute** — it runs as
  `sudo ./anc.py`, and artifact collection archives the root-owned log directory
  with `sudo tar` and then chowns the tarball back to the SSH user. After the
  logs are pulled to the controller they are chowned to the user running the
  test.

---

## 2. Configuration files

Both commands take two mandatory arguments:

- `--cluster_file` — cluster/node definitions (SSH user, private key, nodes).
- `--config_file` — ANC test configuration.

### Cluster file

A starting point lives at `cvs/input/cluster_file/cluster.json`.
Provide your SSH user, private key, and the nodes to target:

```json
{
    "username": "<ssh-user>",
    "priv_key_file": "/home/<ssh-user>/.ssh/id_rsa",
    "node_dict": {
        "<node-hostname-or-ip>": {
            "bmc_ip": "NA",
            "vpc_ip": "<node-ip>",
            "gpu_type": "MI325X"
        }
    }
}
```

- The `node_dict` keys are the SSH targets ANC commands run against.
- `vpc_ip` is used (together with the live hostname) to name the per-node
  artifact folders: `<vpc_ip>_<hostname>`.

### Config file

The ANC config lives at `cvs/input/config_file/anc/anc_config.json`:

```json
{
    "_comment": "ANC test configuration",
    "anc": {
        "description": "AMD Node Check",
        "test_timeout": 7200,
        "install_timeout": 1800,
        "anc_version": "1.4.7",
        "anc_release_url": "https://.../anc-release-helios-nda-1.4.7-rpm-linux-x64.tar.gz",
        "cvs_home": "{home}/cvs",
        "print_all_to_console": "True",
        "log_folder_path": "{home}/cvs_logs/anc_logs/<node>/<test_name>/<timestamp>",
        "ADD_ANC_LOGS_TO_HTML_REPORTS": "False",
        "COLLECT_HTML_REPORTS": "True",
        "html_report_path": "{home}/cvs_logs/html_reports/<node>/<test_name>/<timestamp>"
    }
}
```

Each key is documented inline in the shipped config via a matching
`_comment_<key>` sibling (keys prefixed with `_comment` are ignored at runtime).

| Key | Meaning |
| --- | --- |
| `test_timeout` | Per-group execution timeout in seconds (default 7200 = 2 h). Applies to ANC group runs only, not install. |
| `install_timeout` | Package download+install **inactivity** timeout in seconds (default 1800 = 30 min), used only by `anc_installation` / the install pre-task. It is a per-read (no-output) timeout, not a total budget: the download emits a periodic progress heartbeat so a slow link never trips it, while a genuine stall still fails. Independent of `test_timeout`. |
| `anc_version` | Expected ANC version; install pre-task skips (re)install when already present and post-verifies the match. |
| `anc_release_url` | ANC release archive URL (used by `anc_installation`). Flavour (deb/rpm/tar) is auto-detected from the filename. |
| `cvs_home` | Staging dir on each node for the release download/unpack (tar flavour). `{home}` resolves to the SSH user's home. ANC itself always installs to `/opt/amdtools/anc`. |
| `print_all_to_console` | `True` echoes ANC group output to console; `False` suppresses it (diagnostics still print). |
| `log_folder_path` | Controller-side destination for collected logs. Tokens: `{home}` → `/home/<userid>`, `<node>` → the node's `<ip>_<hostname>` label, `<test_name>` → the group's test name, `<timestamp>` → per-run stamp (appended if omitted). |
| `ADD_ANC_LOGS_TO_HTML_REPORTS` | `True` always bundles the collected ANC log tree (as a `.tar.gz` with a clickable link) into the pytest-html report zip. `False` (default) bundles it **only when the test fails**. |
| `COLLECT_HTML_REPORTS` | `True` (default) auto-generates a pytest-html report even when no `--html` is passed, written to `html_report_path`. `False` disables auto-collection. An explicit `--html` on the command line always overrides `html_report_path`. |
| `html_report_path` | Destination **directory** template for the auto-collected report (`<test_name>.html` is placed inside). Same tokens as `log_folder_path`. Since pytest-html makes one report per session before any node connects, `<node>` here is the **first** cluster node (label from the cluster file only, no SSH). |

`{home}` is resolved on the controller from the cluster file's `username`, so it
works even when the runner and nodes have different home paths. With the default
`log_folder_path`, logs land at
`/home/<userid>/cvs_logs/anc_logs/<node>/<test_name>/<timestamp>`, where `<node>`
is that node's `<ip>_<hostname>` label.

---

## 3. Install ANC on the target nodes

Every ANC validation suite installs ANC as a pre-task, so a separate install
step is **optional**. Run `anc_installation` on its own when you want to
install/refresh ANC without running a validation group. ANC always installs to
`/opt/amdtools/anc` (for all three flavours — deb, rpm, and tar), with the
entrypoint at `/opt/amdtools/anc/anc.py`. You can install it in either of the
ways below.

### Option A - from the head node (recommended)

The **head node** is the controller from which CVS drives the target nodes. Run
the installer there; it installs ANC on every target node listed in the cluster
file:

```bash
cvs run anc_installation \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

This downloads the `anc_release_url` archive, installs ANC to `/opt/amdtools/anc`
on every target node (the flavour — deb/rpm/tar — is auto-detected from the
filename), and validates the install.

### Option B - manually on a target node

If you do not want to run `anc_installation`, install ANC directly on the target
node. The example below is for the **tar** flavour: download and extract the
release archive in a staging dir, then extract the two `anc-tool` and
`anc-content` archives into `/opt/amdtools` so the layout matches the deb/rpm
packages:

```bash
cd "$HOME/cvs"   # staging dir (cvs_home); only used for the download/unpack

# Download and extract the ANC release
wget -q "https://atlartifactory.amd.com:8443/artifactory/HW-ANCRelease-REL-LOCAL/anc-release/helios_nda/1.4.7/anc-release-helios-nda-1.4.7-tar-linux-x64.tar.gz" \
  -O anc-release.tar.gz
tar -xzf anc-release.tar.gz && rm -f anc-release.tar.gz

# Extract the tool and content archives into /opt/amdtools (needs sudo)
sudo rm -rf /opt/amdtools/anc
sudo mkdir -p /opt/amdtools
sudo tar -xzf anc-tool*.tar.gz    -C /opt/amdtools && rm -f anc-tool*.tar.gz
sudo tar -xzf anc-content*.tar.gz -C /opt/amdtools && rm -f anc-content*.tar.gz
```

For the **deb**/**rpm** flavours, install the extracted `anc*.deb` / `anc*.rpm`
packages instead (`sudo dpkg -i` / `sudo dnf install`); they lay ANC down under
`/opt/amdtools/anc` directly. This is the sequence performed by
[`anc_installation.py`](anc_installation.py) (which delegates to
`cvs/lib/anc_lib.py`).

---

## 4. Run the ANC validation tests

Each group function first ensures ANC is installed and ROCm ldconfig is fixed
(session-cached, so it happens once per run), then runs its group on **all**
nodes in parallel with a single `anc.py -g <group>` invocation.

**Single group** — name the group's `test_<group>` function on its suite:

```bash
cvs run anc_test_cpu test_cpu_all \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_gpu test_hbm_lvl1 \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

**All groups in a set** — run the whole suite. ANC install + ldconfig happen
once (session-cached), then every group in the CPU (or GPU) set runs as its own
test with its own log dir:

```bash
cvs run anc_test_cpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_gpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

The exact group lists are defined by `CPU_GROUPS` / `GPU_GROUPS` in
`cvs/lib/anc_lib.py` (see the full list in the "Group suites" section above). To
run a subset, name several functions:

```bash
cvs run anc_test_cpu test_cpu_sanity test_memtest_full \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

---

## 5. Pass / fail criteria

For each test, a node **passes** only when:

- ANC's run produced a `Log directory: <path>` line (the run actually started),
- `console.log` was collected from that directory (the entire log directory is
  pulled back; `console.log` is the only file that must be present), and
- the **final** `return code <NAME> [<int>]` line in `console.log` is
  `ANC_SUCCESS [0]`.

A node **fails** when any of the following occur:

- the run could not be executed (SSH/exec/permission error, no output, or no
  `Log directory` in the output),
- `console.log` is missing or could not be copied, or
- the final ANC return code is non-zero (anything other than `ANC_SUCCESS [0]`).

Failures across multiple parallel nodes are aggregated into a **single** test
failure (one failure per test, not one per node).

---

## 6. Artifacts

Per node and per test, artifacts are downloaded to the resolved
`log_folder_path` (default layout):

```
<runner_log_folder>/anc_logs/<ip>_<hostname>/<test_name>/<timestamp>/
```

- `<runner_log_folder>` comes from `run_config["runner_log_folder"]` (or `{home}`
  in the shipped config) — see the `log_folder_path` token table above.
- `<ip>_<hostname>` is the per-node label; `<test_name>` is the group's test name
  (e.g. `test_cpu_all`); `<timestamp>` keeps repeated runs separate.

Collected files:

- The **entire** ANC log directory is pulled back (whatever ANC wrote:
  `console.log`, `journal.log`, `summary.json`, per-item logs, …).
- **Required:** `console.log` (holds the verdict). If it is missing, the node
  fails. Every other file is collected best-effort as part of the whole-directory
  copy.

### In the HTML report

A pytest-html report is produced whenever `--html` is passed **or**
`COLLECT_HTML_REPORTS` is `True` (the default) — in the latter case the report is
written automatically to `html_report_path` with no `--html` needed. An explicit
`--html` always overrides `html_report_path`. Each group's collected ANC log tree
can also be bundled into the report zip as a single `.tar.gz` with a clickable
**"ANC logs: <test>"** link, controlled by `ADD_ANC_LOGS_TO_HTML_REPORTS`:

- `True` — always attach (pass or fail).
- `False` (default) — attach **only when the test fails** (the link is labelled
  `... (FAILED)`), so passing runs keep the report small while failures always
  ship their full logs.

The full tree is always written to `log_folder_path` regardless of this flag;
the flag only governs what gets embedded in the HTML report bundle.
