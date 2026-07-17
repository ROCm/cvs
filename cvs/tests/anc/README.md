# ANC (Automated Node Checkout) CVS Tests

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

**Per-group suites** — one standalone suite per ANC group, so every group shows
in `cvs list` and runs by name. They live in the `cpu/` and `gpu/` subfolders
and are **generated** from the single source `anc_lib.CPU_GROUPS` /
`GPU_GROUPS`. Each is a thin `AncGroupTest` subclass that installs + verifies
ANC and fixes ROCm ldconfig as pre-tasks, then runs just its group.

> Generated files — do NOT hand-edit. To add/remove a group, edit the lists in
> `cvs/lib/anc_lib.py` and run `make gen-anc-suites` (wraps
> `build_tools/gen_anc_suites.py`), which rewrites the per-group files and
> prunes stale ones. Then reinstall (`make install` / `pip install .`).

```bash
cvs run anc_test_cpu_all   --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_hbm_lvl1  --cluster_file <c.json> --config_file <cfg.json>
```

- CPU (`cpu/`): `anc_test_ampttk_full`, `anc_test_cachewalker_full`,
  `anc_test_cpu_all`, `anc_test_cpu_content_check`, `anc_test_cpu_mfg_l10`,
  `anc_test_cpu_sanity`, `anc_test_difect_full`, `anc_test_fpdeluge_full`,
  `anc_test_hdrt_full`, `anc_test_maxcorestim_full`, `anc_test_memtest_full`,
  `anc_test_miidct_full`, `anc_test_mithac_full`, `anc_test_weighted_sanity`
- GPU (`gpu/`): `anc_test_gpu_content_check`, `anc_test_gpu_mfg_l10`,
  `anc_test_hbm_lvl1` … `anc_test_hbm_lvl5`

**Run-all suites** — install ANC + ldconfig ONCE, then run every group in the
set sequentially (each group is its own parametrized test with its own log dir):

```bash
cvs run anc_test_exec_all_cpu_test --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_exec_all_gpu_test --cluster_file <c.json> --config_file <cfg.json>
```

Shared logic — package install by archive flavour (deb/rpm/tar), version check,
ldconfig fix, group execution, artifact collection, and the per-group base class
`AncGroupTest` — lives in `cvs/lib/anc_lib.py` (group sets are
`CPU_GROUPS` / `GPU_GROUPS`). Shared pytest fixtures live in this directory's
`conftest.py` and apply to the `cpu/` and `gpu/` subfolders too. ANC is invoked
from its installed location `/opt/amdtools/anc/anc.py`.

**Logs & console:** each group's ANC log directory is copied to
`log_folder_path` (default `{runner_log_folder}/anc_logs/<test_name>/<timestamp>`,
under a per-node `<ip>_<hostname>/` subdir); the resolved path is printed before
the run. Set `print_all_to_console` to `False` in config to suppress the ANC
group output on the console (install/ldconfig diagnostics still print). Pass or
fail is read from each run's `console.log` final `ANC_SUCCESS [0]`.

---

## 1. Prerequisites

- The `cvs-internal` package is built and installed, and its virtual environment
  is active. From the repository root:

  ```bash
  make install
  source fremont_venv/bin/activate
  cvs --version          # sanity check
  cvs list               # anc_installation, anc_test_cpu, anc_test_gpu should appear
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
        "description": "Automated Network Connectivity checks",
        "test_timeout": 7200,
        "anc_version": "1.4.7",
        "anc_release_url": "https://.../anc-release-helios-nda-1.4.7-rpm-linux-x64.tar.gz",
        "cvs_home": "{home}/cvs",
        "anc_root_dir": "anc",
        "print_all_to_console": "True",
        "log_folder_path": "{home}/cvs_logs/anc_logs/<test_name>/<timestamp>",
        "ADD_ANC_LOGS_TO_HTML_REPORTS": "False"
    }
}
```

Each key is documented inline in the shipped config via a matching
`_comment_<key>` sibling (keys prefixed with `_comment` are ignored at runtime).

| Key | Meaning |
| --- | --- |
| `test_timeout` | Per-group execution timeout in seconds (default 7200 = 2 h). |
| `anc_version` | Expected ANC version; install pre-task skips (re)install when already present and post-verifies the match. |
| `anc_release_url` | ANC release archive URL (used by `anc_installation`). Flavour (deb/rpm/tar) is auto-detected from the filename. |
| `cvs_home` | Install root on each node. `{home}` resolves to the SSH user's home. |
| `anc_root_dir` | ANC directory name under `cvs_home`. ANC is run from `<cvs_home>/<anc_root_dir>`. |
| `print_all_to_console` | `True` echoes ANC group output to console; `False` suppresses it (diagnostics still print). |
| `log_folder_path` | Controller-side destination for collected logs. Tokens: `{home}` → `/home/<userid>`, `<test_name>` → the group's test name, `<timestamp>` → per-run stamp (appended if omitted). |
| `ADD_ANC_LOGS_TO_HTML_REPORTS` | `True` always bundles the collected ANC log tree (as a `.tar.gz` with a clickable link) into the pytest-html report zip. `False` (default) bundles it **only when the test fails**. |

`{home}` is resolved on the controller from the cluster file's `username`, so it
works even when the runner and nodes have different home paths. With the default
`log_folder_path`, logs land at
`/home/<userid>/cvs_logs/anc_logs/<test_name>/<timestamp>` (under a per-node
`<ip>_<hostname>/` subdir).

---

## 3. Install ANC on the target nodes

The `anc_test_cpu` / `anc_test_gpu` suites install ANC as a pre-task, so a
separate install step is **optional**. Run `anc_installation` on its own when
you want to install/refresh ANC without running a validation group. ANC is
installed to `/opt/amdtools/anc` (deb/rpm), with the entrypoint at
`/opt/amdtools/anc/anc.py`. You can install it in either of the ways below.

### Option A - from the head node (recommended)

The **head node** is the controller from which CVS drives the target nodes. Run
the installer there; it installs ANC on every target node listed in the cluster
file:

```bash
cvs run anc_installation \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

This downloads the `anc_release_url` tarball, extracts ANC into
`<cvs_home>/<anc_root_dir>` on every target node, and validates the install.

### Option B - manually on a target node

If you do not want to run `anc_installation`, install ANC directly on the target
node. Download and extract the release tarball, then install the two `anc-tool`
and `anc-content` archives obtained from that extraction:

```bash
cd "$HOME/cvs"   # <cvs_home>

# Download and extract the ANC release
wget -q "https://atlartifactory.amd.com:8443/artifactory/HW-ANCRelease-REL-LOCAL/anc-release/helios_nda/1.3.2/anc-release-helios-nda-1.3.2-tar-linux-x64.tar.gz" \
  -O anc-release.tar.gz
tar -xzf anc-release.tar.gz && rm -f anc-release.tar.gz

# Install the tool and content archives produced by the extraction above
tar -xzf anc-tool*.tar.gz    && rm -f anc-tool*.tar.gz
tar -xzf anc-content*.tar.gz && rm -f anc-content*.tar.gz
rm -rf anc/content/base      # <anc_root_dir>/content/base
```

This is exactly the sequence performed by `anc_installation` - see
[`anc_installation.py`, lines 273-305](https://github.com/AMD-ROCm-Internal/cvs-internal/blob/ashmishr/anc-cvs-interlock/cvs/tests/anc/anc_installation.py#L273-L305).

---

## 4. Run the ANC validation tests

Each suite first installs/verifies ANC (pre-task), then runs its group set on
**all** nodes in parallel with a single `anc.py -g <groups...>` invocation.

CPU validation:

```bash
cvs run anc_test_cpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

GPU validation:

```bash
cvs run anc_test_gpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

- `anc_test_cpu` runs `sudo ./anc.py -g <CPU_GROUPS>`
- `anc_test_gpu` runs `sudo ./anc.py -g <GPU_GROUPS>`

The exact group lists are defined by `CPU_GROUPS` / `GPU_GROUPS` in
`cvs/lib/anc_lib.py`. To skip the install pre-task and run only the
group test, add pytest's `-k` filter (forwarded by `cvs run`):

```bash
cvs run anc_test_cpu -k test_cpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

---

## 5. Pass / fail criteria

For each test, a node **passes** only when:

- ANC's run produced a `Log Directory: <path>` line (the run actually started),
- `journal.log` and `console.log` were collected from that directory, and
- the **final** `return code <NAME> [<int>]` line in `console.log` is
  `ANC_SUCCESS [0]`.

A node **fails** when any of the following occur:

- the run could not be executed (SSH/exec/permission error, no output, or no
  `Log Directory` in the output),
- a mandatory log (`journal.log` / `console.log`) is missing or could not be
  copied, or
- the final ANC return code is non-zero (anything other than `ANC_SUCCESS [0]`).

Failures across multiple parallel nodes are aggregated into a **single** test
failure (one failure per test, not one per node).

---

## 6. Artifacts

Per node and per test, artifacts are downloaded to:

```
<runner_log_folder>/anc/<ip>_<hostname>/<test_name>/
```

- `<runner_log_folder>` comes from `run_config["runner_log_folder"]` and defaults
  to `/tmp/cvs_results`.
- `<test_name>` is `test_cpu` or `test_gpu`.

Collected files:

- **Mandatory:** `journal.log`, `console.log`.
- **Optional (collected when present):** `summary.json`, `errors.json`,
  `system_monitor.json`.

### In the HTML report

When you pass `--html`, each group's collected ANC log tree can also be bundled
into the report zip as a single `.tar.gz` with a clickable **"ANC logs: <test>"**
link, controlled by `ADD_ANC_LOGS_TO_HTML_REPORTS`:

- `True` — always attach (pass or fail).
- `False` (default) — attach **only when the test fails** (the link is labelled
  `... (FAILED)`), so passing runs keep the report small while failures always
  ship their full logs.

The full tree is always written to `log_folder_path` regardless of this flag;
the flag only governs what gets embedded in the HTML report bundle.
