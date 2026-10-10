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

**Group / item suites** — there are two group suites, `anc_test_cpu` and
`anc_test_gpu` (in the `cpu/` and `gpu/` subfolders), plus per-family
individual-item suites — `anc_test_computerocker`, `anc_test_memrocker`,
`anc_test_oblex`, `anc_test_gemm`, `anc_test_xgmi`, `anc_test_ualink`,
`anc_test_pcie`, `anc_test_babel`, and `anc_test_basic` (a catch-all for one-off
items) — each in its own subfolder. Each holds one `test_<name>` **function**
per ANC group (cpu/gpu) or individual item (the family suites). Running the
whole suite runs every function in that set; naming a function runs just that
one. Each function ensures ANC is installed and ROCm ldconfig is fixed first
(session-cached via `anc_lib.ensure_anc_ready`, so a full-suite run pays the
setup cost once). The suite files are **generated** from the single source in
`anc_lib` — `CPU_GROUPS` / `GPU_GROUPS` and the per-family item lists
(`COMPUTEROCKER_ITEMS`, `MEMROCKER_ITEMS`, `OBLEX_ITEMS`, `GEMM_ITEMS`,
`XGMI_ITEMS`, `UALINK_ITEMS`, `PCIE_ITEMS`, `BABEL_ITEMS`, `BASIC_ITEMS`). The
DIMM/UMC groups live in `anc_test_cpu` because ANC reports them under the CPU
device. Groups run as `anc.py -g <group>`; individual items run as
`anc.py -i <item>`.

> Generated files — do NOT hand-edit. To add/remove a group or item, edit the
> lists in `cvs/lib/anc_lib.py` and run `make gen-anc-suites` (wraps
> `build_tools/gen_anc_suites.py`), which rewrites the suite files and prunes
> stale ones. Then reinstall (`make install` / `pip install .`).

```bash
# run every CPU group (each group its own test + log dir)
cvs run anc_test_cpu                  --cluster_file <c.json> --config_file <cfg.json>
# run a single group by its function name
cvs run anc_test_cpu test_cpu_mfg_l10 --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_cpu test_dimm_content_check --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_gpu test_hbm_lvl1    --cluster_file <c.json> --config_file <cfg.json>
# run a whole item family, or just one item from it
cvs run anc_test_gemm                 --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_gemm test_gemm_fp8_trig --cluster_file <c.json> --config_file <cfg.json>
cvs run anc_test_basic test_no_op     --cluster_file <c.json> --config_file <cfg.json>
# list the per-group/per-item functions in a suite
cvs list anc_test_cpu
cvs list anc_test_computerocker
```

- CPU (`anc_test_cpu`): `test_cpu_content_check`, `test_cpu_mfg_l10`,
  `test_weighted_sanity`, `test_dimm_content_check`, `test_dimm_mfg_l10`,
  `test_dimm_weighted_sanity`
- GPU (`anc_test_gpu`): `test_gpu_content_check`, `test_gpu_mfg_l10`,
  `test_hbm_lvl1` … `test_hbm_lvl5`
- Item families — one `test_<item>` per ANC item, grouped by tool/subsystem:
  `anc_test_computerocker` (24), `anc_test_memrocker` (10), `anc_test_oblex`
  (8), `anc_test_gemm` (3), `anc_test_xgmi` (3), `anc_test_ualink` (3),
  `anc_test_pcie` (3), `anc_test_babel` (2), and `anc_test_basic` (18 one-off
  items, e.g. `test_ampttk`, `test_hdrt`, `test_no_op`, `test_sdma_bidi_peak`).
  See the per-family lists in `cvs/lib/anc_lib.py`, or run
  `cvs list anc_test_<family>`.

Shared logic — package install by archive flavour (deb/rpm/tar), version check,
the session-cached setup guard `ensure_anc_ready`, ldconfig fix, group/item
execution (`run_anc_groups` / `run_anc_items`, both thin wrappers over the
shared `run_anc_selection` core), and artifact collection — lives in
`cvs/lib/anc_lib.py` (group sets are `CPU_GROUPS` / `GPU_GROUPS` with the
DIMM/UMC groups part of `CPU_GROUPS`; individual items are in the per-family
lists above).
Shared pytest fixtures live in this directory's `conftest.py` and apply to the
`cpu/`, `gpu/` and all per-family item subfolders too. ANC is invoked
from its installed location `<prefix>/anc/anc.py` — `<prefix>` is `/opt/amdtools`
by default, or, for **tar** installs only, the relocated `ANC_INSTALL_PATH`
(deb/rpm always use `/opt/amdtools`).

**Logs & console:** each group's ANC log directory is copied under this run's
`run_dir`, where CVS lays down the fixed
`anc_logs/<node>/<test_name>/<timestamp>` structure (`<node>` is that node's
`<ip>_<hostname>` label — so multi-node runs group every test/timestamp under
each node's own folder); the resolved pattern is printed before the run. Set `print_all_to_console` to `False` in config to suppress the
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
  cvs list               # anc_installation, anc_test_cpu, anc_test_gpu, anc_test_<family> (gemm, xgmi, basic, ...)
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
- The per-node artifact folder label is derived **from the cluster file only**
  (no SSH/hostname lookup): it is `<vpc_ip>_<node_dict key>`. When `vpc_ip` is
  missing, `"NA"`, or equal to the node_dict key, the label collapses to just the
  node_dict key (no duplicated name).

### Config file

The ANC config lives at `cvs/input/config_file/anc/anc_config.json`:

```json
{
    "_comment": "ANC test configuration",
    "anc": {
        "description": "AMD Node Check",
        "inactivity_timeout": 900,
        "install_timeout": 1800,
        "anc_version": "1.7.3",
        "anc_release_url": "<changeme>",
        "ANC_INSTALL_PATH": "",
        "print_all_to_console": "True",
        "ADD_ANC_LOGS_TO_HTML_REPORTS": "False"
    }
}
```

`anc_release_url` ships as the `<changeme>` placeholder and **must** be replaced
before running; an unresolved `<changeme>` aborts the run up front (the standard
config resolver hard-exits on it, before any node is contacted).

Each key is documented inline in the shipped config via a matching
`_comment_<key>` sibling (keys prefixed with `_comment` are ignored at runtime).

| Key | Meaning |
| --- | --- |
| `inactivity_timeout` | Per-group **inactivity** timeout in seconds (default 900 = 15 min). Applies to ANC group runs only, not install. A group is aborted only after this many seconds with **no new ANC output**; there is no total wall-clock cap, so a group that keeps producing progress runs as long as it needs. (Replaces the old `test_timeout` total-budget cap, which killed healthy, actively-running groups.) |
| `install_timeout` | Package download+install **inactivity** timeout in seconds (default 1800 = 30 min), used only by `anc_installation` / the install pre-task. It is a per-read (no-output) timeout, not a total budget: the download emits a periodic progress heartbeat so a slow link never trips it, while a genuine stall still fails. Independent of `inactivity_timeout`. |
| `anc_version` | Minimum required ANC version. The install pre-task skips (re)install **only when every expected node already satisfies** it (installed ≥ requested); if any node is below the minimum, the installer runs on **all** nodes, so already-satisfying nodes in a mixed cluster are reinstalled. After installing it post-verifies every node satisfies the version. A release candidate counts as its base release (`1.7.0-rc.1` satisfies `1.7.0`), and rc-vs-rc of the same base compares by number. When set, the version in `anc_release_url` must be **≥** `anc_version` (the archive must be able to satisfy the request) or the run aborts before contacting any node. The installed version is read per node, with each node's packaging generation **auto-detected from its own output**: `anc.py --content-list` (the `anc-release-*` plugin's version column) when that line is present (**direct 1.5.0+**), falling back to `anc.py --version` for **legacy ≤1.4.x** (whose `--version` reports the release version; 1.5.0+ does not). |
| `anc_release_url` | ANC release archive URL (used by `anc_installation`). Both packaging generations are auto-detected from the filename: **legacy (≤1.4.x)** outer tarballs carry a `-deb-`/`-rpm-`/`-tar-` token (e.g. `anc-release-helios-nda-1.4.9-tar-linux-x64.tar.gz`); **direct (1.5.0+)** URLs point straight at a `.deb`/`.rpm`/`.tar.gz` with no flavour token (e.g. `anc-release-helios-nda-1.5.5-x86_64.tar.gz`). The download/unpack is staged in a private temp dir on each node and removed after install (success or failure). deb/rpm install to `/opt/amdtools/anc`; tar installs to `ANC_INSTALL_PATH` (default `/opt/amdtools`). |
| `ANC_INSTALL_PATH` | **Tar installs only:** relocatable prefix ANC is extracted into (its `anc/` dir, tool folders, and content live under here), giving `<prefix>/anc/anc.py`. A leading `~` is expanded and the `{home}`/`{user-id}` placeholders are resolved during config load. The value is validated up front: shell-metacharacters (`'`, `"`, `` ` ``, `$`, `\`, newline) and a prefix that resolves to filesystem root (`/`, `//`) are rejected before any node is contacted. deb/rpm packages carry absolute locations baked into their archive and **ignore** this key. Leave blank or omit to keep the default `/opt/amdtools`. |
| `print_all_to_console` | `True` echoes ANC group output to console; `False` suppresses it (diagnostics still print). |
| `ADD_ANC_LOGS_TO_HTML_REPORTS` | Governs the per-node **ANC logs** tarball links. `True` always bundles each node's collected log tree (one `.tar.gz` + link per node) into the pytest-html report zip. `False` (default) bundles them **only when the test fails**. The per-node `errors.json` links appear regardless of this flag. |

ANC no longer takes an artifact-path config key. Collected logs and the pytest
HTML/log reports all land under this run's `run_dir`
(`<workspace>/cvs_runs/<run_id>/`, resolved by `RunLayout` — the same directory
`cvs run` writes `--html`/`--log-file` into). The collected log tree is laid down
at `<run_dir>/anc_logs/<node>/<test_name>/<timestamp>` (`<node>` → the node's
`<ip>_<hostname>` label, `<test_name>` → the group's or item's test name,
`<timestamp>` → per-run stamp). To send the HTML report or log file elsewhere, pass `--html` /
`--log-file` on the command line; use `--no-html` / `--no-log-file` to suppress
them.

---

## 3. Install ANC on the target nodes

Every ANC validation suite installs ANC as a pre-task, so a separate install
step is **optional**. Run `anc_installation` on its own when you want to
install/refresh ANC without running a validation group. deb/rpm always install to
`/opt/amdtools/anc`; **tar** installs to `ANC_INSTALL_PATH` (default
`/opt/amdtools`), giving the entrypoint `<prefix>/anc/anc.py`. You can install it
in either of the ways below.

### Option A - from the head node (recommended)

The **head node** is the controller from which CVS drives the target nodes. Run
the installer there; it installs ANC on every target node listed in the cluster
file:

```bash
cvs run anc_installation \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

This downloads the `anc_release_url` archive, installs ANC on every target node
(the flavour — deb/rpm/tar — and generation — legacy/direct — are auto-detected
from the filename; deb/rpm land in `/opt/amdtools/anc`, tar in
`ANC_INSTALL_PATH`), and validates the install.

### Option B - manually on a target node

If you do not want to run `anc_installation`, install ANC directly on the target
node. The examples below use `/opt/amdtools` as the prefix; substitute your
`ANC_INSTALL_PATH` for a relocated tar install.

> **Install a version that satisfies the configured `anc_version` minimum**
> (shipped default `1.7.3`). The two blocks below illustrate the two packaging
> **layouts** — the legacy `≤1.4.x` two-tarball form and the direct `1.5.0+`
> single-tree form; the legacy `1.4.9` URL is format-only (it is below the
> default minimum, so a suite run would reinstall over it). Use the direct
> `1.7.3` command for a current manual install, or point the URL at any release
> `≥ anc_version`.

**Legacy (≤1.4.x) tar — packaging-format example only** — the outer archive
holds two inner `anc-tool` and `anc-content` tarballs; extract both into the
prefix so the layout matches the deb/rpm packages:

```bash
STAGE=$(mktemp -d)              # private staging dir, only for the download/unpack
trap 'rm -rf "$STAGE"' EXIT     # auto-remove it on exit (success or failure)
cd "$STAGE"

# Download and extract the ANC release
wget -q "https://atlartifactory.amd.com:8443/artifactory/HW-ANCRelease-REL-LOCAL/anc-release/helios_nda/1.4.9/anc-release-helios-nda-1.4.9-tar-linux-x64.tar.gz" \
  -O outer.tar.gz
tar -xzf outer.tar.gz

# Extract the tool and content archives into /opt/amdtools (needs sudo).
# Replace the top-level folders the release ships so no stale files linger.
sudo mkdir -p /opt/amdtools
sudo tar -xzf anc-tool*.tar.gz    -C /opt/amdtools
sudo tar -xzf anc-content*.tar.gz -C /opt/amdtools
```

**Direct (1.5.0+) tar — current install** — the `.tar.gz` *is* the tree (no
inner archives); a single untar into the prefix lays down `anc/` and the tool
folders. This `1.7.3` command satisfies the shipped `anc_version` default:

```bash
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cd "$STAGE"

wget -q "https://atlartifactory.amd.com:8443/artifactory/HW-ANCRelease-REL-LOCAL/anc-release/helios_nda/1.7.3/anc-release-helios-nda-1.7.3-x86_64.tar.gz" \
  -O anc.tar.gz
sudo mkdir -p /opt/amdtools
sudo tar -xzf anc.tar.gz -C /opt/amdtools
```

> For a relocated tar prefix, `anc_installation` also rewrites the content
> YAMLs' absolute `exe_path` entries from `/opt/amdtools` to the new prefix — do
> the same if you relocate a manual install.

For the **deb**/**rpm** flavours, install the extracted (legacy) or downloaded
(direct) `anc*.deb` / `anc*.rpm` packages instead (`sudo dpkg -i` /
`sudo dnf install`); they lay ANC down under `/opt/amdtools/anc` directly. This
is the sequence performed by [`anc_installation.py`](anc_installation.py) (which
delegates to `cvs/lib/anc_lib.py`).

---

## 4. Run the ANC validation tests

Each group/item function first ensures ANC is installed and ROCm ldconfig is
fixed (session-cached, so it happens once per run), then runs its group on
**all** nodes in parallel with a single `anc.py -g <group>` invocation (the
`anc_test_<family>` item functions run `anc.py -i <item>` instead).

**Single group / item** — name the `test_<group>` (or `test_<item>`) function on
its suite:

```bash
cvs run anc_test_cpu test_cpu_mfg_l10 \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_cpu test_dimm_content_check \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_gpu test_hbm_lvl1 \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_gemm test_gemm_fp8_trig \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

**All groups / items in a set** — run the whole suite. ANC install + ldconfig
happen once (session-cached), then every group/item in that suite (CPU, GPU, or
an item family) runs as its own test with its own log dir (the CPU suite covers
the DIMM/UMC groups too):

```bash
cvs run anc_test_cpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_gpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_computerocker \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json
```

The exact lists are defined by `CPU_GROUPS` / `GPU_GROUPS` and the per-family
item lists in `cvs/lib/anc_lib.py` (see the "Group / item suites" section
above). To run a subset, name several functions:

```bash
cvs run anc_test_cpu test_cpu_content_check test_dimm_content_check \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/input/config_file/anc/anc_config.json

cvs run anc_test_basic test_ampttk test_hdrt test_no_op \
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

Per node and per test, artifacts are downloaded under this run's `run_dir`, in
this fixed layout:

```
<run_dir>/anc_logs/<ip>_<hostname>/<test_name>/<timestamp>/
```

- `<run_dir>` is `<workspace>/cvs_runs/<run_id>/`, resolved by `RunLayout` — the
  same directory `cvs run` writes its `--html`/`--log-file` into. The workspace
  comes from `--workspace`, then `$CVS_WORKSPACE`, then (unmanaged runs only) the
  venv's parent; `<run_id>` is the scheduler job id under a scheduler, else a
  local timestamp.
- `anc_logs/<ip>_<hostname>/<test_name>/<timestamp>` is the fixed structure CVS
  lays down under `run_dir`: `<ip>_<hostname>` is the per-node label,
  `<test_name>` is the group's or item's test name (e.g. `test_cpu_mfg_l10` or
  `test_gemm_fp8_trig`), and `<timestamp>` keeps repeated runs separate.

Collected files:

- The **entire** ANC log directory is pulled back (whatever ANC wrote:
  `console.log`, `journal.log`, `summary.json`, per-item logs, …).
- **Required:** `console.log` (holds the verdict). If it is missing, the node
  fails. Every other file is collected best-effort as part of the whole-directory
  copy.

### In the HTML report

`cvs run` writes a self-contained pytest-html report by default at
`<run_dir>/<test-file-stem>.html` (and a text log at `<run_dir>/<test-file-stem>.log`).
Pass `--html` / `--log-file` to redirect them to an explicit path, or `--no-html` /
`--no-log-file` to suppress them. ANC adds no report machinery of its own.

**Reports section.** Above the results table, this section is failure-focused: it
lists **only the FAILED tests**. Each failed test shows a red **FAILED** verdict
naming the node(s) that failed and a short reason, and directly beneath it nests
that test's **failed-node** ANC log tarball link(s) — so the failed item and its
archive sit together. Passing tests, and passing-node tarballs, are omitted here
(the results table's Links column still carries them). When every ANC test
passed, a single green **All tests passed** note is shown instead. Non-ANC
artifacts (e.g. rccl/preflight) keep a flat bullet list.

**Links column (per test row).** Every row carries, in addition to **Full Log**:

- **One `errors.json` link per node** — `errors.json` (at the root of each node's
  collected log dir) is *copied* (not moved) next to the report as
  `<node>_<test>_<timestamp>_errors.json` and linked as **`errors_occured: <node>`**
  when that node failed, or **`no error: <node>`** when it passed. These links
  always appear (pass or fail); the original stays in the collected log tree.
- **One ANC-logs tarball link per node** — each node's collected tree is archived
  into its own `<node>_<test>_<timestamp>_anc_logs.tar.gz` and linked as
  **`ANC logs: <node>`** (suffixed `(FAILED)` on a failed node), controlled by
  `ADD_ANC_LOGS_TO_HTML_REPORTS`:
  - `True` — always attach (pass or fail).
  - `False` (default) — attach **only when the test fails**, so passing runs keep
    the report small while failures always ship their full logs.

The full tree is always written under `run_dir` regardless of these flags;
they only govern what gets embedded in the HTML report bundle.
