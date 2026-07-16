# ANC (Automated Node Checkout) CVS Tests

This directory packages the ANC validation suite as CVS tests:

| File | `cvs run` command | Purpose |
| --- | --- | --- |
| `anc_installation.py` | `cvs run anc_installation` | Download and install the ANC tool on every node. |
| `anc_test_cpu.py` | `cvs run anc_test_cpu` | Install ANC (pre-task) then run the ANC **CPU** validation groups and collect artifacts. |
| `anc_test_gpu.py` | `cvs run anc_test_gpu` | Install ANC (pre-task) then run the ANC **GPU** validation groups and collect artifacts. |

The `anc_test_cpu` / `anc_test_gpu` suites **install ANC themselves** as a
pre-task (skipping the install when every node already reports the configured
`anc_version`), so you do not need to run `anc_installation` first. Shared
logic — package install by archive flavour (deb/rpm/tar), version check, group
execution, and artifact collection — lives in `cvs/lib/anc_lib.py`;
shared pytest fixtures live in this directory's `conftest.py`.

The CPU/GPU group sets are defined by `CPU_GROUPS` / `GPU_GROUPS` in
`anc_lib.py` and run in a single `anc.py -g <groups...>` invocation per suite.
ANC is invoked from its installed location `/opt/amdtools/anc/anc.py`.

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
  `sudo` on each node. ANC runs as `sudo ./anc.py`, and artifact collection
  reclaims log ownership with `sudo chown`.

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

The ANC config lives at `config/anc_config.json`:

```json
{
    "_comment": "ANC test configuration",
    "anc": {
        "description": "Automated Network Connectivity checks",
        "test_timeout": 7200,
        "anc_release_url": "https://.../anc-release-helios-nda-1.3.2-tar-linux-x64.tar.gz",
        "cvs_home": "{home}/cvs",
        "anc_root_dir": "anc"
    }
}
```

| Key | Meaning |
| --- | --- |
| `test_timeout` | Per-group execution timeout in seconds (default 7200 = 2 h). |
| `anc_release_url` | ANC release tarball URL (used by `anc_installation`). |
| `cvs_home` | Install root on each node. `{home}` resolves to the SSH user's home. |
| `anc_root_dir` | ANC directory name under `cvs_home`. ANC is run from `<cvs_home>/<anc_root_dir>`. |

`{home}` is resolved on the controller from the cluster file's `username`, so it
works even when the runner and nodes have different home paths.

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
  --config_file cvs/tests/anc/config/anc_config.json
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
  --config_file cvs/tests/anc/config/anc_config.json
```

GPU validation:

```bash
cvs run anc_test_gpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/tests/anc/config/anc_config.json
```

- `anc_test_cpu` runs `sudo ./anc.py -g <CPU_GROUPS>`
- `anc_test_gpu` runs `sudo ./anc.py -g <GPU_GROUPS>`

The exact group lists are defined by `CPU_GROUPS` / `GPU_GROUPS` in
`cvs/lib/anc_lib.py`. To skip the install pre-task and run only the
group test, add pytest's `-k` filter (forwarded by `cvs run`):

```bash
cvs run anc_test_cpu -k test_cpu \
  --cluster_file cvs/input/cluster_file/cluster.json \
  --config_file cvs/tests/anc/config/anc_config.json
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
