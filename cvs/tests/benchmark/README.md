The benchmark tests run distributed training benchmarks validated by CVS. The **Aorta** benchmark executes an Aorta-based workload in a Docker container with RCCL, collects PyTorch profiler traces, and validates iteration time, compute ratio, overlap ratio, and rank balance against configurable thresholds.

# How to run the tests

For details on arguments and their purpose, see the main README under the CVS parent folder.

1. **Config file:** Edit `input/config_file/aorta/aorta_benchmark.yaml`. You may use path placeholders (`{user-id}`, `{home}`, etc.); `test_aorta` resolves them after the cluster file is validated, like other CVS suites. Use an absolute path or valid placeholders, and ensure the tree exists (or enable `aorta_auto_clone` with `aorta_clone_url`).
2. **Cluster file:** Provide a valid cluster file (e.g. `input/cluster_file/cluster.json`) with node and user settings. If your SSH key is not under `/home/<username>/.ssh/`, set `priv_key_file` explicitly (the template assumes that layout).

Run from the **CVS package directory**—the directory that contains the `input` folder (the inner `cvs` directory in a normal checkout, next to `tests` and `lib`):

```bash
(myenv) [user@host myproject/cvs]$ pwd
/home/user/myproject/cvs
(myenv) [user@host myproject/cvs]$ cvs run test_aorta --cluster_file input/cluster_file/cluster.json --config_file input/config_file/aorta/aorta_benchmark.yaml --html=logs/www/html/cvs/aorta.html --capture=tee-sys --self-contained-html --log-file=logs/aorta.log -vvv -s
```

With HTML report and full logging (see also `docs/reference/configuration-files/aorta.rst`):

```bash
cvs run test_aorta --cluster_file input/cluster_file/cluster.json --config_file input/config_file/aorta/aorta_benchmark.yaml --html=logs/www/html/cvs/aorta.html --capture=tee-sys --self-contained-html --log-file=logs/aorta.log -v --log-cli-level=INFO
```

# Config and expected results

Configuration options (paths, Docker image, RCCL build, environment, analysis, and expected-result thresholds) are documented in the reference docs under `docs/reference/configuration-files/aorta.rst`. Key settings:

- **aorta_path** – Path to Aorta repo on the host (bind-mounted into the container).
- **expected_results** – Validation thresholds; the test fails if any is not met:
  - **max_avg_iteration_ms** – Maximum acceptable average iteration time (ms).
  - **min_compute_ratio** – Minimum compute ratio (compute time / total iteration time).
  - **min_overlap_ratio** – Minimum compute–communication overlap ratio.
  - **max_time_variance_ratio** – Maximum iteration time variance (e.g. std/mean) across ranks.

The values in `aorta_benchmark.yaml` are **default thresholds for gfx942** (e.g. MI300) and should be **changed as per your testing config** (GPU, node count, workload). The test parses results from host-side trace parsing (raw PyTorch profiler traces or TraceLens reports when present). Artifacts include training logs, profiler traces, and an optional TraceLens analysis directory when enabled.

# Note for users: where to put Aorta (`aorta_path`)

**Prefer a path on local or scratch storage** (e.g. `/scratch/...`) for `aorta_path` when running this benchmark.

If `aorta_path` points to a directory on **NFS** (for example your home directory under `/home`), the container may fail with **Permission denied** when creating the `artifacts/` output directory. Many NFS exports use *root_squash*, so the process running as root inside the container is treated as a non-privileged user on the NFS server and cannot create directories in your tree. Using a path on local disk or on a non–root-squashed filesystem (e.g. `/scratch`) avoids this. No code changes are required—use a suitable path in `aorta_benchmark.yaml` for `aorta_path`.
