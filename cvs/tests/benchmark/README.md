The benchmark tests run distributed training benchmarks validated by CVS. The **Aorta** benchmark executes an Aorta-based workload in a Docker container with RCCL, collects PyTorch profiler traces, and validates iteration time, compute ratio, overlap ratio, and rank balance against configurable thresholds.

# How to run the tests

For details on arguments and their purpose, see the main README under the CVS parent folder.

1. **Config file:** Edit `cvs/input/config_file/aorta/aorta_benchmark.yaml` and set `aorta_path` to the absolute path of your Aorta repository. Do not leave the default `<changeme>`.
2. **Cluster file:** Provide a valid cluster file (e.g. `input/cluster_file/cluster.json`) with node and user settings.

Example from the CVS repo root (directory containing ``cvs`` and ``input``):

```bash
cvs run test_aorta \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/aorta/aorta_benchmark.yaml \
    -v --log-cli-level=INFO
```

With HTML report and full logging (see also `docs/reference/configuration-files/aorta.rst`):

```bash
cvs run test_aorta \
    --cluster_file input/cluster_file/cluster.json \
    --config_file input/config_file/aorta/aorta_benchmark.yaml \
    --html=/var/www/html/cvs/aorta.html --capture=tee-sys --self-contained-html \
    --log-file=/tmp/test.log -vvv -s
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
