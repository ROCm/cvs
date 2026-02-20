The benchmark tests run distributed training benchmarks validated by CVS. The **Aorta** benchmark executes an Aorta-based workload in a Docker container with RCCL, collects PyTorch profiler traces, and validates iteration time, compute ratio, overlap ratio, and rank balance against configurable thresholds.

# How to run the tests

For details on arguments and their purpose, see the main README under the CVS parent folder.

1. **Config file:** Edit `cvs/input/config_file/aorta/aorta_benchmark.yaml` and set `aorta_path` to the absolute path of your Aorta repository. Do not leave the default `<changeme>`.
2. **Cluster file:** Provide a valid cluster file (e.g. `input/cluster_file/cluster.json`) with node and user settings.

Example from the ``cvs`` directory (repository root):

```bash
(myenv) [user@host]~/cvs:(main)$ pwd
/home/user/cvs/cvs
(myenv) [user@host]~/cvs:(main)$ pytest -vvv --log-file=/tmp/test.log -s ./tests/benchmark/test_aorta.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/aorta/aorta_benchmark.yaml --html=/var/www/html/cvs/aorta.html --capture=tee-sys --self-contained-html
```

With verbose logging:

```bash
pytest ./tests/benchmark/test_aorta.py --cluster_file input/cluster_file/cluster.json --config_file input/config_file/aorta/aorta_benchmark.yaml -v --log-cli-level=INFO
```

# Config and expected results

Configuration options (paths, Docker image, RCCL build, environment, analysis, and expected-result thresholds) are documented in the reference docs under `docs/reference/configuration-files/aorta.rst`. Key settings:

- **aorta_path** – Path to Aorta repo on the host (bind-mounted into the container).
- **expected_results** – Optional thresholds for validation: `max_avg_iteration_ms`, `min_compute_ratio`, `min_overlap_ratio`, `max_time_variance_ratio`.

The test parses results from host-side trace parsing (raw PyTorch profiler traces or TraceLens reports when present) and fails if any configured threshold is not met. Artifacts include training logs, profiler traces, and an optional TraceLens analysis directory when enabled.
