# Benchmark suites

The [Aorta suites](aorta/README.md) run RCCL/training benchmarks through the CVS container
orchestrator, collect profiler artifacts and validate performance against configurable
thresholds.

Use `cvs run aorta_single` for one node or `cvs run aorta_distributed` for multiple nodes,
with `--cluster_file` and a JSON `--config_file` from `input/config_file/benchmark/aorta/`.
See the Aorta guide for configuration, artifact paths and migration from the old YAML runner.
