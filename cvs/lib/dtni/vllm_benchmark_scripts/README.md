# vLLM benchmark server scripts (shared)

Shell entrypoints for **`vllm serve`** used by CVS **vllm_single** (`VllmJob` in `cvs.lib.inference.vllm_orch`) and **InferenceMax** host-mounted server flows.

- **`vllm_serve_mi300x.sh`** — default MI300-class server wrapper; the served checkpoint is whatever you set in `MODEL` (the filename is not model-specific).

- Point **vLLM** `paths.benchmark_scripts_dir` (host path, bind-mounted into the container) at a directory that contains copies of—or symlinks to—these files, **or** mount this package directory.
- Point **InferenceMax** `host_benchmark_scripts_relpath` at `lib/dtni/vllm_benchmark_scripts` (relative to the `cvs` Python package root) unless you override `benchmark_server_script_path`.

**Client benchmarks** resolve a driver at runtime inside the container, in order:

1. ``vllm/benchmarks/<bench_serv_script>`` from the installed vLLM package
2. ``/app/bench_serving/<bench_serv_script>`` (cloned from ``benchmark_script_repo`` when the wheel omits benchmarks)
3. ``python -m vllm.entrypoints.cli.main bench serve`` when the CLI exposes it

Python API: ``bundled_scripts_dir()``, ``bash_export_bench_script_from_vllm_install()``, ``clamped_bench_random_range_ratio_str()``, ``validated_bench_script_basename()``.
