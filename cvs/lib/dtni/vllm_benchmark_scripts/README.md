# vLLM benchmark server scripts (shared)

Shell entrypoints for **`vllm serve`** used by CVS **vllm_single** (`VllmJob` in `cvs.lib.inference.vllm_orch`) and **InferenceMax** host-mounted server flows.

- Point **vLLM** `paths.benchmark_scripts_dir` (host path, bind-mounted into the container) at a directory that contains copies of—or symlinks to—these files, **or** mount this package directory.
- Point **InferenceMax** `host_benchmark_scripts_relpath` at `lib/dtni/vllm_benchmark_scripts` (relative to the `cvs` Python package root) unless you override `benchmark_server_script_path`.

Python API: `cvs.lib.dtni.vllm_benchmark_scripts.BENCH_SERVING_GIT_URL`, `bundled_scripts_dir()`.
