# vLLM benchmark server scripts (shared)

Shell entrypoints for **`vllm serve`** used by CVS **vllm_single** (`VllmJob` in `cvs.lib.inference.vllm_orch`) and **InferenceMax** host-mounted server flows.

- **`vllm_serve_mi300x.sh`** — default MI300-class server wrapper; the served checkpoint is whatever you set in `MODEL` (the filename is not model-specific).

- Point **vLLM** `paths.benchmark_scripts_dir` (host path, bind-mounted into the container) at a directory that contains copies of—or symlinks to—these files, **or** mount this package directory.
- Point **InferenceMax** `host_benchmark_scripts_relpath` at `lib/dtni/vllm_benchmark_scripts` (relative to the `cvs` Python package root) unless you override `benchmark_server_script_path`.

**Client benchmarks** use the Python file shipped with the installed **vLLM** package under `vllm/benchmarks/<bench_serv_script>` (resolved at runtime inside the container). CVS no longer clones a third-party `bench_serving` git repo.

Some **vLLM wheels omit** the `benchmarks/` tree; if resolution fails, install bench extras in the image (e.g. `pip install 'vllm[bench]'`) or bind-mount `benchmark_serving.py` from a vLLM checkout that matches your server version.

Resolution exports **`BENCH_PY`** (the `sys.executable` that successfully imported `vllm`) and **`BENCH_SCRIPT`** (path to the driver), trying `python3.13` … `python3` so images where `python3` is not the vLLM interpreter still work.

Python API: `bundled_scripts_dir()`, `bash_export_bench_script_from_vllm_install()`, `validated_bench_script_basename()`.
