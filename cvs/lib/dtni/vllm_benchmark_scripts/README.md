# vLLM benchmark server scripts (shared)

Shell entrypoints for **`vllm serve`** used by CVS **vllm_single** (`VllmJob` in `cvs.lib.inference.vllm_orch`) and **InferenceMax** host-mounted server flows.

- **`vllm_serve_mi300x.sh`** — default MI300-class server wrapper; the served checkpoint is whatever you set in `MODEL` (the filename is not model-specific). Use **`CVS_GPU_MEMORY_UTIL`** in `container_config.env_dict` for `--gpu-memory-utilization` (legacy `VLLM_GPU_MEMORY_UTIL` is still read once then unset so vLLM does not warn). ROCm **AITER** `VLLM_*` tuning vars may still produce vLLM “unknown env” warnings until those names are whitelisted upstream; they are optional performance knobs.

- Point **vLLM** `paths.benchmark_scripts_dir` (host path, bind-mounted into the container) at a directory that contains copies of—or symlinks to—these files, **or** mount this package directory.
- Point **InferenceMax** `host_benchmark_scripts_relpath` at `lib/dtni/vllm_benchmark_scripts` (relative to the `cvs` Python package root) unless you override `benchmark_server_script_path`.

**Client benchmarks** prefer the Python file shipped with the installed **vLLM** package under `vllm/benchmarks/<bench_serv_script>` (resolved at runtime inside the container). If that file is missing (many wheels omit `benchmarks/`), CVS falls back to `python -m vllm.entrypoints.cli.main bench serve` with the same flags. CVS no longer clones a third-party `bench_serving` git repo.

If **both** the script path and the bench CLI are unavailable, install bench-capable vLLM in the image (e.g. `pip install 'vllm[bench]'`) or bind-mount a matching `benchmarks/` tree from a vLLM checkout.

Client invocations also pass ``--temperature 0`` so greedy sampling matches the legacy
``benchmark_serving.py`` default, and clamp ``--random-range-ratio`` when the configured
spread would let ``(ISL+OSL)*(1+r)`` exceed ``max_model_length`` (otherwise vLLM rejects
most random prompts).

Python API: ``bundled_scripts_dir()``, ``bash_export_bench_script_from_vllm_install()``,
``clamped_bench_random_range_ratio_str()``, ``validated_bench_script_basename()``.
