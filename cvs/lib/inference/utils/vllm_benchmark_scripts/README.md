# vLLM benchmark server scripts (shared)

Shell entrypoints for **`vllm serve`** kept for legacy **InferenceBaseJob** flows.

- **`vllm_serve_mi300x.sh`** — reference MI300-class server flags (``enforce-eager``, ``gpu-memory-utilization``, etc.). **InferenceX ATOM** and **vllm_single** encode equivalent flags in ``roles.server.serve_args`` instead of running this script.

**Client benchmarks** use ``vllm bench serve`` (stock results artifact). CVS no longer clones a third-party ``bench_serving`` git repo for InferenceX ATOM.

If **both** the script path and the bench CLI are unavailable, install bench-capable vLLM in the image (e.g. `pip install 'vllm[bench]'`) or bind-mount a matching `benchmarks/` tree from a vLLM checkout.

Client invocations also pass ``--temperature 0`` so greedy sampling matches the legacy
``benchmark_serving.py`` default, and clamp ``--random-range-ratio`` when the configured
spread would let ``(ISL+OSL)*(1+r)`` exceed ``max_model_length`` (otherwise vLLM rejects
most random prompts).

Python API: ``bundled_scripts_dir()``, ``bash_export_bench_script_from_vllm_install()``,
``clamped_bench_random_range_ratio_str()``, ``validated_bench_script_basename()``.
