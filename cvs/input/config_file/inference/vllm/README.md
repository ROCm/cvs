# vLLM inference configs

JSON configuration and threshold files for the `vllm_single` and
`vllm_distributed` suites. Full documentation:

- **Configuration and threshold reference:**
  [`docs/reference/configuration-files/inference/vllm.rst`](../../../../../docs/reference/configuration-files/inference/vllm.rst)
- **How to run the suites:**
  [`docs/how-to/test-suites/inference/vllm.rst`](../../../../../docs/how-to/test-suites/inference/vllm.rst)

Each workload is shipped as a `single` / `distributed` pair:

```text
mi3xx_vllm_<model>_<precision>_<topology>.json
mi325x_vllm_<model>_<precision>_<topology>_threshold.json
```

Each configuration names its sibling threshold file in `threshold_json`.
