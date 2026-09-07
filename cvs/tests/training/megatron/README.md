# Megatron training suite

Unified suites `megatron_single` and `megatron_distributed` run Megatron-LM or Primus pre-training. If `container.image` contains `primus`, the suite uses Primus; otherwise Megatron-LM. Use a `*_single.json` config with `megatron_single` and a `*_distributed.json` config with `megatron_distributed`.

- How to run: [docs/how-to/test-suites/training/megatron.rst](../../../../docs/how-to/test-suites/training/megatron.rst)
- Schema: [docs/reference/configuration-files/training/megatron.rst](../../../../docs/reference/configuration-files/training/megatron.rst)
