# TorchTitan configs

JSON configs and sibling `*_threshold.json` files for `torchtitan_single` / `torchtitan_distributed`. Do not use leftover `mi3xx_*` / `mi35x_*` files with the unified suites.

Sweep combination keys (and matching `sweep.runs` entries and threshold cells) are `MBS=<micro_batch_size>,GBS=<global_batch_size>,PRECISION=<precision>`.

`gpus_per_node` defaults to `8`. Set it when a node has a different GPU count. World size and tokens/s totals use that count.

- Schema: [docs/reference/configuration-files/training/torchtitan.rst](../../../../../docs/reference/configuration-files/training/torchtitan.rst)
- How to run: [docs/how-to/test-suites/training/torchtitan.rst](../../../../../docs/how-to/test-suites/training/torchtitan.rst)
