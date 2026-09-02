.. meta::
  :description: Training test configuration schemas
  :keywords: CVS, JAX, Megatron, TorchTitan, Aorta, config schema

********
Training
********

JSON configuration schemas for distributed training benchmarks under ``cvs/input/config_file/training/``.

- :doc:`JAX MaxText </reference/configuration-files/training/jaxmaxtext>` — MaxText pre-training (single-node and distributed)
- :doc:`Megatron </reference/configuration-files/training/megatron>` — Llama and DeepSeek, single-node and distributed
- :doc:`TorchTitan </reference/configuration-files/training/torchtitan>` — TorchTitan pre-training (single-node and distributed)
- :doc:`Aorta (Distributed Training) </reference/configuration-files/training/aorta>` — Aorta RCCL/training throughput benchmark

How to run these suites: :doc:`/how-to/test-suites/training/index`.
