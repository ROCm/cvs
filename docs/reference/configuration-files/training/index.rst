.. meta::
  :description: Index of CVS training test configuration schemas, covering JAX MaxText, Megatron, TorchTitan, Aorta, and PyTorch Vision distributed training benchmark suites.
  :keywords: CVS, training, JAX, MaxText, Megatron, TorchTitan, Aorta, PyTorch Vision, ROCm, GPU, AMD, distributed, benchmark, JSON

******************************************************************
Cluster Validation Suite (CVS) training test configuration schemas
******************************************************************

JSON configuration schemas for distributed training benchmarks under ``cvs/input/config_file/training/``.

- :doc:`JAX MaxText </reference/configuration-files/training/jaxmaxtext>` — MaxText pre-training (single-node and distributed)
- :doc:`Megatron </reference/configuration-files/training/megatron>` — Llama and DeepSeek, single-node and distributed
- :doc:`PyTorch Vision </reference/configuration-files/training/pytorch_vision>` — torchvision ResNet-50 on ImageNet via rocAL (single-node and distributed)
- :doc:`TorchTitan </reference/configuration-files/training/torchtitan>` — TorchTitan pre-training (single-node and distributed)
- :doc:`Aorta (Distributed Training) </reference/configuration-files/training/aorta>` — Aorta RCCL/training throughput benchmark

See :doc:`/how-to/test-suites/training/index` for more information on running these tests.
