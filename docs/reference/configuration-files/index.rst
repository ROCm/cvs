.. meta::
  :description: Configure the details of each CVS configuration test file
  :keywords: configure, ROCm, test, health, RCCL, platform

************************
Test configuration files
************************

Each CVS test has a corresponding JSON configuration file. You must configure the JSON file for each test you want to run in CVS.

The test configuration files are in the ``cvs/input/config_file`` directory of the cloned repo. You can go to each directory and edit the parameters as necessary for your testing requirements.

.. note::

  Ensure `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ is installed correctly, and the GPU drivers are loaded.

Cluster file
============

In addition to a per-test ``--config_file``, every ``cvs run`` invocation needs a ``--cluster_file`` that declares the SSH credentials, the node list, and the **execution backend** (baremetal or container). See :doc:`/reference/cluster/cluster-file` for the full schema, the container block reference, and which suites consume the orchestrator today.

Test configuration files
========================

Configuration references are grouped by test category. Each page documents parameters and example JSON for suites under ``cvs/input/config_file/``.

- :doc:`Burn-in / Diag </reference/configuration-files/burn-in-diag/index>`
- :doc:`Network </reference/configuration-files/network/index>`
- :doc:`Training </reference/configuration-files/training/index>`
- :doc:`Inference </reference/configuration-files/inference/index>`

Burn-in / Diag
--------------

- :doc:`Platform </reference/configuration-files/burn-in-diag/platform>` — host OS, BIOS, firmware, and PCIe checks
- :doc:`Health </reference/configuration-files/burn-in-diag/health>` — AGFHC, TransferBench, and RVS burn-in configs
- :doc:`Preflight </reference/configuration-files/burn-in-diag/preflight>` — node smoke and cluster preflight checks

Network
-------

- :doc:`InfiniBand (IB Perf) </reference/configuration-files/network/ib>` — IB bandwidth and latency benchmarks
- :doc:`RCCL </reference/configuration-files/network/rccl>` — multi-node collective communication performance
- :doc:`MORI (RDMA Performance) </reference/configuration-files/network/mori>` — RDMA read/write bandwidth and latency

Training
--------

- :doc:`JAX MaxText </reference/configuration-files/training/jaxmaxtext>` — MaxText pre-training (single-node and distributed)
- :doc:`Megatron </reference/configuration-files/training/megatron>` — Llama distributed Megatron training
- :doc:`TorchTitan </reference/configuration-files/training/torchtitan>` — TorchTitan pre-training (single-node and distributed)
- :doc:`Aorta (Distributed Training) </reference/configuration-files/training/aorta>` — Aorta RCCL/training throughput benchmark

Inference
---------

- :doc:`ATOM (vLLM Benchmarking) </reference/configuration-files/inference/atom>` — ATOM inference benchmarks
- :doc:`vLLM Inference </reference/configuration-files/inference/vllm>` — vLLM serving throughput and latency
- :doc:`SGLang Inference </reference/configuration-files/inference/sglang>` — single-node, distributed, and disaggregated LLM serving
- :doc:`xDiT Inference </reference/configuration-files/inference/xdit>` — FLUX.1/FLUX.2 text-to-image and WAN 2.2 image-to-video
