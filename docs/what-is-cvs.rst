.. meta::
  :description: CVS is a collection of test suites that qualify AMD ROCm GPU clusters for production, covering health, networking, training, and inference workloads.
  :keywords: CVS, ROCm, cluster validation, AMD, GPU, test suite, qualification, burn-in, InfiniBand, RCCL, preflight, SSH

What is Cluster Validation Suite (CVS) for ROCm?
=================================================

CVS is a collection of test suites that validate AMD ROCm clusters.
Use CVS to verify GPU cluster health, GPU/CPU node health, host OS configuration, and NIC (network interface card) validation.

Here are the tests available in CVS:

- **Preflight tests**: Run node smoke checks before a full test campaign to catch configuration issues early. Preflight validates per-node SSH connectivity, GPU visibility, driver load, and basic device access so that failures are caught before they consume cluster time.
- **Platform tests**: Perform host OS configuration, BIOS, firmware/driver, and network configuration checks.
- **Burn-in health tests**: Perform `AMD GPU Field Health Check (AGFHC) <https://instinct.docs.amd.com/projects/gpu-operator/en/latest/test/agfhc.html>`_, `TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/install/install.html#install-transferbench>`_, and `ROCm Validation Suite (RVS) <https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/install/installation.html>`_. For MI4XX platforms, the suite also includes AMD Node Check (ANC).
- **InfiniBand (IB Perf)**: Low-level network performance benchmarks that validate the raw communication capabilities of InfiniBand adapters and interconnects. These tests measure the fundamental building blocks on which RCCL and other high-level libraries depend.
- **RCCL network tests**: Perform ping checks and multi-node `ROCm Communication Collectives Library (RCCL) <https://rocm.docs.amd.com/projects/rccl/en/latest/install/installation.html>`_ validations for different collectives.
- **RDMA performance tests**: Validate RDMA (Remote Direct Memory Access) bandwidth and latency with MORI for high-speed inter-node communication using AMD Pensando AINIC and other RDMA-capable devices.
- **Distributed training tests**: Run and validate model trainings on single-node or multi-node clusters.

  - The JAX MaxText suites (``jaxmaxtext_single`` / ``jaxmaxtext_distributed``) use PyTest and a container orchestrator to launch containers and run/verify MaxText pre-training, gating each run on performance and convergence metrics.
  - Megatron training enables scaling transformer models from millions to trillions of parameters by efficiently utilizing hundreds or thousands of GPUs across multiple nodes with Llama and DeepSeek workloads.
  - TorchTitan pre-training benchmarks validate transformer model training across distributed AMD GPU clusters.
  - Aorta distributed training benchmarks validate RCCL performance and training throughput across multi-node clusters.

- **Inference tests**: Validate LLM serving performance and generative AI workloads across AMD GPU clusters.

  - ATOM benchmarks vLLM inference performance for models like GPT-OSS-120B, measuring throughput, time to first token (TTFT), and time per output token (TPOT).
  - vLLM single-node and distributed serving tests cover a packaged MI3xx catalog (Llama 3.3 70B FP8 and 13 other workloads), measuring throughput, latency, and optional accuracy.
  - SGLang disaggregated prefill-decode architecture tests optimize LLM serving by separating prefill and decode phases across different nodes.
  - Flux.1 text-to-image generation tests validate distributed image generation using xDiT with Ulysses and Ring parallelization.
  - WAN 2.2 image-to-video generation tests validate 81-frame video generation with distributed inference.

In addition to test suites, CVS provides cluster utility commands:

- :doc:`cvs exec <how-to/execute-cluster-commands>` runs any shell command across all cluster nodes simultaneously over parallel SSH, returning per-node output for quick ad-hoc diagnostics.
- :doc:`cvs copy <how-to/copy-to-cluster>` copies files or directories to all cluster nodes in parallel, useful for distributing configuration files or binaries before a test run.

CVS can also run test workloads through a :doc:`per-host Docker container backend <how-to/run-with-containers>` instead of directly on the host filesystem. Use the container backend when you want to validate the same image you ship to production, or to keep the host footprint to Docker, the GPU driver, and SSH only.

You can :doc:`monitor cluster health <how-to/monitor/index>` in two ways:

- :doc:`Health reports <how-to/monitor/health-reports/index>`: Run the Cluster Health Checker to generate a self-contained HTML report covering GPU/NIC counters, RAS errors, PCIe/XGMI status, and ``dmesg`` signatures. Use this for point-in-time snapshots before or after test campaigns.
- :doc:`Live dashboards <how-to/monitor/live-dashboards/index>`: Stream real-time GPU and cluster metrics to Grafana using either an agentless SSH-based collector or Prometheus exporters installed on the cluster nodes.

CVS uses the open-source PyTest framework to run the tests and generate reports. You can launch CVS from a head node or any Linux management station that has connectivity to the cluster nodes via SSH. The single node tests run cluster-wide in parallel using the open-source parallel-SSH Python modules to optimize their running time.

.. note::

   CVS has been validated on Ubuntu-based Linux distribution clusters.
