.. meta::
  :description: Choose the CVS test suite config file for your workload
  :keywords: CVS, configure, config_file, test suite

*************************
Choose config template
*************************

See :doc:`/reference/configuration-files/index` for field-level schemas.
Use ``cvs config copy <path> --output <dest>`` to copy a template, or ``cvs config list <path>`` to browse templates in a directory.

Platform, health, RCCL, and other diagnostic/network configs use **fixed filenames** (see **Burn-in / Diag** and **Network** below). **Training** and **inference** workloads use the naming patterns in their respective sections.

**Threshold pairs**

Some training and inference suites ship a matching threshold file for each config: the same basename with ``_threshold`` inserted before ``.json`` (for example ``…_single_threshold.json``). Copy both files and keep them in the same directory. The config references its threshold file via a ``threshold_json`` field.

Burn-in / Diag
==============

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Suite
     - Config location
     - List / README
   * - Platform
     - ``input/config_file/platform/host_config.json``
     - ``cvs config list platform``
   * - Health
     - ``input/config_file/health/mi300_health_config.json``
     - ``cvs config list health``
   * - Preflight
     - ``input/config_file/preflight/preflight_config.json``
     - ``cvs config list preflight``

       `README_preflight_config.md <https://github.com/ROCm/cvs/blob/main/cvs/input/config_file/preflight/README_preflight_config.md>`_

Network
=======

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Suite
     - Config location
     - List / README
   * - IB Perf
     - ``input/config_file/ibperf/ibperf_config.json``
     - ``cvs config list ibperf``
   * - RCCL
     - ``input/config_file/rccl/rccl_config.json``
     - ``cvs config list rccl``
   * - MORI
     - ``input/config_file/mori/mi35x_mori_config.json``
     - ``cvs config list mori``

Training
========

Training workload templates use:

.. code:: text

  {gpu}_{framework}_{model}_{mode}.json

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Segment
     - Meaning
   * - ``{gpu}``
     - Target GPU architecture — for example ``mi300x``, ``mi325x``, ``mi355x``, or ``mi3xx``.
   * - ``{framework}``
     - Training stack — for example ``jaxmaxtext``, ``megatron``, ``torchtitan``.
   * - ``{model}``
     - Model identifier — for example ``llama3_1_70b``, ``llama-3.1-8b``, ``deepseek-v2-lite``.
   * - ``{mode}``
     - Topology — ``single`` (one node) or ``distributed`` (multi-node).

**Example**

.. code:: text

  input/config_file/training/jaxmaxtext/
  ├── mi300x_jaxmaxtext_llama-3.3-70b_single.json
  └── mi325x_jaxmaxtext_llama-3.3-70b_distributed.json

``mi325x_jaxmaxtext_llama-3.3-70b_distributed.json`` → ``mi325x`` · ``jaxmaxtext`` · ``llama-3.3-70b`` · ``distributed``.

**Suites**

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Suite
     - Config location
     - List / README
   * - JAX MaxText
     - ``input/config_file/training/jaxmaxtext/`` — ``mi{gpu}_jaxmaxtext_{model}_{single|distributed}.json`` + ``mi{gpu}_jaxmaxtext_{model}_{single|distributed}_threshold.json``
     - ``cvs config list training/jaxmaxtext``

       `Config README <https://github.com/ROCm/cvs/blob/main/cvs/input/config_file/training/jaxmaxtext/README.md>`_
   * - Megatron
     - ``input/config_file/training/megatron/`` — ``mi{gpu}_megatron_{model}_{single|distributed}.json`` + ``mi{gpu}_megatron_{model}_{single|distributed}_threshold.json``
     - ``cvs config list training/megatron``

       `Config README <https://github.com/ROCm/cvs/blob/main/cvs/input/config_file/training/megatron/README.md>`_
   * - TorchTitan
     - ``input/config_file/training/torchtitan/`` — ``mi{gpu}_torchtitan_{model}_{single|distributed}.json`` + ``mi{gpu}_torchtitan_{model}_{single|distributed}_threshold.json``
     - ``cvs config list training/torchtitan``

       :doc:`Schema reference </reference/configuration-files/training/torchtitan>`
   * - Aorta
     - ``input/config_file/aorta/aorta_benchmark.yaml``
     - ``cvs config list aorta``

Inference
=========

Inference templates use **one of two** filename patterns:

.. code:: text

  {gpu}_{framework}_{model}_{mode}.json
  {gpu}_{framework}_{model}_{precision}_{mode}.json

Use the second form when precision (``fp8``, ``mxfp4``, ``bf16``, and similar) is a separate token after the model name.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Segment
     - Meaning
   * - ``{gpu}``
     - Target GPU architecture — for example ``mi300x``, ``mi30x``, ``mi355x``.
   * - ``{framework}``
     - Inference stack — for example ``atom``, ``vllm``, ``sglang``, ``pytorch_xdit``.
   * - ``{model}``
     - Model identifier — for example ``llama_70b``, ``deepseek-r1``, ``flux1_dev``, ``gpt-oss-120b``.
   * - ``{precision}``
     - Optional quantization or dtype token — for example ``fp8``, ``mxfp4``. Omitted when precision is part of ``{model}`` or not applicable.
   * - ``{mode}``
     - Topology or workload shape — ``single``, ``distributed``, or ``disaggregated`` (SGLang).

**Examples**

Without ``{precision}``:

.. code:: text

  input/config_file/inference/xdit/
  ├── mi3xx_pytorch_xdit_flux1_dev_single.json
  └── mi3xx_pytorch_xdit_wan22_14b_single.json

``mi3xx_pytorch_xdit_flux1_dev_single.json`` → ``mi3xx`` · ``pytorch_xdit`` · ``flux1_dev`` · ``single``.

With ``{precision}``:

.. code:: text

  input/config_file/inference/vllm/
  ├── mi300x_vllm_llama31-70b_fp8_single.json
  └── mi300x_vllm_llama31-70b_fp8_distributed.json

``mi300x_vllm_llama31-70b_fp8_single.json`` → ``mi300x`` · ``vllm`` · ``llama31-70b`` · ``fp8`` · ``single``.

**Suites**

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Suite
     - Config location
     - List / README
   * - ATOM
     - ``input/config_file/inference/atom/`` — ``mi{gpu}_atom_{model}_{precision}_{mode}.json`` + ``mi{gpu}_atom_{model}_{precision}_{mode}_threshold.json``
     - ``cvs config list inference/atom``

       :doc:`How to run </how-to/test-suites/inference/atom>` · :doc:`Config reference </reference/configuration-files/inference/atom>`
   * - vLLM
     - ``input/config_file/inference/vllm/`` — ``mi{gpu}_vllm_{model}_{precision}_{single|distributed}.json``
     - ``cvs config list inference/vllm``
   * - SGLang
     - ``input/config_file/inference/sglang/`` — ``mi{gpu}_sglang_{model}_{single|distributed|disaggregated}.json``
     - ``cvs config list inference/sglang``
   * - xDiT
     - ``input/config_file/inference/xdit/`` — ``mi{gpu}_pytorch_xdit_{model}_{single|distributed}.json``
     - ``cvs config list inference/xdit``
