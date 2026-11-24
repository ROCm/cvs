.. meta::
  :description: Run the PyTorch Training (LLM) CVS test suites
  :keywords: CVS, PyTorch, FSDP2, LLM, Llama, DeepSeek, Qwen, GPT-OSS, training

*****************************
PyTorch Training (LLM)
*****************************

``pytorch_training_single`` and ``pytorch_training_distributed`` pre-train and
fine-tune HuggingFace causal language models with PyTorch FSDP2 in a ROCm
container, then gate the run on performance and correctness metrics with a
PASS/FAIL HTML report.

Overview
========

The harness builds the model from its HuggingFace ``config.json`` on the meta
device, shards every decoder layer with ``fully_shard``, and trains on
tokenized WikiText-103. Every rank therefore holds only its shard, and a 70B
model starts in seconds without reading weights. ``init: pretrained`` loads
the safetensors weights instead, for fine-tuning and benchmark profiles.

The shipped configs cover these scorecard workloads:

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - Id
     - Workload
     - Recipe
   * - W1
     - DeepSeek-V2-Lite (MoE), BF16
     - SL4096, MBS10, GBS80
   * - W2
     - Llama-3.1-8B, BF16
     - SL8192, MBS4, GBS128
   * - W3
     - Llama-3.1-8B, FP8
     - SL8192, MBS4, GBS128
   * - W6
     - Llama-3.3-70B, FP8
     - SL8192, MBS1, GBS16
   * - W7
     - Llama-3.3-70B, BF16
     - SL8192, MBS3, GBS96
   * - W11
     - Qwen2.5-72B, BF16
     - SL2048, MBS16, GBS256
   * - W12
     - Qwen3-30B-A3B (MoE), BF16
     - SL4096, MBS8, GBS512
   * - W13/W14
     - GPT-OSS-20B (MoE), BF16 / FP8
     - SL4096, MBS8, GBS640

W4 and W5 (Llama-3.1-8B MXFP8 and MXFP4) need MI355X; the loader rejects MX
precisions on any other GPU. W8-W10 (Llama-3.1-405B and Mixtral-8x22B) are not
shipped yet.

Quick start
===========

.. code-block:: bash

   cvs run pytorch_training_single \
     --cluster_file ./mi325x_1n_cluster.json \
     --config_file cvs/input/config_file/training/pytorch_training/mi325x_pytorch_training_llama-3.1-8b-bf16_single_perf_config.json \
     --html ./logs/pytorch_training_single.html --self-contained-html --capture=tee-sys

   cvs run pytorch_training_distributed \
     --cluster_file ./mi325x_2n_cluster.json \
     --config_file cvs/input/config_file/training/pytorch_training/mi325x_pytorch_training_llama-3.1-8b-bf16_distributed_smoke_config.json \
     --html ./logs/pytorch_training_distributed.html --self-contained-html --capture=tee-sys

The config's ``training.distributed`` flag must match the suite and the
cluster node count, or the run fails immediately.

Test lifecycle (report rows)
============================

.. list-table::
   :header-rows: 1
   :widths: 8 35 20 37

   * - Order
     - Test
     - Runs on
     - Purpose
   * - 1
     - ``test_launch_container``
     - once
     - Launch and verify the container on every node
   * - 2
     - ``test_verify_environment``
     - once
     - Verify torch, transformers, torchao, HIP, GPU count, and architecture
   * - 3
     - ``test_prepare_data[sweep]``
     - per sweep
     - Tokenize the corpus once per tokenizer into the token cache
   * - 4
     - ``test_training[sweep]``
     - per sweep
     - Train, run the checkpoint round trip, evaluate, and parse the artifact
   * - 5
     - ``test_ga_overhead_pairs``
     - once
     - Pair each GA>1 sweep with the GA=1 sweep at the same micro batch
   * - 6
     - ``test_metric[sweep-metric]``
     - per sweep x metric
     - Threshold PASS/FAIL per metric
   * - 7
     - ``test_loss_curve[sweep]``
     - per sweep
     - Render the loss PNG; gate on a downward trend in accuracy profiles
   * - 8
     - ``test_convergence[sweep]``
     - per sweep
     - Step and time to the configured loss target
   * - 9
     - ``test_print_results_table``
     - once
     - Console table and metric results HTML
   * - 10
     - ``test_teardown``
     - once
     - Tear the container down

Profiles
========

``training.run_mode`` selects the profile:

- ``smoke``: 10 steps with evaluation and the checkpoint round trip.
- ``perf``: 30 measured steps after warmup; evaluation off.
- ``train_5k``: 5,000 steps from random weights, with eval loss, perplexity,
  and loss at steps 100, 500, 1,000, and 5,000.
- ``finetune``: a short run from pretrained weights with evaluation and
  optional lm-evaluation-harness benchmarks.
- ``soak``: runs to ``training.max_duration_seconds``; needs
  ``allow_long_run``.

Checkpoint round trip
=====================

Every sweep saves model and optimizer state with
``torch.distributed.checkpoint`` (DCP), trains ``checkpoint.resume_steps``
more steps, reloads, and replays the same steps.
``checkpoint_state_match`` is 1 only when an exact bit fingerprint of every
shard matches after the reload. ``checkpoint_loss_delta`` is the largest
difference between the original and the replayed losses.

Checkpoints go to node-local disk (``checkpoint.directory``, mounted from
``<changeme-local-scratch-host-path>``), because a 70B checkpoint with
optimizer state is about 840 GB. Each sweep's checkpoint is deleted when the
sweep ends, including on failure, unless ``checkpoint.keep`` is true.

Only the sharded tensors go through DCP. DCP deduplicates replicated entries,
such as Adam step counts and parameter groups, and spreads them over the
ranks' files, so a multi-node reload from node-local disks needed files that
only another node held. The replicated remainder is written on each node with
``torch.save``.

Mixture-of-experts models
=========================

HuggingFace MoE blocks skip any expert that received no tokens. That expert's
gradient is then missing on that rank only, FSDP2 reduce-scatters a different
gradient set on each rank, and the collectives deadlock: DeepSeek-V2-Lite hung
in its first step. A forward hook on every MoE block adds a zero-valued term
over all expert weights, so every rank produces a gradient for every expert.

``compile: true`` compiles each decoder layer, but leaves the expert block
eager. Its Python loop over experts has data-dependent token counts and
recompiled on every new routing.

Large vocabularies and attention
================================

``loss_chunk_tokens`` computes the loss over that many tokens at a time,
recomputing each chunk in backward. Qwen (152k vocabulary) and GPT-OSS (201k)
otherwise need tens of GB for fp32 logits at the recipe micro batch. The
chunked loss matches the unchunked one bit for bit in its gradient.

GPT-OSS uses ``attn_implementation: flex_attention``. The
``flash_attention_2`` path in transformers 4.57 drops its attention sinks
without an error.

Qwen2.5-72B on one node sets ``master_weights: bf16``. With fp32 master
weights, its parameters, gradients, and Adam state need 144 GB per GPU on 8
GPUs, which leaves no room for MBS16 activations.

Metrics
=======

Metrics surfaced (namespace ``training.*``): ``tokens_per_sec``,
``tokens_per_sec_per_gpu``, ``tflops_per_sec_per_gpu``, ``mfu_pct``,
``step_time_ms_{mean,p50,p95}``, ``peak_memory_{allocated,reserved,used}_mb``,
``checkpoint_{save,load,io}_seconds``, ``checkpoint_state_match``,
``checkpoint_loss_delta``, ``gradient_accumulation_overhead_pct``,
``scaling_efficiency_pct``, ``scale_parity_eval_loss_delta``,
``loss_{initial,final}``, ``loss_step_{100,500,1000,5000}``, ``eval_loss``,
``perplexity``, ``convergence_{step,time_seconds}``, ``energy_kwh``,
``tokens_per_kwh``, and the benchmark scores ``mmlu_pro_pct``, ``bbh_pct``,
``math_lvl5_pct``, ``gpqa_pct``, ``musr_pct``, ``gsm8k_pct``, and
``hellaswag_pct``.

FLOPs per token are 6 x the active non-embedding parameters plus the attention
term. MoE expert weights count at ``top_k / num_experts``, and sliding-window
layers count only their window. MFU divides by
``training.peak_tflops_per_gpu`` for the sweep's precision.

Scaling efficiency needs ``scaling_baseline.tokens_per_sec_total`` from a
measured run of the same sweep on ``num_nodes`` nodes; left at 0 it is
omitted.

Scale parity subtracts ``scaling_baseline.eval_loss`` from the run's eval
loss. The two runs evaluate the same held-out windows only when
``eval_batches`` times the total GPU count is equal, so the four-node
fine-tune profile sets ``eval_batches`` to 1 against the single-node
fine-tune's 4. Set it to 2 on two nodes.

Prerequisites
=============

See :ref:`training-common-prerequisites`. In addition:

- The image built by ``build_tools/pytorch_training/Dockerfile.primus-26.3``.
- A models directory (``<changeme-models-host-path>``) with one HuggingFace
  snapshot per model: ``config.json`` and the tokenizer, plus the safetensors
  weights for ``init: pretrained``.
- The WikiText-103 parquet files under ``<changeme-training-data-host-path>``.
  The prepare stage writes the token cache beside them.
- For train-5k, FineWeb-Edu shards as ``train-*.parquet`` and
  ``test-*.parquet`` under ``datasets/fineweb-edu/sample-10BT``. WikiText-103's
  120M tokens would be about 44 epochs of a 5,000-step W2 run; eight
  sample-10BT shards (17 GB) hold about 6B tokens, and their Llama-3.1 token
  cache needs about 24 GB more. On a multi-node run the first node tokenizes
  and the others reuse its cache.
- Node-local scratch space for checkpoints: 12 bytes per parameter.

See :doc:`/reference/configuration-files/training/pytorch_training` for the
config reference.

Measured on MI325X (8 GPUs per node)
====================================

Smoke profiles, one node:

.. list-table::
   :header-rows: 1
   :widths: 30 20 15 35

   * - Workload
     - Tokens/s/GPU
     - MFU
     - Notes
   * - W1 DeepSeek-V2-Lite BF16
     - 7,775
     - 10.8%
     - Eager HF expert loop
   * - W2 Llama-3.1-8B BF16
     - 9,799
     - 43.4%
     - Checkpointing every 4th layer
   * - W3 Llama-3.1-8B FP8
     - 12,440
     - 27.6%
     - MFU against the FP8 peak
   * - W6 Llama-3.3-70B FP8
     - 542
     - 10.0%
     - MBS1
   * - W7 Llama-3.3-70B BF16
     - 1,088
     - 40.1%
     -
   * - W11 Qwen2.5-72B BF16
     - 1,158
     - 39.4%
     - bf16 master weights
   * - W12 Qwen3-30B-A3B BF16
     - 3,701
     - 7.9%
     - Eager HF expert loop
   * - W13 GPT-OSS-20B BF16
     - 7,385
     - 13.6%
     -
   * - W14 GPT-OSS-20B FP8
     - 7,335
     - 6.8%
     - Expert banks are not ``Linear`` layers, so only attention runs in FP8

Every smoke run restored the checkpoint with an exact state match and a loss
delta of 0. W2 perf measured 9,889 tokens/s/GPU at GBS128 (GA4) and 9,827 at
GA1, so accumulation costs nothing measurable. On two nodes the W2 smoke ran
at 9,735 tokens/s/GPU, 99% of the one-node rate, and W2 perf on four nodes
at 9,785 tokens/s/GPU, 98.9% scaling efficiency against the one-node perf run.
