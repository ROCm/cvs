.. meta::
  :description: PyTorch Training (LLM) configuration and threshold reference
  :keywords: CVS, PyTorch, FSDP2, LLM, config schema

*****************************
PyTorch Training (LLM)
*****************************

The PyTorch Training suites use the same configuration split as the other
training suites:

- the cluster file contains SSH access and the selected node(s)
- ``*_config.json`` contains the container and training workload
- the sibling ``*_threshold.json`` contains per-sweep metric gates

Files live in ``cvs/input/config_file/training/pytorch_training/`` and are
named ``<gpu>_pytorch_training_<model>-<precision>_<mode>_<profile>_config.json``
with a matching ``_threshold.json``. Every shipped threshold starts as
``"kind": "info"`` (record-only) until repeated measured runs set real gates.

Top-level fields
================

- ``framework``: must be ``pytorch_training``.
- ``gpu_name``: ``MI300X``, ``MI325X``, or ``MI355X``. MXFP8 and MXFP4 sweeps
  are rejected unless it is ``MI355X``.
- ``enforce_thresholds`` and ``threshold_json``: as in the other suites.
- ``paths``: ``shared_fs``, ``models_dir``, ``log_dir``, and
  ``hf_token_file``. Results go to
  ``{log_dir}/pytorch_training/<sweep-label>/<run-id>/``.
- ``model``: the HuggingFace id of the primary model; ``remote`` stays ``0``.
- ``container``: lifecycle, image, and runtime mounts. The shipped configs
  mount the models directory at ``/models`` (read-only), the corpus and token
  cache at ``/data``, and node-local scratch at ``/scratch``.

Training block
==============

- ``workload``: scorecard id (``W1``-``W14``), recorded in and checked against
  the result artifact.
- ``distributed``: ``false`` for ``pytorch_training_single``, ``true`` for
  ``pytorch_training_distributed``. ``master_port`` is the rendezvous port on
  the first cluster host for multi-node runs.
- ``enabled`` and ``allow_long_run``: safety switches. ``soak`` needs both
  ``allow_long_run`` and ``max_duration_seconds``.
- ``phase`` and ``run_mode``: ``smoke``/``smoke``, ``performance``/``perf``,
  ``accuracy``/``train_5k`` or ``finetune``, and ``soak``/``soak``.
  Accuracy modes need ``eval_enabled`` and ``warmup_steps: 0``.
- ``gpus_per_node``, ``warmup_steps``, ``steps``, ``seed``.
- ``eval_enabled``, ``eval_every_steps`` (0 evaluates once, after training),
  ``eval_batches``: each evaluation forward reads one held-out window per
  rank.
- ``milestone_steps``: steps whose loss is reported as ``loss_step_<N>``.
- ``learning_rate``, ``min_learning_rate_ratio``, ``lr_schedule``
  (``constant`` or ``cosine``), ``lr_warmup_steps``, ``weight_decay``,
  ``adam_beta1``, ``adam_beta2``, ``grad_clip_norm``: AdamW settings.
- ``fsdp_strategy``: ``full_shard`` or ``hybrid_shard`` (shard within a node,
  replicate across nodes).
- ``timeout_s``, ``startup_timeout_s``, ``startup_retries``,
  ``setup_timeout_s``: launch limits. A launch that fails before warmup ends
  is relaunched up to ``startup_retries`` times.
- ``peak_tflops_per_gpu``: dense peak per sweep precision, for MFU.
- ``checkpoint``: ``directory`` (container path on node-local disk), ``keep``
  (default ``false`` deletes it after each sweep), and ``resume_steps``.
- ``data``: ``source`` (WikiText parquet directory), ``text_column``,
  ``train_split``, ``eval_split`` (default ``test``), ``cache_dir``, and
  ``append_eos``. Tokens are cached once per tokenizer under ``cache_dir``.
- ``loss_curve`` and ``convergence``: as in the other training suites.
- ``scaling_baseline``: ``tokens_per_sec_total`` and ``num_nodes`` of a
  measured reference run, plus an optional ``eval_loss`` for scale parity.
  A sweep's own ``scaling_baseline`` overrides this one.
- ``codecarbon``: energy tracking through amd-smi; ``required`` fails the run
  when no energy is reported. The tracker runs in a separate process started
  by rank 0, because its amd-smi polling thread crashed rank 0 when it ran
  inside the trainer.
- ``benchmarks``: lm-evaluation-harness tasks scored on the trained
  checkpoint (``mmlu_pro_pct``, ``bbh_pct``, ``math_lvl5_pct``, ``gpqa_pct``,
  ``musr_pct``, ``gsm8k_pct``, ``hellaswag_pct``), with ``limit``,
  ``batch_size``, ``data_parallel``, and ``timeout_s``.
- ``env_vars`` and ``error_patterns``: exported variables and log signatures
  that fail a run early.
- ``sweeps`` and ``enabled_sweep_list``.

Sweep fields
============

- ``name`` and ``label``: unique ids; the label names the result directory.
- ``model`` and ``model_path``: short architecture id and the container path
  of the HuggingFace snapshot.
- ``init``: ``random`` (default) or ``pretrained``.
- ``precision``: ``BF16``, ``FP8`` (torchao tensorwise float8), ``MXFP8``, or
  ``MXFP4``. ``lm_head`` and MoE routers stay in BF16.
- ``seq_len`` and ``micro_batch_size``.
- ``global_batch_size`` or ``gradient_accumulation_steps``: exactly one. A
  global batch fixes the tokens per step on any topology and must divide by
  ``micro_batch_size x world_size``.
- ``activation_checkpointing_every``: recompute every Nth decoder layer
  (1 = all, 0 = none).
- ``master_weights``: ``fp32`` (default) or ``bf16``.
- ``attn_implementation``: ``flash_attention_2`` (default), ``flex_attention``,
  ``sdpa``, or ``eager``.
- ``compile``: ``torch.compile`` each decoder layer (default ``true``).
- ``loss_chunk_tokens``: compute the loss over this many tokens at a time
  (0 = all at once).
- ``data_mode``: ``tokens`` (default) or ``synthetic``.
- ``training_flops_per_token``: overrides the FLOPs per token derived from
  the model config.
- ``scaling_baseline``: per-sweep reference run.

See :doc:`/how-to/test-suites/training/pytorch_training` for how to run the
suites.
