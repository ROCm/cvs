# JAX MaxText Training — Library

Library code behind the `jaxmaxtext_single` / `jaxmaxtext_distributed` suites.
The pytest suites (`cvs/tests/training/jaxmaxtext/`) are thin wrappers; this
package holds the orchestration-driven training job, the config schema/loader,
and the log-parsing/metric helpers.

- Suites & lifecycle: `cvs/tests/training/jaxmaxtext/README.md`
- Config & threshold reference: `cvs/input/config_file/training/jaxmaxtext/README.md`

## Layout

| Path | Purpose |
|---|---|
| `jaxmaxtext_training_lib.py` | `MaxTextTrainingJob` — one training run driven entirely through an injected `ContainerOrchestrator` (no direct SSH/Docker). |
| `utils/training_config_loader.py` | Pydantic v2 config schema + loader (`load_training_variant`, `validate_thresholds_cover_training`). |
| `utils/maxtext_parsing.py` | Pure log parsing: per-step metrics, aggregates, checkpoint I/O timings, convergence, loss-curve sampling. |
| `utils/loss_curve.py` | Renders the per-sweep training-loss PNG and the decreasing-trend check. |
| `unittests/` | `unittest` tests for the above, using a mock orchestrator (no live cluster). |

## `jaxmaxtext_training_lib.py`

`MaxTextTrainingJob(orch, variant, hf_token, sweep=None)` composes one MaxText
run. All container/SSH plumbing belongs to `orch`; the class only builds
commands, launches, polls, and parses. Key stages (called by the suite in
order):

| Method | What it does |
|---|---|
| `setup_tokenizer()` | Downloads the HF tokenizer — **skipped** when `needs_hf_tokenizer()` is false (all enabled runs are `dataset_type=synthetic`). |
| `setup_training_env()` | Writes the env script (`env_vars`, `XLA_FLAGS`, and the `nccl.*` exports incl. `NCCL_IB_GID_INDEX`) and the MaxText YAML into the container scratch dir. |
| `build_training_cmd()` | Resolves the train entrypoint (first existing of `train_script_paths`) and per-rank JAX distributed env. |
| `start_training()` | Clears each node's stale `training.log`, then launches the per-node launcher with `nohup` (parallel `exec_cmd_list`, one rank per host). |
| `poll_for_completion()` | Streams only NEW node-0 log lines each poll, scans **every** node's new chunk for `error_patterns`/NaN (raises + logs the offending chunk), and detects the completion marker. |
| `parse_results()` | Reads node-0's log once and computes the `training.*` metrics. |
| `scan_dmesg_for_errors()` | Best-effort host `dmesg` scan over the run window (gated by `training.verify_dmesg`). |

Module helpers: `needs_hf_tokenizer(training)` (synthetic → no tokenizer),
`_get_scratch_dir()` (host-user scratch from `paths.temp_dir`, fallback
`/tmp/cvs/jaxmaxtext`), and the error signatures (`_NAN_INF_RE`,
`_TRAINING_ERR_PATTERNS`) used when a config omits `error_patterns`.

Scratch dir note: the scratch base comes from `paths.temp_dir`, **not** the
in-container `id -un` — docker jobs run as root, which would collide on
`/tmp/root` across users on a shared node.

## `utils/training_config_loader.py`

Pydantic models for the config. `TrainingConfig` (extra keys allowed, so
`_*_comment` fields pass through) with nested blocks: `Tokenizer`, `NcclConfig`,
`JaxDistributed`, `RdmaLib`, `ScalingBaseline`, `Convergence`, `LossCurve`,
`SmokeTest`, `CheckpointResume`, `Sweep`. `NcclConfig` hard-exits when a
cluster-specific field (`ib_hca[_list]`, `socket_ifname`, `gloo_socket_ifname`,
`ib_gid_index`) is left as `<changeme>`.

Entry points:

- `load_training_variant(config_file, cluster_dict)` — reads config + sibling
  threshold file, resolves placeholders, returns the typed variant.
- `validate_thresholds_cover_training(...)` — fails fast (or warns when
  `enforce_thresholds=false`) on a sweep-name/threshold-key mismatch.

## `utils/maxtext_parsing.py`

Pure functions (no orchestrator/IO) so they are trivially unit-tested:

- `extract_step_metrics()` / `parse_training_log()` — per-step and aggregate
  `training.*` metrics from the log.
- `TRAINING_METRICS` / `GATED_METRICS` — the metric list the suite parametrizes
  and gates; `CHECKPOINT_METRICS` is kept separate (checkpoint I/O only).
- `extract_checkpoint_timings()` — save/load seconds from orbax logs.
- `compute_convergence()`, `sample_loss_curve()`, `evaluate_loss_decreasing()`
  — convergence and loss-curve helpers.

## Unit tests

`unittests/` uses `unittest` with a `MagicMock` orchestrator — no SSH, no
container, no real sleeps. Run them via the repo suite:

```bash
make ut
# or a single module:
.test_venv/bin/python -m unittest cvs.lib.training.jaxmaxtext.unittests.test_jaxmaxtext_training_lib
```

Every library change here must keep these green (`make ut`) and pass
`make fmt-check` / `make lint`.
