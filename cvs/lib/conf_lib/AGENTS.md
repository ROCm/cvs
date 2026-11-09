# cvs/lib/conf_lib — config blocks and `load_config`

Suite schemas are built from `config_blocks.py`, live in `schemas/` (one module per suite, such
as `schemas/vllm.py`), and are loaded with `loader.load_config`. A suite with a schema here calls
`load_config` directly; it has no loader module of its own.
Substitution (`substitute_placeholders`, and `substitute_config` with its threshold file),
`attach_threshold_file` and `BaseVariantConfig` stay in `cvs/lib/utils/config_loader.py`,
which older suite loaders import.

`__init__.py` must stay empty: `config_loader.py` imports `config_blocks`, and `loader.py`
imports `config_loader`, so a package-level re-export of `load_config` is an import cycle.

## Files

### `config_blocks.py`

The schema blocks every suite composes. `config_loader.py` re-exports `_Forbid`, `_Allow`,
`Paths`, `ModelSpec`, `RuntimeSpec` and `ContainerSpec`, so existing imports keep working.

- `_Forbid` / `_Allow` drop every `_`-prefixed key at the model's own level (a `mode="before"`
  validator). Free-form maps (`env`, `RuntimeSpec.args`, CLI option values) are not walked.
- `ConfigStr` rejects `<changeme>` and `<changeme-...>` (case-insensitive) only when validated
  with `context={"placeholders": "reject"}`. With no context it is a plain `str`, so suites that
  validate a template directly keep working. Applied to `Paths` fields and `ContainerSpec`
  `name`, `image` and `env` values.
- `RuntimeArgs` (`_Allow`) types the `runtime.args` keys the Docker runtime reads; its dump
  omits unset keys. Not wired into `RuntimeSpec`: suites opt in by subclassing it.
- `ThresholdSpec` is a union on `kind`; `ThresholdFile` is the cells plus free-form `accuracy`.
  The verdict engine does not read them yet.

### `schemas/inference.py`

The base inference suite schemas extend. No suite uses it yet.

- `InferenceConfig`: what every inference suite shares (vLLM, SGLang, ATOM, xDiT):
  `enforce_thresholds`, `threshold_json`, `thresholds` (attached with `attach_threshold_file`; free-form
  shape), `paths`, `container` and `server_params.model`. `container.env` is the only env block:
  `runtime.args` is strict and rejects `env`, and a Hugging Face token name in `container.env` raises
  `ContainerEnvTokenError`, not a `ValueError`, so its value never reaches a `ValidationError`.
- `ServingConfig(InferenceConfig)`: what the LLM serving engines share (vLLM, SGLang, ATOM). Adds
  `server_params` `tp`/`pp`/`dp`/`ep` (`ep` above 1 is rejected while reserved), `benchmark_params`,
  `sweeps`, `runs` (nonempty, unique, each a `sweeps` key) and `accuracy`. Cell keys are
  `ISL=,OSL=,TP=,PP=[,DP=][,EP=],CONC=` and must match `server_params`; `DP=1` is rejected, so a
  cell has one spelling. A `sweeps` value is merged onto `benchmark_params` (`engine_args` merged by
  key, `a-b` and `a_b` as one option) and validated as it; `run_benchmark_params(key)` returns the
  result. Threshold cells match sweep keys exactly and every cell's specs must be `ThresholdSpec`s.
  `_comment*`/`_example*` metric keys are dropped from the stored cells; any other `_` key fails.
  With enforcement on, every run needs a cell.
- `server_params` and `benchmark_params` are strict. What CVS only passes to the engine goes in
  `engine_args`: lowercase snake or kebab names, two spellings of one option rejected, `false` and
  nested lists rejected, `<changeme>` rejected, `_` keys dropped. Ports, `max_model_len` and the poll
  fields default to `None`; the engine subclass sets its own defaults. Rendering `engine_args`
  stays in each engine.
- An engine schema subclasses `ServingConfig` (or `InferenceConfig`) and its param blocks, adds only
  fields CVS reads for that engine, and extends two class hooks: `reserved_engine_args()` on a param
  block and `non_cell_threshold_keys()` on the config. Every field of a param block, a subclass's
  included, is already reserved in its `engine_args`; the hook adds the engine's spellings of flags
  CVS renders, with `-` written as `_`. Its metric-name rules go in its own `thresholds` validator.

### `loader.py`

#### `load_config(model_cls, config_path, cluster) -> LoadedConfig(config, cluster, container)`

- `cluster`: path or dict, resolved with `resolve_cluster_config_placeholders` (a cluster
  `<changeme>` still exits). The cluster is not schema-validated.
- Runs `substitute_placeholders(..., all_placeholders=True)`, so pass 1 expands the
  `utils_lib.config_placeholder_map` set (`{user-id}`, `{user}`, `{home}`, `{home-mount-dir}`,
  `{node-dir-name}`, `{run_dir}`) that `resolve_test_config_placeholders` uses. Do not call that
  resolver here: it exits at the first `<changeme>`.
- Validates with `context={"placeholders": "reject", "config_path": Path(config_path)}`. There is
  no switch to allow placeholders.
- Reads no threshold file. A schema that uses one attaches it in a `mode="before"` validator with
  `attach_threshold_file(data, info.context)`, as `schemas/vllm.py` does. Only a named
  `threshold_json` is read, with no sibling lookup. A config that names none gets `{}` and
  `enforce_thresholds` false, so it runs record-only; with a file, the schema's own coverage
  rules apply.
- `ValidationError` becomes `ConfigError` (a `ValueError`) with one `loc: msg` line per error,
  built from `errors(include_input=False)`; never put input values in a message.
- `container`: cluster `container` with `config.container.model_dump()` deep-merged on top
  (dicts by key, lists and scalars replaced, config wins); `None` when the model has none.
- Must not import `cvs.core` (the orchestrator factory imports this package).
