# cvs/lib/conf_lib — config blocks and `load_config`

Suite schemas are built from `config_blocks.py`, live in `schemas/` (one module per suite, such
as `schemas/vllm.py`), and are loaded with `loader.load_config`. A suite with a schema here calls
`load_config` directly; it has no loader module of its own.
Substitution (`substitute_config`) and `BaseVariantConfig` stay in
`cvs/lib/utils/config_loader.py`, which older suite loaders import.

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

### `loader.py`

#### `load_config(model_cls, config_path, cluster) -> LoadedConfig(config, cluster, container)`

- `cluster`: path or dict, resolved with `resolve_cluster_config_placeholders` (a cluster
  `<changeme>` still exits). The cluster is not schema-validated.
- Runs `substitute_config`, attaches `thresholds` when the model has that field, and validates
  with `context={"placeholders": "reject"}`. There is no switch to allow placeholders.
- `ValidationError` becomes `ConfigError` (a `ValueError`) with one `loc: msg` line per error,
  built from `errors(include_input=False)`; never put input values in a message.
- `container`: cluster `container` with `config.container.model_dump()` deep-merged on top
  (dicts by key, lists and scalars replaced, config wins); `None` when the model has none.
- Must not import `cvs.core` (the orchestrator factory imports this package).
