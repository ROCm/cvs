# cvs/lib/conf_lib — config blocks and `ConfigError`

Opt-in suite schemas, such as vLLM's `VllmConfig`, and their loader live in
`cvs/config_validation/`, which raises `loader.ConfigError`.
Substitution (`substitute_placeholders`, and `substitute_config` with its threshold file),
`attach_threshold_file` and `BaseVariantConfig` stay in `cvs/lib/utils/config_loader.py`,
which older suite loaders import.

`__init__.py` must stay empty: `config_loader.py` imports `config_blocks`, so a package-level
re-export is an import cycle.

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

`ConfigError(config_path, problems)`, a `ValueError` that `cvs.config_validation` raises. The
message names the file and has one `location: problem` line per entry in `problems`. Build the
lines from `errors(include_input=False)`; never put input values in a message.
