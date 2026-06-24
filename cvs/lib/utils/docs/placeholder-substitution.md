# Placeholder Substitution — Three-Pass Walkthrough

`substitute_config` resolves placeholders in a variant config JSON in exactly three
passes. Each pass has a defined scope: which keys it reads from and which part of the
document it writes to. Understanding the order matters because a value produced by an
earlier pass becomes available as input to a later one.

Source: `cvs/lib/utils/config_loader.py`, function `substitute_config`.

---

## Worked example

The example uses `llama31_70b_fp8_config.json` (the `w1_llama31_70b_fp8kv` variant).
The cluster dict supplied at runtime is:

```json
{ "username": "jsmith" }
```

### Starting document (before any substitution)

Only the fields that contain placeholders or that receive substituted values are shown;
the rest of the config (model, sweep, params, roles) is omitted for brevity.

```json
{
  "threshold_json": "<changeme>",
  "paths": {
    "shared_fs": "/home/{user-id}",
    "models_dir": "{shared_fs}/models",
    "log_dir": "{shared_fs}/LOGS",
    "hf_token_file": "<changeme>"
  },
  "container": {
    "runtime": {
      "args": {
        "volumes": [
          "/home/{user-id}:/home/{user-id}",
          "{paths.models_dir}:/models"
        ]
      }
    }
  }
}
```

Note: `threshold_json` and `hf_token_file` are shown here with the sentinel value
`"<changeme>"` that appears in the actual file — both fields must be filled in with
real absolute paths before use. The substitution walkthrough below uses illustrative
values (`/shared/cvs/thresholds/llama31_70b_fp8_threshold.json` and
`/shared/cvs/tokens/{user-id}.token` respectively) to keep the example concrete; those
paths do not come from the real config file.

Note: `threshold_json` must always be a fully-resolved absolute path with no
placeholders. Using `{user-id}` or any other token there causes a
`FileNotFoundError` at the literal unresolved path (see the `threshold_json`
section below).

---

## Pass 1 — Cluster placeholders (`{user-id}`) everywhere

**Mapping built:** `_resolve_cluster_mapping(cluster_dict)` reads
`cluster_dict["username"]` (`"jsmith"`), producing `{"user-id": "jsmith"}`. When
`cluster_dict` has no `username` key or the value is falsy (empty string, `None`,
etc.), `getpass.getuser()` is used as the fallback.

**Scope:** `_walk_substitute` is called on the entire document. Every `{user-id}` token
in any string value is replaced.

**After Pass 1:**

```json
{
  "threshold_json": "/shared/cvs/thresholds/llama31_70b_fp8_threshold.json",
  "paths": {
    "shared_fs": "/home/jsmith",
    "models_dir": "{shared_fs}/models",
    "log_dir": "{shared_fs}/LOGS",
    "hf_token_file": "/shared/cvs/tokens/jsmith.token"
  },
  "container": {
    "runtime": {
      "args": {
        "volumes": [
          "/home/jsmith:/home/jsmith",
          "{paths.models_dir}:/models"
        ]
      }
    }
  }
}
```

Key observations:

- `threshold_json` contains no placeholders so it is unchanged by Pass 1. The
  threshold file was already read off disk before this pass ran (see the
  `threshold_json` section below).
- `paths.shared_fs` resolved to `/home/jsmith`, but `paths.models_dir` still shows
  `{shared_fs}/models`. That token is a self-reference within the paths block and is
  not in the cluster mapping. It is resolved in Pass 2.
- `volumes[1]` still shows `{paths.models_dir}:/models`. That cross-block reference
  is resolved in Pass 3.

---

## Pass 2 — Self-reference within `paths` (`{shared_fs}` inside `paths.*`)

**Mapping built:** the keys and string values of the `paths` block at their current
state after Pass 1:

```
shared_fs     -> "/home/jsmith"
models_dir    -> "{shared_fs}/models"
log_dir       -> "{shared_fs}/LOGS"
hf_token_file -> "/shared/cvs/tokens/jsmith.token"
```

**Scope:** `_walk_substitute` is called only on the `paths` block. The mapping is the
paths block's own key-value pairs (string values only). The substitution loop repeats
until the block stops changing, which handles chains: if `models_dir` referenced
`{log_dir}` which in turn referenced `{shared_fs}`, multiple iterations would unwind
the chain completely. The loop is capped at `len(paths_block) + 1` iterations. A cycle
in self-references (e.g. `a = "{b}"`, `b = "{a}"`) exits the cap silently, leaving
both tokens unresolved and verbatim — the same no-error-at-load-time behaviour as any
other unknown token.

**After Pass 2:**

```json
{
  "paths": {
    "shared_fs": "/home/jsmith",
    "models_dir": "/home/jsmith/models",
    "log_dir": "/home/jsmith/LOGS",
    "hf_token_file": "/shared/cvs/tokens/jsmith.token"
  }
}
```

Every `paths.*` value is now fully concrete. The rest of the document is unchanged at
this point — `volumes[1]` still holds `{paths.models_dir}:/models`.

---

## Pass 3 — Cross-block (`{paths.models_dir}` into the rest of the document)

**Mapping built:** `_flatten_paths({"paths": raw.get("paths", {})})` produces dotted keys for
every leaf in the paths block:

```
paths.shared_fs     -> "/home/jsmith"
paths.models_dir    -> "/home/jsmith/models"
paths.log_dir       -> "/home/jsmith/LOGS"
paths.hf_token_file -> "/shared/cvs/tokens/jsmith.token"
```

**Scope:** `_walk_substitute` is called on the entire document with this mapping. Any
`{paths.<key>}` token anywhere in the document is replaced with the fully-resolved
paths value.

Note: `{paths.*}` tokens inside the paths block itself are also resolved by Pass 3 —
Pass 2 only recognises bare keys (e.g. `{shared_fs}`), not dotted keys (e.g.
`{paths.shared_fs}`). A dotted self-reference in paths survives Pass 2 verbatim and is
then expanded by Pass 3.

**After Pass 3 (final document):**

```json
{
  "threshold_json": "/shared/cvs/thresholds/llama31_70b_fp8_threshold.json",
  "paths": {
    "shared_fs": "/home/jsmith",
    "models_dir": "/home/jsmith/models",
    "log_dir": "/home/jsmith/LOGS",
    "hf_token_file": "/shared/cvs/tokens/jsmith.token"
  },
  "container": {
    "runtime": {
      "args": {
        "volumes": [
          "/home/jsmith:/home/jsmith",
          "/home/jsmith/models:/models"
        ]
      }
    }
  }
}
```

`volumes[1]` is now `/home/jsmith/models:/models`. This is the value the container
orchestrator receives for the volume mount.

---

## Special case: `threshold_json` and substitution

`threshold_json` is a **literal absolute path**. The threshold file is read using the
raw (pre-substitution) string value before any pass runs:

```python
raw = json.loads(config_path.read_text())
threshold_path = Path(raw["threshold_json"])   # raw value, before any pass
thresholds = json.loads(threshold_path.read_text())
# ... passes run after this point
```

This has two consequences:

1. **`{user-id}` in `threshold_json` causes a FileNotFoundError.** If your
   `threshold_json` is `/shared/cvs/thresholds/{user-id}/threshold.json`, the file
   open is attempted at that literal string — the substitution that would resolve
   `{user-id}` has not run yet. Write `threshold_json` as a fully-resolved absolute
   path with no placeholders.

   Pass 1 does rewrite the `threshold_json` string value in the in-memory `raw` dict
   (so after substitution the string shows the resolved path), but because the
   threshold file was already opened before Pass 1 ran, that rewrite has no effect on
   which file was loaded.

2. **`{paths.*}` tokens in `threshold_json` are not substituted.** Pass 3 rewrites the
   entire document, so `{paths.log_dir}` in `threshold_json` would technically be
   replaced in the in-memory `raw` dict — but because the threshold file was already
   read before that pass, the token in the string value is irrelevant to which file
   was loaded. Keep `threshold_json` as a plain absolute path.

---

## What a typo'd placeholder looks like

If a placeholder token does not match any key in the current pass's mapping,
`_walk_substitute` leaves it verbatim — curly braces and all. No error is raised at
substitution time.

Example: suppose `volumes` contained a typo in the models dir token:

```json
"volumes": [
  "/home/{user-id}:/home/{user-id}",
  "{paths.modles_dir}:/models"
]
```

After all three passes (with `user-id` resolved and `paths.models_dir` in the mapping
but `paths.modles_dir` not), the output is:

```json
"volumes": [
  "/home/jsmith:/home/jsmith",
  "{paths.modles_dir}:/models"
]
```

The misspelled token `{paths.modles_dir}` survives all three passes unchanged. The
container launch then receives the literal string `{paths.modles_dir}:/models` as a
volume mount argument, which Docker rejects at runtime — not at config-load time.

The same applies to `paths` self-references. A typo'd `{sahred_fs}` in
`paths.models_dir` survives Pass 2 and propagates as a brace-wrapped string into the
paths block that feeds Pass 3.

**Diagnostic rule:** if a mount, log path, or model path looks wrong at runtime,
inspect the `raw` dict returned by `substitute_config`. Any string value containing
a literal `{...}` is a placeholder that failed to resolve, which always indicates a
token name mismatch or a missing mapping key.

---

## Summary table

| Pass | Mapping source | Document scope | Example tokens resolved |
|------|---------------|----------------|------------------------|
| 1 | `cluster_dict["username"]` (falls back to `getpass.getuser()` when the key is absent **or the value is falsy** — empty string, `None`, etc.) | Entire document | `{user-id}` |
| 2 | `paths` block key-value pairs (string values only), iterated until stable | `paths` block only | `{shared_fs}` (any bare key name in paths can be used as a self-reference token) |
| 3 | Flattened `paths` block as dotted keys | Entire document | `{paths.shared_fs}`, `{paths.models_dir}`, `{paths.log_dir}`, `{paths.hf_token_file}` |

Unknown tokens survive all passes verbatim. No error is raised at substitution time.
