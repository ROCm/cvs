# One config API for the training and inference suites

**Scope:** vllm, atom, sglang, megatron, torchtitan, jaxmaxtext.
Health / rccl / ibperf / platform / mori come later.

---

## The problem

Six suites, six config loaders, six sets of conventions.

```
1,778 lines  of loader code across the 6 suites
1,441 lines  of conftest

     6 fixtures — _deep_merge, cluster_dict, hf_token, orch,
                  variant_config, pytest_collection_modifyitems —
                  each defined separately in all 6 conftests
```

Each suite invented its own answer to the same questions: where do paths live, how is
a container declared, what is a sweep, what happens when the config is wrong. An
engineer who knows vLLM cold still cannot read an atom config without reading atom's
loader first.

---

## The idea

```
   BLOCK LIBRARY  (framework owns)
   ┌────────┬───────────┬────────────┬───────────────┬──────────────┐
   │ Paths  │ ModelSpec │ Container  │ ThresholdSpec │  Sweep × 3   │
   │ Model- │           │ Spec       │  (7 kinds)    │ Combo/Matrix/│
   │ Paths  │           │ + Runtime  │               │ Named        │
   └───┬────┴─────┬─────┴──────┬─────┴───────┬───────┴──────┬───────┘
       └──────────┴────────────┴─────────────┴──────────────┘
                               │  compose
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       vllm/schema.py    atom/schema.py   megatron/schema.py   (suite owners)
             │                 │                 │
             └─────────────────┴─────────────────┘
                               │  register("vllm", model=…, rules=[…])
                               ▼
                 load_config(config, suite, cluster)
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
             typed config              Problem[]
             → tests run               → no tests run,
                                         every error at once,
                                         each with a fix
```

**Blocks** — typed, reusable config fragments. A suite composes the ones it needs.
**Registry** — maps the suite name from `cvs run` to its schema. One load path.
**Rules** — the checks no single block can do: sweep-vs-thresholds, config-vs-cluster.

---

## What a suite must implement

Today this contract is implicit, and every suite guessed differently. Making it
explicit is most of the value.

| | Required | Provided by framework |
|---|---|---|
| **Always** | a schema composing blocks | loading, substitution, parse, error format |
| **If it sweeps** | `sweep.entries()` + `cell_key(entry)` | `cells()`, coverage rule, parametrize IDs, report lookup |
| **If it containerizes** | a `container` block | orchestrator handoff |
| **If it gates** | a `thresholds` block | evaluation, record-only rendering |
| **Optional** | `rules` — cross-block and cluster checks | batching, reporting, the `fix` contract |

Two methods. Everything else a suite writes today — `load_variant`,
`validate_sweep_selector`, `validate_thresholds_cover_sweep`, `_check_no_changeme`,
`expected_cells`, `orchestrator_container_from_variant` — moves into the framework.

See **Appendix B** for exactly what each suite implements today and what survives.

---

## The sweep problem, solved once

### What "cell key" means

Every sweep produces cells, and each cell's key is a string that must **exactly match
a key in the threshold file**. Nothing checks that the two agree in format — only that
they agree in content.

Today that string is built three different ways, with three different signatures:

```python
# vllm — formatted from params; PP segment appears only when pp > 1
def cell_key(self, isl, osl, concurrency):
    base = f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},"
    if int(self.params.pipeline_parallel_size) > 1:
        base += f"PP={self.params.pipeline_parallel_size},"
    return base + f"CONC={concurrency}"

# atom — same idea, but the dimension set branches on the driver
def cell_key(self, isl, osl, concurrency):
    key = f"ISL={isl},OSL={osl},TP={p.tensor_parallelism}"
    if p.driver == "atom":
        if nnodes > 1:
            key += f",DP={nnodes},NNODES={nnodes}"
    elif p.driver in ATOM_PP_DRIVERS:
        if pp > 1 or nnodes > 1:
            key += f",PP={p.pipeline_parallel_size}"
        if nnodes > 1:
            key += f",NNODES={p.nnodes}"
    return f"{key},CONC={concurrency}"

# megatron / torchtitan — different signature; the key is a declared combo id
def cell_key(self, combo_key: str) -> str: ...

# sglang — has both of the above, plus a second key function taking nothing
def cell_key(self, isl, osl, concurrency): ...
def perf_cell_key(self): ...

# jaxmaxtext — no method at all; the sweep entry's `name` IS the key
#   "NNODES=2,STEPS=30,PRECISION=BF16,BATCH=3,GBS=48,SEQLEN=8192"
```

**Four distinct signatures** for one conceptual operation — `(isl, osl, concurrency)`,
`(combo_key)`, `()`, and no method at all.

No shared code can call that. Which is why every suite reimplements coverage
checking, parametrize ID generation, and report cell lookup on top of it.

### The fix: split enumeration from formatting

```python
# On the sweep block — shape-specific, knows nothing about params
def entries(self) -> list[SweepEntry]:
    """Enumerate the runs this sweep declares, in order."""

# On the config — one line, identical in every suite
def cells(self) -> list[str]:
    return [self.cell_key(e) for e in self.sweep.entries()]
```

`entries()` is where the three shapes differ. `cells()` is where they stop differing.

`cell_key` stays suite-owned — atom's driver branching is real and cannot be
genericized — but it now has **one signature** across all six, so framework code can
call it.

### What an entry carries

The three shapes share no dimensions at all — `isl/osl/concurrency` vs
`gbs/mbs/precision` vs arbitrary maxtext overrides — so `SweepEntry` cannot have
typed dimension fields. It has three:

```python
@dataclass(frozen=True)
class SweepEntry:
    ref:     str              # stable id — what the config called this run
    dims:    Dict[str, Any]   # the parameters that vary across the sweep
    payload: Dict[str, Any]   # extras the job needs (SLOs, overrides, baselines)
```

`dims` is what `cell_key` formats. `payload` is what the job consumes. Nothing else.

**vllm** — `sequence_combinations[{name, isl, osl, goodput_slo}]` + `runs[{combo, concurrency}]`:

```python
def entries(self):
    by_name = {c.name: c for c in self.sequence_combinations}
    return [
        SweepEntry(
            ref=f"{r.combo}@{r.concurrency}",
            dims={"isl": by_name[r.combo].isl, "osl": by_name[r.combo].osl,
                  "concurrency": r.concurrency},
            payload={"goodput_slo": by_name[r.combo].goodput_slo},
        )
        for r in self.runs
    ]
# → SweepEntry(ref="w1_isl=1000_osl=1000@16",
#              dims={"isl": 1000, "osl": 1000, "concurrency": 16},
#              payload={"goodput_slo": {"ttft_ms": 1e9, ...}})
```

**megatron / torchtitan** — `combinations{id: {...}}` + `runs[id]`:

```python
def entries(self):
    return [SweepEntry(ref=r, dims=self.combinations[r].dims(),
                       payload=self.combinations[r].payload())
            for r in self.runs]
# → SweepEntry(ref="llama3_3_70b-mi325-bs64-mbs1-fp8",
#              dims={"global_batch_size": 64, "micro_batch_size": 1, "precision": "FP8"},
#              payload={"name": "llama3_3_70b_mbs1_gbs64_FP8",
#                       "result_dict": {"throughput_per_gpu": "100", ...}})
```

**jaxmaxtext** — `training.sweeps[]` + `training.enabled_sweep_list[]`:

```python
def entries(self):
    chosen = set(self.enabled_sweep_list or [s.name for s in self.sweeps])
    return [SweepEntry(ref=s.name, dims=s.dims(),
                       payload={"maxtext_overrides": s.maxtext_overrides})
            for s in self.sweeps if s.name in chosen]
# → SweepEntry(ref="NNODES=2,STEPS=30,PRECISION=BF16,BATCH=3,GBS=48,SEQLEN=8192",
#              dims={"nnodes": 2, "steps": 30, "precision": "BF16",
#                    "batch": 3, "gbs": 48, "seqlen": 8192},
#              payload={"maxtext_overrides": {"quantization": "", ...}})
```

`cell_key` then reads `dims` and nothing else:

```python
def cell_key(self, e):                                    # vllm
    d, p = e.dims, self.params
    k = f"ISL={d['isl']},OSL={d['osl']},TP={p.tensor_parallelism},"
    if p.pipeline_parallel_size > 1:
        k += f"PP={p.pipeline_parallel_size},"
    return k + f"CONC={d['concurrency']}"

def cell_key(self, e):                                    # jaxmaxtext
    d = e.dims
    return (f"NNODES={d['nnodes']},STEPS={d['steps']},PRECISION={d['precision']},"
            f"BATCH={d['batch']},GBS={d['gbs']},SEQLEN={d['seqlen']}")
```

### The drift this closes

jaxmaxtext is the case that shows why deriving the key matters. Today its dimensions
exist *only* inside the name string, which is also the threshold key. The config's own
comment states `GBS = per_device_batch_size * total GPUs`, and the arithmetic holds in
all three shipped configs:

```
NNODES=2, BATCH=3, GBS=48   →  3 × 2 × 8 = 48   ✓
NNODES=1, BATCH=5, GBS=40   →  5 × 1 × 8 = 40   ✓
NNODES=2, BATCH=2, GBS=32   →  2 × 2 × 8 = 32   ✓
```

So the name is **derivable but hand-maintained**. Change `per_device_batch_size` to 4
and forget to rename, and the key still resolves against the threshold file — because
the name *is* the key — while advertising a GBS that never ran. Nothing catches it.

Declaring dims and deriving the name closes that. The migration does not require
rewriting configs first: keep the literal `name`, and have load assert derived ==
declared, reporting a `Problem` on mismatch. That is the drift check today and the
deletion path tomorrow.

### What that buys

Everything downstream stops caring which sweep shape it got:

| Consumer | Today | After |
|---|---|---|
| threshold coverage | reimplemented per suite | one rule over `cells()` |
| parametrize IDs | 4 variants, one of which can drift from the key | `cells()` |
| report cell lookup | per-suite key rebuild | `cells()` |

And a fourth sweep shape costs one `entries()` method — nothing else changes.

### Three shapes, one contract

| Block | Config shape | Key origin | Suites |
|---|---|---|---|
| `ComboSweep` | `sweep.sequence_combinations[]` + `sweep.runs[{combo, concurrency}]` | formatted from dims + params | vllm, atom |
| `MatrixSweep` | `sweep.combinations{id: …}` + `sweep.runs[id]` | the combination's declared `name` | megatron, torchtitan |
| `NamedSweep` | `training.sweeps[]` + `training.enabled_sweep_list[]` | the entry's own `name` | jaxmaxtext |

Note jax's sweep is nested under `training`, not at the top level like the other two —
one more thing a newcomer has to discover by reading a loader.

sglang has no sweep block — it derives cells *from* the threshold file, inverting the
direction of truth. That's an open decision, not a solved case.

---

## What changes, per suite

| Suite | Today | After |
|---|---|---|
| **vllm** | standalone model — does not extend the shared base; private copies of `Paths`, `ModelSpec`, `ContainerConfig`; all 19 params typed `str`; drops unknown keys on load; 4 validators | composes shared blocks; `ComboSweep`; real types; unknown keys rejected |
| **atom** | extends the shared base, but imports `Sweep`, `validate_sweep_selector` and `validate_thresholds_cover_sweep` sideways out of **vllm's** module; 6 validators; own `expand_sweep`, `orchestrator_container_from_variant` | `ComboSweep` from the library; no cross-suite import; sweep expansion and orchestrator handoff move to framework |
| **sglang** | extends the base; two load paths (`_is_legacy_root` → legacy vs unified), but **all 5 shipped configs are legacy** — no `schema_version`, no `framework`, top-level `config`/`benchmark_params`; 526-line conftest; two key functions (`cell_key` + `perf_cell_key`); derives cells from thresholds; `orch` subsets cluster hosts | **the largest port by far** — 5 config files restructured, 138 loader lines deleted, and the only suite needing host-subsetting. Not a registration. |
| **megatron** | does **not** extend the base; **no `paths` block** — paths live in a `config` grab-bag with NCCL, topology and run flags; own `MatrixSweep`, `validate_sweep_selector`, `validate_thresholds_cover_sweep`, `_check_no_changeme`; unknown run refs warn and skip | extends base; 4 config files gain `paths`; `MatrixSweep` from library; those four functions become framework; unknown refs are errors |
| **torchtitan** | does **not** extend the base; same missing `paths` block; byte-identical copies of megatron's sweep block and all four functions — differs only in class names | same blocks and same framework code as megatron; 8 config files gain `paths`; the duplicate file shrinks to a schema |
| **jaxmaxtext** | extends the base; `NamedSweep` with name-as-key (already the right pattern); own `validate_thresholds_cover_training`; a bad enable-list entry silently widens the run | `NamedSweep` from library; coverage becomes the shared rule; enable-list mismatches are errors |

### The shared fixtures are not equally shared

"One `orch`, one `hf_token`" hides very different amounts of work. Measured by
comparing the six implementations directly:

| Fixture | Distinct impls | Reality |
|---|---|---|
| `_deep_merge` | 6 copies, **1 body** | AST-identical in all six; only the docstrings drifted. Pure delete. |
| `cluster_dict` | 2 | Five identical; sglang differs. |
| `orch` | 6 files, **3 behaviours** | vllm / megatron / torchtitan / jax bodies are identical (docstring and log text only). **atom differs solely to inject `roles.server.env`** — the one-env-block change erases that. sglang is genuinely different. |
| `hf_token` | 4 | Diverges *semantically*, not cosmetically — see below. |

So the dedup is cheaper than it looks for `_deep_merge` and `orch`, and more expensive
than it looks for `hf_token`, which is not a copy-paste problem but a behaviour
disagreement:

```python
# vllm / sglang — a pre-staged model needs no token
if not os.path.isfile(path):
    if variant_config.model.remote == 0:
        return ""
    pytest.skip(...)

# atom — skips unconditionally
if not os.path.isfile(path):
    pytest.skip(f"hf_token file missing: {path}")

# megatron / torchtitan — different config path entirely
path = variant_config.config['hf_token_file']
```

Identical cluster, identical pre-staged model: vllm **runs**, atom **skips**. Which
you get depends only on which suite you picked. Unifying the fixture forces that
question to be answered once — that is a decision, not a refactor.

### Config files *do* change — for three suites, not one

| Suite | Configs | What has to change |
|---|---|---|
| **sglang** | 5 | Full restructure: no `schema_version`, no `framework`, top-level `config` / `benchmark_params` |
| **megatron** | 4 | Gain a `paths` block; split the `config` grab-bag |
| **torchtitan** | 8 | Same |
| vllm, atom, jax | 45 | No change |

**17 of the 62 config files across these six suites** — not 5. (The 57 "envelope"
configs counted elsewhere in this doc exclude sglang's 5 entirely, because they carry
no `schema_version`. That omission is itself the finding.)

megatron and torchtitan have no `paths` block at all —
they keep paths inside a `config` catch-all that mixes four concerns:

```
paths     hf_token_file, log_dir, data_cache_dir, rocm_dir, scripts_dir, megatron_root
network   nccl_debug, nccl_socket_ifname, gloo_socket_ifname, nccl_ib_gid_index,
          nccl_ib_hca, nccl_ib_hca_list, nic_type
topology  nnodes, master_address
run       training_iterations, verify_network_errors
```

That is why megatron's `hf_token` reads `variant_config.config['hf_token_file']` while
every inference suite reads `variant_config.paths.hf_token_file`.

The payoff is concentrated in sglang. **138 of its 453 loader lines (30%)** exist only
to reverse-engineer a missing `paths` block — `_infer_models_dir` recovers `models_dir`
from `container_config.volume_dict`, `_infer_shared_fs` recovers `shared_fs` by
string-slicing `log_dir`, plus `_legacy_server_env`, `legacy_container_block_from_inference`,
`legacy_paths_from_inference`, `_is_legacy_root`, `_load_legacy_variant`. Give sglang a
real `paths` block and all 138 lines delete.

### What genuinely does not change

- **Test bodies.** `variant_config` still arrives as a typed object. Raw dict access in
  test and job code is already near zero — 0–2 sites per suite — so the port does not
  reach into job internals.
- **Orchestrator and report.** `container.model_dump()` keeps its current contract.
- **45 of the 62 config files** — every vllm, atom, and jaxmaxtext config.

### One capability the framework must decide on

sglang's `orch` rewrites the cluster dict to **scope the orchestrator to a subset of
hosts**, branching three ways (single / distributed / disaggregated) on which suite was
selected. No other suite subsets hosts. Either the framework supports host-subsetting
as a first-class concept or sglang keeps a suite-owned `orch`. That is the one item on
this list that is an architecture question rather than a port.

---

## What a suite author writes

```python
# cvs/lib/inference/atom/schema.py

class AtomConfig(BaseVariantConfig):          # or compose blocks directly
    framework: Literal["atom"]
    params: AtomParams                        # the only suite-specific part
    sweep: ComboSweep

    def cell_key(self, entry) -> str:         # suite-owned formatter
        ...

def check_scaling_baseline(cfg, cluster) -> list[Problem]:
    ...

register("atom", model=AtomConfig, rules=[check_scaling_baseline])
```

Then the conftest calls `load_config()` instead of `load_variant()`. That is the port.

---

## What the user sees when it's wrong

Today: one error, raised at the first problem, sometimes mid-run.

After:

```
config error: mi325x_vllm_deepseek-r1.json

  sweep.runs[0].combo
    "w_deepseek_1k1k" does not name any entry in sweep.sequence_combinations
    fix: use one of: w_deepseek-r1-0528_fp8_1k1k, ..._1k8k, ..._8k1k

  enforce_thresholds
    enforce_thresholds is true but no threshold file was resolved
    fix: set "enforce_thresholds": false to characterize first, then set
         "threshold_json" once you have numbers

2 problems; no tests were run.
```

Every problem at once, before anything launches, each naming the next action. The
`fix` line is mandatory on every rule — if you can't say what the user should do, the
check isn't ready.

---

## Threshold ergonomics

Two questions, answered separately:

| `enforce_thresholds` | threshold file | Behaviour |
|---|---|---|
| `false` | absent | record-only, **no error** — characterizing a new shape |
| `false` | present | report shows spec, bar, margin; no verdict |
| `true` | present | gates the run |
| `true` | absent | error at load |

Flip one boolean to move between gated and ungated, file left in place. Today a
threshold file is mandatory even when nothing is being gated.

---

## Why this is cheap

Most of it already exists in `cvs/lib/utils/config_loader.py` — *"framework-agnostic
config machinery shared by every CVS suite."*

- **6 of 6** already call its `substitute_config`
- **3 of 6** already extend its `BaseVariantConfig` (atom, sglang, jaxmaxtext)
- `Paths`, `ModelSpec`, `ContainerSpec`, `RuntimeSpec` already live there

The genuinely new code is the registry (~30 lines), the `Problem` type, the `entries()`
contract, and the threshold-optional change. The rest is moving blocks that exist into
a library, and deleting the copies.

---

## Phasing

The six do not cost the same, so they do not go in one bucket.

| | | Config files touched |
|---|---|---|
| **1. Framework** | blocks + registry + `Problem` + 3 shared rules. No suite changes. | 0 |
| **2. Two pilots** | **atom** — schema only, proves blocks + registry. **megatron** — proves the config migration, since it needs a `paths` block. One of each kind. | 4 |
| **3. Follow the pilots** | **vllm** and **jax** follow atom (schema move, no config change). **torchtitan** replays megatron's migration. | 8 |
| **4. sglang** | Its own project: restructure 5 configs, delete the 138-line legacy adapter, and settle host-subsetting. | 5 |
| **5. Everything else** | health / rccl / ibperf / platform / mori — scoped separately, later. | — |

Phase 2 is deliberately one of each kind so both paths are proven before Phase 3
replays them. sglang is last because it is the only suite whose port is also an
architecture decision.

Unregistered suites keep their current loader throughout. Nothing is deleted until
its replacement is proven.

---

## The ask

1. Agreement that blocks + registry + rules is the right shape.
2. A suite owner each for the two Phase 2 ports.
3. Two decisions we can't make alone:
   - **Cell keys.** Adopting `cells()` as the single producer means either
     regenerating existing threshold files or keeping per-suite formatters for
     compatibility.
   - **Threshold generation.** Is there anything that produces a threshold file from
     a record-only run? Without it, "flip to enforce" means hand-authoring the file.

---
---

# Appendix A — Block reference

Every block, what it holds, and who declares it.

### `Paths` / `ModelPaths`

```python
class Paths(_Forbid):
    shared_fs: str = Field(min_length=1)
    log_dir:   str = Field(min_length=1)

class ModelPaths(Paths):
    models_dir:     str = Field(min_length=1)
    hf_token_file:  str = Field(min_length=1)
```

Split by whether the suite pulls models. All six training/inference suites use
`ModelPaths`; the later suites use `Paths`.

`min_length=1` is load-bearing — a present-but-empty path currently satisfies `str`
and fails later, mid-run.

Participates in three-pass substitution: `{user-id}` from the cluster file, then
self-reference (`{shared_fs}`), then cross-block (`{paths.log_dir}`).

### `ModelSpec`

```python
class ModelSpec(_Forbid):
    id:        str = Field(min_length=1)   # HF repo id or absolute local path
    remote:    Literal[0, 1] = 0
    precision: str = ""
```

Owns the `remote=1 not implemented` guard. The guard lives here — not on the enclosing
config — so a suite composing `ModelSpec` without the shared envelope still gets it.

### `ContainerSpec` / `RuntimeSpec` / `RuntimeArgs`

```python
class RuntimeArgs(_Allow):        # allow: docker/podman own this vocabulary
    volumes:    List[str] = []
    devices:    List[str] = []
    env:        Dict[str, str] = {}
    network:    str = ""
    ipc:        str = ""
    shm_size:   str = ""
    privileged: bool = False

class RuntimeSpec(_Forbid):
    name: Literal["docker", "podman"] = "docker"
    args: RuntimeArgs

class ContainerSpec(_Forbid):
    lifetime: Literal["no_launch", "per_run", "persistent"] = "per_run"
    name:     str = Field(min_length=1)
    image:    str = Field(min_length=1)
    runtime:  RuntimeSpec
```

`ContainerSpec` is closed — four members. `RuntimeArgs` is open because runtime flags
are the runtime's vocabulary, but the members CVS reads are declared so they're typed
and discoverable.

`container.model_dump()` must keep producing what `OrchestratorConfig` consumes.

### `env` — one block

```python
env: Dict[str, str] = {}          # top level, applied to the container
                                  # and exported into the job process
```

Environment variables are set once for a run. One block, at the top level — which is
what the code already does, as below.

Today there are four places they live, plus a fifth that is declared and unused:

| Location | Configs | |
|---|---|---|
| `roles.server.env` | 42 | vllm, atom |
| `training.env_vars` | 3 | jaxmaxtext |
| hardcoded in `megatron_lib.py:540-560` | — | `TORCH_NCCL_ASYNC_ERROR_HANDLING`, `NCCL_IB_*` |
| hardcoded in `torchtitan_lib.py:389-393` | — | `HSA_FORCE_FINE_GRAIN_PCIE`, `PYTORCH_HIP_ALLOC_CONF` |
| `container.runtime.args.env` | **0** | schema supports it; nothing sets it |

The per-role scoping the current schema allows is not merely unused — the code
actively collapses it. vllm writes one env script and has **both** processes source it:

```python
# vllm_job.py:319  — one script, written once
self.orch.exec("bash -c " + shlex.quote(f"printf '%s' ... > /tmp/server_env_script.sh"))

# vllm_job.py:363  — the server sources it
inner = f"source /tmp/server_env_script.sh && nohup {serve_cmd} ..."

# vllm_job.py:591  — and so does the benchmark client
client_cmd = f"source /tmp/server_env_script.sh && {bench_cmd} ..."
```

So `roles.server.env` already reaches the client. The name says otherwise, which is
exactly the kind of thing a newcomer has to read the job code to discover.

The rest of the evidence agrees:

- `roles` is exactly `{server}` in **42 of 42** configs that have it.
- The one config with genuine prefill/decode roles
  (`mi30x_sglang_deepseek_r1_0528_disaggregated.json`) splits node lists, ports, and
  policies — but not env.

The two hardcoded sets matter most for a newcomer. There is no config field for them,
so tuning NCCL on megatron means finding `megatron_lib.py:551` and editing library
code. Folding them into `env` with defaults makes them visible and overridable.

If disaggregated serving later needs a split, `roles.<role>.env` merging **over** the
base block is purely additive — existing configs keep working, no migration. Of the 8
env keys in the corpus most are cluster-wide (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
`GPU_ARCHS`), but `VLLM_ROCM_USE_AITER` and `AMDGCN_USE_BUFFER_OPS` are kernel-selection
flags that could plausibly differ between a compute-bound prefill node and a
memory-bound decode node. That is the trigger to build it — base plus override, not two
parallel blocks as today. Deferring until then costs nothing.

Redaction is framework-owned against one declared secret-key set, and applies wherever
the block is rendered. Suites do not write their own: a per-suite regex is a per-suite
chance to get the escaping wrong, and the failure is silent.

### `ThresholdSpec`

A tagged union on `kind`. Seven kinds, and they do not share a field set.

| kind | extra fields | meaning |
|---|---|---|
| `info` | — | records, never gates |
| `min` | `value` | lower bound |
| `max` | `value` | upper bound, unit-agnostic (counts) |
| `max_ms` | `value` | upper bound, milliseconds |
| `min_tok_s` | `value` | lower bound, tokens/sec |
| `within` | `value`, `tolerance_pct` | band around a target |
| `min_ratio` | `value`, `reference` | ratio against another metric |

```python
ThresholdSpec = Annotated[Union[...], Field(discriminator="kind")]
Thresholds    = Dict[str, Dict[str, ThresholdSpec]]   # cell -> metric -> spec
```

A union is right here and wrong at the top level: this set is closed and owned by one
evaluator that already switches on exactly these seven strings. Suite schemas are
open-ended, which is why the registry keys on a plain string instead.

A misspelled `kind` currently reaches the evaluator and is reported per-metric at run
time. Here it fails at load.

### The sweep blocks

All three implement `entries() -> list[SweepEntry]`. See the sweep section above for
the entry shape and the three implementations.

```python
class ComboSweep(_Forbid):        # vllm, atom
    sequence_combinations: List[SeqCombo] = Field(min_length=1)
    runs:                  List[Run]      = Field(min_length=1)

class MatrixSweep(_Forbid):       # megatron, torchtitan
    combinations: Dict[str, MatrixCombo] = Field(min_length=1)
    runs:         List[str]              = Field(min_length=1)

class NamedSweep(_Forbid):        # jaxmaxtext — lives under `training`, not top level
    sweeps:              List[NamedEntry] = Field(min_length=1)
    enabled_sweep_list:  List[str] = []   # empty = all
```

Two rules apply to all three:

- An unresolvable reference in `runs` / `enabled_sweep_list` is an **error**. A sweep
  that drops a bad reference reports green for a run that never happened — megatron
  warns and skips today, and jax silently widens the run.
- A selector matching nothing is an error, not an empty sweep.

### `BaseVariantConfig` — preset assembly

```python
class BaseVariantConfig(_Forbid):
    schema_version:     Literal[1]
    framework:          str
    gpu_arch:           str
    enforce_thresholds: bool = True
    threshold_json:     str = ""
    paths:              ModelPaths
    model:              ModelSpec
    container:          ContainerSpec
    thresholds:         Thresholds = {}
```

A convenience, not a requirement. It carries **no validation of its own** — every
guard lives on the block it belongs to. A suite needing a different set composes
blocks directly and loads through the same registry entry.

The membership is not a guess. Across the 57 envelope configs on disk:

```
57/57   schema_version, framework, gpu_arch, enforce_thresholds,
        threshold_json, container      → the shared envelope
54/57   sweep                          → all but the no-sweep suite
45/57   paths, model                   → ModelPaths users
42/57   roles, params                  → suite-specific
```

`framework` and `gpu_arch` are in every config and declared separately in every
suite schema. Six declarations of a universal field is exactly the duplication this
removes.

---
---

# Appendix B — Per-suite implementation inventory

What each suite writes today. ✓ = own implementation, ↗ = imported from another
suite, — = not present.

| | vllm | atom | sglang | megatron | torchtitan | jax |
|---|---|---|---|---|---|---|
| extends `BaseVariantConfig` | — | ✓ | ✓ | — | — | ✓ |
| `load_variant` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `cell_key` | ✓ | ✓ | ✓ ×2 | ✓ | ✓ | — |
| `cell_key` signature | isl,osl,conc | isl,osl,conc | isl,osl,conc | combo_key | combo_key | n/a |
| `expected_cells` | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| `validate_sweep_selector` | ✓ | ↗ vllm | — | ✓ | ✓ | — |
| `validate_thresholds_cover_*` | ✓ | ↗ vllm | — | ✓ | ✓ | ✓ |
| `_check_no_changeme` | — | — | — | ✓ | ✓ | ✓ |
| `expand_sweep` | — | ✓ | — | — | — | — |
| `orchestrator_container_from_variant` | — | ✓ | ✓ | — | — | — |
| legacy load path | — | — | ✓ | — | — | — |
| `@model_validator` | 4 | 5 | 1 | 2 | 2 | 0 |
| `@field_validator` | 1 | 1 | 0 | 0 | 0 | 1 |
| loader lines | 275 | 358 | 453 | 218 | 210 | 264 |
| conftest lines | 180 | 149 | 526 | 187 | 164 | 235 |

**After:** every row above except `cell_key`, the suite schema, and suite-specific
rules moves into the framework. `cell_key` stays, with one signature.

The megatron / torchtitan columns are identical because the files are — their sweep
blocks and all four helper functions differ only in class names.
