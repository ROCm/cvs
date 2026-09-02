# CVS config generation

Companion to [`cvs-config-api-refactor.md`](cvs-config-api-refactor.md). That doc
defines the config *schema* and the `auto` resolution phase. This one is about getting a
config file to exist in the first place.

**It is downstream of the refactor and cannot be read as freestanding.** Generation
emits whatever shape the schema defines; if the schema's sweep departure lands after a
generator ships, every generated file is immediately legacy. Sequencing is treated
explicitly at the end.

Scope is the same six suites: vllm, atom, sglang, megatron, torchtitan, jaxmaxtext.

---

## The problem this solves

`cvs copy-config` is generation today, and it is `shutil.copyfile`
(`copy_config_plugin.py:129`). It hands the user a template with `<changeme>` still in
it. Across `cvs/input/config_file/` that is **75 files carrying 354 occurrences**, out
of 145 configs total. Authoring the rest by hand is the friction.

The asymmetry is the tell:

| Half | State |
|---|---|
| **Cluster file** | *generated.* `cvs generate cluster_json` takes `--hosts` (or a hosts file), expands `192.168.1.10-20` and `host[1-10]`, renders a Jinja template. |
| **Config file** | *copied.* Byte-for-byte. Placeholders survive. |

So CVS already believes in generation — it just stopped at the easier of the two files.

**Generation is not resolution.** Resolution (`"auto"`, in the companion doc) fills
values in memory at run time: zero files, always current, nothing for the user to see or
tune. Generation produces an artifact. Both consume the same probes, so the probe layer
gets built once with two consumers — but they are not substitutes, because a config the
user cannot see is a config the user cannot tune.

### The extension point already exists

`GeneratorPlugin` (`cli_plugins/generate_plugin.py:43-92`) is an ABC with dynamic
discovery over `cvs/input/generate/` and `cvs/reports/generate/`. Adding
`cvs generate config` is **dropping one file into a directory** — no CLI plumbing, no
registry edit. Jinja2 is already a dependency and templates already have a home at
`cvs/input/templates/<kind>/`.

The probes exist too, all currently unwired for this purpose:

| Probe | Where | Used today for |
|---|---|---|
| HCA names + socket netdev, with cross-node asymmetry detection | `lib/utils/ib_discovery.py` | `auto` in atom/vllm fabric resolution |
| total / used / free VRAM | `parse_mem_usage`, `lib/utils/gpu.py` | captured, marked *"Not used as test rows"* |
| model size on disk | `_du_bytes`, `tests/inference/vllm/vllm.py:142` | download-progress poll, not a fit check |
| GPU architecture | `get_model_from_rocm_smi_output` | a torchtitan fixture |

Nothing here needs inventing. It needs wiring and a front door.

---

## Case study: vllm llama3.1-70B

vllm ships exactly two configs for this model — `mi300x_vllm_llama31-70b_fp8_single.json`
(83 lines) and `..._distributed.json` (87). A field-level diff of the two returns **nine
differences, and not one of them is knowledge**:

| Field | single → distributed | What it actually is |
|---|---|---|
| `params.nnodes` | `"1"` → `"2"` | `len(orch.hosts)` |
| `params.master_addr` | absent → `<changeme>` | `orch.hosts[0]` |
| `params.pipeline_parallel_size` | `"1"` → `"2"` | `= nnodes` on the mp backend |
| `roles.server.ib_netdev` | absent → `<changeme>` | probe — `ip -4 -o addr show` on the host's own IP |
| `roles.server.ib_hca_devices` | absent → `"auto"` | **already resolved this way** |
| `params.master_port` | absent → `"29501"` | static |
| `container.name` | `w1_…` → `w2_…` | cosmetic |
| `sweep.sequence_combinations` | `w1_isl=1000_osl=1000` → `w2_…` | **name prefix only** |
| `sweep.runs` | `combo: "w1_…"` → `"w2_…"` | same prefix; concurrency 16 in both |

The last two are worth dwelling on. The sweep entries are otherwise identical — same
`isl`, same `osl`, same `goodput_slo`, same concurrency. And `cell_key`
(`lib/inference/utils/vllm_config_loader.py:226-238`) formats the threshold key from
`isl`, `osl`, `tp`, `pp` and `concurrency`: **the combo `name` never enters the key.**
So the `w1_`/`w2_` prefix changes no behavior, no lookup, and no result. It is a
distinction that distinguishes nothing.

Strip the cosmetics and an entire 87-line file exists to express **five derivable facts
and one port number.**

### It is not a vllm quirk

Counting files whose names differ only by a topology suffix (`_single`,
`_distributed`, `_disaggregated`, `_multinode`):

```
2  inference/atom/mi300x_atom_deepseek-r1_fp8_<TOPO>.json
2  inference/atom/mi355x_atom_deepseek-r1_fp8_<TOPO>.json
3  inference/sglang/mi30x_sglang_deepseek_r1_0528_<TOPO>.json
2  inference/vllm/mi300x_vllm_llama31-70b_fp8_<TOPO>.json
2  training/jax/mi300x_jax_llama3_1_70b_<TOPO>.json
2  training/jaxmaxtext/mi300x_jaxmaxtext_llama-3.3-70b_<TOPO>.json
2  training/megatron/mi325x_megatron_llama-3.3-70b_<TOPO>.json
2  training/megatron/mi3xx_megatron_llama_<TOPO>.json
2  training/torchtitan/mi3xx_torchtitan_deepseek_<TOPO>.json
2  training/torchtitan/mi3xx_torchtitan_llama_<TOPO>.json
2  training/torchtitan/mi3xx_torchtitan_qwen3_<TOPO>.json
```

**23 files in 11 topology sets, spanning all six suites plus jax. 12 of them are the
redundant members.** Every suite independently decided topology is a filename axis.

### And the easy case isn't easy either

The *single-node* file still carries two `<changeme>`: `threshold_json` and
`container.image`. Neither is cluster-specific. Both are knowable. So even the
zero-multinode path is not a zero-edit path today.

---

## Possible solutions

The design question is **what generation reads from**.

| | Source of truth | Trade |
|---|---|---|
| **S1** | Clone-and-retarget the nearest shipped config | Inherits all measured tuning for free. Also inherits its bugs and its shape, and drifts as the configs drift. |
| **S2** | A thin per-(model, arch) recipe behind a template | Clean, queryable, one obvious home for a new model. A second corpus to maintain — and on day one it is a transcription of S1. |
| **S3** | Synthesize from schema defaults | Nothing to maintain. Produces a config nobody tuned: `num_prompts: 320` is not a default, it is somebody's measured result. |
| **S4** | No file at all — resolve `(model, cluster)` at run time | Zero artifacts to rot. Kills the tuning path, which is the stated requirement. |

**Recommendation: S1 now, S2 as the migration target.** S1 ships against the corpus that
already exists and is honest about where its numbers came from. S2 is what S1 becomes
once generation has revealed which fields actually vary across models — that information
does not exist yet, and guessing it now is how the wrong recipe format gets locked in.
S3 and S4 are coherent positions, but each surrenders something already asked for.

### Independent of S1–S4: topology stops being a file axis

One config; `nnodes` from the cluster file; `pp` derived. The 12 redundant files above
stop needing to exist. This is the largest concrete win in the generation story and it
does not depend on which source-of-truth option wins.

---

## The nine ideas

### G1 — Clone-and-retarget, not synthesize

The 145 shipped configs are accumulated tuning knowledge. `num_prompts: 320`,
`client_poll_count: 90`, the `serve_args` — none are schema defaults; they are per-model
results someone measured. Generation picks the nearest shipped config as a base and
re-targets it. Pydantic can dump structure; it cannot dump knowledge.

Falls out for free: the missing-architecture problem. There is no MI325X vllm config
anywhere — but generating from the MI300X one and swapping the threshold reference
produces a usable starting point instead of nothing.

### G2 — Invert the front door: cluster-first

Today the flow is *pick a config, hope it fits the cluster*. Flip it:

```
cvs generate config vllm --cluster cluster.json
  → probes: 3 nodes × 8 GPUs, 192 GB/GPU, models staged in /shared/models
  → offers the (model, topology) pairs that fit
  → writes the one you pick
```

Same fit arithmetic as the preflight gate, run as a **generator** rather than a
validator. Minimum input stays node IPs, plus the two questions nothing can answer for
the user.

### G3 — Freeze the tuning surface, `auto` the hardware

The split rule for what gets written as a literal versus left as `auto`:

| | Treatment | Why |
|---|---|---|
| `num_prompts`, concurrency, `max-model-len`, sweep shape | **frozen literal** | this *is* the tuning surface; invisible is useless |
| HCAs, netdev, `rocm_dir`, `gpu_arch` | **`auto`** | only the hardware knows, and it changes if they re-cable |
| `nnodes`, `master_addr` | **omitted entirely** | restatements of the cluster file; keeping them creates a divergence class |

The failure mode to avoid is freezing everything — a fully concrete config generated on
cluster A silently misdescribes cluster B.

### G4 — Generate the pair, in record mode

Emit the config **and** its threshold file together, zeros throughout,
`enforce_thresholds: false`. That is exactly the intended lifecycle: the first run is
meant to be an easy pass that records.

### G5 — Every generated value carries its provenance

"Why is `num_prompts` 320?" should be answerable from the file. The sources are a small
closed set: `probe`, `cluster-file`, `recipe:<name>`, `default`, `you`. CVS already has
the `_comment_X` convention to carry it.

Generation also gets to **fix an existing problem**: the `_example_` keys in the
megatron and jax configs ship real foreign-cluster values (`bnxt_re0-7`, `ens51f1np1`,
`rocep28s0`). AGENTS.md forbids shipping cluster-specific values precisely because users
copy them. A generator replaces those with what was discovered on *their* cluster.

### G6 — Stamp it, diff it, never clobber it

CSPs will commit these files. Stamp the source recipe and its version; `--diff` shows
what moved upstream since; refuse to overwrite a file the user has edited.
Generated-then-owned is the normal lifecycle, not an edge case.

### G7 — Write outside the package

The harness runs from a `site-packages` copy, not the git source. Generated configs must
land in a user-owned path — writing into `cvs/input/config_file/` puts them where the
next `pip install` erases them.

### G8 — Starter sweeps, not a blank `runs` block

Offer *smoke* / *qualification* / *full envelope* and populate `runs` accordingly. This
is where "set up sweeps across any axis" actually gets delivered: most users want a good
default sweep, and the ones who don't will edit it — which is the point of writing a
file rather than resolving in memory.

### G9 — The round-trip test is the honesty gate

`generate → load → validate` must pass with **zero edits**, for every (suite, model) the
catalog claims to support. One test in CI, and generation cannot rot as schemas move.

It is also the only mechanical acceptance test for "a user with only node IPs can run a
suite." Without it, that goal is an aspiration that cannot fail.

---

## Where this fits

Generation owns the head of the pipe. The refactor owns everything from `load` onward.
They meet at exactly one contract — the schema.

```
cvs generate cluster_json        ← exists today
  → cvs generate config          ← this doc (G1/G2)
    → load + validate            ← refactor doc: the schema
      → resolve auto             ← refactor doc: the new I/O phase
        → materialize the resolved config into the run dir   ← refactor doc
          → run, records results
            → cvs generate thresholds     ← the missing emitter (G4)
              → enforce
```

Two consequences worth stating plainly.

**Generation is the refactor's acceptance test.** G9 is what proves the schema is
actually usable from a standing start, rather than merely well typed.

**Sequencing is a real risk.** The refactor proposes replacing five sweep shapes with one
flat `runs` block. A generator shipped before that lands writes files in the old shape,
and the migration then has two producers to fix instead of one. Either generation
targets the post-departure shape from day one, or it ships after the departure.
Preference: **target the post-departure shape**, and let the generator be the first
consumer that proves it works.

---

## What generation cannot do

Three things stay the user's, and the design should stop pretending otherwise:

1. **Which model.**
2. **What they are proving** — smoke, qualification, or full envelope.
3. **Their SLOs.**

Two questions plus node IPs is the floor. Everything else is derivable, probeable, or
already shipped.

---

## Open questions

1. **Recipe format** — S1 or S2, and if S2, when. The answer probably depends on data
   that only running S1 produces.
2. **Interactive or flags-only?** A wizard fits the day-0 target; flags fit CI. Likely
   both, wizard when a TTY is present.
3. **Does generation require cluster access?** G2's cluster-first flow does. Offline
   generation (`--arch mi355x --nodes 2`) is weaker but works from a laptop. Supporting
   both means offline emits more `auto`.
4. **Who curates recipes** as models are added — the same ownership question as canonical
   model naming in the companion doc.
5. **Does `cvs copy-config` survive?** If generation covers the same ground with better
   output, keeping both means two ways to get a config and one of them ships
   `<changeme>`.
