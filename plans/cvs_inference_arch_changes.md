# CVS Inference Suite: Architectural Changes

The existing training suites (Megatron, JAX, Mori) were designed around a simple model: one test function runs a job, one function verifies it passed. That works when you have one run configuration, one metric, and container lifecycle is a detail. Inference benchmarking broke all three assumptions: runs are parametrized over many ISL/OSL/concurrency combinations, results are dozens of metrics with percentile distributions, and container lifecycle is load-bearing enough to need a policy. This document covers the three architectural changes that address each.

---

## 1. The Orchestrator Handle

### The problem

Every old suite owns its container lifecycle directly. In practice this means each test file constructs a raw `Pssh` handle, calls `docker_lib` functions with hardcoded arguments, and runs teardown as a best-effort step at the end of the happy path. There is no policy, no validation, and no guarantee that teardown runs if a test fails mid-suite.

```
Old pattern (every suite, repeated):

  test_cleanup_stale_containers   →  docker_lib.kill_docker_container(phdl, name)
                                     docker_lib.delete_all_containers_and_volumes(phdl)

  test_launch_*_containers        →  docker_lib.launch_docker_container(
                                         phdl, name, image,
                                         training_dict['container_config']['device_list'],
                                         training_dict['container_config']['volume_dict'],
                                         shm_size='128G', timeout=60*20
                                     )
                                     # no verification the container is actually running

  test_*_training                 →  ... run job ...
                                     update_test_result()  ← fails here

  # teardown never reached; container leaks on every non-happy-path exit
```

If `test_*_training` throws, teardown is skipped. There is no recovery path.

### The change: lifecycle as policy, teardown as a guarantee

Container management moves into `OrchestratorConfig` + `ContainerOrchestrator`. The test file declares a policy in the config; the orchestrator enforces it. The `orch` fixture registers a leak-guard finalizer that fires even if a test throws.

```
New pattern (lifecycle flow):

  ┌─────────────────────────────────────────────────────────┐
  │  Happy path                                             │
  │                                                         │
  │  test_launch_container  →  orch.setup_containers()      │
  │  test_setup_sshd        →  orch.setup_sshd()            │
  │  test_model_fetch       →  verify model bytes present   │
  │  test_vllm_inference    →  run benchmark cell(s)        │
  │  test_teardown          →  orch.teardown_containers()   │
  │                            sets lifecycle.torn_down     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  Failure path                                           │
  │                                                         │
  │  test_launch_container  →  ok                           │
  │  test_setup_sshd        →  ok                           │
  │  test_vllm_inference    →  raises                       │
  │                            lifecycle.failed = True      │
  │                            remaining cells skip         │
  │  test_teardown          →  skipped (lifecycle.failed)   │
  │                                                         │
  │  orch fixture finalizer →  lifecycle.torn_down is False │
  │                            teardown fires here instead  │
  └─────────────────────────────────────────────────────────┘
```

Each lifecycle stage (launch, sshd, teardown) is a first-class pytest test — it appears as its own timed, pass/fail row in the report. The old pattern hides these inside test bodies or fixture code where timing and failure attribution are invisible.

### The policy: `container.lifetime`

Instead of imperative `docker_lib` calls, the config declares a single key:

```json
"container": {
  "lifetime": "per_run",
  "name": "w1_llama31_70b_fp8kv_perf_inference_rocm",
  "image": "<image>",
  "runtime": {
    "name": "docker",
    "args": {
      "network": "host",
      "ipc": "host",
      "privileged": true,
      "volumes": [
        "/home/{user-id}:/home/{user-id}",
        "{paths.models_dir}:/models"
      ]
    }
  }
}
```

`container.lifetime` has three values:

- `per_run` — start fresh on setup, force-remove on teardown. The default.
- `persistent` — attach if already running; cold-start if absent on all nodes; refuse if partially present (avoids destroying a running container's overlay).
- `no_launch` — CVS never starts the container; verify one is already running and attach. Teardown is a no-op.

The old suites had no equivalent of this. Every suite implemented its own variant of "kill anything that exists, launch fresh" with no ability to attach to a pre-existing container or verify a user-managed one.

---

## 2. Typed Config and Separated Thresholds

### The problem

Old suite configs are a single unvalidated JSON blob. Thresholds are buried inside `model_params`, nested alongside training hyperparameters, volume mounts, and container names:

```json
"model_params": {
  "single_node": {
    "llama3_1_70b": {
      "mi300x": {
        "result_dict": {
          "throughput_per_gpu": "4200"
        },
        "tp": 8,
        "container_name": "megatron_training",
        ...
      }
    }
  }
}
```

There is no schema — a typo in a threshold key silently becomes a log warning (`Perf result key X not provided, so will not be checked`) and the threshold is never asserted. There is no way to run record-only without editing code. And the only supported assertion is a single floor check: `float(actual) < float(expected)`.

### The change: three separate artifacts with validated schema

```
input/config_file/inference/vllm_single/
├── cluster_config.json          # nodes, SSH credentials, container base settings
├── mi300x_..._config.json       # variant: model, paths, sweep, container policy
└── mi300x_..._threshold.json    # thresholds only, keyed by cell
```

The variant config is validated by a Pydantic model on load — unknown keys fail loudly, required fields are type-checked, and `{user-id}` placeholders are resolved in a defined three-pass order. The threshold file is declared inside the variant config by path (`threshold_json`), loaded separately, and merged into `variant_config.thresholds` at fixture time.

```json
// mi300x_..._config.json (excerpt)
{
  "schema_version": 1,
  "enforce_thresholds": false,
  "threshold_json": "/path/to/threshold.json",
  "paths": {
    "shared_fs": "/home/{user-id}",
    "models_dir": "{shared_fs}/models",
    "log_dir":    "{shared_fs}/LOGS"
  },
  "model": { "id": "amd/Llama-3.1-70B-Instruct-FP8-KV", "remote": 0 },
  "container": { "lifetime": "per_run", ... },
  "params": { "tensor_parallelism": "8", ... },
  "sweep": { ... }
}
```

`enforce_thresholds: false` makes the entire run record-only — every metric row passes and displays its value, but nothing is asserted. Flip it to `true` once numbers are calibrated.

### Threshold expressiveness

The threshold file is keyed by cell (`ISL=...,OSL=...,TP=...,CONC=...`). Each metric has a typed spec with a `kind` field:

```json
"ISL=1000,OSL=1000,TP=8,CONC=16": {
  "client.total_token_throughput": { "kind": "min_tok_s", "value": 45000  },
  "client.output_throughput":      { "kind": "min_tok_s", "value": 38000  },

  "client.mean_ttft_ms":           { "kind": "max_ms",   "value": 80     },
  "client.p95_ttft_ms":            { "kind": "max_ms",   "value": 120    },
  "client.p99_ttft_ms":            { "kind": "max_ms",   "value": 150    },

  "client.mean_tpot_ms":           { "kind": "max_ms",   "value": 35     },
  "client.p99_tpot_ms":            { "kind": "max_ms",   "value": 55     },

  "client.success_rate":           { "kind": "min",      "value": 0.99   },
  "client.failed":                 { "kind": "max",      "value": 5      }
}
```

Supported kinds: `min`, `max`, `min_tok_s`, `max_ms`, `within` (value ± tolerance_pct), `min_ratio` (actual/reference_metric ≥ value). Adding a new kind means one new branch in `verdict.py` — no test file changes.

In the old design, the only assertion kind that existed was a floor check (`float(actual) < float(expected)`). A ceiling on latency, a tolerance band, or a ratio between two metrics required editing lib code.

---

## 3. Sweep Parametrization and Per-Metric Rows

### The problem

In the old suites, one test function equals one hardcoded run. Adding an ISL/OSL combination means writing a new test function or a new test file. All metric checks are fused inside `verify_training_results()` in the lib — the entire job collapses into one pytest row regardless of how many things are being checked. If it fails, you read log text to find out why.

For inference this does not scale: a typical sweep is 5 ISL/OSL combinations × multiple concurrency levels × ~30 metrics each. That is potentially hundreds of distinct data points that the old model would collapse into a single PASS or FAIL.

### The change: config-driven sweep expansion

The sweep is declared in the variant config, not in Python:

```json
"sweep": {
  "sequence_combinations": [
    { "name": "isl=1000_osl=1000", "isl": "1000", "osl": "1000" },
    { "name": "isl=8000_osl=1000", "isl": "8000", "osl": "1000" },
    { "name": "isl=1000_osl=8000", "isl": "1000", "osl": "8000" },
    { "name": "isl=5000_osl=1024", "isl": "5000", "osl": "1024" }
  ],
  "runs": [
    { "combo": "isl=1000_osl=1000", "concurrency": 16 },
    { "combo": "isl=8000_osl=1000", "concurrency": 16 },
    { "combo": "isl=1000_osl=8000", "concurrency": 16 },
    { "combo": "isl=5000_osl=1024", "concurrency": 16 }
  ]
}
```

`pytest_generate_tests` reads this at collection time and expands it into one `test_vllm_inference` case per run. A reference to an unknown combo name fails collection immediately — before any hardware is touched.

```
Collection time:

  sweep.runs  →  pytest_generate_tests  →  test_vllm_inference[isl=1000_osl=1000-conc16]
                                           test_vllm_inference[isl=8000_osl=1000-conc16]
                                           test_vllm_inference[isl=1000_osl=8000-conc16]
                                           test_vllm_inference[isl=5000_osl=1024-conc16]

  each cell ×  CLIENT_METRICS  →          test_metric[isl=1000_osl=1000-conc16-mean_ttft_ms]
                                           test_metric[isl=1000_osl=1000-conc16-p95_ttft_ms]
                                           test_metric[isl=1000_osl=1000-conc16-p99_ttft_ms]
                                           test_metric[isl=1000_osl=1000-conc16-mean_tpot_ms]
                                           ...
```

Adding a new sweep cell costs one line in `runs`. No Python changes.

### Per-metric rows: `CLIENT_METRICS` and `GATED_METRICS`

`CLIENT_METRICS` is the metric registry in `vllm_parsing.py` — a flat list of `(name, unit)` pairs. Every entry automatically becomes its own `test_metric` row. Adding a metric to the display surface costs one line:

```python
CLIENT_METRICS = [
    # throughput
    ("total_token_throughput",      "tok/s"),
    ("output_throughput",           "tok/s"),
    ("per_gpu_throughput",          "tok/s"),

    # TTFT — full distribution
    ("mean_ttft_ms",                "ms"),
    ("median_ttft_ms",              "ms"),
    ("p90_ttft_ms",                 "ms"),
    ("p95_ttft_ms",                 "ms"),
    ("p99_ttft_ms",                 "ms"),

    # TPOT — full distribution
    ("mean_tpot_ms",                "ms"),
    ("median_tpot_ms",              "ms"),
    ("p90_tpot_ms",                 "ms"),
    ("p95_tpot_ms",                 "ms"),
    ("p99_tpot_ms",                 "ms"),

    # ITL
    ("mean_itl_ms",                 "ms"),
    ("median_itl_ms",               "ms"),
    ("p95_itl_ms",                  "ms"),
    ("p99_itl_ms",                  "ms"),

    # E2E latency
    ("mean_e2el_ms",                "ms"),
    ("p90_e2el_ms",                 "ms"),
    ("p99_e2el_ms",                 "ms"),

    # health
    ("success_rate",                "-"),
    ("failed",                      "-"),
    ...
]
```

`GATED_METRICS` is a separate set — the subset of `CLIENT_METRICS` where a missing threshold spec is a hard failure. A metric in `CLIENT_METRICS` but not in `GATED_METRICS` is record-only by default: it gets a row in the report and shows its value, but the absence of a threshold spec does not fail the run. Moving a metric from record-only to gated means adding its name to `GATED_METRICS` and adding a spec to the threshold JSON — no lib changes, no test file changes.

In the old design, every metric check was gated (or silently skipped) with no middle ground. There was no concept of "show this in the report but don't assert on it yet."
