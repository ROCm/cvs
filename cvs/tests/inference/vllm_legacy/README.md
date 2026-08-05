# Legacy vLLM suites (deprecated)

**Notice given 2026-08-11 · scheduled for removal 2026-11-11.**

These are the pre-#223 single-node vLLM suites. They were hard-deleted by
"vLLM single node refactor (#223)" without a deprecation period; they are
restored here so the 3-month OSS notice window is actually served.

Nothing new should be built on them.

## Replacement

| Deprecated | Replacement |
|---|---|
| `cvs run vllm_gpt_oss_120b_single` | `cvs run vllm` |
| `cvs run vllm_qwen3_80b_single` | `cvs run vllm` |
| `cvs run vllm_qwen3_235b_single` | `cvs run vllm` |
| `cvs run vllm_deepseek31_685b_single` | `cvs run vllm` |
| `cvs/lib/inference/vllm.py` (`VllmJob`) | `cvs/lib/inference/vllm_job.py` (`VllmJob`) |
| `cvs/lib/inference/base_legacy.py` | `cvs/lib/inference/base.py` |
| `input/config_file/inference/vllm_legacy/mi355x_vllm_single.json` | `input/config_file/inference/vllm/*.json` (`schema_version: 1`) |

The unified `vllm` suite covers single-node **and** PP-distributed runs from one
config (topology follows `params.nnodes` / `params.pipeline_parallel_size`), and
adds GPU metrics, vLLM `/metrics` scraping, threshold enforcement against a
sibling threshold file, and the run-deck HTML report.

Migrating means rewriting the config: the legacy shape is `config` +
`benchmark_params` with thresholds inline under `result_dict`, whereas the
unified suite takes a pydantic-validated `schema_version: 1` config with
`paths` / `model` / `container` / `roles` / `params` / `sweep` blocks and a
separate threshold file.

## Running them meanwhile

```
cvs run vllm_gpt_oss_120b_single \
  --cluster_file input/cluster_file/cluster.json \
  --config_file input/config_file/inference/vllm_legacy/mi355x_vllm_single.json \
  --html=/var/www/html/cvs/gpt.html --self-contained-html \
  --capture=tee-sys --log-file=/tmp/gpt.log -vvv -s
```

Importing `cvs.lib.inference.vllm` raises a `DeprecationWarning` naming the
removal date and the replacement.

## Known limitations of the frozen path

These are preserved pre-#223 behaviour, not new defects. They are not fixed
here because doing so would break the freeze that makes this a faithful
deprecation notice; the replacement suite addresses all of them.

1. **The server launch scripts are not distributed with CVS.** The config's
   `server_script` values (`gpt-oss-120b_fp4_mi355x_vllm_docker.sh`,
   `qwen3-235b-bf16_mi355x_vllm_docker.sh`,
   `qwen3-80b-bf16_mi355x_vllm_docker.sh`, `dsr1_fp8_mi355x_vllm_docker.sh`)
   live in no repository. Pre-stage them at `benchmark_server_script_path` on
   every node or the command above cannot run.
2. **Every model block must spell `random_range_ratio`.** The frozen base
   setdefaults the misspelled `random_range_ration`, so the correctly-spelled
   key it later reads has no fallback and a config omitting it raises
   `KeyError` while building the server command.
3. **Server-readiness budget is tight.** 30s precheck + 330s wait + 20 polls x
   60s = 1560s (2160s for `vllm_qwen3_80b_single`, which polls 30 times). A
   first run that downloads weights from HF can exhaust it and fail on a
   readiness timeout rather than on a real fault. Pre-stage the weights. There
   is no config knob for this -- `server_launch_poll_count` is a constructor
   argument, so the only in-tree override is the `VllmJob(...)` call site in
   the suite file.
4. **The bench client is cloned at run time** from
   `https://github.com/kimbochen/bench_serving.git`, a third-party repository
   that is not pinned or mirrored.

## Why this directory, and why no conftest.py

`cvs/tests/inference/vllm/conftest.py` defines `pytest_collection_modifyitems`,
which sorts collected items by a rank table keyed on the *unified* suite's test
names (`test_launch_container`, `test_discover_topology`, ...). These suites'
test names are absent from that table, so they would all tie at the default
rank and be reordered into an unrunnable sequence — `test_vllm_inference` would
run before `test_launch_inference_containers`.

Keeping them in a sibling directory with **no conftest.py** avoids that
entirely. These suites carry their own fixtures inline (`cluster_dict`,
`hf_token`, `s_phdl`, `c_phdl`), so they need nothing from a conftest. The
ancestor `cvs/tests/conftest.py` only defines a lazily-evaluated `orch` fixture
that these suites never request, and `cvs/conftest.py`'s report auto-registration
no-ops for these stems (there is no `cvs.lib.report.presets.<stem>` module).

`cvs list` / `cvs run` discover suites by filename stem, so all four remain
selectable exactly as before.

## Removal checklist (on or after 2026-11-11)

1. Delete `cvs/tests/inference/vllm_legacy/`.
2. Delete `cvs/lib/inference/vllm.py` and `cvs/lib/inference/base_legacy.py`.
3. Delete `cvs/input/config_file/inference/vllm_legacy/`.
4. Delete `cvs/lib/inference/unittests/test_vllm_legacy_deprecation.py`.
5. Drop the `'vllm'` entry from `InferenceJobFactory._FRAMEWORK_CLASSES`
   (`cvs/lib/inference_lib.py`) if nothing else references it.
