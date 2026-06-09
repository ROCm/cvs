# PoC: vllm_single refactor (orch + typed configs)

Branch: `dev/dtni`. Worktree: `/data/atnair/repos/cvs_worktrees/cvs-dtni/`.

Replace 4 byte-similar `vllm_single` wrappers with one parametrized wrapper, per-variant config + threshold dirs, a typed config loader, and a new orch-driven VllmJob. Container lifecycle moves entirely to `ContainerOrchestrator` (`launch: true`). Existing `cvs/lib/inference/{base,vllm,inference_max}.py` untouched (other suites still use them).

## Approach

- Add `cvs/lib/dtni/__init__.py`, `cvs/lib/dtni/verdict.py` (~60 LOC; 5 kinds: `min`, `max_ms`, `within`, `min_tok_s`, `min_ratio`; `evaluate_all(actuals, thresholds)`).
- Add `cvs/lib/dtni/config_loader.py` (~110 LOC). Pydantic models with `extra="forbid"`: top-level `schema_version: Literal[1]`, `framework: Literal["vllm_single"]`, `gpu_arch`, `paths` (5 required keys), `model {id, remote: Literal[0,1], precision}`, `image {tag, remote}`, `container` (passthrough shape matching `cvs/core/orchestrators/factory.OrchestratorConfig` — `enabled`, `launch`, `name`, `runtime {name, args}`; `runtime.args` is `extra="allow"`), `roles.server`, `params`, `benchmark_params`, `sweep`. `load_variant(config_path, cluster_dict) -> VariantConfig` loads `config.json` + sibling `threshold.json`, runs 3-pass placeholder substitution: cluster (`{user-id}`) → self-ref (`{shared_fs}` in `paths.*`) → cross-block (`{paths.models_dir}` in `container.runtime.args.volumes`). `model.remote=1` raises `NotImplementedError` pointing at cvs-dtni-v1 `resource_resolver.py`. `enumerate_variants(root_dir) -> list[Path]` walks `cvs/input/dtni/vllm_single/*/`.
- Add `cvs/lib/inference/vllm_orch.py` (~250 LOC). Standalone `VllmJob` — does NOT inherit `InferenceBaseJob`. Takes `orch: ContainerOrchestrator`. Methods: `build_server_cmd`, `start_server`, `is_ready`, `wait_ready`, `run_client`, `wait_client_complete`, `parse_results`, `stop_server`. All exec via `orch.exec` / `orch.exec_on_head` (orch already routes into the container). Drops: dead distributed branch (`self.port_no`), `random_range_ration` typo, `globals.error_list` indirection, silent-skip in `verify_inference_results`.
- Add `cvs/tests/inference/vllm/conftest.py` (~70 LOC): `cluster_dict`, `variant_config`, `orch` (constructs `OrchestratorConfig` from cluster_dict identity + `variant_config.container.dict()`; calls `orch.setup_containers()`, yields, calls `orch.teardown_containers()`), `hf_token`, `inf_res_dict` (session-scoped table). `pytest_generate_tests` parametrizes `test_vllm_inference` over `(seq_combo × concurrency)` from the loaded variant.
- Add `cvs/tests/inference/vllm/_shared.py` (~30 LOC): `test_print_results_table(inf_res_dict)`. `test_cleanup_stale_containers` + `test_launch_inference_containers` collapse into the `orch` fixture lifecycle.
- Add `cvs/tests/inference/vllm/vllm_single.py` (~60 LOC): `from ._shared import *`. Single `test_vllm_inference(orch, variant_config, hf_token, seq_combo, concurrency, inf_res_dict)` — constructs `VllmJob(orch=orch, ...)`, runs `stop_server → build_server_cmd → start_server → wait_ready → run_client → wait_client_complete → parse_results → evaluate_all(actuals, variant_config.thresholds)`.
- Add 4 per-variant dirs `cvs/input/dtni/vllm_single/<full-model-name>_perf/{config.json, threshold.json}` for: `Qwen3-Next-80B-A3B-Instruct`, `Qwen3-235B-A22B-Instruct-2507-FP8`, `DeepSeek-V3.1-Terminus`, `gpt-oss-120b`. Each `config.json` carries the existing per-model slice from `mi355x_vllm_single.json` plus the new `paths`/`model`/`image`/`container` blocks. `threshold.json` ports the model's `result_dict` keys.
- Delete `cvs/tests/inference/vllm/vllm_qwen3_80b_single.py` (480), `vllm_qwen3_235b_single.py` (449), `vllm_deepseek31_685b_single.py` (453), `vllm_gpt_oss_120b_single.py` (451), and `cvs/input/config_file/inference/vllm/mi355x_vllm_single.json` (95).

## Out of scope / Future additions

- Accuracy variants (`*_accuracy/`, `test_vllm_accuracy`, lm_eval harness, dataset download).
- `cvs/lib/inference/{base,vllm,inference_max}.py` cleanup. Old VllmJob stays; InferenceMaxJob still inherits InferenceBaseJob.
- Orphan `cvs/lib/inference_lib.py` (`InferenceJobFactory` importing non-existent module) — separate cleanup PR.
- `model.remote=1` implementation. Schema present, raises NotImplementedError; port from cvs-dtni-v1 `resource_resolver.py` later.
- `cvs/lib/docker_lib.py` deletion / `cvs/lib/parallel_ssh_lib.py` deprecation cleanup.
- Switching `cvs/core/*` off the deprecated `parallel_ssh_lib` shim (DeprecationWarning will appear in test output — not blocking).
- Other suites (sglang, inferencemax, pytorch_xdit, megatron, jax) — untouched.
- `cvs migrate-config` tool — the 4 variant configs are hand-written.
- Manifest / sidecar / `cvs export` (v1 W4).
- Topology / sweep semantics rework (v1 W5).
- Dev guide (separate doc, written after PoC lands and validates).

## Verification

1. `cd /data/atnair/repos/cvs_worktrees/cvs-dtni && python -m pytest --collect-only cvs/tests/inference/vllm/vllm_single.py --config_file=cvs/input/dtni/vllm_single/Qwen3-Next-80B-A3B-Instruct_perf/config.json --cluster_file=<existing mi355x cluster file>` — expect collection lists `test_vllm_inference[balanced-conc16]` plus `test_print_results_table`. No errors.
2. `cvs list vllm_single` — expect enumeration of `test_vllm_inference[...]` for all 4 variants (uses default-walk of `cvs/input/dtni/vllm_single/` when `--config_file=dummy`). Plus `test_print_results_table`.
3. `python -m pytest cvs/tests/inference/vllm/vllm_single.py --config_file=cvs/input/dtni/vllm_single/Qwen3-Next-80B-A3B-Instruct_perf/config.json --cluster_file=<mi355x cluster>` on hardware — expect exit 0. Cell `balanced-conc16` produces a results dict; `test_print_results_table` prints a row matching (within ±10%) the numbers in `vllm_7.14_run.zip / test_print_results_table_*.html`: `Req/s`, `Total tok/s`, `Mean TTFT (ms)`, `Mean TPOT (ms)`, `P99 ITL (ms)`.
4. During the run, `ssh <mi355x node> docker ps` shows the `vllm_inference_rocm` container appear at `orch.setup_containers()` (not earlier from `docker_lib`) and disappear at `orch.teardown_containers()`. Container is launched by orch, not by VllmJob.
5. `python -m pytest cvs/tests/inference/vllm/vllm_single.py --config_file=<missing-model-dir>/config.json --cluster_file=...` where `model.remote=0` and `models_dir/<id>` does NOT exist — expect a clean assertion failure naming the missing path, not a runtime crash deep in VllmJob.
6. `python -c "from cvs.lib.dtni.config_loader import load_variant; load_variant('<config with typo: percentiles_metrics>', {})"` — expect Pydantic `ValidationError` mentioning the unknown field. (Catches the v1 typo class the spec called out.)

## Open questions

1. Cluster file path for verification 1–4 — which existing mi355x cluster file should the PoC default to?
2. `test_cleanup_stale_containers` + `test_launch_inference_containers` — accept that they vanish from the HTML report (fixture lifecycle replaces them), or keep 5-line stub tests that re-check orch state to preserve the report shape?
3. New Job class name — `cvs.lib.inference.vllm_orch.VllmJob` (same class name, different module) or `VllmOrchJob` (clearly different during transition)?
