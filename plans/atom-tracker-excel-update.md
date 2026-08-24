# Multi-Node Validation – ATOM tracker — Excel update reference

**Branch:** `hnimrama/atom-accuracy` (through `b41a0c49`, rebased on `main`)  
**Scope:** CVS `cvs run atom` only — not InferenceX runner. Cross-engine compare uses `driver=vllm_atom` / `driver=sglang` inside the atom suite.  
**Last updated:** 2026-08-18 (automation pass — mark wired rows **Completed**; lab notes in Comments)

Use this file when updating the **Excel Online** tracker. Columns mirror the live sheet:


| #   | Category | Test / Metric | Priority | Automation Status | Comments |
| --- | -------- | ------------- | -------- | ----------------- | -------- |


**Status key**


| Value           | Meaning                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| **Completed**   | CVS config stem + harness wired (`cvs run atom`); add `lab pending` in Comments until smoke passes         |
| **In Progress** | Partial automation only — stem missing, blocked dataset, or parser not implemented                         |
| *(blank)*       | Not started — no atom stem yet                                                                             |


**Default for this sheet:** if a row has a shipped config + test hook, mark **Completed** and note lab/threshold status in Comments. After a passing smoke, replace `lab pending` with the lab date + score.

**P1-first rule (lab):** Run P1 lab smokes before spending cluster time on P2 workloads. Automation status is independent — P2 rows can be **Completed** when stems are shipped.

---

## Lab validation log (MI300X jumphost — 2026-08-18)


| Run                     | Config                                 | Result     | Notes                                                                                                              |
| ----------------------- | -------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------ |
| W1 perf smoke           | `mi300x_atom_deepseek-r1_fp8_single`   | **PASS**   | `11 passed, 1 skipped` in ~14 min; `w1_1k_1k-conc128` 1000/1000; image `rocm/atom-dev:latest`; node `10.32.80.112` |
| W1 accuracy (attempt 1) | `mi300x_atom_deepseek-r1_fp8_accuracy` | **FAIL**   | `test_openai_compatible_smoke` — DeepSeek R1 returns `reasoning_content`, probe expected `content`                 |
| Probe fix (R1 content)  | `c4877d6d`                             | **merged** | `OpenAIProbe` accepts `reasoning_content` when `content` empty                                                     |
| Probe fix (fenced JSON) | `36497404`                             | **merged** | FUNC-1 structured book probe strips `` ```json `` fences; rerun accuracy smoke after `make install`                |
| lm-eval 422 fix         | `b41a0c49`                             | **merged** | `tokenized_requests=False` in accuracy lm-eval cmd — stops token-id arrays hitting `/v1/completions`               |


**Jumphost sync before accuracy rerun:**

```bash
cd ~/cvs && git fetch origin && git checkout hnimrama/atom-accuracy && git pull
make install && source .cvs_venv/bin/activate
```

**Lab order (remaining):**

1. ~~W1 perf smoke~~ → **done** (closes **#3**, **#24–27**; partial **#33**)
2. **Next:** W1 accuracy gsm8k smoke (`gsm8k_flex`, pull `36497404`) → lab proof for **#45** / FUNC-1/2 on **#3** (rows already **Completed** for automation)
3. Full W1 accuracy (all lm-eval tasks) → calibrate **#40**, **#46**
4. M4 triple: `_single` / `_vllm_single` / `_sglang_single` → **#1**, **#2**
5. P1 workloads **#18**, **#7**, **#8**, then MI355X **#6**

---

## P1 lab sprint (target ≥80% lab-proven on MI300X)

**Scoreboard:** 18 P1 Excel rows — **8 lab-proven** today (#3, #24–27, #29, #31 from W1 perf). **10 remaining** on MI300X (+ **#6** MI355X needs different hardware).

| P1 # | Item | Lab status | Sprint run | Unblocks |
| ---- | ---- | ---------- | ---------- | -------- |
| 3 | ATOM native | partial | **A** | FUNC-1/2 Comments (perf done) |
| 24–27 | Core perf | **done** | — | — |
| 29, 31 | E2E, goodput | **done** | — | — |
| 45 | GSM8K | pending | **A** | ACC-1 gate |
| 40 | MMLU-PRO | pending | **A′** | baseline score |
| 46 | HellaSwag | pending | **A′** | baseline score |
| 1 | vLLM parity | pending | **B** | M4 compare panel inputs |
| 2 | SGLang parity | pending | **B** | M4 compare panel inputs |
| 4 | ATOM + MTP | pending | **C** | MTP info metrics |
| 18 | Kimi K2.7 Code | pending | **D** | W13 perf smoke |
| 7 | GPT-oss-120b | pending | **E** | W2 perf smoke |
| 8 | Qwen3.5 FP8 | pending | **F** | W3 perf smoke |
| 10 | DeepSeek V4 Pro | pending | **G** | atom longctx path first |
| 6 | R1 MI355X | pending | **H** | needs MI355X node |

**After sprint A+A′:** 11/18 P1 lab-proven (61%). **After B–G on MI300X:** 17/18 (94%) — only **#6** MI355X left.

### Lab container images (MI300X jumphost — confirmed 2026-08-18)

| Driver | `container.image` | Config stems |
| ------ | ----------------- | ------------ |
| `atom` | `amdaccelcloud/atom:10.0.0rc0-pytorch2.11` | `…_single`, `…_accuracy`, `…_mtp3`, W2/W3/W13 atom paths |
| `vllm_atom` | `rocm/ufb-private:vllm-0.23.0-ubuntu24.04-py3.14-prereleases-device-all-cdna-rocm7.14.0rc3-0fc695fc6-sshd` | `…_vllm_single`, `…_distributed` |
| `sglang` | `rocm/ufb-private:sglang-v0.5.12.post1-ubuntu24.04-py3.14-prereleases-device-all-rocm7.14.0rc0-497dd42881` | `…_sglang_single`, `…_sglang_distributed` |

`rocm/ufb-private` pulls need `registry` in `atom_cluster.json` (see `cvs/input/cluster_file/README.md`).

**One-time patch** (jumphost copied configs under `~/input/…`; requires `jq`):

```bash
ATOM_IMG='amdaccelcloud/atom:10.0.0rc0-pytorch2.11'
VLLM_IMG='rocm/ufb-private:vllm-0.23.0-ubuntu24.04-py3.14-prereleases-device-all-cdna-rocm7.14.0rc3-0fc695fc6-sshd'
SGLANG_IMG='rocm/ufb-private:sglang-v0.5.12.post1-ubuntu24.04-py3.14-prereleases-device-all-rocm7.14.0rc0-497dd42881'

patch_img() { jq --arg img "$2" '.container.image = $img' "$1" > "$1.tmp" && mv "$1.tmp" "$1"; }

patch_img ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_single.json "$ATOM_IMG"
patch_img ~/input/config_file/inference/atom/accuracy/mi300x_atom_deepseek-r1_fp8_accuracy.json "$ATOM_IMG"
patch_img ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_mtp3.json "$ATOM_IMG"
patch_img ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_vllm_single.json "$VLLM_IMG"
patch_img ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_sglang_single.json "$SGLANG_IMG"
```

### Prereq (every session)

```bash
cd ~/cvs
git fetch origin && git checkout hnimrama/atom-accuracy && git pull
make install && source .cvs_venv/bin/activate
export TS=$(date +%Y%m%d_%H%M%S)
export CLUSTER=~/input/cluster_file/atom_cluster.json
export RESULTS=~/cvs_results
mkdir -p "$RESULTS"
```

### Run A — W1 accuracy gsm8k (+ FUNC-1/2) · ~2–4 h · do first

Validates probe fixes; closes **#45** and FUNC-1/2 lab proof on **#3**.

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/accuracy/mi300x_atom_deepseek-r1_fp8_accuracy.json \
  -k "test_launch_container or test_setup_sshd or test_discover_topology or test_model_fetch or test_openai_compatible_smoke or test_server_health or acc_warmup-conc1 or gsm8k_flex or test_teardown" \
  --html="$RESULTS/${TS}_w1-acc-gsm8k.html" --self-contained-html \
  --log-file="$RESULTS/${TS}_w1-acc-gsm8k.log" \
  --maxfail=1 -vvv -s
```

### Run A′ — W1 full accuracy (same config, all lm-eval) · ~4–8 h · chain after A passes

One job; server already warm from A if you skip teardown between runs — otherwise single invocation:

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/accuracy/mi300x_atom_deepseek-r1_fp8_accuracy.json \
  -k "test_launch_container or test_setup_sshd or test_discover_topology or test_model_fetch or test_openai_compatible_smoke or test_server_health or acc_warmup-conc1 or test_accuracy_eval or test_atom_quant_parity or test_teardown" \
  --html="$RESULTS/${TS}_w1-acc-full.html" --self-contained-html \
  --log-file="$RESULTS/${TS}_w1-acc-full.log" \
  --maxfail=1 -vvv -s
```

Closes **#40**, **#46**; paste scores into Comments; matrix W1 GSM8K / MMLU-PRO / HellaSwag → **Y/A**.

### Run B — M4 vLLM + SGLang perf smokes · ~30 min each · images above

```bash
for DRIVER in vllm sglang; do
  cvs run atom \
    --cluster_file "$CLUSTER" \
    --config_file ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_${DRIVER}_single.json \
    -k "w1_1k_1k-conc128" \
    --html="$RESULTS/${TS}_w1-${DRIVER}-conc128.html" --self-contained-html \
    --log-file="$RESULTS/${TS}_w1-${DRIVER}-conc128.log" \
    -vvv -s
done
```

Closes **#1**, **#2** (lab proof; parity panel is render-only).

### Run C — W1 MTP · ~20 min

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-r1_fp8_mtp3.json \
  -k "w1_1k_1k-conc128" \
  --html="$RESULTS/${TS}_w1-mtp3-conc128.html" --self-contained-html \
  -vvv -s
```

Closes **#4**.

### Run D — Kimi K2.7 Code perf · ~20 min · P1 workload priority

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/single/mi355x_atom_kimi-k2.7-code_single.json \
  -k "w13_1k_1k-conc128" \
  --html="$RESULTS/${TS}_w13-conc128.html" --self-contained-html \
  -vvv -s
```

Closes **#18** perf path (ACC-12 NIAH = separate `…_code_accuracy` run if needed).

### Run E — GPT-oss-120b · ~30 min

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/single/mi300x_atom_gpt-oss-120b_mxfp4_single.json \
  -k "w2_8k_1k-conc32" \
  --html="$RESULTS/${TS}_w2-conc32.html" --self-contained-html \
  -vvv -s
```

Closes **#7**.

### Run F — Qwen3.5 FP8 · ~30 min

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/single/mi300x_atom_qwen3.5-397b-a17b_fp8_single.json \
  -k "qwen_1k_8k-conc32" \
  --html="$RESULTS/${TS}_w3-conc32.html" --self-contained-html \
  -vvv -s
```

Closes **#8**.

### Run G — DeepSeek V4 Pro (atom longctx) · ~30 min · model must be cached

```bash
cvs run atom \
  --cluster_file "$CLUSTER" \
  --config_file ~/input/config_file/inference/atom/single/mi300x_atom_deepseek-v4-pro_longctx_single.json \
  -k "v4pro_5k_1k-conc16" \
  --html="$RESULTS/${TS}_w5-v4pro-conc16.html" --self-contained-html \
  -vvv -s
```

Closes **#10** atom path; vLLM/SGLang V4 stems still need V4-capable images (optional follow-up).

### Run H — MI355X R1 · hardware gate

`mi355x_atom_deepseek-r1_fp8_single` on MI355X cluster file; closes **#6** MI355X half.

### After each pass — update tracker

1. Excel Comments: `lab validated YYYY-MM-DD` + key metric/score  
2. Matrix: flip cell **Y/P → Y/A** for that workload/metric  
3. Accuracy thresholds: if gsm8k/mmlu_pro/hellaswag pass, flip `"kind": "info"` → `"kind": "min"` in threshold JSON (separate commit)

**Fastest path tonight:** **A → A′** (if A passes) gets you to **11/18**. Queue **B** while accuracy runs if someone can set vLLM/SGLang images in parallel.

---

## ATOM – FRAMEWORK PATHS (cross-compare) (#1–5)


| #   | Category  | Test / Metric                                                                     | Priority | Automation Status | Comments                                                                                                                                                                                            |
| --- | --------- | --------------------------------------------------------------------------------- | -------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | ATOM Path | vLLM (ROCm) – baseline engine for InferenceX parity and cross-compare cards       | P1       | **Completed**     | CVS M4: `mi300x_atom_deepseek-r1_fp8_vllm_single` (`driver=vllm_atom`); W2: `…_gpt-oss-120b_mxfp4_vllm_single`. Run-deck parity panel (`CVS_ATOM_PARITY_REF_JSON`). **Lab pending** (compare vs #3). |
| 2   | ATOM Path | SGLang (ROCm) – baseline engine for InferenceX parity and cross-compare cards     | P1       | **Completed**     | CVS M4: `mi300x_atom_deepseek-r1_fp8_sglang_single`; W2: `…_gpt-oss-120b_mxfp4_sglang_single`. Same workload cards as ATOM where comparable. **Lab pending.**                                       |
| 3   | ATOM Path | ATOM – InferenceX framework: atom (AiTer Optimized Model serving path)            | P1       | **Completed**     | `driver=atom`, `mi300x_atom_deepseek-r1_fp8_single`. **Lab perf smoke passed 2026-08-18** (1000/1000 @ conc128). FUNC-1/2 lab proof pending accuracy rerun (probe fixes `c4877d6d` + `36497404`). INF-6 dmesg opt-in. |
| 4   | ATOM Path | ATOM + MTP – InferenceX *-atom-mtp recipes (speculative / multi-token prediction) | P1       | **Completed**     | `mi300x_atom_deepseek-r1_fp8_mtp3`, `…_mtp3_accuracy`; `test_atom_mtp_quality` (ACC-4/5/13). MTP thresholds `"kind": "info"` until lab. Chat-template required. **Lab pending.**                  |
| 5   | ATOM Path | ATOM-Disagg – InferenceX framework: atom-disagg (disaggregated prefill/decode)    | P2       | *(blank)*         | Not in atom suite — use SGLang disagg path. Wide EP + custom comms kernels.                                                                                                                         |


---

## WORKLOAD COVERAGE – ATOM recipes (#6–23)


| #   | Category | Test / Metric                                                                | Priority | Automation Status | Comments                                                                                                                                                                                    |
| --- | -------- | ---------------------------------------------------------------------------- | -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6   | Workload | DeepSeek R1 FP8 | MI355X | InferenceX framework: atom | dsr1-fp8-mi355x-atom | P1       | **Completed**     | CVS: `mi{300,355}x_atom_deepseek-r1_fp8_{single,accuracy,distributed,mtp3,baseline_sweep}`. W1 **1K/1K** TP8. MI355X `enforce_thresholds: false` until lab confirm. **Lab pending** (MI300X perf done). |
| 7   | Workload | GPT-oss-120b – openai/gpt-oss-120b, TP=4 (MXFP4)                             | P1       | **Completed**     | `mi{300,355}x_atom_gpt-oss-120b_mxfp4_{single,accuracy}`; M4 `…_vllm_single` / `…_sglang_single`. **8K/1K** TP4 MXFP4. **Lab pending.**                                                         |
| 8   | Workload | Qwen3.5-397B-A17B-FP8 – amd/Qwen3.5-397B-A17B-FP8, TP=8                      | P1       | **Completed**     | `mi{300,355}x_atom_qwen3.5-397b-a17b_fp8_single` + thresholds. **1K/8K** TP8. **Lab pending.**                                                                                                  |
| 9   | Workload | GLM 5.1 – zai-org/GLM-5.1-FP8, TP=8 (FP8)                                    | P2       | **Completed**     | `mi{300,355}x_atom_glm-5.1_{single,accuracy}`. **Lab pending** — defer lab until P1 smokes closed.                                                                                              |
| 10  | Workload | DeepSeek V4 Pro – deepseek-ai/DeepSeek-V4-Pro, TP=8 (FP4+FP8)                | P1       | **Completed**     | Automation: `mi{300,355}x_atom_deepseek-v4-pro_{longctx_single,vllm_single,sglang_single,distributed}`; 5000/1024; V4 vLLM DSv4 recipe. **Lab pending** (vLLM path needs V4-capable image). |
| 11  | Workload | DeepSeek V4 Flash – deepseek-ai/DeepSeek-V4-Flash, TP=4 (FP4+FP8)            | P2       | **Completed**     | `mi{300,355}x_atom_deepseek-v4-flash_single`. **Lab pending.**                                                                                                                                  |
| 12  | Workload | Kimi K2.6 Thinking – uniquealexx/Kimi-K2.6-Thinking-200x, TP=4 (INT4)        | P2       | **Completed**     | `mi355x_atom_kimi-k2.6-thinking_{single,accuracy}`. **Lab pending.**                                                                                                                      |
| 13  | Workload | GLM 5.2 MXFP4 – amd/GLM-5.2-MXFP4, TP=8                                      | P2       | **Completed**     | `mi{300,355}x_atom_glm-5.2-mxfp4_single`. MI355X + ATOM + vLLM parity stems shipped. **Lab pending.**                                                                                            |
| 14  | Workload | Kimi K2.5 MXFP4 – amd/Kimi-K2.5-MXFP4, TP=4                                  | P2       | **Completed**     | `mi355x_atom_kimi-k2.5-mxfp4_single` (MI355X only). **Lab pending.**                                                                                                                                    |
| 15  | Workload | Qwen 3.5 397B A17B – Qwen/Qwen3.5-397B-A17B, TP=8 (FP8)                      | P2       | **Completed**     | `mi{300,355}x_atom_qwen3.5-397b-a17b_single` (BF16); distinct from #8 amd FP8 variant. **Lab pending.**                                                                                         |
| 16  | Workload | GLM 5.2 – zai-org/GLM-5.2-FP8, TP=8 (FP8)                                    | P2       | **Completed**     | `mi{300,355}x_atom_glm-5.2-fp8_single`. **Lab pending.**                                                                                                                                        |
| 17  | Workload | GLM 5.2 – zai-org/GLM-5.2, TP=8 (MXFP4)                                      | P2       | **Completed**     | `mi{300,355}x_atom_glm-5.2_single`. MI355X. **Lab pending.**                                                                                                                                    |
| 18  | Workload | Kimi-K2.7-Code – moonshotai/Kimi-K2.7-Code, TP=8 (MXFP4)                     | P1       | **Completed**     | **Highest P1 workload lab priority.** `mi355x_atom_kimi-k2.7-code_{single,longctx_single,code_accuracy}` (MI355X only). ACC-12 NIAH on accuracy stem. **Lab pending.**                                 |
| 19  | Workload | MiniMax-M3 – MiniMaxAI/MiniMax-M3 (BF16)                                     | P2       | **Completed**     | `mi{300,355}x_atom_minimax-m3_single`. **Lab pending.**                                                                                                                                         |
| 20  | Workload | Qwen3.5-397B-A17B-MXFP4 – amd/Qwen3.5-397B-A17B-MXFP4, TP=8                  | P2       | **Completed**     | `mi{300,355}x_atom_qwen3.5-397b-a17b-mxfp4_single`. **Lab pending.**                                                                                                                              |
| 21  | Workload | Mistral Large 3 – mistralai/Mistral-Large-3-675B-Instruct-2512, TP=8 (FP8)   | P2       | **Completed**     | `mi{300,355}x_atom_mistral-large-3_single`. **Lab pending.**                                                                                                                                    |
| 22  | Workload | DeepSeek-R1-0528-MXFP4 – amd/DeepSeek-R1-0528-MXFP4, TP=8 MXFP4              | P2       | **Completed**     | `mi{300,355}x_atom_deepseek-r1_mxfp4_{single,accuracy}`. MI355X. **Lab pending.**                                                                                                               |
| 23  | Workload | MiMo-v2.5-Pro – XiaomiMiMo/MiMo-V2.5-Pro, TP=8 (BF16)                        | P2       | **Completed**     | `mi{300,355}x_atom_mimo-v2.5-pro_single`. **Lab pending.**                                                                                                                                      |


---

## PERFORMANCE BENCHMARK METRICS (#24–37)


| #   | Category    | Test / Metric                                                           | Priority | Automation Status | Comments                                                                                                                 |
| --- | ----------- | ----------------------------------------------------------------------- | -------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 24  | Performance | Throughput per GPU – total tokens/sec (tput_per_gpu)                    | P1       | **Completed**     | **Lab validated 2026-08-18** — `client.per_gpu_throughput` ~6894 total tok/s cluster in W1 smoke artifact.               |
| 25  | Performance | Output throughput per GPU – decode tokens/sec/GPU (output_tput_per_gpu) | P1       | **Completed**     | **Lab validated 2026-08-18** — ~3445 output tok/s/GPU in W1 smoke.                                                       |
| 26  | Performance | TTFT – mean & p99 (ms)                                                  | P1       | **Completed**     | **Lab validated 2026-08-18** — mean TTFT ~564 ms, p99 ~4793 ms (W1 smoke).                                               |
| 27  | Performance | TPOT – mean & p95 (ms)                                                  | P1       | **Completed**     | **Lab validated 2026-08-18** — mean TPOT ~35 ms, p99 ~38 ms (W1 smoke).                                                  |
| 28  | Performance | Prefill latency – p50 / p95 (ms)                                        | P2       | *(blank)*         | Not emitted separately from TTFT on `driver=atom`.                                                                       |
| 29  | Performance | End-to-end latency – mean / p95 / p99 (ms)                              | P1       | **Completed**     | `client.mean_e2el_ms`, p95/p99 tails — emitted in artifact; gate optional.                                               |
| 30  | Performance | Latency vs load – P95 & P99 at each QPS/concurrency step                | P2       | **Completed**     | `*_baseline_sweep` (14 cells); sweep knee in report. **Lab pending** on W1 trim/full sweep.                                  |
| 31  | Performance | Goodput – successful requests / total requests under load               | P1       | **Completed**     | `client.goodput` when finite `request_rate`; record-only on W1 inf rate.                                                 |
| 32  | Performance | Scaling efficiency % – multi-GPU / multi-node vs 1× reference           | P2       | **Completed**     | `*_distributed`; `scaling.efficiency_pct`. IB/UE fabric in run card + `test_discover_topology`. **Lab pending.**            |
| 33  | Performance | Peak GPU memory – max allocated / reserved (MB)                         | P2       | **Completed**     | **INF-7:** `platform.gpu_metrics_poll` → `gpu.peak_gpu_memory_mb`. Emitted on W1 perf smoke; **threshold calibration pending**. |
| 34  | Performance | KV cache memory footprint (GB) at target batch × sequence               | P2       | *(blank)*         | No ATOM KV telemetry parser yet.                                                                                         |
| 35  | Performance | Request success rate & error mix                                        | P2       | **Completed**     | `client.success_rate`, `client.failed`. **INF-6:** optional `platform.dmesg_scan` → `test_verify_dmesg`.                 |
| 36  | Performance | Model load time (s) + load memory (MB) – cold start                     | P2       | **Completed**     | `gpu.model_load_s`, `gpu.model_load_memory_mb` when poll enabled; `server.model_cache_bytes`. **Lab pending.**               |
| 37  | Performance | Time-to-ready — server (container start → readiness regex match)        | P2       | **Completed**     | `server.time_to_ready_s` lifecycle metrics. **Lab pending.**                                                                 |


---

## OPTIONAL QUALITY GATES (#38–39)


| #   | Category | Test / Metric                                                                     | Priority | Automation Status | Comments                                                                                                  |
| --- | -------- | --------------------------------------------------------------------------------- | -------- | ----------------- | --------------------------------------------------------------------------------------------------------- |
| 38  | Quality  | MTP acceptance rate / degenerate-spec decode checks (chat-formatted prompts only) | P2       | **Completed**     | ACC-4/5/13 via `test_atom_mtp_quality` on `*_mtp3`; `"kind": "info"` until lab. W3/W4/W7/W15 MTP recipes. **Lab pending.** |
| 39  | Quality  | Quantization output parity – FP8/FP4 vs BF16 reference                            | P2       | **Completed**     | **ACC-7:** `quant_parity` + `test_atom_quant_parity` on W1 `*_accuracy`. BF16 reference pairing optional. **Lab pending.** |


---

## ACCURACY BENCHMARK METRICS (#40–50)


| #   | Category                | Test / Metric                                                    | Priority | Automation Status | Comments                                                                                                                                                          |
| --- | ----------------------- | ---------------------------------------------------------------- | -------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 40  | MMLU-PRO                | Refined MMLU, 10 choices; Accuracy % (5-shot)                    | P1       | **Completed**     | ACC-15; lm-eval `mmlu_pro` on W1 `mi300x_atom_deepseek-r1_fp8_accuracy`; info gate until lab → flip to min. **Lab pending.** |
| 41  | BBH                     | Big Bench Hard; Normalized accuracy % (3-shot)                   | P2       | **Completed**     | ACC-16; lm-eval `bbh` on W1 accuracy stem; info threshold. **Lab pending.**                                                    |
| 42  | MATH Level 5            | High-school competition math; Exact match % (4-shot)             | P2       | **Completed**     | ACC-11; `hendrycks_math` on W7 `…_thinking_accuracy` (Level 5 metadata). **Lab pending.**                                    |
| 43  | GPQA                    | Graduate-level science Q&A; Normalized accuracy % (0-shot)       | P2       | *(blank)*         | Gated HF dataset — not wired in CVS.                                                                                          |
| 44  | MuSR                    | Multistep soft reasoning; Normalized accuracy % (0-shot)         | P2       | **Completed**     | ACC-17; lm-eval `musr` on W1 accuracy stem; info threshold. **Lab pending.**                                                  |
| 45  | GSM8K                   | Grade-school math; Exact-match accuracy %                        | P1       | **Completed**     | ACC-1..3; lm-eval on `mi300x_atom_deepseek-r1_fp8_accuracy`; gate ≥ **0.94** flexible-extract. **Lab pending** (pull `36497404`, rerun accuracy smoke). |
| 46  | HellaSwag               | Commonsense sentence completion; Accuracy %                      | P1       | **Completed**     | ACC-14; lm-eval `hellaswag` 0-shot; info → min after first lab run. **Lab pending.**                                           |
| 47  | MMLU                    | Legacy MMLU, 57 subjects; Accuracy %                             | P2       | **Completed**     | ACC-6 scaffold on W3 `glm-5.1_accuracy`; prefer #40 MMLU-PRO for P1. **Lab pending.**                                        |
| 48  | ARC-Challenge           | AI2 Reasoning Challenge; Accuracy %                              | P2       | **Completed**     | ACC-18; lm-eval `arc_challenge` on W1 accuracy stem. **Lab pending.**                                                        |
| 49  | WinoGrande              | Commonsense pronoun resolution; Accuracy %                       | P2       | **Completed**     | ACC-19; lm-eval `winogrande` on W1 accuracy stem. **Lab pending.**                                                           |
| 50  | Scale Parity – Accuracy | Same model/weights → scores match 1 GPU / multi-GPU / multi-node | P2       | **Completed**     | `mi300x_atom_deepseek-r1_fp8_distributed_accuracy`; `CVS_ATOM_SCALE_ACCURACY_REF_JSON` / scale_accuracy report panel. **Lab pending.** |


---

## INFRA / CROSS-CUTTING (not numbered in Excel — map to Comments)

These items **do not get their own rows** in the Excel sheet. Copy the text into **Comments** on the row listed.


| Item                           | Priority | Automation Status | Map to Excel row # | Comments (paste into sheet)                                                                                                         |
| ------------------------------ | -------- | ----------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| INF-6 time-bounded dmesg scan  | —        | **Completed**     | **#3**, **#35**    | `platform.dmesg_scan: true` → `test_verify_dmesg` (opt-in)                                                                          |
| INF-7 GPU metrics poll         | —        | **Completed**     | **#33**            | `platform.gpu_metrics_poll` wired; W1 perf smoke emitted `gpu.peak_gpu_memory_mb`. **Threshold calibration pending.**         |
| Framework parity report panels | —        | **Completed**     | **#1**, **#2**     | M4 run-deck: `CVS_ATOM_PARITY_REF_JSON`; `compare.vllm.`* / `compare.sglang.*` panels (render-only)                         |
| FUNC-1 API smoke               | —        | **Completed**     | **#3**, **#45**    | `test_openai_compatible_smoke`; probe fixes `c4877d6d` + `36497404` for DeepSeek R1. **Lab pass pending rerun.**              |
| FUNC-2 health check            | —        | **Completed**     | **#3**, **#45**    | `functional.health_check: true` → `test_server_health` (/health, model list, max_tokens=1). **Lab pending** with accuracy run. |
| ACC-7 quant parity probe       | —        | **Completed**     | **#39**            | `quant_parity` block + `test_atom_quant_parity` on W1 accuracy variant. **Lab pending.**                                      |
| ACC-12 NIAH long-context       | —        | **Completed**     | **#18**            | `…_code_accuracy` + `test_atom_long_context_accuracy`. **Lab pending.**                                                       |


---

## P1 closure checklist (flip to **Completed** in Excel)


| Step | Lab run                                                                 | Status                           | Rows to close                                      |
| ---- | ----------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------- |
| 1    | Perf smoke: `mi300x_atom_deepseek-r1_fp8_single`, `-k w1_1k_1k-conc128` | **Done**                         | **#3**, **#24–27**; partial **#33**                |
| 2    | Accuracy gsm8k: `mi300x_atom_deepseek-r1_fp8_accuracy`, `-k gsm8k_flex` | **Next** (pull `36497404`) | **#45** lab proof; FUNC-1/2 lab proof on **#3** Comments (automation **Completed**) |
| 3    | Full W1 accuracy (all lm-eval tasks)                                    | Pending                          | Calibrate **#40**, **#46** (flip info → min gates) |
| 4    | M4 triple: `_single` / `_vllm_single` / `_sglang_single`                | Pending                          | **#1**, **#2**                                     |
| 5    | Kimi K2.7 lab                                                           | Pending                          | **#18**                                            |
| 6    | GPT-oss / Qwen / MI355X R1                                              | Pending                          | **#7**, **#8**, **#6**                             |


**Accuracy smoke command (after `make install`):**

```bash
cvs run atom \
  --cluster_file ~/input/cluster_file/atom_cluster.json \
  --config_file ~/input/config_file/inference/atom/accuracy/mi300x_atom_deepseek-r1_fp8_accuracy.json \
  -k "test_launch_container or test_setup_sshd or test_discover_topology or test_model_fetch or test_openai_compatible_smoke or test_server_health or acc_warmup-conc1 or gsm8k_flex or test_teardown" \
  --maxfail=1 -vvv -s
```

---

## Summary counts (for Excel footer — 2026-08-18 automation pass)

**Framework + metric rows (#1–50):**


| Status                  | Count (#1–50) | Notes                                                                 |
| ----------------------- | ------------- | --------------------------------------------------------------------- |
| **Completed**           | 46            | All wired stems + test hooks; #24–27 also **lab validated** on MI300X   |
| **In Progress**         | 0             | —                                                                     |
| **Blank / Not started** | 4             | #5 (disagg), #28 (prefill), #34 (KV cache), #43 (GPQA gated HF)       |


**P1 rows only (18 total in Excel):** #1–4, #6–8, #10, #18, #24–27, #29, #31, #40, #45–46


| P1 metric                               | Value                                                                 |
| --------------------------------------- | --------------------------------------------------------------------- |
| # of P1                                 | 18                                                                    |
| # of P1 **Completed** (automation)      | **18** (100%)                                                         |
| # of P1 **lab validated**               | **8** (#3, #24–27, #29, #31) — rest **lab pending** in Comments       |
| % of P1 automation closed               | **100%**                                                              |
| % of P1 lab closed (target ≥80%)        | **44%** now → gsm8k + M4 triple + Kimi K2.7 lab runs raise this       |


*Recalculate footer after each lab milestone. Paste lab date + pass/fail into **Comments** when flipping rows.*

---

## Changelog


| Date       | Summary                                                                                                                                 |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-18 | Automation pass: flip wired rows **Completed** (46/50); **In Progress** only for unwired (#5, #28, #34, #43); P1 automation 18/18        |
| 2026-08-18 | Atom Matrix: Excel ↔ Y/A/Y/P crosswalk; **Y/P** = automation **Completed** (lab pending)                                                  |
| 2026-08-18 | Post jumphost lab: W1 perf smoke **PASS** → **#3**, **#24–27** Completed; accuracy FUNC-1 fail + probe fixes noted                      |
| 2026-08-18 | V4 Pro (#10) automation complete: longctx + vllm + sglang + distributed stems (mi300x/mi355x); orch V4 recipe guards                    |
| 2026-08-18 | Reconcile with live Excel sheet; P1-first statuses; INFRA → row mapping; fix #6 row; V4 Pro as P1 per tracker                           |
| 2026-08-18 | Added **Atom Matrix** — workload × benchmark coverage grid (Y/A vs Y/P vs --)                                                           |
| 2026-08-11 | Initial cheat sheet (`357a82c8`)                                                                                                        |


---

## Atom Matrix

Workload × benchmark coverage for CVS `cvs run atom`. Updated **2026-08-18** (automation pass — aligned with Excel **Completed** rows above).

**Cell codes** (matrix ≠ Excel Automation Status — use this crosswalk when copying to Excel):


| Code    | Meaning                                                                         | Excel Automation Status equivalent                          |
| ------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Y/A** | CVS harness wired **and** lab validated on MI300X                               | **Completed** + lab date/score in Comments                  |
| **Y/P** | CVS harness wired; lab pending or threshold calibration open                    | **Completed** + `lab pending` / `threshold calibration pending` in Comments |
| **--**  | Not in CVS for this workload / metric (see [No coverage](#no-coverage-reason) ) | *(blank)* or **In Progress** only if partial wiring         |


**Matrix roll-up (performance + quality + accuracy grids):** **Y/A** = W1 perf core only (8 metric rows × 1 workload); **Y/P** = all other wired cells; **--** = unwired by design (GPQA, KV cache, prefill, etc.).


### Workload legend (InferenceX / ATOM)


| Code | Workload                                               | CVS stem (mi300x)                                                          | P   |
| ---- | ------------------------------------------------------ | -------------------------------------------------------------------------- | --- |
| W1   | DeepSeek R1 FP8 | MI355X | atom | dsr1-fp8-mi355x-atom | `…_deepseek-r1_fp8_{single,accuracy,distributed,baseline_sweep,…}`         | P1  |
| W2   | GPT-oss-120b MXFP4 TP4                                 | `…_gpt-oss-120b_mxfp4_{single,accuracy,vllm_single,sglang_single}`         | P1  |
| W3   | Qwen3.5-397B-A17B-FP8 TP8                              | `…_qwen3.5-397b-a17b_fp8_single`                                           | P1  |
| W4   | GLM 5.1 FP8 TP8                                        | `…_glm-5.1_{single,accuracy}`                                              | P2  |
| W5   | DeepSeek V4 Pro FP4+FP8 TP8                            | `…_deepseek-v4-pro_{longctx_single,vllm_single,sglang_single,distributed}` | P1  |
| W6   | DeepSeek V4 Flash FP4+FP8 TP4                          | `…_deepseek-v4-flash_single`                                               | P2  |
| W7   | Kimi K2.6 Thinking INT4 TP4                            | `…_kimi-k2.6-thinking_{single,accuracy}`                                   | P2  |
| W8   | GLM 5.2 MXFP4 TP8                                      | `…_glm-5.2-mxfp4_single`                                                   | P2  |
| W9   | Kimi K2.5 MXFP4 TP4                                    | `…_kimi-k2.5-mxfp4_single`                                                 | P2  |
| W10  | Qwen3.5-397B-A17B BF16 TP8                             | `…_qwen3.5-397b-a17b_single`                                               | P2  |
| W11  | GLM 5.2 FP8 TP8                                        | `…_glm-5.2-fp8_single`                                                     | P2  |
| W12  | GLM 5.2 MXFP4 TP8                                      | `…_glm-5.2_single`                                                         | P2  |
| W13  | Kimi-K2.7-Code MXFP4 TP8                               | `…_kimi-k2.7-code_{single,longctx_single,code_accuracy}`                   | P1  |
| W14  | MiniMax-M3 BF16                                        | `…_minimax-m3_single`                                                      | P2  |
| W15  | Qwen3.5-397B-A17B-MXFP4 TP8                            | `…_qwen3.5-397b-a17b-mxfp4_single`                                         | P2  |
| W16  | Mistral Large 3 FP8 TP8                                | `…_mistral-large-3_single`                                                 | P2  |
| W17  | DeepSeek-R1-0528-MXFP4 TP8                             | `…_deepseek-r1_mxfp4_{single,accuracy}`                                    | P2  |
| W18  | MiMo-v2.5-Pro BF16 TP8                                 | `…_mimo-v2.5-pro_single`                                                   | P2  |


### Performance metrics


| Benchmark Workload             | W1      | W2  | W3  | W4  | W5  | W6  | W7  | W8  | W9  | W10 | W11 | W12 | W13 | W14 | W15 | W16 | W17 | W18 |
| ------------------------------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Throughput / GPU (total tok/s) | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| TTFT (Time to First Token)     | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| TPOT (Time Per Output Token)   | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Prefill Latency                | --      | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| Normalized TTFT                | Y/P     | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| P99/P50 Decode Latency Ratio   | Y/P     | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Queue Wait Time                | --      | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| End-to-End Request Latency     | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Latency vs Load Curve          | Y/P     | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| Global Throughput              | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Decode Throughput              | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Max Concurrent Requests        | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Scaling Efficiency %           | Y/P     | --  | --  | --  | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| Peak GPU Memory                | Y/P     | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| KV Cache Memory Footprint      | --      | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| Model Load Time + Memory       | Y/P     | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| GPU Memory Bandwidth Util %    | --      | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| GPU Compute Util % (MFU)       | --      | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| Goodput                        | **Y/A** | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P | Y/P |
| Quantization Output Parity     | Y/P     | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |


*W1 **Y/A** = `mi300x_atom_deepseek-r1_fp8_single`, `-k w1_1k_1k-conc128`, 2026-08-18 jumphost (1000/1000 @ conc128). All other **Y/P** = stem + gate wired on branch `36497404`; no MI300X lab run yet.*

### Optional quality


| Benchmark Workload           | W1  | W2  | W3  | W4  | W5  | W6  | W7  | W8  | W9  | W10 | W11 | W12 | W13 | W14 | W15 | W16 | W17 | W18 |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Quant / logit parity vs BF16 | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |


### Accuracy benchmarks


| Benchmark Workload      | W1  | W2  | W3  | W4  | W5  | W6  | W7  | W8  | W9  | W10 | W11 | W12 | W13 | W14 | W15 | W16 | W17 | W18 |
| ----------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMLU-PRO                | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| BBH (Big Bench Hard)    | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| MATH Level 5            | --  | --  | --  | --  | --  | --  | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| GPQA                    | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| MuSR                    | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| GSM8K                   | Y/P | Y/P | --  | Y/P | --  | --  | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | Y/P | --  |
| HellaSwag               | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| MMLU                    | --  | --  | --  | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| ARC-Challenge           | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| WinoGrande              | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |
| Scale Parity – Accuracy | Y/P | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  | --  |


*W1 GSM8K / FUNC-1: automation **Completed**; accuracy smoke failed 2026-08-18 (`reasoning_content` + fenced JSON); fixes shipped (`c4877d6d`, `36497404`) — **lab pending**. W13 uses HumanEval/MBPP on `…_code_accuracy` (not rows above).*

### No coverage reason


| Metric / benchmark                  | Why `--`                                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| Prefill Latency                     | Not emitted separately from TTFT on `driver=atom`                                        |
| Queue Wait Time                     | No client queue telemetry in bench artifact                                              |
| KV Cache Memory Footprint           | No ATOM KV telemetry parser in CVS                                                       |
| GPU Memory Bandwidth / MFU          | Not collected by atom bench client                                                       |
| Latency vs Load (W2–W18)            | Only W1 ships `*_baseline_sweep` today                                                   |
| Scaling Efficiency (most workloads) | Requires `*_distributed` stem + 2-node lab; only W1/W5 wired                             |
| Peak GPU Memory (W2–W18)            | `platform.gpu_metrics_poll` enabled on W1 perf path only so far                          |
| Quant parity (W2–W18)               | `quant_parity` block on W1 `*_accuracy` only                                             |
| GPQA (all)                          | Gated HF dataset — not wired in CVS                                                      |
| MMLU-PRO / BBH / … (most workloads) | lm-eval tasks live on W1 `*_accuracy` (and W4 MMLU, W7 MATH) — perf-only stems elsewhere |
| W5 accuracy columns                 | No V4 Pro `*_accuracy` stem yet                                                          |
| W13 standard accuracy rows          | Code workload uses HumanEval/MBPP + ACC-12 NIAH, not gsm8k matrix                        |


---

## Related repo docs

- [atom-workload-tracker.md](atom-workload-tracker.md) — CVS automation map
- [atom-accuracy-test-catalog.md](atom-accuracy-test-catalog.md) — ACC-* detail
- [cvs/input/config_file/inference/atom/README.md](../cvs/input/config_file/inference/atom/README.md) — config stem inventory + lab commands

