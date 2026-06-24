# PR #238 — InferenceX ATOM W1 (MI300X perf gates)

**PR:** https://github.com/ROCm/cvs/pull/238  
**Head:** `hnimrama/IX-atom` → **Base:** `dev/dtni`

---

## Motivation

CVS needs a first-class **InferenceX ATOM** automation path aligned with the **DTNI Validation Tracker (IX ATOM)** — not the legacy `inferencemax_single` uplift. This PR:

- Introduces the **`inferencex_atom_single`** suite with an ATOM-native driver (`atom.entrypoints.openai_server` + `atom.benchmarks.benchmark_serving`).
- Ships **W1 DeepSeek R1 FP8** variant configs for **MI300X** and **MI355X** (perf, smoke, MTP3) with calibrated / CI-seeded thresholds.
- Closes **M1 / Phase A** on MI300X: W1 perf with `enforce_thresholds: true` after lab confirmation.
- Removes legacy `inferencemax/` configs and documents the full IX-atom roadmap (W1–W18, accuracy, metrics tiers, parity frameworks).

**MI355X** variant dirs and CI-seeded thresholds ship with `enforce_thresholds: false` until hardware is available — they do not block merge or MI300X milestone work.

**Base branch:** `dev/dtni`.

---

## Technical Details

### Suite and orchestration

- Rename **`inferencemax_single` → `inferencex_atom_single`**; legacy configs removed.
- New **`InferenceXAtomJob`** (`inferencex_atom_orch.py`):
  - `params.driver=atom` → ATOM server + `benchmark_serving`; JSON artifacts parsed via `to_client_metrics`.
  - `params.driver=vllm` retained for interim uplift variants only.
- Canonical config layout: `cvs/input/config_file/inference/inferencex_atom_single/<variant>/` with `schema_version: 1`, typed loader, and `ix_recipes.json` recipe pins.
- Cluster examples: `mi300x_atom_single.json`, `mi355x_atom_single.json`; container names pinned (`inferencex_atom_mi300x` / `inferencex_atom_mi355x`).

### W1 variants shipped

| Variant | Arch | `enforce_thresholds` | Notes |
|---------|------|----------------------|-------|
| `deepseek_r1_fp8_mi300x_atom_perf` | MI300X | `true` | M1 gate — ISL=OSL=1024, CONC=128/256, 1000 prompts |
| `deepseek_r1_fp8_mi300x_atom_smoke` | MI300X | `false` | 128-prompt pre-gate |
| `deepseek_r1_fp8_mi300x_atom_mtp3` | MI300X | `false` | MTP3 recipe |
| `deepseek_r1_fp8_mi355x_atom_perf` | MI355X | `false` | CI-seeded thresholds (plan Section 4.3) |
| `deepseek_r1_fp8_mi355x_atom_mtp3` | MI355X | `false` | CI-seeded |

### Threshold / metrics plumbing

- **`to_client_metrics`**: derive `client.failed` when ATOM omits it (`num_prompts - completed`), then compute `client.success_rate`.
- **`test_metric`**: skip enforcement when a gated metric is absent from the artifact (p90/p95 placeholders stay record-only; W1 benchmark emits p99).
- MI300X W1 perf thresholds calibrated from lab run (throughput × 0.9, latency × 1.1 guard band).

### Platform / shared infra (supporting changes)

- `cvs/lib/utils/` — shared `config_loader`, `verdict`, sweep selector.
- `vllm_single` suite refactor to the same lifecycle / metric pattern.
- `plans/inferencex-atom-cvs-automation-plan.md` — implementation plan (branch state, lab policy, W1–W18 matrix, Section 12 extended coverage).
- Runbook: `cvs/input/config_file/inference/inferencex_atom_single/README.md`.

### Out of scope (follow-up on `dev/dtni`)

- M2 gsm8k accuracy (`deepseek_r1_fp8_mi300x_atom_accuracy`)
- M3 P1 workloads (W2, W3, W13, W17)
- M4 `inferencex_atom_vllm_single` / `inferencex_atom_sglang_single` parity frameworks

---

## Test Plan

### CI / unit (no GPU)

- [ ] `pytest cvs/tests/inference/inferencex_atom/ -q`
- [ ] `pytest cvs/tests/inference/vllm/test_vllm_orch_parse.py -q` (parser + `failed` / `success_rate` derivation)
- [ ] Config loader / sweep selector tests pass

### Lab — MI300X W1 perf (M1 gate)

**Prerequisites**

1. From repo root on the lab host: **`make install`** (builds sdist into `.cvs_venv/`). Do not rely on a stale `site-packages/cvs` from an older install.
2. Activate `.cvs_venv` (or invoke `cvs` from `.cvs_venv/bin/cvs`).
3. `cvs copy-config` for variant config + threshold + cluster; set cluster node IPs and `container.image`.
4. HF token at `paths.hf_token_file`.

```bash
make install
source .cvs_venv/bin/activate   # or equivalent

cvs copy-config inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_config.json \
  --output ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_config.json
cvs copy-config inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_threshold.json \
  --output ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_threshold.json

cvs run inferencex_atom_single \
  --cluster_file ~/input/cluster_file/mi300x_atom_single.json \
  --config_file ~/input/config_file/inference/inferencex_atom_single/deepseek_r1_fp8_mi300x_atom_perf/deepseek_r1_fp8_mi300x_atom_perf_config.json \
  --html=/tmp/inferencex_atom_w1_mi300x.html -vvv -s
```

- [ ] Lifecycle stages pass (sshd skip on single-node, model fetch, server start, benchmark client).
- [ ] Both sweep cells run: `CONC=128`, `CONC=256` (ISL=OSL=1024, TP=8).
- [ ] Gated metrics pass under `enforce_thresholds: true`.
- [ ] HTML report and logs attached to PR.

### Lab — smoke (optional)

- [ ] `deepseek_r1_fp8_mi300x_atom_smoke` — 128 prompts, record-only thresholds.

### MI355X

- [ ] Not required for merge — configs collect; flip `enforce_thresholds` when hardware is available.

---

## Test Result

### MI300X lab — `deepseek_r1_fp8_mi300x_atom_perf`

- **Branch:** `hnimrama/IX-atom` → **`dev/dtni`**
- **Install:** `make install` from repo root (`.cvs_venv`)
- **Image:** `rocm/atom-dev:latest`
- **Outcome:** **81 passed, 0 failed**

| Cell | `client.output_throughput` (measured) | Threshold (min) | Result |
|------|----------------------------------------|-----------------|--------|
| CONC=128 | ~2842 tok/s | 2590.98 tok/s | PASS |
| CONC=256 | ~4419 tok/s | 3942.77 tok/s | PASS |

- Mean TTFT / TPOT within calibrated max gates for both cells.
- Artifacts: attach HTML (`--html`) and log bundle.

### Unit tests

- [ ] CI / local unit suite green on PR branch.

---

## Submission Checklist

- [x] PR open: [#238](https://github.com/ROCm/cvs/pull/238)
- [ ] Base branch is **`dev/dtni`**
- [ ] Lab run used **`make install`** from this branch (not a stale global / site-packages install)
- [ ] MI300X W1 perf HTML report attached
- [ ] No secrets in logs (rotate HF token if it appeared in verbose capture)
- [ ] `enforce_thresholds: true` only on MI300X W1 perf — confirmed in lab
- [ ] MI355X variants left at `enforce_thresholds: false` (pending hardware)
- [ ] Plan doc reviewed for milestone alignment
- [ ] Reviewer aware M2 (gsm8k) and M3 (W2/W3/W13/W17) are follow-ups on `dev/dtni`
