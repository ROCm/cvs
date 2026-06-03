# Specification: Structured Metrics Parsing & Display for SGLang Disagg Inference Tests

Status: implemented
Owner: CVS sglang disaggregated-inference suite
Files: `cvs/lib/sglang_disagg_lib.py`, tests under `cvs/tests/inference/sglang/`

## 1. Problem

The disaggregated PD inference suite runs two benchmark tests against the
proxy router:

- `gsm8k` (`run_gsm8k_benchmark_test`)
- `bench_serv_random` (`benchserv_test_random`)

Their result reporting is inconsistent:

- `bench_serv_random` parses ~15 metrics into `self.inference_results_dict`
  and `verify_inference_results` compares them against thresholds.
- `gsm8k` greps a single `Output throughput` value inline, never stores it,
  and logs **nothing** on success (only the raw benchmark stdout is captured
  incidentally). There is no logged actual-vs-threshold verdict.

Result: for gsm8k the run log shows the benchmark's own stdout (Accuracy,
Latency, Output throughput) but no structured metrics and no pass/fail
evidence, so a reviewer cannot tell from the harness output what was measured
or how it compared to the gate.

## 2. Goals

1. Every benchmark test parses its raw stdout into a structured per-node
   metrics dict (`self.inference_results_dict`).
2. Every benchmark test emits a uniform, greppable, human-readable metrics
   block to the log.
3. For each metric that has a configured threshold, the log shows
   `actual`, the comparison direction, the `expected` threshold, and a
   `PASS`/`FAIL` verdict.
4. No change to pass/fail semantics: a test fails iff a thresholded metric
   violates its bound (or the benchmark produced no parseable result).

## 3. Metrics captured

`gsm8k` (keys chosen so the throughput key matches its threshold key
`tokens_per_sec`):

| key | source line | direction |
|---|---|---|
| `accuracy` | `Accuracy: <f>` | higher better (display only) |
| `invalid` | `Invalid: <f>` | lower better (display only) |
| `latency_s` | `Latency: <f> s` | lower better (display only) |
| `tokens_per_sec` | `Output throughput: <f> token/s` | higher better (gated) |

`bench_serv_random` (unchanged set; gated keys are
`output_throughput_per_sec`, `mean_ttft_ms`, `mean_tpot_ms`).

## 4. Direction rule

A metric is "lower is better" iff its name matches
`_ms$ | latency | ttft | tpot | itl | e2el` (case-insensitive); otherwise
"higher is better". This reproduces the existing `verify_inference_results`
behavior for all current gated keys (throughput → higher, `*_ms` latency →
lower) while being explicit and extensible.

## 5. Display format

One block per node, logged line-by-line at INFO so it survives `--log-file`:

```
================ METRICS [gsm8k] node=10.245.135.19 ================
  accuracy        = 0.940
  invalid         = 0.000
  latency_s       = 98.449
  tokens_per_sec  = 1017.889   [expected >= 1000]  PASS
==================================================================
```

- Only metrics present in the parsed dict are shown.
- The `[expected <op> <threshold>]  <verdict>` suffix appears only for metrics
  that have a configured threshold.
- Verdict is `PASS`/`FAIL`, or `UNPARSED` if the value can't be coerced to a
  float.

## 6. API (in `sglang_disagg_lib.py`)

Module-level (pure, unit-testable without a cluster):

- `parse_gsm8k_metrics(text) -> dict` — regex-extract gsm8k metrics from one
  node's stdout.
- `format_metrics_table(test_name, results_dict, expected_dict=None) -> str` —
  render the block above; computes per-metric verdict when `expected_dict`
  is given.
- `_is_lower_better(metric_name) -> bool` — the §4 direction rule.

Methods on `SglangDisaggPD`:

- `get_gsm8k_results_dict(out_dict)` — fill `self.inference_results_dict` from
  gsm8k stdout via `parse_gsm8k_metrics`.
- `log_metrics(test_name, expected_result_dict=None)` — log the formatted
  table for `self.inference_results_dict`.
- `verify_inference_results(test_name, expected_result_dict, check_dmesg=True)`
  — log the table, then per gated metric log
  `metric <name>: actual=<a> expected <op> <t> -> PASS/FAIL` and `fail_test`
  on violation; run dmesg checks only when `check_dmesg=True`.

## 7. Test wiring

- `run_gsm8k_benchmark_test`: keep the "no `Output throughput` => fail" guard,
  then `get_gsm8k_results_dict` -> `log_metrics('gsm8k', expected)` ->
  `verify_inference_results('gsm8k', expected, check_dmesg=False)`.
  (`check_dmesg=False` preserves the current behavior where gsm8k does not run
  the dmesg sweep; `bench_serv` runs it at the end of the suite.)
- `benchserv_test_random`: unchanged flow; `verify_inference_results('bench_serv', …)`
  now also emits the table + per-metric verdicts (dmesg as before).

## 8. Full aggregate + per-item detail (extension)

### 8.1 Full aggregate (every field of the result block)

`get_inference_results_dict` now parses the **entire** sglang.bench_serving
`Serving Benchmark Result` block via `parse_bench_serv_metrics`, an ordered
`(label, key)` table matched with a generic `<escaped label>:\\s+<number>`
rule. This:

- Fixes the prior unescaped-paren guard bug that silently dropped
  `median_ttft_ms`, `p99_ttft_ms`, `p99_tpot_ms`.
- Fixes the `Mean E2EL (ms)` vs actual `Mean E2E Latency (ms)` mismatch.
- Adds the fields that were never parsed: input/total/peak token throughput,
  concurrency, peak concurrent requests, full E2E latency (mean/median/p90/p99),
  p95/max ITL, and the retokenized generated-token count.

Gated keys (`output_throughput_per_sec`, `mean_ttft_ms`, `mean_tpot_ms`) keep
their names so threshold checks are unchanged. Result: ~28 aggregate fields are
captured and shown in the METRICS block.

### 8.2 Per-item detail (every scoring)

Both benchmarks emit per-item detail to an artifact on the benchmark node and a
compact per-item table to `cvs.log`:

- `bench_serv`: command adds `--output-file <log_dir>/benchmark_node/bench_serv_details.jsonl --output-details`;
  `parse_bench_serv_per_request` reads the per-request arrays (`input_lens`,
  `output_lens`, `ttfts`, `itls`, `errors`) → columns `req, input_len,
  output_len, ttft_ms, gen_tokens, error`.
- `gsm8k`: command adds `--raw-result-file <log_dir>/benchmark_node/gsm8k_per_question.jsonl`;
  `parse_gsm8k_per_question` reads `{prompt_id, prompt, output, correct}` →
  columns `q, correct, output` (preview).

`log_per_item_detail(test_name, remote_path, parser_fn, columns)` fetches the
file via the benchmark `Pssh` handle, logs a one-line summary (correct/total
for gsm8k; succeeded/errored for bench_serv) and the per-item table via
`format_per_request_table`. Display rows are capped by
`config.per_item_display_limit` (unset/0 => all); the full JSONL stays on the
node as an artifact.

## 9. Out of scope / future

- Renaming the pre-existing odd bench key (now normalized to
  `total_generated_tokens`).
- Copying the per-item JSONL artifacts into the devbox run dir / bundle (today
  they live under `<log_dir>/benchmark_node/` on the benchmark node).
- Per-percentile gating (p99 etc.) — only mean/throughput keys are gated today.

## 10. Verification

- Offline: `parse_gsm8k_metrics`, `parse_bench_serv_metrics`,
  `parse_bench_serv_per_request`, `parse_gsm8k_per_question` +
  `format_metrics_table` / `format_per_request_table` run against the real
  captured gsm8k/bench_serv stdout from a prior run reproduce the expected dict
  and a correct PASS verdict.
- `python -m py_compile` clean.
- On-cluster (optional): a suite re-run shows a METRICS block with verdicts for
  both `gsm8k` and `bench_serv` in `cvs.log`.
