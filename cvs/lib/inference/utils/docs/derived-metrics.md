# Derived metrics

`to_client_metrics` in `vllm_parsing.py` appends five derived metrics on top of
the stock `vllm bench serve` scalars. Every derived metric is guarded by `_safe_div`.

---

## `_safe_div` contract

```python
def _safe_div(num, den):
    if num is None or den is None:
        return None
    try:
        den = float(den)
        if den == 0:
            return None
        return float(num) / den
    except (TypeError, ValueError):
        return None
```

Returns `None` when:

- Either operand is `None` (missing key or explicit null in the artifact)
- The divisor is `0`
- Either operand cannot be converted to `float` (`TypeError` or `ValueError`)

Never raises `TypeError` or `ValueError`; never returns a bogus `0`. Note: `OverflowError` is not caught — callers must ensure artifact values are within the representable float range.

### What `None` means downstream

| Context | Behaviour |
|---|---|
| HTML results table | Displayed as `-` |
| Record-only metric (not in `GATED_METRICS`) | Silently recorded; no assertion |
| Gated metric with a threshold spec | `evaluate_all` raises `ThresholdViolation` — the violation string names the metric and states the value is `None` (metric unavailable for this run) |

A `None` gated metric is always a loud failure, never a silent skip.

---

## Derived metrics

### `client.per_gpu_throughput`

**Formula**: `total_token_throughput / tp`

**Inputs from artifact**: `total_token_throughput` (stock scalar)

**Out-of-band input**: `tp` — tensor parallelism, passed by the caller from
`params.tensor_parallelism`

**None when**: `total_token_throughput` is missing or `None`, `tp` is `0` or
unconvertible to `float`.

**Gated or record-only**: record-only. Secondary throughput used for per-GPU
capacity analysis. `total_token_throughput` (the primary throughput) is gated
instead.

---

### `client.normalized_ttft_ms_per_tok`

**Formula**: `mean_ttft_ms / isl`

**Inputs from artifact**: `mean_ttft_ms` (stock scalar)

**Out-of-band input**: `isl` — input sequence length, passed by the caller from
the sweep `SeqCombo.isl`

**None when**: `mean_ttft_ms` is missing or `None`, `isl` is `0` or
unconvertible.

**Gated or record-only**: record-only. Diagnostic derivation — normalises TTFT
by input length to expose prefill efficiency across cells with different ISLs.
The per-quantile `*_ttft_ms` metrics are gated instead.

---

### `client.decode_latency_ratio`

**Formula**: `p99_itl_ms / p50_itl_ms`

**Inputs from artifact**: `p99_itl_ms` (stock scalar), `p50_itl_ms` (looked up
via `raw.get("p50_itl_ms")`)

**Important**: real `vllm bench serve` artifacts emit `median_itl_ms` for the
50th-percentile ITL, **not** `p50_itl_ms`. The source calls
`raw.get("p50_itl_ms")`, so this key is absent from every real artifact and the
denominator is always `None`. As a result, `client.decode_latency_ratio` is
**always `None` on real runs** — it is record-only by design and the `None`
value is silently recorded (no assertion, displayed as `-` in the results
table).

**None when**: `p50_itl_ms` is absent from the artifact (always, with real
runs), or either input is `None`, or `p50_itl_ms` is `0`.

**Gated or record-only**: record-only. Diagnostic derivation — intended to
measure tail latency inflation (how much worse p99 ITL is than the median). The
individual quantile ITL metrics (`p95_itl_ms`, `p99_itl_ms`, etc.) are gated
instead.

---

### `client.decode_throughput_p50`

**Formula**: `1000.0 / median_tpot_ms`

**Inputs from artifact**: `median_tpot_ms` (stock scalar)

**None when**: `median_tpot_ms` is missing, `None`, or `0`.

**Gated or record-only**: record-only. Secondary throughput — converts median
time-per-output-token to tokens-per-second for human-readable throughput
comparisons. `median_tpot_ms` (the latency metric) is gated instead.

---

### `client.success_rate`

**Formula**: `completed / (completed + failed)`

**Inputs from artifact**: `completed`, `failed` (both stock scalars)

**Implementation note**: the denominator is computed as
`completed + failed` only when both are non-`None`; if either is `None`, the
total is set to `None` and `_safe_div` returns `None`.

```python
total_req = None if completed is None or failed is None else completed + failed
m["client.success_rate"] = _safe_div(completed, total_req)
```

**None when**: either `completed` or `failed` is missing or `None`.

**Gated or record-only**: **gated**. Run health: a floor on the fraction of
requests that completed without error. A success rate below threshold indicates
infrastructure failures, OOM, or timeouts — not a performance regression.

---

## `client.goodput` alias

```python
m["client.goodput"] = raw.get("request_goodput")
```

Stock `vllm bench serve` emits the key `request_goodput`. The results table and
`threshold.json` reference it as `client.goodput` (without the `request_` prefix)
for readability. The alias is set unconditionally alongside the stock
`client.request_goodput` entry that the 1:1 namespace copy also produces.

If `--goodput` was not passed to `vllm bench serve`, stock emits
`request_goodput: null`, so `client.goodput` will be `None`.

**Gated or record-only**: record-only. Goodput is only meaningful when a
`GoodputSlo` was configured in the sweep, and the value reflects the SLO input
rather than an independent throughput measurement.
