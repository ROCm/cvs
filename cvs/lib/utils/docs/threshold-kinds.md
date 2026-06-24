# Threshold Kinds Reference

Full reference for every threshold kind understood by `_check_one` and
`evaluate_all` in `cvs/lib/utils/verdict.py`.

---

## How thresholds are evaluated

`evaluate_all(actuals, thresholds)` iterates over every key in `thresholds`.
Before calling `_check_one` on a spec, it handles two conditions centrally:

- **Metric missing from actuals** — appends
  `"{metric}: missing from actuals"` as a violation and skips `_check_one`.
- **Metric value is `None`** — appends
  `"{metric}: value is None (metric unavailable for this run)"` as a violation
  and skips `_check_one`. This prevents a `float(None)` TypeError and surfaces
  the problem explicitly.

> **Note:** only `None` and missing keys are handled centrally. Any other
> non-float-convertible value — for example, a string `"N/A"` or `"error"`
> produced by a metric-extraction bug — passes both guards and reaches
> `_check_one`, where `_to_float` raises a raw `ValueError`, not a
> `ThresholdViolation`. Because `evaluate_all` has no try/except around
> `_check_one`, that `ValueError` propagates uncaught and breaks out of the
> violations-collection loop entirely. Callers that may populate `actuals` from
> untrusted or partially-failed data should coerce or validate metric values to
> `float`-or-`None` before passing to `evaluate_all`.

After these guards, `evaluate_all` calls `_check_one(metric, actual, spec)`.
If `_check_one` returns a truthy (non-empty, non-`None`) string, that string is
added to the violation list. Note: `_check_one` must return `None` (not `""`)
on the passing path — the collector uses `if v:` rather than
`if v is not None`, so an empty string would be silently dropped. All violations are collected before raising; `ThresholdViolation`
carries the full list in `.violations` and its message is the newline-joined
string of all of them.

---

## Kinds

### `min`

**JSON shape**

```json
{ "kind": "min", "value": <number> }
```

| Field   | Type   | Required |
|---------|--------|----------|
| `kind`  | string | yes      |
| `value` | number | yes      |

**Comparison**

Passes when `actual >= value`. Fails when `actual < value`.

**Failure message**

```
{metric}: actual {actual} < min {target}
```

**When to use**

Use for dimensionless or mixed-unit lower bounds where the unit is either
implicit from the metric name or irrelevant to the failure message. Examples:
token counts, request counts, dimensionless ratios that do not already have a
dedicated kind. Prefer `min_tok_s` when the metric is a token-per-second
throughput — it produces a more readable failure message with explicit units.

---

### `max`

**JSON shape**

```json
{ "kind": "max", "value": <number> }
```

| Field   | Type   | Required |
|---------|--------|----------|
| `kind`  | string | yes      |
| `value` | number | yes      |

**Comparison**

Passes when `actual <= value`. Fails when `actual > value`.

**Failure message**

```
{metric}: actual {actual} > max {target}
```

**When to use**

Use for upper bounds on metrics whose unit is not milliseconds. The canonical
case is `failed` (failed request count): a milliseconds suffix in the message
would be a unit lie. Use `max_ms` when the metric is a latency in milliseconds.
The comparison logic is identical to `max_ms`; only the failure message differs.

---

### `max_ms`

**JSON shape**

```json
{ "kind": "max_ms", "value": <number> }
```

| Field   | Type   | Required |
|---------|--------|----------|
| `kind`  | string | yes      |
| `value` | number | yes      |

**Comparison**

Passes when `actual <= value`. Fails when `actual > value`.

**Failure message**

```
{metric}: actual {actual} ms > max {target} ms
```

**When to use**

Use for latency upper bounds where the metric is expressed in milliseconds (TTFT,
TPOT, E2EL, ITL). The `ms` suffix in both slots of the failure message makes the
unit explicit. Use `max` instead when the metric is not a time measurement.

---

### `within`

**JSON shape**

```json
{ "kind": "within", "value": <number>, "tolerance_pct": <number> }
```

| Field           | Type   | Required |
|-----------------|--------|----------|
| `kind`          | string | yes      |
| `value`         | number | yes      |
| `tolerance_pct` | number | yes      |

**Comparison**

Computes an acceptable band:

```
lo = value * (1 - tolerance_pct / 100.0)
hi = value * (1 + tolerance_pct / 100.0)
```

Passes when `lo <= actual <= hi`. Fails otherwise.

> **Gotcha — `value` must be positive.** A negative `value` inverts the band:
> for example, `value = -100` with `tolerance_pct = 10` yields
> `lo = -90` and `hi = -110`, so `lo > hi` and the test `lo <= actual <= hi`
> can never pass. Every check fails with the normal outside-band message,
> giving no hint that the spec itself is the problem. There is no guard in
> `_check_one` or `evaluate_all` against this.
>
> A `value` of `0` collapses the band to a single point (`lo = hi = 0`); only
> `actual == 0` passes.

**Failure message**

```
{metric}: actual {actual} outside {target} ±{pct}%
```

**When to use**

Use when the acceptable range is symmetric around a target value and you want
to express the tolerance as a percentage rather than an absolute bound. Useful
for stability metrics or regressions against a known baseline where ±N% drift
is acceptable. Prefer `min` or `max` when the bound is one-sided.

---

### `min_tok_s`

**JSON shape**

```json
{ "kind": "min_tok_s", "value": <number> }
```

| Field   | Type   | Required |
|---------|--------|----------|
| `kind`  | string | yes      |
| `value` | number | yes      |

**Comparison**

Passes when `actual >= value`. Fails when `actual < value`.

**Failure message**

```
{metric}: actual {actual} tok/s < min {target} tok/s
```

**When to use**

Use for token throughput lower bounds (`client.total_token_throughput`,
`client.output_throughput`, `client.per_gpu_throughput`). Functionally identical
to `min` but annotates `tok/s` in both slots of the failure message. Use `min`
for lower bounds on metrics that are not token-per-second rates.

---

### `min_ratio`

**JSON shape**

```json
{ "kind": "min_ratio", "value": <number>, "reference": "<metric_name>" }
```

| Field       | Type   | Required |
|-------------|--------|----------|
| `kind`      | string | yes      |
| `value`     | number | yes      |
| `reference` | string | yes      |

**Comparison**

Computes `observed = actual / actuals[reference]` and passes when
`observed >= value`. Fails when `observed < value`.

`reference` names another metric that must also appear in `actuals`. The
observed value of that metric is used as the denominator.

**Failure message (ratio below minimum)**

```
{metric}: observed ratio {observed:.3f} < min {ratio} (vs {ref_metric})
```

**When to use**

Use when the threshold for one metric is expressed as a fraction of another
metric from the same cell run. Example: asserting that `client.output_throughput`
is at least 0.8 of `client.total_token_throughput`. This avoids hard-coding an
absolute number that would need to be recalibrated every time the reference
metric changes with hardware or model configuration.

Do not use `min_ratio` when an absolute lower bound suffices — the ratio
interpretation adds coupling between two metrics and the failure message is
harder to read than a plain `min` or `min_tok_s` violation.

---

#### The `_actuals` injection mechanism

Callers of `evaluate_all` never set `_actuals` in a spec dict. `evaluate_all`
injects it automatically for every `min_ratio` spec:

```python
spec_with_actuals = dict(spec)
if spec.get("kind") == "min_ratio":
    spec_with_actuals["_actuals"] = actuals
v = _check_one(metric, actuals[metric], spec_with_actuals)
```

Inside `_check_one`, `_actuals` is read with `.get("_actuals", {})` to look up
the reference metric's value at check time. This means:

- The threshold JSON never contains `_actuals`.
- The reference metric is resolved at evaluation time from the same `actuals`
  dict that holds the primary metric's value.
- Passing all cell actuals to `evaluate_all` (not just the single metric being
  asserted) is required for `min_ratio` to work. The `test_metric` pattern in
  `cvs/lib/inference/ADDING_A_SUITE.md` calls
  `evaluate_all(full_cell_actuals, {metric: spec})` for exactly this reason.

---

#### `min_ratio` failure modes

There are four failure conditions distinct from the ratio comparison itself.
Each produces its own violation string and short-circuits before the ratio is
computed.

**Reference metric missing from actuals**

Triggered when the metric named by `reference` is not a key in the `actuals`
dict at all (e.g., the key was never populated, or has a typo in `reference`).

```
{metric}: reference metric '{ref_metric}' missing from actuals
```

**Reference metric is `None`**

Triggered when the reference metric key exists in `actuals` but its value is
`None`. This happens when a derived metric could not be computed for the run
(e.g., a zero-divisor upstream in `to_client_metrics`).

```
{metric}: reference '{ref_metric}' is None (metric unavailable for this run)
```

**Reference metric is zero**

Triggered when `float(actuals[ref_metric]) == 0`. Division by zero is blocked
explicitly before the ratio is computed.

```
{metric}: reference '{ref_metric}' is 0; cannot compute ratio
```

**Reference metric is non-numeric and non-`None`**

Triggered when `actuals[ref_metric]` is neither `None` nor float-convertible
(e.g., the string `"error"` produced by a metric-extraction bug). `_to_float`
raises `ValueError` uncaught; `evaluate_all` has no `try/except` around
`_check_one`, so the exception propagates and breaks out of the
violations-collection loop. Callers must coerce reference metric values to
`float`-or-`None` before calling `evaluate_all`.

---

## Unknown kind

If `kind` does not match any of the six strings above, `_check_one` returns:

```
{metric}: unknown threshold kind '{kind}'
```

This surfaces as a violation collected by `evaluate_all`. There is no exception
at parse time — the error only appears when `evaluate_all` runs against a spec
with an unrecognised kind.

---

## Quick reference

| Kind        | Bound         | Unit in message | Required fields beyond `kind`   |
|-------------|---------------|-----------------|---------------------------------|
| `min`       | lower         | none            | `value`                         |
| `max`       | upper         | none            | `value`                         |
| `max_ms`    | upper         | `ms`            | `value`                         |
| `within`    | lower + upper | none            | `value`, `tolerance_pct`        |
| `min_tok_s` | lower         | `tok/s`         | `value`                         |
| `min_ratio` | lower (ratio) | ratio (3 d.p.)  | `value`, `reference`            |
