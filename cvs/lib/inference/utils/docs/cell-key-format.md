# Cell key format

The cell key is a string that uniquely identifies one sweep cell. It is the single source
of truth shared between:
- `VariantConfig.cell_key()` — generates the key
- `threshold.json` top-level keys — what the key must match
- The test's verdict lookup — reads threshold specs from `variant_config.thresholds` keyed by this string
- pytest parametrize ids — `pytest_generate_tests` builds ids as
  `combo_name + "-conc" + concurrency + "-" + metric_short`
  (e.g. `w1_isl=1000_osl=1000-conc16-mean_ttft_ms`); the cell key string never
  appears in pytest parametrize ids

---

## Format specification

```
ISL={isl},OSL={osl},TP={tensor_parallelism},CONC={concurrency}
```

**All four fields are required, in this exact order, with no spaces.**

| Field | Source | Description |
|---|---|---|
| `ISL=` | `SeqCombo.isl` (string from config) | Input sequence length in tokens |
| `OSL=` | `SeqCombo.osl` (string from config) | Output sequence length in tokens |
| `TP=` | `Params.tensor_parallelism` (string from config) | Tensor parallelism degree; comes from `params`, not from the run |
| `CONC=` | `Run.concurrency` (integer from config) | Number of concurrent requests for this run |

Values are interpolated verbatim from the config fields — no numeric normalization.
`isl` and `osl` are string-typed in the schema; `concurrency` is an integer but
Python's f-string renders it directly. What appears in the config is what appears in
the key.

**Implementation** (from `VariantConfig.cell_key`):
```python
f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},CONC={concurrency}"
```

---

## Where it is used

**`threshold.json` top-level keys** — each key names one sweep cell. The coverage
check (`_check_thresholds_cover_sweep`) enforces a two-way match at load time:
every cell key produced by `expected_cells()` must appear as a threshold key, and
every threshold key must name a cell the sweep actually runs. Axis 2 of the same
check verifies every present cell has a spec for every `GATED_METRICS` member.
The spec keys the coverage check looks for are prefixed with `client.`
(e.g., `client.mean_ttft_ms`, `client.output_throughput`) — bare `GATED_METRICS`
names without this prefix will not satisfy the check.

**`test_metric` verdict lookup** — the test calls
`variant_config.cell_key(isl, osl, concurrency)` to build the key it looks up in
the threshold specs (`variant_config.thresholds.get(cell)`). A format mismatch means
no spec is found, and the test falls through to the record-only branch — PASS with
zero assertions, silently, even under `enforce_thresholds=true`.

Note: `inf_res_dict` is keyed by the tuple
`(model_id, gpu_arch, isl, osl, combo_name, concurrency)`, which `test_metric`
builds independently before `cell_key` is called. The cell_key string is not used
to index `inf_res_dict`.

---

## Worked example — `w1_llama31_70b_fp8_config.json`

The config declares `tensor_parallelism: "8"` in `params`, and the sweep has five
cells all at `concurrency: 16`:

```json
"params": { "tensor_parallelism": "8" },
"sweep": {
  "sequence_combinations": [
    { "name": "w1_isl=1000_osl=1000", "isl": "1000", "osl": "1000" },
    { "name": "w1_isl=8000_osl=1000", "isl": "8000", "osl": "1000" },
    { "name": "w1_isl=1000_osl=8000", "isl": "1000", "osl": "8000" },
    { "name": "w1_isl=1000_osl=4000", "isl": "1000", "osl": "4000" },
    { "name": "w1_isl=5000_osl=1024", "isl": "5000", "osl": "1024" }
  ],
  "runs": [
    { "combo": "w1_isl=1000_osl=1000", "concurrency": 16 },
    { "combo": "w1_isl=8000_osl=1000", "concurrency": 16 },
    { "combo": "w1_isl=1000_osl=8000", "concurrency": 16 },
    { "combo": "w1_isl=1000_osl=4000", "concurrency": 16 },
    { "combo": "w1_isl=5000_osl=1024", "concurrency": 16 }
  ]
}
```

`expected_cells()` returns — and `llama31_70b_fp8_threshold.json` must use as top-level keys:

```
ISL=1000,OSL=1000,TP=8,CONC=16
ISL=8000,OSL=1000,TP=8,CONC=16
ISL=1000,OSL=8000,TP=8,CONC=16
ISL=1000,OSL=4000,TP=8,CONC=16
ISL=5000,OSL=1024,TP=8,CONC=16
```

---

## Common mistakes

| Mistake | Effect |
|---|---|
| Adding a space: `ISL=1000, OSL=1000,...` | Key mismatch — no threshold found, no verdict |
| Changing field order: `TP=8,ISL=1000,...` | Key mismatch — no threshold found, no verdict |
| Using a different separator: `ISL=1000\|OSL=1000\|...` | Key mismatch — no threshold found, no verdict |
| Normalizing values: `ISL=1024` when config says `"1000"` | Key mismatch — values are verbatim from the config string |
| Typo in threshold.json key | Axis 1 of `_check_thresholds_cover_sweep` catches this at load time |

Any key mismatch silently drops the cell — the test finds no spec and falls through to the
record-only branch, reporting PASS with zero assertions. The load-time coverage check
(`_check_thresholds_cover_sweep`) is what catches this before the test runs: when
`enforce_thresholds=true` it raises `ValueError`; when false it emits a warning.
