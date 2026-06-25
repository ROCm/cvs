# CVS Suite Reports — Architecture



Three **independent** report tracks share `cvs/lib/report/` plumbing (registry,

session store, `HtmlReportManager.generate_suite_reports`, zip bundling). They

do not block one another.



## Layering: generic vs presets vs suite wiring



```text

Generic engines (no suite ids)

  cvs/lib/report/inference.py

  cvs/lib/report/training.py

  cvs/lib/report/parity/inference.py

  cvs/lib/report/panels/*

        ↑ configured by

Presets (suite- or program-specific)

  cvs/lib/report/presets/<suite>.py

  cvs/lib/report/presets/inference_parity.py   # W1 IX only

        ↑ registered in

Suite conftest (demo: inferencex_atom)

  cvs/tests/inference/inferencex_atom/conftest.py

```



- **Generic modules** accept arbitrary `suite_id`, framework ids, and metric specs.

  They are exported from `cvs.lib.report` only.

- **Presets** hold column layouts, chart series, W1 parity keys, etc. Import

  presets explicitly: `from cvs.lib.report.presets.inferencex_atom import ...`

- **Suite conftest** uses ``cvs.lib.report.inference_wiring`` (three calls) plus a
  preset import. inferencex_atom is the reference implementation, not a library dependency.

```text
cvs/lib/report/
├── inference.py          # sweep cells, gates, concurrency charts
├── inference_wiring.py   # configure / bind / row-extra hooks for suite conftest

├── training.py           # per-node throughput tables (Megatron / JAX)

├── registry.py           # register_suite_report + session bind

├── presets/              # suite-specific InferenceReportConfig bindings

├── panels/

│   ├── scaling.py        # multi-node scaling — any inference run with nnodes>1

│   └── training_parity.py  # training run-to-run / framework compare

└── parity/

    └── inference.py      # generic framework parity (arbitrary framework ids)

```



## Track comparison



| Track | Question it answers | Inputs | Blocks on |

|-------|---------------------|--------|-----------|

| **Inference suite report** | How did this run perform? | `inf_res_dict`, lifecycle | Nothing |

| **Scaling panel** | Did we scale efficiently across nodes? | Same run + optional single-node baseline JSON | Nothing (M4 not required) |

| **Training report** | How did training perform per node? | `training_res_dict` | Nothing |

| **Training parity** | Megatron vs JAX, or run vs last green? | 2+ `*_report.json` from training runs | Nothing |

| **Inference parity** | Same sweep cell across frameworks? | 2+ inference `*_report.json` | Nothing for scaling/training |



## Design rules



1. **Panels are optional sections**, not separate zip roots. A single

   `*_report.html` can include a Scaling section when data exists.

2. **JSON sidecar** carries `panels.scaling`, `panels.training_parity`, etc. so

   CI can merge without re-parsing HTML.

3. **Baseline JSON is always optional.** Scaling efficiency uses

   `compare.scaling.efficiency_pct` when a single-node reference JSON is

   provided; without it the panel shows per-node breakdown only.

4. **Registration is per domain:** `InferenceReportConfig` vs

   `TrainingReportConfig`. `generate_suite_reports` dispatches on config type.



## Scaling panel (M5, not gated on M4)



Emitted when:



- `variant_config.params.nnodes > 1`, or

- multiple hosts appear in `inf_res_dict` for the same sweep cell



Metrics (from automation plan §6.1 Tier 5):



- `scaling.efficiency_pct` — actual cluster throughput / (single-node baseline × nnodes)

- Per-node `client.output_throughput` (already in full results table; panel summarizes)



Optional baseline: path in preset `scaling_baseline_json` or prior run JSON in CI.



## Training parity (independent)



Examples:



- Megatron 8B single vs distributed same model

- Run vs `compare.prev_run.throughput_per_gpu_ratio`

- JAX vs Megatron on same cluster config (manual merge of two JSON files)



Does not use `compare.vllm.*` or inference cell keys.



## Inference parity — generic module



`cvs/lib/report/parity/inference.py` merges aligned sweep cells from multiple

inference `*_report.json` sidecars. Framework ids are caller-defined strings.



**Types:** `InferenceParitySource`, `InferenceParityMetric`, `InferenceParityConfig`



**Generic helpers:** `build_inference_parity_config`, `default_parity_metrics`,

`build_session_parity_config` (session-end hook via `parity_compare_jsons` on

`InferenceReportConfig`)



**Standalone CLI:**



```bash

python -m cvs.lib.report.parity.inference \

  --reference ref_report.json \

  --reference-id ref \

  --compare other=other_report.json \

  --out inference_parity_report.html

```



**Python API:**



```python

from pathlib import Path

from cvs.lib.report import (

    InferenceParitySource,

    build_inference_parity_config,

    write_inference_parity_report,

)



config = build_inference_parity_config(

    reference=InferenceParitySource("ref", "Reference", "ref.json"),

    comparators=(InferenceParitySource("b", "B", "b.json"),),

)

write_inference_parity_report(Path("parity.html"), config=config)

```



**W1 IX preset** (optional): `cvs/lib/report/presets/inference_parity.py` —

`w1_triple_parity_config` with fixed `atom` / `vllm` / `sglang` ids.



**Optional zip attach:** `publish_inference_parity_report(config, report_manager=..., pytest_config=...)`



## Demo order (recommended)



1. Inference suite report — **done** (inferencex_atom W1 perf)

2. Scaling panel — when multi-node lab or synthetic multi-host dict

3. Training report — Megatron 8B single pilot

4. Training parity — two training JSONs

5. Inference parity — triple-framework run (W1 preset or generic config)

## Interactive viewer (Phases B–D)

`*_report_viewer.html` is written whenever ``interactive_viewer`` is enabled on the preset
(not only when the static HTML truncates). It loads the JSON sidecar and provides:

- ISL / OSL / policy / host / tier filters + search
- Chart.js concurrency line charts (zoom / pan) driven by ``chart_config`` + filtered cells
- Throughput heatmap (concurrency × ISL/OSL slice)
- Gate matrix from ``gate_matrix`` in JSON
- Export filtered rows to CSV

Static ``*_report.html`` truncates inline cell cards when ``len(cells) > viewer_cell_threshold``;
full cell data remains in JSON and the viewer.

## Inference parity (Phase E)

Session-end hook ``publish_session_inference_parity`` runs after the reference suite JSON is written.

Configure comparators via preset ``parity_compare_jsons`` and/or env::

    CVS_INFERENCE_PARITY_COMPARE=vllm=/path/vllm_report.json,sglang=/path/sglang_report.json

Outputs ``inference_parity_report.html/json`` in the zip when all JSON paths exist.

## Training reports (Phase F)

``training_res_dict`` (node → metrics) + ``TrainingReportConfig`` preset + ``training_wiring`` in
suite conftest. Megatron pilot: ``cvs/tests/training/megatron/conftest.py``.

Optional training parity: ``parity_baseline_json`` or ``CVS_TRAINING_PARITY_BASELINE_JSON``.

