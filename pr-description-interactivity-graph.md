# InferenceX interactivity viewer and run provenance

**Base branch:** `dev/dtni`  
**Head branch:** `hnimrama/interactivity_graph`

## Summary

Adds a SemiAnalysis-style **interactivity chart** to the inference interactive HTML viewer and improves **run deck reproducibility metadata** (git ref, container image, launch commands). Render-only reporting changes—no multinode orchestration or sweep config changes on this branch.

## Interactivity chart (interactive viewer)

- **Token Throughput per GPU vs. Interactivity** (one line per ISL/OSL shape; points by concurrency).
- **Interactivity** = `1000 / mean TPOT` (tok/s/user), aligned with [InferenceX](https://inferencex.semianalysis.com/inference).
- **Y-axis:** total token throughput per GPU.
- Hover **detail card** (click point to pin): date, image, throughput split, TP/GPUs, concurrency, precision, CI/upstream link.
- Zoom/pan: scroll zoom, shift+drag pan, drag box-zoom, reset / double-click.
- **Layout:** Filters → Overview → Interactivity → other sweep panels.

## Run deck & provenance

- Git ref, container image display, image digest at launch, IX launch command provenance.
- Run deck surfaces reproducibility fields; bundle copy skip when artifact already present.

## Test plan

- [ ] `pytest cvs/lib/report/unittests/test_viewer_scaffold.py cvs/lib/report/unittests/test_provenance.py cvs/lib/report/unittests/test_inference_report.py`
- [ ] `pytest cvs/lib/inference/unittests/test_inferencex_atom_launch.py cvs/core/orchestrators/unittests/test_container.py`
- [ ] Open `*_viewer.html` from a sweep report: interactivity chart, pin card, links, filters.

## Follow-up (separate PR)

Multinode IX-atom work is preserved on **`hnimrama/interactivity_graph-multinode-backup`** (and `hnimrama/ix-atom-multinode`) for a later merge after this lands.
