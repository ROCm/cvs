## Title

`docs(dtni): add DTNI validation tracker for team review`

## Summary

Adds the **DTNI validation tracker** workbook as the agreed reference for multi-node training and inference validation. After review and merge, this file guides **automation uplift** on `dev/dtni`.

**Path:** `docs/dtni/validation_tracker/DTNI_Validation_Tracker.xlsx`

**Sheets (8):** JAX, Megatron, TorchTitan, InferenceMax, PyTorch, SGLang, vLLM, XDIT

> GitHub will not show cell-level diffs for the xlsx. Please open the file from this branch locally or download from the PR.

## Reviewers

| Sheet | Suggested reviewer | Focus |
|-------|-------------------|--------|
| JAX | @ | Training smoke/stability rows; CVS JAX test mapping |
| Megatron | @… | Megatron distributed coverage |
| TorchTitan | @… | TorchTitan path and gaps |
| InferenceMax | @… | InferenceMax workloads |
| PyTorch | @… | PyTorch distributed items |
| SGLang | @… | SGLang multi-node inference |
| vLLM | @… | vLLM inference matrix |
| XDIT | @… | xDiT / diffusion inference |

## Review checklist

Please comment on the PR as **`Sheet / Row # / note`**.

- [ ] **Completeness** — Missing test types we already run manually on DTNI?
- [ ] **Priorities** — P1 = true minimum bar before automation; P2 ordering sensible?
- [ ] **CVS mapping** — Each P1 row: existing CVS test, partial, or net-new? (state in Comments)
- [ ] **Acceptance criteria** — Comments include model, scale, metrics, and pass/fail where needed
- [ ] **Consistency** — No duplicate tests across sheets with conflicting priority
- [ ] **Automation scope** — P1 set is feasible as first tranche on `dev/dtni`

## Post-merge

- [ ] Link automation epics/issues to `docs/dtni/validation_tracker/`
