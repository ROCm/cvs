# DTNI validation tracker

Canonical checklist for *multi-node distributed training and inference validation* on DTNI clusters. After team review, this workbook is the reference for **automation uplift** on the [`dev/dtni`](https://github.com/ROCm/cvs/tree/dev/dtni) branch.

## Files

| File | Role |
|------|------|
| `DTNI_Validation_Tracker.xlsx` | Source of truth (human-edited) |
| `README.md` | How to read, review, and update the tracker |

## Workbook layout

One sheet per framework or stack. Each sheet uses the same column headers:

| Column | Meaning |
|--------|---------|
| **#** | Row id within the sheet |
| **Category** | Area (e.g. Training, Inference, Platform) |
| **Test / Metric** | What must be validated |
| **Priority** | **P1** = minimum bar before automation; **P2** = next wave |
| **Comments** | Models, configs, node counts, thresholds, Mapping notes |

| Sheet | automation scope |
|-------|--------------------------------|
| JAX | Distributed JAX training tests |
| Megatron | Megatron-LM distributed training |
| TorchTitan | TorchTitan training path |
| InferenceMax | InferenceMax single/multi-node |
| PyTorch | PyTorch distributed workloads |
| SGLang | SGLang distributed inference |
| vLLM | vLLM inference |
| XDIT | PyTorch xDiT inference (Flux, Wan, etc.) |

## How to review (GitHub PR)

Git does not diff `.xlsx` cell-by-cell. Reviewers should:

1. Check out the PR branch and open `DTNI_Validation_Tracker.xlsx` locally (Excel).
2. Review using the PR checklist in `PR_DESCRIPTION.md` (or the PR body).
3. Leave PR comments as **`Sheet / Row # / issue`** (e.g. `JAX / 12 / P1 should be P2`).
4. Authors apply edits in the workbook, push to the same branch, and re-request review.

### Review sign-off criteria

- **P1** rows are agreed as the first automation tranche.
- Each P1 row has enough **Comments** for an implementer (model, scale, pass/fail, metrics).
- No duplicate or conflicting priorities across sheets.

## After merge

- Treat **`dev/dtni`** (per team agreement) as the only authoritative copy
- Changes go through a **new PR**; describe sheet-level changes in the PR description.
- Automation issues and design docs should link to this path:
  `docs/dtni/validation_tracker/DTNI_Validation_Tracker.xlsx`
