---
ledger_for: /data/atnair/.claude/jobs/76513a86/tmp/spec-a1-final.md
plan_success_criterion: "client.per_gpu_throughput reflects the true GPU count (tp * pp) for every cell, single-node or distributed, while every existing single-node behavior (pp=1) is numerically unchanged; the authored suite is green and non-vacuous (mutation gate passes); no regression in vllm_job.py's existing test suite."
generated: 2026-07-16T16:47:17Z
total: 3
passing: 0
failing: 3
---

# Acceptance ledger: spec-a1-ledger

> Definition of done (plan success_criterion): "(plan has no success_criterion)"
> `/goal` terminates when failing == 0 — not on the model's feeling of done.
> HUMAN CHECKPOINT (t=0): glance over this list and confirm it IS what done means.
> A missing/wrong item is invisible to review+validation and silently corrupts the run.

## tests
- [ ] (t1) `/data/atnair/miniconda3/envs/py310/bin/python3 -m unittest cvs.lib.inference.unittests.test_vllm_parsing -v` — authored greenfield suite for vllm_parsing.py must go fully green (AC1-AC5, AC7)
- [ ] (t2) `/data/atnair/miniconda3/envs/py310/bin/python3.10 /data/atnair/.claude/bin/test-mutation-gate.py cvs/lib/inference/utils/vllm_parsing.py --test-cmd "/data/atnair/miniconda3/envs/py310/bin/python3 -m unittest cvs.lib.inference.unittests.test_vllm_parsing -v"` — mutation gate proves the suite is non-vacuous -- run only after t1 is green
- [ ] (t3) `/data/atnair/miniconda3/envs/py310/bin/python3 -m unittest cvs.lib.inference.unittests.test_vllm_job_ray_backend -v` — AC6 regression guard: parse_results call site (pp=self.pp) does not break existing VllmJob ray-backend suite

