'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Shared implementations for the JAX MaxText training suites. This is NOT a
runnable suite (leading underscore -> excluded by `cvs list`/`cvs run`); the two
suite files import these helpers and wrap each as an explicit `test_*` method:

  - jaxmaxtext_single.py       (single-node: no RDMA stage)
  - jaxmaxtext_distributed.py  (adds the test_setup_rdma stage)

Kept deliberately simple: plain shared functions + a couple of small helpers,
no framework-y generalization. The single vs distributed mode is recorded in
`training_res_dict["mode"]` and reflected in the console tables, the metric
results HTML title, and the loss-curve title/artifact.
'''

import html as _html
import json
import re
import shlex
import time
import uuid as _uuid
from pathlib import Path as _Path
from types import SimpleNamespace

import pytest
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib import MaxTextTrainingJob, needs_hf_tokenizer
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import (
    TRAINING_METRICS,
    TRAINING_METRIC_UNITS,
    compute_scaling_efficiency,
    compute_convergence,
    sample_loss_curve,
    evaluate_loss_decreasing,
    extract_checkpoint_timings,
)
from cvs.lib.training.jaxmaxtext.utils.loss_curve import render_loss_curve_png
from cvs.lib.utils.verdict import evaluate_all, ThresholdViolation
from cvs.lib.utils_lib import fail_test, update_test_result

log = globals.log

_STATUS_COLORS = {
    "PASS": "#2e7d32",
    "FAIL": "#c62828",
    "N/A": "#f9a825",
    "RECORD": "#555555",
}


# ---------- small helpers ----------


def _sweep_label(name):
    """Compact, unique-per-sweep id used in every parametrized test row and in
    the reports: PRECISION[-SL<seqlen>][-B<batch>], e.g. "BF16-SL4096-B3".

    The full sweep name still drives results/threshold lookups; this is only the
    display label. Falls back to a sanitized full name when PRECISION is absent.
    """
    name = name or ""

    def _tok(key):
        m = re.search(rf"{key}=([^,]+)", name)
        return m.group(1).strip() if m else None

    precision = _tok("PRECISION")
    seqlen = _tok("SEQLEN")
    batch = _tok("BATCH")

    parts = []
    if precision:
        parts.append(precision)
    if seqlen:
        parts.append(f"SL{seqlen}")
    if batch:
        parts.append(f"B{batch}")
    if parts:
        return "-".join(parts)
    return (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")) or "default"


def _enabled_sweep_names(config_file):
    """Read sweep names to run from the raw config (collection time, no fixtures).

    Honors training.enabled_sweep_list (subset selector); falls back to every
    declared sweep, or a single implicit "default" when none are declared.
    """
    try:
        with open(config_file) as fp:
            raw = json.load(fp)
    except Exception:
        return ["default"]
    training = raw.get("training", {})
    names = [s.get("name") for s in training.get("sweeps", []) if s.get("name")]
    if not names:
        return ["default"]
    enabled = training.get("enabled_sweep_list") or names
    return [n for n in enabled if n in names] or names


def _find_sweep(variant_config, sweep_name):
    for s in variant_config.enabled_sweeps():
        if s.name == sweep_name:
            return s
    return None


def _mode(variant_config):
    return "distributed" if variant_config.training.distributed else "single"


def _format_expected(spec):
    """Human-readable expected-threshold string for the console log + summary file."""
    if not spec:
        return "-"
    kind = spec.get("kind")
    value = spec.get("value")
    if kind == "info":
        return f"info ({value})" if value is not None else "info"
    if kind in ("min", "min_tok_s"):
        return f">= {value}"
    if kind == "max":
        return f"<= {value}"
    if kind == "max_ms":
        return f"<= {value} ms"
    if kind == "within":
        return f"{value} +/-{spec.get('tolerance_pct')}%"
    if kind == "min_ratio":
        return f">= {value} x {spec.get('reference')}"
    return str(spec)


def _format_value(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# ---------- lifecycle stage implementations ----------
# Plain helpers (no test_ prefix): the suite files wrap each as a `test_*`
# method with a docstring so the suites read as self-documenting.


def _precreate_tmp_bind_mounts(orch):
    """Create host-side ``/tmp/...`` bind-mount source dirs before launch.

    Docker auto-creates a missing bind-mount source directory owned by root; a
    leftover root-owned dir (``docker system prune`` does not remove host bind
    dirs) then blocks the next user on a shared GPU node with a permission
    error. Creating them here over SSH -- via ``exec_on_host``, which runs on
    the cluster host OS as the invoking user -- makes them user-owned instead.

    Only ``/tmp/`` sources are touched so device/system mounts (``/dev/*``,
    ``/lib/*``) are never created.
    """
    exec_host = getattr(orch, "exec_on_host", None)
    if not callable(exec_host):
        return
    try:
        volumes = orch.get_volumes()
    except Exception:  # noqa: BLE001 - best-effort; docker still auto-creates
        return
    sources, seen = [], set()
    for vol in volumes or []:
        src = str(vol).split(":", 1)[0].strip()
        if src.startswith("/tmp/") and src not in seen:
            seen.add(src)
            sources.append(src)
    if not sources:
        return
    quoted = " ".join(shlex.quote(p) for p in sources)
    try:
        exec_host(f"mkdir -p {quoted}")
    except Exception:  # noqa: BLE001 - non-fatal; fall back to docker auto-create
        pass


def launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container. Verify it is running."""
    t = time.monotonic()
    _precreate_tmp_bind_mounts(orch)
    ok = orch.setup_containers()
    lifecycle.record(request.node.nodeid, "container_launch", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        name = orch.get_container_name(orch.container_config, orch.container_config["image"])
        pytest.fail(f"setup_containers() returned False for {name}")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def setup_rdma(orch, variant_config, hf_token, lifecycle, request):
    """Distributed-only: copy RDMA library into container (thor2 NIC only)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not variant_config.training.distributed:
        pytest.skip("single-node: RDMA not needed")
    if not variant_config.training.nic_type or "thor" not in variant_config.training.nic_type.lower():
        pytest.skip(f"nic_type={variant_config.training.nic_type}: RDMA lib copy not needed")
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_rdma_lib()
    lifecycle.record(request.node.nodeid, "rdma_setup", time.monotonic() - t)


def setup_tokenizer(orch, variant_config, hf_token, lifecycle, request):
    """Download HF tokenizer into models dir (skipped for synthetic data)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    if not needs_hf_tokenizer(variant_config.training):
        reason = (
            "dataset_type=synthetic: training uses random token ids in "
            "[0, vocab_size); HuggingFace tokenizer download is not required"
        )
        log.info("skipping tokenizer setup: %s", reason)
        pytest.skip(reason)
    t = time.monotonic()
    job = MaxTextTrainingJob(orch, variant_config, hf_token)
    job.setup_tokenizer()
    lifecycle.record(request.node.nodeid, "tokenizer_setup", time.monotonic() - t)


# Smoke test: smallest fixed run that confirms the model loads and trains a few
# steps without hitting any error_patterns. Overrides only per_device_batch_size,
# max_target_length, precision and step count; keeps the config's model_name +
# tokenizer so the vocab/tokenizer stay consistent. No metric/threshold checks.
_SMOKE_STEPS = 5
_SMOKE_BATCH = 1
_SMOKE_SEQLEN = 2048


def smoke(orch, variant_config, hf_token, lifecycle, request):
    """Smoke test: the model loads and runs _SMOKE_STEPS steps without any
    error_pattern firing.

    Runs one short training with small fixed per_device_batch_size / seqlen and
    BF16, overriding the config for this run only. Completing the steps without an
    error signature (scanned by poll_for_completion) is the only pass criterion --
    there is no metric or threshold verification. A failure sets lifecycle.failed
    so downstream stages skip.

    Enabled by default; skipped when training.smoke.enabled=false (opt-OUT, e.g.
    during iterative experiments). steps/batch/seqlen come from training.smoke.
    """
    cfg = variant_config.training.smoke
    if not getattr(cfg, "enabled", True):
        pytest.skip("smoke test disabled (training.smoke.enabled=false)")
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    steps = getattr(cfg, "steps", _SMOKE_STEPS)
    batch = getattr(cfg, "per_device_batch_size", _SMOKE_BATCH)
    seqlen = getattr(cfg, "max_target_length", _SMOKE_SEQLEN)

    # Isolated deep copy so the smoke run's tiny steps/overrides never leak into
    # the real per-sweep training_run tests (they share the module-scoped config).
    smoke_variant = variant_config.model_copy(deep=True)
    smoke_variant.training.steps = steps
    smoke_sweep = SimpleNamespace(
        name="SMOKE",
        maxtext_overrides={
            "per_device_batch_size": batch,
            "max_target_length": seqlen,
            "dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "quantization": "",
        },
    )

    job = MaxTextTrainingJob(orch, smoke_variant, hf_token, sweep=smoke_sweep)
    t = time.monotonic()
    try:
        job.setup_training_env()
        job.build_training_cmd()
        job.start_training()
        # poll scans each node's log for error_patterns/NaN every iteration and
        # raises on the first match or on timeout; returns cleanly once the run
        # reaches step steps-1.
        job.poll_for_completion()
    except Exception as e:  # noqa: BLE001
        lifecycle.failed = True
        pytest.fail(f"smoke test failed (model did not run {steps} steps cleanly): {e}")
    finally:
        # Reap ranks so the smoke run leaves no orphan processes for the next stage.
        try:
            job.stop_training()
        except Exception:  # noqa: BLE001
            pass

    lifecycle.record(request.node.nodeid, "smoke", time.monotonic() - t)
    log.info(
        "smoke PASSED | model=%s steps=%s batch=%s seqlen=%s",
        smoke_variant.training.maxtext_config.get("model_name", "<base.yml default>"),
        steps,
        batch,
        seqlen,
    )


def training_run(orch, variant_config, hf_token, sweep_name, training_res_dict, lifecycle, request):
    """Per sweep: build the command, train, poll, parse results.

    Runs once per enabled sweep with that sweep's maxtext overrides. A failure is
    isolated to this sweep's row (it does NOT set lifecycle.failed) so the other
    sweeps still run and report.
    """
    training_res_dict.setdefault("mode", _mode(variant_config))
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    sweep = _find_sweep(variant_config, sweep_name)
    job = MaxTextTrainingJob(orch, variant_config, hf_token, sweep=sweep)
    try:
        job.setup_training_env()
        job.build_training_cmd()
        t = time.monotonic()
        job.start_training()
        job.poll_for_completion()
        wall_time = time.monotonic() - t
        results = job.parse_results()
        # Scan host dmesg over this run's window for GPU/HW/kernel faults. Uses
        # fail_test internally (rolled into the aggregated failure summary) and
        # is best-effort, so it never raises here.
        job.scan_dmesg_for_errors()
    except Exception as e:  # noqa: BLE001 - isolate the failure to this sweep
        log.error("training run failed for sweep '%s': %s", sweep_name, e)
        # Reap any lingering ranks so the next sweep does not launch on top of
        # them (and so persistent containers are not left with orphan processes).
        try:
            job.stop_training()
        except Exception:  # noqa: BLE001
            pass
        pytest.fail(f"training run failed for sweep '{sweep_name}': {e}")

    # wall_time is this sweep's measured wall-clock; logged for diagnostics only.
    # Convergence is surfaced/asserted via the registered steps_to_target /
    # time_to_target_seconds metrics below -- the old ad-hoc
    # training.wall_time_seconds / convergence_* keys were never in
    # TRAINING_METRICS, so nothing displayed or gated them.
    log.info("[training] sweep '%s' wall-clock: %.1fs", sweep_name, wall_time)

    baseline = variant_config.training.scaling_baseline
    results["training.scaling_efficiency_pct"] = compute_scaling_efficiency(
        results.get("training.tokens_per_sec_total"),
        job.num_nodes,
        baseline.tokens_per_sec_total,
        baseline.num_nodes,
    )

    conv = variant_config.training.convergence
    steps_to_target, time_to_target = compute_convergence(
        job.step_metrics,
        job.eval_metrics,
        conv.target_metric,
        conv.target_value,
    )
    results["training.steps_to_target"] = steps_to_target
    results["training.time_to_target_seconds"] = time_to_target

    training_res_dict.setdefault("sweeps", {})[sweep_name] = {
        "results": results,
        "step_metrics": job.step_metrics,
        "eval_metrics": job.eval_metrics,
        "num_nodes": job.num_nodes,
    }


def _latest_checkpoint_step_path(orch, ckpt_dir):
    """Path to the newest (highest-numbered) orbax checkpoint step dir under
    `ckpt_dir`, or None if none exist. Used as a pre-Phase-2 sanity check that
    Phase 1 actually wrote a checkpoint for MaxText to auto-resume from."""
    try:
        out = orch.exec("bash -c " + shlex.quote(f"ls -1 {shlex.quote(ckpt_dir)} 2>/dev/null"))
    except Exception:  # noqa: BLE001
        return None
    raw = (out or {}).get(orch.hosts[0], "")
    text = raw if isinstance(raw, str) else (raw or {}).get("output", "")
    steps = [int(s.strip().rstrip("/")) for s in (text or "").splitlines() if s.strip().rstrip("/").isdigit()]
    if not steps:
        return None
    return f"{ckpt_dir}/{max(steps)}"


def checkpoint_resume(orch, variant_config, hf_token, training_res_dict, lifecycle, request):
    """Opt-in: checkpoint save + resume + I/O timing (one sweep, two phases).

    Phase 1 trains `steps_before_ckpt` steps with checkpointing on (a checkpoint
    is written at `checkpoint_period`); Phase 2 resumes from it (same
    out_dir/run_name -> MaxText auto-restores the latest checkpoint) and trains
    `steps_after_resume` more. PASS = the resumed run restarts from a non-zero
    (restored) step AND the loss at the resume boundary matches Phase 1 within
    `loss_tolerance` (state restored, not reinitialized). Also benchmarks
    checkpoint_save_seconds / checkpoint_load_seconds, gated inline against
    max_save_seconds / max_load_seconds when > 0 (else record-only).

    Skipped unless training.checkpoint_resume.enabled. Isolated: a failure here
    does NOT set lifecycle.failed. Runs on ONE sweep only; smoke_model_overrides
    can shrink the model (keeping the tokenizer/vocab) for a fast I/O check.
    """
    cfg = variant_config.training.checkpoint_resume
    if not getattr(cfg, "enabled", False):
        pytest.skip("checkpoint_resume disabled (training.checkpoint_resume.enabled=false)")
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    # Pick the sweep to exercise: config.sweep if set, else the first enabled one.
    enabled = variant_config.enabled_sweeps()
    chosen = None
    if getattr(cfg, "sweep", ""):
        chosen = next((s for s in enabled if s.name == cfg.sweep), None)
    if chosen is None:
        chosen = enabled[0] if enabled else SimpleNamespace(name="default", maxtext_overrides={})

    base_overrides = dict(getattr(chosen, "maxtext_overrides", None) or {})
    base_overrides.update(cfg.smoke_model_overrides or {})

    def _mk_job(total_steps, extra_overrides, enable_ckpt):
        # Deep copy so the CKPT run's steps/checkpointing never leak into the
        # shared, module-scoped variant used by the perf sweeps.
        v = variant_config.model_copy(deep=True)
        v.training.steps = total_steps
        v.training.enable_checkpointing = enable_ckpt
        ov = dict(base_overrides)
        ov.update(extra_overrides or {})
        sweep = SimpleNamespace(name="CKPT", maxtext_overrides=ov)
        return MaxTextTrainingJob(orch, v, hf_token, sweep=sweep)

    # Phase 1: train + SAVE (checkpointing on, sync so the save flushes). Clean any
    # stale CKPT output first so Phase 2 can only restore THIS run's checkpoint.
    job1 = _mk_job(
        cfg.steps_before_ckpt,
        {"checkpoint_period": cfg.checkpoint_period, "async_checkpointing": False},
        True,
    )
    run_name = f"jaxmaxtext_{variant_config.model.id}_{job1.sweep_tag}"
    ckpt_dir = f"{job1.out_dir}/{run_name}/checkpoints"
    try:
        orch.exec("bash -c " + shlex.quote(f"rm -rf {shlex.quote(job1.out_dir)} 2>/dev/null || true"))
    except Exception:  # noqa: BLE001
        pass

    t0 = time.monotonic()
    try:
        job1.setup_training_env()
        job1.build_training_cmd()
        job1.start_training()
        job1.poll_for_completion()
        job1.parse_results()
    except Exception as e:  # noqa: BLE001
        try:
            job1.stop_training()
        except Exception:  # noqa: BLE001
            pass
        pytest.fail(f"checkpoint save phase failed: {e}")
    finally:
        try:
            job1.stop_training()
        except Exception:  # noqa: BLE001
            pass

    p1_steps = list(job1.step_metrics or [])
    p1_log = getattr(job1, "raw_log", "") or ""

    # Phase 1 MUST have written a checkpoint, else Phase 2 has nothing to
    # restore and the resume/loss checks are meaningless. This happens when
    # checkpoint_period > steps_before_ckpt (no periodic save fires within
    # Phase 1). Fail loudly here rather than silently continue into a Phase 2
    # that cannot actually resume.
    ckpt_step_path = _latest_checkpoint_step_path(orch, ckpt_dir)
    if not ckpt_step_path:
        pytest.fail(
            f"Phase 1 wrote no checkpoint under {ckpt_dir}: checkpoint_period="
            f"{cfg.checkpoint_period} likely exceeds steps_before_ckpt="
            f"{cfg.steps_before_ckpt}. Set checkpoint_period <= steps_before_ckpt "
            f"so a checkpoint is saved during Phase 1."
        )
    try:
        checkpoint_step = int(str(ckpt_step_path).rstrip("/").rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        checkpoint_step = None
    log.info("[checkpoint] Phase 1 saved checkpoint at step %s (%s)", checkpoint_step, ckpt_step_path)

    # Phase 2: RESUME. Auto-restore Phase 1's latest checkpoint from the SAME
    # base_output_directory/run_name (both phases use the "CKPT" sweep + same
    # model, so out_dir/run_name are identical and MaxText auto-resumes the
    # latest checkpoint). enable_checkpointing MUST stay true: this MaxText
    # version rejects loading a checkpoint when enable_checkpointing=false
    # (incl. via load_full_state_path) -> "You must set enable_checkpointing=True
    # to load a checkpoint". To keep Phase 2 a resume (no extra checkpoint churn),
    # push checkpoint_period beyond the total step count so no NEW periodic
    # checkpoint is written while we train the extra steps.
    total_steps = cfg.steps_before_ckpt + cfg.steps_after_resume
    job2 = _mk_job(total_steps, {"checkpoint_period": total_steps + 1000}, True)
    try:
        job2.setup_training_env()
        job2.build_training_cmd()
        job2.start_training()
        job2.poll_for_completion()
        job2.parse_results()
    except Exception as e:  # noqa: BLE001
        try:
            job2.stop_training()
        except Exception:  # noqa: BLE001
            pass
        pytest.fail(f"checkpoint resume phase failed: {e}")
    finally:
        try:
            job2.stop_training()
        except Exception:  # noqa: BLE001
            pass

    p2_steps = list(job2.step_metrics or [])
    p2_log = getattr(job2, "raw_log", "") or ""

    # Free the (large) checkpoint files now that both phases are done, unless the
    # user asked to keep them (delete_ckpt_dir=false) for post-test inspection.
    if getattr(cfg, "delete_ckpt_dir", True):
        try:
            orch.exec("bash -c " + shlex.quote(f"rm -rf {shlex.quote(ckpt_dir)} 2>/dev/null || true"))
            log.info("[checkpoint] deleted checkpoint dir %s (delete_ckpt_dir=true)", ckpt_dir)
        except Exception:  # noqa: BLE001
            pass
    else:
        log.info("[checkpoint] keeping checkpoint dir %s (delete_ckpt_dir=false)", ckpt_dir)

    # --- resume correctness ---
    # A fresh (non-resumed) run logs step 0 first; a restored run starts at the
    # checkpoint step. So a non-zero first step means the checkpoint (step +
    # weights + optimizer) was restored.
    resumed_step = p2_steps[0]["step"] if p2_steps else None
    resume_ok = resumed_step is not None and resumed_step > 0

    # --- loss continuity at the resume boundary ---
    # Compare loss at the SAME step in both phases whenever they overlap: a
    # correct restore reproduces Phase 1's loss at that step (delta ~ 0), so this
    # is a true "state restored" check that does not depend on step alignment.
    # Only when Phase 2 does NOT re-emit any Phase-1 step (it continued past the
    # checkpoint without re-logging it) do we fall back to adjacent-boundary
    # continuity (Phase 1's last loss vs Phase 2's first) -- looser, since those
    # are one training step apart, but the best signal available.
    p1_loss_by_step = {
        s["step"]: s["loss"] for s in p1_steps if isinstance(s.get("loss"), (int, float)) and s.get("step") is not None
    }
    p2_loss_by_step = {
        s["step"]: s["loss"] for s in p2_steps if isinstance(s.get("loss"), (int, float)) and s.get("step") is not None
    }
    common_steps = sorted(set(p1_loss_by_step) & set(p2_loss_by_step))
    if common_steps:
        cmp_step = common_steps[0]
        p1_cmp_loss, p2_cmp_loss = p1_loss_by_step[cmp_step], p2_loss_by_step[cmp_step]
        loss_basis = f"matched step {cmp_step}"
    else:
        cmp_step = None
        p1_cmp_loss = next((s["loss"] for s in reversed(p1_steps) if isinstance(s.get("loss"), (int, float))), None)
        p2_cmp_loss = next((s["loss"] for s in p2_steps if isinstance(s.get("loss"), (int, float))), None)
        loss_basis = "adjacent boundary (no overlapping step; p1 last vs p2 first)"
    loss_delta = abs(p2_cmp_loss - p1_cmp_loss) if (p1_cmp_loss is not None and p2_cmp_loss is not None) else None
    loss_ok = loss_delta is not None and loss_delta <= cfg.loss_tolerance

    # --- checkpoint I/O timings ---
    save_seconds = extract_checkpoint_timings(p1_log).get("save_seconds")
    if save_seconds is None:
        # Fallback: with async_checkpointing=false the checkpoint step blocks, so
        # it is the slow outlier -> save_seconds ~= max step time - median step time.
        secs = sorted(s["seconds"] for s in p1_steps if isinstance(s.get("seconds"), (int, float)))
        if len(secs) >= 3:
            median = secs[len(secs) // 2]
            delta = secs[-1] - median
            save_seconds = delta if delta > 0 else None
    load_seconds = extract_checkpoint_timings(p2_log).get("load_seconds")  # best-effort; None -> record-only

    # --- record rows into the consolidated metric-results table (CKPT label) ---
    rows = training_res_dict.setdefault("metric_rows", [])

    def _io_row(metric_name, value, max_bound):
        gated = bool(max_bound and max_bound > 0)
        if value is None:
            status = "N/A"
        elif gated:
            status = "PASS" if value <= max_bound else "FAIL"
        else:
            status = "RECORD"
        rows.append(
            {
                "sweep": "CKPT",
                "metric": metric_name,
                "expected": (f"<= {max_bound}" if gated else "record"),
                "actual": _format_value(value),
                "unit": "s",
                "status": status,
            }
        )
        return status

    save_status = _io_row("checkpoint_save_seconds", save_seconds, cfg.max_save_seconds)
    load_status = _io_row("checkpoint_load_seconds", load_seconds, cfg.max_load_seconds)

    # Stash the checkpoint I/O results in their own dict so print_results_table
    # can render them as a separate section (after the per-sweep tables and loss
    # curves) rather than mixing them into the perf-sweep metric tables.
    training_res_dict["checkpoint_io"] = {
        "resumed_step": resumed_step,
        "loss_delta": loss_delta,
        "loss_tolerance": cfg.loss_tolerance,
        "save_seconds": save_seconds,
        "save_max": cfg.max_save_seconds,
        "save_status": save_status,
        "load_seconds": load_seconds,
        "load_max": cfg.max_load_seconds,
        "load_status": load_status,
    }

    lifecycle.record(request.node.nodeid, "checkpoint_resume", time.monotonic() - t0)
    log.info(
        "[checkpoint] resumed_step=%s loss_delta=%s (tol=%s, basis=%s) | save=%ss (%s) load=%ss (%s)",
        resumed_step,
        loss_delta,
        cfg.loss_tolerance,
        loss_basis,
        save_seconds,
        save_status,
        load_seconds,
        load_status,
    )

    # --- verdict: resume correctness (hard) + I/O gates (when configured) ---
    failures = []
    if not resume_ok:
        failures.append(f"resume did not restore a checkpoint (phase-2 first step={resumed_step}, expected > 0)")
    if not loss_ok:
        failures.append(
            f"loss discontinuity at resume ({loss_basis}): |{p2_cmp_loss} - {p1_cmp_loss}| = "
            f"{loss_delta} > tol {cfg.loss_tolerance}"
        )
    if save_status == "FAIL":
        failures.append(f"checkpoint save {save_seconds}s > max {cfg.max_save_seconds}s")
    if load_status == "FAIL":
        failures.append(f"checkpoint load {load_seconds}s > max {cfg.max_load_seconds}s")

    if failures:
        training_res_dict.setdefault("metric_failures", []).extend(f"[CKPT] {m}" for m in failures)
        pytest.fail("; ".join(failures))
    log.info("checkpoint_resume PASSED")


def metric(sweep_name, metric, training_res_dict, variant_config, lifecycle, request):
    """One test (row) per (sweep, metric). Threshold-driven PASS/FAIL; logs
    `sweep | metric | expected | actual | status` and collects rows for the
    single metric-results HTML file (linked from every metric row)."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    rec = training_res_dict.get("sweeps", {}).get(sweep_name)
    results = rec.get("results") if rec else None
    if not results:
        pytest.skip(f"no results for sweep '{sweep_name}' (training did not complete)")

    label = _sweep_label(sweep_name)
    full = "training." + metric
    value = results.get(full)
    unit = TRAINING_METRIC_UNITS.get(metric, "-")
    # The sweep name IS the threshold cell key.
    spec = (variant_config.thresholds.get(sweep_name) or {}).get(full)
    expected = _format_expected(spec)
    actual = _format_value(value)

    rows = training_res_dict.setdefault("metric_rows", [])

    def _record(status):
        rows.append(
            {
                "sweep": label,
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "unit": unit,
                "status": status,
            }
        )

    if value is None:
        log.info("[metric] %-6s %-24s | expected %-14s | actual None | %s -> N/A", label, metric, expected, unit)
        _record("N/A")
        pytest.skip(f"{metric}: no value produced this run")

    if spec is None or not variant_config.enforce_thresholds:
        log.info(
            "[metric] %-6s %-24s | expected %-14s | actual %s | %s -> RECORD", label, metric, expected, actual, unit
        )
        _record("RECORD")
        return

    try:
        evaluate_all(results, {full: spec})
    except ThresholdViolation as e:
        log.error(
            "[metric] %-6s %-24s | expected %-14s | actual %s | %s -> FAIL", label, metric, expected, actual, unit
        )
        _record("FAIL")
        training_res_dict.setdefault("metric_failures", []).append(
            f"[{label}] {metric}: expected {expected}, actual {actual}"
        )
        pytest.fail(str(e))
    else:
        log.info("[metric] %-6s %-24s | expected %-14s | actual %s | %s -> PASS", label, metric, expected, actual, unit)
        _record("PASS")


def loss_curve(sweep_name, training_res_dict, variant_config, lifecycle, request):
    """Row 32 (per sweep): sample the training loss, render a PNG, gate on trend."""
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")

    label = _sweep_label(sweep_name)
    mode = _mode(variant_config)
    rec = training_res_dict.get("sweeps", {}).get(sweep_name)
    step_metrics = rec.get("step_metrics") if rec else None
    if not step_metrics:
        pytest.skip(f"no step metrics for sweep '{sweep_name}' (training did not complete)")

    cfg = variant_config.training.loss_curve
    points = sample_loss_curve(step_metrics, cfg.sample_every, cfg.milestone_steps)
    verdict = evaluate_loss_decreasing(points, cfg.max_slope)

    mgr = getattr(request.config, "_html_report_manager", None)
    if mgr is not None and getattr(mgr, "is_enabled", False):
        out_dir = mgr.log_dir
    else:
        out_dir = _Path(variant_config.paths.log_dir)
    png_path = None
    try:
        _Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = f"loss_curve_{variant_config.model.id}_{mode}_{label}_{str(_uuid.uuid4()).split('-')[-1]}.png"
        abs_path = _Path(out_dir) / fname
        title = f"Training Loss Curve — {variant_config.model.id} [{mode}/{label}]"
        png_path = render_loss_curve_png(points, abs_path, title=title)
    except Exception as e:  # noqa: BLE001 - plotting must never break the verdict
        log.warning("loss curve: could not prepare PNG output (%s)", e)

    if png_path and mgr is not None and getattr(mgr, "is_enabled", False):
        try:
            rel_path = str(_Path(png_path).relative_to(mgr.htmlpath.parent))
            lifecycle.add_artifact(request.node.nodeid, f"Loss Curve [{mode}/{label}]", rel_path, str(png_path))
        except Exception as e:  # noqa: BLE001
            log.warning("loss curve: could not register report link (%s)", e)

    if verdict is not None:
        _decreasing, _slope, detail = verdict
        log.info("loss curve: %s", detail)

    if verdict is None:
        pytest.skip(f"loss curve needs >= 2 sampled points (got {len(points)})")
    decreasing, _slope, detail = verdict
    if cfg.enforce and not decreasing:
        pytest.fail(f"training loss is not decreasing: {detail}")


# ---------- reporting ----------


def _write_metric_results_html(training_res_dict, request):
    """Write ALL metric verdicts to ONE HTML file in the report bundle dir."""
    metric_rows = training_res_dict.get("metric_rows") or []
    mgr = getattr(request.config, "_html_report_manager", None)
    if not metric_rows or mgr is None or not getattr(mgr, "is_enabled", False):
        return
    mode = training_res_dict.get("mode", "")
    try:
        out_dir = mgr.log_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "metric_results.html"
        body = ""
        for r in metric_rows:
            color = _STATUS_COLORS.get(r["status"], "#000000")
            body += (
                "<tr>"
                f"<td>{_html.escape(str(r.get('sweep', '-')))}</td>"
                f"<td>{_html.escape(str(r['metric']))}</td>"
                f"<td>{_html.escape(str(r['expected']))}</td>"
                f"<td>{_html.escape(str(r['actual']))}</td>"
                f"<td>{_html.escape(str(r['unit']))}</td>"
                f"<td style=\"color:{color};font-weight:bold;\">{_html.escape(str(r['status']))}</td>"
                "</tr>"
            )
        title = f"Training Metric Results ({mode})" if mode else "Training Metric Results"
        doc = (
            f"<html><head><meta charset='utf-8'><title>{_html.escape(title)}</title></head>"
            f"<body><h2>{_html.escape(title)}</h2>"
            "<table border='1' cellpadding='6' cellspacing='0'>"
            "<tr><th>Sweep</th><th>Metric</th><th>Expected</th><th>Actual</th><th>Unit</th><th>Status</th></tr>"
            f"{body}</table></body></html>"
        )
        path.write_text(doc, encoding="utf-8")
        log.info("wrote metric results HTML: %s", path)
    except Exception as e:  # noqa: BLE001 - reporting must never break the run
        log.warning("could not write metric results HTML: %s", e)


def _print_sweep_tables(training_res_dict):
    """Log a per-sweep metric table + loss curve to the console."""
    sweeps = training_res_dict.get("sweeps", {})
    mode = training_res_dict.get("mode", "")
    if not sweeps:
        log.info("no sweep results to print")
        return
    for sweep_name, rec in sweeps.items():
        results = rec.get("results", {})
        rows = []
        for short, unit in TRAINING_METRICS:
            val = results.get("training." + short)
            rows.append([short, f"{val:.4f}" if isinstance(val, float) else str(val), unit])
        log.info(
            "\n[%s | sweep %s]\n%s",
            mode,
            sweep_name,
            tabulate(rows, headers=["Metric", "Value", "Unit"], tablefmt="github"),
        )
        loss_rows = [[s["step"], f"{s['loss']:.6f}"] for s in rec.get("step_metrics", []) if "loss" in s]
        if loss_rows:
            log.info(
                "\nLoss Curve [%s]:\n%s", sweep_name, tabulate(loss_rows, headers=["Step", "Loss"], tablefmt="github")
            )


def _print_checkpoint_io(training_res_dict):
    """Log the checkpoint save/load I/O results as a separate section, printed
    after the per-sweep metric tables and loss curves. No-op when the (opt-in)
    checkpoint_resume test did not run."""
    io = training_res_dict.get("checkpoint_io")
    if not io:
        return
    tol = io.get("loss_tolerance")

    def _bound(mx):
        return f"<= {mx}" if (mx and mx > 0) else "record"

    rows = [
        [
            "checkpoint_save_seconds",
            _format_value(io.get("save_seconds")),
            _bound(io.get("save_max")),
            io.get("save_status"),
        ],
        [
            "checkpoint_load_seconds",
            _format_value(io.get("load_seconds")),
            _bound(io.get("load_max")),
            io.get("load_status"),
        ],
    ]
    log.info(
        "\n[Checkpoint I/O]  resumed_step=%s  loss_delta=%s (tol=%s)\n%s",
        io.get("resumed_step"),
        _format_value(io.get("loss_delta")),
        tol,
        tabulate(rows, headers=["Metric", "Value (s)", "Threshold", "Status"], tablefmt="github"),
    )


def print_results_table(training_res_dict, request):
    """Summarize all sweeps: console tables, single metric-results HTML, and a
    consolidated PASS/FAIL summary recorded via globals.error_list for the pytest
    final summary."""
    if not training_res_dict.get("sweeps"):
        log.info("training_res_dict empty, nothing to print")
        return

    _print_sweep_tables(training_res_dict)
    _print_checkpoint_io(training_res_dict)
    _write_metric_results_html(training_res_dict, request)

    failures = training_res_dict.get("metric_failures", [])
    globals.error_list = []
    for f in failures:
        fail_test(f)
    update_test_result()


def teardown(orch, lifecycle, request):
    """Final stage: explicit container teardown."""
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
