"""
Extra FID samples after the timed benchmark, stopped before the command timeout.

Rank 0 picks the next step count. Every rank runs that same sample, so the
process group stays aligned. A 25% slack on the last measured pipe time keeps
the next sample from running into the deadline.
"""

import json
import os
import time


def estimate_sample_s(reference_pipe_s, reference_steps, nfe, budget_s):
    """Return a conservative duration for one sample at ``nfe`` steps."""
    if reference_pipe_s <= 0 or reference_steps <= 0 or budget_s <= 0:
        return budget_s + 1
    return reference_pipe_s * (float(nfe) / float(reference_steps)) * 1.25


def next_fid_sample(steps, counts, samples, elapsed, budget_s, reference_pipe_s, reference_steps):
    """Pick the step count with the fewest saved samples that still fits."""
    if budget_s <= 0:
        return None
    pending = [int(step) for step in steps if counts.get(int(step), 0) < samples]
    pending.sort(key=lambda step: (counts.get(step, 0), step))
    for step in pending:
        needed = estimate_sample_s(reference_pipe_s, reference_steps, step, budget_s)
        if elapsed + needed <= budget_s:
            return step
    return None


def broadcast_value(value, rank):
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return value
    box = [value if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def count_step_videos(directory):
    """MP4s saved for one NFE. Frame PNGs beside them are for the scorer, not extra samples."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for name in os.listdir(directory) if name.lower().endswith(".mp4"))


def rewrite_manifest_from_disk(path, steps, samples, budget_s):
    """Record videos that exist, so a failed export is not counted as a finished sample."""
    root = os.path.dirname(path)
    counts = {int(step): count_step_videos(os.path.join(root, f"nfe_{int(step)}")) for step in steps}
    previous = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
    write_manifest(path, budget_s, samples, counts, previous.get("produced") or [], previous.get("error") or "")


def write_manifest(path, budget_s, samples, counts, produced, error=""):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "budget_s": budget_s,
        "samples_requested": samples,
        "stopped_early": any(count < samples for count in counts.values()) or bool(error),
        "counts": {str(step): count for step, count in sorted(counts.items())},
        "produced": produced,
        "error": error,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"FID sweep manifest {path}: counts={payload['counts']} stopped_early={payload['stopped_early']}")


def run_fid_sweep(
    rank,
    steps,
    samples,
    budget_s,
    started,
    reference_pipe_s,
    reference_steps,
    generate_one,
    manifest_path,
):
    """Generate samples until the budget is gone. ``generate_one(nfe, index)`` returns seconds."""
    counts = {int(step): 0 for step in steps}
    produced = []
    error = ""
    try:
        while True:
            decision = None
            if rank == 0:
                elapsed = time.monotonic() - started
                step = next_fid_sample(
                    steps,
                    counts,
                    samples,
                    elapsed,
                    budget_s,
                    reference_pipe_s,
                    reference_steps,
                )
                if step is not None:
                    decision = (int(step), counts[int(step)])
            decision = broadcast_value(decision, rank)
            if not decision:
                break
            step, sample_index = decision
            elapsed_sample = float(generate_one(int(step), int(sample_index)))
            if rank == 0:
                counts[int(step)] += 1
                produced.append({"nfe": int(step), "index": int(sample_index), "seconds": elapsed_sample})
                if elapsed_sample > 0 and int(step) > 0:
                    reference_pipe_s = elapsed_sample * float(reference_steps) / float(step)
            reference_pipe_s = broadcast_value(reference_pipe_s if rank == 0 else None, rank)
    except Exception as exc:
        error = str(exc)
        print(f"FID sweep stopped: {exc}")
        raise
    finally:
        if rank == 0:
            write_manifest(manifest_path, budget_s, samples, counts, produced, error)
    return produced
