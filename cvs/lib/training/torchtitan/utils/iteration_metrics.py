'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Per-step TorchTitan log parsing for training curves.

TorchTitan's metrics logger prints one rank-0 line per step, for example:
    step: 10  loss: 8.12345  grad_norm: 0.1234  memory: 45.67GiB(12.34%)
    tps: 12,345  tflops: 234.56  mfu: 32.10%

parse_iteration_metrics — fields from each ``step:`` line
sample_metric_curve     — downsample a field the same way as the loss sampler
summarize_step_metrics  — post-warmup means used by the threshold gate
'''

import math
import re

_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_STEP = re.compile(r'\bstep:\s*(\d+)\b', re.I)

# Early steps are noisy (compile / cache). Curves and means start after this fraction.
WARMUP_FRAC = 0.1

# tps / tflops are printed with thousands separators (``12,345``).
_COMMA_NUM = r'([0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?)'
_PLAIN_NUM = r'([0-9.eE+\-]+)'

_FIELDS = {
    'loss': [r'\bloss:\s*' + _PLAIN_NUM],
    'grad_norm': [r'grad_norm:\s*' + _PLAIN_NUM],
    'memory_gib': [r'memory:\s*' + _PLAIN_NUM + r'\s*GiB', r'mem:\s*' + _PLAIN_NUM + r'\s+GB'],
    'tps': [r'\btps:\s*' + _COMMA_NUM, r'tok/s:\s*' + _COMMA_NUM],
    'tflops': [r'\btflops:\s*' + _COMMA_NUM],
    'mfu': [r'\bmfu:\s*' + _PLAIN_NUM],
    'learning_rate': [r'\blr:\s*' + _PLAIN_NUM],
    'elapsed_s': [r'\btime:\s*' + _PLAIN_NUM + r'\s*s\b', r'end_to_end\(s\):\s*' + _PLAIN_NUM],
}

# Gate keys (training.<name>) and the per-step field they summarize.
_SUMMARY_FIELDS = (
    ('tokens_per_sec', 'tps'),
    ('loss', 'loss'),
    ('mem_usage_gb', 'memory_gib'),
    ('tflops', 'tflops'),
    ('grad_norm', 'grad_norm'),
)

CURVE_STORE_KEYS = (
    ('loss', '_loss_curve'),
    ('perplexity', '_perplexity_curve'),
    ('grad_norm', '_grad_norm_curve'),
    ('learning_rate', '_learning_rate_curve'),
    ('tps', '_tps_curve'),
    ('tflops', '_tflops_curve'),
    ('memory_gib', '_memory_curve'),
    ('mfu', '_mfu_curve'),
)


def _strip_ansi(text):
    return _ANSI.sub('', text or '')


def _first_float(line, patterns):
    for pat in patterns:
        match = re.search(pat, line, re.I)
        if not match:
            continue
        raw = match.group(1).replace(',', '')
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _perplexity_from_nll(loss):
    """Token perplexity from mean NLL in nats (TorchTitan global average loss)."""
    try:
        ppl = math.exp(float(loss))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(ppl):
        return None
    return ppl


def _elapsed_ms_from_tps(tps, global_batch_size, seq_length, world_size):
    """Step time when the log prints per-device tps but not elapsed time.

    TorchTitan's ``tps`` is tokens/s/device. Tokens that device consumes in one
    step are ``global_batch_size * seq_length / world_size``.
    """
    try:
        tps = float(tps)
        global_batch_size = float(global_batch_size)
        seq_length = float(seq_length)
        world_size = float(world_size)
    except (TypeError, ValueError):
        return None
    if tps <= 0 or global_batch_size <= 0 or seq_length <= 0 or world_size <= 0:
        return None
    tokens_per_device = global_batch_size * seq_length / world_size
    return tokens_per_device / tps * 1000.0


def parse_iteration_metrics(log_text, seq_length=None, global_batch_size=None, world_size=None):
    """Extract per-step fields from TorchTitan ``step:`` lines.

    Duplicate steps (multi-rank copies of the same line) keep the last row.
    When ``seq_length``, ``global_batch_size``, and ``world_size`` are given and
    the line has ``tps`` but no explicit step time, ``elapsed_ms`` is derived
    from that step's tps.
    """
    text = _strip_ansi(log_text)
    by_step = {}
    for match in _STEP.finditer(text):
        step = int(match.group(1))
        line_end = text.find('\n', match.start())
        line = text[match.start() : line_end if line_end >= 0 else None]
        row = {'step': step}
        for key, patterns in _FIELDS.items():
            val = _first_float(line, patterns)
            if val is not None:
                row[key] = val
        if 'loss' not in row and 'tps' not in row:
            continue
        if 'loss' in row:
            ppl = _perplexity_from_nll(row['loss'])
            if ppl is not None:
                row['perplexity'] = ppl
        if 'elapsed_ms' not in row and 'elapsed_s' in row:
            row['elapsed_ms'] = row['elapsed_s'] * 1000.0
        if 'elapsed_ms' not in row and 'tps' in row:
            elapsed = _elapsed_ms_from_tps(row['tps'], global_batch_size, seq_length, world_size)
            if elapsed is not None:
                row['elapsed_ms'] = elapsed
        by_step[step] = row
    rows = [by_step[step] for step in sorted(by_step)]
    planned = max((row['step'] for row in rows), default=0)
    for row in rows:
        row['total'] = planned
    return rows


def _warmup_cutoff(rows, warmup_frac=None):
    """First ``warmup_frac`` of planned (or observed) steps are dropped from curves."""
    frac = WARMUP_FRAC if warmup_frac is None else warmup_frac
    if not frac or frac <= 0:
        return 0
    planned = [int(row['total']) for row in (rows or []) if row.get('total')]
    observed = [int(row['step']) for row in (rows or []) if row.get('step') is not None]
    last = max(planned) if planned else (max(observed) if observed else 0)
    return int(last * frac)


def sample_metric_curve(rows, value_key, sample_every=10, milestone_steps=None, warmup_frac=None):
    """Downsample ``value_key`` after dropping the warmup prefix.

    Keeps a point when its step is a multiple of ``sample_every``, is one of
    the milestone steps, or is the first or last step that remains after warmup.
    """
    cutoff = _warmup_cutoff(rows, warmup_frac)
    milestones = set(milestone_steps or [])
    every = sample_every if sample_every and sample_every > 0 else 1
    keyed = [
        row
        for row in (rows or [])
        if row.get('step') is not None and row['step'] > cutoff and isinstance(row.get(value_key), (int, float))
    ]
    if not keyed:
        return []
    first_step = keyed[0]['step']
    last_step = keyed[-1]['step']
    picked = {}
    for row in keyed:
        step = row['step']
        if step % every == 0 or step in milestones or step in (first_step, last_step):
            picked[step] = row[value_key]
    return [(step, picked[step]) for step in sorted(picked)]


def parse_all_loss_points(log_text):
    """Loss-bearing rows from ``parse_iteration_metrics`` for slope-gate callers."""
    return parse_iteration_metrics(log_text)


def sample_loss_curve(step_metrics, sample_every=10, milestone_steps=None):
    return sample_metric_curve(step_metrics, 'loss', sample_every, milestone_steps)


def sample_training_curves(rows, sample_every=10, milestone_steps=None):
    """Sample loss / PPL / grad_norm / tps / tflops / memory into ``_``-prefixed lists."""
    out = {}
    for value_key, store_key in CURVE_STORE_KEYS:
        points = sample_metric_curve(rows, value_key, sample_every, milestone_steps)
        if points:
            out[store_key] = [[step, val] for step, val in points]
    return out


def _percentile(values, q):
    """Linear-interpolated percentile (q in [0, 100])."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def derive_step_time_stats(rows, warmup_frac=None):
    """P50 / p95 of per-step elapsed ms after warmup.

    Empty when the log has no step time and tps could not be inverted.
    Mean is omitted — it duplicates a single elapsed-time gate if one is added later.
    """
    cutoff = _warmup_cutoff(rows, warmup_frac)
    samples = [
        float(row['elapsed_ms'])
        for row in (rows or [])
        if row.get('step') is not None
        and row['step'] > cutoff
        and isinstance(row.get('elapsed_ms'), (int, float))
        and row['elapsed_ms'] > 0
    ]
    if not samples:
        return {}
    return {
        'step_time_p50_ms': [str(_percentile(samples, 50))],
        'step_time_p95_ms': [str(_percentile(samples, 95))],
    }


def summarize_step_metrics(rows, warmup_frac=None):
    """Post-warmup mean of each gate metric, as a one-element string list.

    The threshold check reads ``values[-1]``. A single mean avoids treating the
    last (often still-noisy) step as the run result. Metrics with no samples
    are omitted so an optional threshold can skip them.
    """
    cutoff = _warmup_cutoff(rows, warmup_frac)
    out = {}
    for dest, src in _SUMMARY_FIELDS:
        values = [
            float(row[src])
            for row in (rows or [])
            if row.get('step') is not None and row['step'] > cutoff and isinstance(row.get(src), (int, float))
        ]
        if values:
            out[dest] = [str(sum(values) / len(values))]
    return out
