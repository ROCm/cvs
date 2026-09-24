'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Per-iteration Megatron-LM / Primus log parsing for training curves.

parse_iteration_metrics — dialect-specific fields from each ``iteration N/`` line
sample_metric_curve     — downsample a field the same way as the loss sampler
'''

import re

_NUM = r'([0-9.eE+\-]+)'
_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_ITER = re.compile(r'iteration\s+(\d+)\s*/\s*\d+', re.I)

# Megatron-LM: single values on the iteration line (no tokens/s/GPU).
MEGATRON_FIELDS = {
    'elapsed_ms': [r'elapsed time per iteration \(ms\):\s*' + _NUM],
    'global_batch_size': [r'global batch size:\s*(\d+)'],
    'throughput_per_gpu': [r'throughput per GPU(?:\s*\([^)]*\))?\s*:\s*' + _NUM],
    'learning_rate': [r'learning rate:\s*' + _NUM],
    'loss': [r'\blm loss:\s*' + _NUM],
    'grad_norm': [r'grad norm:\s*' + _NUM],
}

# Primus: throughput/tokens may be inst/avg as X/Y — capture instantaneous X.
PRIMUS_FIELDS = {
    'elapsed_ms': [r'elapsed time per iteration \(ms\):\s*' + _NUM],
    'global_batch_size': [r'global batch size:\s*(\d+)'],
    'throughput_per_gpu': [r'throughput per GPU \(TFLOP/s/GPU\):\s*' + _NUM],
    'tokens_per_gpu': [
        r'tokens/s/GPU inst/harmonic mean:\s*' + _NUM,
        r'tokens per GPU \(tokens/s/GPU\):\s*' + _NUM,
    ],
    'learning_rate': [r'learning rate:\s*' + _NUM],
    'loss': [r'\blm loss:\s*' + _NUM],
    'grad_norm': [r'grad norm:\s*' + _NUM],
}

_DIALECTS = {
    'megatron': MEGATRON_FIELDS,
    'primus': PRIMUS_FIELDS,
}

CURVE_STORE_KEYS = (
    ('loss', '_loss_curve'),
    ('grad_norm', '_grad_norm_curve'),
    ('throughput_per_gpu', '_throughput_curve'),
    ('tokens_per_gpu', '_tokens_curve'),
)


def dialect_from_image(image):
    if image and 'primus' in str(image).lower():
        return 'primus'
    return 'megatron'


def _strip_ansi(text):
    return _ANSI.sub('', text or '')


def _first_float(line, patterns):
    for pat in patterns:
        m = re.search(pat, line, re.I)
        if not m:
            continue
        try:
            return float(m.group(1))
        except (TypeError, ValueError, IndexError):
            continue
    return None


def _tokens_per_gpu_from_elapsed(gbs, seq_length, elapsed_ms, world_size):
    try:
        gbs = float(gbs)
        seq_length = float(seq_length)
        elapsed_ms = float(elapsed_ms)
        world_size = float(world_size)
    except (TypeError, ValueError):
        return None
    if elapsed_ms <= 0 or world_size <= 0 or seq_length <= 0 or gbs <= 0:
        return None
    return gbs * seq_length * 1000.0 / (elapsed_ms * world_size)


def parse_iteration_metrics(log_text, dialect='megatron', seq_length=None, world_size=None):
    """Extract per-step fields from Megatron-LM or Primus iteration lines.

    Megatron-LM does not print tokens/s/GPU per step; when ``seq_length`` and
    ``world_size`` are given, tokens/GPU/s is computed with the training-script
    formula using that step's elapsed ms (not the run mean).

    Duplicate steps (multi-rank Primus logs) keep the last parsed row.
    """
    fields = _DIALECTS.get(dialect)
    if fields is None:
        raise ValueError("dialect must be 'megatron' or 'primus', got %r" % (dialect,))

    text = _strip_ansi(log_text)
    by_step = {}
    for m in _ITER.finditer(text):
        step = int(m.group(1))
        line_end = text.find('\n', m.start())
        line = text[m.start() : line_end if line_end >= 0 else None]
        row = {'step': step}
        for key, patterns in fields.items():
            val = _first_float(line, patterns)
            if val is not None:
                row[key] = val
        if dialect == 'megatron' and 'tokens_per_gpu' not in row:
            tgs = _tokens_per_gpu_from_elapsed(
                row.get('global_batch_size'),
                seq_length,
                row.get('elapsed_ms'),
                world_size,
            )
            if tgs is not None:
                row['tokens_per_gpu'] = tgs
        by_step[step] = row
    return [by_step[s] for s in sorted(by_step)]


def sample_metric_curve(rows, value_key, sample_every=10, milestone_steps=None):
    """Downsample ``value_key`` with the same stride / first / last rules as loss."""
    milestones = set(milestone_steps or [])
    every = sample_every if sample_every and sample_every > 0 else 1
    keyed = [
        s
        for s in (rows or [])
        if s.get('step') is not None and isinstance(s.get(value_key), (int, float))
    ]
    if not keyed:
        return []
    first_step = keyed[0]['step']
    last_step = keyed[-1]['step']
    picked = {}
    for s in keyed:
        step = s['step']
        if step % every == 0 or step in milestones or step in (first_step, last_step):
            picked[step] = s[value_key]
    return [(step, picked[step]) for step in sorted(picked)]


def parse_all_loss_points(log_text, dialect='megatron'):
    """Loss-only view of ``parse_iteration_metrics`` for existing slope-gate callers."""
    return parse_iteration_metrics(log_text, dialect)


def sample_loss_curve(step_metrics, sample_every=10, milestone_steps=None):
    return sample_metric_curve(step_metrics, 'loss', sample_every, milestone_steps)


def sample_training_curves(rows, sample_every=10, milestone_steps=None):
    """Sample loss / grad_norm / TFLOPS / tokens into ``_``-prefixed store lists."""
    out = {}
    for value_key, store_key in CURVE_STORE_KEYS:
        points = sample_metric_curve(rows, value_key, sample_every, milestone_steps)
        if points:
            out[store_key] = [[step, val] for step, val in points]
    return out
