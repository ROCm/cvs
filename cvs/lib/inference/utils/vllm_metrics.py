'''Canonical metric contract for CVS vLLM benchmark results.'''

import math
from collections import namedtuple
from types import MappingProxyType

from cvs.lib.utils.gpu import GPU_METRICS as _SHARED_GPU_METRICS


MetricDefinition = namedtuple(
    'MetricDefinition',
    (
        'name',
        'raw_source',
        'derivation',
        'required_inputs',
        'unit',
        'datasource',
        'category',
        'direction',
    ),
)

METRIC_CONTRACT = MappingProxyType({'id': 'vllm-bare', 'version': 1})
_METADATA_FIELDS = frozenset(
    {
        'date',
        'endpoint_type',
        'backend',
        'label',
        'model_id',
        'tokenizer_id',
        'burstiness',
        'request_rate',
    }
)


def _raw(name, unit, category, direction, raw_source=None):
    source = raw_source or name
    return MetricDefinition(name, source, 'identity', (source,), unit, 'client', category, direction)


def _derived(name, unit, category, direction, derivation, required_inputs):
    return MetricDefinition(name, None, derivation, tuple(required_inputs), unit, 'client', category, direction)


def _gpu_definitions():
    definitions = []
    for name, unit in _SHARED_GPU_METRICS:
        if name == 'peak_gpu_memory_mb':
            derivation = 'poll_max'
            required_inputs = ('gpu.used_vram',)
            direction = 'max'
        elif name == 'model_load_memory_mb':
            derivation = 'snapshot_delta'
            required_inputs = ('gpu.used_vram.before', 'gpu.used_vram.after')
            direction = 'max'
        elif name == 'model_load_s':
            derivation = 'server_readiness_elapsed'
            required_inputs = ('server_start', 'server_ready')
            direction = 'max'
        elif name == 'gpu_bandwidth_util_pct':
            derivation = 'poll_mean'
            required_inputs = ('gpu.umc_activity',)
            direction = 'min'
        elif name == 'gpu_compute_util_pct':
            derivation = 'poll_mean'
            required_inputs = ('gpu.gfx_activity',)
            direction = 'min'
        else:
            raise ValueError(f'unsupported shared GPU metric {name!r}')
        definitions.append(
            MetricDefinition(
                name,
                None,
                derivation,
                required_inputs,
                unit,
                'gpu',
                'gpu',
                direction,
            )
        )
    return tuple(definitions)


METRIC_REGISTRY = (
    _raw('max_concurrency', '-', 'run_health', 'min'),
    _raw('max_concurrent_requests', '-', 'run_health', 'min'),
    _raw('num_prompts', '-', 'run_health', 'min'),
    _raw('completed', '-', 'run_health', 'min'),
    _raw('failed', '-', 'run_health', 'max'),
    _derived(
        'success_rate',
        '-',
        'run_health',
        'min',
        'success_rate',
        ('completed', 'failed'),
    ),
    _raw('duration', 's', 'run_health', 'max'),
    _raw('request_throughput', 'req/s', 'throughput', 'min'),
    _raw('goodput', 'req/s', 'throughput', 'min', raw_source='request_goodput'),
    _raw('output_throughput', 'tok/s', 'throughput', 'min'),
    _raw('total_token_throughput', 'tok/s', 'throughput', 'min'),
    _derived(
        'per_gpu_throughput',
        'tok/s',
        'throughput',
        'min',
        'per_gpu_throughput',
        ('total_token_throughput', 'tp', 'pp'),
    ),
    _derived(
        'decode_throughput_p50',
        'tok/s',
        'throughput',
        'min',
        'decode_throughput_p50',
        ('median_tpot_ms',),
    ),
    _raw('max_output_tokens_per_s', 'tok/s', 'throughput', 'min'),
    _raw('rtfx', '-', 'throughput', 'min'),
    _raw('total_input_tokens', '-', 'throughput', 'min'),
    _raw('total_output_tokens', '-', 'throughput', 'min'),
    _raw('mean_ttft_ms', 'ms', 'ttft', 'max'),
    _raw('median_ttft_ms', 'ms', 'ttft', 'max'),
    _raw('std_ttft_ms', 'ms', 'ttft', 'max'),
    _raw('p50_ttft_ms', 'ms', 'ttft', 'max'),
    _raw('p90_ttft_ms', 'ms', 'ttft', 'max'),
    _raw('p95_ttft_ms', 'ms', 'ttft', 'max'),
    _raw('p99_ttft_ms', 'ms', 'ttft', 'max'),
    _derived(
        'normalized_ttft_ms_per_tok',
        'ms/tok',
        'ttft',
        'max',
        'normalized_ttft_ms_per_tok',
        ('mean_ttft_ms', 'isl'),
    ),
    _raw('mean_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('median_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('std_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('p50_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('p90_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('p95_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('p99_tpot_ms', 'ms', 'tpot', 'max'),
    _raw('mean_itl_ms', 'ms', 'itl', 'max'),
    _raw('median_itl_ms', 'ms', 'itl', 'max'),
    _raw('std_itl_ms', 'ms', 'itl', 'max'),
    _raw('p50_itl_ms', 'ms', 'itl', 'max'),
    _raw('p90_itl_ms', 'ms', 'itl', 'max'),
    _raw('p95_itl_ms', 'ms', 'itl', 'max'),
    _raw('p99_itl_ms', 'ms', 'itl', 'max'),
    _derived(
        'decode_latency_ratio',
        '-',
        'itl',
        'max',
        'decode_latency_ratio',
        ('p99_itl_ms', 'p50_itl_ms'),
    ),
    _raw('mean_e2el_ms', 'ms', 'e2el', 'max'),
    _raw('median_e2el_ms', 'ms', 'e2el', 'max'),
    _raw('std_e2el_ms', 'ms', 'e2el', 'max'),
    _raw('p50_e2el_ms', 'ms', 'e2el', 'max'),
    _raw('p90_e2el_ms', 'ms', 'e2el', 'max'),
    _raw('p95_e2el_ms', 'ms', 'e2el', 'max'),
    _raw('p99_e2el_ms', 'ms', 'e2el', 'max'),
    *_gpu_definitions(),
    MetricDefinition(
        'queue_time_p50_ms',
        None,
        'histogram_quantile_p50',
        ('vllm:request_queue_time_seconds.before', 'vllm:request_queue_time_seconds.after'),
        'ms',
        'prometheus',
        'prometheus',
        'max',
    ),
    MetricDefinition(
        'queue_time_p95_ms',
        None,
        'histogram_quantile_p95',
        ('vllm:request_queue_time_seconds.before', 'vllm:request_queue_time_seconds.after'),
        'ms',
        'prometheus',
        'prometheus',
        'max',
    ),
    MetricDefinition(
        'prefill_time_p50_ms',
        None,
        'histogram_quantile_p50',
        ('vllm:request_prefill_time_seconds.before', 'vllm:request_prefill_time_seconds.after'),
        'ms',
        'prometheus',
        'prometheus',
        'max',
    ),
    MetricDefinition(
        'prefill_time_p95_ms',
        None,
        'histogram_quantile_p95',
        ('vllm:request_prefill_time_seconds.before', 'vllm:request_prefill_time_seconds.after'),
        'ms',
        'prometheus',
        'prometheus',
        'max',
    ),
)

METRICS_BY_NAME = MappingProxyType({definition.name: definition for definition in METRIC_REGISTRY})
METRIC_NAMES = tuple(METRICS_BY_NAME)
METRIC_UNITS = MappingProxyType({definition.name: definition.unit for definition in METRIC_REGISTRY})
METRIC_CATEGORIES = tuple(dict.fromkeys(definition.category for definition in METRIC_REGISTRY))
CLIENT_METRICS = tuple(
    (definition.name, definition.unit) for definition in METRIC_REGISTRY if definition.datasource == 'client'
)
VLLM_GPU_METRICS = tuple(
    (definition.name, definition.unit) for definition in METRIC_REGISTRY if definition.datasource == 'gpu'
)
PROM_METRICS = tuple(
    (definition.name, definition.unit) for definition in METRIC_REGISTRY if definition.datasource == 'prometheus'
)
PROM_METRIC_UNITS = MappingProxyType(dict(PROM_METRICS))
_RAW_CLIENT_METRICS = MappingProxyType(
    {
        definition.raw_source: definition.name
        for definition in METRIC_REGISTRY
        if definition.datasource == 'client' and definition.raw_source is not None
    }
)
_DERIVED_CLIENT_METRICS = tuple(
    definition
    for definition in METRIC_REGISTRY
    if definition.datasource == 'client' and definition.raw_source is None
)
_NAMES_BY_DATASOURCE = MappingProxyType(
    {
        datasource: frozenset(
            definition.name for definition in METRIC_REGISTRY if definition.datasource == datasource
        )
        for datasource in ('client', 'gpu', 'prometheus')
    }
)

VLLM_RESULTS_COLUMNS = (
    ('Model', None),
    ('GPU', None),
    ('ISL', None),
    ('OSL', None),
    ('Policy', None),
    ('Conc', None),
    ('Host', None),
    ('Req/s', 'request_throughput'),
    ('Total tok/s', 'total_token_throughput'),
    ('Mean TTFT (ms)', 'mean_ttft_ms'),
    ('P95 TTFT (ms)', 'p95_ttft_ms'),
    ('Mean TPOT (ms)', 'mean_tpot_ms'),
    ('P95 TPOT (ms)', 'p95_tpot_ms'),
    ('P99 ITL (ms)', 'p99_itl_ms'),
    ('Goodput (req/s)', 'goodput'),
)


def is_finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_contract():
    return dict(METRIC_CONTRACT)


def metric_definitions():
    return METRIC_REGISTRY


def report_metric_units():
    return dict(METRIC_UNITS)


def tier_metric_specs(thresholds_cell, tier):
    return {
        definition.name: thresholds_cell[definition.name]
        for definition in METRIC_REGISTRY
        if definition.category == tier and definition.name in thresholds_cell
    }


def _number(value):
    if type(value) in (int, float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator, denominator):
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _derive_metric(definition, values, context):
    inputs = {name: context.get(name, values.get(name)) for name in definition.required_inputs}
    if definition.derivation == 'success_rate':
        completed = inputs['completed']
        failed = inputs['failed']
        total = None if completed is None or failed is None else completed + failed
        return _safe_ratio(completed, total)
    if definition.derivation == 'per_gpu_throughput':
        tp = _number(inputs['tp'])
        pp = _number(inputs['pp'])
        gpu_count = None if tp is None or pp is None else tp * pp
        return _safe_ratio(inputs['total_token_throughput'], gpu_count)
    if definition.derivation == 'decode_throughput_p50':
        return _safe_ratio(1000.0, inputs['median_tpot_ms'])
    if definition.derivation == 'normalized_ttft_ms_per_tok':
        return _safe_ratio(inputs['mean_ttft_ms'], inputs['isl'])
    if definition.derivation == 'decode_latency_ratio':
        return _safe_ratio(inputs['p99_itl_ms'], inputs['p50_itl_ms'])
    raise ValueError(f'unknown vLLM metric derivation {definition.derivation!r}')


def project_vllm_metrics(raw, *, tp, isl, pp='1', artifact_path='results'):
    '''Project one vLLM benchmark artifact into finite canonical bare metrics.'''
    if not isinstance(raw, dict):
        raise ValueError(f'vLLM results artifact must contain an object: {artifact_path}')

    metrics = {}
    for raw_name, value in raw.items():
        metric_name = _RAW_CLIENT_METRICS.get(raw_name)
        if metric_name is not None:
            if is_finite_number(value):
                metrics[metric_name] = value
            continue
        if raw_name in _METADATA_FIELDS or not is_finite_number(value):
            continue
        raise ValueError(f'unknown finite numeric vLLM result field {raw_name!r}: {artifact_path}')

    context = {'tp': tp, 'pp': pp, 'isl': isl}
    for definition in _DERIVED_CLIENT_METRICS:
        value = _derive_metric(definition, metrics, context)
        if is_finite_number(value):
            metrics[definition.name] = value
    return metrics


def merge_metric_sources(client_metrics, gpu_metrics, prometheus_metrics):
    '''Validate and disjointly merge the three vLLM metric datasources.'''
    merged = {}
    for datasource, source in (
        ('client', client_metrics),
        ('gpu', gpu_metrics),
        ('prometheus', prometheus_metrics),
    ):
        collisions = set(merged) & set(source)
        if collisions:
            raise ValueError(f'vLLM metric source collision: {sorted(collisions)}')
        unknown = set(source) - _NAMES_BY_DATASOURCE[datasource]
        if unknown:
            raise ValueError(f'unknown {datasource} vLLM metrics: {sorted(unknown)}')
        merged.update(source)
    return merged


def validate_threshold_spec(metric, spec):
    definition = METRICS_BY_NAME.get(metric)
    if definition is None:
        raise ValueError(f'unknown vLLM threshold metric {metric!r}')
    if type(spec.get('kind')) is not str or spec['kind'] != definition.direction:
        raise ValueError(
            f'{metric} threshold kind must be {definition.direction!r}, got {spec.get("kind")!r}'
        )
    if not is_finite_number(spec.get('value')):
        raise ValueError(f'{metric} threshold value must be a finite built-in int or float')


def metric_verdict(metric, actual, spec):
    '''Evaluate one canonical vLLM min/max threshold without coercion.'''
    try:
        validate_threshold_spec(metric, spec)
    except (AttributeError, KeyError, ValueError) as exc:
        return 'fail', str(exc)
    if not is_finite_number(actual):
        return 'fail', f'{metric}: actual must be a finite built-in int or float, got {actual!r}'

    direction = METRICS_BY_NAME[metric].direction
    target = spec['value']
    if direction == 'min' and actual < target:
        return 'fail', f'{metric}: actual {actual} < min {target}'
    if direction == 'max' and actual > target:
        return 'fail', f'{metric}: actual {actual} > max {target}'
    return 'pass', ''
