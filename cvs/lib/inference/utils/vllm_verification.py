'''Threshold verdict helpers for vLLM metric verification subtests.'''

from __future__ import annotations

from typing import Any, Mapping

from cvs.lib.inference.utils.vllm_parsing import CLIENT_METRICS
from cvs.lib.inference.utils.vllm_server_metrics import PROM_METRICS
from cvs.lib.utils.gpu import GPU_METRICS
from cvs.lib.utils.verdict import ThresholdViolation, evaluate_all


def metric_definitions() -> tuple[dict[str, str], ...]:
    '''Return ordered vLLM metric metadata keyed by fully qualified names.'''
    definitions = []
    for prefix, metrics, missing_status in (
        ('client.', CLIENT_METRICS, 'fail'),
        ('gpu.', GPU_METRICS, 'skip'),
        ('prom.', PROM_METRICS, 'skip'),
    ):
        for short_name, unit in metrics:
            definitions.append(
                {
                    'metric': f'{prefix}{short_name}',
                    'unit': unit,
                    'missing_status': missing_status,
                }
            )
    return tuple(definitions)


def report_metric_units() -> dict[str, str]:
    '''Return units for fully qualified metrics and vLLM's client shorthand.'''
    units = {definition['metric']: definition['unit'] for definition in metric_definitions()}
    units.update(
        {metric.removeprefix('client.'): unit for metric, unit in units.items() if metric.startswith('client.')}
    )
    return units


def active_metric_specs(
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    enforce_thresholds: bool,
) -> tuple[dict[str, Any], ...]:
    '''Return configured, enforced, non-informational metric specs in display order.'''
    if not enforce_thresholds:
        return ()

    specs = []
    for definition in metric_definitions():
        spec = thresholds.get(definition['metric'])
        if not isinstance(spec, Mapping) or spec.get('kind') == 'info':
            continue
        specs.append({**definition, 'spec': dict(spec)})
    return tuple(specs)


def evaluate_metric_verdicts(
    actuals_by_host: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Mapping[str, Any]],
    *,
    enforce_thresholds: bool,
) -> list[dict[str, Any]]:
    '''Evaluate all active vLLM metric specs without stopping after a failure.'''
    verdicts = []
    active_specs = active_metric_specs(thresholds, enforce_thresholds=enforce_thresholds)
    for host, actuals in actuals_by_host.items():
        for definition in active_specs:
            metric = definition['metric']
            value = actuals.get(metric)
            verdict = {
                'node': str(host),
                'metric': metric,
                'unit': definition['unit'],
                'actual': value,
                'spec': definition['spec'],
                'status': 'pass',
                'reason': '',
            }
            if value is None:
                verdict['status'] = definition['missing_status']
                verdict['reason'] = f'{metric}: value is None (metric unavailable for this run)'
            else:
                try:
                    evaluate_all(actuals, {metric: definition['spec']})
                except ThresholdViolation as exc:
                    verdict['status'] = 'fail'
                    verdict['reason'] = str(exc)
            verdicts.append(verdict)
    return verdicts
