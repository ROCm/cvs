'''Threshold verdict helpers for vLLM metric verification subtests.'''

from cvs.lib.inference.utils.vllm_metrics import (
    METRIC_CATEGORIES,
    METRIC_CONTRACT,
    METRIC_NAMES,
    METRIC_REGISTRY,
    METRIC_UNITS,
    metric_contract,
    metric_definitions,
    metric_verdict,
    report_metric_units,
    tier_metric_specs,
)
from cvs.lib.inference.utils.vllm_metrics import is_finite_number as _is_finite_number

__all__ = [
    'METRIC_CATEGORIES',
    'METRIC_CONTRACT',
    'METRIC_NAMES',
    'METRIC_REGISTRY',
    'METRIC_UNITS',
    'active_metric_specs',
    'evaluate_metric_verdicts',
    'metric_contract',
    'metric_definitions',
    'metric_verdict',
    'report_metric_units',
    'reportable_metric_specs',
    'tier_metric_specs',
]


def active_metric_specs(thresholds, *, enforce_thresholds):
    '''Return configured and enforced metric specs in registry order.'''
    if not enforce_thresholds:
        return ()
    return reportable_metric_specs(thresholds)


def reportable_metric_specs(thresholds):
    '''Return configured metric specs in registry order.'''
    specs = []
    for definition in METRIC_REGISTRY:
        spec = thresholds.get(definition.name)
        if not isinstance(spec, dict):
            continue
        specs.append(
            {
                'metric': definition.name,
                'unit': definition.unit,
                'datasource': definition.datasource,
                'category': definition.category,
                'direction': definition.direction,
                'spec': dict(spec),
            }
        )
    return tuple(specs)


def evaluate_metric_verdicts(actuals_by_host, thresholds, *, enforce_thresholds):
    '''Build all vLLM rows before the parent emits threshold subtests.'''
    verdicts = []
    for host, actuals in actuals_by_host.items():
        for definition in METRIC_REGISTRY:
            metric = definition.name
            spec = thresholds.get(metric)
            value = actuals.get(metric)
            if not _is_finite_number(value) and spec is None:
                continue
            enforced = bool(enforce_thresholds and spec is not None)
            verdict = {
                'node': str(host),
                'metric': metric,
                'unit': definition.unit,
                'actual': value,
                'spec': spec,
                'enforced': enforced,
                'status': 'pass' if enforced else 'record',
                'reason': '',
            }
            if enforced:
                verdict['status'], verdict['reason'] = metric_verdict(metric, value, spec)
            elif spec is None:
                verdict['reason'] = 'metric produced without a configured threshold'
            else:
                verdict['reason'] = 'threshold enforcement disabled; no threshold asserted'
                if not _is_finite_number(value):
                    verdict['reason'] = (
                        f'{metric}: unavailable or non-finite actual recorded; no threshold asserted'
                    )
            verdicts.append(verdict)
    return verdicts
