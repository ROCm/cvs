'''Report RCCL cases as pytest sub-tests and per-case HTML rows.'''

from cvs.lib.report.benchmark_metric_registry import record_benchmark_metric_rows

RCCL_CASE_TESTS = ('test_rccl_perf', 'test_rccl_pairwise', 'test_rccl_incremental')


def _measurement(actual, threshold, unit):
    if actual is None:
        return ''
    text = f': {actual:.2f} {unit}'
    if threshold is not None:
        text += f' (threshold >= {threshold:.2f} {unit})'
    return text


class RcclCaseReporter:
    """Report cases while leaving cluster work and parent failure gating outside subtests."""

    def __init__(self, request, subtests):
        self._node = request.node
        self._subtests = subtests
        self.rows = []

    def report(self, case_id, label, passed, reason='', actual=None, threshold=None, unit=None, **context):
        self.rows.append(
            {
                'node': '',
                'metric': case_id,
                'label': label,
                'actual': actual,
                'unit': unit,
                'spec': None if threshold is None else {'kind': '>=', 'value': threshold},
                'enforced': True,
                'status': 'pass' if passed else 'fail',
                'reason': '' if passed else reason,
            }
        )
        record_benchmark_metric_rows(self._node, self.rows)
        with self._subtests.test(**context):
            assert passed, reason

    def report_verdicts(self, verdicts, collective):
        """Report each verifier comparison with its collective, check and row labels."""
        for index, verdict in enumerate(verdicts):
            labels = {key: verdict[key] for key in ('check', 'dtype', 'size') if verdict.get(key) is not None}
            name = ' '.join([collective] + [f'size={v}' if k == 'size' else str(v) for k, v in labels.items()])
            self.report(
                f'{index}:{verdict["check"]}',
                name + _measurement(verdict.get('actual'), verdict.get('threshold'), verdict.get('unit')),
                verdict['status'] == 'pass',
                verdict['message'],
                actual=verdict.get('actual'),
                threshold=verdict.get('threshold'),
                unit=verdict.get('unit'),
                collective=collective,
                **labels,
            )

    def report_phase(self, phase, node, run_label, passed, reason, best_bw=None, min_bw=None):
        """Report a pairwise or incremental candidate."""
        self.report(
            f'phase{phase}:{node}',
            f'phase {phase} {node} ({run_label})' + _measurement(best_bw, min_bw, 'GB/s'),
            passed,
            reason,
            actual=best_bw,
            threshold=min_bw,
            unit=None if best_bw is None else 'GB/s',
            phase=phase,
            node=node,
        )
