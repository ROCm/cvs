'''Recognise pytest sub-test reports in report hooks.'''

import inspect

try:
    from _pytest.subtests import SubtestReport as _BuiltinSubtestReport
except ImportError:
    _BuiltinSubtestReport = None
try:
    from pytest_subtests.plugin import SubTestReport as _PluginSubtestReport
except ImportError:
    _PluginSubtestReport = None

_SUBTEST_MODULES = ('/_pytest/subtests.py', '/pytest_subtests/plugin.py')


def is_subtest_report(report):
    """True for a finished sub-test report."""
    return any(cls is not None and isinstance(report, cls) for cls in (_BuiltinSubtestReport, _PluginSubtestReport))


def called_from_subtest_context():
    """True while a sub-test context builds its report.

    The inner makereport receives the parent item before the report becomes a
    SubtestReport, so its class cannot identify the context there.
    """
    frame = inspect.currentframe()
    while frame is not None:
        filename = frame.f_code.co_filename.replace('\\', '/')
        if frame.f_code.co_name == '__exit__' and filename.endswith(_SUBTEST_MODULES):
            return True
        frame = frame.f_back
    return False
