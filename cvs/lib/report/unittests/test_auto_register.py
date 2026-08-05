'''Tests for automatic inference report preset registration.'''

from types import SimpleNamespace

from cvs.lib.report.auto_register import try_auto_register_inference_suite_report
from cvs.lib.report.registry import register_suite_report
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


def test_auto_register_skips_when_already_configured():
    cfg = make_inference_report_config(
        suite_id="x",
        results_columns=(),
        metric_units={},
        tier_metric_specs=lambda _c, _t: {},
    )
    config = SimpleNamespace(_suite_name="missing_module")
    register_suite_report(config, cfg)
    assert try_auto_register_inference_suite_report(config) is False


def test_auto_register_missing_profile():
    config = SimpleNamespace(_suite_name="no_such_suite_xyz")
    assert try_auto_register_inference_suite_report(config) is False
