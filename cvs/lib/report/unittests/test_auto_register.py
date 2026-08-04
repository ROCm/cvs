'''Tests for automatic inference report preset registration.'''

from types import SimpleNamespace

from cvs.lib.report.auto_register import try_auto_register_inference_suite_report
from cvs.lib.report.registry import get_resolved_profile, register_suite_report
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


def test_auto_register_loads_json_profile():
    config = SimpleNamespace(_suite_name="inferencex_atom_single")
    assert try_auto_register_inference_suite_report(config) is True
    preset = get_resolved_profile(config)
    assert preset is not None
    assert preset["suite_id"] == "inferencex_atom"


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


def test_auto_register_loads_inferencex_atom_single_profile():
    config = SimpleNamespace(_suite_name="inferencex_atom_single")
    assert try_auto_register_inference_suite_report(config) is True
    preset = get_resolved_profile(config)
    assert preset is not None
    suite_id = preset["suite_id"] if isinstance(preset, dict) else preset.suite_id
    assert suite_id == "inferencex_atom"
