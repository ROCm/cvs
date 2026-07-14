'''Tests for automatic inference report preset registration.'''

from types import SimpleNamespace

from cvs.lib.report.auto_register import try_auto_register_inference_suite_report
from cvs.lib.report.presets.builder import make_inference_report_config
from cvs.lib.report.registry import get_suite_report_config, register_suite_report


def test_auto_register_loads_stem_preset(monkeypatch):
    cfg = make_inference_report_config(
        suite_id="stem_demo",
        results_columns=(),
        metric_units={},
        tier_metric_specs=lambda _c, _t: {},
    )
    fake = SimpleNamespace(STEM_DEMO_REPORT_CONFIG=cfg)
    monkeypatch.setitem(
        __import__("sys").modules,
        "cvs.lib.report.presets.stem_demo",
        fake,
    )

    config = SimpleNamespace(_suite_name="stem_demo")
    assert try_auto_register_inference_suite_report(config) is True
    assert get_suite_report_config(config) is cfg


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


def test_auto_register_missing_module():
    config = SimpleNamespace(_suite_name="no_such_suite_xyz")
    assert try_auto_register_inference_suite_report(config) is False


def test_auto_register_loads_inferencex_atom_single_preset():
    config = SimpleNamespace(_suite_name="inferencex_atom_single")
    assert try_auto_register_inference_suite_report(config) is True
    preset = get_suite_report_config(config)
    assert preset is not None
    assert preset.suite_id == "inferencex_atom"
