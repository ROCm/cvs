'''Foundation tests for the CVS Run Deck package (Milestone 1).'''

from types import SimpleNamespace

import pytest

from cvs.lib.report.presets.builder import make_inference_report_config
from cvs.lib.report.profile import DEFAULT_SOURCES, sources_for_profile
from cvs.lib.report.registry import (
    bind_session_results,
    clear_session_results,
    get_session_results,
    get_sources,
    register_deck_profile,
    register_suite_report,
)
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets, register_dataset_builder


def test_default_sources_for_legacy_preset():
    cfg = make_inference_report_config(
        suite_id="demo",
        results_columns=(),
        metric_units={},
        tier_metric_specs=lambda _c, _t: {},
    )
    assert sources_for_profile(cfg) == DEFAULT_SOURCES


def test_get_sources_from_registered_preset():
    cfg = make_inference_report_config(
        suite_id="demo",
        results_columns=(),
        metric_units={},
        tier_metric_specs=lambda _c, _t: {},
    )
    config = SimpleNamespace()
    register_suite_report(config, cfg)
    assert get_sources(config) == DEFAULT_SOURCES


def test_session_store_accepts_cvs_results_dict_alias():
    clear_session_results()
    bind_session_results(cvs_results_dict={"cell": {"host": {"m": 1}}})
    store = get_session_results()
    assert store["cvs_results_dict"] == {"cell": {"host": {"m": 1}}}
    assert store["inf_res_dict"] == store["cvs_results_dict"]


def test_json_profile_sources_override_defaults():
    profile = {
        "schema_version": 1,
        "sources": {
            "results": "inf_res_dict",
            "variant": "variant_config",
            "lifecycle": "lifecycle",
        },
    }
    config = SimpleNamespace()
    register_deck_profile(config, profile)
    assert get_sources(config)["results"] == "inf_res_dict"


def test_build_datasets_returns_empty_when_builder_missing():
    assert build_datasets("sweep", {}, {}) == {}


def test_register_dataset_builder():
    @register_dataset_builder("demo")
    def _demo_builder(sources, profile):
        return {"demo": True}

    assert build_datasets("demo", {}, {}) == {"demo": True}


def test_pytest_hooks_resolve_inf_res_dict_alias(monkeypatch):
    from cvs.lib.report import pytest_hooks

    cfg = make_inference_report_config(
        suite_id="demo",
        results_columns=(),
        metric_units={},
        tier_metric_specs=lambda _c, _t: {},
    )
    request = SimpleNamespace(
        config=SimpleNamespace(_suite_report_config=cfg),
        _finalizers=[],
    )

    def fake_getfixturevalue(name):
        if name == "inf_res_dict":
            return {"k": 1}
        if name == "variant_config":
            return object()
        if name == "lifecycle":
            return SimpleNamespace(report={})
        raise pytest.FixtureLookupError(name)

    request.getfixturevalue = fake_getfixturevalue
    request.addfinalizer = request._finalizers.append

    clear_session_results()
    gen = pytest_hooks.cvs_rundeck_bind_module_fixture(request, None)
    next(gen)
    for fn in request._finalizers:
        fn()
    with pytest.raises(StopIteration):
        next(gen)

    store = get_session_results()
    assert store["cvs_results_dict"] == {"k": 1}
