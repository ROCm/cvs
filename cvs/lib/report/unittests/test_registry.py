'''Unit tests for Run Deck session registry.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profile import DEFAULT_SOURCES
from cvs.lib.report.registry import (
    bind_session_results,
    clear_session_results,
    get_session_results,
    get_sources,
    register_deck_profile,
    register_suite_report,
)
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


class TestRegistry(unittest.TestCase):
    def setUp(self):
        clear_session_results()

    def tearDown(self):
        clear_session_results()

    def test_get_sources_from_registered_preset(self):
        cfg = make_inference_report_config(
            suite_id="demo",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
        )
        config = SimpleNamespace()
        register_suite_report(config, cfg)
        self.assertEqual(get_sources(config), DEFAULT_SOURCES)

    def test_session_store_accepts_cvs_results_dict_alias(self):
        bind_session_results(cvs_results_dict={"cell": {"host": {"m": 1}}})
        store = get_session_results()
        self.assertEqual(store["cvs_results_dict"], {"cell": {"host": {"m": 1}}})
        self.assertEqual(store["inf_res_dict"], store["cvs_results_dict"])

    def test_json_profile_sources_override_defaults(self):
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
        self.assertEqual(get_sources(config)["results"], "inf_res_dict")


if __name__ == "__main__":
    unittest.main()
