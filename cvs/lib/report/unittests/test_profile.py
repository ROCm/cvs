'''Unit tests for deck profile source resolution and inheritance.'''

import unittest

from cvs.lib.report.profile import DEFAULT_SOURCES, load_json_profile, sources_for_profile
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


class TestProfile(unittest.TestCase):
    def test_default_sources_for_legacy_preset(self):
        cfg = make_inference_report_config(
            suite_id="demo",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
        )
        self.assertEqual(sources_for_profile(cfg), DEFAULT_SOURCES)

    def test_sglang_stems_share_one_profile(self):
        for stem in ("sglang_single", "sglang_distributed", "sglang_disagg_distributed"):
            profile = load_json_profile(stem)
            self.assertIsNotNone(profile, stem)
            self.assertEqual(profile["suite_id"], "sglang")
            self.assertEqual(profile["report_basename"], "sglang_run_deck")
            self.assertEqual(
                profile["hooks"]["run_card_display"],
                "cvs.lib.report.profiles.hooks.sglang_run_card:sglang_run_card_display",
            )

    def test_vllm_hooks_point_at_inference_parsing(self):
        profile = load_json_profile("vllm")
        self.assertEqual(
            profile["hooks"]["metric_units"],
            "cvs.lib.inference.utils.vllm_parsing:CLIENT_METRIC_UNITS",
        )
        self.assertNotIn("run_card_display", profile.get("hooks", {}))


if __name__ == "__main__":
    unittest.main()
