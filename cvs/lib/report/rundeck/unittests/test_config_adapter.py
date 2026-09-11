'''Unit tests for Run Deck profile configuration resolution.'''

import unittest

from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile


class TestConfigAdapter(unittest.TestCase):
    def test_series_profile_resolves_run_card_hook(self):
        profile = {
            "dataset_builder": "series",
            "suite_id": "demo",
            "metric_contract": {"id": "demo-series", "version": 1},
            "hooks": {
                "run_card_display": "cvs.lib.report.rundeck.config_builder:_default_run_card",
            },
        }

        config = build_inference_config_from_profile(profile)

        self.assertEqual(config.run_card_display_builder.__name__, "_default_run_card")
        self.assertFalse(config.interactive_viewer)
        self.assertEqual(config.metric_contract, {"id": "demo-series", "version": 1})


if __name__ == "__main__":
    unittest.main()
