'''Unit tests for deck profile source resolution.'''

import unittest

from cvs.lib.report.profile import DEFAULT_SOURCES, sources_for_profile
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


if __name__ == "__main__":
    unittest.main()
