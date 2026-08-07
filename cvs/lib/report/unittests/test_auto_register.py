'''Tests for automatic deck profile registration.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.auto_register import try_auto_register_suite_report
from cvs.lib.report.registry import register_suite_report
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


class TestAutoRegister(unittest.TestCase):
    def test_skips_when_already_configured(self):
        cfg = make_inference_report_config(
            suite_id="x",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
        )
        config = SimpleNamespace(_suite_name="missing_module")
        register_suite_report(config, cfg)
        self.assertFalse(try_auto_register_suite_report(config))

    def test_missing_profile(self):
        config = SimpleNamespace(_suite_name="no_such_suite_xyz")
        self.assertFalse(try_auto_register_suite_report(config))


if __name__ == "__main__":
    unittest.main()
