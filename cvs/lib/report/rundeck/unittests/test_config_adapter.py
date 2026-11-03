'''Tests for non-sweep profile hooks.'''

import unittest

from cvs.lib.report.profiles.hooks.rccl_run_card import rccl_run_card_display
from cvs.lib.report.rundeck.config_adapter import resolve_report_config


class TestConfigAdapter(unittest.TestCase):
    def test_series_and_matrix_resolve_custom_hooks(self):
        hook = "cvs.lib.report.profiles.hooks.rccl_run_card:rccl_run_card_display"
        for builder in ("series", "matrix"):
            with self.subTest(builder=builder):
                config = resolve_report_config(
                    {
                        "suite_id": "custom",
                        "dataset_builder": builder,
                        "hooks": {"run_card_display": hook, "launch_provenance": hook},
                    }
                )
                self.assertIs(config.run_card_display_builder, rccl_run_card_display)
                self.assertIs(config.launch_provenance_builder, rccl_run_card_display)


if __name__ == "__main__":
    unittest.main()
