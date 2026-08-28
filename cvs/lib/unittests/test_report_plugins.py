'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/report_plugins.py :: HtmlReportManager environment-table
config/cluster file links.
'''

import html
import json
import re
import unittest

from cvs.lib.report_plugins import HtmlReportManager


def _make_html(environment):
    """Wrap an environment dict in a pytest-html-style data-jsonblob table."""
    blob = html.escape(json.dumps({"environment": environment, "tests": {}}), quote=True)
    return f'<table id="results-table" data-jsonblob="{blob}"></table>'


def _read_env(html_content):
    """Extract the environment dict back out of the data-jsonblob."""
    m = re.search(r'data-jsonblob="([^"]*)"', html_content)
    return json.loads(html.unescape(m.group(1)))["environment"]


class UpdateEnvironmentConfigLinksTests(unittest.TestCase):
    def _mgr(self, config_files):
        mgr = HtmlReportManager.__new__(HtmlReportManager)  # bypass __init__ (only _config_files is used)
        mgr._config_files = config_files
        return mgr

    def test_config_file_without_config_substring_still_links(self):
        # Regression: a config named "..._distributed.json" has no "config"
        # substring, but must still get a link (role is tracked, not guessed).
        cluster = "/x/p3_2n_danell_cluster.json"
        config = "/x/mi325x_jaxmaxtext_llama-3.1-8b_distributed.json"
        mgr = self._mgr(
            {
                cluster: ("Cluster File", "logs/cluster_p3_2n_danell_cluster.json"),
                config: ("Config File", "logs/config_mi325x_jaxmaxtext_llama-3.1-8b_distributed.json"),
            }
        )
        html_in = _make_html(
            {
                "Cluster File": "p3_2n_danell_cluster.json",
                "Config File": "mi325x_jaxmaxtext_llama-3.1-8b_distributed.json",
            }
        )

        env = _read_env(mgr._update_environment_config_links(html_in))

        self.assertIn('<a href="logs/config_mi325x_jaxmaxtext_llama-3.1-8b_distributed.json"', env["Config File"])
        self.assertIn("mi325x_jaxmaxtext_llama-3.1-8b_distributed.json</a>", env["Config File"])
        self.assertIn('<a href="logs/cluster_p3_2n_danell_cluster.json"', env["Cluster File"])

    def test_missing_env_key_is_skipped(self):
        # If the environment table has no such row, don't crash or add one.
        mgr = self._mgr({"/x/c.json": ("Config File", "logs/config_c.json")})
        html_in = _make_html({"Cluster File": "c.json"})
        env = _read_env(mgr._update_environment_config_links(html_in))
        self.assertNotIn("Config File", env)

    def test_no_config_files_returns_unchanged(self):
        mgr = self._mgr({})
        html_in = _make_html({"Config File": "x.json"})
        self.assertEqual(mgr._update_environment_config_links(html_in), html_in)


if __name__ == "__main__":
    unittest.main()
