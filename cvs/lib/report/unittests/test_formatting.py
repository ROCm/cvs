'''Unit tests for report HTML formatting helpers.'''

import unittest

from cvs.lib.report.formatting import link_or_text_html


class TestLinkOrTextHtml(unittest.TestCase):
    def test_http_url(self):
        out = link_or_text_html("https://example.com/run", "Upstream")
        self.assertIn('href="https://example.com/run"', out)
        self.assertIn('target="_blank"', out)
        self.assertIn(">Upstream</a>", out)

    def test_local_path_uses_basename(self):
        out = link_or_text_html("/home/user/cvs_results/run.html", "Pytest report")
        self.assertIn('href="run.html"', out)
        self.assertIn(">Pytest report</a>", out)
        self.assertNotIn("target=", out)

    def test_preserves_bundle_relative_path(self):
        out = link_or_text_html("../atom_single.html", "Pytest report")
        self.assertIn('href="../atom_single.html"', out)

    def test_empty(self):
        self.assertEqual(link_or_text_html("", "Pytest report"), "\u2014")


if __name__ == "__main__":
    unittest.main()
