'''Tests for the Run Deck page shell: CVS header, theme, and deck content.'''

import unittest

from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.rundeck.runtime.theme import OVERVIEW_LINK_MARKER, THEME_BOOT_SCRIPT
from cvs.lib.report.testing.fixtures import generic_sweep_profile, generic_variant, two_cell_inf_res


class TestDeckPage(unittest.TestCase):
    def setUp(self):
        store = {"inf_res_dict": two_cell_inf_res(), "variant_config": generic_variant(), "lifecycle_report": {}}
        payload = build_rundeck_payload(profile=generic_sweep_profile(), store=store, cvs_version="1")
        self.page = render_rundeck_html(payload)

    def test_page_starts_with_the_cvs_header_and_theme_toggle(self):
        body = self.page.split("<body>", 1)[1]
        self.assertTrue(body.startswith('<div class="wrap" id="top">\n<div class="cvs-header"'))
        self.assertIn(OVERVIEW_LINK_MARKER, body)
        self.assertIn('class="cvs-theme-toggle"', body)

    def test_theme_is_set_before_the_styles_apply(self):
        head = self.page.split("</head>", 1)[0]
        self.assertLess(head.index(THEME_BOOT_SCRIPT), head.index("<style>"))
        self.assertIn(":root[data-theme='light']", head)

    def test_deck_content_is_unchanged(self):
        self.assertIn('<div class="hero-head">', self.page)
        self.assertIn('class="status-badge status-', self.page)
        self.assertIn("report-nav", self.page)


if __name__ == "__main__":
    unittest.main()
