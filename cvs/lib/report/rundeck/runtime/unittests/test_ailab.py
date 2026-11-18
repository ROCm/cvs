'''Tests for the AI Lab look shared by Run Decks, viewers, and the consolidated report.'''

import unittest

from cvs.lib.report.rundeck.runtime import ailab


class TestTokens(unittest.TestCase):
    def test_dark_tokens_feed_the_page_variables(self):
        css = ailab.tokens_css(light=False)
        self.assertIn("--f-canvas: #0A0A0B", css)
        self.assertIn("--panel: var(--f-surface-1)", css)
        self.assertIn("--pass: var(--f-ok)", css)
        self.assertNotIn("data-theme='light'", css)

    def test_light_theme_is_chosen_by_data_theme(self):
        css = ailab.tokens_css()
        self.assertIn(":root[data-theme='light']", css)
        self.assertIn("--f-canvas: #FAFAFA", css)


class TestThemeCss(unittest.TestCase):
    def test_deck_theme_has_light_tokens_status_marks_and_print_rules(self):
        css = ailab.theme_css("deck")
        self.assertIn(":root[data-theme='light']", css)
        self.assertIn(".status-badge.status-fail::before", css)
        self.assertIn(".ail-header", css)
        self.assertIn("@media print", css)

    def test_viewer_theme_stays_dark_and_leaves_deck_components_alone(self):
        css = ailab.theme_css("viewer")
        self.assertNotIn("data-theme='light'", css)
        self.assertNotIn(".ail-theme", css)
        self.assertNotIn(".summary-stat", css)
        self.assertIn(".status-badge.status-fail::before", css)
        self.assertIn(".ail-header", css)
        self.assertEqual(ailab.viewer_theme_head(), "<style>%s</style>" % css)


class TestHeader(unittest.TestCase):
    def test_header_has_brand_context_and_the_overview_link_slot(self):
        header = ailab.header_html("Run Deck <viewer>")
        self.assertIn("<strong>AMD</strong> Cluster Validation Suite", header)
        self.assertIn('<span class="ail-context">Run Deck &lt;viewer&gt;</span>', header)
        self.assertIn(ailab.OVERVIEW_LINK_MARKER, header)
        self.assertNotIn("ail-theme", header)

    def test_toggle_flips_and_stores_the_theme(self):
        header = ailab.header_html("Run Deck", toggle=True)
        self.assertIn('class="ail-theme"', header)
        handler = header.split('onclick="', 1)[1].split('">', 1)[0]
        self.assertIn("localStorage.setItem('cvs-theme',t)", handler)
        self.assertNotIn('"', handler)

    def test_boot_script_reads_the_shared_theme_key(self):
        self.assertTrue(ailab.THEME_BOOT_SCRIPT.startswith("<script>"))
        self.assertIn("localStorage.getItem('cvs-theme')", ailab.THEME_BOOT_SCRIPT)


if __name__ == "__main__":
    unittest.main()
