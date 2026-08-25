"""Unit tests for cell card rendering functionality."""

import unittest

from cvs.lib.report.render.cell_card import (
    CellCardConfig,
    CellCardRenderer,
    _THEME_TOKENS,
)


class TestCellCardConfig(unittest.TestCase):
    """Test CellCardConfig dataclass functionality."""

    def test_default_values(self):
        """Test that CellCardConfig has sensible defaults."""
        config = CellCardConfig()

        self.assertEqual(config.tier_order, ())
        self.assertEqual(config.headline_metric, "")
        self.assertEqual(config.enforce, False)
        self.assertEqual(config.cell_lifecycle_labels, ())
        self.assertEqual(config.compact, False)
        self.assertIsNone(config.highlight_metric)
        self.assertIsNone(config.pytest_html_basename)
        self.assertEqual(config.theme, "pytest")

    def test_custom_configuration(self):
        """Test creating config with custom values."""
        config = CellCardConfig(
            tier_order=("T1", "T2", "T3"),
            headline_metric="throughput",
            enforce=True,
            cell_lifecycle_labels=("setup", "execution", "teardown"),
            compact=True,
            highlight_metric="latency",
            pytest_html_basename="report.html",
            theme="report",
        )

        self.assertEqual(config.tier_order, ("T1", "T2", "T3"))
        self.assertEqual(config.headline_metric, "throughput")
        self.assertTrue(config.enforce)
        self.assertEqual(config.cell_lifecycle_labels, ("setup", "execution", "teardown"))
        self.assertTrue(config.compact)
        self.assertEqual(config.highlight_metric, "latency")
        self.assertEqual(config.pytest_html_basename, "report.html")
        self.assertEqual(config.theme, "report")

    def test_immutable_config(self):
        """Test that CellCardConfig is frozen/immutable."""
        config = CellCardConfig(tier_order=("T1",))

        with self.assertRaises(AttributeError):
            config.tier_order = ("T2",)

    def test_theme_validation(self):
        """Test that only valid themes are accepted."""
        # Valid themes should work
        CellCardConfig(theme="pytest")
        CellCardConfig(theme="report")

        # Note: Type hints would catch invalid themes at development time
        # but runtime validation would need additional implementation


class TestCellCardRenderer(unittest.TestCase):
    """Test CellCardRenderer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.basic_config = CellCardConfig(
            tier_order=("T1", "T2"),
            headline_metric="throughput",
            enforce=False,
            cell_lifecycle_labels=("setup", "run"),
        )

        self.sample_cell = {
            "policy": "test_policy",
            "isl": 1024,
            "osl": 2048,
            "concurrency": 4,
            "cell_id": "test_cell_123",
            "host": "test_host",
            "show_host_in_label": True,
            "metrics": [
                {
                    "metric": "throughput",
                    "actual": 1234.5,
                    "label": "Throughput",
                    "unit": "tok/s",
                    "status": "pass",
                    "bar_pct": 80,
                    "spec": {"value": 1000},
                    "margin": "+23%",
                },
                {
                    "metric": "latency",
                    "actual": 50.2,
                    "label": "Latency",
                    "unit": "ms",
                    "status": "fail",
                    "bar_pct": 120,
                    "spec": {"value": 40},
                    "margin": "-25%",
                },
            ],
            "tiers": {"T1": "pass", "T2": "fail"},
            "cell_lifecycle": {"setup": 5.0, "run": 15.0},
        }

    def test_renderer_creation(self):
        """Test creating a CellCardRenderer."""
        renderer = CellCardRenderer(self.basic_config)

        self.assertEqual(renderer.config, self.basic_config)
        self.assertEqual(renderer._theme_tokens, _THEME_TOKENS["pytest"])
        self.assertIsNone(renderer._cell)

    def test_render_complete_cell(self):
        """Test rendering a complete cell to HTML."""
        renderer = CellCardRenderer(self.basic_config)
        html = renderer.render(self.sample_cell)

        # Check basic HTML structure
        self.assertIn("<article class='cell-card'>", html)
        self.assertIn("</article>", html)

        # Check header content
        self.assertIn("test_policy", html)
        self.assertIn("ISL=1024", html)
        self.assertIn("OSL=2048", html)
        self.assertIn("C=4", html)

        # Check headline metric (numbers are formatted with commas)
        self.assertIn("1,234.5", html)  # throughput value
        self.assertIn("tok/s", html)  # unit

        # Check tiers
        self.assertIn("chip-pass", html)
        self.assertIn("chip-fail", html)

        # Check footer
        self.assertIn("test_cell_123", html)
        self.assertIn("test_host", html)

    def test_render_compact_mode(self):
        """Test rendering in compact mode."""
        compact_config = CellCardConfig(
            tier_order=("T1",), headline_metric="throughput", enforce=False, cell_lifecycle_labels=(), compact=True
        )
        renderer = CellCardRenderer(compact_config)
        html = renderer.render(self.sample_cell)

        self.assertIn("cell-card-compact", html)
        # Timeline should not appear in compact mode
        self.assertNotIn("cell-mini-tl", html)

    def test_render_with_highlight_metric(self):
        """Test rendering with highlighted metric."""
        highlight_config = CellCardConfig(
            tier_order=("T1",),
            headline_metric="throughput",
            enforce=False,
            cell_lifecycle_labels=(),
            highlight_metric="latency",
        )
        renderer = CellCardRenderer(highlight_config)
        html = renderer.render(self.sample_cell)

        self.assertIn("metric-row-highlight", html)

    def test_render_with_pytest_link(self):
        """Test rendering with pytest HTML link."""
        pytest_config = CellCardConfig(
            tier_order=("T1",),
            headline_metric="throughput",
            enforce=False,
            cell_lifecycle_labels=(),
            pytest_html_basename="report.html",
        )

        cell_with_nodeid = self.sample_cell.copy()
        cell_with_nodeid["pytest_metrics_nodeid"] = "test_module::test_function"

        renderer = CellCardRenderer(pytest_config)
        html = renderer.render(cell_with_nodeid)

        self.assertIn("report.html", html)

    def test_render_enforced_thresholds(self):
        """Test rendering with enforced thresholds (gate vs floor)."""
        enforced_config = CellCardConfig(
            tier_order=("T1",),
            headline_metric="throughput",
            enforce=True,  # This should show "gate" instead of "floor"
            cell_lifecycle_labels=(),
        )
        renderer = CellCardRenderer(enforced_config)
        html = renderer.render(self.sample_cell)

        self.assertIn("gate", html)

        # Test non-enforced (should show "floor")
        non_enforced_config = CellCardConfig(
            tier_order=("T1",), headline_metric="throughput", enforce=False, cell_lifecycle_labels=()
        )
        renderer = CellCardRenderer(non_enforced_config)
        html = renderer.render(self.sample_cell)

        self.assertIn("floor", html)

    def test_render_missing_headline_metric(self):
        """Test rendering when headline metric is not found."""
        config = CellCardConfig(
            tier_order=("T1",), headline_metric="nonexistent_metric", enforce=False, cell_lifecycle_labels=()
        )
        renderer = CellCardRenderer(config)
        html = renderer.render(self.sample_cell)

        # Should show em dash for missing metric
        self.assertIn("\u2014", html)

    def test_render_empty_cell_lifecycle(self):
        """Test rendering cell with empty lifecycle data."""
        cell_no_lifecycle = self.sample_cell.copy()
        cell_no_lifecycle["cell_lifecycle"] = {}

        renderer = CellCardRenderer(self.basic_config)
        html = renderer.render(cell_no_lifecycle)

        # Should not contain timeline elements
        self.assertNotIn("cell-mini-tl", html)

    def test_render_metrics_with_null_actual(self):
        """Test rendering metrics that have null actual values."""
        cell_with_null = self.sample_cell.copy()
        cell_with_null["metrics"] = [
            {
                "metric": "throughput",
                "actual": None,  # Null value should be skipped
                "label": "Throughput",
                "unit": "tok/s",
            }
        ]

        renderer = CellCardRenderer(self.basic_config)
        html = renderer.render(cell_with_null)

        # Should not contain the metric with null value
        self.assertNotIn("Throughput", html)

    def test_cell_state_cleanup(self):
        """Test that cell state is properly cleaned up after rendering."""
        renderer = CellCardRenderer(self.basic_config)

        # Initially no cell
        self.assertIsNone(renderer._cell)

        # Render should set and clean up cell
        renderer.render(self.sample_cell)

        # Should be cleaned up after render
        self.assertIsNone(renderer._cell)

    def test_css_generation(self):
        """Test CSS generation."""
        renderer = CellCardRenderer(self.basic_config)
        css = renderer.get_css()

        self.assertIsInstance(css, str)
        self.assertGreater(len(css), 100)  # Should be substantial CSS

        # Check for key CSS classes
        self.assertIn(".cell-card", css)
        self.assertIn(".headline", css)
        self.assertIn(".chip", css)

    def test_css_generation_compact_mode(self):
        """Test CSS generation in compact mode."""
        compact_config = CellCardConfig(compact=True)
        renderer = CellCardRenderer(compact_config)
        css = renderer.get_css()

        self.assertIn("cell-card-compact", css)

    def test_css_generation_report_theme(self):
        """Test CSS generation with report theme."""
        report_config = CellCardConfig(theme="report")
        renderer = CellCardRenderer(report_config)
        css = renderer.get_css()

        # Report theme should use CSS variables
        self.assertIn("var(--", css)
        self.assertIn(".cells {", css)  # Grid rule for report theme

    def test_theme_tokens_selection(self):
        """Test that correct theme tokens are selected."""
        pytest_renderer = CellCardRenderer(CellCardConfig(theme="pytest"))
        report_renderer = CellCardRenderer(CellCardConfig(theme="report"))

        self.assertEqual(pytest_renderer._theme_tokens, _THEME_TOKENS["pytest"])
        self.assertEqual(report_renderer._theme_tokens, _THEME_TOKENS["report"])


class TestThemeTokens(unittest.TestCase):
    """Test theme token definitions."""

    def test_theme_tokens_exist(self):
        """Test that required theme tokens exist."""
        required_themes = ["pytest", "report"]

        for theme in required_themes:
            self.assertIn(theme, _THEME_TOKENS)

        required_keys = ["card_bg", "border", "text", "accent", "muted", "pass", "fail", "record", "na", "card_font"]

        for theme_name, tokens in _THEME_TOKENS.items():
            for key in required_keys:
                self.assertIn(key, tokens, f"Missing {key} in {theme_name} theme")

    def test_pytest_theme_has_colors(self):
        """Test that pytest theme has actual color values."""
        pytest_tokens = _THEME_TOKENS["pytest"]

        # Should have hex colors
        self.assertTrue(pytest_tokens["card_bg"].startswith("#"))
        self.assertTrue(pytest_tokens["border"].startswith("#"))
        self.assertTrue(pytest_tokens["accent"].startswith("#"))

    def test_report_theme_has_css_variables(self):
        """Test that report theme uses CSS variables."""
        report_tokens = _THEME_TOKENS["report"]

        # Should use CSS custom properties
        self.assertTrue(report_tokens["card_bg"].startswith("var("))
        self.assertTrue(report_tokens["border"].startswith("var("))
        self.assertTrue(report_tokens["accent"].startswith("var("))


if __name__ == "__main__":
    unittest.main()
