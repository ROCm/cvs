import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.report_plugins import HtmlReportManager


class TestHtmlReportManager(unittest.TestCase):
    def test_registers_file_already_in_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SimpleNamespace(
                option=SimpleNamespace(htmlpath=str(root / "report.html")),
                _test_html_dir="report_html",
            )
            manager = HtmlReportManager(config)
            manager.log_dir.mkdir()
            artifact = manager.log_dir / "run_deck.html"
            artifact.write_text("deck")

            relative = manager.add_html_to_report(artifact, link_name="Run Deck")

            self.assertEqual(relative, "report_html/run_deck.html")
            self.assertEqual(manager._custom_test_reports[0]["name"], "Run Deck")
            self.assertEqual(artifact.read_text(), "deck")


if __name__ == "__main__":
    unittest.main()
