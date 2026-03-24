'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import datetime
import shutil
import zipfile
from pathlib import Path
import uuid

import pytest_html
from cvs.lib import globals

log = globals.log

REPORT_STYLE_OVERRIDES = """<style>
    .collapse { display: none !important; }
    .col-result:hover::after { content: none !important; }
    .col-result.collapsed:hover::after { content: none !important; }
    .collapsible td:not(.col-links) { cursor: default !important; }
    .extras-row { display: none !important; }
</style>"""


class HtmlReportManager:
    """Manages pytest-html report externalization, styling, and zip bundling."""

    def __init__(self, config):
        self._config = config
        self._htmlpath = getattr(config.option, "htmlpath", None)
        self._test_html_dir = getattr(config, "_test_html_dir", "test_html")

    @property
    def is_enabled(self):
        return self._htmlpath is not None

    @property
    def htmlpath(self):
        return Path(self._htmlpath) if self._htmlpath else None

    @property
    def log_dir(self):
        if not self.htmlpath:
            return None
        return self.htmlpath.parent / self._test_html_dir

    def setup_log_dir(self):
        """Remove stale log directory from a previous run and create a fresh one."""
        if not self.is_enabled:
            log.info("Skipping log directory setup because HTML reporting is disabled.")
            return

        # Keep all per-test html logs next to the main pytest-html report.
        log_dir = Path(self._htmlpath).resolve().parent / self._test_html_dir
        log.info("Preparing report log directory: %s", log_dir)
        if log_dir.is_dir():
            try:
                # Start each run with a clean directory to avoid stale links/files.
                shutil.rmtree(log_dir)
                log.info("Removed stale log directory: %s", log_dir)
            except Exception as e:
                log.info(f"Failed to remove stale log directory: {log_dir} - {e}")
        log_dir.mkdir(parents=True, exist_ok=True)

    def write_test_log(self, report, test_name=None):
        """Write an external HTML log file for a single test and return the extras list."""
        extras = getattr(report, "extras", [])

        # Only externalize per-test logs for the actual test call phase.
        if not self.is_enabled or report.when != "call":
            return extras

        # Use the bare test function name (without parametrize params) to keep filenames short.
        if not test_name:
            test_name = report.nodeid.split("::")[-1].split("[")[0]
        temp_id = str(uuid.uuid4()).split("-")[-1]
        safe_name = f"{test_name}_{temp_id}"

        log_dir = self.htmlpath.parent / self._test_html_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{safe_name}.html"

        log_content = []
        for section_name, section_content in report.sections:
            log_content.append(f"<h3>{section_name}</h3><pre>{section_content}</pre>")

        if log_content:
            # Persist a standalone html log page per test.
            log_path.write_text(
                f"<html><body><h1>{report.nodeid}</h1>{''.join(log_content)}</body></html>",
                encoding="utf-8",
            )
            log.info("Wrote external test log: %s", log_path)

            # Link must be relative to the main report location, not the log directory itself.
            rel_path = log_path.relative_to(self.htmlpath.parent)
            rel_path_str = str(rel_path)

            # Avoid duplicate links if another hook/plugin invocation already added the same URL extra.
            already_added = any(
                isinstance(extra, dict)
                and extra.get("format_type") == "url"
                and extra.get("name") == "Full Log"
                and extra.get("content") == rel_path_str
                for extra in extras
            )
            if already_added:
                log.info("Skipped duplicate Full Log link for test '%s': %s", report.nodeid, rel_path_str)
            else:
                extras.append(pytest_html.extras.url(rel_path_str, name="Full Log"))
                log.info("Added Full Log link for test '%s': %s", report.nodeid, rel_path_str)
        else:
            log.info("No captured sections found for test '%s'; no external log file created.", report.nodeid)

        return extras

    @staticmethod
    def replace_table_html(report, data):
        """Replace inline log content with a message pointing to the external log file."""
        del data[:]
        if report.failed:
            data.append("<div class='empty log'>See 'Full Log' link for details.</div>")
        else:
            data.append("<div class='empty log'>Log externalized (see link above).</div>")

    @staticmethod
    def inject_style_overrides(prefix):
        """Inject CSS to hide show/hide details UI elements."""
        prefix.extend([REPORT_STYLE_OVERRIDES])

    def create_zip_bundle(self, session):
        """Bundle the HTML report and per-test log files into a timestamped zip archive."""
        if not self.is_enabled:
            log.info("Skipping zip bundle creation because HTML reporting is disabled.")
            return

        # Resolve path after session completes; report file is expected to exist by now.
        htmlpath = Path(self._htmlpath).resolve()
        if not htmlpath.is_file():
            log.info("Skipping zip bundle creation because HTML report was not found: %s", htmlpath)
            return

        report_dir = htmlpath.parent
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

        suite_name_part = getattr(session.config, "_suite_name", htmlpath.stem)

        zip_path = report_dir / f"{suite_name_part}_{timestamp}.zip"
        log_dir = report_dir / self._test_html_dir

        log.info("Creating report archive: %s", zip_path)
        files_added = 1  # Main report HTML
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Main summary report at zip root.
            zf.write(htmlpath, htmlpath.name)

            if log_dir.is_dir():
                for filepath in sorted(log_dir.iterdir()):
                    if filepath.is_file():
                        # Preserve log folder in archive so report links continue to work after extraction.
                        zf.write(filepath, Path(self._test_html_dir) / filepath.name)
                        files_added += 1
            else:
                log.info("Log directory not found while zipping (continuing): %s", log_dir)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        log.info("Report archive created: %s (%.1f MB, files=%d)", zip_path, size_mb, files_added)
