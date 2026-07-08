'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import datetime
import html
import re
import shutil
import sys
import zipfile
from pathlib import Path
import uuid

import pytest_html
from cvs.lib import globals

log = globals.log


def cli_option_value(name, argv=None):
    """Return the value passed for CLI option `name`, or None if absent.

    Supports both `--name value` (space-separated) and `--name=value` (equals)
    forms. Needed because callers invoke the suite either way (e.g. via
    `cvs run ... --config_file=...`), and a plain `arg == name` scan silently
    misses the equals form -- which is why the HTML Environment table showed
    "Not specified" and the config files were never bundled/linked.
    """
    argv = sys.argv if argv is None else argv
    prefix = name + "="
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


REPORT_STYLE_OVERRIDES = """<style>
    .collapse { display: none !important; }
    .col-result:hover::after { content: none !important; }
    .col-result.collapsed:hover::after { content: none !important; }
    .collapsible td:not(.col-links) { cursor: default !important; }
    .extras-row { display: none !important; }
    /* Separate the per-node links in the Links column with a "|" divider. */
    .col-links a + a::before { content: " | "; color: #888; }
</style>"""


class HtmlReportManager:
    """Manages pytest-html report externalization, styling, and zip bundling."""

    def __init__(self, config):
        self._config = config
        self._htmlpath = getattr(config.option, "htmlpath", None)
        self._test_html_dir = getattr(config, "_test_html_dir", "test_html")
        self._custom_test_reports = []  # Track reports added via add_html_to_report
        self._config_files = {}  # {original_path: {"rel": relative_path, "kind": "cluster"|"config"}}
        # Per-test PASS/FAIL verdict lines rendered in the Reports section.
        # Each entry: {"name": test_name, "passed": bool, "detail": str}.
        self._anc_verdicts = []

        # Store reference for access from pytest hooks
        HtmlReportManager._current_instance = self

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
                log.warning(f"Failed to remove stale log directory: {log_dir} - {e}")
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
            log_content.append(
                f"<h3>{html.escape(section_name)}</h3>"
                f"<pre>{html.escape(section_content)}</pre>"
            )

        if log_content:
            # Persist a standalone html log page per test.
            log_path.write_text(
                f"<html><body><h1>{html.escape(report.nodeid)}</h1>"
                f"{''.join(log_content)}</body></html>",
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

    def add_html_to_report(
        self,
        html_file,
        link_name=None,
        request=None,
        dest_name=None,
        track_in_reports=True,
        test_name=None,
        passed=None,
    ):
        """Copy an external file into the report directory for inclusion in the ZIP bundle.

        Args:
            html_file (str or Path): Path to the file to include in the report
            link_name (str, optional): If provided, adds a clickable link in the pytest-html report
            request (pytest request fixture, optional): Required if link_name is provided
            dest_name (str, optional): Filename to copy to inside the report dir. Defaults to the
                source filename. Use a unique name (e.g. "<node>_<test>_<ts>_errors.json") to avoid
                one node's artifact overwriting another's when several land in the same directory.
            track_in_reports (bool): When True (default) the file is listed in the Reports section
                bullet list. Set False for per-row artifacts (errors.json, per-node ANC tarballs)
                that are already linked from the results table and would clutter Reports.
            test_name (str, optional): Owning test function name; lets the Reports section group
                this artifact under its test's verdict line.
            passed (bool, optional): Whether the owning test passed; the Reports section lists
                artifacts only for FAILED tests, so passing-test artifacts are skipped there.

        Returns:
            str: Relative path to the copied file (for linking), or None if copying failed
        """
        if not self.is_enabled:
            log.info("Skipping HTML file copy because HTML reporting is disabled.")
            return None

        try:
            source_path = Path(html_file)
            if not source_path.exists():
                log.warning("File not found, cannot add to report: %s", source_path)
                return None

            # Ensure log directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Copy file to log directory (optionally under a caller-supplied unique name)
            dest_path = self.log_dir / (dest_name or source_path.name)
            shutil.copy2(source_path, dest_path)

            log.info("Added file to report bundle: %s -> %s", source_path, dest_path)

            # Return relative path for potential linking
            rel_path = dest_path.relative_to(self.htmlpath.parent)
            rel_path_str = str(rel_path)

            # Track the added report (unless it's a per-row artifact linked elsewhere)
            if track_in_reports:
                report_info = {
                    'name': link_name or source_path.name,
                    'path': rel_path_str,
                    'original_path': str(source_path),
                    'test_name': test_name,
                    'passed': passed,
                }
                self._custom_test_reports.append(report_info)

            # Add clickable link to pytest-html report if requested
            if link_name and request:
                try:
                    extra = pytest_html.extras.url(rel_path_str, name=link_name)
                    request.node.user_properties.append(("pytest_html_extra", extra))
                    log.info("Added pytest-html link: %s -> %s", link_name, rel_path_str)
                except Exception as e:
                    log.error("Failed to add pytest-html link: %s - %s", link_name, e)

            return rel_path_str

        except Exception as e:
            log.error("Failed to add HTML file to report: %s - %s", html_file, e)
            return None

    def copy_config_files_to_bundle(self, cluster_file_path, config_file_path):
        """Copy cluster and config files to the report bundle directory.

        Args:
            cluster_file_path (str): Path to cluster file
            config_file_path (str): Path to config file

        Returns:
            dict: Mapping of file types to relative paths in bundle
        """
        if not self.is_enabled:
            return {}

        copied_files = {}

        try:
            # Ensure log directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Copy cluster file. The "cluster_" prefix namespaces it in the
            # bundle so it can't collide with a same-named config/artifact. The
            # Environment-table link is keyed by `kind` (not a filename
            # substring), so the prefix is purely for bundle hygiene.
            if cluster_file_path and cluster_file_path != "Not specified":
                cluster_path = Path(cluster_file_path)
                if cluster_path.exists():
                    dest_cluster = self.log_dir / f"cluster_{cluster_path.name}"
                    shutil.copy2(cluster_path, dest_cluster)
                    rel_cluster = dest_cluster.relative_to(self.htmlpath.parent)
                    copied_files['cluster'] = str(rel_cluster)
                    self._config_files[cluster_file_path] = {"rel": str(rel_cluster), "kind": "cluster"}
                    log.info("Copied cluster file to bundle: %s -> %s", cluster_path, dest_cluster)
                else:
                    log.warning("Cluster file not found: %s", cluster_path)

            # Copy config file with a "config_" prefix -- see the cluster-file
            # note above.
            if config_file_path and config_file_path != "Not specified":
                config_path = Path(config_file_path)
                if config_path.exists():
                    dest_config = self.log_dir / f"config_{config_path.name}"
                    shutil.copy2(config_path, dest_config)
                    rel_config = dest_config.relative_to(self.htmlpath.parent)
                    copied_files['config'] = str(rel_config)
                    self._config_files[config_file_path] = {"rel": str(rel_config), "kind": "config"}
                    log.info("Copied config file to bundle: %s -> %s", config_path, dest_config)
                else:
                    log.warning("Config file not found: %s", config_path)

        except Exception as e:
            log.error("Failed to copy config files to bundle: %s", e)

        return copied_files

    def copy_config_files_from_args(self):
        """Copy config files from command line arguments to bundle."""

        # Parse command line arguments to get file paths.
        # Handles both `--opt value` and `--opt=value` forms.
        cluster_arg = cli_option_value("--cluster_file")
        config_arg = cli_option_value("--config_file")
        cluster_file_path = str(Path(cluster_arg).resolve()) if cluster_arg else None
        config_file_path = str(Path(config_arg).resolve()) if config_arg else None

        # Copy files to bundle if we found them
        if cluster_file_path or config_file_path:
            return self.copy_config_files_to_bundle(cluster_file_path, config_file_path)

        return {}

    def inject_reports_section_into_html(self, htmlpath):
        """Inject Reports section and update Environment table with config file links."""
        try:
            # Read the HTML file
            with open(htmlpath, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Update Environment table with clickable config file links
            html_content = self._update_environment_config_links(html_content)

            # Generate Reports section (test reports only)
            reports_html = self.generate_reports_section()

            # Inject Reports section if we have test reports
            if reports_html:
                # Find the Environment table and inject Reports section after it
                env_pos = html_content.find('<table id="environment">')
                if env_pos != -1:
                    # Find the closing </table> after the environment table
                    table_end_pos = html_content.find('</table>', env_pos)
                    if table_end_pos != -1:
                        # Insert Reports section after the environment table
                        insertion_pos = table_end_pos + len('</table>')

                        # Add some spacing and the Reports section
                        reports_section = f'\n    {reports_html}\n'

                        html_content = html_content[:insertion_pos] + reports_section + html_content[insertion_pos:]

                        log.info("Injected Reports section between Environment and Summary")
                    else:
                        log.warning("Could not find Environment table end in HTML report")
                else:
                    log.warning("Could not find Environment table in HTML report")

            # Write back the modified HTML
            with open(htmlpath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            log.info("Updated Environment table with config file links")

        except Exception as e:
            log.error("Failed to inject Reports section: %s", e)

    def _update_environment_config_links(self, html_content):
        """Update the JSON data to make config files clickable in Environment table."""
        import json

        if not self._config_files:
            return html_content

        # Find the JSON data container
        json_pattern = r'data-jsonblob="([^"]*)"'
        match = re.search(json_pattern, html_content)

        if not match:
            log.warning("Could not find JSON data in HTML report")
            return html_content

        # Decode the HTML-encoded JSON
        import html

        json_str = html.unescape(match.group(1))

        try:
            data = json.loads(json_str)

            # Update environment data with clickable links for config files.
            # Keyed by the recorded `kind` (cluster/config), not a filename
            # substring -- so a config file whose name no longer contains
            # "config" (e.g. mi325x_..._distributed.json) still gets linked.
            row_by_kind = {"cluster": "Cluster File", "config": "Config File"}
            for original_path, info in self._config_files.items():
                filename = Path(original_path).name
                row = row_by_kind.get(info["kind"])
                if row:
                    data["environment"][row] = f'<a href="{info["rel"]}" target="_blank">{filename}</a>'

            # Re-encode the JSON and update the HTML
            updated_json = json.dumps(data)
            encoded_json = html.escape(updated_json, quote=True)

            # Function-form replacement avoids re.sub's backslash-escape
            # processing on string replacements (which would convert JSON's
            # 2-char `\n` / `\t` escapes into raw 0x0A / 0x09 bytes, breaking
            # the data-jsonblob and emptying the results table in the browser).
            html_content = re.sub(json_pattern, lambda _m: f'data-jsonblob="{encoded_json}"', html_content)

        except json.JSONDecodeError as e:
            log.error("Failed to parse JSON data: %s", e)

        return html_content

    @staticmethod
    def replace_table_html(report, data):
        """Replace inline log content with a message pointing to the external log file."""
        del data[:]
        if report.failed:
            data.append("<div class='empty log'>See 'Full Log' link for details.</div>")
        else:
            data.append("<div class='empty log'>Log externalized (see link above).</div>")

    def record_anc_verdict(self, test_name, passed, detail=""):
        """Record a per-test PASS/FAIL verdict for the Reports section.

        Args:
            test_name (str): The test function name (e.g. "test_cpu_all").
            passed (bool): True if the test passed on every node.
            detail (str): On pass, the node list ("PASS on all nodes [...]"); on fail,
                which node(s) failed and a short reason.
        """
        self._anc_verdicts.append({"name": test_name, "passed": bool(passed), "detail": detail})

    @staticmethod
    def _escape(text):
        """Minimal HTML escaping for text injected into the Reports section."""
        import html as _html

        return _html.escape(str(text), quote=True)

    def generate_reports_section(self):
        """Generate HTML for the Reports section.

        Failure-focused: lists ONLY the FAILED tests, and under each failed test's
        red verdict line, nests that test's failed-node ANC log zip(s) so the item
        and its archive sit together. Passing tests (and passing-node zips) are
        omitted here on purpose — the Summary table's Links column already carries
        them. When every ANC test passed, a single green "All tests passed" note is
        shown instead. Non-ANC artifacts (rccl/preflight, which carry no test_name)
        keep their existing flat bullet list.
        """
        if not self._anc_verdicts and not self._custom_test_reports:
            return ""

        html = '<div><h2>Reports</h2><ul>'
        for report in self._custom_test_reports:
            path = html.escape(str(report["path"]))
            name = html.escape(str(report["name"]))
            html += f'<li><a href="{path}" target="_blank">{name}</a></li>'
        html += "</ul></div>"

        return html

    def generate_suite_reports(self, session):
        """Write registered suite report HTML/JSON into the pytest bundle before zip."""
        if not self.is_enabled:
            return

        from cvs.lib.report.inference import publish_inference_suite_report
        from cvs.lib.report.registry import get_session_results, get_suite_report_config
        from cvs.lib.report.types import InferenceReportConfig

        report_config = get_suite_report_config(session.config)
        if report_config is None:
            suite_name = getattr(session.config, "_suite_name", "unknown")
            log.info(
                "Skipping suite report generation: no preset registered for suite '%s'",
                suite_name,
            )
            return

        store = get_session_results()

        if not isinstance(report_config, InferenceReportConfig):
            log.warning("Unknown suite report config type: %s", type(report_config).__name__)
            return

        inf_res_dict = store.get("inf_res_dict")
        if not inf_res_dict:
            log.info("Skipping suite report generation: no results in session store")
            return

        publish_inference_suite_report(
            report_config,
            variant_config=store.get("variant_config"),
            inf_res_dict=inf_res_dict,
            lifecycle_report=store.get("lifecycle_report") or {},
            report_manager=self,
            pytest_config=session.config,
        )

    @staticmethod
    def inject_style_overrides(prefix):
        """Inject CSS to hide show/hide details UI elements."""
        prefix.extend([REPORT_STYLE_OVERRIDES])

    def create_zip_bundle(self, session):
        """Bundle the HTML report and per-test log files into a timestamped zip archive."""
        if not self.is_enabled:
            log.info("Skipping zip bundle creation because HTML reporting is disabled.")
            return

        # Copy config files before creating ZIP (tracking handled internally)
        self.copy_config_files_from_args()

        # Resolve path after session completes; report file is expected to exist by now.
        htmlpath = Path(self._htmlpath).resolve()
        if not htmlpath.is_file():
            log.warning("Skipping zip bundle creation because HTML report was not found: %s", htmlpath)
            return

        # Inject Reports section into main HTML report
        self.inject_reports_section_into_html(htmlpath)

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

            # Include assets directory if pytest-html created it (for CSS, etc.)
            # This happens when --self-contained-html=false (the default)
            assets_dir = report_dir / "assets"
            if assets_dir.is_dir():
                # Check if self-contained mode is disabled (default behavior)
                self_contained = getattr(session.config.option, 'self_contained_html', False)
                if not self_contained:
                    log.info("Including assets directory in ZIP bundle (external CSS mode)")
                    for filepath in sorted(assets_dir.iterdir()):
                        if filepath.is_file():
                            zf.write(filepath, Path("assets") / filepath.name)
                            files_added += 1
                            log.info("Added asset to ZIP: %s", filepath.name)
                else:
                    log.info("Skipping assets directory (self-contained HTML mode enabled)")
            else:
                log.info("No assets directory found (likely self-contained HTML mode)")

            if log_dir.is_dir():
                for filepath in sorted(log_dir.iterdir()):
                    if filepath.is_file():
                        # Preserve log folder in archive so report links continue to work after extraction.
                        zf.write(filepath, Path(self._test_html_dir) / filepath.name)
                        files_added += 1
            else:
                log.warning("Log directory not found while zipping (continuing): %s", log_dir)

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        log.info("Report archive created: %s (%.1f MB, files=%d)", zip_path, size_mb, files_added)
