"""
Report Generation Module

This module provides functionality for generating preflight test reports in various formats.
"""

import csv
import html
import json
from pathlib import Path

from cvs.lib.preflight.base import PreflightCheck
from cvs.lib import globals

log = globals.log


class PreflightReportGenerator(PreflightCheck):
    """Generate preflight test reports."""

    def __init__(self, phdl, results, config_dict=None):
        """
        Initialize report generator.

        Args:
            phdl: Parallel SSH handle for cluster nodes
            results: Preflight test results to generate reports from
            config_dict: Optional configuration dictionary
        """
        super().__init__(phdl, config_dict)
        self.results = results

    def run(self):
        """
        Generate all configured reports.

        Returns:
            dict: Report generation results
        """
        report_results = {}

        summary = self._generate_preflight_summary()
        report_results['summary'] = summary
        self.results['summary'] = summary

        if self.config_dict.get('generate_html_report', 'true').lower() == 'true':
            html_path, rdma_csv_path = self._generate_html_report()
            report_results['html_report'] = html_path
            if rdma_csv_path:
                report_results['rdma_pairs_csv'] = rdma_csv_path

        return report_results

    def _generate_preflight_summary(self):
        """Build summary dict from ``self.results`` (preflight_results bundle)."""
        gid_results = self.results.get('gid_consistency', {})
        connectivity_results = self.results.get('rdma_connectivity', {})
        rocm_results = self.results.get('rocm_versions', {})
        interface_results = self.results.get('interface_names', {})
        reachability_results = self.results.get('node_reachability')
        ssh_connectivity_results = self.results.get('ssh_connectivity')
        summary = {
            'overall_status': 'PASS',
            'checks': {
                'ssh_reachability': self._summarize_reachability_results(reachability_results),
                'gid_consistency': self._summarize_gid_results(gid_results),
                'rdma_connectivity': self._summarize_connectivity_results(connectivity_results),
                'rocm_versions': self._summarize_rocm_results(rocm_results),
                'interface_names': self._summarize_interface_results(interface_results),
            },
            'recommendations': [],
        }

        if ssh_connectivity_results:
            summary['checks']['ssh_connectivity'] = self._summarize_ssh_connectivity_results(ssh_connectivity_results)

        # Determine overall status (skipped tests don't affect overall status)
        for check_name, check_summary in summary['checks'].items():
            if check_summary['status'] == 'FAIL':
                summary['overall_status'] = 'FAIL'

        # Generate recommendations
        if summary['checks']['gid_consistency']['status'] == 'FAIL':
            summary['recommendations'].append(
                "Fix GID configuration on RDMA interfaces before running performance tests"
            )

        if summary['checks']['rdma_connectivity']['status'] == 'FAIL':
            summary['recommendations'].append("Address RDMA connectivity issues between node pairs")
        elif summary['checks']['rdma_connectivity']['status'] == 'SKIPPED':
            summary['recommendations'].append("Consider running RDMA connectivity tests for comprehensive validation")

        if summary['checks']['rocm_versions']['status'] == 'FAIL':
            summary['recommendations'].append("Ensure consistent ROCm versions across all cluster nodes")

        if summary['checks']['interface_names']['status'] == 'FAIL':
            summary['recommendations'].append("Standardize RDMA interface naming across cluster nodes")

        if summary['overall_status'] == 'PASS':
            summary['recommendations'].append("All preflight checks passed - cluster is ready for performance testing")

        return summary

    def _generate_html_report(self, output_path=None):
        """Write HTML report for preflight test results.

        Returns:
            tuple: (html_path_str, rdma_pairs_csv_path_str_or_None)
        """
        from datetime import datetime

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config_dict.get('report_output_dir', '/tmp/preflight_reports')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir) / f"preflight_report_{timestamp}.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html_content()

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        log.info(f"HTML report generated: {output_path}")

        rdma_csv_path = None
        if self.config_dict.get('generate_rdma_pairs_csv', 'true').lower() == 'true':
            rdma_csv_path = self._write_rdma_pairs_csv(output_path)
            if rdma_csv_path:
                log.info(f"RDMA pairs CSV generated: {rdma_csv_path}")

        return str(output_path), rdma_csv_path

    def _summarize_gid_results(self, gid_results):
        """Summarize GID consistency check results."""
        total_interfaces = 0
        ok_interfaces = 0
        failed_nodes = []

        for node, result in gid_results.items():
            for interface, interface_result in result['interfaces'].items():
                total_interfaces += 1
                if interface_result.get('status') == 'OK':
                    ok_interfaces += 1

            if result['status'] == 'FAIL':
                failed_nodes.append(node)

        return {
            'status': 'PASS' if not failed_nodes else 'FAIL',
            'total_interfaces': total_interfaces,
            'ok_interfaces': ok_interfaces,
            'failed_nodes': failed_nodes,
            'summary': f"{ok_interfaces}/{total_interfaces} interfaces have valid GID",
        }

    def _summarize_connectivity_results(self, connectivity_results):
        """Summarize RDMA connectivity check results."""
        # Handle skipped tests (either by configuration or due to failures)
        if connectivity_results.get('skipped', False) or connectivity_results.get('status') == 'SKIPPED':
            msg = connectivity_results.get('message', 'RDMA connectivity test skipped')
            excl_if = connectivity_results.get('excluded_nodes_interface_check') or []
            excl_gid = connectivity_results.get('excluded_nodes_gid') or []
            extra = []
            if excl_if:
                extra.append(f"{len(excl_if)} failed interface presence")
            if excl_gid:
                extra.append(f"{len(excl_gid)} failed GID consistency")
            if extra:
                msg += " (" + "; ".join(extra) + ")"
            return {
                'status': 'SKIPPED',
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'mode': connectivity_results.get('mode', 'unknown'),
                'summary': msg,
            }

        # Handle normal test results
        if 'failed_pairs' not in connectivity_results:
            # Fallback for malformed results
            return {
                'status': 'ERROR',
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'mode': connectivity_results.get('mode', 'unknown'),
                'summary': 'RDMA connectivity test failed to complete properly',
            }

        summary = f"{connectivity_results['successful_pairs']}/{connectivity_results['total_pairs']} pairs connected successfully"
        pruned = connectivity_results.get('pruned_nodes_after_intra') or []
        if pruned:
            summary += f"; {len(pruned)} node(s) excluded from inter-group after Round 1 (intra-group)"
        excl_iface = connectivity_results.get('excluded_nodes_interface_check') or []
        if excl_iface:
            summary += f"; {len(excl_iface)} node(s) excluded from RDMA (failed interface presence check before mesh)"
        excl_gid = connectivity_results.get('excluded_nodes_gid') or []
        if excl_gid:
            summary += f"; {len(excl_gid)} node(s) excluded from RDMA (failed GID consistency before mesh)"

        return {
            'status': 'PASS' if connectivity_results['failed_pairs'] == 0 else 'FAIL',
            'total_pairs': connectivity_results['total_pairs'],
            'successful_pairs': connectivity_results['successful_pairs'],
            'failed_pairs': connectivity_results['failed_pairs'],
            'mode': connectivity_results['mode'],
            'summary': summary,
        }

    def _summarize_rocm_results(self, rocm_results):
        """Summarize ROCm version check results."""
        total_nodes = len(rocm_results)
        consistent_nodes = sum(1 for result in rocm_results.values() if result['status'] == 'PASS')
        failed_nodes = [node for node, result in rocm_results.items() if result['status'] == 'FAIL']

        return {
            'status': 'PASS' if consistent_nodes == total_nodes else 'FAIL',
            'total_nodes': total_nodes,
            'consistent_nodes': consistent_nodes,
            'failed_nodes': failed_nodes,
            'summary': f"{consistent_nodes}/{total_nodes} nodes have consistent ROCm version",
        }

    def _summarize_interface_results(self, interface_results):
        """Summarize interface name check results."""
        total_nodes = len(interface_results)
        compliant_nodes = sum(1 for result in interface_results.values() if result['status'] == 'PASS')
        failed_nodes = [node for node, result in interface_results.items() if result['status'] == 'FAIL']

        return {
            'status': 'PASS' if compliant_nodes == total_nodes else 'FAIL',
            'total_nodes': total_nodes,
            'compliant_nodes': compliant_nodes,
            'failed_nodes': failed_nodes,
            'summary': f"{compliant_nodes}/{total_nodes} nodes have compliant interface names",
        }

    def _summarize_reachability_results(self, reachability_results):
        """Summarize SSH reachability check results."""
        if not reachability_results:
            return {
                'status': 'UNKNOWN',
                'total_nodes': 0,
                'reachable_nodes': 0,
                'unreachable_nodes': [],
                'summary': 'SSH reachability test not performed',
            }

        total_nodes = reachability_results.get('total_nodes', 0)
        reachable_nodes = reachability_results.get('reachable_nodes', 0)
        unreachable_nodes = reachability_results.get('unreachable_nodes', [])

        status = 'PASS' if len(unreachable_nodes) == 0 else 'WARNING'

        return {
            'status': status,
            'total_nodes': total_nodes,
            'reachable_nodes': reachable_nodes,
            'unreachable_nodes': unreachable_nodes,
            'summary': f"{reachable_nodes}/{total_nodes} nodes reachable",
        }

    def _summarize_ssh_connectivity_results(self, ssh_connectivity_results):
        """Summarize SSH full mesh connectivity check results."""
        if not ssh_connectivity_results:
            return {
                'status': 'UNKNOWN',
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'summary': 'SSH connectivity test not performed',
            }

        if ssh_connectivity_results.get('skipped', False):
            return {
                'status': 'SKIPPED',
                'total_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'summary': 'SSH connectivity test skipped',
            }

        total_pairs = ssh_connectivity_results.get('total_pairs', 0)
        successful_pairs = ssh_connectivity_results.get('successful_pairs', 0)
        failed_pairs = ssh_connectivity_results.get('failed_pairs', 0)

        # Determine status based on results
        if 'error' in ssh_connectivity_results:
            status = 'ERROR'
            summary = f"SSH connectivity test failed: {ssh_connectivity_results['error']}"
        elif failed_pairs == 0:
            status = 'PASS'
            summary = f"{successful_pairs}/{total_pairs} pairs connected successfully"
        else:
            status = 'FAIL'
            success_rate = (successful_pairs / total_pairs * 100) if total_pairs > 0 else 0
            summary = f"{successful_pairs}/{total_pairs} pairs connected successfully ({success_rate:.1f}%)"

        return {
            'status': status,
            'total_pairs': total_pairs,
            'successful_pairs': successful_pairs,
            'failed_pairs': failed_pairs,
            'summary': summary,
        }

    def _generate_html_content(self):
        """Generate the complete HTML content for the preflight report."""
        from datetime import datetime

        results = self.results
        config_dict = self.config_dict
        summary = results.get('summary', {})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GPU Cluster Preflight Check Report</title>
        <style>
            {self._get_html_styles()}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>GPU Cluster Preflight Check Report</h1>
                <div class="report-meta">
                    <p><strong>Generated:</strong> {timestamp}</p>
                    <p><strong>Overall Status:</strong> <span class="status-{summary.get('overall_status', 'UNKNOWN').lower()}">{summary.get('overall_status', 'UNKNOWN')}</span></p>
                </div>
            </header>

            {self._generate_executive_summary_html(summary)}
            {self._generate_gid_consistency_html(results.get('gid_consistency', {}))}
            {self._generate_connectivity_html(results.get('rdma_connectivity', {}))}
            {self._generate_ssh_connectivity_html(results.get('ssh_connectivity', {}))}
            {self._generate_rocm_versions_html(results.get('rocm_versions', {}))}
            {self._generate_interface_names_html(results.get('interface_names', {}))}
            {self._generate_configuration_html(config_dict)}
            {self._generate_recommendations_html(summary.get('recommendations', []))}
        </div>
    </body>
    </html>
    """
        return html

    def _get_html_styles(self):
        """Return CSS styles for the HTML report."""
        return """
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            header {
                border-bottom: 3px solid #007acc;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            h1 {
                color: #333;
                margin: 0;
            }
            h2 {
                color: #007acc;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 10px;
            }
            h3 {
                color: #555;
            }
            .report-meta {
                margin-top: 10px;
                color: #666;
            }
            .status-pass {
                color: #28a745;
                font-weight: bold;
            }
            .status-fail {
                color: #dc3545;
                font-weight: bold;
            }
            .status-skipped {
                color: #6c757d;
                font-weight: bold;
            }
            .error-summary {
                color: #dc3545;
                font-weight: bold;
                margin-bottom: 15px;
                padding: 10px;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 4px;
            }
            .summary-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .summary-table th {
                background-color: #f8f9fa;
                font-weight: bold;
                padding: 15px 12px;
                text-align: left;
                border-bottom: 2px solid #dee2e6;
            }
            .summary-table td {
                padding: 12px;
                border-bottom: 1px solid #dee2e6;
            }
            .summary-table tr:hover {
                background-color: #f8f9fa;
            }
            .check-name {
                font-weight: 600;
                color: #495057;
            }
            .status-cell {
                text-align: center;
                width: 120px;
            }
            .status-badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
                white-space: nowrap;
            }
            .status-badge.status-pass {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status-badge.status-fail {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .status-badge.status-skipped {
                background-color: #e2e3e5;
                color: #6c757d;
                border: 1px solid #d6d8db;
            }
            .results-cell {
                text-align: center;
                font-weight: 600;
                color: #495057;
                width: 100px;
            }
            .details-cell {
                color: #6c757d;
                font-size: 0.95em;
            }
            .summary-row-pass {
                border-left: 4px solid #28a745;
            }
            .summary-row-fail {
                border-left: 4px solid #dc3545;
            }
            .summary-row-skipped {
                border-left: 4px solid #6c757d;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .gid-cell {
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.9em;
                word-break: break-all;
                max-width: 200px;
            }
            .gid-cell small {
                color: #666;
                font-weight: normal;
            }
            code {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 3px;
                padding: 2px 6px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.85em;
                color: #495057;
                word-break: break-all;
                display: block;
                margin: 2px 0;
            }
            .connectivity-matrix {
                display: grid;
                gap: 2px;
                margin: 20px 0;
            }
            .matrix-cell {
                padding: 8px;
                text-align: center;
                border: 1px solid #ddd;
                font-size: 12px;
            }
            .matrix-cell.header {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .matrix-cell.pass {
                background-color: #d4edda;
                color: #155724;
            }
            .matrix-cell.fail {
                background-color: #f8d7da;
                color: #721c24;
            }
            .matrix-cell.not-tested {
                background-color: #e2e3e5;
                color: #6c757d;
            }
            .error-list {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 4px;
                padding: 15px;
                margin: 10px 0;
            }
            .error-list ul {
                margin: 0;
                padding-left: 20px;
            }
            .error-list li {
                color: #721c24;
            }
            .recommendations {
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 4px;
                padding: 20px;
                margin: 20px 0;
            }
            .recommendations h3 {
                color: #0c5460;
                margin-top: 0;
            }
            .recommendations ul {
                margin: 0;
                padding-left: 20px;
            }
            .config-section {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                margin: 10px 0;
            }
            .config-section pre {
                background-color: #e9ecef;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
            }
        
            /* Collapsible details styling */
            details {
                border: 1px solid #ddd;
                border-radius: 6px;
                margin: 15px 0;
                overflow: hidden;
            }
            details summary {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 12px 15px;
                cursor: pointer;
                font-weight: 600;
                border-bottom: 1px solid #ddd;
                outline: none;
                transition: background-color 0.2s ease;
            }
            details summary:hover {
                background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            }
            details[open] summary {
                background: linear-gradient(135deg, #007acc 0%, #0056b3 100%);
                color: white;
                border-bottom-color: #0056b3;
            }
            details summary::marker {
                font-size: 1.2em;
            }
            details .content {
                padding: 15px;
                background-color: #fafbfc;
            }
            details table {
                margin-top: 0;
            }
        """

    def _generate_executive_summary_html(self, summary):
        """Generate summary section with table layout."""
        if not summary or 'checks' not in summary:
            return "<section><h2>Summary</h2><p>No summary data available.</p></section>"

        html = """
        <section>
            <h2>Summary</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Check</th>
                        <th>Status</th>
                        <th>Results</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        """

        for check_name, check_summary in summary['checks'].items():
            status = check_summary['status']
            if status == 'PASS':
                status_class = 'pass'
                status_icon = '✅'
            elif status == 'FAIL':
                status_class = 'fail'
                status_icon = '❌'
            else:  # SKIPPED
                status_class = 'skipped'
                status_icon = '⏭️'

            display_name = check_name.replace('_', ' ').title()

            # Extract key metrics from summary for the Results column
            summary_text = check_summary['summary']
            results_text = summary_text

            # Try to extract numbers for cleaner display
            if '/' in summary_text:
                # Extract fraction like "102/102" or "48/51"
                import re

                fraction_match = re.search(r'(\d+/\d+)', summary_text)
                if fraction_match:
                    results_text = fraction_match.group(1)

            html += f"""
                    <tr class="summary-row-{status_class}">
                        <td class="check-name">{display_name}</td>
                        <td class="status-cell">
                            <span class="status-badge status-{status_class}">
                                {status_icon} {status}
                            </span>
                        </td>
                        <td class="results-cell">{results_text}</td>
                        <td class="details-cell">{summary_text}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </section>
        """
        return html

    def _generate_gid_consistency_html(self, gid_results):
        """Generate GID inconsistencies section - only show failed nodes."""
        if not gid_results:
            return ""

        # Filter to only failed nodes
        failed_nodes = {node: result for node, result in gid_results.items() if result['status'] == 'FAIL'}

        if not failed_nodes:
            return ""  # No failures, no section needed

        html = """
        <section>
            <h2>GID Inconsistencies</h2>
            <p class="error-summary">The following nodes have GID consistency issues:</p>
            <table>
                <thead>
                    <tr>
                        <th>Node</th>
                        <th>Failed Interfaces</th>
                        <th>Issues</th>
                    </tr>
                </thead>
                <tbody>
        """

        for node, result in failed_nodes.items():
            # Build details for failed interfaces only
            failed_interfaces = []
            issues = []

            for iface_name, iface_result in result.get('interfaces', {}).items():
                if iface_result.get('status') != 'OK':
                    failed_interfaces.append(iface_name)
                    status = iface_result.get('status', 'UNKNOWN')
                    error = iface_result.get('error', '')
                    issues.append(f"{iface_name}: {status} ({error})" if error else f"{iface_name}: {status}")

            # Add general errors
            if result.get('errors'):
                issues.extend(result['errors'])

            failed_ifaces_str = ', '.join(failed_interfaces) if failed_interfaces else 'Multiple'
            issues_str = '; '.join(issues)

            html += f"""
                <tr>
                    <td>{node}</td>
                    <td>{failed_ifaces_str}</td>
                    <td>{issues_str}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </section>
        """
        return html

    @staticmethod
    def _rdma_pair_row_fields(pair_key, pair_result):
        """Parse one RDMA pair result for HTML/CSV rows. Returns None if ``pair_result`` is not a dict."""
        if not isinstance(pair_result, dict):
            return None
        error_details = pair_result.get('error_details', [])
        error_msg = 'Connection failed'
        if error_details:
            for err in error_details:
                if '|' in err:
                    error_msg = err.split('|')[0].strip()
                else:
                    error_msg = err
        if '(' in pair_key:
            base_pair = pair_key.split(' (')[0]
            interface = pair_key.split('(')[1].rstrip(')')
        else:
            base_pair = pair_key
            interface = pair_result.get('interface', 'default')
        return {
            'base_pair': base_pair,
            'interface': interface,
            'server_cmd': pair_result.get('server_cmd', 'N/A'),
            'client_cmd': pair_result.get('client_cmd', 'N/A'),
            'server_node': pair_result.get('server_node', 'unknown'),
            'client_node': pair_result.get('client_node', 'unknown'),
            'error_msg': error_msg,
            'server_output': pair_result.get('server_output', ''),
            'client_output': pair_result.get('client_output', ''),
            'status': pair_result.get('status'),
        }

    @staticmethod
    def _rdma_pair_error_plaintext(fields, max_log_chars=4096):
        """Build plain-text error column for CSV (mirrors HTML failure detail content without markup)."""
        if fields.get('status') != 'FAIL':
            return ''
        parts = [f"Error: {fields['error_msg']}"]
        so = '' if fields.get('server_output') is None else str(fields.get('server_output'))
        co = '' if fields.get('client_output') is None else str(fields.get('client_output'))
        if so.strip() and 'EMPTY_LOG' not in so:
            parts.append(f"Server Log:\n{so.strip()[:max_log_chars]}")
        elif 'EMPTY_LOG' in so:
            parts.append('Server Log: EMPTY_LOG')
        if co.strip() and 'EMPTY_LOG' not in co:
            parts.append(f"Client Log:\n{co.strip()[:max_log_chars]}")
        elif 'EMPTY_LOG' in co:
            parts.append('Client Log: EMPTY_LOG')
        return '\n'.join(parts)

    def _write_rdma_pairs_csv(self, html_path: Path):
        """
        Write **failed** RDMA ``pair_results`` rows to a UTF-8 CSV next to the HTML report.

        If there are no failures, no file is written (returns None).

        Columns match the HTML failure table: node pair, interface, ssh-wrapped commands, error text.
        """
        rc = self.results.get('rdma_connectivity') or {}
        if rc.get('skipped') or not rc.get('pair_results'):
            return None
        pair_results = rc['pair_results']
        failed_rows = []
        for pair_key in sorted(pair_results.keys()):
            pr = pair_results[pair_key]
            fields = self._rdma_pair_row_fields(pair_key, pr)
            if fields is None or fields.get('status') != 'FAIL':
                continue
            failed_rows.append(
                [
                    fields['base_pair'],
                    fields['interface'],
                    f'ssh {fields["server_node"]} "{fields["server_cmd"]}"',
                    f'ssh {fields["client_node"]} "{fields["client_cmd"]}"',
                    self._rdma_pair_error_plaintext(fields),
                ]
            )
        if not failed_rows:
            return None
        out_path = html_path.parent / f'{html_path.stem}_rdma_pairs.csv'
        headers = [
            'Failed Node Pair',
            'Interface',
            'Server Command',
            'Client Command',
            'Error Details',
        ]
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(failed_rows)
        return str(out_path)

    def _generate_rdma_topology_html(self, connectivity_results):
        """
        Partition groups, hosts per group, inter-group mode, and multi-wave schedule for full_mesh runs.
        """
        partition = connectivity_results.get('partition_groups') or {}
        if not partition:
            return ""

        inter_groups = connectivity_results.get('inter_groups') or {}
        mode = connectivity_results.get('inter_group_mode') or ''
        waves = connectivity_results.get('inter_group_waves') or []

        def _group_block(title, groups_map):
            parts = [f"<h4>{html.escape(title)}</h4>", '<dl style="margin:0 0 12px 0;">']
            for gid in sorted(groups_map.keys(), key=lambda x: (str(type(x)), str(x))):
                hosts = groups_map[gid]
                if not isinstance(hosts, (list, tuple)):
                    hosts = [hosts]
                host_list = ', '.join(html.escape(str(h)) for h in hosts)
                parts.append(
                    f"<dt style=\"margin-top:8px;font-weight:600;\"><code>{html.escape(str(gid))}</code></dt>"
                    f"<dd style=\"margin:4px 0 0 1em;\">{host_list}</dd>"
                )
            parts.append('</dl>')
            return '\n'.join(parts)

        blocks = [
            '<div class="rdma-topology" style="margin:12px 0;padding:12px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;">',
            '<h3>Topology (partition groups)</h3>',
            '<p style="margin-top:0;">Reachable nodes were partitioned for intra-group (Round 1) and inter-group (Round 2) testing.</p>',
            _group_block('Groups and hosts (initial partition)', partition),
        ]

        if inter_groups and inter_groups != partition:
            blocks.append(_group_block('Groups and hosts (inter-group phase, after prune)', inter_groups))

        mode_label = {
            'single_shot': 'single shot (one coordination round for all ordered group-pairs)',
            'multi_wave': 'multi-wave (ordered group-pairs split across waves)',
            'none': 'no inter-group phase (single group or empty)',
        }.get(mode, html.escape(str(mode)))

        blocks.append(f'<p><strong>Inter-group mode:</strong> {mode_label}</p>')

        ng = len(inter_groups) if inter_groups else len(partition)
        chunk = connectivity_results.get('inter_group_wave_chunk')
        if chunk is None and ng >= 2:
            chunk = max(1, ng - 1)
        elif chunk is None:
            chunk = 1
        if ng >= 2 and mode == 'multi_wave':
            n_ordered = ng * (ng - 1)
            n_waves_approx = (n_ordered + chunk - 1) // chunk
            blocks.append(
                '<p style="font-size:0.95em;color:#333;">'
                '<em>Why this many waves?</em> With <strong>G</strong> partition groups, inter-group '
                'testing schedules <strong>G×(G−1)</strong> <em>ordered</em> directed group-pairs '
                '(every Gi→Gj with i≠j), not one wave per source group where one group is server for all others. '
                'Multi-wave splits that ordered list into chunks of up to '
                f'<code>rdma_inter_group_wave_group_pairs</code> (resolved chunk size <strong>{chunk}</strong>; '
                'default <strong>max(1, G−1)</strong> when unset or <code>auto</code>), so the wave count is '
                f'<strong>ceil(G×(G−1) / {chunk})</strong>. With G={ng}, that is {n_ordered} ordered pairs '
                f'→ <strong>{n_waves_approx}</strong> wave(s).'
                '</p>'
            )

        if waves:
            blocks.append('<h4>Inter-group waves (ordered group-pair keys)</h4>')
            blocks.append('<ul style="margin-top:4px;">')
            total_w = len(waves)
            for w in waves:
                wi = w.get('wave', '?')
                n_gp = w.get('num_group_pairs', 0)
                keys = w.get('group_pair_keys') or []
                preview = ', '.join(html.escape(str(k)) for k in keys[:8])
                if len(keys) > 8:
                    preview += f' … (+{len(keys) - 8} more)'
                blocks.append(
                    f'<li style="margin-bottom:10px;"><strong>Wave {wi}/{total_w}</strong> — '
                    f'{n_gp} ordered group-pair(s)'
                    f'<br><span style="font-size:0.9em;color:#444;">{preview}</span></li>'
                )
            blocks.append('</ul>')

        blocks.append('</div>')
        return '\n'.join(blocks)

    def _generate_pruned_nodes_after_intra_html(self, records):
        """HTML block listing nodes removed after Round 1 before inter-group tests."""
        if not records:
            return ""
        rows = []
        for r in records:
            rows.append(
                f"<tr><td><code>{html.escape(str(r.get('node', '')))}</code></td>"
                f"<td>{html.escape(str(r.get('group_id', '')))}</td>"
                f"<td>{html.escape(str(r.get('reason', '')))}</td></tr>"
            )
        return f"""
        <div class="pruned-nodes" style="margin:15px 0;padding:12px;background:#fff3cd;border:1px solid #ffc107;border-radius:4px;">
            <h3>Nodes excluded before Round 2 (inter-group)</h3>
            <p style="margin-top:0;">These nodes met the configured <strong>intra-group peer failure fraction</strong> threshold (see <code>rdma_prune_peer_failure_threshold</code>): enough distinct peers in the same partition group had at least one failing intra test versus that node. They were omitted from inter-group tests.</p>
            <table class="summary-table">
                <thead><tr><th>Node</th><th>Group</th><th>Reason</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """

    def _rdma_failure_detail_table_rows(self, pair_results, round_tag, max_detail_rows, inter_wave=None):
        """
        Build <tr>…</tr> fragments for failed pairs. ``round_tag`` None = all failures.
        ``inter_wave`` filters inter-group rows to one wave (legacy rows without ``inter_wave`` count as wave 1).
        Returns (html_fragment, rows_emitted).
        """
        row_fragments = []
        if not pair_results:
            return '', 0

        for pair_key, pair_result in pair_results.items():
            if not isinstance(pair_result, dict) or pair_result.get('status') != 'FAIL':
                continue
            if round_tag is not None and pair_result.get('round') != round_tag:
                continue
            if inter_wave is not None:
                if pair_result.get('round') != 'inter_group':
                    continue
                pw = pair_result.get('inter_wave') or 1
                if pw != inter_wave:
                    continue
            if max_detail_rows > 0 and len(row_fragments) >= max_detail_rows:
                break

            fields = self._rdma_pair_row_fields(pair_key, pair_result)
            if fields is None:
                continue
            base_pair = fields['base_pair']
            interface = fields['interface']
            server_cmd = fields['server_cmd']
            client_cmd = fields['client_cmd']
            server_node = fields['server_node']
            client_node = fields['client_node']
            error_msg = fields['error_msg']

            error_details_html = f"<strong>Error:</strong> {html.escape(str(error_msg))}<br>"

            server_output = fields.get('server_output', '')
            if server_output and server_output.strip() and 'EMPTY_LOG' not in server_output:
                so = html.escape(server_output.strip()[:4096])
                error_details_html += f"<br><strong>Server Log:</strong><br><pre style='font-size:0.8em;max-height:100px;overflow-y:auto;background:#f5f5f5;padding:5px;'>{so}</pre>"
            elif 'EMPTY_LOG' in str(server_output):
                error_details_html += "<br><strong>Server Log:</strong> <span style='color:#888;'>EMPTY_LOG</span>"

            client_output = fields.get('client_output', '')
            if client_output and client_output.strip() and 'EMPTY_LOG' not in client_output:
                co = html.escape(client_output.strip()[:4096])
                error_details_html += f"<br><strong>Client Log:</strong><br><pre style='font-size:0.8em;max-height:100px;overflow-y:auto;background:#f5f5f5;padding:5px;'>{co}</pre>"
            elif 'EMPTY_LOG' in str(client_output):
                error_details_html += "<br><strong>Client Log:</strong> <span style='color:#888;'>EMPTY_LOG</span>"

            row_fragments.append(
                f"""
                        <tr>
                            <td>{html.escape(base_pair)}</td>
                            <td>{html.escape(interface)}</td>
                            <td><code>ssh {html.escape(str(server_node))} "{html.escape(str(server_cmd))}"</code></td>
                            <td><code>ssh {html.escape(str(client_node))} "{html.escape(str(client_cmd))}"</code></td>
                            <td style="max-width:400px;">{error_details_html}</td>
                        </tr>
                    """
            )

        return ''.join(row_fragments), len(row_fragments)

    def _generate_connectivity_html(self, connectivity_results):
        """Generate RDMA connectivity section: topology, pruned nodes, and failure analysis."""
        if not connectivity_results:
            return ""

        if connectivity_results.get('skipped', False):
            ex_if = connectivity_results.get('excluded_nodes_interface_check') or []
            ex_gid = connectivity_results.get('excluded_nodes_gid') or []
            if ex_if or ex_gid:
                msg = html.escape(str(connectivity_results.get('message', 'RDMA connectivity test skipped')))
                iface_items = ''.join(f'<li><code>{html.escape(str(n))}</code></li>' for n in ex_if)
                gid_items = ''.join(f'<li><code>{html.escape(str(n))}</code></li>' for n in ex_gid)
                iface_block = (
                    f"""<div style="margin:12px 0;padding:12px;background:#e7f3ff;border:1px solid #b8daff;border-radius:4px;">
            <h3>Excluded (interface presence)</h3><ul style="margin:0;">{iface_items}</ul></div>"""
                    if ex_if
                    else ""
                )
                gid_block = (
                    f"""<div style="margin:12px 0;padding:12px;background:#fff8e7;border:1px solid #f0d28c;border-radius:4px;">
            <h3>Excluded (GID consistency)</h3><ul style="margin:0;">{gid_items}</ul></div>"""
                    if ex_gid
                    else ""
                )
                return f"""
        <section>
            <h2>RDMA connectivity</h2>
            <p class="error-summary">{msg}</p>
            {iface_block}
            {gid_block}
        </section>
        """
            return ""

        failed_pairs = connectivity_results.get('failed_pairs', 0)
        pruned = connectivity_results.get('pruned_nodes_after_intra') or []
        pair_results = connectivity_results.get('pair_results') or {}
        partition_groups = connectivity_results.get('partition_groups') or {}

        topology_html = self._generate_rdma_topology_html(connectivity_results) if partition_groups else ""

        excluded_iface = connectivity_results.get('excluded_nodes_interface_check') or []
        excluded_iface_html = ''
        if excluded_iface:
            items = ''.join(f'<li><code>{html.escape(str(n))}</code></li>' for n in excluded_iface)
            excluded_iface_html = f"""
        <div class="excluded-interface-nodes" style="margin:12px 0;padding:12px;background:#e7f3ff;border:1px solid #b8daff;border-radius:4px;">
            <h3>Nodes excluded before RDMA mesh (interface presence)</h3>
            <p style="margin-top:0;">These nodes failed the <strong>interface presence</strong> check (missing, down, or inactive expected RDMA interfaces) and were not included in RDMA connectivity testing.</p>
            <ul style="margin:0;">{items}</ul>
        </div>
        """

        excluded_gid = connectivity_results.get('excluded_nodes_gid') or []
        excluded_gid_html = ''
        if excluded_gid:
            gitems = ''.join(f'<li><code>{html.escape(str(n))}</code></li>' for n in excluded_gid)
            excluded_gid_html = f"""
        <div class="excluded-gid-nodes" style="margin:12px 0;padding:12px;background:#fff8e7;border:1px solid #f0d28c;border-radius:4px;">
            <h3>Nodes excluded before RDMA mesh (GID consistency)</h3>
            <p style="margin-top:0;">These nodes failed <strong>GID consistency</strong> for the configured index and were not included in RDMA connectivity testing.</p>
            <ul style="margin:0;">{gitems}</ul>
        </div>
        """

        excluded_panels_html = excluded_iface_html + excluded_gid_html

        if failed_pairs == 0 and not pruned and not topology_html and not excluded_panels_html:
            return ""

        cfg = self.config_dict or {}
        try:
            max_detail_rows = int(cfg.get('rdma_report_max_detail_rows', 5000))
        except (TypeError, ValueError):
            max_detail_rows = 5000

        pruned_html = self._generate_pruned_nodes_after_intra_html(pruned)

        if failed_pairs == 0:
            return f"""
        <section>
            <h2>RDMA connectivity</h2>
            {excluded_panels_html}
            {topology_html}
            {pruned_html}
        </section>
        """

        use_rounds = any(
            isinstance(p, dict) and p.get('round') in ('intra_group', 'inter_group') for p in pair_results.values()
        )

        table_header = """
                    <table>
                <thead>
                    <tr>
                        <th>Failed Node Pair</th>
                        <th>Interface</th>
                        <th>Server Command</th>
                        <th>Client Command</th>
                        <th>Error Details</th>
                    </tr>
                </thead>
                <tbody>
        """

        blocks = []

        def append_round_block(title, rtag, inter_wave_idx=None):
            if rtag == 'inter_group' and inter_wave_idx is not None:
                n_fail = sum(
                    1
                    for p in pair_results.values()
                    if isinstance(p, dict)
                    and p.get('status') == 'FAIL'
                    and p.get('round') == 'inter_group'
                    and (p.get('inter_wave') or 1) == inter_wave_idx
                )
            else:
                n_fail = sum(
                    1
                    for p in pair_results.values()
                    if isinstance(p, dict) and p.get('status') == 'FAIL' and p.get('round') == rtag
                )
            if n_fail == 0:
                return
            log_preview = n_fail if max_detail_rows <= 0 else min(n_fail, max_detail_rows)
            iw = inter_wave_idx if rtag == 'inter_group' else None
            detail_body, n_shown = self._rdma_failure_detail_table_rows(
                pair_results, rtag, max_detail_rows, inter_wave=iw
            )
            omitted = n_fail - n_shown if max_detail_rows > 0 and n_fail > n_shown else 0
            omit_note = (
                f"<p><em>{omitted:,} additional failures in this round omitted (rdma_report_max_detail_rows).</em></p>"
                if omitted
                else ""
            )
            safe_title = html.escape(title)
            blocks.append(
                f"""
            <details style="margin-bottom:18px;">
                <summary><strong>{safe_title}</strong> ({n_fail} failed tests)</summary>
                <div class="content" style="margin-top:10px;">
                    <p><em>Detailed rows (up to {log_preview} shown):</em></p>
                    {omit_note}
                    {table_header}
                    {detail_body}
                </tbody>
            </table>
                </div>
            </details>
                    """
            )

        if use_rounds:
            append_round_block('Round 1 — Intra-group failures', 'intra_group')

            inter_mode = connectivity_results.get('inter_group_mode') or ''
            inter_waves = connectivity_results.get('inter_group_waves') or []
            split_inter = inter_mode == 'multi_wave' and len(inter_waves) > 1

            if split_inter:
                total_w = len(inter_waves)
                for wmeta in inter_waves:
                    widx = wmeta.get('wave', 1)
                    n_gp = wmeta.get('num_group_pairs', 0)
                    title = f'Inter-group wave {widx}/{total_w} ({n_gp} ordered group-pairs in this wave)'
                    append_round_block(title, 'inter_group', inter_wave_idx=widx)
            else:
                append_round_block('Round 2 — Inter-group failures', 'inter_group', inter_wave_idx=None)

            rounds_body = '\n'.join(blocks) if blocks else '<p><em>No per-round failure breakdown available.</em></p>'
        else:
            log_preview = failed_pairs if max_detail_rows <= 0 else min(failed_pairs, max_detail_rows)
            detail_body, n_shown = self._rdma_failure_detail_table_rows(pair_results, None, max_detail_rows)
            omitted = failed_pairs - n_shown if max_detail_rows > 0 and failed_pairs > n_shown else 0
            omit_note = (
                f"<p><em>{omitted:,} additional failures omitted (rdma_report_max_detail_rows).</em></p>"
                if omitted
                else ""
            )
            rounds_body = f"""
            <details style="margin-bottom:18px;">
                <summary><strong>All failed connection tests</strong> ({failed_pairs} failed tests)</summary>
                <div class="content" style="margin-top:10px;">
                    <p><em>Detailed rows (up to {log_preview} shown):</em></p>
                    {omit_note}
                    {table_header}
                    {detail_body}
                </tbody>
            </table>
                </div>
            </details>
            """

        return f"""
        <section>
            <h2>RDMA connectivity</h2>
            {excluded_panels_html}
            {topology_html}
            <h3>Connection failures</h3>
            <p class="error-summary">Found {failed_pairs} failed connection(s) out of {connectivity_results.get('total_pairs', 0)} total pairs tested.</p>
            {pruned_html}
            {rounds_body}
        </section>
        """

    def _generate_ssh_connectivity_html(self, ssh_results):
        """Generate SSH connection failures section - only show failed pairs."""
        if not ssh_results:
            return ""

        if ssh_results.get('skipped', False):
            return ""  # Skip section entirely if test was skipped

        # Check if there are any failures
        failed_pairs = ssh_results.get('failed_pairs', 0)
        if failed_pairs == 0:
            return ""  # No failures, no section needed

        html = f"""
        <section>
            <h2>SSH Connection Failures</h2>
            <p class="error-summary">Found {failed_pairs} failed SSH connection(s) out of {ssh_results.get('total_pairs', 0)} total pairs tested:</p>
            <table>
                <thead>
                    <tr>
                        <th>Source Node</th>
                        <th>Target Node</th>
                        <th>Error Details</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Only show failed pairs
        if ssh_results.get('pair_results'):
            # Sort failed pairs for consistent display
            failed_pairs_list = []
            for pair_key, status in ssh_results['pair_results'].items():
                if status.startswith('FAILED'):
                    # Parse pair key (format: "source → target")
                    if ' → ' in pair_key:
                        source_node, target_node = pair_key.split(' → ', 1)
                        error_msg = (
                            status.replace('FAILED - ', '') if 'FAILED - ' in status else 'SSH connection failed'
                        )
                        failed_pairs_list.append((source_node.strip(), target_node.strip(), error_msg))

            # Sort by source node, then target node
            failed_pairs_list.sort(key=lambda x: (x[0], x[1]))

            # Generate table rows
            for source_node, target_node, error_msg in failed_pairs_list:
                html += f"""
                    <tr>
                        <td>{source_node}</td>
                        <td>{target_node}</td>
                        <td class="error-details">{error_msg}</td>
                    </tr>
                """

        html += """
                </tbody>
            </table>
        </section>
        """

        return html

    def _generate_rocm_versions_html(self, rocm_results):
        """Generate ROCm version inconsistencies section - only show failed nodes."""
        if not rocm_results:
            return ""

        # Filter to only failed nodes
        failed_nodes = {node: result for node, result in rocm_results.items() if result['status'] == 'FAIL'}

        if not failed_nodes:
            return ""  # No failures, no section needed

        html = """
        <section>
            <h2>ROCm Version Inconsistencies</h2>
            <p class="error-summary">The following nodes have ROCm version mismatches:</p>
            <table>
                <thead>
                    <tr>
                        <th>Node</th>
                        <th>Detected Version</th>
                        <th>Expected Version</th>
                        <th>Issue</th>
                    </tr>
                </thead>
                <tbody>
        """

        for node, result in failed_nodes.items():
            errors = ', '.join(result.get('errors', [])) if result.get('errors') else 'Version mismatch'

            html += f"""
                <tr>
                    <td>{node}</td>
                    <td>{result.get('detected_version', 'Unknown')}</td>
                    <td>{result.get('expected_version', 'Unknown')}</td>
                    <td>{errors}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </section>
        """
        return html

    def _generate_interface_names_html(self, interface_results):
        """Generate RDMA interface inconsistencies section - only show failed nodes."""
        if not interface_results:
            return ""

        # Filter to only failed nodes
        failed_nodes = {node: result for node, result in interface_results.items() if result['status'] == 'FAIL'}

        if not failed_nodes:
            return ""  # No failures, no section needed

        html = """
        <section>
            <h2>RDMA Interface Inconsistencies</h2>
            <p class="error-summary">The following nodes have RDMA interface issues:</p>
            <table>
                <thead>
                    <tr>
                        <th>Node</th>
                        <th>Missing</th>
                        <th>Inactive</th>
                        <th>Down</th>
                        <th>Issues</th>
                    </tr>
                </thead>
                <tbody>
        """

        for node, result in failed_nodes.items():
            missing_interfaces = ', '.join(result.get('missing_interfaces', [])) or 'None'
            inactive_interfaces = ', '.join(result.get('inactive_interfaces', [])) or 'None'
            down_interfaces = ', '.join(result.get('down_interfaces', [])) or 'None'
            errors = ', '.join(result.get('errors', [])) if result.get('errors') else 'Interface issues detected'

            html += f"""
                <tr>
                    <td>{node}</td>
                    <td>{missing_interfaces}</td>
                    <td>{inactive_interfaces}</td>
                    <td>{down_interfaces}</td>
                    <td>{errors}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </section>
        """
        return html

    def _generate_configuration_html(self, config_dict):
        """Generate configuration section."""
        html = """
        <section>
            <h2>Test Configuration</h2>
            <div class="config-section">
                <pre>
        """

        # Filter out comment fields for cleaner display
        clean_config = {k: v for k, v in config_dict.items() if not k.startswith('_comment')}
        html += json.dumps(clean_config, indent=2)

        html += """
                </pre>
            </div>
        </section>
        """
        return html

    def _generate_recommendations_html(self, recommendations):
        """Generate recommendations section."""
        if not recommendations:
            return ""

        html = """
        <section>
            <div class="recommendations">
                <h3>Recommendations</h3>
                <ul>
        """

        for recommendation in recommendations:
            html += f"<li>{recommendation}</li>"

        html += """
                </ul>
            </div>
        </section>
        """
        return html
