'''Pytest-html and JUnit integration coverage for vLLM metric subtests.'''

import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path


class TestVllmReportingIntegration(unittest.TestCase):
    def test_bare_parent_rows_subtests_html_and_junit_property(self):
        source = textwrap.dedent(
            '''
            from types import SimpleNamespace

            import pytest

            from cvs.tests.inference.vllm._common import (
                test_verify_cell_metrics as verify_cell_metrics,
            )

            CELL = "ISL=128,OSL=128,TP=1,PP=1,CONC=1"

            @pytest.fixture
            def run():
                return SimpleNamespace(
                    cell=SimpleNamespace(
                        key=CELL,
                        isl=128,
                        osl=128,
                        concurrency=1,
                    )
                )

            @pytest.fixture
            def variant_config():
                return SimpleNamespace(
                    model_id="model",
                    enforce_thresholds=True,
                    thresholds={
                        CELL: {
                            "output_throughput": {"kind": "min", "value": 1}
                        }
                    },
                )

            @pytest.fixture
            def inf_res_dict(run, variant_config):
                key = (
                    variant_config.model_id,
                    "",
                    "128",
                    "128",
                    CELL,
                    1,
                )
                return {
                    key: {
                        "head": {
                            "output_throughput": 2.0,
                            "mean_ttft_ms": 3.0,
                        }
                    }
                }

            @pytest.fixture
            def lifecycle():
                return SimpleNamespace(record=lambda *args: None)

            def test_verify_cell_metrics(
                run,
                inf_res_dict,
                variant_config,
                lifecycle,
                request,
                subtests,
            ):
                verify_cell_metrics(
                    run,
                    inf_res_dict,
                    variant_config,
                    lifecycle,
                    request,
                    subtests,
                )
            '''
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_path = root / "test_vllm_reporting.py"
            html_path = root / "report.html"
            xml_path = root / "report.xml"
            test_path.write_text(source)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_path),
                    "-p",
                    "cvs.tests.inference.vllm.conftest",
                    f"--html={html_path}",
                    "--self-contained-html",
                    f"--junitxml={xml_path}",
                    "-q",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            html = html_path.read_text()
            xml_root = element_tree.parse(xml_path).getroot()

        self.assertIn("output_throughput", html)
        self.assertIn("mean_ttft_ms", html)
        self.assertNotIn("client.output_throughput", html)
        properties = [
            prop
            for prop in xml_root.findall(".//property")
            if prop.attrib.get("name") == "cvs_vllm_metrics_v1"
        ]
        self.assertEqual(len(properties), 1)
        value = properties[0].attrib["value"]
        self.assertIn('"metric_contract":{"id":"vllm-bare","version":1}', value)
        self.assertIn('"actuals_by_host":{"head":', value)
        self.assertIn('"output_throughput":2.0', value)
        self.assertNotIn("client.", value)


if __name__ == "__main__":
    unittest.main()
