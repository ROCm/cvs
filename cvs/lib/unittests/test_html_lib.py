import unittest
import tempfile
import os
import json
from io import StringIO
from unittest.mock import patch

# Import the module under test
import cvs.lib.html_lib as html_lib


class TestNormalizeBytes(unittest.TestCase):
    def test_bytes_only(self):
        self.assertEqual(html_lib.normalize_bytes(932), "932 B")

    def test_kilobytes_binary(self):
        self.assertEqual(html_lib.normalize_bytes(2048), "2 KB")

    def test_kilobytes_decimal(self):
        self.assertEqual(html_lib.normalize_bytes(2000, si=True), "2 kB")

    def test_megabytes(self):
        self.assertEqual(html_lib.normalize_bytes(5 * 1024 * 1024), "5 MB")

    def test_gigabytes(self):
        self.assertEqual(html_lib.normalize_bytes(3 * 1024**3), "3 GB")

    def test_negative_bytes(self):
        self.assertEqual(html_lib.normalize_bytes(-1024), "-1 KB")

    def test_precision(self):
        self.assertEqual(html_lib.normalize_bytes(1536, precision=1), "1.5 KB")

    def test_type_error(self):
        with self.assertRaises(TypeError):
            html_lib.normalize_bytes("not a number")


class TestBuildHtmlMemUtilizationTable(unittest.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8")
        self.filename = self.tmp_file.name

    def tearDown(self):
        self.tmp_file.close()
        os.remove(self.filename)

    def test_single_node_valid_input(self):
        use_dict = {
            "node1": {
                **{
                    f"card{i}": {
                        "GPU Memory Allocated (VRAM%)": f"{i * 10}%",
                        "GPU Memory Read/Write Activity (%)": f"{i * 5}%",
                        "Memory Activity": f"{i * 3}%",
                        "Avg. Memory Bandwidth": f"{i * 2} GB/s",
                    }
                    for i in range(8)
                }
            }
        }

        amd_dict = {
            "node1": [
                {
                    "mem_usage": {
                        "total_vram": {"value": "16384"},
                        "used_vram": {"value": "8192"},
                        "free_vram": {"value": "8192"},
                    }
                }
                for _ in range(8)
            ]
        }

        html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)
        with open(self.filename, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("GPU Memory Utilization", content)
            self.assertIn("G0 Tot VRAM MB", content)
            self.assertIn("node1", content)
            self.assertIn("8192", content)
            self.assertIn("10%", content)

    def test_multiple_nodes(self):
        use_dict = {
            f"node{i}": {
                **{
                    f"card{j}": {
                        "GPU Memory Allocated (VRAM%)": f"{j * 10}%",
                        "GPU Memory Read/Write Activity (%)": f"{j * 5}%",
                        "Memory Activity": f"{j * 3}%",
                        "Avg. Memory Bandwidth": f"{j * 2} GB/s",
                    }
                    for j in range(8)
                }
            }
            for i in range(2)
        }

        amd_dict = {
            f"node{i}": [
                {
                    "mem_usage": {
                        "total_vram": {"value": "16384"},
                        "used_vram": {"value": "8192"},
                        "free_vram": {"value": "8192"},
                    }
                }
                for _ in range(8)
            ]
            for i in range(2)
        }

        html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)
        with open(self.filename, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("node0", content)
            self.assertIn("node1", content)

    def test_rocm7_style_gpu_data(self):
        use_dict = {
            "node1": {
                **{
                    f"card{i}": {
                        "GPU Memory Allocated (VRAM%)": f"{i * 10}%",
                        "GPU Memory Read/Write Activity (%)": f"{i * 5}%",
                        "Memory Activity": f"{i * 3}%",
                        "Avg. Memory Bandwidth": f"{i * 2} GB/s",
                    }
                    for i in range(8)
                }
            }
        }

        amd_dict = {
            "node1": {
                "gpu_data": [
                    {
                        "mem_usage": {
                            "total_vram": {"value": "16384"},
                            "used_vram": {"value": "8192"},
                            "free_vram": {"value": "8192"},
                        }
                    }
                    for _ in range(8)
                ]
            }
        }

        html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)
        with open(self.filename, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("GPU Memory Utilization", content)
            self.assertIn("G0 Tot VRAM MB", content)
            self.assertIn("node1", content)

    def test_missing_gpu_key_raises_keyerror(self):
        use_dict = {
            "node1": {
                "card0": {
                    "GPU Memory Allocated (VRAM%)": "10%",
                    "GPU Memory Read/Write Activity (%)": "20%",
                    "Memory Activity": "30%",
                    "Avg. Memory Bandwidth": "40 GB/s",
                }
                # Missing card1 to card7
            }
        }

        amd_dict = {
            "node1": [
                {
                    "mem_usage": {
                        "total_vram": {"value": "16384"},
                        "used_vram": {"value": "8192"},
                        "free_vram": {"value": "8192"},
                    }
                }
                for _ in range(8)
            ]
        }

        with self.assertRaises(KeyError):
            html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)


class TestBuildRcclHeatmapTable(unittest.TestCase):
    def setUp(self):
        """Create temporary files for HTML output and JSON data."""
        self.html_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".html")
        self.html_filename = self.html_file.name

        self.actual_json_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".json")
        self.actual_json_filename = self.actual_json_file.name

        self.ref_json_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".json")
        self.ref_json_filename = self.ref_json_file.name

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.html_file, self.actual_json_file, self.ref_json_file]:
            f.close()
        for fname in [self.html_filename, self.actual_json_filename, self.ref_json_filename]:
            if os.path.exists(fname):
                os.remove(fname)

    def _create_json_file(self, filepath, data):
        """Helper to write JSON data to a file."""
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def test_3part_keys_matching_3part_reference(self):
        """Test 3-part keys in both actual and reference data."""
        actual_data = {
            "all_reduce_perf-float-8": {
                "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"},
                "2097152": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "84.0"},
            }
        }
        ref_data = {
            "all_reduce_perf-float-8": {
                "1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"},
                "2097152": {"bus_bw": "255.0", "alg_bw": "127.5", "time": "82.0"},
            }
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Test Heatmap", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("Test Heatmap", content)
            self.assertIn("all_reduce_perf", content)
            self.assertIn("float", content)
            self.assertIn("8", content)
            self.assertIn("245.5", content)
            self.assertIn("250.0", content)
            self.assertIn("<table id=\"rccltable\"", content)

    def test_4part_keys_with_3part_reference_fallback(self):
        """Test 4-part keys in actual falling back to 3-part reference keys."""
        actual_data = {
            "all_reduce_perf-float-8-chdefault": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}}
        }
        ref_data = {
            "all_reduce_perf-float-8": {  # 3-part key
                "1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}
            }
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Test Fallback", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("all_reduce_perf", content)
            self.assertIn("245.5", content)
            self.assertIn("250.0", content)  # Reference data should be used

    def test_4part_keys_matching_4part_reference(self):
        """Test 4-part keys matching 4-part reference directly."""
        actual_data = {
            "all_reduce_perf-float-8-chdefault": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}}
        }
        ref_data = {
            "all_reduce_perf-float-8-chdefault": {  # 4-part key matches
                "1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}
            }
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Test 4-part", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("245.5", content)
            self.assertIn("250.0", content)

    def test_4part_numeric_channel_config_fallback(self):
        """Test 4-part keys with numeric channel config (ch16-16) falling back to 3-part reference."""
        actual_data = {
            "all_reduce_perf-float-8-ch16-16": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}},
            "broadcast_perf-bfloat16-16-ch32-32": {"2097152": {"bus_bw": "200.0", "alg_bw": "100.0", "time": "50.0"}},
        }
        ref_data = {
            "all_reduce_perf-float-8": {  # 3-part key
                "1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}
            },
            "broadcast_perf-bfloat16-16": {  # 3-part key
                "2097152": {"bus_bw": "210.0", "alg_bw": "105.0", "time": "48.0"}
            },
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Channel Sweep Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            # Both keys should match using fallback
            self.assertIn("all_reduce_perf", content)
            self.assertIn("broadcast_perf", content)
            self.assertIn("245.5", content)
            self.assertIn("250.0", content)  # Reference for all_reduce
            self.assertIn("200.0", content)
            self.assertIn("210.0", content)  # Reference for broadcast

    def test_wrapped_reference_format(self):
        """Test reference JSON with metadata/result wrapper."""
        actual_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}}}
        ref_data_wrapped = {
            "metadata": {"test_date": "2026-01-20", "platform": "MI300X"},
            "result": {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}},
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data_wrapped)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Wrapped Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("245.5", content)
            self.assertIn("250.0", content)

    def test_unwrapped_reference_format(self):
        """Test reference JSON without wrapper (backward compatibility)."""
        actual_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}}}
        ref_data_unwrapped = {
            "all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data_unwrapped)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Unwrapped Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("245.5", content)
            self.assertIn("250.0", content)

    @patch('sys.stdout', new_callable=StringIO)
    def test_missing_keys_summary_not_spam(self, mock_stdout):
        """Test that missing keys show summary, not individual warnings."""
        actual_data = {
            "all_reduce_perf-float-8": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}},
            "broadcast_perf-bfloat16-16": {"1048576": {"bus_bw": "200.0", "alg_bw": "100.0", "time": "50.0"}},
            "reduce_scatter_perf-float-4": {"1048576": {"bus_bw": "180.0", "alg_bw": "90.0", "time": "55.0"}},
        }
        ref_data = {
            "all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}
            # Missing the other two keys
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Missing Keys Test", self.actual_json_filename, self.ref_json_filename
        )

        output = mock_stdout.getvalue()
        # Should have summary message
        self.assertIn("2 test keys not found in reference data", output)
        self.assertIn("skipped from table", output)
        # Should show sample keys
        self.assertTrue("broadcast_perf-bfloat16-16" in output or "reduce_scatter_perf-float-4" in output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_missing_message_sizes_summary(self, mock_stdout):
        """Test that missing message sizes are counted and summarized."""
        actual_data = {
            "all_reduce_perf-float-8": {
                "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"},
                "2097152": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "84.0"},
                "4194304": {"bus_bw": "255.0", "alg_bw": "127.5", "time": "165.0"},
            }
        }
        ref_data = {
            "all_reduce_perf-float-8": {
                "1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}
                # Missing 2097152 and 4194304
            }
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Missing Msg Sizes", self.actual_json_filename, self.ref_json_filename
        )

        output = mock_stdout.getvalue()
        # Should show count of missing message sizes
        self.assertIn("2 msg_size entries not found in reference data", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_invalid_key_format_skipped(self, mock_stdout):
        """Test that keys with invalid format are skipped with warning."""
        actual_data = {
            "invalid-key": {  # Only 2 parts
                "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}
            },
            "all_reduce_perf-float-8": {  # Valid 3 parts
                "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}
            },
        }
        ref_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}}

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Invalid Key Test", self.actual_json_filename, self.ref_json_filename
        )

        output = mock_stdout.getvalue()
        # Should warn about invalid format
        self.assertIn("Invalid key format", output)
        self.assertIn("invalid-key", output)

    def test_mixed_3part_and_4part_keys(self):
        """Test dataset with both 3-part and 4-part keys."""
        actual_data = {
            "all_reduce_perf-float-8": {  # 3-part
                "1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}
            },
            "broadcast_perf-bfloat16-16-chdefault": {  # 4-part
                "1048576": {"bus_bw": "200.0", "alg_bw": "100.0", "time": "50.0"}
            },
        }
        ref_data = {
            "all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}},
            "broadcast_perf-bfloat16-16": {  # 3-part for fallback
                "1048576": {"bus_bw": "210.0", "alg_bw": "105.0", "time": "48.0"}
            },
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Mixed Keys Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("all_reduce_perf", content)
            self.assertIn("broadcast_perf", content)
            self.assertIn("245.5", content)
            self.assertIn("200.0", content)

    def test_performance_regression_color_coding(self):
        """Test that performance regressions are highlighted with danger class."""
        actual_data = {
            "all_reduce_perf-float-8": {
                "1048576": {"bus_bw": "200.0", "alg_bw": "100.0", "time": "50.0"}  # Worse than reference
            }
        }
        ref_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}}

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Regression Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            # Lower bandwidth and higher time should be marked as danger
            self.assertIn("label-danger", content)

    def test_performance_improvement_no_danger(self):
        """Test that performance improvements are not marked as danger."""
        actual_data = {
            "all_reduce_perf-float-8": {
                "1048576": {"bus_bw": "260.0", "alg_bw": "130.0", "time": "38.0"}  # Better than reference
            }
        }
        ref_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}}

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Improvement Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            # Should have the improved values without danger class
            self.assertIn("260.0", content)
            self.assertIn("130.0", content)
            self.assertIn("38.0", content)

    def test_table_structure(self):
        """Test that HTML table has correct structure."""
        actual_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}}}
        ref_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": "250.0", "alg_bw": "125.0", "time": "40.0"}}}

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        html_lib.build_rccl_heatmap_table(
            self.html_filename, "Structure Test", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            # Check for required HTML elements
            self.assertIn('<table id="rccltable"', content)
            self.assertIn('<thead>', content)
            self.assertIn('<th>Collective</th>', content)
            self.assertIn('<th>DataType</th>', content)
            self.assertIn('<th>Number of GPUs</th>', content)
            self.assertIn('<th>Msg Size</th>', content)
            self.assertIn('</table>', content)


class TestBuildRcclHeatmap(unittest.TestCase):
    """Test cases for build_rccl_heatmap function."""

    def setUp(self):
        """Create temporary files for HTML output and JSON data."""
        self.html_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".html")
        self.html_filename = self.html_file.name

        self.actual_json_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".json")
        self.actual_json_filename = self.actual_json_file.name

        self.ref_json_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".json")
        self.ref_json_filename = self.ref_json_file.name

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.html_file, self.actual_json_file, self.ref_json_file]:
            f.close()
        for fname in [self.html_filename, self.actual_json_filename, self.ref_json_filename]:
            if os.path.exists(fname):
                os.remove(fname)

    def _create_json_file(self, filepath, data):
        """Helper to write JSON data to a file."""
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def test_wrapped_reference_format_heatmap(self):
        """Test heatmap generation with wrapped reference format."""
        actual_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": 245.5, "alg_bw": 122.75, "time": 42.3}}}
        ref_data_wrapped = {
            "metadata": {"test_date": "2026-01-20"},
            "result": {"all_reduce_perf-float-8": {"1048576": {"bus_bw": 250.0, "alg_bw": 125.0, "time": 40.0}}},
        }

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data_wrapped)

        html_lib.build_rccl_heatmap(
            self.html_filename, "heatmapchart", "Test Heatmap", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("Test Heatmap", content)
            # Should generate chart content
            self.assertTrue(len(content) > 0)

    def test_unwrapped_reference_format_heatmap(self):
        """Test heatmap generation with unwrapped reference format."""
        actual_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": 245.5, "alg_bw": 122.75, "time": 42.3}}}
        ref_data_unwrapped = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": 250.0, "alg_bw": 125.0, "time": 40.0}}}

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data_unwrapped)

        html_lib.build_rccl_heatmap(
            self.html_filename, "heatmapchart", "Test Unwrapped", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            self.assertIn("Test Unwrapped", content)
            self.assertTrue(len(content) > 0)

    def test_heatmap_chart_name(self):
        """Test that chart name is properly used in heatmap."""
        actual_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": 245.5, "alg_bw": 122.75, "time": 42.3}}}
        ref_data = {"all_reduce_perf-float-8": {"1048576": {"bus_bw": 250.0, "alg_bw": 125.0, "time": 40.0}}}

        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, ref_data)

        chart_name = "my_custom_chart_div"
        html_lib.build_rccl_heatmap(
            self.html_filename, chart_name, "Test Chart Name", self.actual_json_filename, self.ref_json_filename
        )

        with open(self.html_filename, 'r') as f:
            content = f.read()
            # Chart name should appear in the output
            self.assertTrue(len(content) > 0)


if __name__ == "__main__":
    unittest.main()
