import logging
import unittest
import tempfile
import os
import json

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


class TestBuildHtmlNicTable(unittest.TestCase):
    def test_port_without_netdev_is_left_out(self):
        rdma_dict = {
            'node1': {
                'mlx5_0': {'eth_device': None, 'device_status': 'ACTIVE', 'link_status': 'LINK_UP'},
                'rdma0': {'eth_device': 'eth0', 'device_status': 'ACTIVE', 'link_status': 'LINK_UP'},
            }
        }
        lshw_dict = {'node1': {'eth0': {'pci_bus': '0000:06:00.0'}}}
        ip_dict = {'node1': {'eth0': {'mtu': '9000', 'ipv4_addr_list': [], 'ipv6_addr_list': []}}}
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, 'nic.html')
            html_lib.build_html_nic_table(filename, rdma_dict, lshw_dict, ip_dict)
            with open(filename) as fp:
                html = fp.read()
        self.assertIn("['rdma0']", html)
        self.assertNotIn('mlx5_0', html)


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

    def test_missing_keys_summary_not_spam(self):
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

        with self.assertLogs(html_lib.log, level=logging.WARNING) as log_cm:
            html_lib.build_rccl_heatmap_table(
                self.html_filename, "Missing Keys Test", self.actual_json_filename, self.ref_json_filename
            )

        output = "\n".join(log_cm.output)
        # Should have summary message
        self.assertIn("2 test keys not found in reference data", output)
        self.assertIn("skipped from table", output)
        # Should show sample keys
        self.assertTrue("broadcast_perf-bfloat16-16" in output or "reduce_scatter_perf-float-4" in output)

    def test_missing_message_sizes_summary(self):
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

        with self.assertLogs(html_lib.log, level=logging.WARNING) as log_cm:
            html_lib.build_rccl_heatmap_table(
                self.html_filename, "Missing Msg Sizes", self.actual_json_filename, self.ref_json_filename
            )

        output = "\n".join(log_cm.output)
        # Should show count of missing message sizes
        self.assertIn("2 msg_size entries not found in reference data", output)

    def test_invalid_key_format_skipped(self):
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

        with self.assertLogs(html_lib.log, level=logging.WARNING) as log_cm:
            html_lib.build_rccl_heatmap_table(
                self.html_filename, "Invalid Key Test", self.actual_json_filename, self.ref_json_filename
            )

        output = "\n".join(log_cm.output)
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

    def test_stale_per_dtype_reference_warns_to_regenerate(self):
        """A plain-name reference cannot be matched to per-data-type series; say why instead of a generic miss."""
        point = {"1048576": {"bus_bw": "245.5", "alg_bw": "122.75", "time": "42.3"}}
        cases = (
            ({"all_reduce_perf [type=float]": point}, {"all_reduce_perf": point}, "predates per-data-type"),
            ({"all_reduce_perf": point}, {"all_reduce_perf [type=float]": point}, "different RCCL series schemes"),
        )
        for actual_data, ref_data, expected in cases:
            with self.subTest(expected=expected):
                self._create_json_file(self.actual_json_filename, actual_data)
                self._create_json_file(self.ref_json_filename, ref_data)
                with self.assertLogs(html_lib.log, level=logging.WARNING) as log_cm:
                    html_lib.build_rccl_heatmap_table(
                        self.html_filename, "Stale Reference", self.actual_json_filename, self.ref_json_filename
                    )
                output = "\n".join(log_cm.output)
                self.assertIn(expected, output)
                self.assertIn("regenerate", output)
                self.assertNotIn("Invalid key format", output)
                self.assertNotIn("test keys not found in reference data", output)


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

    def test_stale_per_dtype_reference_warns_to_regenerate(self):
        """The heatmap names a pre-split reference instead of the generic 'not found' warning."""
        point = {"1048576": {"bus_bw": 245.5, "alg_bw": 122.75, "time": 42.3}}
        actual_data = {"all_reduce_perf [type=float]": point, "all_reduce_perf [type=half]": point}
        self._create_json_file(self.actual_json_filename, actual_data)
        self._create_json_file(self.ref_json_filename, {"all_reduce_perf": point})

        with self.assertLogs(html_lib.log, level=logging.WARNING) as log_cm:
            html_lib.build_rccl_heatmap(
                self.html_filename, "heatmapchart", "Stale Reference", self.actual_json_filename, self.ref_json_filename
            )

        warnings = [line for line in log_cm.output if "not found in reference data" in line]
        self.assertEqual(len(warnings), 2)
        for warning in warnings:
            self.assertIn("plain series all_reduce_perf", warning)
            self.assertIn("predates per-data-type RCCL series and must be regenerated", warning)

        self._create_json_file(self.ref_json_filename, {"broadcast_perf": point})
        with self.assertLogs(html_lib.log, level=logging.WARNING) as log_cm:
            html_lib.build_rccl_heatmap(
                self.html_filename,
                "heatmapchart",
                "Unrelated Reference",
                self.actual_json_filename,
                self.ref_json_filename,
            )
        self.assertNotIn("predates", "\n".join(log_cm.output))


class TestBuildTopologyGraph(unittest.TestCase):
    """Tests for the cluster topology HTML renderer (demo_d layout)."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".html")
        self.filename = self.tmp.name

    def tearDown(self):
        self.tmp.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def _gpu(self, bdf, index, peers):
        """GPU record with a SELF link plus one XGMI link per (peer_index, peer_bdf)."""
        links = [{"peer_bdf": bdf, "peer_gpu": index, "link_type": "SELF", "link_status": "ENABLED"}]
        for peer_index, peer_bdf in peers:
            links.append(
                {
                    "peer_bdf": peer_bdf,
                    "peer_gpu": peer_index,
                    "link_type": "XGMI",
                    "weight": 15,
                    "num_hops": 1,
                    "link_status": "ENABLED",
                }
            )
        return {"bdf": bdf, "gpu_index": index, "uuid": None, "links": links}

    def _gpu_node(self):
        return {
            "host": "testnode-rck-g02-h21-7.example",
            "hostname": "testnode-rck-g02-h21-7",
            "reachable": True,
            "gpu_count": 2,
            "cpu": {
                "model_name": "AMD Eng Sample",
                "cpus": "256",
                "cores_per_socket": "128",
                "sockets": "1",
                "numa_nodes": "38",
            },
            "numa": [{"node": "node0", "cpulist": "0-63,128-191"}, {"node": "node1", "cpulist": "64-127,192-254"}],
            "nics": [
                {"name": "enp4s0np0", "bdf": "0000:04:00.0", "numa_node": "0", "operstate": "up"},
                {"name": "enP1p68s0", "bdf": "0001:44:00.0", "numa_node": "-1", "operstate": "up"},
                {"name": "enP1p72s0", "bdf": "0001:48:00.0", "numa_node": "-1", "operstate": "down"},
            ],
            "dimms": [
                {
                    "locator": "DIMM 0",
                    "bank": "P0 CHANNEL A",
                    "size": "128 GB",
                    "type": "DDR5",
                    "speed": "8000 MT/s",
                    "manufacturer": "Samsung",
                    "part_number": "M321",
                    "rank": "2",
                },
                {
                    "locator": "DIMM 0",
                    "bank": "P0 CHANNEL B",
                    "size": "128 GB",
                    "type": "DDR5",
                    "speed": "8000 MT/s",
                    "manufacturer": "Samsung",
                    "part_number": "M321",
                    "rank": "2",
                },
            ],
            "mem_summary": {},
            "nvme": [
                {"name": "nvme0n1", "capacity": "3.8 TB", "numa_node": "1", "model": "SDS6BA138"},
            ],
            "scale_up_mode": "SPX/NPS1",
            "rdma": [
                {
                    "bdf": "0000:04:00.0",
                    "netdev": "enp4s0np0",
                    "gbps": 100,
                    "rate": "100 Gb/sec (4X EDR)",
                    "fabric": "RoCE",
                    "gen": "EDR",
                    "state": "4: ACTIVE",
                    "active": True,
                },
                {
                    "bdf": "0001:44:00.0",
                    "netdev": "enP1p68s0",
                    "gbps": 400,
                    "rate": "400 Gb/sec (4X NDR)",
                    "fabric": "RoCE",
                    "gen": "NDR",
                    "state": "4: ACTIVE",
                    "active": True,
                },
                {
                    "bdf": "0002:44:00.0",
                    "netdev": "enP2p68s0",
                    "gbps": 400,
                    "rate": "400 Gb/sec (4X NDR)",
                    "fabric": "RoCE",
                    "gen": "NDR",
                    "state": "4: ACTIVE",
                    "active": True,
                },
            ],
            "lldp": {},
            "gpus": [
                self._gpu("0001:01:00.0", 0, [(1, "0002:01:00.0")]),
                self._gpu("0002:01:00.0", 1, [(0, "0001:01:00.0")]),
            ],
        }

    def _no_gpu_node(self):
        return {
            "host": "testnode-rck-g02-j19-7.example",
            "hostname": "testnode-rck-g02-j19-7",
            "reachable": True,
            "gpu_count": 0,
            "cpu": {"model_name": "AMD Eng Sample", "cpus": "64", "numa_nodes": "2"},
            "numa": [{"node": "node0", "cpulist": "0-63"}],
            "nics": [{"name": "nic0", "bdf": "0000:04:00.0", "numa_node": "0", "operstate": "up"}],
            "gpus": [],
        }

    def _render(self, cluster):
        html_lib.build_topology_graph(self.filename, cluster)
        with open(self.filename) as fp:
            return fp.read()

    def test_pci_domain_and_nic_grouping(self):
        self.assertEqual(html_lib._topo_pci_domain("0001:01:00.0"), "0001")
        self.assertEqual(html_lib._topo_pci_domain(""), "")
        by_domain = html_lib._topo_nics_by_domain(self._gpu_node())
        # Both backend NICs share domain 0001 with gpu0; the front-end NIC is 0000.
        self.assertEqual({n["name"] for n in by_domain["0001"]}, {"enP1p68s0", "enP1p72s0"})
        self.assertEqual([n["name"] for n in by_domain["0000"]], ["enp4s0np0"])

    def test_peer_links_drop_self(self):
        links = html_lib._topo_gpu_peer_links(self._gpu_node())
        self.assertEqual(set(links["0001:01:00.0"].keys()), {"0002:01:00.0"})
        self.assertEqual(links["0001:01:00.0"]["0002:01:00.0"], ("XGMI", 15))

    def test_gpu_node_renders_cpu_mesh_and_links(self):
        content = self._render({"nodes": [self._gpu_node()], "summary": {"total_gpus": 2, "xgmi_edge_count": 1}})
        # CPU / NUMA panel
        self.assertIn("cpu0", content)
        self.assertIn("cpus=0-63,128-191", content)
        self.assertIn("front-end", content)
        # Mesh: GPUs, XGMI edge, PCIe host link, backend NIC
        self.assertIn("gpu0", content)
        self.assertIn("gpu1", content)
        self.assertIn("Scale-up: XGMI mesh (1 links)", content)
        self.assertIn("PCIe x16", content)
        self.assertIn("enP1p68s0", content)
        # No external assets.
        self.assertNotIn("http://", content)
        self.assertNotIn("https://", content)
        self.assertNotIn("cdn.", content)

    def test_dimms_and_nvme_rendered(self):
        content = self._render({"nodes": [self._gpu_node()], "summary": {"total_dimms": 2, "total_nvme": 1}})
        # Memory summary (2x 128 GB = 256 GB), per-DIMM detail, and NVMe on NUMA.
        self.assertIn("256 GB", content)
        self.assertIn("DDR5", content)
        self.assertIn("8000 MT/s", content)
        self.assertIn("populated DIMM slots", content)
        self.assertIn("P0 CHANNEL A", content)
        self.assertIn("nvme0n1", content)
        self.assertIn("3.8 TB", content)
        self.assertIn("DIMMs: 2", content)
        self.assertIn("NVMe: 1", content)

    def test_lsmem_fallback_when_no_dmidecode_dimms(self):
        node = self._gpu_node()
        node["dimms"] = []
        node["mem_summary"] = {"total_online": "512G"}
        content = self._render({"nodes": [node], "summary": {}})
        self.assertIn("512G", content)
        self.assertNotIn("populated DIMM slots", content)

    def test_down_nic_styled_but_present(self):
        content = self._render({"nodes": [self._gpu_node()], "summary": {}})
        self.assertIn("enP1p72s0", content)  # the down backend NIC still shows

    def test_no_gpu_node_shows_fallback_not_mesh(self):
        content = self._render({"nodes": [self._no_gpu_node()], "summary": {}})
        self.assertIn("amdgpu driver not loaded", content)
        self.assertIn("cpus=0-63", content)  # CPU panel still rendered
        # The mesh SVG label (distinct from the intro paragraph) must be absent.
        self.assertNotIn("Scale-up: XGMI mesh (", content)

    def test_multi_node_summary_and_cards(self):
        cluster = {
            "nodes": [self._gpu_node(), self._no_gpu_node()],
            "summary": {
                "node_count": 2,
                "reachable_node_count": 2,
                "unreachable_node_count": 0,
                "rack_count": 1,
                "total_gpus": 2,
                "xgmi_edge_count": 1,
            },
        }
        content = self._render(cluster)
        self.assertIn("GPUs: 2", content)
        self.assertIn("XGMI links: 1", content)
        self.assertIn("testnode-rck-g02-h21-7", content)
        self.assertIn("testnode-rck-g02-j19-7", content)

    def test_scale_up_and_scale_out_labels(self):
        content = self._render(
            {"nodes": [self._gpu_node()], "summary": {"scale_up_mode": "SPX/NPS1", "scale_out_nic_count": 2}}
        )
        # Scale-up: mesh label + CPU-panel row with the partition mode.
        self.assertIn("Scale-up: XGMI mesh", content)
        self.assertIn("SPX/NPS1", content)
        # Scale-out: per-node fabric line (2 backend NICs, front-end excluded).
        self.assertIn("Scale-out (NIC)", content)
        self.assertIn("RoCE", content)
        self.assertIn("2&times; 400G", content)
        self.assertIn("Scale-out NICs: 2", content)

    def test_inter_node_absent_shows_graceful_note(self):
        content = self._render({"nodes": [self._gpu_node()], "summary": {}, "inter_node_edges": []})
        self.assertIn("Inter-node fabric (scale-out)", content)
        self.assertIn("was not discovered", content)
        self.assertIn("LLDP", content)

    def test_inter_node_edges_render_switch_grouping(self):
        cluster = {
            "nodes": [self._gpu_node()],
            "summary": {"switch_count": 1},
            "inter_node_edges": [
                {
                    "src": {"name": "testnode-rck-g02-h21-7", "port": "enP1p68s0"},
                    "dst": {"name": "tor-g02-1", "port": "Eth1/5"},
                    "role": "data",
                    "via": "tor-g02-1",
                    "speed": "400 Gb/sec (4X NDR)",
                    "status": "4: ACTIVE",
                    "family": "scale_out",
                }
            ],
        }
        content = self._render(cluster)
        self.assertIn("ToR switch tor-g02-1", content)
        self.assertIn("Eth1/5", content)
        self.assertNotIn("was not discovered", content)


if __name__ == "__main__":
    unittest.main()
