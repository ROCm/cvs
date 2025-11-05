import unittest
import tempfile
import os

# Import the module under test
import lib.html_lib as html_lib

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
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8')
        self.filename = self.tmp_file.name

    def tearDown(self):
        self.tmp_file.close()
        os.remove(self.filename)

    def test_single_node_valid_input(self):
        use_dict = {
            "node1": {
                **{f"card{i}": {
                    "GPU Memory Allocated (VRAM%)": f"{i*10}%",
                    "GPU Memory Read/Write Activity (%)": f"{i*5}%",
                    "Memory Activity": f"{i*3}%",
                    "Avg. Memory Bandwidth": f"{i*2} GB/s"
                } for i in range(8)}
            }
        }

        amd_dict = {
            "node1": [
                {
                    "mem_usage": {
                        "total_vram": {"value": "16384"},
                        "used_vram": {"value": "8192"},
                        "free_vram": {"value": "8192"}
                    }
                } for _ in range(8)
            ]
        }

        html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)
        with open(self.filename, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("GPU Memory Utilization", content)
            self.assertIn("G0 Tot VRAM MB", content)
            self.assertIn("node1", content)
            self.assertIn("8192", content)
            self.assertIn("10%", content)

    def test_multiple_nodes(self):
        use_dict = {
            f"node{i}": {
                **{f"card{j}": {
                    "GPU Memory Allocated (VRAM%)": f"{j*10}%",
                    "GPU Memory Read/Write Activity (%)": f"{j*5}%",
                    "Memory Activity": f"{j*3}%",
                    "Avg. Memory Bandwidth": f"{j*2} GB/s"
                } for j in range(8)}
            } for i in range(2)
        }

        amd_dict = {
            f"node{i}": [
                {
                    "mem_usage": {
                        "total_vram": {"value": "16384"},
                        "used_vram": {"value": "8192"},
                        "free_vram": {"value": "8192"}
                    }
                } for _ in range(8)
            ] for i in range(2)
        }

        html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)
        with open(self.filename, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("node0", content)
            self.assertIn("node1", content)

    def test_rocm7_style_gpu_data(self):
        use_dict = {
            "node1": {
                **{f"card{i}": {
                    "GPU Memory Allocated (VRAM%)": f"{i*10}%",
                    "GPU Memory Read/Write Activity (%)": f"{i*5}%",
                    "Memory Activity": f"{i*3}%",
                    "Avg. Memory Bandwidth": f"{i*2} GB/s"
                } for i in range(8)}
            }
        }

        amd_dict = {
            "node1": {
                "gpu_data": [
                    {
                        "mem_usage": {
                            "total_vram": {"value": "16384"},
                            "used_vram": {"value": "8192"},
                            "free_vram": {"value": "8192"}
                        }
                    } for _ in range(8)
                ]
            }
        }

        html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)
        with open(self.filename, 'r', encoding='utf-8') as f:
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
                    "Avg. Memory Bandwidth": "40 GB/s"
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
                        "free_vram": {"value": "8192"}
                    }
                } for _ in range(8)
            ]
        }

        with self.assertRaises(KeyError):
            html_lib.build_html_mem_utilization_table(self.filename, use_dict, amd_dict)


if __name__ == '__main__':
    unittest.main()