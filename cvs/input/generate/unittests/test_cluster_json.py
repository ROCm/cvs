import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to sys.path to import generate modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cvs.input.generate.cluster_json import ClusterJsonGenerator


class TestClusterJsonGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ClusterJsonGenerator()

    def test_get_name(self):
        """Test generator name"""
        self.assertEqual(self.generator.get_name(), "cluster_json")

    def test_get_description(self):
        """Test generator description"""
        self.assertEqual(self.generator.get_description(), "Generate cluster JSON configuration file from hosts list")

    def test_get_parser(self):
        """Test parser creation"""
        parser = self.generator.get_parser()

        # Check required arguments
        self.assertTrue(parser.get_default('input_hosts_file') is None)  # Should be required
        self.assertTrue(parser.get_default('output_json_file') is None)  # Should be required
        self.assertTrue(parser.get_default('username') is None)  # Should be required
        self.assertTrue(parser.get_default('key_file') is None)  # Should be required
        self.assertIsNone(parser.get_default('head_node'))  # Optional

    def test_expand_ip_range_single_ip(self):
        """Test expanding single IP (no range)"""
        result = self.generator.expand_ip_range("192.168.1.10")
        self.assertEqual(result, ["192.168.1.10"])

    def test_expand_ip_range_valid_range(self):
        """Test expanding valid IP range"""
        result = self.generator.expand_ip_range("192.168.1.10-15")
        expected = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14", "192.168.1.15"]
        self.assertEqual(result, expected)

    def test_expand_ip_range_invalid_range(self):
        """Test expanding invalid IP range returns original"""
        result = self.generator.expand_ip_range("192.168.1.10-invalid")
        self.assertEqual(result, ["192.168.1.10-invalid"])

    def test_expand_ip_range_malformed_ip(self):
        """Test expanding malformed IP returns original"""
        result = self.generator.expand_ip_range("invalid.ip.range")
        self.assertEqual(result, ["invalid.ip.range"])

    @patch('builtins.open')
    def test_parse_hosts_file(self, mock_open):
        """Test parsing hosts file with ranges"""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = ["192.168.1.10", "# comment", "", "192.168.1.11-13", "192.168.2.1"]
        mock_open.return_value = mock_file

        result = self.generator.parse_hosts_file("/fake/hosts.txt")

        expected = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.2.1"]
        self.assertEqual(result, expected)

    def test_determine_head_node_specified(self):
        """Test head node determination when specified"""
        node_list = ["192.168.1.10", "192.168.1.11", "192.168.1.12"]
        head_node = self.generator.determine_head_node(node_list, "192.168.1.11")

        self.assertEqual(head_node, "192.168.1.11")
        # Head node should be first in list
        self.assertEqual(node_list[0], "192.168.1.11")

    def test_determine_head_node_not_in_list(self):
        """Test head node determination when specified head not in list"""
        node_list = ["192.168.1.10", "192.168.1.11"]
        head_node = self.generator.determine_head_node(node_list, "192.168.1.12")

        # Should return head node but not add it to the list
        self.assertEqual(head_node, "192.168.1.12")
        self.assertNotIn("192.168.1.12", node_list)
        # Node list should remain unchanged
        self.assertEqual(node_list, ["192.168.1.10", "192.168.1.11"])

    def test_determine_head_node_default(self):
        """Test head node determination using default (first in list)"""
        node_list = ["192.168.1.10", "192.168.1.11", "192.168.1.12"]
        head_node = self.generator.determine_head_node(node_list, None)

        self.assertEqual(head_node, "192.168.1.10")

    @patch('cvs.input.generate.cluster_json.resources.files')
    @patch('builtins.open')
    @patch('cvs.input.generate.cluster_json.Template')
    def test_generate_success(self, mock_template_class, mock_open, mock_resources_files):
        """Test successful generation"""
        # Mock template
        mock_template = MagicMock()
        mock_template.render.return_value = '{"test": "json"}'
        mock_template_class.return_value = mock_template

        # Mock resources
        mock_files = MagicMock()
        mock_text = MagicMock()
        mock_text.read_text.return_value = "template content"
        mock_files.joinpath.return_value = mock_text
        mock_resources_files.return_value = mock_files

        # Mock file writing
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Create args
        args = MagicMock()
        args.input_hosts_file = "/fake/hosts.txt"
        args.output_json_file = "/fake/output.json"
        args.username = "testuser"
        args.key_file = "/fake/key"
        args.port = "22"
        args.head_node = "192.168.1.10"

        with patch.object(self.generator, 'parse_hosts_file', return_value=["192.168.1.10", "192.168.1.11"]):
            with patch.object(self.generator, 'determine_head_node', return_value="192.168.1.10"):
                with patch('builtins.print') as mock_print:
                    self.generator.generate(args)

                    # Verify template was rendered with correct data
                    mock_template.render.assert_called_once_with(
                        username="testuser",
                        priv_key_file="/fake/key",
                        port="22",
                        head_node_ip="192.168.1.10",
                        node_list=["192.168.1.10", "192.168.1.11"],
                    )

                    # Verify file was written
                    mock_file.write.assert_called_once_with('{"test": "json"}')

                    # Verify success messages
                    success_calls = [
                        call for call in mock_print.call_args_list if "Generated cluster JSON file" in str(call)
                    ]
                    self.assertTrue(len(success_calls) > 0)

    @patch('builtins.open', side_effect=FileNotFoundError("Hosts file not found"))
    def test_generate_hosts_file_error(self, mock_open):
        """Test error when hosts file not found"""
        args = MagicMock()
        args.input_hosts_file = "/nonexistent/hosts.txt"

        with self.assertRaises(FileNotFoundError):
            self.generator.parse_hosts_file("/nonexistent/hosts.txt")

    @patch('cvs.input.generate.cluster_json.resources.files', side_effect=Exception("Resource error"))
    def test_generate_resource_error(self, mock_resources_files):
        """Test error when resources cannot be loaded"""
        args = MagicMock()
        args.input_hosts_file = "/fake/hosts.txt"
        args.output_json_file = "/fake/output.json"
        args.username = "testuser"
        args.key_file = "/fake/key"

        with patch.object(self.generator, 'parse_hosts_file', return_value=["192.168.1.10"]):
            with self.assertRaises(Exception):
                self.generator.generate(args)


if __name__ == '__main__':
    unittest.main()
