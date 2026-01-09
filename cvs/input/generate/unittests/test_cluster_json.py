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

    def test_expand_hostname_bracket_range_single_hostname(self):
        """Test expanding single hostname (no range)"""
        result = self.generator.expand_hostname_bracket_range("server01")
        self.assertEqual(result, ["server01"])

    def test_expand_hostname_bracket_range_valid_range(self):
        """Test expanding valid hostname bracket range"""
        result = self.generator.expand_hostname_bracket_range("mia1-p01-g[24-30]")
        expected = [
            "mia1-p01-g24",
            "mia1-p01-g25",
            "mia1-p01-g26",
            "mia1-p01-g27",
            "mia1-p01-g28",
            "mia1-p01-g29",
            "mia1-p01-g30",
        ]
        self.assertEqual(result, expected)

    def test_expand_hostname_bracket_range_with_padding(self):
        """Test expanding hostname bracket range with leading zeros"""
        result = self.generator.expand_hostname_bracket_range("server[01-05]")
        expected = ["server01", "server02", "server03", "server04", "server05"]
        self.assertEqual(result, expected)

    def test_expand_hostname_bracket_range_with_suffix(self):
        """Test expanding hostname bracket range with suffix"""
        result = self.generator.expand_hostname_bracket_range("node[1-3].example.com")
        expected = ["node1.example.com", "node2.example.com", "node3.example.com"]
        self.assertEqual(result, expected)

    def test_expand_range_hostname_bracket(self):
        """Test unified expand_range with hostname bracket notation"""
        result = self.generator.expand_range("server[10-12]")
        expected = ["server10", "server11", "server12"]
        self.assertEqual(result, expected)

    def test_expand_range_ip_range(self):
        """Test unified expand_range with IP range"""
        result = self.generator.expand_range("192.168.1.10-12")
        expected = ["192.168.1.10", "192.168.1.11", "192.168.1.12"]
        self.assertEqual(result, expected)

    def test_expand_range_no_range(self):
        """Test unified expand_range with no range notation"""
        result = self.generator.expand_range("simple-hostname")
        self.assertEqual(result, ["simple-hostname"])

    def test_parse_hosts_list_simple(self):
        """Test parsing simple comma-separated hosts"""
        result = self.generator.parse_hosts_list("host1,host2,host3")
        self.assertEqual(result, ["host1", "host2", "host3"])

    def test_parse_hosts_list_with_ip_ranges(self):
        """Test parsing hosts list with IP ranges"""
        result = self.generator.parse_hosts_list("192.168.1.10-12,192.168.2.1")
        expected = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.2.1"]
        self.assertEqual(result, expected)

    def test_parse_hosts_list_with_bracket_ranges(self):
        """Test parsing hosts list with hostname bracket ranges"""
        result = self.generator.parse_hosts_list("server[1-3],host5")
        expected = ["server1", "server2", "server3", "host5"]
        self.assertEqual(result, expected)

    def test_parse_hosts_list_mixed_ranges(self):
        """Test parsing hosts list with mixed range types"""
        result = self.generator.parse_hosts_list("mia1-p01-g20,mia1-p01-g[24-26],192.168.1.10-11")
        expected = ["mia1-p01-g20", "mia1-p01-g24", "mia1-p01-g25", "mia1-p01-g26", "192.168.1.10", "192.168.1.11"]
        self.assertEqual(result, expected)

    def test_parse_hosts_list_with_whitespace(self):
        """Test parsing hosts list with whitespace"""
        result = self.generator.parse_hosts_list("host1 , host2 , host3")
        self.assertEqual(result, ["host1", "host2", "host3"])

    def test_parse_hosts_list_empty_entries(self):
        """Test parsing hosts list with empty entries"""
        result = self.generator.parse_hosts_list("host1,,host2")
        self.assertEqual(result, ["host1", "host2"])

    @patch('builtins.open')
    def test_parse_hosts_file(self, mock_open):
        """Test parsing hosts file with ranges"""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = [
            "192.168.1.10",
            "# comment",
            "",
            "192.168.1.11-13",
            "server[1-2]",
            "192.168.2.1",
        ]
        mock_open.return_value = mock_file

        result = self.generator.parse_hosts_file("/fake/hosts.txt")

        expected = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13", "server1", "server2", "192.168.2.1"]
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
        args.hosts = None
        args.output_json_file = "/fake/output.json"
        args.username = "testuser"
        args.key_file = "/fake/key"
        args.head_node = "192.168.1.10"

        with patch.object(self.generator, 'parse_hosts_file', return_value=["192.168.1.10", "192.168.1.11"]):
            with patch.object(self.generator, 'determine_head_node', return_value="192.168.1.10"):
                with patch('builtins.print') as mock_print:
                    self.generator.generate(args)

                    # Verify template was rendered with correct data
                    mock_template.render.assert_called_once_with(
                        username="testuser",
                        priv_key_file="/fake/key",
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

    @patch('cvs.input.generate.cluster_json.resources.files')
    @patch('builtins.open')
    @patch('cvs.input.generate.cluster_json.Template')
    def test_generate_with_hosts_arg(self, mock_template_class, mock_open, mock_resources_files):
        """Test successful generation using --hosts argument"""
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

        # Create args using --hosts
        args = MagicMock()
        args.input_hosts_file = None
        args.hosts = "host1,host[2-3]"
        args.output_json_file = "/fake/output.json"
        args.username = "testuser"
        args.key_file = "/fake/key"
        args.head_node = None

        with patch.object(self.generator, 'parse_hosts_list', return_value=["host1", "host2", "host3"]):
            with patch.object(self.generator, 'determine_head_node', return_value="host1"):
                with patch('builtins.print'):
                    self.generator.generate(args)

                    # Verify template was rendered with correct data
                    mock_template.render.assert_called_once_with(
                        username="testuser",
                        priv_key_file="/fake/key",
                        head_node_ip="host1",
                        node_list=["host1", "host2", "host3"],
                    )

                    # Verify file was written
                    mock_file.write.assert_called_once_with('{"test": "json"}')

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
