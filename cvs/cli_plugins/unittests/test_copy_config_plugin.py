import unittest
from unittest.mock import patch
import os
import tempfile
import argparse
import io
from contextlib import redirect_stdout

from cvs.cli_plugins.copy_config_plugin import CopyConfigPlugin


class TestCopyConfigPlugin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Hardcode expected root directories using the cvs package path
        import cvs

        cvs_dir = os.path.dirname(cvs.__file__)
        cls.expected_roots = [
            os.path.join(cvs_dir, "input", "config_file"),
            os.path.join(cvs_dir, "input", "cluster_file"),
        ]

    def setUp(self):
        self.plugin = CopyConfigPlugin()

    def test_find_config_root(self):
        """Test finding config root directories"""
        roots = self.plugin._find_config_root()

        # Should match the expected roots
        self.assertEqual(sorted(roots), sorted(self.expected_roots))

    @patch("builtins.print")
    def test_run_list_mode(self, mock_print):
        """Test listing configs when no output specified"""
        args = argparse.Namespace()
        args.output = None
        args.list = True
        args.path = None
        args.all = False
        args.force = False

        self.plugin.run(args)

        # Should print configs
        mock_print.assert_called()

    def test_run_copy_all_success(self):
        """Test successful copy of all configs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace()
            args.output = temp_dir
            args.path = None
            args.all = True

            self.plugin.run(args)

            # Check that directories for all expected roots are created and contain files
            for expected_root in self.expected_roots:
                dir_name = os.path.basename(expected_root)
                expected_dir = os.path.join(temp_dir, dir_name)

                # Directory should exist
                self.assertTrue(os.path.exists(expected_dir), f"Directory {dir_name} should be created")
                self.assertTrue(os.path.isdir(expected_dir), f"{expected_dir} should be a directory")

                # Directory should contain files (recursively)
                files_in_dir = []
                for root, dirs, files in os.walk(expected_dir):
                    files_in_dir.extend(files)

                self.assertGreater(len(files_in_dir), 0, f"Directory {dir_name} should contain config files")
                print(f"Directory {dir_name} contains {len(files_in_dir)} files")

    def test_run_overwrite_behavior(self):
        """Test overwrite behavior with and without --force flag"""

        # Use a specific config file for testing
        test_config_path = "training/jax/mi300x_distributed_llama3_1_70b.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_config.json")

            # First, copy the config file to create an existing file
            args_initial = argparse.Namespace()
            args_initial.output = output_path
            args_initial.path = test_config_path
            args_initial.all = False
            args_initial.force = False
            args_initial.list = False

            # Copy the file initially
            self.plugin.run(args_initial)

            # Verify the file was created
            self.assertTrue(os.path.exists(output_path), "Initial file should be created")

            # Get the original file content
            with open(output_path, "r") as f:
                original_content = f.read()

            # Modify the file content to simulate a different file
            modified_content = '{"modified": true}'
            with open(output_path, "w") as f:
                f.write(modified_content)

            # Test 1: Try to copy without --force (should fail)
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                args_no_force = argparse.Namespace()
                args_no_force.output = output_path
                args_no_force.path = test_config_path
                args_no_force.all = False
                args_no_force.force = False
                args_no_force.list = False

                self.plugin.run(args_no_force)

            output_no_force = captured_output.getvalue()

            # Should print error about existing file
            self.assertIn("Error: File", output_no_force)
            self.assertIn("already exists", output_no_force)
            self.assertIn("--force", output_no_force)

            # File should still exist with modified content (not overwritten)
            self.assertTrue(os.path.exists(output_path), "File should still exist")
            with open(output_path, "r") as f:
                current_content = f.read()
            self.assertEqual(current_content, modified_content, "File content should be unchanged without --force")

            # Test 2: Try to copy with --force (should succeed)
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                args_with_force = argparse.Namespace()
                args_with_force.output = output_path
                args_with_force.path = test_config_path
                args_with_force.all = False
                args_with_force.force = True
                args_with_force.list = False

                self.plugin.run(args_with_force)

            output_with_force = captured_output.getvalue()

            # Should not print error about existing file
            self.assertNotIn("Error: File", output_with_force)
            self.assertNotIn("already exists", output_with_force)

            # File should exist with original content (overwritten)
            self.assertTrue(os.path.exists(output_path), "File should still exist")
            with open(output_path, "r") as f:
                final_content = f.read()
            self.assertEqual(final_content, original_content, "File content should be overwritten with --force")

    def test_run_single_file_overwrite_behavior(self):
        """Test overwrite behavior for single file copy with and without --force"""

        # Use a specific config file for testing
        test_config_path = "training/jax/mi300x_distributed_llama3_1_70b.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_config.json")

            # First, copy the config file to create an existing file
            args_initial = argparse.Namespace()
            args_initial.output = output_path
            args_initial.path = test_config_path
            args_initial.all = False
            args_initial.force = False
            args_initial.list = False

            # Copy the file initially
            self.plugin.run(args_initial)

            # Verify the file was created
            self.assertTrue(os.path.exists(output_path), "Initial file should be created")

            # Get the original file content
            with open(output_path, "r") as f:
                original_content = f.read()

            # Modify the file content to simulate a different file
            modified_content = '{"modified": true}'
            with open(output_path, "w") as f:
                f.write(modified_content)

            # Test 1: Try to copy without --force (should fail and not overwrite)
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                args_no_force = argparse.Namespace()
                args_no_force.output = output_path
                args_no_force.path = test_config_path
                args_no_force.all = False
                args_no_force.force = False
                args_no_force.list = False

                self.plugin.run(args_no_force)

            output_no_force = captured_output.getvalue()

            # Should print error about existing file
            self.assertIn("Error: File", output_no_force)
            self.assertIn("already exists", output_no_force)
            self.assertIn("--force", output_no_force)

            # File should still exist with modified content (not overwritten)
            self.assertTrue(os.path.exists(output_path), "File should still exist")
            with open(output_path, "r") as f:
                current_content = f.read()
            self.assertEqual(current_content, modified_content, "File content should be unchanged without --force")

            # Test 2: Try to copy with --force (should succeed and overwrite)
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                args_with_force = argparse.Namespace()
                args_with_force.output = output_path
                args_with_force.path = test_config_path
                args_with_force.all = False
                args_with_force.force = True
                args_with_force.list = False

                self.plugin.run(args_with_force)

            output_with_force = captured_output.getvalue()

            # Should not print error about existing file
            self.assertNotIn("Error: File", output_with_force)
            self.assertNotIn("already exists", output_with_force)

            # File should exist with original content (overwritten)
            self.assertTrue(os.path.exists(output_path), "File should still exist")
            with open(output_path, "r") as f:
                current_content = f.read()
            self.assertNotEqual(current_content, modified_content, "File content should be overwritten")
            self.assertEqual(current_content, original_content, "File content should match original config")


if __name__ == "__main__":
    unittest.main()
