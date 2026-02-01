'''
Unit tests for test_utils.py using unittest (not pytest)
This test reproduces the infinite recursion bug where meta-test discovers itself.
'''

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock


class TestCVSTestSuiteRunnerDiscovery(unittest.TestCase):
    """Test CVSTestSuiteRunner discovery."""

    def test_discover_excludes_meta_test_itself(self):
        """Test that the meta-test file itself is excluded from discovery to prevent infinite recursion."""
        # Import here to use the source code
        from cvs.lib.test_utils import CVSTestSuiteRunner

        # Create temp directory with test files
        with tempfile.TemporaryDirectory() as tmp_path:
            # Create test files including the meta-test itself
            meta_test_file = os.path.join(tmp_path, "singlenode_all_models_vllm.py")
            test_a = os.path.join(tmp_path, "singlenode_qwen3_80b_vllm.py")
            test_b = os.path.join(tmp_path, "singlenode_deepseek31_vllm.py")

            with open(meta_test_file, 'w') as f:
                f.write("# meta test")
            with open(test_a, 'w') as f:
                f.write("# test a")
            with open(test_b, 'w') as f:
                f.write("# test b")

            mock_log = Mock()
            mock_config = Mock()
            mock_config.getoption.return_value = None

            with patch('sys.argv', ['singlenode_all_models_vllm']):
                with patch('inspect.currentframe') as mock_frame:
                    frame_mock = MagicMock()
                    frame_mock.f_back.f_code.co_filename = meta_test_file
                    mock_frame.return_value = frame_mock

                    runner = CVSTestSuiteRunner(mock_log, mock_config, test_pattern='singlenode_*_vllm.py')
                    runner.test_dir = tmp_path
                    modules = runner.discover_test_modules(pattern='singlenode_*_vllm.py')

            # Should find 2 suites, NOT including the meta-test itself
            self.assertEqual(len(modules), 2, f"Expected 2 modules but found {len(modules)}: {list(modules.keys())}")
            self.assertIn('singlenode_qwen3_80b_vllm', modules)
            self.assertIn('singlenode_deepseek31_vllm', modules)
            self.assertNotIn(
                'singlenode_all_models_vllm', modules, "Meta-test should be excluded to prevent infinite recursion!"
            )


class TestCVSTestSuiteRunnerInitialization(unittest.TestCase):
    """Test CVSTestSuiteRunner initialization."""

    def test_initialization_detects_exclude_patterns(self):
        """Test that __init__ auto-detects calling file to exclude."""
        from cvs.lib.test_utils import CVSTestSuiteRunner
        
        mock_log = Mock()
        mock_config = Mock()
        mock_config.getoption.return_value = None

        with patch('sys.argv', ['runner.py']):
            with patch('inspect.currentframe') as mock_frame:
                frame_mock = MagicMock()
                frame_mock.f_back.f_code.co_filename = '/path/to/my_meta_test.py'
                mock_frame.return_value = frame_mock

                runner = CVSTestSuiteRunner(mock_log, mock_config)

        self.assertEqual(runner.exclude_patterns, ['my_meta_test', 'run'])

    def test_initialization_sets_default_exclude_flags(self):
        """Test that default exclude flags are set."""
        from cvs.lib.test_utils import CVSTestSuiteRunner
        
        mock_log = Mock()
        mock_config = Mock()
        mock_config.getoption.return_value = None

        with patch('sys.argv', ['test.py']):
            with patch('inspect.currentframe') as mock_frame:
                frame_mock = MagicMock()
                frame_mock.f_back.f_code.co_filename = '/tmp/test.py'
                mock_frame.return_value = frame_mock

                runner = CVSTestSuiteRunner(mock_log, mock_config)

        self.assertIn('-k', runner.exclude_flags)
        self.assertIn('--setup-only', runner.exclude_flags)
        # --collect-only should NOT be excluded (for cvs list support)
        self.assertNotIn('--collect-only', runner.exclude_flags)


class TestCVSTestSuiteRunnerArguments(unittest.TestCase):
    """Test CLI argument preparation and filtering."""

    def setUp(self):
        """Set up common test fixtures."""
        self.tmp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.tmp_dir, "meta_test.py")
        with open(self.test_file, 'w') as f:
            f.write("# meta test")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_exclude_patterns_filters_matching_args(self):
        """Test that arguments matching exclude patterns are filtered out."""
        from cvs.lib.test_utils import CVSTestSuiteRunner
        
        mock_log = Mock()
        mock_config = Mock()
        mock_config.getoption.return_value = None

        with patch('sys.argv', ['meta_test.py', '--html', 'report.html', 'meta_test', '--other']):
            with patch('inspect.currentframe') as mock_frame:
                frame_mock = MagicMock()
                frame_mock.f_back.f_code.co_filename = self.test_file
                mock_frame.return_value = frame_mock

                runner = CVSTestSuiteRunner(mock_log, mock_config, test_pattern='*.py')
                args = runner.prepare_suite_arguments('test_suite')

        self.assertNotIn('meta_test', args)
        self.assertIn('--html', args)
        self.assertIn('--other', args)

    def test_exclude_flags_removes_flag_and_value(self):
        """Test that excluded flags and their values are removed."""
        from cvs.lib.test_utils import CVSTestSuiteRunner
        
        mock_log = Mock()
        mock_config = Mock()
        mock_config.getoption.return_value = None

        with patch('sys.argv', ['test.py', '-k', 'test_name', '--html', 'report.html']):
            with patch('inspect.currentframe') as mock_frame:
                frame_mock = MagicMock()
                frame_mock.f_back.f_code.co_filename = self.test_file
                mock_frame.return_value = frame_mock

                runner = CVSTestSuiteRunner(mock_log, mock_config)
                args = runner.prepare_suite_arguments('suite')

        self.assertNotIn('-k', args)
        self.assertNotIn('test_name', args)
        self.assertIn('--html', args)


if __name__ == '__main__':
    unittest.main()
