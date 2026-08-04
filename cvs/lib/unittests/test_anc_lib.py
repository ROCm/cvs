# cvs/lib/unittests/test_anc_lib.py
import os
import unittest
from unittest.mock import patch

import cvs.lib.anc_lib as anc_lib


class TestResolveAncInstallPrefix(unittest.TestCase):
    '''resolve_anc_install_prefix: tar honours ANC_INSTALL_PATH; deb/rpm ignore it.'''

    def test_tar_with_custom_path(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc"}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), "/home/u/my/anc")

    def test_tar_blank_falls_back_to_default(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": ""}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), anc_lib.ANC_TOOLS_PREFIX)

    def test_tar_whitespace_only_falls_back(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "   "}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), anc_lib.ANC_TOOLS_PREFIX)

    def test_tar_missing_key_falls_back(self):
        cfg = {"anc": {}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), anc_lib.ANC_TOOLS_PREFIX)

    def test_deb_ignores_custom_path(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc"}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "deb"), anc_lib.ANC_TOOLS_PREFIX)

    def test_rpm_ignores_custom_path(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc"}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "rpm"), anc_lib.ANC_TOOLS_PREFIX)

    def test_tilde_is_expanded(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "~/my/anc"}}
        expected = os.path.abspath(os.path.expanduser("~/my/anc"))
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), expected)

    def test_trailing_slash_normalised(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc/"}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), "/home/u/my/anc")


class TestResolveAncPaths(unittest.TestCase):
    '''resolve_anc_paths: derives anc_dir/anc_bin under the resolved prefix.'''

    def test_relocated_tar_paths(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc"}}
        paths = anc_lib.resolve_anc_paths(cfg, "tar")
        self.assertEqual(paths.prefix, "/home/u/my/anc")
        self.assertEqual(paths.anc_dir, "/home/u/my/anc/anc")
        self.assertEqual(paths.anc_bin, "/home/u/my/anc/anc/anc.py")

    def test_default_tar_paths_match_module_constants(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": ""}}
        paths = anc_lib.resolve_anc_paths(cfg, "tar")
        self.assertEqual(paths.prefix, anc_lib.ANC_TOOLS_PREFIX)
        self.assertEqual(paths.anc_dir, anc_lib.ANC_DIR)
        self.assertEqual(paths.anc_bin, anc_lib.ANC_BIN)

    def test_deb_paths_are_default(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc"}}
        paths = anc_lib.resolve_anc_paths(cfg, "deb")
        self.assertEqual(paths.anc_bin, anc_lib.ANC_BIN)


class TestResolveAncPathsFromConfig(unittest.TestCase):
    '''resolve_anc_paths_from_config: pkg flavour inferred from the release URL.'''

    def test_tar_url_relocated(self):
        cfg = {
            "anc": {
                "anc_release_url": "http://x/anc-release-1.4.9-tar-linux-x64.tar.gz",
                "ANC_INSTALL_PATH": "/home/u/my/anc",
            }
        }
        self.assertEqual(anc_lib.resolve_anc_paths_from_config(cfg).anc_bin, "/home/u/my/anc/anc/anc.py")

    def test_deb_url_ignores_install_path(self):
        cfg = {
            "anc": {
                "anc_release_url": "http://x/anc-release-1.4.9-deb-linux-x64.tar.gz",
                "ANC_INSTALL_PATH": "/home/u/my/anc",
            }
        }
        self.assertEqual(anc_lib.resolve_anc_paths_from_config(cfg).anc_bin, anc_lib.ANC_BIN)

    def test_missing_url_falls_back_to_default(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/my/anc"}}
        self.assertEqual(anc_lib.resolve_anc_paths_from_config(cfg).anc_bin, anc_lib.ANC_BIN)

    def test_unrecognised_url_falls_back_to_default(self):
        cfg = {"anc": {"anc_release_url": "http://x/mystery.bin", "ANC_INSTALL_PATH": "/home/u/my/anc"}}
        self.assertEqual(anc_lib.resolve_anc_paths_from_config(cfg).anc_bin, anc_lib.ANC_BIN)


class TestSudoPrefixSnippet(unittest.TestCase):
    '''_sudo_prefix_snippet: emits a shell probe that selects sudo by writability.'''

    def test_snippet_mentions_prefix_and_writability_branch(self):
        snippet = anc_lib._sudo_prefix_snippet("/home/u/my/anc")
        self.assertIn("/home/u/my/anc", snippet)
        self.assertIn("SUDO=''", snippet)
        self.assertIn("SUDO='sudo'", snippet)
        self.assertIn("-w", snippet)


class TestNodeVersionMatchesUsesAncBin(unittest.TestCase):
    '''node_version_matches queries the passed anc_bin path.'''

    def test_custom_anc_bin_in_command(self):
        class FakePhdl:
            def __init__(self):
                self.cmd = None

            def exec(self, cmd, timeout=None):
                self.cmd = cmd
                return {"node1": "version 1.4.9"}

        phdl = FakePhdl()
        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(phdl, "1.4.9", anc_bin="/home/u/my/anc/anc/anc.py")
        self.assertIn("/home/u/my/anc/anc/anc.py --version", phdl.cmd)
        self.assertTrue(result["node1"])


if __name__ == "__main__":
    unittest.main()
