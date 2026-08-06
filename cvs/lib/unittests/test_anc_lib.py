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


class TestDetectPackageFlavour(unittest.TestCase):
    '''detect_package_flavour: flavour + legacy/direct generation from the name.'''

    def test_legacy_tar_token(self):
        url = "http://x/anc-release-helios-nda-1.4.9-tar-linux-x64.tar.gz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("tar", False))

    def test_legacy_deb_token(self):
        url = "http://x/anc-release-helios-nda-1.4.9-deb-linux-x64.tar.gz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("deb", False))

    def test_legacy_rpm_token(self):
        url = "http://x/anc-release-helios-nda-1.4.9-rpm-linux-x64.tar.gz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("rpm", False))

    def test_legacy_dot_token_tar(self):
        # Trailing "-tar." before the extension is legacy, NOT a direct tar.
        url = "http://x/anc-release-helios-nda-1.4.9-tar.tar.gz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("tar", False))

    def test_legacy_dot_token_deb(self):
        url = "http://x/anc-release-helios-nda-1.4.9-deb.tar.gz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("deb", False))

    def test_direct_deb(self):
        url = "http://x/anc-release-helios-nda_1.5.5_amd64.deb"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("deb", True))

    def test_direct_rpm(self):
        url = "http://x/anc-release-helios-nda-1.5.5-1.x86_64.rpm"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("rpm", True))

    def test_direct_tar(self):
        url = "http://x/anc-release-helios-nda-1.5.5-x86_64.tar.gz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("tar", True))

    def test_direct_tgz(self):
        url = "http://x/anc-release-helios-nda-1.5.5-x86_64.tgz"
        self.assertEqual(anc_lib.detect_package_flavour(url), anc_lib.PackageFlavour("tar", True))

    def test_unrecognised_raises(self):
        with self.assertRaises(ValueError):
            anc_lib.detect_package_flavour("http://x/mystery.bin")

    def test_detect_package_type_wrapper_direct_tar(self):
        self.assertEqual(anc_lib.detect_package_type("http://x/anc-1.5.5-x86_64.tar.gz"), "tar")

    def test_detect_package_type_wrapper_legacy_deb(self):
        self.assertEqual(anc_lib.detect_package_type("http://x/anc-1.4.9-deb-linux-x64.tar.gz"), "deb")


class TestParseVersionFromUrl(unittest.TestCase):
    '''parse_version_from_url: first dotted-numeric run in the filename.'''

    def test_legacy_url(self):
        self.assertEqual(
            anc_lib.parse_version_from_url("http://x/anc-release-helios-nda-1.4.9-tar-linux-x64.tar.gz"),
            "1.4.9",
        )

    def test_direct_tar_url(self):
        self.assertEqual(
            anc_lib.parse_version_from_url("http://x/anc-release-helios-nda-1.5.5-x86_64.tar.gz"),
            "1.5.5",
        )

    def test_direct_deb_url(self):
        self.assertEqual(
            anc_lib.parse_version_from_url("http://x/anc-release-helios-nda_1.5.5_amd64.deb"),
            "1.5.5",
        )

    def test_no_version_returns_none(self):
        self.assertIsNone(anc_lib.parse_version_from_url("http://x/anc-release.tar.gz"))

    def test_empty_returns_none(self):
        self.assertIsNone(anc_lib.parse_version_from_url(""))


class TestCheckVersionMatchesUrl(unittest.TestCase):
    '''check_version_matches_url: abort when configured version != URL version.'''

    def test_match_returns_none(self):
        cfg = {"anc": {"anc_version": "1.5.5", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_mismatch_returns_problem(self):
        cfg = {"anc": {"anc_version": "1.5.4", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        problem = anc_lib.check_version_matches_url(cfg)
        self.assertIsNotNone(problem)
        self.assertIn("1.5.4", problem)
        self.assertIn("1.5.5", problem)

    def test_blank_version_skips(self):
        cfg = {"anc": {"anc_version": "", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_unparseable_url_skips(self):
        cfg = {"anc": {"anc_version": "1.5.5", "anc_release_url": "http://x/anc-release.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_legacy_url_match(self):
        cfg = {"anc": {"anc_version": "1.4.9", "anc_release_url": "http://x/anc-1.4.9-tar-linux-x64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))


class _RecordingPhdl:
    '''Minimal phdl stand-in: records the install command, returns a success line.'''

    def __init__(self, response):
        self.cmd = None
        self._response = response

    def exec(self, cmd, timeout=None):  # noqa: ARG002
        self.cmd = cmd
        return dict(self._response)


class TestDirectInstallers(unittest.TestCase):
    '''Direct 1.5.0+ installers issue single-artifact download + install commands.'''

    CLUSTER = {"node_dict": {"node1": {}}, "username": "u", "priv_key_file": "k"}

    def _cfg(self, url, install_path=""):
        return {"anc": {"anc_release_url": url, "ANC_INSTALL_PATH": install_path}}

    def test_deb_direct_command_shape(self):
        cfg = self._cfg("http://x/anc-release-helios-nda_1.5.5_amd64.deb")
        phdl = _RecordingPhdl({"node1": "ANC_INSTALL_SUCCESS"})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
            anc_lib._install_anc_deb_direct(phdl, self.CLUSTER, cfg)
        self.assertIn("dpkg -i --force-depends ./anc.deb", phdl.cmd)
        self.assertIn("anc-release-helios-nda_1.5.5_amd64.deb", phdl.cmd)
        # No outer-tarball extraction for a direct package.
        self.assertNotIn("outer.tar.gz", phdl.cmd)

    def test_rpm_direct_command_shape(self):
        cfg = self._cfg("http://x/anc-release-helios-nda-1.5.5-1.x86_64.rpm")
        phdl = _RecordingPhdl({"node1": "ANC_INSTALL_SUCCESS"})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
            anc_lib._install_anc_rpm_direct(phdl, self.CLUSTER, cfg)
        self.assertIn("dnf install -y ./anc.rpm", phdl.cmd)
        self.assertNotIn("outer.tar.gz", phdl.cmd)

    def test_tar_direct_default_prefix_no_rewrite(self):
        cfg = self._cfg("http://x/anc-release-helios-nda-1.5.5-x86_64.tar.gz")
        phdl = _RecordingPhdl({"node1": "ANC_INSTALL_SUCCESS"})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "_validate_exe_paths"):
            anc_lib._install_anc_tar_direct(phdl, self.CLUSTER, cfg)
        self.assertIn(f"tar -xzf anc.tar.gz -C '{anc_lib.ANC_TOOLS_PREFIX}'", phdl.cmd)
        # Single archive, no inner anc-tool/anc-content tarballs.
        self.assertNotIn("anc-tool", phdl.cmd)
        self.assertNotIn("anc-content", phdl.cmd)
        # Default prefix -> no exe_path rewrite.
        self.assertNotIn("Rewriting exe_path", phdl.cmd)

    def test_tar_direct_relocated_rewrites_exe_path(self):
        cfg = self._cfg("http://x/anc-release-helios-nda-1.5.5-x86_64.tar.gz", install_path="/home/u/anc")
        phdl = _RecordingPhdl({"node1": "ANC_INSTALL_SUCCESS"})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "_validate_exe_paths"):
            anc_lib._install_anc_tar_direct(phdl, self.CLUSTER, cfg)
        self.assertIn("tar -xzf anc.tar.gz -C '/home/u/anc'", phdl.cmd)
        self.assertIn("Rewriting exe_path", phdl.cmd)
        self.assertIn(f"s#{anc_lib.ANC_TOOLS_PREFIX}#/home/u/anc#g", phdl.cmd)


class _ProbePhdl:
    '''phdl stand-in for resolve_anc_install_location: records the probe command
    and returns a canned per-host response.'''

    def __init__(self, response):
        self.cmd = None
        self._response = response

    def exec(self, cmd, timeout=None):  # noqa: ARG002
        self.cmd = cmd
        return dict(self._response)


class TestResolveAncInstallLocation(unittest.TestCase):
    '''resolve_anc_install_location: probe by URL flavour, verify, cache.'''

    def setUp(self):
        anc_lib._ANC_INSTALL_PATHS = None
        globals_patcher = patch.object(anc_lib.globals, "error_list", [])
        globals_patcher.start()
        self.addCleanup(globals_patcher.stop)
        self.addCleanup(setattr, anc_lib, "_ANC_INSTALL_PATHS", None)

    def _cluster(self):
        return {"node_dict": {"node1": {}}, "username": "u", "priv_key_file": "k"}

    def test_deb_probes_default_prefix(self):
        cfg = {"anc": {"anc_release_url": "http://x/anc_1.5.5_amd64.deb", "ANC_INSTALL_PATH": "/home/u/anc"}}
        phdl = _ProbePhdl({"node1": "ANC_PRESENT"})
        with patch.object(anc_lib, "print_test_output"):
            paths = anc_lib.resolve_anc_install_location(phdl, self._cluster(), cfg)
        self.assertEqual(paths.anc_bin, anc_lib.ANC_BIN)
        self.assertIn(anc_lib.ANC_BIN, phdl.cmd)
        self.assertIs(anc_lib._ANC_INSTALL_PATHS, paths)

    def test_tar_probes_relocated_prefix(self):
        cfg = {"anc": {"anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz", "ANC_INSTALL_PATH": "/home/u/anc"}}
        phdl = _ProbePhdl({"node1": "ANC_PRESENT"})
        with patch.object(anc_lib, "print_test_output"):
            paths = anc_lib.resolve_anc_install_location(phdl, self._cluster(), cfg)
        self.assertEqual(paths.anc_bin, "/home/u/anc/anc/anc.py")
        self.assertIn("/home/u/anc/anc/anc.py", phdl.cmd)

    def test_missing_fails_and_clears_cache(self):
        cfg = {"anc": {"anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz", "ANC_INSTALL_PATH": "/home/u/anc"}}
        phdl = _ProbePhdl({"node1": ""})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "fail_test") as ft:
            result = anc_lib.resolve_anc_install_location(phdl, self._cluster(), cfg)
        self.assertIsNone(result)
        self.assertIsNone(anc_lib._ANC_INSTALL_PATHS)
        ft.assert_called_once()
        self.assertIn("/home/u/anc/anc/anc.py", ft.call_args[0][0])

    def test_partial_coverage_fails(self):
        cluster = {"node_dict": {"node1": {}, "node2": {}}, "username": "u", "priv_key_file": "k"}
        cfg = {"anc": {"anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz", "ANC_INSTALL_PATH": "/home/u/anc"}}
        phdl = _ProbePhdl({"node1": "ANC_PRESENT"})  # node2 unreachable / absent
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "fail_test") as ft:
            result = anc_lib.resolve_anc_install_location(phdl, cluster, cfg)
        self.assertIsNone(result)
        ft.assert_called_once()
        self.assertIn("node2", ft.call_args[0][0])


class TestRunAncGroupsUsesCachedPath(unittest.TestCase):
    '''run_anc_groups builds the command from the session-cached install path.'''

    def setUp(self):
        self.addCleanup(setattr, anc_lib, "_ANC_INSTALL_PATHS", None)

    def test_cached_path_used_in_command(self):
        cached = anc_lib.AncPaths(prefix="/home/u/anc", anc_dir="/home/u/anc/anc", anc_bin="/home/u/anc/anc/anc.py")
        anc_lib._ANC_INSTALL_PATHS = cached
        cluster = {"node_dict": {"node1": {}}, "username": "u", "priv_key_file": "k"}
        cfg = {"anc": {"print_all_to_console": "True", "log_folder_path": "/tmp/logs"}}

        captured = {}

        class FakePhdl:
            def exec(self, cmd, inactivity_timeout=None):  # noqa: ARG002
                captured["cmd"] = cmd
                return {}

        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
            anc_lib.run_anc_groups(FakePhdl(), cluster, cfg, ["cpu_sanity"], "test_cpu_sanity")

        self.assertIn("cd '/home/u/anc/anc' && sudo ./anc.py -g cpu_sanity", captured["cmd"])


if __name__ == "__main__":
    unittest.main()
