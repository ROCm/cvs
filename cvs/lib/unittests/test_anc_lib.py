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

    def test_direct_uses_content_list(self):
        content_out = (
            "Available content plugins (2):\n"
            "  Name                      Version Description\n"
            "  anc-release-helios-nda    1.5.5   Helios NDA Release\n"
            "  base                      1.0.0   Base ANC items\n"
        )

        class FakePhdl:
            def __init__(self):
                self.cmd = None

            def exec(self, cmd, timeout=None):
                self.cmd = cmd
                return {"node1": content_out}

        phdl = FakePhdl()
        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(phdl, "1.5.5", anc_bin="/opt/amdtools/anc/anc.py", is_direct=True)
        self.assertIn("/opt/amdtools/anc/anc.py --content-list", phdl.cmd)
        self.assertTrue(result["node1"])

    def test_direct_mismatch_is_false(self):
        content_out = "  anc-release-helios-nda    1.5.4   Helios NDA Release\n"

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {"node1": content_out}

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(
                FakePhdl(), "1.5.5", anc_bin="/opt/amdtools/anc/anc.py", is_direct=True
            )
        self.assertFalse(result["node1"])


class TestParseReleaseVersionFromContentList(unittest.TestCase):
    '''parse_release_version_from_content_list: read the anc-release-* version column.'''

    def test_direct_155_output(self):
        out = (
            "Start Time: 2026-08-07 12:08:45\n"
            "Available content plugins (5):\n"
            "  Name                      Version Description\n"
            "  anc-release-helios-nda    1.5.5   Helios NDA Release\n"
            "  base                      1.0.0   Base ANC items\n"
            "Program exiting with return code ANC_SUCCESS [0]\n"
        )
        self.assertEqual(anc_lib.parse_release_version_from_content_list(out), "1.5.5")

    def test_legacy_two_column_returns_none(self):
        # Legacy <=1.4.x has no version column and no anc-release-* plugin.
        out = (
            "Available content plugins (4):\n"
            "  base - Base ANC items and utilities...\n"
            "  helios_nda - Helios NDA Test Content\n"
        )
        self.assertIsNone(anc_lib.parse_release_version_from_content_list(out))

    def test_hardware_discovery_failure_returns_none(self):
        out = "FATAL: Error occurred during hardware discovery\n"
        self.assertIsNone(anc_lib.parse_release_version_from_content_list(out))

    def test_empty_returns_none(self):
        self.assertIsNone(anc_lib.parse_release_version_from_content_list(""))
        self.assertIsNone(anc_lib.parse_release_version_from_content_list(None))

    def test_two_part_version(self):
        out = "  anc-release-venice-nda    2.0   Venice NDA Release\n"
        self.assertEqual(anc_lib.parse_release_version_from_content_list(out), "2.0")


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


class TestAssertShellSafe(unittest.TestCase):
    '''_assert_shell_safe rejects every shell-metacharacter that can subvert the
    remote command, not just a single quote.'''

    def test_accepts_clean_value(self):
        anc_lib._assert_shell_safe({"k": "/home/u/my/anc"}, ("k",))  # no raise

    def test_accepts_missing_key(self):
        anc_lib._assert_shell_safe({}, ("k",))  # None value is skipped, no raise

    def test_rejects_each_unsafe_char(self):
        for ch in ("'", '"', "`", "$", "\\", "\n"):
            with self.subTest(ch=ch):
                with self.assertRaises(ValueError):
                    anc_lib._assert_shell_safe({"k": f"/home/u{ch}anc"}, ("k",))

    def test_rejects_command_substitution(self):
        with self.assertRaises(ValueError):
            anc_lib._assert_shell_safe({"ANC_INSTALL_PATH": "/home/$(whoami)/anc"}, ("ANC_INSTALL_PATH",))


class TestResolveAncInstallPrefixValidation(unittest.TestCase):
    '''resolve_anc_install_prefix rejects unsafe/degenerate prefixes.'''

    def test_rejects_shell_unsafe_prefix(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/a$b"}}
        with self.assertRaises(ValueError):
            anc_lib.resolve_anc_install_prefix(cfg, "tar")

    def test_rejects_quote_prefix(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/o'brien/anc"}}
        with self.assertRaises(ValueError):
            anc_lib.resolve_anc_install_prefix(cfg, "tar")

    def test_rejects_root_prefix(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/"}}
        with self.assertRaises(ValueError):
            anc_lib.resolve_anc_install_prefix(cfg, "tar")

    def test_rejects_double_slash_root(self):
        # abspath preserves a leading "//" (POSIX-special); it is still root.
        for value in ("//", "///", "//x/.."):
            with self.subTest(value=value):
                cfg = {"anc": {"ANC_INSTALL_PATH": value}}
                with self.assertRaises(ValueError):
                    anc_lib.resolve_anc_install_prefix(cfg, "tar")

    def test_rejects_root_after_normalisation(self):
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/.."}}  # abspath -> "/home" not "/"
        # "/home/u/.." normalises to "/home", which is allowed; a value that
        # normalises to "/" (e.g. "/..") is rejected.
        cfg_root = {"anc": {"ANC_INSTALL_PATH": "/.."}}
        with self.assertRaises(ValueError):
            anc_lib.resolve_anc_install_prefix(cfg_root, "tar")
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "tar"), "/home")

    def test_deb_ignores_unsafe_install_path(self):
        # deb/rpm never read ANC_INSTALL_PATH, so an unsafe value there is moot.
        cfg = {"anc": {"ANC_INSTALL_PATH": "/home/u/a$b"}}
        self.assertEqual(anc_lib.resolve_anc_install_prefix(cfg, "deb"), anc_lib.ANC_TOOLS_PREFIX)


class TestSedReplacementSafe(unittest.TestCase):
    '''_sed_replacement_safe escapes sed-replacement metacharacters (# & \\).'''

    def test_escapes_delimiter_and_ampersand(self):
        self.assertEqual(anc_lib._sed_replacement_safe("/home/a#b&c"), "/home/a\\#b\\&c")

    def test_escapes_backslash_first(self):
        self.assertEqual(anc_lib._sed_replacement_safe("a\\b"), "a\\\\b")

    def test_plain_value_unchanged(self):
        self.assertEqual(anc_lib._sed_replacement_safe("/home/u/anc"), "/home/u/anc")


class TestRmRfSinkIsSingleQuoted(unittest.TestCase):
    '''The top-level cleanup loop single-quotes the (untrusted) prefix so a
    double-quoted expansion can never command-substitute or mis-target.'''

    CLUSTER = {"node_dict": {"node1": {}}, "username": "u", "priv_key_file": "k"}

    def _run_direct_tar(self, install_path):
        cfg = {"anc": {"anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz", "ANC_INSTALL_PATH": install_path}}
        phdl = _RecordingPhdl({"node1": "ANC_INSTALL_SUCCESS"})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "_validate_exe_paths"):
            anc_lib._install_anc_tar_direct(phdl, self.CLUSTER, cfg)
        return phdl.cmd

    def test_prefix_single_quoted_name_double_quoted(self):
        cmd = self._run_direct_tar("/home/u/anc")
        # prefix single-quoted, archive-derived $name double-quoted.
        self.assertIn("rm -rf '/home/u/anc'/\"$name\"", cmd)
        # The old double-quoted "{prefix}/$name" form must be gone.
        self.assertNotIn('rm -rf "/home/u/anc/$name"', cmd)


class TestPerUserTmpNamespacing(unittest.TestCase):
    '''Remote temp paths are namespaced under /tmp/<user> to avoid cross-user
    ownership collisions on shared nodes.'''

    def test_remote_user_tmp(self):
        self.assertEqual(anc_lib._remote_user_tmp("alice"), "/tmp/alice")

    def test_remote_user_tmp_sanitizes(self):
        # A user value with a path separator collapses to one safe component.
        self.assertEqual(anc_lib._remote_user_tmp("a/b"), "/tmp/a_b")

    def test_validate_exe_paths_uses_per_user_tmp(self):
        cluster = {"node_dict": {"node1": {}}, "username": "alice", "priv_key_file": "k"}
        cmds = []

        class FakePhdl:
            def exec(self, cmd, timeout=None):  # noqa: ARG002
                cmds.append(cmd)
                return {"node1": "VALIDATION_SUCCESS"}

            def upload_file(self, local, remote):
                cmds.append(f"UPLOAD {remote}")

        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib.globals, "error_list", []):
            anc_lib._validate_exe_paths(FakePhdl(), cluster, "/opt/amdtools/anc/content")

        joined = "\n".join(cmds)
        self.assertIn("mkdir -p '/tmp/alice'", joined)
        self.assertIn("/tmp/alice/validate_exe_paths.py", joined)

    def test_run_anc_groups_quiet_uses_per_user_tmp(self):
        anc_lib._ANC_INSTALL_PATHS = anc_lib.AncPaths("/opt/amdtools", "/opt/amdtools/anc", "/opt/amdtools/anc/anc.py")
        self.addCleanup(setattr, anc_lib, "_ANC_INSTALL_PATHS", None)
        cluster = {"node_dict": {"node1": {}}, "username": "bob", "priv_key_file": "k"}
        cfg = {"anc": {"print_all_to_console": "False", "log_folder_path": "/tmp/logs"}}
        captured = {}

        class FakePhdl:
            def exec(self, cmd, inactivity_timeout=None):  # noqa: ARG002
                captured["cmd"] = cmd
                return {}

        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
            anc_lib.run_anc_groups(FakePhdl(), cluster, cfg, ["cpu_sanity"], "test_cpu_sanity")

        self.assertIn("/tmp/bob/anc_run_$$.out", captured["cmd"])
        self.assertIn("mkdir -p '/tmp/bob'", captured["cmd"])


class TestValidateClusterUsername(unittest.TestCase):
    '''validate_cluster_username rejects usernames unsafe for shell interpolation.'''

    def test_clean_username_ok(self):
        self.assertIsNone(anc_lib.validate_cluster_username({"username": "ashmishr"}))

    def test_missing_or_blank_is_deferred(self):
        self.assertIsNone(anc_lib.validate_cluster_username({}))
        self.assertIsNone(anc_lib.validate_cluster_username({"username": "  "}))

    def test_rejects_space(self):
        self.assertIsNotNone(anc_lib.validate_cluster_username({"username": "a b"}))

    def test_rejects_command_injection(self):
        problem = anc_lib.validate_cluster_username({"username": "x; rm -rf /root"})
        self.assertIsNotNone(problem)


class TestValidateAncConfig(unittest.TestCase):
    '''validate_anc_config aggregates the fail-fast config problems.'''

    def _cfg(self, **anc):
        base = {"anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz", "ANC_INSTALL_PATH": ""}
        base.update(anc)
        return {"anc": base}

    def _cluster(self, username="ashmishr"):
        return {"username": username}

    def test_clean_config_no_problems(self):
        problems = anc_lib.validate_anc_config(self._cfg(), self._cluster(), require_log_folder=False)
        self.assertEqual(problems, [])

    def test_blank_url_flagged(self):
        problems = anc_lib.validate_anc_config(self._cfg(anc_release_url=""), self._cluster(), require_log_folder=False)
        self.assertTrue(any("anc_release_url" in p for p in problems))

    def test_version_mismatch_flagged(self):
        cfg = self._cfg(anc_version="1.5.4")  # url is 1.5.5
        problems = anc_lib.validate_anc_config(cfg, self._cluster(), require_log_folder=False)
        self.assertTrue(any("does not match" in p for p in problems))

    def test_unsafe_prefix_flagged_cleanly(self):
        cfg = self._cfg(ANC_INSTALL_PATH="//")
        problems = anc_lib.validate_anc_config(cfg, self._cluster(), require_log_folder=False)
        # The resolve_anc_paths_from_config ValueError is surfaced, not raised.
        self.assertTrue(any("root" in p.lower() for p in problems))

    def test_bad_username_flagged(self):
        problems = anc_lib.validate_anc_config(self._cfg(), self._cluster(username="a b"), require_log_folder=False)
        self.assertTrue(any("username" in p for p in problems))

    def test_log_folder_required_for_group_suites(self):
        cfg = self._cfg(log_folder_path="")
        without = anc_lib.validate_anc_config(cfg, self._cluster(), require_log_folder=False)
        with_req = anc_lib.validate_anc_config(cfg, self._cluster(), require_log_folder=True)
        self.assertFalse(any("log_folder_path" in p for p in without))
        self.assertTrue(any("log_folder_path" in p for p in with_req))


class TestTarCleanupFiltersDotDot(unittest.TestCase):
    '''Both tar installers filter '.'/'..' from the top-level cleanup list.'''

    CLUSTER = {"node_dict": {"node1": {}}, "username": "u", "priv_key_file": "k"}

    def _cmd(self, installer, url):
        cfg = {"anc": {"anc_release_url": url, "ANC_INSTALL_PATH": ""}}
        phdl = _RecordingPhdl({"node1": "ANC_INSTALL_SUCCESS"})
        with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "_validate_exe_paths"):
            installer(phdl, self.CLUSTER, cfg)
        return phdl.cmd

    def test_direct_tar_filters_and_guards(self):
        cmd = self._cmd(anc_lib._install_anc_tar_direct, "http://x/anc-1.5.5-x86_64.tar.gz")
        # tops filter drops '.' and '..'
        self.assertIn("grep -vE '^([.]{1,2})?$'", cmd)
        # per-name defensive guard against a path-separator/dot component
        self.assertIn('case "$name" in */*|.|..) continue;; esac', cmd)

    def test_legacy_tar_filters_and_guards(self):
        cmd = self._cmd(anc_lib._install_anc_tar, "http://x/anc-1.4.9-tar-linux-x64.tar.gz")
        self.assertIn("grep -vE '^([.]{1,2})?$'", cmd)
        self.assertIn('case "$name" in */*|.|..) continue;; esac', cmd)


class TestChownArgumentQuoted(unittest.TestCase):
    '''_pull_log_dir single-quotes the chown user argument (defense in depth).'''

    def test_chown_user_single_quoted(self):
        captured = {}

        class FakeSingle:
            def exec(self, cmd, timeout=None):  # noqa: ARG002
                captured.setdefault("first", cmd)
                return {}

            def download_file(self, remote, local):  # noqa: ARG002
                return {}

        # download_file returns {} so _pull_log_dir bails after the archive_cmd;
        # we only care that the archive command quotes the user.
        anc_lib._pull_log_dir(FakeSingle(), "node1", "alice", "/root/logs/run1", "/tmp/dest")
        self.assertIn("sudo chown 'alice'", captured["first"])


if __name__ == "__main__":
    unittest.main()
