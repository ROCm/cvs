# cvs/lib/unittests/test_anc_lib.py
import os
import unittest
from unittest.mock import MagicMock, patch

import cvs.lib.anc_lib as anc_lib


def _patch_run_dir(run_dir):
    '''Patch RunLayout.get() so resolve_anc_log_folder resolves to a fixed run_dir
    without touching the filesystem or the scheduler environment.'''
    layout = MagicMock()
    layout.run_dir = run_dir
    return patch("cvs.core.run_layout.RunLayout.get", return_value=layout)


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
        # Legacy node (no anc-release-* line): content-list probe yields nothing,
        # so it falls back to --version, read from the labelled Release line.
        class FakePhdl:
            def __init__(self):
                self.cmd = None

            def exec(self, cmd, timeout=None):
                self.cmd = cmd
                return {"node1": "Release Version: 1.4.9\n"}

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
            result = anc_lib.node_version_matches(phdl, "1.5.5", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertIn("/opt/amdtools/anc/anc.py --content-list", phdl.cmd)
        self.assertTrue(result["node1"])

    def test_direct_mismatch_is_false(self):
        # Installed 1.5.4 does NOT satisfy a request for 1.5.5 (installed < required).
        content_out = "  anc-release-helios-nda    1.5.4   Helios NDA Release\n"

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {"node1": content_out}

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.5.5", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertFalse(result["node1"])

    def test_direct_higher_installed_satisfies(self):
        # Installed 1.6.0 satisfies a request for 1.5.5 (installed >= required).
        content_out = "  anc-release-helios-nda    1.6.0   Helios NDA Release\n"

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {"node1": content_out}

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.5.5", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertTrue(result["node1"])

    def test_direct_installed_rc_satisfies_base_request(self):
        # Installed 1.7.0-rc.1 satisfies a request for base 1.7.0 (rc == base).
        content_out = "  anc-release-helios-nda    1.7.0-rc.1 Helios NDA Release\n"

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {"node1": content_out}

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.7.0", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertTrue(result["node1"])

    def test_direct_installed_rc1_does_not_satisfy_rc2(self):
        # Installed 1.7.0-rc.1 does NOT satisfy a request for 1.7.0-rc.2.
        content_out = "  anc-release-helios-nda    1.7.0-rc.1 Helios NDA Release\n"

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {"node1": content_out}

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.7.0-rc.2", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertFalse(result["node1"])

    def test_legacy_reads_release_version_label(self):
        # Legacy (<=1.4.x) node: no anc-release-* line -> fall back to --version,
        # read from the labelled "Release Version:" record.
        def exec_impl(cmd, timeout=None):
            if "--content-list" in cmd:
                return {"node1": "Available content plugins (2):\n  base - Base items\n"}
            return {"node1": "Release Name: helios-nda\nRelease Version: 1.4.10\n"}

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return exec_impl(cmd, timeout)

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.4.9", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertTrue(result["node1"])  # legacy 1.4.10 >= 1.4.9

    def test_generation_detected_from_node_not_url(self):
        # A direct build is present (anc-release-* line); its --version would
        # report an unrelated tool version, but that is never consulted because
        # the content-list line is authoritative. No is_direct flag is passed.
        cmds = []

        def exec_impl(cmd, timeout=None):
            cmds.append(cmd)
            if "--content-list" in cmd:
                return {"node1": "  anc-release-helios-nda    1.6.0   Helios NDA Release\n"}
            return {"node1": "Framework Version: 0.9.9"}

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return exec_impl(cmd, timeout)

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.5.5", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertTrue(result["node1"])  # 1.6.0 (content-list) >= 1.5.5, not 0.9.9
        self.assertFalse(any("--version" in c for c in cmds))  # no fallback probe needed

    def test_absent_anc_does_not_satisfy(self):
        # No content-list line AND no Release Version line -> node does not satisfy.
        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {"node1": "anc.py: command not found"}

        with patch.object(anc_lib, "print_test_output"):
            result = anc_lib.node_version_matches(FakePhdl(), "1.4.9", anc_bin="/opt/amdtools/anc/anc.py")
        self.assertFalse(result["node1"])


class TestCompareAncVersions(unittest.TestCase):
    '''compare_anc_versions / parse_anc_version: ANC version ordering.'''

    def test_parse_base_and_rc(self):
        self.assertEqual(anc_lib.parse_anc_version("1.7.0"), ((1, 7, 0), None))
        self.assertEqual(anc_lib.parse_anc_version("1.7.0-rc.1"), ((1, 7, 0), 1))
        self.assertEqual(anc_lib.parse_anc_version("1.4.9"), ((1, 4, 9), None))

    def test_parse_none_when_no_version(self):
        self.assertIsNone(anc_lib.parse_anc_version(""))
        self.assertIsNone(anc_lib.parse_anc_version("no-version-here"))

    def test_parse_bare_integer_is_not_a_version(self):
        self.assertIsNone(anc_lib.parse_anc_version("7"))

    def test_parse_rc_kept_with_trailing_text(self):
        # The rc suffix is part of the version token, so surrounding text does
        # not drop it (regression for an end-anchored rc parser).
        self.assertEqual(anc_lib.parse_anc_version("ANC 1.7.0-rc.1 build"), ((1, 7, 0), 1))
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0-rc.1 build", "1.7.0-rc.2"), -1)

    def test_is_strict_anc_version(self):
        # Strict gate for user-configured values: whole-string match only.
        self.assertTrue(anc_lib.is_strict_anc_version("1.7.0"))
        self.assertTrue(anc_lib.is_strict_anc_version("1.7.0-rc.1"))
        self.assertTrue(anc_lib.is_strict_anc_version(" 1.4.9 "))
        for bad in ("1.7.0-rc.bad", "release-1.7.0", "1.7.0 build", "1.7.0-RC.2", "7", "", None):
            self.assertFalse(anc_lib.is_strict_anc_version(bad), bad)

    def test_rc_equals_its_base(self):
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0-rc.1", "1.7.0"), 0)
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0", "1.7.0-rc.1"), 0)

    def test_rc_to_rc_numeric(self):
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0-rc.1", "1.7.0-rc.2"), -1)
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0-rc.2", "1.7.0-rc.1"), 1)
        # numeric, not lexical: rc.2 < rc.10
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0-rc.2", "1.7.0-rc.10"), -1)

    def test_base_ordering(self):
        self.assertEqual(anc_lib.compare_anc_versions("1.6.0", "1.7.0-rc.1"), -1)
        self.assertEqual(anc_lib.compare_anc_versions("1.7.0-rc.5", "1.8.0"), -1)
        self.assertEqual(anc_lib.compare_anc_versions("1.4.9", "1.4.10"), -1)

    def test_zero_padding(self):
        self.assertEqual(anc_lib.compare_anc_versions("1.7", "1.7.0"), 0)

    def test_unparseable_raises(self):
        with self.assertRaises(ValueError):
            anc_lib.compare_anc_versions("garbage", "1.7.0")

    def test_satisfies(self):
        self.assertTrue(anc_lib.anc_version_satisfies("1.7.0-rc.1", "1.7.0"))
        self.assertTrue(anc_lib.anc_version_satisfies("1.8.0", "1.7.0"))
        self.assertFalse(anc_lib.anc_version_satisfies("1.7.0-rc.1", "1.7.0-rc.2"))
        self.assertFalse(anc_lib.anc_version_satisfies("1.6.0", "1.7.0"))
        self.assertFalse(anc_lib.anc_version_satisfies("garbage", "1.7.0"))


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

    def test_rc_version_from_real_node_output(self):
        # Shape of a real --content-list run that carries an rc release version.
        out = (
            "Start Time: 2026-09-23 07:43:13\n"
            "Log Directory: /home/testuser/logs/anc_20260923-074313\n\n"
            "Available content plugins (6):\n"
            "  Name                      Version Description\n"
            "  ainic-nda                 1.0.0   NDA Test Content for AMD AINIC\n"
            "  anc-release-helios-nda    1.7.0-rc.1 Helios NDA Release\n"
            "  base                      1.1.0   Base ANC items and utilities available for all test plans\n"
            "  helios-nda                1.0.3   Helios NDA Test Content\n"
            "Program exiting with return code ANC_SUCCESS [0]\n"
        )
        self.assertEqual(anc_lib.parse_release_version_from_content_list(out), "1.7.0-rc.1")

    def test_malformed_rc_in_column_is_rejected(self):
        out = "  anc-release-helios-nda    1.7.0-rc.bad   Helios NDA Release\n"
        self.assertIsNone(anc_lib.parse_release_version_from_content_list(out))


class TestParseVersionFromVersionOutput(unittest.TestCase):
    '''_parse_version_from_version_output: read ONLY the labelled Release Version
    record, never an unrelated number from stdout/stderr.'''

    def test_reads_release_version_line(self):
        out = "Release Name: helios-nda\nRelease Version: 1.4.9\n"
        self.assertEqual(anc_lib._parse_version_from_version_output(out), "1.4.9")

    def test_label_is_case_insensitive_token_is_not(self):
        self.assertEqual(anc_lib._parse_version_from_version_output("release version: 1.4.9"), "1.4.9")
        self.assertEqual(anc_lib._parse_version_from_version_output("ANC Release Version: 1.7.0-rc.1"), "1.7.0-rc.1")
        # The label is case-insensitive but the -rc TOKEN is not: an uppercase RC
        # is rejected (None) rather than silently truncated to its base.
        self.assertIsNone(anc_lib._parse_version_from_version_output("Release Version: 1.7.0-RC.2"))

    def test_ignores_unrelated_numbers_and_paths(self):
        # A warning number before the version, or the version inside a "No such
        # file" path when ANC is absent, must NOT be read as the release.
        self.assertEqual(
            anc_lib._parse_version_from_version_output("WARNING: libfoo 2.3\nRelease Version: 1.4.9\n"), "1.4.9"
        )
        self.assertIsNone(anc_lib._parse_version_from_version_output("bash: /opt/anc-1.6.0/anc/anc.py: No such file\n"))

    def test_no_release_line_is_none(self):
        self.assertIsNone(anc_lib._parse_version_from_version_output("anc.py: command not found"))
        self.assertIsNone(anc_lib._parse_version_from_version_output(""))
        self.assertIsNone(anc_lib._parse_version_from_version_output(None))


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

    def test_rc_url_keeps_suffix(self):
        # The rc suffix is part of the version; the trailing package revision is not.
        self.assertEqual(
            anc_lib.parse_version_from_url("http://x/anc-release-helios-nda-1.7.0-rc.1-1.x86_64.rpm"),
            "1.7.0-rc.1",
        )

    def test_malformed_or_uppercase_rc_rejected(self):
        # Attached malformed / uppercase rc must NOT truncate to the base.
        self.assertIsNone(anc_lib.parse_version_from_url("http://x/anc-1.7.0-rc.bad.rpm"))
        self.assertIsNone(anc_lib.parse_version_from_url("http://x/anc-1.7.0-rc.1foo.rpm"))
        self.assertIsNone(anc_lib.parse_version_from_url("http://x/anc-1.7.0-RC.1.rpm"))

    def test_malformed_first_token_not_skipped(self):
        # A malformed leading token must be rejected, not skipped for a later
        # unrelated number (the 2.31 in a -glibc-2.31 suffix).
        self.assertIsNone(anc_lib.parse_version_from_url("http://x/anc-1.7.0-rc.bad-glibc-2.31.tar.gz"))

    def test_package_revision_still_allowed(self):
        self.assertEqual(
            anc_lib.parse_version_from_url("http://x/anc-1.7.0-rc.10-2.x86_64.rpm"),
            "1.7.0-rc.10",
        )


class TestCheckVersionMatchesUrl(unittest.TestCase):
    '''check_version_matches_url: abort when configured version is invalid or > URL version.'''

    def test_match_returns_none(self):
        cfg = {"anc": {"anc_version": "1.5.5", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_config_lower_than_url_is_ok(self):
        # config < url is fine: the archive can satisfy the requested version.
        cfg = {"anc": {"anc_version": "1.5.4", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_config_higher_than_url_returns_problem(self):
        cfg = {"anc": {"anc_version": "1.5.6", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        problem = anc_lib.check_version_matches_url(cfg)
        self.assertIsNotNone(problem)
        self.assertIn("1.5.6", problem)
        self.assertIn("1.5.5", problem)

    def test_rc_config_matches_base_url(self):
        # rc is treated equal to its base, so config rc.1 vs url base is OK.
        cfg = {"anc": {"anc_version": "1.7.0-rc.1", "anc_release_url": "http://x/anc-1.7.0-x86_64.rpm"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_base_config_matches_rc_url(self):
        # config base vs url rc of same base is OK (equal).
        cfg = {"anc": {"anc_version": "1.7.0", "anc_release_url": "http://x/anc-1.7.0-rc.1-1.x86_64.rpm"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_higher_rc_config_than_url_rc_aborts(self):
        cfg = {"anc": {"anc_version": "1.7.0-rc.2", "anc_release_url": "http://x/anc-1.7.0-rc.1-1.x86_64.rpm"}}
        problem = anc_lib.check_version_matches_url(cfg)
        self.assertIsNotNone(problem)
        self.assertIn("1.7.0-rc.2", problem)

    def test_blank_version_skips(self):
        cfg = {"anc": {"anc_version": "", "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_unparseable_url_skips(self):
        cfg = {"anc": {"anc_version": "1.5.5", "anc_release_url": "http://x/anc-release.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_legacy_url_match(self):
        cfg = {"anc": {"anc_version": "1.4.9", "anc_release_url": "http://x/anc-1.4.9-tar-linux-x64.tar.gz"}}
        self.assertIsNone(anc_lib.check_version_matches_url(cfg))

    def test_invalid_configured_version_is_problem(self):
        # URL parses, but the configured value is not a strict version -> report
        # a config problem instead of silently reading it as a base version.
        for bad in ("garbage", "release-1.7.0", "1.7.0-rc.bad", "1.7.0-RC.2"):
            cfg = {"anc": {"anc_version": bad, "anc_release_url": "http://x/anc-1.7.0-x86_64.tar.gz"}}
            problem = anc_lib.check_version_matches_url(cfg)
            self.assertIsNotNone(problem, bad)
            self.assertIn("not a valid ANC version", problem)

    def test_falsy_nonblank_version_is_not_treated_as_omitted(self):
        # A present-but-falsy value (0 / False) must NOT be read as "omitted"
        # (which would bypass validation and disable install verification); it
        # reaches the strict check and is reported.
        for bad in (0, False, "0"):
            cfg = {"anc": {"anc_version": bad, "anc_release_url": "http://x/anc-1.7.0-x86_64.tar.gz"}}
            problem = anc_lib.check_version_matches_url(cfg)
            self.assertIsNotNone(problem, repr(bad))
            self.assertIn("not a valid ANC version", problem)

    def test_none_or_blank_version_is_omitted(self):
        for omitted in (None, "", "   "):
            cfg = {"anc": {"anc_version": omitted, "anc_release_url": "http://x/anc-1.7.0-x86_64.tar.gz"}}
            self.assertIsNone(anc_lib.check_version_matches_url(cfg), repr(omitted))


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
        cfg = {"anc": {"print_all_to_console": "True"}}

        captured = {}

        class FakePhdl:
            def exec(self, cmd, inactivity_timeout=None):  # noqa: ARG002
                captured["cmd"] = cmd
                return {}

        with _patch_run_dir("/ws/cvs_runs/1"):
            with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
                anc_lib.run_anc_groups(FakePhdl(), cluster, cfg, ["cpu_content_check"], "test_cpu_content_check")

        self.assertIn("cd '/home/u/anc/anc' && sudo ./anc.py -g cpu_content_check", captured["cmd"])


class TestRunAncItems(unittest.TestCase):
    '''run_anc_items runs ``anc.py -i <item>`` (the item selector, not -g).'''

    def setUp(self):
        self.addCleanup(setattr, anc_lib, "_ANC_INSTALL_PATHS", None)
        anc_lib._ANC_INSTALL_PATHS = anc_lib.AncPaths("/opt/amdtools", "/opt/amdtools/anc", "/opt/amdtools/anc/anc.py")

    def _run(self, cfg, item="gemm_fp8_trig"):
        cluster = {"node_dict": {"node1": {}}, "username": "bob", "priv_key_file": "k"}
        captured = {}

        class FakePhdl:
            def exec(self, cmd, inactivity_timeout=None):  # noqa: ARG002
                captured["cmd"] = cmd
                return {}

        with _patch_run_dir("/ws/cvs_runs/1"):
            with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
                anc_lib.run_anc_items(FakePhdl(), cluster, cfg, [item], f"test_{item}")
        return captured["cmd"]

    def test_item_selector_print_all(self):
        cmd = self._run({"anc": {"print_all_to_console": "True"}})
        self.assertIn("sudo ./anc.py -i gemm_fp8_trig", cmd)
        self.assertNotIn("-g gemm_fp8_trig", cmd)

    def test_item_selector_quiet_mode(self):
        cmd = self._run({"anc": {"print_all_to_console": "False"}})
        self.assertIn("sudo ./anc.py -i gemm_fp8_trig", cmd)
        # Quiet mode greps for a missing group OR item FATAL line.
        self.assertIn("FATAL: (Group|Item)", cmd)


class TestNotFoundRegexScopedToUnit(unittest.TestCase):
    '''anc_not_found_re matches ONLY its own unit's FATAL line (group vs item).'''

    def test_group_matcher_matches_group_line(self):
        self.assertTrue(anc_lib.anc_not_found_re("group").search("FATAL: Group 'cpu_mfg_l10' not found"))

    def test_group_matcher_ignores_item_line(self):
        # A group run must NOT be downgraded to "not available" by an item's
        # FATAL line (e.g. a missing leaf item inside an installed group).
        self.assertIsNone(anc_lib.anc_not_found_re("group").search("FATAL: Item 'gemm_fp8_trig' not found"))

    def test_item_matcher_matches_item_line(self):
        self.assertTrue(anc_lib.anc_not_found_re("item").search("FATAL: Item 'gemm_fp8_trig' not found"))

    def test_item_matcher_ignores_group_line(self):
        self.assertIsNone(anc_lib.anc_not_found_re("item").search("FATAL: Group 'cpu_mfg_l10' not found"))

    def test_unrelated_line_does_not_match(self):
        self.assertIsNone(anc_lib.anc_not_found_re("item").search("All items passed"))


class TestResolveAncLogFolder(unittest.TestCase):
    '''resolve_anc_log_folder lays the fixed anc_logs tree under this run's run_dir.'''

    def test_substitutes_node_test_and_timestamp_under_run_dir(self):
        with _patch_run_dir("/ws/cvs_runs/job-42"):
            path = anc_lib.resolve_anc_log_folder("test_cpu", "20261006-010203", node="10.0.0.5_node01")
        self.assertEqual(path, "/ws/cvs_runs/job-42/anc_logs/10.0.0.5_node01/test_cpu/20261006-010203")

    def test_node_token_left_intact_when_node_none(self):
        with _patch_run_dir("/ws/cvs_runs/job-42"):
            path = anc_lib.resolve_anc_log_folder("test_cpu", "20261006-010203")
        # The banner pattern keeps "<node>" so it can be shown before any node is known.
        self.assertEqual(path, "/ws/cvs_runs/job-42/anc_logs/<node>/test_cpu/20261006-010203")


class TestConsoleLogUnder(unittest.TestCase):
    '''_console_log_under locates the collected console.log under a node dir.'''

    def test_finds_nested_console_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            nested = os.path.join(root, "anc_run", "deep")
            os.makedirs(nested)
            target = os.path.join(nested, anc_lib.CONSOLE_LOG)
            open(target, "w").close()
            self.assertEqual(anc_lib._console_log_under(root), target)

    def test_none_dest_dir(self):
        self.assertIsNone(anc_lib._console_log_under(None))

    def test_missing_dir(self):
        self.assertIsNone(anc_lib._console_log_under("/no/such/dir/xyz"))

    def test_empty_dir_has_no_console_log(self):
        import tempfile

        with tempfile.TemporaryDirectory() as empty:
            self.assertIsNone(anc_lib._console_log_under(empty))


class TestNodeStatusMapping(unittest.TestCase):
    '''_node_status maps a NodeResult reason to a deck status.'''

    def _result(self, reason):
        return anc_lib.NodeResult(reason=reason, dest_dir=None, label="n1", errors_json=None)

    def test_none_reason_is_pass(self):
        self.assertEqual(anc_lib._node_status(self._result(None)), "pass")

    def test_not_available_reason_is_na(self):
        reason = "This test is not available on the remote system [10.0.0.1_n1]"
        self.assertEqual(anc_lib._node_status(self._result(reason)), "na")

    def test_other_reason_is_fail(self):
        self.assertEqual(anc_lib._node_status(self._result("ANC returned ANC_PROG_FAIL_IN_ITEM [19]")), "fail")


class TestCaptureRundeckResultsWithoutFixture(unittest.TestCase):
    '''_capture_rundeck_results is a no-op when the suite has no anc_res_dict fixture.'''

    def test_missing_fixture_is_noop(self):
        import pytest

        class FakeRequest:
            config = type("C", (), {"_suite_name": "anc_installation"})()

            def getfixturevalue(self, name):
                # Match what pytest raises for an undefined fixture; construct via
                # __new__ to avoid the real ctor needing a live request stack.
                raise pytest.FixtureLookupError.__new__(pytest.FixtureLookupError)

        # Must not raise; simply returns without touching any store.
        anc_lib._capture_rundeck_results(
            FakeRequest(), {}, {}, ["cpu_content_check"], "test_cpu_content_check", "ts", [], {}, {}
        )


class TestAttachErrorsJsonReturnsRenamedHref(unittest.TestCase):
    '''_attach_node_errors_json returns the RENAMED file's basename per node.

    Regression guard: the Run Deck must link the copy actually placed next to the
    report (``<label>_<test>_<ts>_errors.json``), not the source ``errors.json``.
    '''

    def test_href_is_renamed_copy_not_source_basename(self):
        result = anc_lib.NodeResult(
            reason=None, dest_dir="/x", label="10.0.0.1_n1", errors_json="/some/src/errors.json"
        )

        class FakeMgr:
            def add_html_to_report(self, src, request=None, dest_name=None, track_in_reports=True):  # noqa: ARG002
                # add_html_to_report returns the RELATIVE path of the copied,
                # renamed file (mirrors the real manager).
                return f"anc_test_gpu_html/{dest_name}"

        with patch.object(anc_lib.os.path, "isfile", return_value=True), patch.object(anc_lib, "_stash_report_link"):
            hrefs = anc_lib._attach_node_errors_json(None, FakeMgr(), "test_hbm_lvl1", "20260101-000000", [result])

        self.assertEqual(hrefs["10.0.0.1_n1"], "10.0.0.1_n1_test_hbm_lvl1_20260101-000000_errors.json")
        self.assertNotEqual(hrefs["10.0.0.1_n1"], "errors.json")


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
        cfg = {"anc": {"print_all_to_console": "False"}}
        captured = {}

        class FakePhdl:
            def exec(self, cmd, inactivity_timeout=None):  # noqa: ARG002
                captured["cmd"] = cmd
                return {}

        with _patch_run_dir("/ws/cvs_runs/1"):
            with patch.object(anc_lib, "print_test_output"), patch.object(anc_lib, "update_test_result"):
                anc_lib.run_anc_groups(FakePhdl(), cluster, cfg, ["cpu_content_check"], "test_cpu_content_check")

        self.assertIn("/tmp/bob/anc_run_$$.out", captured["cmd"])
        self.assertIn("mkdir -p '/tmp/bob'", captured["cmd"])


class TestValidateClusterUsername(unittest.TestCase):
    '''validate_cluster_username rejects usernames unsafe for shell interpolation.'''

    def test_clean_username_ok(self):
        self.assertIsNone(anc_lib.validate_cluster_username({"username": "testuser"}))

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

    def _cluster(self, username="testuser"):
        return {"username": username}

    def test_clean_config_no_problems(self):
        problems = anc_lib.validate_anc_config(self._cfg(), self._cluster())
        self.assertEqual(problems, [])

    def test_blank_url_flagged(self):
        problems = anc_lib.validate_anc_config(self._cfg(anc_release_url=""), self._cluster())
        self.assertTrue(any("anc_release_url" in p for p in problems))

    def test_config_newer_than_url_flagged(self):
        cfg = self._cfg(anc_version="1.5.6")  # url is 1.5.5; config newer than archive
        problems = anc_lib.validate_anc_config(cfg, self._cluster())
        self.assertTrue(any("is newer than" in p for p in problems))

    def test_config_older_than_url_ok(self):
        cfg = self._cfg(anc_version="1.5.4")  # url is 1.5.5; archive can satisfy request
        problems = anc_lib.validate_anc_config(cfg, self._cluster())
        self.assertFalse(any("newer than" in p or "does not match" in p for p in problems))

    def test_invalid_version_flagged(self):
        cfg = self._cfg(anc_version="garbage")  # url parses, configured value does not
        problems = anc_lib.validate_anc_config(cfg, self._cluster())
        self.assertTrue(any("not a valid ANC version" in p for p in problems))

    def test_unsafe_prefix_flagged_cleanly(self):
        cfg = self._cfg(ANC_INSTALL_PATH="//")
        problems = anc_lib.validate_anc_config(cfg, self._cluster())
        # The resolve_anc_paths_from_config ValueError is surfaced, not raised.
        self.assertTrue(any("root" in p.lower() for p in problems))

    def test_bad_username_flagged(self):
        problems = anc_lib.validate_anc_config(self._cfg(), self._cluster(username="a b"))
        self.assertTrue(any("username" in p for p in problems))


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


class TestInstallAncVersionGating(unittest.TestCase):
    '''install_anc: anc_version drives the precheck skip and post-install verify.'''

    CLUSTER = {"node_dict": {"node1": {}}, "username": "u", "priv_key_file": "k"}

    def setUp(self):
        p = patch.object(anc_lib.globals, "error_list", [])
        p.start()
        self.addCleanup(p.stop)

    def _cfg(self, version):
        return {
            "anc": {
                "anc_release_url": "http://x/anc-1.5.5-x86_64.tar.gz",
                "anc_version": version,
                "ANC_INSTALL_PATH": "",
            }
        }

    @staticmethod
    def _content(version):
        return f"  anc-release-helios-nda    {version}   Helios NDA Release\n" if version else ""

    def _phdl(self, state):
        content = self._content

        class FakePhdl:
            def exec(self, cmd, timeout=None):
                return {host: content(ver) for host, ver in state.items()}

        return FakePhdl()

    def test_precheck_skip_when_installed_satisfies(self):
        # node already runs 1.6.0 (>= requested 1.5.5): skip, no installer call.
        with (
            patch.object(anc_lib, "print_test_output"),
            patch.object(anc_lib, "update_test_result") as upd,
            patch.object(anc_lib, "_install_anc_tar_direct") as installer,
            patch.object(anc_lib, "fail_test") as ft,
        ):
            anc_lib.install_anc(self._phdl({"node1": "1.6.0"}), self.CLUSTER, self._cfg("1.5.5"))
        installer.assert_not_called()
        ft.assert_not_called()
        upd.assert_called_once()

    def test_mixed_cluster_does_not_skip_all_or_nothing(self):
        # node1 satisfies (1.6.0) but node2 is below (1.5.4): installer runs for
        # the whole set, then the post-verify passes once node2 is bumped.
        cluster = {"node_dict": {"node1": {}, "node2": {}}, "username": "u", "priv_key_file": "k"}
        state = {"node1": "1.6.0", "node2": "1.5.4"}

        def do_install(*args, **kwargs):
            state["node2"] = "1.5.5"

        with (
            patch.object(anc_lib, "print_test_output"),
            patch.object(anc_lib, "update_test_result") as upd,
            patch.object(anc_lib, "_install_anc_tar_direct", side_effect=do_install) as installer,
            patch.object(anc_lib, "fail_test") as ft,
        ):
            anc_lib.install_anc(self._phdl(state), cluster, self._cfg("1.5.5"))
        installer.assert_called_once()
        ft.assert_not_called()
        upd.assert_called_once()

    def test_post_verify_fails_when_still_below(self):
        state = {"node1": None}  # absent at precheck; installer leaves it too low

        def do_install(*args, **kwargs):
            state["node1"] = "1.5.4"

        with (
            patch.object(anc_lib, "print_test_output"),
            patch.object(anc_lib, "update_test_result"),
            patch.object(anc_lib, "_install_anc_tar_direct", side_effect=do_install),
            patch.object(anc_lib, "fail_test") as ft,
        ):
            anc_lib.install_anc(self._phdl(state), self.CLUSTER, self._cfg("1.5.5"))
        ft.assert_called_once()
        self.assertIn("node1", ft.call_args[0][0])


class TestSafeTarExtract(unittest.TestCase):
    '''_safe_tar_extract: only regular files/dirs within dest; links/devices rejected.'''

    def test_extracts_safe_member(self):
        import tarfile as _tar
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            payload = os.path.join(root, "payload")
            with open(payload, "w") as fh:
                fh.write("x")
            tar_path = os.path.join(root, "a.tar")
            with _tar.open(tar_path, "w") as tf:
                tf.add(payload, arcname="logs/console.log")
            dest = os.path.join(root, "dest")
            os.makedirs(dest)
            with _tar.open(tar_path) as tf:
                anc_lib._safe_tar_extract(tf, dest)
            self.assertTrue(os.path.isfile(os.path.join(dest, "logs", "console.log")))

    def test_rejects_parent_traversal(self):
        import tarfile as _tar
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            payload = os.path.join(root, "payload")
            with open(payload, "w") as fh:
                fh.write("x")
            tar_path = os.path.join(root, "a.tar")
            with _tar.open(tar_path, "w") as tf:
                tf.add(payload, arcname="../escape.txt")
            dest = os.path.join(root, "dest")
            os.makedirs(dest)
            with _tar.open(tar_path) as tf:
                with self.assertRaises(_tar.TarError):
                    anc_lib._safe_tar_extract(tf, dest)
            self.assertFalse(os.path.exists(os.path.join(root, "escape.txt")))

    def test_rejects_links_and_special_files(self):
        import tarfile as _tar
        import tempfile

        for type_const, linkname in (
            (_tar.SYMTYPE, "/etc/passwd"),
            (_tar.SYMTYPE, "../out"),
            (_tar.LNKTYPE, "../out"),
            (_tar.FIFOTYPE, None),
            (_tar.CHRTYPE, None),
            (_tar.BLKTYPE, None),
        ):
            with self.subTest(type=type_const):
                with tempfile.TemporaryDirectory() as root:
                    tar_path = os.path.join(root, "a.tar")
                    with _tar.open(tar_path, "w") as tf:
                        info = _tar.TarInfo("member")
                        info.type = type_const
                        if linkname is not None:
                            info.linkname = linkname
                        tf.addfile(info)
                    dest = os.path.join(root, "dest")
                    os.makedirs(dest)
                    with _tar.open(tar_path) as tf:
                        with self.assertRaises(_tar.TarError):
                            anc_lib._safe_tar_extract(tf, dest)

    def test_rejects_write_through_preexisting_symlink(self):
        # dest_dir is created with exist_ok=True, so it may already contain a
        # symlink. A regular member "logs/pwn.txt" whose parent "logs" is a
        # pre-existing symlink out of dest must be rejected (realpath resolves
        # the link); a lexical check would let it write outside.
        import tarfile as _tar
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            outside = os.path.join(root, "outside")
            os.makedirs(outside)
            dest = os.path.join(root, "dest")
            os.makedirs(dest)
            os.symlink(outside, os.path.join(dest, "logs"))  # dest/logs -> /outside

            payload = os.path.join(root, "payload")
            with open(payload, "w") as fh:
                fh.write("x")
            tar_path = os.path.join(root, "a.tar")
            with _tar.open(tar_path, "w") as tf:
                tf.add(payload, arcname="logs/pwn.txt")

            with _tar.open(tar_path) as tf:
                with self.assertRaises(_tar.TarError):
                    anc_lib._safe_tar_extract(tf, dest)
            self.assertFalse(os.path.exists(os.path.join(outside, "pwn.txt")))


if __name__ == "__main__":
    unittest.main()
