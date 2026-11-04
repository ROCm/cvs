#
# Copyright 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Unit tests for build_tools/security_scan.py. Exercises every rule against both
# the blocked shapes and the intentionally-allowed ones, so the allowlists do
# not silently widen into a leak and the leak checks do not regress into noise.

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from build_tools import security_scan  # noqa: E402


def _rules(text):
    return {rule for rule, _ in security_scan.scan_line(text)}


class TestAmdHostRule(unittest.TestCase):
    def test_public_docs_hosts_allowed(self):
        for url in (
            "https://rocm.docs.amd.com/projects/cvs/en/latest/",
            "https://rocmdocs.amd.com/",
            "https://rocm-handbook.amd.com/",
            "https://instinct.docs.amd.com/",
            "https://www.amd.com/en/resources/product-security.html",
            "https://developer.amd.com/resources/",
            "https://repo.amd.com/rocm/rvs/tarball/",
            "https://stable.repo.amd.com/rocm/",
            "https://nightly.repo.amd.com/rocm/",
            "https://rocm.prereleases.amd.com/",
            "https://rocm.nightlies.amd.com/",
            "https://rocm.blogs.amd.com/",
            "see https://amd.com/cvs/rundeck/profile.schema.json",
        ):
            self.assertNotIn("internal-amd-host", _rules(url), url)

    def test_internal_amd_hosts_blocked(self):
        for url in (
            "https://chi-metrics-exp-mon.amd.com",
            "https://atlartifactory.amd.com:8443/artifactory/HW-ANCRelease",
            "host = internal-build.amd.com",
            "https://jenkins.ci.amd.com/job/x",
        ):
            self.assertIn("internal-amd-host", _rules(url), url)

    def test_non_amd_url_ignored(self):
        self.assertEqual(_rules("https://example.com/path"), set())


class TestHttpsPortRule(unittest.TestCase):
    def test_standard_port_allowed(self):
        self.assertNotIn("exposed-https-port", _rules("https://host.example.com:443/x"))

    def test_loopback_and_private_allowed(self):
        for url in (
            "https://127.0.0.1:6443",
            "https://localhost:8080",
            "https://10.0.0.5:2379",
            "https://192.168.1.10:9090",
            "https://172.16.0.1:10257",
        ):
            self.assertNotIn("exposed-https-port", _rules(url), url)

    def test_placeholder_host_allowed(self):
        self.assertNotIn("exposed-https-port", _rules("https://<grafana-host>:3000"))

    def test_routable_nonstandard_port_blocked(self):
        for url in (
            "https://grafana.example.com:3000",
            "https://1.2.3.4:8443",
            "https://metrics.internal.net:9443",
        ):
            self.assertIn("exposed-https-port", _rules(url), url)

    def test_ipv6_routable_nonstandard_port_blocked(self):
        for url in (
            "https://[2001:db8::1]:9443",
            "https://[2606:4700:4700::1111]:8443",
        ):
            self.assertIn("exposed-https-port", _rules(url), url)

    def test_ipv6_loopback_and_private_allowed(self):
        for url in (
            "https://[::1]:8443",
            "https://[fe80::1]:8080",
            "https://[fc00::1]:9000",
            "https://[2001:db8::1]:443",
        ):
            self.assertNotIn("exposed-https-port", _rules(url), url)

    def test_userinfo_does_not_hide_port(self):
        # A URL with userinfo (user@ or user:pass@) must still be checked.
        for url in (
            "https://alice@metrics.example.net:9443/x",
            "https://user:pass@grafana.example.com:3000",
            "https://alice@[2001:db8::1]:9443/x",
        ):
            self.assertIn("exposed-https-port", _rules(url), url)


class TestRealClusterHostnameRule(unittest.TestCase):
    def test_real_hostnames_blocked(self):
        # Every example the user listed must be caught.
        for host in (
            "ctheliosr-1b114-g07-1",
            "ctheliosr-rck-g02-j17-7",
            "ctheliosm-rck-g02-h11-01",
            "heliosp-1b114-c06-3",
            "heliosr-1b114-b01-1",
            "heliosr-1b114-c01-4",
            "heliosr-1b114-c03-4",
        ):
            self.assertIn("real-cluster-hostname", _rules("node: " + host), host)

    def test_real_hostname_blocked_inside_url_and_path(self):
        self.assertIn("real-cluster-hostname", _rules("ssh ashmishr@heliosr-1b114-b01-1.rck.dcgpu"))
        self.assertIn("real-cluster-hostname", _rules("/scratch/heliosr-2b805-d5-3/out.log"))

    def test_family_names_allowed(self):
        # The user's explicit valid list.
        for name in ("heliosr", "heliosr-rck", "heliosm-rck", "heliosp"):
            self.assertNotIn("real-cluster-hostname", _rules('cluster = "' + name + '"'), name)

    def test_family_bare_words_allowed(self):
        for text in (
            'meta = {"cluster": "helios"}',
            "# follows ARC development across heliosr, heliosp, and heliosm.",
            "arc-tool-heliosr-1.8.1.tar.gz",
            "arc-tool-heliosr_1.8.1_amd64.deb",
            '"heliosr_mfg_l11",',
        ):
            self.assertNotIn("real-cluster-hostname", _rules(text), text)

    def test_rck_family_without_coordinates_allowed(self):
        self.assertNotIn("real-cluster-hostname", _rules('profile = "heliosm-rck"'))


class TestSecretRules(unittest.TestCase):
    def test_known_token_shapes_blocked(self):
        samples = (
            "github-token",
            "huggingface-token",
            "aws-access-key",
            "slack-token",
        )
        lines = {
            "github-token": "token = ghp_" + "a" * 36,
            "huggingface-token": "HF_TOKEN=hf_" + "b" * 34,
            "aws-access-key": "key = AKIAIOSFODNN7EXAMPLE",
            "slack-token": "slack = xoxb-123456789012-abcdefghijkl",
        }
        for rule in samples:
            self.assertIn(rule, _rules(lines[rule]), rule)

    def test_private_key_header_blocked(self):
        # Assemble the PEM header so the verbatim literal is not in this file;
        # GitHub push-protection flags the contiguous string even in a fixture.
        header = "-----BEGIN RSA " + "PRIVATE" + " KEY-----"
        self.assertIn("private-key", _rules(header))

    def test_generic_hardcoded_secret_blocked(self):
        self.assertIn("hardcoded-secret", _rules('password = "s3cr3tP@ssw0rd99"'))
        self.assertIn("hardcoded-secret", _rules("api_key: 'A1b2C3d4E5f6G7h8'"))

    def test_quoted_credential_key_blocked(self):
        # JSON / dict form: the key itself is quoted. Must detect the same as
        # the bare assignment above.
        for text in (
            '"password": "s3cr3tP@ssw0rd99"',
            "'api_key': 'A1b2C3d4E5f6G7h8'",
            '{"client_secret": "Zx9Qw8Er7Ty6Ui5"}',
        ):
            self.assertIn("hardcoded-secret", _rules(text), text)

    def test_placeholder_first_does_not_hide_later_secret(self):
        # A leading placeholder assignment must not shadow a real secret later
        # on the same line.
        text = '{"password": "<changeme>", "api_key": "A1b2C3d4E5f6G7h8"}'
        self.assertIn("hardcoded-secret", _rules(text), text)

    def test_angle_bracket_in_value_still_flagged(self):
        # Angle brackets are valid password characters; a secret containing one
        # must not be excused as a placeholder.
        for text in (
            'password = "s3cr3tP@ssw0rd99<"',
            'password = "s3cr3t>P@ssw0rd9"',
            "api_key = 'A1b2<C3d4E5f6G7'",
        ):
            self.assertIn("hardcoded-secret", _rules(text), text)

    def test_complete_placeholder_tokens_allowed(self):
        # A value that IS a <token>/${token} placeholder stays allowed.
        for text in (
            'password = "<your-password>"',
            'password = "${DB_PASSWORD}"',
        ):
            self.assertNotIn("hardcoded-secret", _rules(text), text)

    def test_placeholders_and_env_lookups_allowed(self):
        for text in (
            'password = "<changeme>"',
            'token = os.environ["HF_TOKEN"]',
            'secret = os.getenv("CLIENT_SECRET")',
            'api_key = "your_api_key_here"',
            'password = "example-password"',
            'passwd = "********"',
        ):
            self.assertNotIn("hardcoded-secret", _rules(text), text)
            self.assertEqual(
                [r for r in _rules(text) if r.endswith("token") or r == "hardcoded-secret"],
                [],
                text,
            )


class TestPragma(unittest.TestCase):
    def test_pragma_suppresses_line(self):
        text = "url = https://chi-metrics-exp-mon.amd.com  # cvs-security-allow"
        self.assertEqual(security_scan.scan_line(text), [])


class TestDiffScanning(unittest.TestCase):
    def _diff(self, path, added):
        body = "".join("+" + line + "\n" for line in added)
        return "diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -0,0 +1,{n} @@\n".format(p=path, n=len(added)) + body

    def test_added_line_flagged_with_path_and_lineno(self):
        diff = self._diff("cvs/foo.py", ["ok = 1", "host = heliosr-1b114-b01-1"])
        results = security_scan.scan_diff(diff)
        self.assertEqual(len(results), 1)
        path, lineno, rule, _msg = results[0]
        self.assertEqual(path, "cvs/foo.py")
        self.assertEqual(lineno, 2)
        self.assertEqual(rule, "real-cluster-hostname")

    def test_removed_lines_ignored(self):
        diff = (
            "diff --git a/cvs/foo.py b/cvs/foo.py\n"
            "--- a/cvs/foo.py\n"
            "+++ b/cvs/foo.py\n"
            "@@ -1,1 +0,0 @@\n"
            "-host = heliosr-1b114-b01-1\n"
        )
        self.assertEqual(security_scan.scan_diff(diff), [])

    def test_scanner_own_files_exempt(self):
        diff = self._diff("build_tools/security_scan.py", ["host = heliosr-1b114-b01-1"])
        self.assertEqual(security_scan.scan_diff(diff), [])
        diff2 = self._diff("build_tools/unittests/test_security_scan.py", ["host = heliosr-1b114-b01-1"])
        self.assertEqual(security_scan.scan_diff(diff2), [])

    def test_same_basename_elsewhere_not_exempt(self):
        # A file that merely shares the basename must still be scanned.
        diff = self._diff("cvs/elsewhere/security_scan.py", ["host = heliosr-1b114-b01-1"])
        results = security_scan.scan_diff(diff)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][2], "real-cluster-hostname")

    def test_header_shaped_added_content_scanned(self):
        # An added line whose content starts with "++ " (raw "+++ ...") must be
        # treated as content, not a file header -- otherwise it could switch the
        # path to an exempt file and hide the real leak on the next line.
        diff = (
            "diff --git a/real.py b/real.py\n"
            "--- a/real.py\n"
            "+++ b/real.py\n"
            "@@ -0,0 +1,2 @@\n"
            "++ build_tools/security_scan.py\n"
            "+host = heliosr-1b114-b01-1\n"
        )
        results = security_scan.scan_diff(diff)
        rules = [r for _, _, r, _ in results]
        self.assertIn("real-cluster-hostname", rules)
        # The real file path must be preserved, not switched to the exempt one.
        self.assertEqual(results[0][0], "real.py")

    def test_no_newline_marker_not_counted(self):
        # Replacing an unterminated final line: the "\ No newline at end of
        # file" marker is diff metadata, not a new-side line. It must not
        # consume the hunk budget, or the replacement addition is skipped.
        diff = (
            "diff --git a/f b/f\n"
            "--- a/f\n"
            "+++ b/f\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "\\ No newline at end of file\n"
            "+host = heliosr-1b114-b01-1\n"
            "\\ No newline at end of file\n"
        )
        results = security_scan.scan_diff(diff)
        self.assertEqual([r for _, _, r, _ in results], ["real-cluster-hostname"])

    def test_clean_diff_passes(self):
        diff = self._diff(
            "docs/x.rst",
            ["See https://rocm.docs.amd.com/projects/cvs/ for details.", 'cluster = "heliosr"'],
        )
        self.assertEqual(security_scan.scan_diff(diff), [])

    def test_colored_diff_still_scanned(self):
        # A diff produced with color.ui=always carries ANSI escapes before the
        # +/- marker; the scanner must strip them and still flag added lines.
        esc = "\x1b[32m"
        reset = "\x1b[m"
        diff = (
            "diff --git a/cvs/foo.py b/cvs/foo.py\n"
            "--- a/cvs/foo.py\n"
            "{e}+++ b/cvs/foo.py{r}\n"
            "@@ -0,0 +1 @@\n"
            "{e}+host = heliosr-1b114-b01-1{r}\n"
        ).format(e=esc, r=reset)
        results = security_scan.scan_diff(diff)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][2], "real-cluster-hostname")


if __name__ == "__main__":
    unittest.main()
