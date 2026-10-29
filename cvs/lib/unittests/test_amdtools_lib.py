'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Tests for shared AMD tool helpers.
'''

import io
import os
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cvs.lib import amdtools_lib


class TestVersionsAndConfig(unittest.TestCase):
    def test_version_api_and_url_names(self):
        self.assertTrue(amdtools_lib.is_strict_version("1.8.1-rc.2"))
        self.assertFalse(amdtools_lib.is_strict_version("1.8.1-RC.2"))
        self.assertEqual(amdtools_lib.parse_version("arc 1.8.1-rc.2"), ((1, 8, 1), 2))
        self.assertEqual(amdtools_lib.compare_versions("1.8.1", "1.8.0"), 1)
        self.assertTrue(amdtools_lib.version_satisfies("1.8.1", "1.8.1-rc.2"))
        for name in (
            "arc-tool-heliosr-1.8.1.tar.gz",
            "arc-tool-heliosr_1.8.1_amd64.deb",
            "arc-tool-heliosr-1.8.1-1.x86_64.rpm",
        ):
            self.assertEqual(amdtools_lib.parse_version_from_url(name), "1.8.1")

    def test_flavours_and_tool_label(self):
        for suffix, pkg in ((".tar.gz", "tar"), (".deb", "deb"), (".rpm", "rpm")):
            self.assertEqual(
                amdtools_lib.detect_package_flavour("arc-1.8.1" + suffix, "ARC"), amdtools_lib.PackageFlavour(pkg, True)
            )
        with self.assertRaisesRegex(ValueError, "ARC"):
            amdtools_lib.detect_package_flavour("arc.bin", "ARC")
        with self.assertRaisesRegex(ValueError, "ARC config value"):
            amdtools_lib.assert_shell_safe({"url": "bad$"}, ("url",), "ARC")
        self.assertIn("ARC", amdtools_lib.validate_cluster_username({"username": "a b"}, "ARC"))

    def test_shell_snippets(self):
        self.assertIn("ARC download progress", amdtools_lib.download_with_progress_snippet("url", "dest", "ARC"))
        self.assertIn("ARC install: prefix", amdtools_lib.sudo_prefix_snippet("/opt/amdtools", "ARC"))
        expected = (
            "echo 'Determining top-level tool folders...'; "
            "tops=$( tar -tzf anc.tar.gz | sed 's#^\\./##' | awk -F/ '{print $1}' | grep -vE '^([.]{1,2})?$' | sort -u ); "
            "$SUDO mkdir -p '/opt/amdtools'; "
            "echo 'Replacing top-level folders under /opt/amdtools...'; "
            'for name in $tops; do case "$name" in */*|.|..) continue;; esac; '
            'echo "  /opt/amdtools/$name"; $SUDO rm -rf \'/opt/amdtools\'/"$name"; done; '
            "echo 'Extracting archive to /opt/amdtools...'; "
            "$SUDO tar -xzf anc.tar.gz -C '/opt/amdtools'; "
        )
        self.assertEqual(amdtools_lib.replace_tops_and_extract_snippet("anc.tar.gz", "/opt/amdtools"), expected)

    def test_unreachable_and_labels(self):
        with patch.object(amdtools_lib, "fail_test") as fail:
            self.assertEqual(amdtools_lib.fail_unreachable_nodes(["a", "b"], {"a": "ok"}, "install", "ARC"), ["b"])
            fail.assert_called_once()
            self.assertIn("ARC install failed on b", fail.call_args.args[0])
        self.assertEqual(amdtools_lib.remote_user_tmp("a/b"), "/tmp/a_b")
        self.assertEqual(
            amdtools_lib.node_label_from_file({"node_dict": {"host": {"vpc_ip": "ip"}}}, "host"), "ip_host"
        )
        self.assertTrue(amdtools_lib.as_bool("True"))
        self.assertFalse(amdtools_lib.as_bool(None, default=False))

    def test_run_path(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("cvs.core.run_layout.RunLayout.get", return_value=SimpleNamespace(run_dir=tmp)),
        ):
            self.assertEqual(
                amdtools_lib.resolve_run_log_folder("arc_logs/<node>/<test_name>/<timestamp>", "test_x", "ts", "host"),
                os.path.join(tmp, "arc_logs", "host", "test_x", "ts"),
            )


class TestLogsAndReports(unittest.TestCase):
    def test_pull_remote_dir_and_missing_download(self):
        class FakeOrch:
            def __init__(self, missing=False):
                self.calls = []
                self.missing = missing

            def exec(self, cmd, **kwargs):
                self.calls.append((cmd, kwargs))
                return {"host": ""}

            def download_file(self, remote, local, **kwargs):
                self.calls.append((remote, kwargs))
                if self.missing:
                    return {}
                with tarfile.open(local, "w:gz") as tf:
                    data = b"summary"
                    info = tarfile.TarInfo("session/ARC/summary.json")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
                return {"host": local}

        with tempfile.TemporaryDirectory() as tmp:
            orch = FakeOrch()
            root, reason = amdtools_lib.pull_remote_dir(orch, "host", "user", "/logs/session", tmp, "ARC")
            self.assertIsNone(reason)
            self.assertTrue(os.path.isfile(os.path.join(root, "ARC", "summary.json")))
            self.assertTrue(any("/arc_session.tar.gz" in cmd for cmd, _ in orch.calls))
            self.assertTrue(all(kwargs.get("hosts") == ["host"] for _, kwargs in orch.calls))
            root, reason = amdtools_lib.pull_remote_dir(
                FakeOrch(missing=True), "host", "user", "/logs/session", tmp, "ARC"
            )
            self.assertIsNone(root)
            self.assertIn("not created", reason)

    def test_safe_extract_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "bad.tar.gz")
            with tarfile.open(archive, "w:gz") as tf:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                tf.addfile(info, io.BytesIO(b"x"))
            with tarfile.open(archive) as tf, self.assertRaises(tarfile.TarError):
                amdtools_lib.safe_tar_extract(tf, os.path.join(tmp, "dest"))

    def test_links_status_and_tarball(self):
        request = SimpleNamespace(node=SimpleNamespace(name="test"))
        with patch.dict(
            "sys.modules", {"pytest_html": SimpleNamespace(extras=SimpleNamespace(url=lambda href, name: (href, name)))}
        ):
            amdtools_lib.stash_report_link(request, "file.html", "ARC logs", "_arc_html_extras")
        self.assertEqual(request.node._arc_html_extras, [("file.html", "ARC logs")])
        self.assertEqual(
            [amdtools_lib.deck_status(x) for x in (None, "not available on the remote system", "bad")],
            ["pass", "na", "fail"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            logs = os.path.join(tmp, "logs")
            os.mkdir(logs)
            with open(os.path.join(logs, "summary.json"), "w") as fh:
                fh.write("{}")
            mgr = Mock()
            mgr.add_html_to_report.return_value = "/reports/tar.gz"
            result = SimpleNamespace(dest_dir=logs, label="host", reason="failed")
            with patch.object(amdtools_lib, "stash_report_link"):
                hrefs = amdtools_lib.attach_node_log_tarballs(
                    request, mgr, [result], "test_x", "ts", tmp, "ARC", "_arc_html_extras"
                )
            self.assertEqual(hrefs, {"host": "tar.gz"})
            self.assertTrue(os.path.isfile(os.path.join(tmp, "host_test_x_ts_arc_logs.tar.gz")))
