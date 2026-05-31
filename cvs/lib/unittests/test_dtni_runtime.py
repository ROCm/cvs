"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

import unittest

from cvs.lib.runtime.container_handle import ContainerHandle


class TestContainerHandle(unittest.TestCase):
    def test_non_privileged_default(self):
        # The default-emitted run command carries no privilege escalation and is
        # tagged with the run-id label that scoped teardown later filters on.
        cmd = ContainerHandle("img:tag", "run1", runner=None).build_run_command()
        self.assertNotIn("--privileged", cmd)
        self.assertNotIn("seccomp=unconfined", cmd)
        self.assertIn("cvs_run_id=run1", cmd)

    def test_opt_in_privileged(self):
        # Escalation flags are emitted only when the caller explicitly opts in.
        cmd = ContainerHandle("img", "r", runner=None, privileged=True, seccomp_unconfined=True).build_run_command()
        self.assertIn("--privileged", cmd)
        self.assertIn("seccomp=unconfined", cmd)

    def test_injection_chars_quoted_in_run_command(self):
        # Every interpolated field is shell-quoted at the build boundary, so an
        # injected `; rm -rf /` / `$(...)` stays inside one argument instead of
        # running as a second command on the host. Cover several field kinds so a
        # per-site quoting regression (not just the volume path) is caught.
        cmd = ContainerHandle(
            "img;rm -rf /",
            "run1",
            runner=None,
            network="$(reboot)",
            volumes={"/a b; rm -rf /": "/data"},
            ports={"80; rm": "90 x"},
            env={"K; rm": "v; rm -rf /"},
        ).build_run_command()
        self.assertIn("--network '$(reboot)'", cmd)
        self.assertIn("-v '/a b; rm -rf /':/data", cmd)
        self.assertIn("-p '80; rm':'90 x'", cmd)
        self.assertIn("-e 'K; rm'='v; rm -rf /'", cmd)
        self.assertIn("'img;rm -rf /'", cmd)


class FakeRunner:
    """Records every command it is handed and returns a canned result."""

    def __init__(self, output=""):
        self.cmds = []
        self.output = output

    def exec(self, cmd, timeout=None):
        self.cmds.append(cmd)
        return self.output


class TestContainerHandleRuntime(unittest.TestCase):
    def test_enter_exit_captures_then_removes(self):
        # Happy path: launch, then on exit forensics are captured before the
        # label-scoped removal (teardown order matters for diagnostics).
        r = FakeRunner()
        with ContainerHandle("img", "run1", r, readiness=lambda h: True):
            pass
        self.assertTrue(any("docker run -d" in c for c in r.cmds))
        log_idx = next(i for i, c in enumerate(r.cmds) if "docker logs" in c)
        rm_idx = next(i for i, c in enumerate(r.cmds) if "docker rm -f" in c)
        self.assertLess(log_idx, rm_idx)

    def test_exit_removes_and_does_not_suppress_body_exception(self):
        # An exception in the with-body still triggers teardown and propagates
        # (the context manager must not swallow it).
        r = FakeRunner()
        with self.assertRaises(ValueError):
            with ContainerHandle("img", "run1", r, readiness=lambda h: True):
                raise ValueError("boom")
        self.assertTrue(any("docker rm -f" in c for c in r.cmds))

    def test_enter_readiness_timeout_tears_down(self):
        # Regression: __enter__ raising on a readiness timeout must still capture
        # and remove the already-started container (else it leaks, since __exit__
        # never runs when __enter__ raises).
        r = FakeRunner()
        h = ContainerHandle(
            "img",
            "run1",
            r,
            readiness=lambda h: False,
            readiness_timeout_s=0.01,
            readiness_interval_s=0.001,
        )
        with self.assertRaises(TimeoutError):
            with h:
                pass
        self.assertTrue(any("docker run -d" in c for c in r.cmds))
        self.assertTrue(any("docker logs" in c for c in r.cmds))
        self.assertTrue(any("docker rm -f" in c for c in r.cmds))

    def test_remove_is_label_scoped_not_prune(self):
        # Teardown removes strictly by the run-id label and never prunes.
        r = FakeRunner()
        ContainerHandle("img", "run1", r).remove()
        removed = [c for c in r.cmds if "docker rm -f" in c]
        self.assertTrue(removed)
        self.assertIn("--filter label=cvs_run_id=run1", removed[0])
        self.assertNotIn("prune", removed[0])

    def test_run_flattens_per_host_dict(self):
        # A per-host dict result (Pssh) is flattened to one newline-joined string.
        r = FakeRunner(output={"hostA": "out-a", "hostB": "out-b"})
        self.assertEqual(ContainerHandle("img", "run1", r)._run("cmd"), "out-a\nout-b")


if __name__ == "__main__":
    unittest.main()
