"""Unit tests for Primus auto_setup preflight helper."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.primus_setup import (
    build_primus_clone_or_update_command,
    build_primus_venv_install_command,
    build_wait_for_shared_primus_command,
    parse_setup_output,
    _venv_root_from_activate,
)


class TestPrimusSetupCommands(unittest.TestCase):
    def test_clone_command_uses_branch_single_branch(self):
        cmd = build_primus_clone_or_update_command(
            primus_dir="/home/user/Primus",
            git_url="https://github.com/AMD-AIG-AIMA/Primus.git",
            git_branch="dev/preflight-direct-test",
            recurse_submodules=False,
        )
        self.assertIn("--branch dev/preflight-direct-test", cmd)
        self.assertIn("--single-branch", cmd)
        self.assertNotIn("--recurse-submodules", cmd)
        self.assertIn("git fetch origin", cmd)

    def test_clone_with_submodules_when_enabled(self):
        cmd = build_primus_clone_or_update_command(
            primus_dir="/home/user/Primus",
            git_url="https://github.com/AMD-AIG-AIMA/Primus.git",
            git_branch="dev/preflight-direct-test",
            recurse_submodules=True,
        )
        self.assertIn("--recurse-submodules", cmd)

    def test_force_reclone_removes_existing(self):
        cmd = build_primus_clone_or_update_command(
            primus_dir="/home/user/Primus",
            git_url="https://github.com/AMD-AIG-AIMA/Primus.git",
            git_branch="dev/preflight-direct-test",
            force_reclone=True,
        )
        self.assertIn("rm -rf", cmd)

    def test_clone_removes_broken_partial_directory(self):
        cmd = build_primus_clone_or_update_command(
            primus_dir="/home/user/Primus",
            git_url="https://github.com/AMD-AIG-AIMA/Primus.git",
            git_branch="dev/preflight-direct-test",
        )
        self.assertIn("[ ! -d /home/user/Primus/.git ]", cmd)
        self.assertIn("rm -rf /home/user/Primus", cmd)

    def test_wait_for_shared_primus_polls(self):
        cmd = build_wait_for_shared_primus_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            max_wait=60,
        )
        self.assertIn("runner/primus-cli", cmd)
        self.assertIn("import torch", cmd)
        self.assertIn("while [ $i -lt 12 ]", cmd)
        self.assertIn("CVS_PRIMUS_SETUP_OK", cmd)
        self.assertNotIn("bash -c", cmd)

    def test_venv_minimal_installs_torch_not_editable(self):
        activate = "/home/user/envs/preflight/.venv/bin/activate"
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate=activate,
            pip_install_mode="minimal",
        )
        self.assertEqual(_venv_root_from_activate(activate), "/home/user/envs/preflight/.venv")
        self.assertIn("python3 -m venv", cmd)
        self.assertIn("pip install torch", cmd)
        self.assertNotIn("pip install -e .", cmd)
        self.assertIn("runner/primus-cli", cmd)
        self.assertIn("import torch", cmd)

    def test_pathspec_error_is_git_not_pip(self):
        parsed = parse_setup_output("error: pathspec 'dev/preflight-direct-test' did not match any file(s) known to git\n")
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("git", parsed["errors"][0])

    def test_pip_error_not_classified_as_git_error(self):
        parsed = parse_setup_output(
            "Already on 'dev/preflight-direct-test'\n"
            "ERROR: file:///home/user%40example.com/Primus does not appear to be a Python project\n"
        )
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("pip", parsed["errors"][0])


class TestParseSetupOutput(unittest.TestCase):
    def test_git_fatal_fails(self):
        parsed = parse_setup_output("Cloning...\nfatal: repository not found\n")
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("git", parsed["errors"][0])

    def test_git_lock_config_suggests_shared_install(self):
        parsed = parse_setup_output(
            "error: could not lock config file /home/user/Primus/.git/config: No such file or directory\n"
            "fatal: could not set 'core.repositoryformatversion' to '0'\n"
        )
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("shared_install", parsed["errors"][0])

    def test_bash_lock_redirect_error_fails(self):
        parsed = parse_setup_output(
            "bash: line 1: /home/user/Primus/.cvs_primus_setup.lock: No such file or directory\n"
        )
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("shell error", parsed["errors"][0])

    def test_clean_output_passes_with_marker(self):
        parsed = parse_setup_output("Successfully installed torch\nCVS_PRIMUS_SETUP_OK\n")
        self.assertEqual(parsed["status"], "PASS")

    def test_output_without_marker_fails(self):
        parsed = parse_setup_output("Successfully installed torch\n")
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("did not report success", parsed["errors"][0])

    def test_empty_output_fails(self):
        parsed = parse_setup_output("")
        self.assertEqual(parsed["status"], "FAIL")
        self.assertIn("empty setup output", parsed["errors"][0])


if __name__ == "__main__":
    unittest.main()
