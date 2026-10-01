"""Unit tests for Primus auto_setup preflight helper."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.primus_setup import (
    PrimusSetup,
    build_primus_clone_or_update_command,
    build_primus_venv_install_command,
    build_wait_for_shared_primus_command,
    _hip_matches_torch_index,
    parse_setup_output,
    _resolve_setting_from_sections,
    _torch_compat_python,
    _venv_root_from_activate,
)
from cvs.lib.preflight.node_smoke import (
    LEGACY_NODE_SMOKE_SECTION,
    NODE_SMOKE_TIER1_SECTION,
)


class TestPrimusSetupConfigResolution(unittest.TestCase):
    def test_resolve_setting_from_tier1_section(self):
        cfg = {
            "node_smoke_tier1": {
                "primus_dir": "/home/user/Primus",
                "venv_activate": "/home/user/envs/preflight/.venv/bin/activate",
            }
        }
        setup = PrimusSetup(None, ["node1"], cfg)
        self.assertEqual(setup.primus_dir, "/home/user/Primus")
        self.assertEqual(setup.venv_activate, "/home/user/envs/preflight/.venv/bin/activate")

    def test_resolve_setting_falls_back_to_legacy_node_smoke(self):
        cfg = {
            "node_smoke": {
                "primus_dir": "/legacy/Primus",
                "venv_activate": "/legacy/.venv/bin/activate",
            }
        }
        setup = PrimusSetup(None, ["node1"], cfg)
        self.assertEqual(setup.primus_dir, "/legacy/Primus")
        self.assertEqual(setup.venv_activate, "/legacy/.venv/bin/activate")

    def test_tier1_wins_over_legacy_when_both_present(self):
        cfg = {
            "node_smoke_tier1": {"primus_dir": "/tier1/Primus"},
            "node_smoke": {"primus_dir": "/legacy/Primus"},
        }
        value = _resolve_setting_from_sections(
            cfg,
            (NODE_SMOKE_TIER1_SECTION, LEGACY_NODE_SMOKE_SECTION),
            "primus_dir",
            "",
        )
        self.assertEqual(value, "/tier1/Primus")


class TestHipMatchesTorchIndex(unittest.TestCase):
    def test_cpu_or_cuda_wheel_does_not_match(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        self.assertFalse(_hip_matches_torch_index(None, index))
        self.assertFalse(_hip_matches_torch_index("", index))

    def test_older_rocm_wheel_does_not_match_new_index(self):
        self.assertFalse(_hip_matches_torch_index("6.2.41134", "https://download.pytorch.org/whl/rocm7.2"))

    def test_matching_hip_minor(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        self.assertTrue(_hip_matches_torch_index("7.2.0", index))
        self.assertTrue(_hip_matches_torch_index("7.2.26015", index))
        self.assertTrue(_hip_matches_torch_index("7.2+git", index))

    def test_minor_is_not_a_prefix(self):
        self.assertFalse(_hip_matches_torch_index("7.10.0", "https://download.pytorch.org/whl/rocm7.1"))
        self.assertFalse(_hip_matches_torch_index("7.1.0", "https://download.pytorch.org/whl/rocm7.10"))

    def test_unversioned_index_accepts_any_hip_build(self):
        self.assertTrue(_hip_matches_torch_index("6.2.0", "https://example.invalid/torch"))
        self.assertFalse(_hip_matches_torch_index(None, "https://example.invalid/torch"))

    def test_shell_probe_agrees_with_helper(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        for hip in (None, "", "6.2.41134", "7.2.0", "7.2.26015", "7.10.1", "7.2+git"):
            for match_index in (True, False):
                self.assertEqual(
                    _probe_exit_code(hip, index, match_index_version=match_index) == 0,
                    _hip_matches_torch_index(hip, index, match_index_version=match_index),
                    f"hip={hip!r} match_index={match_index}",
                )

    def test_hip_only_accepts_older_rocm_wheel(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        self.assertTrue(_hip_matches_torch_index("6.2.41134", index, match_index_version=False))
        self.assertFalse(_hip_matches_torch_index(None, index, match_index_version=False))


def _shared_setup_commands(pip_install_mode):
    """Commands PrimusSetup sends when two nodes share one venv."""
    phdl = MagicMock()
    phdl.reachable_hosts = ["nodeA", "nodeB"]
    phdl.exec_cmd_list.return_value = {
        "nodeA": "CVS_PRIMUS_SETUP_OK\n",
        "nodeB": "CVS_PRIMUS_SETUP_OK\n",
    }
    orch = MagicMock()
    orch.all = phdl
    cfg = {
        "node_smoke_tier1": {
            "primus_dir": "/home/user/Primus",
            "venv_activate": "/home/user/envs/preflight/.venv/bin/activate",
            "pip_install_mode": pip_install_mode,
            "torch_pip_index_url": "https://download.pytorch.org/whl/rocm7.2",
        }
    }
    PrimusSetup(orch, ["nodeA", "nodeB"], cfg).run()
    return phdl.exec_cmd_list.call_args[0][0]


def _probe_exit_code(hip, index, match_index_version=True):
    import types

    snippet = _torch_compat_python(index, match_index_version=match_index_version).replace('\\"', '"')
    torch_mod = types.ModuleType("torch")
    torch_mod.version = types.SimpleNamespace(hip=hip)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_mod
    try:
        try:
            exec(snippet, {"__name__": "__main__"})
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


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
        self.assertIn('parts[0]==\\"7\\"', cmd)
        self.assertIn('parts[1].split(\\"+\\")[0]==\\"2\\"', cmd)
        self.assertNotIn('python -c \\"import torch\\"', cmd)
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
        self.assertIn("pip install --upgrade --force-reinstall torch", cmd)
        self.assertIn("https://download.pytorch.org/whl/rocm7.2", cmd)
        self.assertIn('parts[0]==\\"7\\"', cmd)
        self.assertIn('parts[1].split(\\"+\\")[0]==\\"2\\"', cmd)
        self.assertNotIn("pip install -e .", cmd)
        self.assertIn("runner/primus-cli", cmd)

    def test_venv_reinstalls_when_index_rocm_version_differs(self):
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            pip_install_mode="minimal",
            torch_pip_index_url="https://download.pytorch.org/whl/rocm6.2",
        )
        self.assertIn("--force-reinstall", cmd)
        self.assertIn('parts[0]==\\"6\\"', cmd)
        self.assertIn('parts[1].split(\\"+\\")[0]==\\"2\\"', cmd)
        self.assertNotIn('parts[0]==\\"7\\"', cmd)

    def test_skip_mode_checks_hip_build_not_index_version(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            pip_install_mode="skip",
            torch_pip_index_url=index,
        )
        probe = _torch_compat_python(index, match_index_version=False)
        self.assertIn(probe, cmd)
        self.assertNotIn("--force-reinstall", cmd)
        self.assertNotIn('parts[0]==\\"7\\"', cmd)
        self.assertEqual(_probe_exit_code("6.2.41134", index, match_index_version=False), 0)
        self.assertNotEqual(_probe_exit_code(None, index, match_index_version=False), 0)

    def test_requirements_mode_checks_hip_build_not_index_version(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            pip_install_mode="requirements",
            torch_pip_index_url=index,
        )
        probe = _torch_compat_python(index, match_index_version=False)
        self.assertIn("pip install -r requirements.txt", cmd)
        self.assertIn(probe, cmd)
        self.assertNotIn("--force-reinstall", cmd)
        self.assertNotIn("--index-url", cmd)
        self.assertNotIn('parts[0]==\\"7\\"', cmd)
        self.assertEqual(_probe_exit_code("6.2.41134", index, match_index_version=False), 0)
        self.assertNotEqual(_probe_exit_code("", index, match_index_version=False), 0)

    def test_shared_followers_do_not_pin_index_outside_minimal_mode(self):
        hip_only = _torch_compat_python("", match_index_version=True)
        for mode in ("skip", "requirements", "SKIP", " Requirements "):
            commands = _shared_setup_commands(mode)
            follower = commands[1]
            self.assertIn(hip_only, follower, mode)
            self.assertNotIn('parts[0]==\\"7\\"', follower, mode)
            self.assertNotIn("rocm7.2", follower, mode)
        self.assertEqual(_probe_exit_code("6.2.41134", ""), 0)
        self.assertNotEqual(_probe_exit_code(None, ""), 0)

    def test_shared_followers_pin_index_in_minimal_mode(self):
        commands = _shared_setup_commands("minimal")
        follower = commands[1]
        self.assertIn('parts[0]==\\"7\\"', follower)
        self.assertIn('parts[1].split(\\"+\\")[0]==\\"2\\"', follower)
        self.assertNotEqual(_probe_exit_code("6.2.41134", "https://download.pytorch.org/whl/rocm7.2"), 0)

    def test_wait_uses_same_hip_probe_as_install(self):
        index = "https://download.pytorch.org/whl/rocm7.1"
        install = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            torch_pip_index_url=index,
        )
        wait = build_wait_for_shared_primus_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            torch_pip_index_url=index,
        )
        probe = _torch_compat_python(index)
        self.assertIn(probe, install)
        self.assertIn(probe, wait)

    def test_pathspec_error_is_git_not_pip(self):
        parsed = parse_setup_output(
            "error: pathspec 'dev/preflight-direct-test' did not match any file(s) known to git\n"
        )
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
