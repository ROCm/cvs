"""Unit tests for Primus auto_setup preflight helper."""

import os
import shlex
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.primus_setup import (
    PrimusSetup,
    build_primus_clone_or_update_command,
    build_primus_venv_install_command,
    build_wait_for_shared_primus_command,
    parse_setup_output,
    rocm_major_minor_from_index_url,
    _resolve_setting_from_sections,
    _torch_hip_check_script,
    _torch_hip_match_shell,
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
        self.assertNotIn('if ! python -c "import torch"', cmd)

    def test_minimal_install_replaces_importable_incompatible_torch(self):
        index = "https://download.pytorch.org/whl/rocm7.2"
        activate = "/home/user/envs/preflight/.venv/bin/activate"
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate=activate,
            pip_install_mode="minimal",
            torch_pip_index_url=index,
        )
        self.assertEqual(rocm_major_minor_from_index_url(index), "7.2")
        self.assertIn(shlex.quote(_torch_hip_check_script("7.2")), cmd)
        self.assertIn("pip uninstall -y torch", cmd)
        self.assertLess(cmd.index("if !"), cmd.index("pip uninstall -y torch"))
        self.assertNotIn('if ! python -c "import torch"', cmd)
        subprocess.run(["bash", "-n", "-c", cmd], check=True)

    def test_index_without_rocm_token_always_reinstalls(self):
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            pip_install_mode="minimal",
            torch_pip_index_url="https://example.invalid/simple",
        )
        self.assertIn("pip uninstall -y torch", cmd)
        self.assertNotIn("if !", cmd)
        self.assertNotIn("torch.version", cmd)

    def test_skip_mode_does_not_reinstall_torch(self):
        cmd = build_primus_venv_install_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            pip_install_mode="skip",
            torch_pip_index_url="https://download.pytorch.org/whl/rocm7.2",
        )
        self.assertNotIn("pip uninstall", cmd)
        self.assertNotIn("pip install torch", cmd)

    def test_wait_requires_matching_hip_when_index_encodes_rocm(self):
        cmd = build_wait_for_shared_primus_command(
            primus_dir="/home/user/Primus",
            venv_activate="/home/user/envs/preflight/.venv/bin/activate",
            max_wait=60,
            torch_pip_index_url="https://download.pytorch.org/whl/rocm7.2",
        )
        self.assertIn(shlex.quote(_torch_hip_check_script("7.2")), cmd)
        self.assertNotIn('python -c "import torch"', cmd)
        self.assertNotIn("bash -c", cmd)
        subprocess.run(["bash", "-n", "-c", cmd], check=True)

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


class TestTorchHipCheckScript(unittest.TestCase):
    def _run(self, hip=None, *, block_torch=False):
        script = _torch_hip_check_script("7.2")
        if block_torch:
            prelude = (
                "import builtins\n"
                "_real = builtins.__import__\n"
                "def _block(name, *args, **kwargs):\n"
                "    if name == 'torch' or name.startswith('torch.'):\n"
                "        raise ModuleNotFoundError(name)\n"
                "    return _real(name, *args, **kwargs)\n"
                "builtins.__import__ = _block\n"
            )
        else:
            prelude = (
                "import sys, types\n"
                "mod = types.ModuleType('torch')\n"
                f"mod.version = types.SimpleNamespace(hip={hip!r})\n"
                "sys.modules['torch'] = mod\n"
            )
        proc = subprocess.run(
            [sys.executable, "-c", prelude + script],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode

    def test_importable_rocm64_wheel_does_not_match_rocm72_index(self):
        self.assertNotEqual(self._run("6.4.43484-aaa"), 0)

    def test_matching_hip_major_minor_is_accepted(self):
        self.assertEqual(self._run("7.2.25424"), 0)

    def test_cpu_or_cuda_wheel_with_no_hip_is_replaced(self):
        self.assertNotEqual(self._run(None), 0)

    def test_missing_torch_fails_the_check(self):
        self.assertNotEqual(self._run(block_torch=True), 0)

    def test_shell_check_rejects_importable_incompatible_wheel(self):
        self.assertNotEqual(self._shell_check("6.4.43484-aaa"), 0)
        self.assertEqual(self._shell_check("7.2.25424"), 0)
        self.assertNotEqual(self._shell_check(None), 0)

    def _shell_check(self, hip):
        with tempfile.TemporaryDirectory() as tmp:
            bindir = os.path.join(tmp, "bin")
            os.makedirs(bindir)
            python_path = os.path.join(bindir, "python")
            with open(python_path, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        f"""\
                        #!{sys.executable}
                        import sys
                        import types
                        mod = types.ModuleType("torch")
                        mod.version = types.SimpleNamespace(hip={hip!r})
                        sys.modules["torch"] = mod
                        if len(sys.argv) > 2 and sys.argv[1] == "-c":
                            exec(sys.argv[2])
                        """
                    )
                )
            os.chmod(python_path, os.stat(python_path).st_mode | stat.S_IEXEC)
            cmd = _torch_hip_match_shell(shlex.quote(python_path), "7.2")
            proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
        return proc.returncode


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
