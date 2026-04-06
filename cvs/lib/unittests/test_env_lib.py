"""
test_env_lib.py

Unit tests for env_lib.build_env_prefix using Python's built-in unittest framework.
"""

import unittest
from env_lib import build_env_prefix


class TestBuildEnvPrefix(unittest.TestCase):
    def test_empty_env_vars(self):
        self.assertEqual(build_env_prefix({}), "")

    def test_literal_env_var(self):
        env = {"FOO": "bar"}
        result = build_env_prefix(env)
        self.assertEqual(result, "export FOO=bar")

    def test_literal_env_var_with_spaces(self):
        env = {"FOO": "hello world"}
        result = build_env_prefix(env)
        self.assertEqual(result, "export FOO='hello world'")

    def test_path_prepend(self):
        env = {"PATH": "/usr/bin:/custom/bin:$PATH"}
        result = build_env_prefix(env)
        self.assertEqual(result, "export PATH=/usr/bin:/custom/bin:$PATH")

    def test_path_append(self):
        env = {"PATH": "$PATH:/custom/bin"}
        result = build_env_prefix(env)
        self.assertEqual(result, "export PATH=$PATH:/custom/bin")

    def test_ld_library_path_prepend(self):
        env = {"LD_LIBRARY_PATH": "/opt/lib:$LD_LIBRARY_PATH"}
        result = build_env_prefix(env)
        self.assertEqual(result, "export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH")

    def test_multiple_env_vars_mixed(self):
        env = {
            "PATH": "/usr/bin:$PATH",
            "FOO": "bar",
            "BAZ": "hello world",
        }
        result = build_env_prefix(env)

        self.assertEqual(result, "export PATH=/usr/bin:$PATH ; export FOO=bar ; export BAZ='hello world'")

    def test_cross_variable_expansion_is_not_allowed(self):
        env = {"FOO": "$PATH"}
        result = build_env_prefix(env)

        # Expansion should be blocked and treated as a literal
        self.assertEqual(result, "export FOO='$PATH'")

    def test_shell_injection_attempt_is_quoted(self):
        env = {"FOO": "$(rm -rf /)"}
        result = build_env_prefix(env)

        # Must be fully quoted to prevent execution
        self.assertEqual(result, "export FOO='$(rm -rf /)'")


if __name__ == "__main__":
    unittest.main()
