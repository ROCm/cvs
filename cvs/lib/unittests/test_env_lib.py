"""
test_env_lib.py

Unit tests for env_lib.build_env_prefix and env_lib.redact_secrets using Python's
built-in unittest framework.
"""

import shlex
import unittest
from cvs.lib.env_lib import build_env_prefix, redact_secrets

FAKE_CRED = "fake-cred"
TMP = "/tmp"


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


class TestRedactSecrets(unittest.TestCase):
    def test_masks_token_assignment(self):
        self.assertEqual(redact_secrets('export HF_TOKEN=fake-cred'), 'export HF_TOKEN=<redacted>')

    def test_leaves_token_file_path(self):
        text = 'export HF_TOKEN_FILE=/home/user/.hf_token'
        self.assertEqual(redact_secrets(text), text)

    def test_empty_text(self):
        self.assertEqual(redact_secrets(''), '')

    def assertRedacted(self, text, expected):
        result = redact_secrets(text)
        self.assertEqual(result, expected)
        self.assertNotIn(FAKE_CRED, result)

    def test_masks_shell_quoted_values(self):
        nested = f"export SECRET={shlex.quote(FAKE_CRED + ' x')}; run"
        for text, expected in (
            (build_env_prefix({"DB_PASSWORD": f"{FAKE_CRED} x"}) + " ; run", "export DB_PASSWORD=<redacted> ; run"),
            (f'export API_KEY="{FAKE_CRED} x"; run', "export API_KEY=<redacted>; run"),
            (f"bash -c {shlex.quote(nested)}", f"bash -c {shlex.quote('export SECRET=<redacted>; run')}"),
        ):
            with self.subTest(text=text):
                self.assertRedacted(text, expected)

    def test_masks_values_containing_quote_characters(self):
        # shlex.quote splits a value containing ' into adjacent quoted fragments; the secret sits after the split.
        value = f"x'\"{FAKE_CRED} y"
        nested = f"export SECRET={shlex.quote(value)}; run"
        for text, expected in (
            (build_env_prefix({"HF_TOKEN": value}) + " ; run", "export HF_TOKEN=<redacted> ; run"),
            (f'export API_KEY="x\\"{FAKE_CRED} y"; run', "export API_KEY=<redacted>; run"),
            (f"run --hf-token {shlex.quote(value)} --y", "run --hf-token <redacted> --y"),
            (f"run --token={shlex.quote(value)} --y", "run --token=<redacted> --y"),
            (f"docker run -e {shlex.quote('HF_TOKEN=' + value)} img", "docker run -e 'HF_TOKEN=<redacted>' img"),
            (f"bash -c {shlex.quote(nested)}", f"bash -c {shlex.quote('export SECRET=<redacted>; run')}"),
        ):
            with self.subTest(text=text):
                self.assertRedacted(text, expected)

    def test_masks_vllm_env_script_written_through_docker_exec(self):
        env_script = f"export HF_TOKEN={FAKE_CRED}\nexport HF_HUB_CACHE=/models\n"
        inner = f"printf '%s' {shlex.quote(env_script)} > {TMP}/server_env_script.sh"
        result = redact_secrets(f"docker exec c bash -c {shlex.quote(inner)}")
        self.assertNotIn(FAKE_CRED, result)
        self.assertIn("export HF_TOKEN=<redacted>\nexport HF_HUB_CACHE=/models\n", result)

    def test_masks_escaped_ansi_c_and_newline_separated_values(self):
        for text, expected in (
            (f"run --token {FAKE_CRED}\\ x --y", "run --token <redacted> --y"),
            (f"export TOKEN=$'{FAKE_CRED} x'; run", "export TOKEN=<redacted>; run"),
            (f"run --token\n{FAKE_CRED}", "run --token\n<redacted>"),
            (f"run --token \\\n{FAKE_CRED}", "run --token \\\n<redacted>"),
        ):
            with self.subTest(text=text):
                self.assertRedacted(text, expected)

    def test_leaves_non_secrets_unchanged(self):
        for text in (
            "export TOKENIZERS_PARALLELISM=false ; export MAX_NUM_BATCHED_TOKENS=8192",
            "export HF_TOKEN_PATH=/run/secrets/hf ; hf_token_file=/home/u/.hf_token password_file=/p",
            "cat f | docker login --username u --password-stdin",
            "vllm serve m --tokenizer-mode auto --max-tokens 5",
        ):
            with self.subTest(text=text):
                self.assertEqual(redact_secrets(text), text)

    def test_long_identifiers_scan_in_linear_time(self):
        # A nested-quantifier name regex backtracks quadratically here and never finishes.
        name = "TOKEN_" * 100000 + "X"
        self.assertRedacted(f"{name}={FAKE_CRED} ; {name}_PATH=/p", f"{name}=<redacted> ; {name}_PATH=/p")


if __name__ == "__main__":
    unittest.main()
