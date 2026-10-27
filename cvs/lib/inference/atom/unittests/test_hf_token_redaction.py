'''Unit tests for ATOM Hugging Face token log redaction.'''

import io
import logging
import unittest

from cvs.lib.inference.atom.hf_token_redaction import (
    HfTokenLogFilter,
    install_hf_token_log_redaction,
    redact_hf_token_text,
)


class TestRedactHfTokenText(unittest.TestCase):
    def test_redacts_assignment_and_keeps_token_file_path(self):
        secret = "hf_supersecretvalue"
        text = (
            f"cmd = export HF_TOKEN={secret} path=/home/user/.hf_token "
            f"{{'HF_TOKEN': '{secret}', 'hf_token_file': '/home/user/.hf_token'}}"
        )
        redacted = redact_hf_token_text(text)
        self.assertNotIn(secret, redacted)
        self.assertIn("/home/user/.hf_token", redacted)
        self.assertIn("<redacted>", redacted)

    def test_empty_and_non_string_pass_through(self):
        self.assertEqual(redact_hf_token_text(""), "")
        self.assertIsNone(redact_hf_token_text(None))

    def test_leaves_hf_transfer_package_name(self):
        text = "No module named 'hf_transfer'"
        self.assertEqual(redact_hf_token_text(text), text)

    def test_redacts_long_hf_value(self):
        raw_value = "hf_" + ("a" * 34)
        redacted = redact_hf_token_text(f"saw {raw_value} in /home/user/.hf_token")
        self.assertNotIn(raw_value, redacted)
        self.assertIn("/home/user/.hf_token", redacted)
        self.assertIn("<redacted>", redacted)


class TestHfTokenLogFilter(unittest.TestCase):
    def test_log_filter_redacts_formatted_message(self):
        # Not named "secret": CodeQL treats that name as sensitive data written to a log.
        raw_value = "hf_supersecretvalue"
        logger = logging.getLogger("atom-hf-token-redaction-test")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addFilter(HfTokenLogFilter())
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            logger.info("cmd = export HF_TOKEN=%s path=%s", raw_value, "/home/user/.hf_token")
        finally:
            logger.removeHandler(handler)
            logger.removeFilter(logger.filters[-1])
        text = stream.getvalue()
        self.assertNotIn(raw_value, text)
        self.assertIn("/home/user/.hf_token", text)
        self.assertIn("<redacted>", text)

    def test_install_attaches_filter_to_root_and_handlers(self):
        root = logging.getLogger()
        handler = logging.StreamHandler(io.StringIO())
        root.addHandler(handler)
        try:
            install_hf_token_log_redaction()
            install_hf_token_log_redaction()
            self.assertEqual(sum(isinstance(item, HfTokenLogFilter) for item in root.filters), 1)
            self.assertEqual(sum(isinstance(item, HfTokenLogFilter) for item in handler.filters), 1)
        finally:
            root.removeHandler(handler)
            root.filters[:] = [item for item in root.filters if not isinstance(item, HfTokenLogFilter)]


if __name__ == "__main__":
    unittest.main()
