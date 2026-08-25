'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.atom.atom_mtp_quality.
'''

import json
import unittest

from cvs.lib.inference.atom.atom_mtp_quality import (
    chat_template_ok,
    chat_template_sha256,
    degenerate_decode_ratio,
    extract_completion_text,
    parse_mtp_log_metrics,
)


class TestAtomMtpQuality(unittest.TestCase):
    def test_parse_mtp_log_metrics_finds_acceptance_rate(self):
        log = "INFO draft acceptance rate=0.8123 speculative tokens avg"
        out = parse_mtp_log_metrics(log)
        self.assertAlmostEqual(out["mtp.acceptance_rate"], 0.8123)

    def test_degenerate_decode_ratio_empty_is_one(self):
        self.assertEqual(degenerate_decode_ratio(""), 1.0)
        self.assertEqual(degenerate_decode_ratio("abc"), 1.0)

    def test_degenerate_decode_ratio_repeat_is_one(self):
        self.assertEqual(degenerate_decode_ratio("hellohellohellohellohello"), 1.0)

    def test_degenerate_decode_ratio_normal_is_zero(self):
        self.assertEqual(
            degenerate_decode_ratio("The model served a reasonable completion for the probe."),
            0.0,
        )

    def test_chat_template_sha256_stable(self):
        h1 = chat_template_sha256("hello")
        h2 = chat_template_sha256("hello")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, chat_template_sha256("world"))

    def test_chat_template_ok_without_expected_returns_none(self):
        self.assertIsNone(chat_template_ok("anything", ""))

    def test_chat_template_ok_matches_expected_hash(self):
        text = "probe response"
        expected = chat_template_sha256(text)
        self.assertEqual(chat_template_ok(text, expected), 1.0)
        self.assertEqual(chat_template_ok("other", expected), 0.0)

    def test_extract_completion_text_from_chat_json(self):
        payload = {
            "choices": [{"message": {"content": "Hi there"}}],
        }
        self.assertEqual(extract_completion_text(json.dumps(payload)), "Hi there")


if __name__ == "__main__":
    unittest.main()
