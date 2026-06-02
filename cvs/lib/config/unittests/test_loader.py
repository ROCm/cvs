"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Unit tests for cvs/lib/config/loader.py: parse_config dispatch + placeholder
# resolution. Generic over every framework via iter_bases() (env refs are
# injected into container.env, a field every base shares).
#
# Pinned invariants:
#   - Dispatch routes each framework to its registered class and fails closed on
#     an unknown framework or any unexpected top-level key (extra="forbid").
#   - B3 env resolution: an unset ${env:VAR} raises, a deliberately-empty one
#     ("") is allowed, and an embedded (non-whole-value) ${env:...} is rejected
#     so a missing credential never silently becomes "".

import os
import unittest

from cvs.lib.config import parse_config
from cvs.lib.config.loader import ConfigError

from ._fixtures import iter_bases, make_base


class TestConfigDispatch(unittest.TestCase):
    def test_parses_and_dispatches(self):
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                cfg = parse_config(base)
                self.assertEqual(cfg.framework, framework)

    def test_rejects_unknown_framework(self):
        with self.assertRaises(ConfigError):
            parse_config({**make_base(), "framework": "vlm"})

    def test_rejects_non_string_framework_as_configerror(self):
        # A non-str framework must fail closed as ConfigError. An unhashable
        # value (list/dict) previously leaked a raw TypeError from the registry
        # dict.get(); int/bool can never match a registered key.
        for bad in ([], {}, 1, True):
            with self.subTest(framework=bad):
                with self.assertRaises(ConfigError):
                    parse_config({**make_base(), "framework": bad})

    def test_rejects_extra_key(self):
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                with self.assertRaises(ConfigError):
                    parse_config({**base, "bogus": 1})


class TestLoaderEnv(unittest.TestCase):
    def test_b3_unset_env_raises(self):
        os.environ.pop("DTNI_TEST_MISSING", None)
        self.addCleanup(os.environ.pop, "DTNI_TEST_MISSING", None)
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                cfg = {**base, "container": {"env": {"HF_TOKEN": "${env:DTNI_TEST_MISSING}"}}}
                with self.assertRaises(ConfigError):
                    parse_config(cfg)

    def test_b3_empty_env_allowed(self):
        os.environ["DTNI_TEST_EMPTY"] = ""
        self.addCleanup(os.environ.pop, "DTNI_TEST_EMPTY", None)
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                cfg = parse_config({**base, "container": {"env": {"HF_TOKEN": "${env:DTNI_TEST_EMPTY}"}}})
                self.assertEqual(cfg.container.env["HF_TOKEN"], "")

    def test_b3_embedded_env_ref_rejected(self):
        os.environ["DTNI_TEST_SET"] = "x"
        self.addCleanup(os.environ.pop, "DTNI_TEST_SET", None)
        for framework, base in iter_bases():
            with self.subTest(framework=framework):
                cfg = {**base, "container": {"env": {"HF_TOKEN": "run ${env:DTNI_TEST_SET}.sh"}}}
                with self.assertRaises(ConfigError):
                    parse_config(cfg)


if __name__ == "__main__":
    unittest.main()
