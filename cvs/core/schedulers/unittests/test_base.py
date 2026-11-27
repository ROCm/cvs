'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/schedulers/base.py: env_int, which every backend uses for
# rank and world size, returns the integer or raises RuntimeError naming the variable.

import os
import unittest
from unittest.mock import patch

from cvs.core.schedulers.base import env_int


class TestEnvInt(unittest.TestCase):
    def test_reads_integer_values(self):
        for value, expected in (("0", 0), ("7", 7), ("-1", -1)):
            with self.subTest(value=value), patch.dict(os.environ, {"CVS_TEST_INT": value}, clear=True):
                self.assertEqual(env_int("CVS_TEST_INT"), expected)

    def test_unset_empty_or_non_integer_names_the_variable(self):
        # A KeyError or ValueError escaping here would reach the user as a bare
        # traceback that names neither the variable nor the managed run.
        for env in ({}, {"CVS_TEST_INT": ""}, {"CVS_TEST_INT": "two"}, {"CVS_TEST_INT": "1.5"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "CVS_TEST_INT"):
                    env_int("CVS_TEST_INT")


if __name__ == "__main__":
    unittest.main()
