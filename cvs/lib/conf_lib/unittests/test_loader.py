'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.conf_lib.loader.
'''

import unittest
from pathlib import Path

from cvs.lib.conf_lib.loader import ConfigError


class TestConfigError(unittest.TestCase):
    def test_the_message_names_the_file_then_one_problem_per_indented_line(self):
        problems = ["server_params.tp: Field required", "cells: Extra inputs are not permitted"]
        error = ConfigError(Path("/configs/vllm.json"), problems)
        self.assertEqual(
            str(error),
            "/configs/vllm.json is not a valid config:\n"
            "  server_params.tp: Field required\n"
            "  cells: Extra inputs are not permitted",
        )
        self.assertIs(error.problems, problems)

    def test_it_is_a_value_error_and_an_empty_problem_list_still_names_the_file(self):
        error = ConfigError("vllm.json", [])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "vllm.json is not a valid config:\n")


if __name__ == "__main__":
    unittest.main()
