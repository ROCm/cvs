'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import logging
import tempfile
import unittest
from pathlib import Path

from cvs.core.agent.logger import disable_rank_log, enable_rank_log, rank_log_enabled
from cvs.lib.globals import set_verbosity, verbose_log


class TestRankLog(unittest.TestCase):
    def tearDown(self):
        disable_rank_log()
        set_verbosity(0)

    def test_writes_debug_while_root_is_warning(self):
        set_verbosity(1)
        root = logging.getLogger()
        original_level = root.level
        root.setLevel(logging.WARNING)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "rank0.log"
        enable_rank_log(path)
        try:
            self.assertTrue(rank_log_enabled())
            verbose_log(logging.getLogger("cvs.test.verbose_log"), "startup line", 1)
            self.assertIn("startup line", path.read_text(encoding="utf-8"))
        finally:
            root.setLevel(original_level)

    def test_disable_closes_the_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        enable_rank_log(Path(tmp.name) / "rank1.log")
        self.assertTrue(rank_log_enabled())
        disable_rank_log()
        self.assertFalse(rank_log_enabled())
