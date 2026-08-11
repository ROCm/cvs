'''Unit tests for platform_config schema.'''

import unittest

from cvs.lib.inference.utils.platform_config import PlatformConfig


class TestPlatformConfig(unittest.TestCase):
    def test_defaults_dmesg_scan_false(self):
        cfg = PlatformConfig()
        self.assertFalse(cfg.dmesg_scan)


if __name__ == "__main__":
    unittest.main()
