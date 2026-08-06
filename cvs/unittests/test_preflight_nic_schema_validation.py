import unittest

from pydantic import ValidationError

from cvs.parsers.schemas import PreflightNicDriverVersionConfig, PreflightNicFirmwareConfig


class TestPreflightNicDriverVersionSchema(unittest.TestCase):
    def test_default_nic_type_is_broadcom(self):
        parsed = PreflightNicDriverVersionConfig()
        self.assertEqual(parsed.nic_type, ["broadcom"])

    def test_unknown_vendor_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PreflightNicDriverVersionConfig(nic_type=["broadcomm"])
        self.assertIn("unknown vendor(s)", str(ctx.exception))

    def test_duplicate_vendor_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PreflightNicDriverVersionConfig(nic_type=["ainic", "ainic"])
        self.assertIn("duplicate vendor names", str(ctx.exception))

    def test_empty_nic_type_rejected_when_enabled(self):
        with self.assertRaises(ValidationError) as ctx:
            PreflightNicDriverVersionConfig(enabled=True, nic_type=[])
        self.assertIn("must not be empty when enabled is true", str(ctx.exception))

    def test_empty_nic_type_allowed_when_disabled(self):
        parsed = PreflightNicDriverVersionConfig(enabled=False, nic_type=[])
        self.assertEqual(parsed.nic_type, [])

    def test_vendor_sub_block_rejects_unknown_key(self):
        with self.assertRaises(ValidationError):
            PreflightNicDriverVersionConfig(broadcom={"unknown_field": "x"})

    def test_top_level_rejects_unknown_key(self):
        with self.assertRaises(ValidationError):
            PreflightNicDriverVersionConfig(unknown_field="x")


class TestPreflightNicFirmwareSchema(unittest.TestCase):
    def test_default_nic_type_is_ainic(self):
        parsed = PreflightNicFirmwareConfig()
        self.assertEqual(parsed.nic_type, ["ainic"])

    def test_unknown_vendor_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PreflightNicFirmwareConfig(nic_type=["cisco"])
        self.assertIn("unknown vendor(s)", str(ctx.exception))

    def test_duplicate_vendor_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            PreflightNicFirmwareConfig(nic_type=["mellanox", "mellanox"])
        self.assertIn("duplicate vendor names", str(ctx.exception))

    def test_empty_nic_type_rejected_when_enabled(self):
        with self.assertRaises(ValidationError) as ctx:
            PreflightNicFirmwareConfig(enabled=True, nic_type=[])
        self.assertIn("must not be empty when enabled is true", str(ctx.exception))

    def test_empty_nic_type_allowed_when_disabled(self):
        parsed = PreflightNicFirmwareConfig(enabled=False, nic_type=[])
        self.assertEqual(parsed.nic_type, [])

    def test_vendor_sub_block_rejects_unknown_key(self):
        with self.assertRaises(ValidationError):
            PreflightNicFirmwareConfig(ainic={"unknown_field": "x"})

    def test_top_level_rejects_unknown_key(self):
        with self.assertRaises(ValidationError):
            PreflightNicFirmwareConfig(unknown_field="x")

    def test_multi_vendor_nic_type_accepted(self):
        parsed = PreflightNicFirmwareConfig(nic_type=["ainic", "broadcom", "mellanox"])
        self.assertEqual(parsed.nic_type, ["ainic", "broadcom", "mellanox"])


if __name__ == "__main__":
    unittest.main()
