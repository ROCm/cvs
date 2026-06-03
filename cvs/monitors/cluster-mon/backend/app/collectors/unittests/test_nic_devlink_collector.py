"""
Contract tests for NICDevlinkCollector.
"""

import json
import unittest

from app.unittests.testing import FakeSshManager
from app.collectors.nic_devlink_collector import NICDevlinkCollector


class TestNICDevlinkCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collect_devlink_info_normalizes_vendor_and_fw(self):
        devlink = json.dumps(
            {
                "info": {
                    "pci/0000:76:00.0": {
                        "driver": "bnxt_en",
                        "serial_number": "SER123",
                        "board.serial_number": "BSER456",
                        "versions": {
                            "fixed": {"board.id": "BID", "fw.psid": "PSID", "asic.id": "AID", "asic.rev": "A1"},
                            "running": {"fw": "1.2.3", "fw.mgmt": "9.9"},
                        },
                    }
                }
            }
        )
        ssh = FakeSshManager(["node1"], command_map={"devlink dev info --json": {"node1": devlink}})

        result = await NICDevlinkCollector().collect_devlink_info(ssh)

        dev = result["node1"]["pci/0000:76:00.0"]
        self.assertEqual(dev["pci_address"], "0000:76:00.0")
        self.assertEqual(dev["driver"], "bnxt_en")
        self.assertEqual(dev["vendor"], "Broadcom Thor2")
        self.assertEqual(dev["fw_version"], "1.2.3")
        self.assertEqual(dev["fw_psid"], "PSID")
        self.assertEqual(dev["fw_mgmt"], "9.9")
        self.assertEqual(dev["board_id"], "BID")

    async def test_collect_devlink_info_empty_object(self):
        ssh = FakeSshManager(["node1"], command_map={"devlink dev info --json": {"node1": "{}"}})
        result = await NICDevlinkCollector().collect_devlink_info(ssh)
        self.assertEqual(result["node1"], {})

    async def test_collect_devlink_info_error_marker(self):
        ssh = FakeSshManager(
            ["node1"], command_map={"devlink dev info --json": {"node1": "ABORT: Host Unreachable Error"}}
        )
        result = await NICDevlinkCollector().collect_devlink_info(ssh)
        self.assertEqual(result["node1"], {"error": "ABORT: Host Unreachable Error"})


if __name__ == "__main__":
    unittest.main()
