'''Unit tests for host inventory normalization.'''

import unittest

from cvs.lib.platform.host_inventory import (
    collected_fact_keys,
    parse_gpu_count,
    parse_iommu,
    parse_kernel_version,
    parse_os_release,
    parse_pci_realloc,
    parse_pcie_link,
    parse_rocm_version,
    record_gpu_firmware,
    record_indexed_node_facts,
    record_node_facts,
)


class TestHostInventory(unittest.TestCase):
    def test_parses_scalar_command_outputs(self):
        self.assertEqual(parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\n'), "Ubuntu 24.04.1 LTS")
        self.assertEqual(parse_kernel_version("Linux node-a 6.8.0-60-generic #63 SMP"), "6.8.0-60-generic")
        self.assertEqual(parse_rocm_version("ROCm version: 7.0.2\nAMDSMI version: 26.0"), "7.0.2")
        self.assertEqual(parse_pci_realloc("BOOT_IMAGE=/vmlinuz pci=realloc=off iommu=pt"), "off")
        self.assertEqual(parse_iommu("BOOT_IMAGE=/vmlinuz pci=realloc=off iommu=pt"), "pt")
        self.assertEqual(parse_pcie_link("LnkSta: Speed 32GT/s, Width x16"), "32GT/s \u00b7 x16")
        self.assertEqual(
            parse_gpu_count(
                "01:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] "
                "VGA compatible controller\n"
                "02:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI]"
            ),
            "1",
        )

    def test_records_node_and_indexed_facts(self):
        results = {}
        record_node_facts(results, "bios", {"node-b": "B2\n", "node-a": "A1\n"})
        record_indexed_node_facts(
            results,
            "gpu_pcie",
            "card0",
            {"node-a": "LnkSta: Speed 32GT/s, Width x16"},
            parse_pcie_link,
        )

        self.assertEqual(results["nodes"]["node-a"]["bios"], "A1")
        self.assertEqual(results["nodes"]["node-a"]["gpu_pcie"]["card0"], "32GT/s \u00b7 x16")

    def test_records_gpu_firmware(self):
        results = {}
        record_gpu_firmware(
            results,
            {
                "node-a": [
                    {
                        "gpu": 0,
                        "fw_list": [
                            {"fw_id": "CP_MEC1", "fw_version": "32945"},
                            {"fw_id": "RLC", "fw_version": "65"},
                        ],
                    }
                ]
            },
        )

        self.assertEqual(results["firmware"]["node-a"]["0"]["CP_MEC1"], "32945")
        self.assertEqual(collected_fact_keys(results), ["firmware"])


if __name__ == "__main__":
    unittest.main()
