'''Unit tests for host inventory normalization.'''

import unittest

from cvs.lib.platform.host_inventory import (
    collected_fact_keys,
    normalize_output,
    parse_gpu_count,
    parse_iommu,
    parse_kernel_version,
    parse_numa_balancing,
    parse_online_memory,
    parse_os_release,
    parse_pci_acs,
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

    def test_failed_firmware_collection_is_not_a_fact_category(self):
        results = {}
        record_gpu_firmware(results, {"n0": "sudo: a password is required", "n1": []})
        self.assertNotIn("firmware", results)
        self.assertEqual(collected_fact_keys(results), [])

    def test_unparseable_remote_output_is_unavailable(self):
        self.assertEqual(
            normalize_output("ssh: connect to host 10.0.3.14 port 22: Connection timed out"), "unavailable"
        )
        self.assertEqual(
            parse_kernel_version("Permission denied (publickey,password) for root@10.0.3.14"), "unavailable"
        )
        self.assertEqual(parse_online_memory("Total online memory: 1.3T"), "1.3T")
        self.assertEqual(parse_numa_balancing("kernel.numa_balancing = 0"), "0")
        self.assertEqual(parse_pci_acs("ACSCtl: SrcValid+"), "enabled")
        self.assertEqual(parse_pci_acs(""), "disabled")


if __name__ == "__main__":
    unittest.main()
