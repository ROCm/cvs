"""Unit tests for preflight configuration schema (config_file/preflight/config.py)."""

import unittest
import warnings

from pydantic import ValidationError

from cvs.schema.config_file.preflight.config import (
    PreflightConfigFile,
    normalize_legacy_preflight_node_smoke_sections,
    normalize_legacy_preflight_rdma_config,
)


class TestPreflightRdmaConfigSchema(unittest.TestCase):
    def test_legacy_rdma_inventory_is_normalized_with_one_warning(self):
        legacy = {
            "node_check": {
                "expected_rocm_version": "7.15.0",
                "gid_index": "7",
                "rdma_interfaces": ["enp4s0np0"],
            },
            "connectivity_check": {"rdma": {"connectivity_mode": "basic"}},
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = PreflightConfigFile.model_validate(legacy)

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, FutureWarning)
        self.assertIn("legacy paths will be removed in a future release", str(caught[0].message))
        self.assertEqual(config.connectivity_check.rdma.gid_index, "7")
        self.assertEqual(config.connectivity_check.rdma.interfaces, ["enp4s0np0"])
        self.assertNotIn("gid_index", config.node_check.model_dump())
        self.assertNotIn("rdma_interfaces", config.node_check.model_dump())

    def test_matching_legacy_and_canonical_rdma_values_are_accepted(self):
        with self.assertWarns(FutureWarning):
            config = PreflightConfigFile.model_validate(
                {
                    "node_check": {
                        "gid_index": "7",
                        "rdma_interfaces": ["enp4s0np0"],
                    },
                    "connectivity_check": {
                        "rdma": {
                            "gid_index": "7",
                            "interfaces": ["enp4s0np0"],
                        }
                    },
                }
            )

        self.assertEqual(config.connectivity_check.rdma.gid_index, "7")
        self.assertEqual(config.connectivity_check.rdma.interfaces, ["enp4s0np0"])

    def test_conflicting_legacy_and_canonical_rdma_values_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Conflicting preflight RDMA options"):
            PreflightConfigFile.model_validate(
                {
                    "node_check": {"gid_index": "3"},
                    "connectivity_check": {"rdma": {"gid_index": "7"}},
                }
            )

        with self.assertRaisesRegex(ValidationError, "Conflicting preflight RDMA options"):
            PreflightConfigFile.model_validate(
                {
                    "node_check": {"rdma_interfaces": ["enp4s0np0"]},
                    "connectivity_check": {"rdma": {"interfaces": ["mlx5_0"]}},
                }
            )

    def test_invalid_legacy_interface_value_uses_canonical_validation(self):
        with self.assertWarns(FutureWarning):
            with self.assertRaises(ValidationError):
                PreflightConfigFile.model_validate(
                    {"node_check": {"rdma_interfaces": "enp4s0np0"}},
                )

    def test_canonical_rdma_inventory_under_connectivity_check(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = PreflightConfigFile.model_validate(
                {
                    "connectivity_check": {
                        "rdma": {
                            "connectivity_mode": "skip",
                            "gid_index": "7",
                            "interfaces": ["enp4s0np0"],
                        }
                    },
                    "reporting": {
                        "generate_html_report": True,
                        "generate_rdma_pairs_csv": False,
                    },
                }
            )

        self.assertEqual(caught, [])
        self.assertEqual(config.connectivity_check.rdma.gid_index, "7")
        self.assertEqual(config.connectivity_check.rdma.interfaces, ["enp4s0np0"])


class TestNodeHealthConfigSchema(unittest.TestCase):
    def test_documentation_pseudo_fields_stripped_typos_rejected(self):
        config = PreflightConfigFile.model_validate(
            {
                "_comment": "Preflight settings",
                "node_check": {
                    "_comment": "Node checks",
                    "_example_gpus_per_node": 8,
                    "enabled": True,
                    "gpus_per_node": 4,
                    "expected_rocm_version": "7.15.0",
                },
                "connectivity_check": {
                    "ifoe": {
                        "_comment": "IFoE checks",
                        "l2ping": {
                            "_comment_enabled": "Enable strict L2 validation",
                            "enabled": True,
                        },
                    }
                },
            }
        )

        self.assertEqual(config.node_check.gpus_per_node, 4)
        self.assertTrue(config.connectivity_check.ifoe.l2ping.enabled)
        self.assertNotIn("_comment", config.node_check.model_extra or {})

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {"node_check": {"enabled": True, "gpus_per_nod": 4}},
            )

    def test_node_check_accepts_only_documented_customer_fields(self):
        config = PreflightConfigFile.model_validate(
            {
                "node_check": {
                    "enabled": True,
                    "gpus_per_node": 4,
                    "expected_rocm_version": "7.15.0",
                },
                "connectivity_check": {"ifoe": {"fabric_checks": True}},
            }
        )

        self.assertTrue(config.node_check.enabled)
        self.assertEqual(config.node_check.gpus_per_node, 4)
        self.assertTrue(config.connectivity_check.ifoe.fabric_checks)

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "node_check": {
                        "enabled": True,
                        "gpus_per_node": 4,
                        "failure_mode": "report",
                    }
                }
            )

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "node_health": {
                        "enabled": True,
                        "gpus_per_node": 4,
                        "fabric_checks": True,
                    }
                }
            )


class TestTransferBenchConfigSchema(unittest.TestCase):
    def test_accepts_only_six_customer_facing_options(self):
        config = PreflightConfigFile.model_validate(
            {
                "connectivity_check": {
                    "ifoe": {
                        "transferbench": {
                            "enabled": True,
                            "scope": "cluster",
                            "profile": "smoketest",
                            "message_sizes": ["1K", "16M"],
                            "iterations": 3,
                            "warmup_iterations": 1,
                        }
                    }
                }
            }
        )

        transferbench = config.connectivity_check.ifoe.transferbench
        self.assertTrue(transferbench.enabled)
        self.assertEqual(transferbench.scope, "cluster")

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "connectivity_check": {
                        "ifoe": {
                            "transferbench": {
                                "enabled": True,
                                "scope": "node",
                                "profile": "bandwidth",
                            }
                        }
                    }
                }
            )

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "connectivity_check": {
                        "ifoe": {
                            "transferbench": {
                                "enabled": True,
                                "tb_binary": "/custom/TransferBench",
                            }
                        }
                    }
                }
            )

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate({"transferbench": {"enabled": True}})


class TestL2PingConfigSchema(unittest.TestCase):
    def test_accepts_only_two_customer_facing_options(self):
        config = PreflightConfigFile.model_validate(
            {
                "connectivity_check": {
                    "ifoe": {
                        "l2ping": {
                            "enabled": True,
                            "pings_per_port": 5,
                        }
                    }
                }
            }
        )

        self.assertTrue(config.connectivity_check.ifoe.l2ping.enabled)
        self.assertEqual(config.connectivity_check.ifoe.l2ping.pings_per_port, 5)

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "connectivity_check": {
                        "ifoe": {
                            "l2ping": {
                                "enabled": True,
                                "pings_per_port": 3,
                                "loss_threshold_pct": 1.0,
                            }
                        }
                    }
                }
            )

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {"l2ping": {"enabled": True, "pings_per_port": 3}},
            )


class TestLegacyNodeSmokeNormalization(unittest.TestCase):
    def test_legacy_node_smoke_copied_to_tier1(self):
        cfg = {"node_smoke": {"connectivity_mode": "run", "primus_dir": "/home/user/Primus"}}
        normalized, warning = normalize_legacy_preflight_node_smoke_sections(cfg)
        self.assertIsNotNone(warning)
        self.assertEqual(normalized["node_smoke_tier1"]["primus_dir"], "/home/user/Primus")

    def test_canonical_tier1_not_overwritten_by_legacy(self):
        cfg = {
            "node_smoke_tier1": {"primus_dir": "/tier1/Primus"},
            "node_smoke": {"primus_dir": "/legacy/Primus"},
        }
        normalized, warning = normalize_legacy_preflight_node_smoke_sections(cfg)
        self.assertIsNone(warning)
        self.assertEqual(normalized["node_smoke_tier1"]["primus_dir"], "/tier1/Primus")


class TestLegacyRdmaNormalizer(unittest.TestCase):
    def test_normalize_legacy_rdma_returns_warning_message(self):
        normalized, warning = normalize_legacy_preflight_rdma_config(
            {
                "node_check": {
                    "gid_index": "7",
                    "rdma_interfaces": ["enp4s0np0"],
                },
            }
        )
        self.assertIsNotNone(warning)
        self.assertEqual(normalized["connectivity_check"]["rdma"]["gid_index"], "7")
        self.assertEqual(normalized["connectivity_check"]["rdma"]["interfaces"], ["enp4s0np0"])


class TestFabricPrerequisites(unittest.TestCase):
    def test_fabric_checks_requires_node_check_enabled(self):
        with self.assertRaisesRegex(ValidationError, "fabric_checks requires node_check.enabled"):
            PreflightConfigFile.model_validate(
                {
                    "node_check": {"enabled": False},
                    "connectivity_check": {"ifoe": {"fabric_checks": True}},
                }
            )


if __name__ == "__main__":
    unittest.main()
