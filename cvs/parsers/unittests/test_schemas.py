"""Configuration validation entry-point compatibility."""

import json
import getpass
import tempfile
import unittest
from pathlib import Path

from cvs.lib.benchmark.aorta.aorta_config_loader import AortaVariantConfig
from cvs.lib.benchmark.aorta.unittests.fixtures import variant_dict
from cvs.parsers.schemas import PreflightConfigFile, RacksBlock, validate_config_file
from cvs.unittests.support import FAKE_CRED


class TestRacksBlock(unittest.TestCase):
    def test_credentials_are_not_racks(self):
        block = RacksBlock.model_validate(
            {
                "switch_ssh_user": "admin",
                "bmc_user": "root",
                "bmc_password": FAKE_CRED,
                "_racks_comment": "ignored",
                "rack-01": {"platform": "HeliosM", "rmc": ["a", "b"], "power_shelves": ["p0"]},
            }
        )
        self.assertEqual(block.bmc_user, "root")
        self.assertEqual(list(block.get_racks()), ["rack-01"])
        rack = block.get_racks()["rack-01"]
        self.assertEqual((rack.rmc, rack.power_shelves), (["a", "b"], ["p0"]))

    def test_single_rmc_string_becomes_a_list(self):
        rack = RacksBlock.model_validate({"rack-01": {"rmc": "10.0.4.1"}}).get_racks()["rack-01"]
        self.assertEqual(rack.rmc, ["10.0.4.1"])

    def test_shipped_rack_sample_parses(self):
        path = Path(__file__).resolve().parents[2] / "input" / "cluster_file" / "cluster_rack.json"
        racks = RacksBlock.model_validate(json.loads(path.read_text())["racks"]).get_racks()
        self.assertEqual(racks["rack-01"].platform, "HeliosP")
        self.assertEqual(len(racks["rack-01"].back_end_switches), 6)


class TestAortaConfigDispatch(unittest.TestCase):
    def test_generic_validator_delegates_to_shared_variant_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = variant_dict()
            thresholds = raw.pop("thresholds")
            (root / "test_threshold.json").write_text(json.dumps(thresholds))
            path = root / "aorta.json"
            path.write_text(json.dumps(raw))
            for mode in ("auto", "aorta"):
                with self.subTest(mode=mode):
                    config = validate_config_file(path, config_type=mode)
                    self.assertIsInstance(config, AortaVariantConfig)
                    self.assertEqual(config.thresholds, thresholds)

    def test_generic_validator_uses_local_user_without_cluster_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = variant_dict()
            thresholds = raw.pop("thresholds")
            raw["container"]["name"] = "{user-id}_aorta"
            (root / "test_threshold.json").write_text(json.dumps(thresholds))
            path = root / "aorta.json"
            path.write_text(json.dumps(raw))
            config = validate_config_file(path)
            self.assertEqual(config.container.name, f"{getpass.getuser()}_aorta")


class TestPreflightRdmaGidType(unittest.TestCase):
    def test_default_gid_type(self):
        config = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'gid_index': '3'}}})
        self.assertEqual(config.connectivity_check.rdma.gid_type, 'RoCE v2')

    def test_any_gid_type_round_trips(self):
        config = PreflightConfigFile.model_validate({'connectivity_check': {'rdma': {'gid_type': 'any'}}})
        self.assertEqual(config.connectivity_check.rdma.gid_type, 'any')
        self.assertEqual(config.model_dump()['connectivity_check']['rdma']['gid_type'], 'any')
