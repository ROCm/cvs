'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Tests for building ARC rack files from the cluster file's racks block.
'''

import copy
import json
import unittest
from pathlib import Path

from cvs.lib import arc_rack_lib
from cvs.unittests.support import FAKE_CRED


def _slot(interfaces, tray_id=None):
    slot = {"serial_number": "", "interfaces": {name: {"hostname": "", "username": ""} for name in interfaces}}
    if tray_id is None:
        slot["location"] = {"row": 0, "position": 0}
    else:
        slot["tray_id"] = tray_id
    return slot


TEMPLATE = {
    "platform_id": "HeliosR",
    "serial_number": "",
    "rack": {
        "compute": [_slot(("eth1", "bmc"), tray) for tray in range(3)],
        "manager": [_slot(("bmc",), 0)],
        "switch": [_slot(("eth1", "bmc"), tray) for tray in range(2)],
        "front_end_switch": [_slot(("eth1", "bmc"))],
        "back_end_switch": [_slot(("eth1", "bmc"))],
    },
}

CLUSTER = {
    "username": "runner",
    "priv_key_file": "/home/runner/.ssh/id_rsa",
    "node_dict": {
        "node-b": {"rack_id": "rack-01", "vpc_ip": "10.0.0.2", "bmc_ip": "10.0.1.2"},
        "node-a": {"rack_id": "rack-01", "vpc_ip": "10.0.0.1", "bmc_ip": "NA"},
        "node-c": {"rack_id": "rack-02", "vpc_ip": "10.0.0.3"},
    },
    "racks": {
        "_racks_comment": "ignored",
        "switch_ssh_user": "admin",
        "switch_ssh_password": FAKE_CRED,
        "bmc_user": "root",
        "bmc_password": FAKE_CRED,
        "rack-01": {
            "platform": "HeliosR",
            "switch_model": "celestica",
            "switch_trays": ["10.0.2.1", "10.0.2.2"],
            "switch_bmc": ["10.0.3.1", "10.0.3.2"],
            "rmc": ["10.0.4.1"],
            "front_end_switches": ["10.0.5.1"],
            "back_end_switches": ["10.0.6.1"],
        },
        "rack-02": {"platform": "HeliosR", "arc_controller": "node-c"},
    },
}


def cluster(**rack_changes):
    data = copy.deepcopy(CLUSTER)
    data["racks"]["rack-01"].update(rack_changes)
    return data


class TestRackSelection(unittest.TestCase):
    def test_entries_skip_credentials_and_comments(self):
        self.assertEqual(list(arc_rack_lib.rack_entries(CLUSTER)), ["rack-01", "rack-02"])

    def test_rack_groups_alias(self):
        data = copy.deepcopy(CLUSTER)
        data["rack_groups"] = data.pop("racks")
        self.assertEqual(list(arc_rack_lib.rack_entries(data)), ["rack-01", "rack-02"])

    def test_nodes_keep_file_order(self):
        self.assertEqual(arc_rack_lib.rack_nodes(CLUSTER, "rack-01"), ["node-b", "node-a"])

    def test_controller_defaults_to_first_node_and_honors_override(self):
        self.assertEqual(arc_rack_lib.rack_controller(CLUSTER, "rack-01"), "node-b")
        self.assertEqual(arc_rack_lib.rack_controller(cluster(arc_controller="node-a"), "rack-01"), "node-a")
        self.assertEqual(arc_rack_lib.rack_controllers(CLUSTER), {"node-b": "rack-01", "node-c": "rack-02"})

    def test_no_racks_block(self):
        self.assertEqual(arc_rack_lib.rack_entries({"node_dict": {}}), {})
        self.assertEqual(arc_rack_lib.rack_controllers({"node_dict": {}}), {})


class TestValidateRacks(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(arc_rack_lib.validate_racks(CLUSTER), [])

    def test_problems(self):
        cases = (
            ("platform", cluster(platform=None)),
            ("letters, digits", cluster(platform="helios;rm")),
            ("not in node_dict", cluster(arc_controller="ghost")),
            ("share controller", cluster(arc_controller="node-c")),
            ("switch_bmc has 1", cluster(switch_bmc=["10.0.3.1"])),
        )
        for expected, data in cases:
            with self.subTest(expected=expected):
                problems = arc_rack_lib.validate_racks(data)
                self.assertTrue(any(expected in problem for problem in problems), problems)

    def test_missing_racks_block_and_unknown_rack_id(self):
        self.assertIn("no racks block", arc_rack_lib.validate_racks({"node_dict": {"n": {}}})[0])
        data = copy.deepcopy(CLUSTER)
        data["node_dict"]["node-d"] = {"rack_id": "rack-09"}
        self.assertTrue(any("rack-09" in problem for problem in arc_rack_lib.validate_rack_controllers(data)))

    def test_placeholders_are_rejected(self):
        data = cluster(rmc=["<changeme>"])
        data["racks"]["bmc_user"] = "<changeme>"
        data["racks"]["_bmc_comment"] = "<changeme> in a comment is fine"
        problems = [p for p in arc_rack_lib.validate_racks(data) if "<changeme>" in p]
        self.assertEqual(
            sorted(problems),
            [
                "racks.bmc_user is still <changeme>; set it or remove it",
                "racks.rack-01.rmc[0] is still <changeme>; set it or remove it",
            ],
        )

    def test_shipped_sample_needs_only_bmc_login(self):
        path = Path(arc_rack_lib.__file__).resolve().parents[1] / "input" / "cluster_file" / "cluster_rack.json"
        with open(path, encoding="utf-8") as fh:
            sample = json.load(fh)
        problems = arc_rack_lib.validate_racks(sample)
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(all("bmc_" in problem and "<changeme>" in problem for problem in problems))
        self.assertEqual(arc_rack_lib.rack_controllers(sample), {"10.0.0.1": "rack-01"})

    def test_rack_without_nodes(self):
        data = copy.deepcopy(CLUSTER)
        data["racks"]["rack-03"] = {"platform": "HeliosR"}
        self.assertTrue(any("rack-03" in problem for problem in arc_rack_lib.validate_rack_controllers(data)))


class TestFillRackTemplate(unittest.TestCase):
    def setUp(self):
        self.data = arc_rack_lib.fill_rack_template(TEMPLATE, CLUSTER, "rack-01")
        self.rack = self.data["rack"]

    def test_template_is_not_modified(self):
        self.assertEqual(TEMPLATE["rack"]["compute"][0]["serial_number"], "")

    def test_rack_identity_and_models(self):
        self.assertEqual(self.data["serial_number"], "rack-01")
        self.assertEqual(self.data["models"], {"switch": "celestica"})
        self.assertEqual(self.data["platform_id"], "HeliosR")

    def test_compute_in_tray_order(self):
        first, second, unused = self.rack["compute"]
        self.assertEqual((first["tray_id"], first["serial_number"]), (0, "rack-01-compute-0"))
        self.assertEqual(
            first["interfaces"]["eth1"],
            {"hostname": "10.0.0.2", "username": "runner", "ssh_key_filename": "/home/runner/.ssh/id_rsa"},
        )
        self.assertEqual(
            first["interfaces"]["bmc"], {"hostname": "10.0.1.2", "username": "root", "password": FAKE_CRED}
        )
        self.assertEqual(second["interfaces"]["eth1"]["hostname"], "10.0.0.1")
        self.assertEqual(second["interfaces"]["bmc"]["hostname"], "", "an NA bmc_ip leaves the slot empty")
        self.assertEqual(unused["serial_number"], "")

    def test_switches_rmc_and_unmapped_locations(self):
        switch = self.rack["switch"][1]
        self.assertEqual(
            switch["interfaces"]["eth1"], {"hostname": "10.0.2.2", "username": "admin", "password": FAKE_CRED}
        )
        self.assertEqual(switch["interfaces"]["bmc"]["hostname"], "10.0.3.2")
        self.assertEqual(self.rack["manager"][0]["interfaces"]["bmc"]["hostname"], "10.0.4.1")
        self.assertNotIn("eth1", self.rack["manager"][0]["interfaces"])
        locations = [self.rack[key][0]["location"] for key in ("front_end_switch", "back_end_switch")]
        self.assertEqual(locations, [{"row": 0, "position": 0}, {"row": 0, "position": 1}])
        self.assertEqual(self.rack["back_end_switch"][0]["serial_number"], "rack-01-back_end_switch-0")

    def test_switch_key_file_replaces_password(self):
        data = cluster()
        data["racks"]["switch_ssh_key_file"] = "/keys/switch"
        eth1 = arc_rack_lib.fill_rack_template(TEMPLATE, data, "rack-01")["rack"]["switch"][0]["interfaces"]["eth1"]
        self.assertEqual(eth1["ssh_key_filename"], "/keys/switch")
        self.assertNotIn("password", eth1)

    def test_without_bmc_login_bmc_stays_empty(self):
        data = cluster(rmc=[])
        del data["racks"]["bmc_user"]
        rack = arc_rack_lib.fill_rack_template(TEMPLATE, data, "rack-01")["rack"]
        self.assertEqual(rack["compute"][0]["interfaces"]["bmc"]["hostname"], "")
        self.assertEqual(rack["switch"][0]["interfaces"]["bmc"]["hostname"], "")

    def test_rmc_without_bmc_login_is_an_error(self):
        data = cluster()
        del data["racks"]["bmc_user"]
        with self.assertRaisesRegex(ValueError, "bmc_user"):
            arc_rack_lib.fill_rack_template(TEMPLATE, data, "rack-01")

    def test_more_entries_than_slots(self):
        with self.assertRaisesRegex(ValueError, "3 switch entries listed, but the platform declares only 2"):
            arc_rack_lib.fill_rack_template(TEMPLATE, cluster(switch_trays=["a", "b", "c"], switch_bmc=[]), "rack-01")
        with self.assertRaisesRegex(ValueError, "declares none"):
            arc_rack_lib.fill_rack_template(TEMPLATE, cluster(power_shelves=["p"]), "rack-01")

    def test_single_string_rmc(self):
        rack = arc_rack_lib.fill_rack_template(TEMPLATE, cluster(rmc="10.0.4.9"), "rack-01")["rack"]
        self.assertEqual(rack["manager"][0]["interfaces"]["bmc"]["hostname"], "10.0.4.9")

    def test_serials_are_sanitized(self):
        data = copy.deepcopy(CLUSTER)
        data["racks"]["_x rack/1"] = data["racks"].pop("rack-01")
        for info in data["node_dict"].values():
            if info["rack_id"] == "rack-01":
                info["rack_id"] = "_x rack/1"
        filled = arc_rack_lib.fill_rack_template(TEMPLATE, data, "_x rack/1")
        self.assertEqual(filled["serial_number"], "r_x-rack-1")
        self.assertEqual(filled["rack"]["compute"][0]["serial_number"], "r_x-rack-1-compute-0")


class TestRedacted(unittest.TestCase):
    def test_masks_password_values_only(self):
        out = arc_rack_lib.redacted(CLUSTER)
        self.assertEqual(out["racks"]["bmc_password"], "***")
        self.assertEqual(out["racks"]["switch_ssh_password"], "***")
        self.assertEqual(out["racks"]["bmc_user"], "root")
        self.assertEqual(CLUSTER["racks"]["bmc_password"], FAKE_CRED)
        self.assertEqual(
            arc_rack_lib.redacted({"password": "", "items": [{"password": "x"}]}),
            {"password": "", "items": [{"password": "***"}]},
        )


if __name__ == "__main__":
    unittest.main()
