'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication
and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Tests for cvs generate rack_cluster_json.
'''

import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout

from cvs.input.generate.rack_cluster_json import RackClusterJsonGenerator, split_rack_groups
from cvs.lib import arc_rack_lib
from cvs.parsers.schemas import ClusterConfigFile
from cvs.unittests.support import FAKE_CRED

GLOBAL = ["--username", "runner", "--key_file", "/home/runner/.ssh/id_rsa"]


class TestSplitRackGroups(unittest.TestCase):
    def test_splits_on_numbered_markers(self):
        global_args, groups = split_rack_groups(
            ["--username", "u", "--rack0", "--computes", "a", "--rack1", "--id", "r"]
        )
        self.assertEqual(global_args, ["--username", "u"])
        self.assertEqual(groups, {"--rack0": ["--computes", "a"], "--rack1": ["--id", "r"]})


class TestGenerate(unittest.TestCase):
    def run_generator(self, argv):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cluster.json")
            out = io.StringIO()
            with redirect_stdout(out):
                RackClusterJsonGenerator().generate(Namespace(_raw_args=["--output_json_file", path] + argv))
            with open(path, encoding="utf-8") as fh:
                return json.load(fh), out.getvalue()

    def assert_exits(self, argv, message):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(out), self.assertRaises(SystemExit):
            RackClusterJsonGenerator().generate(
                Namespace(_raw_args=GLOBAL + ["--output_json_file", os.path.join(tmp, "c.json")] + argv)
            )
        self.assertIn(message, out.getvalue())

    def test_compute_only_rack_is_ready_for_arc(self):
        cluster, _ = self.run_generator(GLOBAL + ["--rack0", "--id", "rack-01", "--computes", "10.0.0.1-3"])
        self.assertEqual(list(cluster["node_dict"]), ["10.0.0.1", "10.0.0.2", "10.0.0.3"])
        self.assertEqual(cluster["node_dict"]["10.0.0.2"], {"bmc_ip": "NA", "vpc_ip": "10.0.0.2", "rack_id": "rack-01"})
        self.assertEqual(cluster["racks"], {"rack-01": {"platform": "HeliosP"}})
        self.assertEqual(cluster["head_node_dict"]["mgmt_ip"], "10.0.0.1")
        self.assertEqual(arc_rack_lib.validate_racks(cluster), [])
        ClusterConfigFile.model_validate(cluster)

    def test_full_rack_fields(self):
        cluster, out = self.run_generator(
            GLOBAL
            + ["--switch_ssh_user", "admin", "--switch_ssh_password", FAKE_CRED, "--bmc_user", "root"]
            + ["--rack0", "--id", "rack-01", "--platform", "HeliosM", "--computes", "10.0.0.1-2"]
            + ["--bmcs", "10.0.1.1-2", "--arc_controller", "10.0.0.2", "--switch_model", "CelesticaPolarisSwitch"]
            + ["--switches", "10.0.2.1-2", "--switch_bmc", "10.0.3.1-2", "--rmc", "10.0.4.1-4"]
            + ["--front_end_switches", "10.0.5.1,10.0.5.2", "--power_shelves", "10.0.7.1"]
            + ["--rack1", "--id", "rack-02", "--computes", "node[1-2]"]
        )
        racks = cluster["racks"]
        self.assertEqual(racks["switch_ssh_password"], FAKE_CRED)
        self.assertEqual((racks["bmc_user"], racks["bmc_password"]), ("root", "<changeme>"))
        self.assertEqual(
            racks["rack-01"],
            {
                "platform": "HeliosM",
                "switch_model": "CelesticaPolarisSwitch",
                "arc_controller": "10.0.0.2",
                "switch_trays": ["10.0.2.1", "10.0.2.2"],
                "switch_bmc": ["10.0.3.1", "10.0.3.2"],
                "rmc": ["10.0.4.1", "10.0.4.2", "10.0.4.3", "10.0.4.4"],
                "front_end_switches": ["10.0.5.1", "10.0.5.2"],
                "power_shelves": ["10.0.7.1"],
            },
        )
        self.assertEqual(racks["rack-02"], {"platform": "HeliosP"})
        self.assertEqual(ClusterConfigFile.model_validate(cluster).racks.get_racks()["rack-01"].rmc[3], "10.0.4.4")
        self.assertEqual(cluster["node_dict"]["10.0.0.2"]["bmc_ip"], "10.0.1.2")
        self.assertEqual(cluster["node_dict"]["node2"]["rack_id"], "rack-02")
        self.assertEqual(arc_rack_lib.rack_controllers(cluster), {"10.0.0.2": "rack-01", "node1": "rack-02"})
        self.assertIn("Replace every <changeme>", out)
        problems = arc_rack_lib.validate_racks(cluster)
        self.assertEqual(problems, ["racks.bmc_password is still <changeme>; set it or remove it"])

    def test_passwords_with_quotes_stay_valid_json(self):
        cluster, out = self.run_generator(
            GLOBAL + ["--bmc_user", "root", "--bmc_password", 'p"w\\d', "--rack0", "--computes", "10.0.0.1"]
        )
        self.assertEqual(cluster["racks"]["bmc_password"], 'p"w\\d')
        self.assertNotIn("<changeme>", out)

    def test_inconsistent_input_exits(self):
        cases = (
            ([], "No rack groups"),
            (["--rack0", "--computes", "a,b", "--bmcs", "x"], "--bmcs has 1 entries; --computes has 2"),
            (
                ["--rack0", "--computes", "a", "--switches", "s1,s2", "--switch_bmc", "b"],
                "switch_bmc has 1 entries; switch_trays has 2",
            ),
            (["--rack0", "--computes", "a", "--arc_controller", "z"], "arc_controller z is not one of its computes"),
            (["--rack0", "--id", "r", "--switches", "s1"], "Rack r is not in the file; give its --computes"),
            (["--rack0", "--computes", "a", "--rack1", "--computes", "a"], "already in another rack: a"),
            (
                ["--rack0", "--id", "r", "--computes", "a", "--rack1", "--id", "r", "--computes", "b"],
                "repeats rack id r",
            ),
            (["--rack0", "--bogus"], "Failed to parse args for --rack0"),
        )
        for argv, message in cases:
            with self.subTest(message=message):
                self.assert_exits(argv, message)


RACK_01 = ["--rack0", "--id", "rack-01", "--computes", "10.0.0.1-2", "--switches", "10.0.2.1"]
RACK_02 = ["--rack0", "--id", "rack-02", "--computes", "10.1.0.1"]


class TestAmend(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "cluster.json")
        self.generate(GLOBAL + ["--switch_ssh_user", "admin", "--bmc_user", "root"] + RACK_01)
        cluster = self.read()
        cluster["racks"]["bmc_password"] = FAKE_CRED
        cluster["env_vars"] = {"PATH": "/opt/rocm/bin:$PATH"}
        cluster["_comment"] = "kept"
        cluster["node_dict"]["10.0.0.1"]["extra"] = "kept"
        self.write(cluster)

    def write(self, cluster):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(cluster, fh)

    def read(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    def generate(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            RackClusterJsonGenerator().generate(Namespace(_raw_args=["--output_json_file", self.path] + argv))
        return out.getvalue()

    def assert_exits(self, argv, message):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            RackClusterJsonGenerator().generate(Namespace(_raw_args=["--output_json_file", self.path] + argv))
        self.assertIn(message, out.getvalue())

    def test_adds_rack_and_keeps_existing_content(self):
        out = self.generate(
            ["--amend", "--switch_ssh_user", "admin", "--bmc_user", "root"]
            + ["--rack0", "--id", "rack-02", "--platform", "HeliosR", "--computes", "10.1.0.1-2", "--rmc", "10.1.4.1"]
        )
        cluster = self.read()
        self.assertIn("Amended rack cluster JSON", out)
        self.assertNotIn("<changeme>", out)
        self.assertEqual(list(cluster["node_dict"]), ["10.0.0.1", "10.0.0.2", "10.1.0.1", "10.1.0.2"])
        self.assertEqual(cluster["node_dict"]["10.0.0.1"]["extra"], "kept")
        self.assertEqual(cluster["node_dict"]["10.1.0.2"]["rack_id"], "rack-02")
        self.assertEqual(cluster["racks"]["rack-02"], {"platform": "HeliosR", "rmc": ["10.1.4.1"]})
        self.assertEqual(cluster["racks"]["bmc_password"], FAKE_CRED)
        self.assertEqual(cluster["env_vars"], {"PATH": "/opt/rocm/bin:$PATH"})
        self.assertEqual(cluster["_comment"], "kept")
        self.assertEqual(cluster["username"], "runner")
        self.assertEqual(cluster["head_node_dict"]["mgmt_ip"], "10.0.0.1")
        self.assertEqual(arc_rack_lib.validate_racks(cluster), [])

    def test_fills_placeholder_and_missing_logins(self):
        cluster = self.read()
        cluster["racks"]["bmc_password"] = "<changeme>"
        self.write(cluster)
        self.generate(["--amend", "--bmc_password", FAKE_CRED, "--switch_ssh_key_file", "/keys/sw"] + RACK_02)
        racks = self.read()["racks"]
        self.assertEqual(racks["bmc_password"], FAKE_CRED)
        self.assertEqual(racks["switch_ssh_key_file"], "/keys/sw")

    def test_rerunning_the_same_rack_changes_nothing(self):
        before = self.read()
        self.generate(["--amend"] + RACK_01)
        self.assertEqual(self.read(), before)

    def test_compute_only_rack_gains_switch_details_later(self):
        self.generate(GLOBAL + ["--rack0", "--id", "rack-01", "--platform", "HeliosR", "--computes", "10.0.0.1-2"])
        self.generate(
            ["--amend", "--switch_ssh_user", "admin", "--rack0", "--id", "rack-01"]
            + ["--switches", "10.0.2.1-2", "--bmcs", "10.0.1.1-2"]
        )
        self.generate(["--amend", "--bmc_user", "root", "--rack0", "--id", "rack-01", "--switch_bmc", "10.0.3.1-2"])
        cluster = self.read()
        self.assertEqual(
            cluster["racks"]["rack-01"],
            {"platform": "HeliosR", "switch_trays": ["10.0.2.1", "10.0.2.2"], "switch_bmc": ["10.0.3.1", "10.0.3.2"]},
        )
        self.assertEqual([info["bmc_ip"] for info in cluster["node_dict"].values()], ["10.0.1.1", "10.0.1.2"])
        self.assertEqual(cluster["racks"]["switch_ssh_user"], "admin")
        self.assertEqual(cluster["racks"]["bmc_password"], "<changeme>")

    def test_replace_overwrites_only_the_fields_passed(self):
        self.generate(["--amend", "--replace", "--rack0", "--id", "rack-01", "--switches", "10.0.9.1", "--rmc", "r1"])
        cluster = self.read()
        self.assertEqual(
            cluster["racks"]["rack-01"], {"platform": "HeliosP", "switch_trays": ["10.0.9.1"], "rmc": ["r1"]}
        )
        self.assertEqual(list(cluster["node_dict"]), ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(cluster["node_dict"]["10.0.0.1"]["extra"], "kept")

    def test_replace_computes_swaps_nodes_in_place(self):
        self.generate(["--amend"] + RACK_02)
        self.generate(["--amend", "--replace", "--rack0", "--id", "rack-01", "--computes", "10.0.0.9,10.0.0.1"])
        cluster = self.read()
        self.assertEqual(list(cluster["node_dict"]), ["10.0.0.9", "10.0.0.1", "10.1.0.1"])
        self.assertEqual(cluster["node_dict"]["10.0.0.1"]["extra"], "kept")
        self.assertEqual(cluster["node_dict"]["10.0.0.9"], {"bmc_ip": "NA", "vpc_ip": "10.0.0.9", "rack_id": "rack-01"})
        self.assertEqual(cluster["racks"]["rack-01"]["switch_trays"], ["10.0.2.1"])

    def test_replace_still_rejects_computes_of_another_rack(self):
        self.generate(["--amend"] + RACK_02)
        self.assert_exits(
            ["--amend", "--replace", "--rack0", "--id", "rack-01", "--computes", "10.1.0.1"],
            "under another rack: 10.1.0.1",
        )

    def test_keeps_file_permissions_and_leaves_no_temp_file(self):
        os.chmod(self.path, 0o600)
        self.generate(["--amend"] + RACK_02)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(os.listdir(self.dir), ["cluster.json"])

    def test_rejections_leave_file_untouched(self):
        cases = (
            (["--amend", "--rack0", "--computes", "10.1.0.1"], "needs --id with --amend"),
            (
                ["--amend", "--rack0", "--id", "rack-01", "--computes", "10.0.0.2,10.0.0.1"],
                "differs from the file's list",
            ),
            (
                ["--amend", "--rack0", "--id", "rack-01", "--switches", "10.0.9.9"],
                "racks.rack-01.switch_trays is ['10.0.2.1'] in the file, not ['10.0.9.9']; pass --replace",
            ),
            (
                ["--amend", "--rack0", "--id", "rack-01", "--switch_bmc", "a,b"],
                "switch_bmc has 2 entries; switch_trays has 1",
            ),
            (["--amend", "--rack0", "--id", "rack-01", "--bmcs", "10.0.1.1"], "has 1 entries; the rack has 2 computes"),
            (["--amend", "--rack0", "--id", "rack-01", "--arc_controller", "10.1.0.1"], "is not one of its computes"),
            (["--amend", "--rack0", "--id", "rack-02", "--computes", "10.0.0.2"], "under another rack: 10.0.0.2"),
            (["--amend", "--bmc_user", "admin"] + RACK_02, "racks.bmc_user is 'root' in the file, not 'admin'"),
            (["--amend", "--username", "other"] + RACK_02, "username is 'runner' in the file"),
            (["--amend", "--head_node", "10.9.9.9"] + RACK_02, "head_node_dict.mgmt_ip is '10.0.0.1'"),
            (["--replace"] + GLOBAL + RACK_02, "--replace works only with --amend"),
            (RACK_02, "--username and --key_file are required unless --amend"),
        )
        before = self.read()
        for argv, message in cases:
            with self.subTest(message=message):
                self.assert_exits(argv, message)
                self.assertEqual(self.read(), before)

    def test_missing_or_non_object_file(self):
        os.remove(self.path)
        self.assert_exits(["--amend"] + RACK_02, "--amend needs an existing")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("[1, 2]")
        self.assert_exits(["--amend"] + RACK_02, "is not a cluster JSON object")


if __name__ == "__main__":
    unittest.main()
