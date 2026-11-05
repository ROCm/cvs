'''Unit tests for the Mori config schema in cvs.lib.conf_lib.schemas.mori.'''

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.conf_lib.loader import ConfigError, load_config
from cvs.lib.conf_lib.schemas.mori import MoriConfig
from cvs.lib.mori_lib import MoriBenchmark
from cvs.unittests.support import write_filled_config

SAMPLE = Path(__file__).resolve().parents[4] / "input" / "config_file" / "mori" / "mi35x_mori_config.json"
CLUSTER = {"username": "jdoe"}


class TestMoriConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.filled = json.loads(write_filled_config(SAMPLE, self._tmp.name).read_text())

    def _load(self, config):
        path = Path(self._tmp.name) / "config.json"
        path.write_text(json.dumps(config))
        return load_config(MoriConfig, path, CLUSTER)

    def test_filled_sample_dumps_every_key_mori_lib_reads(self):
        dump = self._load(self.filled).config.model_dump()
        self.assertEqual(dump["mori_dir"], "/home/jdoe/INSTALL/mori")
        self.assertEqual(dump["log_dir"], "/home/jdoe/LOGS/mori")
        self.assertEqual(dump["master_port"], 1234)
        self.assertEqual((dump["git_ref"], dump["install_system_packages"]), ("main", True))
        self.assertEqual(set(dump["expected_results"]), {"ibgda_write", "io_read", "io_write"})
        orch = SimpleNamespace(orchestrator_type="container", hosts=["n0", "n1"], head_node="n0")
        bench = MoriBenchmark(orch, dump)
        self.assertEqual(bench.master_addr, "filled")
        self.assertEqual(bench.mori_device_list, "filled")

    def test_config_from_0_3_0_loads_with_install_defaults(self):
        config = dict(self.filled)
        del config["git_ref"], config["install_system_packages"]
        loaded = self._load(config).config
        self.assertEqual(loaded.git_url, "https://github.com/ROCm/mori")
        self.assertEqual(loaded.git_ref, "main")
        self.assertTrue(loaded.install_system_packages)

    def test_install_system_packages_accepts_the_old_string_form(self):
        for value, expected in (("True", True), ("False", False), (False, False)):
            with self.subTest(value=value):
                self.assertIs(
                    self._load(dict(self.filled, install_system_packages=value)).config.install_system_packages,
                    expected,
                )
        with self.assertRaises(ConfigError):
            self._load(dict(self.filled, install_system_packages="ture"))

    def test_changeme_fields_are_named_in_one_error(self):
        config = dict(self.filled, mori_dir="<changeme>", git_ref="<changeme>")
        config["env"] = dict(config["env"], MORI_RDMA_DEVICES="<changeme>")
        with self.assertRaises(ConfigError) as ctx:
            self._load(config)
        named = sorted(problem.split(":", 1)[0] for problem in ctx.exception.problems)
        self.assertEqual(named, ["env.MORI_RDMA_DEVICES", "git_ref", "mori_dir"])


if __name__ == "__main__":
    unittest.main()
