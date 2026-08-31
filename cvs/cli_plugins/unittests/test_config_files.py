import os
import tempfile
import unittest

from cvs.cli_plugins import config_files


class TestConfigFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cvs

        cvs_dir = os.path.dirname(cvs.__file__)
        cls.expected_roots = [
            os.path.join(cvs_dir, "input", "config_file"),
            os.path.join(cvs_dir, "input", "cluster_file"),
            os.path.join(cvs_dir, "input", "env_file"),
        ]

    def test_find_config_roots(self):
        roots = config_files.find_config_roots()
        self.assertEqual(sorted(roots), sorted(self.expected_roots))

    def test_parse_scope_with_root_prefix(self):
        root_filter, subpath = config_files.parse_scope("config_file/rccl")
        self.assertEqual(root_filter, "config_file")
        self.assertEqual(subpath, "rccl")

    def test_parse_scope_without_root_prefix(self):
        root_filter, subpath = config_files.parse_scope("training/torchtitan")
        self.assertIsNone(root_filter)
        self.assertEqual(subpath, "training/torchtitan")

    def test_group_files_by_dir(self):
        grouped = config_files.group_files_by_dir(
            [
                "training/jax/a.json",
                "training/jax/b.json",
                "cluster.json",
            ]
        )
        self.assertEqual(grouped["training/jax"], ["training/jax/a.json", "training/jax/b.json"])
        self.assertEqual(grouped[""], ["cluster.json"])

    def test_group_dirs_by_first_segment(self):
        groups = config_files.group_dirs_by_first_segment(["training/jax", "training/torchtitan", "inference/atom"])
        self.assertEqual(
            groups["training"],
            ["training/jax/", "training/torchtitan/"],
        )
        self.assertEqual(groups["inference"], ["inference/atom/"])

    def test_collect_dir_entries(self):
        dirs, root_files = config_files.collect_dir_entries(["rccl/rccl_config.json", "cluster.json"])
        self.assertEqual(dirs, ["rccl"])
        self.assertEqual(root_files, ["cluster.json"])

    def test_list_config_files_training_scope(self):
        root = self.expected_roots[0]
        files = config_files.list_config_files(root, "training/torchtitan")
        self.assertTrue(files)
        self.assertTrue(all(path.startswith("training/torchtitan/") for path in files))

    def test_find_config_file(self):
        found = config_files.find_config_file(
            self.expected_roots,
            "platform/host_config.json",
        )
        self.assertIsNotNone(found)
        self.assertTrue(os.path.isfile(found))

    def test_copy_single_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "host_config.json")
            ok = config_files.copy_single_config(
                self.expected_roots,
                "platform/host_config.json",
                dest,
            )
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(dest))

    def test_copy_all_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ok = config_files.copy_all_configs(self.expected_roots, tmpdir)
            self.assertTrue(ok)
            for root in self.expected_roots:
                label = os.path.basename(root)
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, label)))


if __name__ == "__main__":
    unittest.main()
