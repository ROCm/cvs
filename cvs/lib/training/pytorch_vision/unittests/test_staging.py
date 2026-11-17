import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cvs.lib.training.pytorch_vision.staging import remove_staged_dataset, stage_dataset, staged_container
from cvs.lib.training.pytorch_vision.unittests.test_config_loader import _config, _thresholds
from cvs.lib.training.pytorch_vision.utils.config_loader import load_vision_variant


class _LocalHosts:
    """Runs the staging command on this machine as if it were one host."""

    def __init__(self):
        self.calls = 0

    def exec(self, command, timeout=None, detailed=False, print_console=True):
        self.calls += 1
        done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"node0": {"output": done.stdout + done.stderr, "exit_code": done.returncode}}


def _staging(source, destination, enabled=True):
    return SimpleNamespace(
        enabled=enabled, source=str(source), destination=str(destination), parallel_copies=4, timeout_s=60
    )


@unittest.skipUnless(shutil.which("flock") and shutil.which("bash"), "needs bash and flock")
class TestStageDataset(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "imagenet"
        for split, cls, name in (("train", "0001", "a.JPEG"), ("train", "0002", "b.JPEG"), ("val", "0001", "c.JPEG")):
            (self.source / split / cls).mkdir(parents=True, exist_ok=True)
            (self.source / split / cls / name).write_bytes(name.encode() * 100)
        (self.source / "MANIFEST.json").write_text("{}")
        (self.source / "val" / "README").write_text("second level")
        (self.source / "train" / "0001" / "extra").mkdir()
        (self.source / "train" / "0001" / "extra" / "deep.JPEG").write_bytes(b"deep")
        self.destination = self.root / "shm" / "cvs-imagenet"
        self.hosts = _LocalHosts()

    def _stage(self):
        return stage_dataset(self.hosts, ["node0"], _staging(self.source, self.destination))["node0"]

    def test_copies_every_file_and_marks_the_copy(self):
        summary = self._stage()
        self.assertIn("CVS_STAGE copied 6 files", summary)
        self.assertEqual((self.destination / "train/0002/b.JPEG").read_bytes(), b"b.JPEG" * 100)
        self.assertTrue((self.destination / ".cvs_staged").is_file())
        self.assertFalse((self.destination / ".cvs_staging").exists())

    def test_reuses_a_copy_that_still_matches_the_source(self):
        self._stage()
        self.assertIn("CVS_STAGE reused 6 files", self._stage())

    def test_recopies_when_the_source_changes(self):
        self._stage()
        (self.source / "val/0001/d.JPEG").write_bytes(b"new")
        self.assertIn("CVS_STAGE copied 7 files", self._stage())
        self.assertTrue((self.destination / "val/0001/d.JPEG").is_file())

    def test_resumes_over_an_interrupted_copy(self):
        self.destination.mkdir(parents=True)
        (self.destination / ".cvs_staging").touch()
        (self.destination / "partial").write_text("x")
        self.assertIn("CVS_STAGE copied 6 files", self._stage())
        self.assertFalse((self.destination / "partial").exists())

    def test_leaves_a_directory_it_did_not_create(self):
        self.destination.mkdir(parents=True)
        (self.destination / "keep.txt").write_text("user data")
        with self.assertRaisesRegex(RuntimeError, "not created by CVS dataset staging"):
            self._stage()
        self.assertEqual((self.destination / "keep.txt").read_text(), "user data")

    def test_removes_a_staged_copy_after_the_run(self):
        self._stage()
        summary = remove_staged_dataset(self.hosts, ["node0"], _staging(self.source, self.destination))["node0"]
        self.assertIn("CVS_STAGE removed", summary)
        self.assertFalse(self.destination.exists())
        self.assertTrue((self.source / "train/0001/a.JPEG").is_file())

    def test_removal_keeps_a_directory_it_did_not_create(self):
        self.destination.mkdir(parents=True)
        (self.destination / "keep.txt").write_text("user data")
        summary = remove_staged_dataset(self.hosts, ["node0"], _staging(self.source, self.destination))["node0"]
        self.assertIn("CVS_STAGE kept", summary)
        self.assertTrue((self.destination / "keep.txt").is_file())

    def test_reports_a_missing_source(self):
        shutil.rmtree(self.source)
        with self.assertRaisesRegex(RuntimeError, "is not a directory"):
            self._stage()


class TestStagedContainer(unittest.TestCase):
    def test_mounts_the_staged_copy_in_place_of_the_source(self):
        container = {
            "runtime": {
                "args": {"volumes": ["/home/u:/home/u", "/weka/imagenet/:/datasets/imagenet-1k:ro"]},
            }
        }
        staged = staged_container(container, _staging("/weka/imagenet", "/dev/shm/cvs-imagenet"))
        self.assertEqual(
            staged["runtime"]["args"]["volumes"],
            ["/home/u:/home/u", "/dev/shm/cvs-imagenet:/datasets/imagenet-1k:ro"],
        )
        self.assertEqual(container["runtime"]["args"]["volumes"][1], "/weka/imagenet/:/datasets/imagenet-1k:ro")

    def test_disabled_staging_keeps_the_container_spec(self):
        container = {"runtime": {"args": {"volumes": ["/weka/imagenet:/datasets/imagenet-1k:ro"]}}}
        self.assertIs(staged_container(container, _staging("/weka/imagenet", "/dev/shm/x", enabled=False)), container)


class TestStagingConfig(unittest.TestCase):
    def _load(self, staging, volumes=("/weka/imagenet:/datasets/imagenet-1k:ro",)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        config = _config()
        config["container"]["runtime"]["args"] = {"volumes": list(volumes)}
        config["training"]["dataset_staging"] = staging
        (root / "w1_config.json").write_text(json.dumps(config))
        (root / "w1_threshold.json").write_text(json.dumps(_thresholds()))
        return load_vision_variant(root / "w1_config.json", {"username": "tester"})

    def test_accepts_a_source_that_is_a_volume(self):
        variant = self._load({"enabled": True, "source": "/weka/imagenet/", "destination": "/dev/shm/cvs-imagenet"})
        self.assertEqual(variant.training.dataset_staging.parallel_copies, 48)
        self.assertFalse(variant.training.dataset_staging.remove_after_run)

    def test_rejects_a_source_that_no_volume_mounts(self):
        with self.assertRaisesRegex(ValueError, "host side of a container volume"):
            self._load({"enabled": True, "source": "/other", "destination": "/dev/shm/cvs-imagenet"})

    def test_rejects_nested_or_top_level_destinations(self):
        for destination, message in (
            ("/weka/imagenet/copy", "must not contain each other"),
            ("/dev", "dedicated directory"),
            ("relative/path", "absolute paths"),
        ):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, message):
                self._load({"enabled": True, "source": "/weka/imagenet", "destination": destination})

    def test_disabled_staging_needs_no_paths(self):
        self.assertFalse(self._load({"enabled": False}).training.dataset_staging.enabled)


if __name__ == "__main__":
    unittest.main()
