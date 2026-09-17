import unittest

from cvs.lib.inference.xdit.pytorch_xdit_model_verify import (
    build_diffusers_local_model_required_checks,
    build_hf_snapshot_download_cmd,
    container_snapshot_to_host,
    download_hf_snapshot,
    first_required_check_failure,
    is_local_model_path,
    parse_hf_snapshot_output,
    resolve_wan_local_model_required_checks,
    wan_native_hf_snapshot_required_checks,
)


class TestDiffusersLocalModelChecks(unittest.TestCase):
    def test_includes_model_index_and_transformer(self):
        checks = build_diffusers_local_model_required_checks("/models/flux")
        self.assertIn("model_index.json", checks)
        self.assertIn("transformer weights", checks)
        self.assertIn("/models/flux/model_index.json", checks["model_index.json"])


class TestWanLocalModelChecks(unittest.TestCase):
    def test_native_uses_wan_snapshot_fingerprint(self):
        checks = resolve_wan_local_model_required_checks(
            "/models/Wan2.2-I2V-A14B",
            model_repo="/models/Wan2.2-I2V-A14B",
        )
        self.assertIn("configuration.json", checks)
        self.assertIn("low_noise_model/config.json", checks)

    def test_diffusers_uses_diffusers_tree_checks(self):
        checks = resolve_wan_local_model_required_checks(
            "/models/Wan2.2-I2V-A14B-Diffusers",
            model_repo="/models/Wan2.2-I2V-A14B-Diffusers",
        )
        self.assertIn("model_index.json", checks)
        self.assertIn("vae weights", checks)


class TestWanHfSnapshotRequiredChecks(unittest.TestCase):
    def test_native_repo_returns_checks(self):
        checks = wan_native_hf_snapshot_required_checks("/cache/snap", "Wan-AI/Wan2.2-I2V-A14B")
        self.assertIsNotNone(checks)
        self.assertIn("configuration.json", checks)

    def test_diffusers_repo_returns_none(self):
        self.assertIsNone(wan_native_hf_snapshot_required_checks("/cache/snap", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"))


class TestFirstRequiredCheckFailure(unittest.TestCase):
    def test_returns_first_failing_label(self):
        class FakePhdl:
            def exec(self, cmd, print_console=False):
                if "model_index.json" in cmd:
                    return {"n1": "MISSING", "n2": "OK"}
                return {"n1": "OK", "n2": "OK"}

        failure = first_required_check_failure(
            FakePhdl(),
            build_diffusers_local_model_required_checks("/models/flux"),
        )
        self.assertEqual(failure, ("model_index.json", ["n1"]))


class TestHfSnapshotDownload(unittest.TestCase):
    def test_local_path_detection(self):
        self.assertTrue(is_local_model_path("/data/models/FLUX.1-dev"))
        self.assertFalse(is_local_model_path("black-forest-labs/FLUX.1-dev"))
        self.assertFalse(is_local_model_path("Wan-AI/Wan2.2-I2V-A14B-Diffusers"))

    def test_download_cmd_uses_repo_and_hf_home(self):
        cmd = build_hf_snapshot_download_cmd(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            revision="main",
            hf_home="/hf_home",
            token="hf_secret",
        )
        self.assertIn("snapshot_download", cmd)
        self.assertIn("XDIT_HF_REPO=Wan-AI/Wan2.2-I2V-A14B-Diffusers", cmd)
        self.assertIn("XDIT_HF_REVISION=main", cmd)
        self.assertIn("HF_HOME=/hf_home", cmd)
        self.assertIn("HF_TOKEN=hf_secret", cmd)

    def test_download_cmd_omits_token_when_empty(self):
        cmd = build_hf_snapshot_download_cmd("org/model")
        self.assertNotIn("HF_TOKEN=", cmd)
        self.assertNotIn("XDIT_HF_REVISION=", cmd)

    def test_parse_snapshot_and_error(self):
        snapshot, error = parse_hf_snapshot_output("noise\nHF_SNAPSHOT=/hf_home/hub/snap\n")
        self.assertEqual(snapshot, "/hf_home/hub/snap")
        self.assertIsNone(error)
        snapshot, error = parse_hf_snapshot_output({"output": "HF_DOWNLOAD_ERROR=gated repo"})
        self.assertIsNone(snapshot)
        self.assertEqual(error, "gated repo")

    def test_container_snapshot_maps_to_host_hf_home(self):
        host = container_snapshot_to_host(
            "/hf_home/hub/models--org--name/snapshots/abc",
            {"hf_home": "/home/user/.cache/huggingface", "hf_home_container": "/hf_home"},
        )
        self.assertEqual(host, "/home/user/.cache/huggingface/hub/models--org--name/snapshots/abc")

    def test_download_records_per_host_snapshot(self):
        class FakeOrch:
            def exec(self, cmd, timeout=None):
                self.cmd = cmd
                self.timeout = timeout
                return {"n1": "HF_SNAPSHOT=/hf_home/hub/snap"}

        orch = FakeOrch()
        snapshots, errors = download_hf_snapshot(
            orch,
            {"model_repo": "org/model", "hf_home_container": "/hf_home"},
            token="tok",
        )
        self.assertEqual(errors, [])
        self.assertEqual(snapshots, {"n1": "/hf_home/hub/snap"})
        self.assertIn("XDIT_HF_REPO=org/model", orch.cmd)
        self.assertIn("HF_TOKEN=tok", orch.cmd)


if __name__ == "__main__":
    unittest.main()
