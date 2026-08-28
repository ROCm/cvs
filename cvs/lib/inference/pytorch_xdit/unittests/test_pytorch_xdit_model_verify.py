import unittest

from cvs.lib.inference.pytorch_xdit.pytorch_xdit_model_verify import (
    build_diffusers_local_model_required_checks,
    first_required_check_failure,
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


if __name__ == "__main__":
    unittest.main()
