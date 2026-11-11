'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the support-matrix compatibility gate. All cluster access is
mocked via a fake handle whose .exec() returns canned per-host output.
'''

import os
import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.support_matrix import support_matrix_lib as sm


def _handle(arch_text, rocm_text, hosts=("node0",)):
    """A fake orchestrator handle: .exec(cmd) -> {host: output}.

    The arch probe contains 'rocm-smi'; the ROCm probe does not.
    """
    handle = MagicMock()

    def _exec(cmd, *args, **kwargs):
        if "rocm-smi" in cmd:
            return {host: arch_text for host in hosts}
        return {host: rocm_text for host in hosts}

    handle.exec.side_effect = _exec
    return handle


class VersionTests(unittest.TestCase):
    def test_parse_variants(self):
        self.assertEqual(sm.parse_version("7.0"), (7, 0, 0))
        self.assertEqual(sm.parse_version("6.4.0-123"), (6, 4, 0))
        self.assertEqual(sm.parse_version("6.4.1"), (6, 4, 1))
        self.assertIsNone(sm.parse_version(""))
        self.assertIsNone(sm.parse_version(None))

    def test_ordering(self):
        self.assertTrue(sm.parse_version("6.4.1") < sm.parse_version("7.0"))
        self.assertTrue(sm.parse_version("7.0") >= sm.parse_version("7.0.0"))


class ArchTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(sm.normalize_arch("MI325"), "mi325x")
        self.assertEqual(sm.normalize_arch("mi355"), "mi355x")
        self.assertEqual(sm.normalize_arch("mi300x"), "mi300x")
        self.assertEqual(sm.normalize_arch(" Mi350 "), "mi350x")


class EnvSwitchTests(unittest.TestCase):
    def test_default_is_enforce(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(sm.gate_enforced())

    def test_on_values_enforce(self):
        for val in ("1", "ON", "on", "TRUE", "true", "yes"):
            with patch.dict(os.environ, {sm.GATE_ENV: val}, clear=True):
                self.assertTrue(sm.gate_enforced(), val)

    def test_off_values_disable(self):
        for val in ("0", "OFF", "off", "FALSE", "false", "disabled"):
            with patch.dict(os.environ, {sm.GATE_ENV: val}, clear=True):
                self.assertFalse(sm.gate_enforced(), val)


class ExtractRocmTests(unittest.TestCase):
    def test_version_file_text(self):
        self.assertEqual(sm._extract_rocm_version("6.4.0-123\n"), "6.4.0")

    def test_amd_smi_json(self):
        self.assertEqual(sm._extract_rocm_version('[{"rocm_version": "7.0.0"}]'), "7.0.0")

    def test_empty(self):
        self.assertIsNone(sm._extract_rocm_version(""))


class ConfidentArchTests(unittest.TestCase):
    def test_marketing_names(self):
        self.assertEqual(sm._confident_arch("Card series: AMD Instinct MI355X"), "mi355x")
        self.assertEqual(sm._confident_arch("AMD Instinct MI300X"), "mi300x")
        self.assertEqual(sm._confident_arch("Market Name: Instinct MI325X"), "mi325x")

    def test_device_id_fallback(self):
        self.assertEqual(sm._confident_arch("Device ID: 0x75a0\nGFX Version: gfx950"), "mi350x")

    def test_unrecognized_is_none_not_mi300x(self):
        # A banner or an arch the parser does not know must NOT become mi300x.
        self.assertIsNone(sm._confident_arch("============ ROCm System Management ============"))
        self.assertIsNone(sm._confident_arch("Device ID: 0x9999"))
        self.assertIsNone(sm._confident_arch(""))


class DetectPlatformTests(unittest.TestCase):
    def test_detects_arch_and_rocm(self):
        handle = _handle("Card series: AMD Instinct MI355X", "7.0.0-42", hosts=("n0", "n1"))
        platform = sm.detect_platform(handle)
        self.assertEqual(platform["n0"], {"arch": "mi355x", "rocm": "7.0.0"})
        self.assertEqual(platform["n1"]["arch"], "mi355x")

    def test_missing_outputs(self):
        handle = _handle("", "", hosts=("n0",))
        platform = sm.detect_platform(handle)
        self.assertIsNone(platform["n0"]["arch"])
        self.assertIsNone(platform["n0"]["rocm"])

    def test_probe_passes_timeout(self):
        handle = _handle("AMD Instinct MI300X", "7.0.0")
        sm.detect_platform(handle)
        self.assertTrue(all("timeout" in c.kwargs for c in handle.exec.call_args_list))


_ENTRY = {
    "unsupported_gpu_arch": ["mi300x"],
    "required_rocm_min_version": "7.0",
    "required_rocm_max_version": "",
    "unsupported_params": {"precision": {"mxfp4": ["mi300x", "mi325x"]}},
}


class EvaluateTests(unittest.TestCase):
    def test_model_level_arch_deny(self):
        violations, _ = sm.evaluate(_ENTRY, {}, {"n0": {"arch": "mi300x", "rocm": "7.0"}})
        self.assertTrue(any("not supported for this model" in v for v in violations))

    def test_precision_deny(self):
        violations, _ = sm.evaluate(_ENTRY, {"precision": ["mxfp4"]}, {"n0": {"arch": "mi325x", "rocm": "7.0"}})
        self.assertTrue(any("precision=mxfp4 is not supported on mi325x" in v for v in violations))

    def test_precision_supported_elsewhere(self):
        violations, _ = sm.evaluate(_ENTRY, {"precision": ["mxfp4", "bf16"]}, {"n0": {"arch": "mi355x", "rocm": "7.0"}})
        self.assertEqual(violations, [])

    def test_rocm_below_min(self):
        violations, _ = sm.evaluate(_ENTRY, {}, {"n0": {"arch": "mi355x", "rocm": "6.4.0"}})
        self.assertTrue(any("below the required minimum" in v for v in violations))

    def test_rocm_above_max(self):
        entry = dict(_ENTRY, required_rocm_max_version="7.0", unsupported_gpu_arch=[])
        violations, _ = sm.evaluate(entry, {}, {"n0": {"arch": "mi355x", "rocm": "7.1.0"}})
        self.assertTrue(any("above the supported maximum" in v for v in violations))

    def test_undetected_arch_is_violation(self):
        violations, _ = sm.evaluate(_ENTRY, {"precision": ["mxfp4"]}, {"n0": {"arch": None, "rocm": "7.0"}})
        self.assertTrue(any("could not detect a supported GPU arch" in v for v in violations))

    def test_undetected_rocm_is_warning_only(self):
        entry = {"required_rocm_min_version": "7.0", "unsupported_gpu_arch": [], "unsupported_params": {}}
        violations, warnings = sm.evaluate(entry, {}, {"n0": {"arch": "mi355x", "rocm": None}})
        self.assertEqual(violations, [])
        self.assertTrue(any("could not detect ROCm" in w for w in warnings))


class ServedModelNameTests(unittest.TestCase):
    def test_model_object_with_id(self):
        vc = MagicMock(spec=["model"])
        vc.model = MagicMock(spec=["id"])
        vc.model.id = "llama-3.1-8b"
        self.assertEqual(sm.served_model_name(vc), "llama-3.1-8b")

    def test_server_params_dict(self):
        vc = MagicMock(spec=["model", "server_params"])
        vc.model = None
        vc.server_params = {"model": "openai/gpt-oss-120b"}
        self.assertEqual(sm.served_model_name(vc), "openai/gpt-oss-120b")

    def test_inference_dict_model(self):
        vc = MagicMock(spec=["model", "server_params", "inference"])
        vc.model = None
        vc.server_params = None
        vc.inference = {"model": "deepseek-ai/DeepSeek-R1-0528"}
        self.assertEqual(sm.served_model_name(vc), "deepseek-ai/DeepSeek-R1-0528")

    def test_model_repo_path_is_ignored(self):
        # model_repo is a filesystem path, not a model id, so it is not used.
        vc = MagicMock(spec=["model", "server_params", "inference"])
        vc.model = None
        vc.server_params = None
        vc.inference = {"model_repo": "/data/models/FLUX.1-dev"}
        self.assertEqual(sm.served_model_name(vc), "")

    def test_top_level_model_name(self):
        vc = MagicMock(spec=["model_name"])
        vc.model_name = "llama3.1_8B"
        self.assertEqual(sm.served_model_name(vc), "llama3.1_8B")

    def test_top_level_hf_model_id(self):
        vc = MagicMock(spec=["hf_model_id"])
        vc.hf_model_id = "meta-llama/Llama-3.1-8B"
        self.assertEqual(sm.served_model_name(vc), "meta-llama/Llama-3.1-8B")

    def test_train_params_model_name(self):
        vc = MagicMock(spec=["train_params"])
        vc.train_params = {"model_name": "deepseek_v2_lite"}
        self.assertEqual(sm.served_model_name(vc), "deepseek_v2_lite")

    def test_none_found(self):
        vc = MagicMock(spec=[])
        self.assertEqual(sm.served_model_name(vc), "")


class FindEntryTests(unittest.TestCase):
    def setUp(self):
        self.matrix = sm.load_matrix("training")

    def test_by_key(self):
        self.assertIsNotNone(sm.find_model_entry(self.matrix, "jaxmaxtext", "llama-3.1-8b"))

    def test_by_alias(self):
        matrix = {"frameworks": {"jaxmaxtext": {"models": {"llama-3.1-8b": {"aliases": ["meta-llama/Llama-3.1-8B"]}}}}}
        self.assertIsNotNone(sm.find_model_entry(matrix, "jaxmaxtext", "meta-llama/Llama-3.1-8B"))
        self.assertIsNotNone(sm.find_model_entry(matrix, "JaxMaxText", "LLAMA-3.1-8B"))

    def test_absent_model(self):
        self.assertIsNone(sm.find_model_entry(self.matrix, "jaxmaxtext", "no-such-model"))

    def test_absent_framework(self):
        self.assertIsNone(sm.find_model_entry(self.matrix, "nope", "llama-3.1-8b"))

    def test_matches_config_spellings(self):
        # The strings the suites actually pass: megatron llama3.1_8B, torchtitan
        # llama3_1_8b, and the jaxmaxtext MaxText preset alias deepseek2-16b.
        self.assertIsNotNone(sm.find_model_entry(self.matrix, "megatron", "llama3.1_8B"))
        self.assertIsNotNone(sm.find_model_entry(self.matrix, "torchtitan", "llama3_1_8b"))
        self.assertIsNotNone(sm.find_model_entry(self.matrix, "jaxmaxtext", "deepseek2-16b"))

    def test_matches_path_basename(self):
        self.assertIsNotNone(sm.find_model_entry(self.matrix, "jaxmaxtext", "/root/models/Llama-3.1-8B"))


class LoadMatrixValidationTests(unittest.TestCase):
    def test_real_files_load(self):
        self.assertIn("jaxmaxtext", sm.load_matrix("training")["frameworks"])
        self.assertIn("vllm", sm.load_matrix("inference")["frameworks"])

    def test_unknown_arch_token_rejected(self):
        bad = {
            "known_arch_tokens": ["mi300x"],
            "frameworks": {"f": {"models": {"m": {"unsupported_gpu_arch": ["mi999x"]}}}},
        }
        with patch("builtins.open"), patch("json.load", return_value=bad):
            with self.assertRaises(ValueError):
                sm.load_matrix("training")


class RunGateTests(unittest.TestCase):
    def test_absent_model_allows(self):
        handle = _handle("MI300X", "7.14.0")
        with patch.dict(os.environ, {}, clear=True):
            result = sm.run_support_matrix_gate("training", "jaxmaxtext", "no-such", {}, handle)
        self.assertTrue(result.passed)
        self.assertTrue(result.not_in_matrix)
        handle.exec.assert_not_called()

    def test_supported_passes(self):
        handle = _handle("AMD Instinct MI355X", "7.14.0")
        with patch.dict(os.environ, {}, clear=True):
            result = sm.run_support_matrix_gate(
                "training", "jaxmaxtext", "llama-3.1-8b", {"precision": ["mxfp4", "bf16"]}, handle
            )
        self.assertTrue(result.passed)
        self.assertEqual(result.violations, [])

    def test_violation_enforced_blocks(self):
        handle = _handle("AMD Instinct MI300X", "7.14.0")
        with patch.dict(os.environ, {sm.GATE_ENV: "ON"}, clear=True):
            result = sm.run_support_matrix_gate(
                "training", "jaxmaxtext", "llama-3.1-8b", {"precision": ["mxfp4"]}, handle
            )
        self.assertFalse(result.passed)
        self.assertTrue(result.violations)

    def test_violation_disabled_continues(self):
        handle = _handle("AMD Instinct MI300X", "7.14.0")
        with patch.dict(os.environ, {sm.GATE_ENV: "OFF"}, clear=True):
            result = sm.run_support_matrix_gate(
                "training", "jaxmaxtext", "llama-3.1-8b", {"precision": ["mxfp4"]}, handle
            )
        self.assertTrue(result.passed)
        self.assertTrue(result.violations)

    def test_normalized_model_spelling_blocks(self):
        # A different separator spelling of the model still matches the entry.
        handle = _handle("AMD Instinct MI300X", "7.14.0")
        with patch.dict(os.environ, {sm.GATE_ENV: "ON"}, clear=True):
            result = sm.run_support_matrix_gate(
                "training", "jaxmaxtext", "LLAMA3_1_8B", {"precision": ["mxfp4"]}, handle
            )
        self.assertFalse(result.passed)

    def test_undetected_gpu_fails_when_enforced(self):
        handle = _handle("==== ROCm System Management ====", "7.0.0")
        with patch.dict(os.environ, {sm.GATE_ENV: "ON"}, clear=True):
            result = sm.run_support_matrix_gate(
                "training", "jaxmaxtext", "llama-3.1-8b", {"precision": ["bf16"]}, handle
            )
        self.assertFalse(result.passed)
        self.assertTrue(any("could not detect" in v for v in result.violations))

    def test_matrix_load_failure_fails_when_enforced(self):
        handle = _handle("AMD Instinct MI300X", "7.0.0")
        with patch.object(sm, "load_matrix", side_effect=ValueError("bad json")):
            with patch.dict(os.environ, {sm.GATE_ENV: "ON"}, clear=True):
                self.assertFalse(
                    sm.run_support_matrix_gate("training", "jaxmaxtext", "llama-3.1-8b", {}, handle).passed
                )
            with patch.dict(os.environ, {sm.GATE_ENV: "OFF"}, clear=True):
                self.assertTrue(sm.run_support_matrix_gate("training", "jaxmaxtext", "llama-3.1-8b", {}, handle).passed)


if __name__ == "__main__":
    unittest.main()
