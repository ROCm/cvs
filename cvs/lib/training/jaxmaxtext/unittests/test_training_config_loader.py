'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/utils/training_config_loader.py: schema
defaults for the metric add-ons (scaling_baseline / convergence / loss_curve),
the expected_cells (sweep-name) contract, the threshold-coverage validator, and
a round-trip load of a real jaxmaxtext config file.
'''

import getpass
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.training.jaxmaxtext.utils.training_config_loader import (
    CheckpointResume,
    Convergence,
    LossCurve,
    NcclConfig,
    ScalingBaseline,
    SmokeTest,
    load_training_variant,
    normalize_training_config,
    validate_thresholds_cover_training,
)


def _new_format_config():
    """A minimal single-node config in the current on-disk layout.

    Single-node (no NCCL IB device env) so it loads without cluster-specific
    <changeme> values -- exercises the full normalize + validate path.
    """
    return {
        "gpu_name": "mi300x",
        "threshold_json": "cfg_threshold.json",
        "enforce_thresholds": False,
        "gpus_per_node": 8,
        "paths": {
            "shared_fs": "/home/{user-id}",
            "models_dir": "{shared_fs}/cache/maxtext",
            "log_dir": "{shared_fs}/LOGS/jaxmaxtext",
            "hf_token_file": "{shared_fs}/.hf_token",
            "temp_dir": "/tmp/{user-id}/jaxmaxtext",
        },
        "container": {
            "lifetime": "per_run",
            "name": "rocm-jaxmaxtext-test",
            "image": "rocm/jax-training:test",
            "runtime": {"name": "docker", "args": {"network": "host"}},
            "_env_comment": "static env",
            "env": {
                "GPU_MAX_HW_QUEUES": "2",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
                "JAX_COORDINATOR_PORT": "12346",
                "JAX_DISTRIBUTED_HEARTBEAT_TIMEOUT_SECONDS": "900",
            },
        },
        "train_params": {
            "hf_model_id": "NousResearch/Meta-Llama-3-70B",
            "train_script_paths": ["/workspace/maxtext/a.py", "/workspace/maxtext/b.py"],
            "maxtext_config": {
                "model_name": "llama3.3-70b",
                "tokenizer_path": "{paths.models_dir}/Meta-Llama-70-B",
                "steps": 30,
                "enable_checkpointing": False,
                "dtype": "bfloat16",
                "dataset_type": "synthetic",
                "per_device_batch_size": 2,
            },
            "xla_flags": {"xla_gpu_autotune_level": "0"},
        },
        "scaling_baseline": {"tokens_per_sec_total": 100.0, "num_nodes": 1},
        "convergence": {"target_metric": "auto", "target_value": 0.0},
        "loss_curve": {"sample_every": 10, "enforce": True},
        "smoke": {"enabled": True, "steps": 5},
        "checkpoint_resume": {"enabled": False},
        "error_patterns": {"NCCL ERROR": "NCCL ERROR"},
        "sweeps": {
            "CELL_A": {"per_device_batch_size": 2, "quantization": ""},
            "CELL_B": {"per_device_batch_size": 2, "quantization": "nanoo_fp8"},
        },
        "runs": ["CELL_A", "CELL_B"],
    }


def _write_config(tmpdir, cfg):
    """Write cfg + a sibling threshold file, return the config path."""
    cfg_path = Path(tmpdir) / "cfg.json"
    thr_path = Path(tmpdir) / "cfg_threshold.json"
    cfg_path.write_text(json.dumps(cfg))
    thr_path.write_text(json.dumps({"CELL_A": {}, "CELL_B": {}}))
    return cfg_path


class SchemaDefaultsTests(unittest.TestCase):
    def test_scaling_baseline_defaults(self):
        sb = ScalingBaseline()
        self.assertEqual(sb.tokens_per_sec_total, 0.0)
        self.assertEqual(sb.num_nodes, 1)

    def test_convergence_defaults(self):
        c = Convergence()
        self.assertEqual(c.target_metric, "auto")
        self.assertEqual(c.target_value, 0.0)

    def test_loss_curve_defaults(self):
        lc = LossCurve()
        self.assertEqual(lc.sample_every, 10)
        self.assertEqual(lc.milestone_steps, [100, 500, 1000, 5000])
        self.assertEqual(lc.max_slope, 0.0)
        self.assertTrue(lc.enforce)

    def test_smoke_defaults(self):
        s = SmokeTest()
        self.assertTrue(s.enabled)  # opt-OUT: on by default
        self.assertEqual(s.steps, 5)
        self.assertEqual(s.per_device_batch_size, 1)
        self.assertEqual(s.max_target_length, 2048)

    def test_checkpoint_resume_defaults(self):
        cr = CheckpointResume()
        self.assertFalse(cr.enabled)  # opt-in: off by default
        self.assertEqual(cr.sweep, "")
        self.assertEqual(cr.steps_before_ckpt, 6)
        self.assertEqual(cr.steps_after_resume, 6)
        self.assertEqual(cr.checkpoint_period, 5)
        self.assertEqual(cr.loss_tolerance, 0.1)
        self.assertEqual(cr.max_save_seconds, 0.0)
        self.assertEqual(cr.max_load_seconds, 0.0)
        self.assertEqual(cr.smoke_model_overrides, {})


class NcclConfigTests(unittest.TestCase):
    def test_defaults(self):
        n = NcclConfig()
        self.assertEqual(n.ib_gid_index, "3")
        # ib_tc / ib_sl were removed from nccl (they live in env_vars now); the
        # _Allow base still tolerates them if an old config carries them.
        self.assertFalse(hasattr(NcclConfig(), "ib_tc") and "ib_tc" in NcclConfig().model_fields)

    def test_ib_gid_index_changeme_rejected(self):
        with self.assertRaises(ValidationError):
            NcclConfig(ib_gid_index="<changeme>")

    def test_ib_gid_index_concrete_value_ok(self):
        self.assertEqual(NcclConfig(ib_gid_index="1").ib_gid_index, "1")


class ValidateThresholdsCoverTrainingTests(unittest.TestCase):
    _GATED = {
        "training.tflops_per_sec_per_gpu": {"kind": "min", "value": 1},
        "training.tokens_per_sec_per_gpu": {"kind": "min", "value": 1},
        "training.final_loss": {"kind": "max", "value": 15},
        "training.loss_decreased": {"kind": "min", "value": 1},
    }

    def test_missing_cell_raises_when_enforced(self):
        with self.assertRaises(ValueError):
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={},
                enforce_thresholds=True,
            )

    def test_missing_cell_warns_when_not_enforced(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={},
                enforce_thresholds=False,
            )
        self.assertTrue(any("does not match" in str(w.message) for w in caught))

    def test_gated_metric_gap_raises_when_enforced(self):
        # Cell present but missing the gated-metric specs -> coverage failure.
        with self.assertRaises(ValueError):
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={"CELL_A": {}},
                enforce_thresholds=True,
            )

    def test_full_coverage_passes(self):
        # No exception, no warning when every cell + gated metric is covered.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_thresholds_cover_training(
                expected_cells=["CELL_A"],
                thresholds={"CELL_A": dict(self._GATED)},
                enforce_thresholds=True,
            )


class NormalizeTrainingConfigTests(unittest.TestCase):
    """Direct tests of the on-disk-layout -> internal-shape normalizer."""

    def test_missing_train_params_raises(self):
        with self.assertRaises(ValueError):
            normalize_training_config({"gpu_name": "mi300x", "container": {}})

    def test_env_kept_on_container_for_docker_run(self):
        raw = _new_format_config()
        # add NCCL device env so this normalizes as a distributed run
        raw["container"]["env"].update({"NCCL_IB_HCA": "rdma0", "NCCL_SOCKET_IFNAME": "eno0", "NCCL_IB_GID_INDEX": "3"})
        internal = normalize_training_config(raw)
        cenv = internal["container"]["env"]
        # env stays on the container (verbatim) so the orchestrator forwards it as -e
        self.assertEqual(cenv["NCCL_IB_HCA"], "rdma0")
        self.assertEqual(cenv["NCCL_SOCKET_IFNAME"], "eno0")
        self.assertEqual(cenv["GPU_MAX_HW_QUEUES"], "2")
        self.assertEqual(cenv["JAX_COORDINATOR_PORT"], "12346")
        # NCCL IB device selection present -> inferred distributed
        self.assertTrue(internal["training"]["distributed"])

    def test_xla_flags_folded_into_container_env_quoted(self):
        internal = normalize_training_config(_new_format_config())
        xf = internal["container"]["env"]["XLA_FLAGS"]
        # wrapped in double quotes so it survives as one docker -e token
        self.assertTrue(xf.startswith('"') and xf.endswith('"'))
        self.assertIn("--xla_gpu_autotune_level=0", xf)

    def test_empty_xla_flags_omits_xla_flags_env(self):
        raw = _new_format_config()
        raw["train_params"]["xla_flags"] = {}
        internal = normalize_training_config(raw)
        self.assertNotIn("XLA_FLAGS", internal["container"]["env"])

    def test_changeme_in_container_env_raises(self):
        raw = _new_format_config()
        raw["container"]["env"]["NCCL_IB_HCA"] = "<changeme>"
        with self.assertRaises(ValueError):
            normalize_training_config(raw)

    def test_single_node_when_no_nccl_devices(self):
        internal = normalize_training_config(_new_format_config())
        self.assertFalse(internal["training"]["distributed"])

    def test_sweeps_map_and_runs_become_list_and_enabled(self):
        internal = normalize_training_config(_new_format_config())
        t = internal["training"]
        self.assertEqual([s["name"] for s in t["sweeps"]], ["CELL_A", "CELL_B"])
        self.assertEqual(t["enabled_sweep_list"], ["CELL_A", "CELL_B"])
        self.assertEqual(t["sweeps"][1]["maxtext_overrides"]["quantization"], "nanoo_fp8")

    def test_synthesized_model_and_steps(self):
        internal = normalize_training_config(_new_format_config())
        # model.id defaults to maxtext_config.model_name; precision from dtype
        self.assertEqual(internal["model"]["id"], "llama3.3-70b")
        self.assertEqual(internal["model"]["remote"], 0)
        self.assertEqual(internal["model"]["precision"], "bfloat16")
        self.assertEqual(internal["framework"], "jaxmaxtext")
        self.assertEqual(internal["gpu_arch"], "mi300x")
        # steps sourced from maxtext_config
        self.assertEqual(internal["training"]["steps"], 30)

    def test_explicit_model_id_overrides_model_name(self):
        raw = _new_format_config()
        raw["train_params"]["model_id"] = "friendly-label"
        internal = normalize_training_config(raw)
        self.assertEqual(internal["model"]["id"], "friendly-label")

    def test_tokenizer_path_read_from_maxtext_config(self):
        internal = normalize_training_config(_new_format_config())
        self.assertTrue(internal["training"]["tokenizer"]["tokenizer_path"].endswith("/Meta-Llama-70-B"))

    def test_container_env_kept_but_comment_stripped(self):
        internal = normalize_training_config(_new_format_config())
        # env stays on the container (for docker -e); the nested _comment is dropped
        self.assertIn("env", internal["container"])
        self.assertNotIn("_env_comment", internal["container"])


class SweepKeyParsingTests(unittest.TestCase):
    """The sweep KEY (BS/PRECISION/SL) is parsed into maxtext overrides."""

    def _overrides(self, gpu, key, sweep_val):
        cfg = _new_format_config()
        cfg["gpu_name"] = gpu
        cfg["sweeps"] = {key: sweep_val}
        cfg["runs"] = [key]
        return normalize_training_config(cfg)["training"]["sweeps"][0]["maxtext_overrides"]

    def test_key_parsed_bs_sl_and_cdna3_fp8(self):
        ov = self._overrides("mi325x", "BS=3,PRECISION=FP8,SL=4096", {"_comment": "x"})
        self.assertEqual(ov["per_device_batch_size"], 3)
        self.assertEqual(ov["max_target_length"], 4096)
        self.assertEqual(ov["quantization"], "nanoo_fp8")  # CDNA3

    def test_fp8_maps_to_fp8_on_mi35x(self):
        ov = self._overrides("mi35x", "BS=5,PRECISION=FP8,SL=8192", {})
        self.assertEqual(ov["quantization"], "fp8")  # CDNA4

    def test_bf16_clears_quant_comment_stripped_extra_override_kept(self):
        ov = self._overrides("mi300x", "BS=2,PRECISION=BF16,SL=8192", {"_comment": "note", "steps": 300})
        self.assertEqual(ov["quantization"], "")
        self.assertNotIn("_comment", ov)  # underscore keys dropped from overrides
        self.assertEqual(ov["steps"], 300)  # extra override preserved

    def test_explicit_override_takes_precedence_over_parsed_key(self):
        # A dict value wins over the parsed key (e.g. pin a different batch size).
        ov = self._overrides("mi325x", "BS=2,PRECISION=BF16,SL=8192", {"per_device_batch_size": 8})
        self.assertEqual(ov["per_device_batch_size"], 8)


class RoundTripLoadTests(unittest.TestCase):
    """Full load_training_variant round-trip on a temp new-format config."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = _write_config(self._tmp.name, _new_format_config())
        # Empty cluster dict -> {user-id} resolves to the local OS user.
        self.cfg = load_training_variant(str(cfg_path), {})

    def tearDown(self):
        self._tmp.cleanup()

    def test_metric_addon_blocks_present(self):
        t = self.cfg.training
        self.assertIsInstance(t.scaling_baseline, ScalingBaseline)
        self.assertIsInstance(t.convergence, Convergence)
        self.assertIsInstance(t.loss_curve, LossCurve)

    def test_tokenizer_resolved_from_config(self):
        tok = self.cfg.training.tokenizer
        # hf_model_id from train_params; tokenizer_path from maxtext_config (with
        # {paths.models_dir} resolved).
        self.assertEqual(tok.hf_model_id, "NousResearch/Meta-Llama-3-70B")
        self.assertTrue(tok.tokenizer_path.endswith("/cache/maxtext/Meta-Llama-70-B"))

    def test_expected_cells_are_declared_sweep_names(self):
        expected = self.cfg.expected_cells()
        declared = [s.name for s in self.cfg.training.sweeps]
        self.assertEqual(expected, declared)
        for cell in expected:
            self.assertIn(cell, self.cfg.thresholds)

    def test_xla_flags_exported_via_container_env(self):
        # xla_flags{} is folded into container.env as a single quoted XLA_FLAGS
        # var (docker -e), so editing xla_flags{} takes effect with no code change.
        xf = self.cfg.container.env.get("XLA_FLAGS", "")
        self.assertIn("--xla_gpu_autotune_level=0", xf)

    def test_temp_dir_is_user_namespaced(self):
        expected = f"/tmp/{getpass.getuser()}/jaxmaxtext"
        self.assertEqual(self.cfg.paths.temp_dir, expected)


if __name__ == "__main__":
    unittest.main()
