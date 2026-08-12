'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/training/jaxmaxtext/jaxmaxtext_training_lib.py::MaxTextTrainingJob.

The job talks to the outside world only through an injected orchestrator
(`orch.exec` / `orch.exec_cmd_list`), so every test builds a job with a
MagicMock orch and a lightweight SimpleNamespace variant -- no SSH, no
container, no real sleeps (mirrors test_megatron_training_lib.py).
'''

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib import MaxTextTrainingJob


def _training(**overrides):
    t = SimpleNamespace(
        steps=3,
        distributed=True,
        enable_checkpointing=False,
        train_script="/workspace/maxtext/src/MaxText/train.py",
        maxtext_config={
            "per_device_batch_size": 2,
            "max_target_length": 8192,
            "scan_layers": True,
            "mlp_activations": ["silu", "linear"],
        },
        nic_type="thor2",
        env_vars={"NCCL_DEBUG": "ERROR"},
        xla_flags={"xla_gpu_autotune_level": "0", "xla_gpu_enable_triton_gemm": "False"},
        nccl=SimpleNamespace(
            ib_hca="rdma0",
            ib_hca_list="rdma0,rdma1",
            socket_ifname="eno0",
            gloo_socket_ifname="eno0",
        ),
        jax_distributed=SimpleNamespace(
            coordinator_ip="auto",
            coordinator_port="12346",
            initialization_timeout_seconds="1800",
            heartbeat_timeout_seconds="900",
        ),
        rdma_lib=SimpleNamespace(container_mount_file="", container_dest_file=""),
        tokenizer=SimpleNamespace(hf_model_id="", tokenizer_path="/models/tok"),
    )
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def _make_job(hosts=None, **training_overrides):
    hosts = hosts or ["h0"]
    orch = MagicMock()
    orch.hosts = list(hosts)
    orch.exec = MagicMock(return_value={})
    orch.exec_cmd_list = MagicMock(return_value={})
    variant = SimpleNamespace(
        training=_training(**training_overrides),
        model=SimpleNamespace(id="llama3.3-70b"),
        paths=SimpleNamespace(log_dir="/logs", models_dir="/models"),
    )
    return MaxTextTrainingJob(orch, variant, hf_token="dummy"), orch


def _wire_container_exec(orch, user="tester", script="/workspace/maxtext/src/MaxText/train.py"):
    """Answer the in-container probes the job runs before building launchers.

    ``build_training_cmd`` resolves the scratch dir (``id -un``) and the train
    script (a ``[ -f ... ]`` probe) via ``orch.exec``; without wiring these the
    default empty response would make ``_resolve_train_script`` raise.
    """

    def _side(cmd, *a, **k):
        text = str(cmd)
        if "id -un" in text:
            return {h: user for h in orch.hosts}
        if "train.py" in text:
            return {h: script for h in orch.hosts}
        return {}

    orch.exec.side_effect = _side


def _log(steps=3):
    lines = []
    for i in range(steps):
        lines.append(
            f"I0804 08:14:00 1 metric_logger.py:196] completed step: {i}, seconds: 0.5, "
            f"TFLOP/s/device: 200.0, Tokens/s/device: 25000.0, total_weights: 1, loss: {9.0 - i}"
        )
    return "\n".join(lines) + "\n"


class ConstructorTests(unittest.TestCase):
    def test_node_and_gpu_counts(self):
        job, _ = _make_job(hosts=["h0", "h1"])
        self.assertEqual(job.num_nodes, 2)
        self.assertEqual(job.num_gpus, 16)
        self.assertEqual(job.out_dir, "/logs/jaxmaxtext")

    def test_build_xla_flags_str(self):
        job, _ = _make_job()
        s = job._build_xla_flags_str()
        self.assertIn("--xla_gpu_autotune_level=0", s)
        self.assertIn("--xla_gpu_enable_triton_gemm=False", s)

    def test_gpus_per_node_from_config(self):
        # num_gpus derives from config gpus_per_node, not a hardcoded 8.
        job, _ = _make_job(hosts=["h0", "h1"], gpus_per_node=4)
        self.assertEqual(job.gpus_per_node, 4)
        self.assertEqual(job.num_gpus, 8)

    def test_gpus_per_node_defaults_to_8(self):
        job, _ = _make_job(hosts=["h0"])  # fake config has no gpus_per_node
        self.assertEqual(job.gpus_per_node, 8)
        self.assertEqual(job.num_gpus, 8)


class StopTrainingTests(unittest.TestCase):
    @patch("cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib.time.sleep")
    def test_uses_bracketed_self_safe_pattern(self, _sleep):
        job, orch = _make_job(hosts=["h0"])
        job.stop_training()
        cmd = orch.exec.call_args.args[0]
        # Bracketed first char so the pkill wrapper's own cmdline is not matched.
        self.assertIn("[m]axtext_config.yml", cmd)
        self.assertIn("[t]raining_launcher_node", cmd)


class IsCompleteTests(unittest.TestCase):
    def test_all_nodes_complete(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        orch.exec_cmd_list.return_value = {"h0": "1", "h1": "1"}
        self.assertTrue(job.is_complete())

    def test_one_node_incomplete(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        orch.exec_cmd_list.return_value = {"h0": "1", "h1": "0"}
        self.assertFalse(job.is_complete())

    def test_missing_host_output(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        orch.exec_cmd_list.return_value = {"h0": "1"}
        self.assertFalse(job.is_complete())

    def test_dict_shaped_result(self):
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": {"output": "1"}}
        self.assertTrue(job.is_complete())


class ScanForErrorsTests(unittest.TestCase):
    def test_clean_log_no_raise(self):
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": _log()}
        job._scan_for_errors()  # should not raise

    def test_nccl_error_raises(self):
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": "some log\nNCCL ERROR: unhandled\n"}
        with self.assertRaises(RuntimeError):
            job._scan_for_errors()

    def test_nan_metric_raises(self):
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": "completed step: 1, TFLOP/s/device: NaN\n"}
        with self.assertRaises(RuntimeError):
            job._scan_for_errors()

    def test_config_error_patterns_replace_defaults(self):
        # A config-provided error_patterns set fully REPLACES the built-in defaults.
        job, orch = _make_job(hosts=["h0"], error_patterns={"custom": "MY_CUSTOM_ERR"})
        # The default NCCL signature is no longer active -> no raise.
        orch.exec_cmd_list.return_value = {"h0": "some log\nNCCL ERROR: unhandled\n"}
        job._scan_for_errors()
        # The custom signature IS active -> raises.
        orch.exec_cmd_list.return_value = {"h0": "boom MY_CUSTOM_ERR here\n"}
        with self.assertRaises(RuntimeError):
            job._scan_for_errors()

    def test_default_error_patterns_used_when_config_empty(self):
        # No config error_patterns -> built-in defaults apply.
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": "RESOURCE_EXHAUSTED: Out of memory\n"}
        with self.assertRaises(RuntimeError):
            job._scan_for_errors()

    def test_default_segfault_pattern_raises(self):
        # segfault is part of the built-in default signatures.
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": "worker: Segmentation fault (core dumped)\n"}
        with self.assertRaises(RuntimeError):
            job._scan_for_errors()


class ParseResultsTests(unittest.TestCase):
    def test_parses_from_node0_log(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        orch.exec_cmd_list.return_value = {"h0": _log(steps=3), "h1": ""}
        summary = job.parse_results()
        self.assertEqual(len(job.step_metrics), 3)
        self.assertIn("training.final_loss", summary)
        self.assertAlmostEqual(summary["training.final_loss"], 7.0)

    def test_empty_log_raises(self):
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": "   "}
        with self.assertRaises(RuntimeError):
            job.parse_results()


class SetupRdmaLibTests(unittest.TestCase):
    def test_skip_when_paths_unset(self):
        job, orch = _make_job()  # rdma_lib defaults are empty strings
        job.setup_rdma_lib()
        orch.exec.assert_not_called()

    def test_raises_when_devinfo_mismatch(self):
        job, orch = _make_job(rdma_lib=SimpleNamespace(container_mount_file="/src.so", container_dest_file="/dst.so"))
        orch.exec.return_value = {"h0": "no matching hca here"}
        with self.assertRaises(RuntimeError):
            job.setup_rdma_lib()

    def test_ok_when_devinfo_matches(self):
        job, orch = _make_job(rdma_lib=SimpleNamespace(container_mount_file="/src.so", container_dest_file="/dst.so"))
        orch.exec.return_value = {"h0": "hca_id: bnxt_re0\n"}
        job.setup_rdma_lib()  # should not raise


class SetupTokenizerTests(unittest.TestCase):
    def test_skips_download_when_no_model_id(self):
        job, orch = _make_job()  # hf_model_id="" by default
        job.setup_tokenizer()
        # Only the mkdir exec fires; no huggingface-cli download command.
        joined = " ".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertNotIn("huggingface-cli", joined)

    def test_downloads_when_model_id_set(self):
        job, orch = _make_job(tokenizer=SimpleNamespace(hf_model_id="org/model", tokenizer_path="/models/tok"))
        job.setup_tokenizer()
        joined = " ".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertIn("huggingface-cli download", joined)
        self.assertIn("org/model", joined)


class BuildTrainingCmdTests(unittest.TestCase):
    def test_distributed_per_rank_indices(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        _wire_container_exec(orch)
        job.build_training_cmd()
        cmds = orch.exec_cmd_list.call_args.args[0]
        self.assertEqual(len(cmds), 2)
        self.assertIn("JAX_PROCESS_INDEX=0", cmds[0])
        self.assertIn("NODE_RANK=0", cmds[0])
        self.assertIn("JAX_PROCESS_INDEX=1", cmds[1])
        self.assertIn("NODE_RANK=1", cmds[1])
        # coordinator IP is host 0
        self.assertIn("JAX_COORDINATOR_IP=h0", cmds[0])
        # resolved train script and user-namespaced scratch dir are wired in
        self.assertIn("/workspace/maxtext/src/MaxText/train.py", cmds[0])
        self.assertIn("/tmp/tester/jax/maxtext_env.sh", cmds[0])
        self.assertIn("/tmp/tester/jax/maxtext_config.yml", cmds[0])

    def test_single_node_localhost_coordinator(self):
        job, orch = _make_job(hosts=["h0"], distributed=False)
        _wire_container_exec(orch)
        job.build_training_cmd()
        cmds = orch.exec_cmd_list.call_args.args[0]
        self.assertEqual(len(cmds), 1)
        self.assertIn("JAX_COORDINATOR_IP=localhost", cmds[0])
        self.assertIn("JAX_PROCESS_INDEX=0", cmds[0])

    def test_auto_coordinator_uses_first_host(self):
        # coordinator_ip "auto" -> first cluster node (orch.hosts[0]).
        job, orch = _make_job(hosts=["10.0.0.5", "10.0.0.6"])
        _wire_container_exec(orch)
        job.build_training_cmd()
        cmds = orch.exec_cmd_list.call_args.args[0]
        self.assertIn("JAX_COORDINATOR_IP=10.0.0.5", cmds[0])

    def test_explicit_coordinator_ip_overrides_auto(self):
        # A concrete coordinator_ip in the config wins over the first host.
        job, orch = _make_job(
            hosts=["10.0.0.5", "10.0.0.6"],
            jax_distributed=SimpleNamespace(
                coordinator_ip="10.9.9.9",
                coordinator_port="12346",
                initialization_timeout_seconds="1800",
                heartbeat_timeout_seconds="900",
            ),
        )
        _wire_container_exec(orch)
        job.build_training_cmd()
        cmds = orch.exec_cmd_list.call_args.args[0]
        self.assertIn("JAX_COORDINATOR_IP=10.9.9.9", cmds[0])


class TrainScriptResolveTests(unittest.TestCase):
    def test_returns_first_existing_probed_path(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        v264 = "/workspace/maxtext/src/maxtext/trainers/pre_train/train.py"
        orch.exec.side_effect = lambda cmd, *a, **k: (
            {h: v264 for h in orch.hosts} if "train.py" in str(cmd) else {}
        )
        self.assertEqual(job._resolve_train_script(), v264)

    def test_raises_when_no_candidate_exists(self):
        job, orch = _make_job()
        orch.exec.return_value = {"h0": ""}
        with self.assertRaises(RuntimeError):
            job._resolve_train_script()

    def test_result_is_cached(self):
        job, orch = _make_job()
        orch.exec.return_value = {"h0": "/workspace/maxtext/src/MaxText/train.py"}
        first = job._resolve_train_script()
        count_after_first = orch.exec.call_count
        second = job._resolve_train_script()
        self.assertEqual(first, second)
        self.assertEqual(orch.exec.call_count, count_after_first)


class ScratchDirTests(unittest.TestCase):
    def test_user_namespaced(self):
        job, orch = _make_job()
        orch.exec.return_value = {"h0": "alice"}
        self.assertEqual(job._get_scratch_dir(), "/tmp/alice/jax")

    def test_falls_back_to_default_when_unresolved(self):
        job, orch = _make_job()
        orch.exec.return_value = {}
        self.assertEqual(job._get_scratch_dir(), "/tmp/cvs/jax")

    def test_result_is_cached(self):
        job, orch = _make_job()
        orch.exec.return_value = {"h0": "bob"}
        job._get_scratch_dir()
        count_after_first = orch.exec.call_count
        job._get_scratch_dir()
        self.assertEqual(orch.exec.call_count, count_after_first)


class WriteMaxtextYamlTests(unittest.TestCase):
    def test_yaml_content_has_run_name_steps_and_bools(self):
        job, orch = _make_job()
        job._write_maxtext_yaml()
        written = " ".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertIn("run_name: jaxmaxtext_llama3.3-70b", written)
        self.assertIn("steps: 3", written)
        # enable_checkpointing False -> rendered as lowercase yaml bool
        self.assertIn("enable_checkpointing: false", written)
        # scan_layers True -> lowercase bool
        self.assertIn("scan_layers: true", written)


class StartTrainingTests(unittest.TestCase):
    @patch("cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib.time.sleep")
    def test_launches_per_node_backgrounded(self, _sleep):
        job, orch = _make_job(hosts=["h0", "h1"])
        job.start_training()
        cmds = orch.exec_cmd_list.call_args.args[0]
        self.assertEqual(len(cmds), 2)
        self.assertTrue(all("nohup bash" in c for c in cmds))
        _sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
