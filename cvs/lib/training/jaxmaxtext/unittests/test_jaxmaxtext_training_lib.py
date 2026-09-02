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

from cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib import MaxTextTrainingJob, needs_hf_tokenizer


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
            ib_gid_index="3",
        ),
        jax_distributed=SimpleNamespace(
            coordinator_ip="auto",
            coordinator_port="12346",
            initialization_timeout_seconds="1800",
            heartbeat_timeout_seconds="900",
        ),
        rdma_lib=SimpleNamespace(container_mount_file="", container_dest_file=""),
        tokenizer=SimpleNamespace(hf_model_id="", tokenizer_path="/models/tok"),
        sweeps=[],
        enabled_sweep_list=[],
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
        paths=SimpleNamespace(log_dir="/logs", models_dir="/models", temp_dir="/tmp/tester/jaxmaxtext"),
    )
    return MaxTextTrainingJob(orch, variant, hf_token="dummy"), orch


def _wire_container_exec(orch, script="/workspace/maxtext/src/MaxText/train.py"):
    """Answer the in-container train-script probe ``build_training_cmd`` runs."""

    def _side(cmd, *a, **k):
        text = str(cmd)
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
        job, _ = _make_job(hosts=["h0"])
        job._scan_chunk_for_errors("h0", 0, _log())  # should not raise

    def test_empty_chunk_no_raise(self):
        job, _ = _make_job(hosts=["h0"])
        job._scan_chunk_for_errors("h0", 0, "")  # should not raise

    def test_nccl_error_raises(self):
        job, _ = _make_job(hosts=["h0"])
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, "some log\nNCCL ERROR: unhandled\n")

    def test_nan_metric_raises(self):
        job, _ = _make_job(hosts=["h0"])
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, "completed step: 1, TFLOP/s/device: NaN\n")

    def test_nan_loss_raises_even_when_throughput_numeric(self):
        # Real failure signature: loss/lm_loss/perplexity go NaN while
        # TFLOP/s/device and Tokens/s/device stay numeric.
        job, _ = _make_job(hosts=["h0"])
        line = (
            "completed step: 3, seconds: 6.783, TFLOP/s/device: 39.268, "
            "Tokens/s/device: 301.929, total_weights: 65536, loss: nan, "
            "lm_loss: nan, perplexity: nan, moe_lb_loss: 0.000\n"
        )
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, line)

    def test_aborting_nan_loss_line_raises(self):
        job, _ = _make_job(hosts=["h0"])
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, "metric_logger.py:270] Aborting training due to NaN loss.\n")

    def test_failure_chunk_is_logged_before_raising(self):
        # The offending chunk (e.g. a compile traceback) must be logged so it
        # reaches the console/--log-file -- the truncated exception message alone
        # can miss the root cause, and non-node-0 chunks are not otherwise streamed.
        job, _ = _make_job(hosts=["h0", "h1"])
        chunk = "some log\nValueError: Compiler params for platform tpu cannot be used for gpu lowering.\ngrpc tail\n"
        with patch("cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib.log") as mock_log:
            with self.assertRaises(RuntimeError):
                job._scan_chunk_for_errors("h1", 1, chunk)
        logged = " ".join(str(c) for c in mock_log.error.call_args_list)
        self.assertIn("FAILURE chunk", logged)
        self.assertIn("Compiler params for platform tpu", logged)

    def test_healthy_step_with_large_perplexity_no_raise(self):
        # Regression guard: valid numeric metrics (incl. a big perplexity) must
        # NOT trip the NaN detector.
        job, _ = _make_job(hosts=["h0"])
        job._scan_chunk_for_errors(
            "h0",
            0,
            "completed step: 2, TFLOP/s/device: 38.530, Tokens/s/device: 296.258, "
            "loss: 11.267, lm_loss: 11.267, perplexity: 78210.594, moe_lb_loss: 0.000\n",
        )

    def test_config_error_patterns_replace_defaults(self):
        # A config-provided error_patterns set fully REPLACES the built-in defaults.
        job, _ = _make_job(hosts=["h0"], error_patterns={"custom": "MY_CUSTOM_ERR"})
        # The default NCCL signature is no longer active -> no raise.
        job._scan_chunk_for_errors("h0", 0, "some log\nNCCL ERROR: unhandled\n")
        # The custom signature IS active -> raises.
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, "boom MY_CUSTOM_ERR here\n")

    def test_default_error_patterns_used_when_config_empty(self):
        # No config error_patterns -> built-in defaults apply.
        job, _ = _make_job(hosts=["h0"])
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, "RESOURCE_EXHAUSTED: Out of memory\n")

    def test_default_segfault_pattern_raises(self):
        # segfault is part of the built-in default signatures.
        job, _ = _make_job(hosts=["h0"])
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, "worker: Segmentation fault (core dumped)\n")

    def test_import_error_caught_by_always_on_even_with_custom_patterns(self):
        # A config that fully overrides error_patterns (no ImportError/Traceback)
        # must STILL catch a fatal Python crash via the always-on set, so an
        # import failure fails fast instead of running to the poll timeout.
        job, _ = _make_job(hosts=["h0"], error_patterns={"custom": "MY_CUSTOM_ERR"})
        chunk = (
            "Traceback (most recent call last):\n"
            "  File \".../jax_flash_attention.py\", line 21, in <module>\n"
            "ImportError: cannot import name 'must_fuse_call' from 'jax.experimental.xla_metadata'\n"
        )
        with self.assertRaises(RuntimeError):
            job._scan_chunk_for_errors("h0", 0, chunk)


class DrainNewLogLinesTests(unittest.TestCase):
    def test_advances_cursor_and_returns_new_text(self):
        job, orch = _make_job(hosts=["h0", "h1"])
        orch.exec_cmd_list.return_value = {"h0": "l1\nl2\nl3\n", "h1": "a\nb\n"}
        new = job._drain_new_log_lines()
        self.assertEqual(new[0], "l1\nl2\nl3\n")
        self.assertEqual(new[1], "a\nb\n")
        # cursor advances by the number of lines read on each node.
        self.assertEqual(job._log_line_cursor, [3, 2])

    def test_uses_cursor_offset_in_tail_and_no_console_echo(self):
        job, orch = _make_job(hosts=["h0"])
        job._log_line_cursor = [5]
        orch.exec_cmd_list.return_value = {"h0": "l6\n"}
        job._drain_new_log_lines()
        # tail starts at the line after the cursor (5 -> +6) ...
        cmd = orch.exec_cmd_list.call_args.args[0][0]
        self.assertIn("tail -n +6", cmd)
        # ... and the bulk read is NOT echoed to the console.
        self.assertEqual(orch.exec_cmd_list.call_args.kwargs.get("print_console"), False)

    def test_empty_output_leaves_cursor_unchanged(self):
        job, orch = _make_job(hosts=["h0"])
        orch.exec_cmd_list.return_value = {"h0": ""}
        new = job._drain_new_log_lines()
        self.assertEqual(new, {})
        self.assertEqual(job._log_line_cursor, [0])


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

    def test_cp_is_guarded_by_source_existence(self):
        # With a direct read-only mount the legacy ".host" source is absent, so
        # the copy must be guarded ([ -f <src> ]) rather than failing.
        job, orch = _make_job(rdma_lib=SimpleNamespace(container_mount_file="/src.so", container_dest_file="/dst.so"))
        orch.exec.return_value = {"h0": "hca_id: bnxt_re0\n"}
        job.setup_rdma_lib()
        cp_cmd = orch.exec.call_args_list[0].args[0]  # first exec == the guarded copy
        self.assertIn("[ -f /src.so ]", cp_cmd)
        self.assertIn("cp /src.so /dst.so", cp_cmd)


class SetupTokenizerTests(unittest.TestCase):
    def test_skips_download_when_no_model_id(self):
        job, orch = _make_job()  # hf_model_id="" by default
        job.setup_tokenizer()
        joined = " ".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertNotIn("hf download", joined)
        self.assertNotIn("huggingface-cli", joined)

    def test_downloads_when_model_id_set(self):
        job, orch = _make_job(tokenizer=SimpleNamespace(hf_model_id="org/model", tokenizer_path="/models/tok"))
        job.setup_tokenizer()
        joined = " ".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertIn("hf download", joined)
        self.assertNotIn("huggingface-cli", joined)
        self.assertIn("org/model", joined)

    def test_skips_download_when_dataset_is_synthetic(self):
        job, orch = _make_job(
            tokenizer=SimpleNamespace(hf_model_id="org/model", tokenizer_path="/models/tok"),
            maxtext_config={"dataset_type": "synthetic"},
        )
        job.setup_tokenizer()
        orch.exec.assert_not_called()


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
        self.assertIn("/tmp/tester/jaxmaxtext/maxtext_env.sh", cmds[0])
        self.assertIn("/tmp/tester/jaxmaxtext/maxtext_config.yml", cmds[0])

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
        orch.exec.side_effect = lambda cmd, *a, **k: ({h: v264 for h in orch.hosts} if "train.py" in str(cmd) else {})
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
    def test_uses_paths_temp_dir(self):
        job, orch = _make_job()
        self.assertEqual(job._get_scratch_dir(), "/tmp/tester/jaxmaxtext")
        orch.exec.assert_not_called()

    def test_falls_back_when_temp_dir_unset(self):
        job, orch = _make_job()
        job.variant.paths = SimpleNamespace(log_dir="/logs", models_dir="/models")
        self.assertEqual(job._get_scratch_dir(), "/tmp/cvs/jaxmaxtext")
        orch.exec.assert_not_called()

    def test_strips_trailing_slash(self):
        job, _ = _make_job()
        job.variant.paths.temp_dir = "/tmp/alice/jaxmaxtext/"
        self.assertEqual(job._get_scratch_dir(), "/tmp/alice/jaxmaxtext")

    def test_result_is_cached(self):
        job, orch = _make_job()
        job.variant.paths.temp_dir = "/tmp/bob/jaxmaxtext"
        first = job._get_scratch_dir()
        job.variant.paths.temp_dir = "/tmp/other/jaxmaxtext"
        self.assertEqual(job._get_scratch_dir(), first)
        orch.exec.assert_not_called()


class NeedsHfTokenizerTests(unittest.TestCase):
    def test_synthetic_does_not_need_tokenizer(self):
        job, _ = _make_job(maxtext_config={"dataset_type": "synthetic"})
        self.assertFalse(needs_hf_tokenizer(job.training))

    def test_missing_dataset_type_needs_tokenizer(self):
        job, _ = _make_job()
        self.assertTrue(needs_hf_tokenizer(job.training))

    def test_sweep_override_to_hf_needs_tokenizer(self):
        job, _ = _make_job(
            maxtext_config={"dataset_type": "synthetic"},
            sweeps=[SimpleNamespace(name="c4", maxtext_overrides={"dataset_type": "hf"})],
            enabled_sweep_list=["c4"],
        )
        self.assertTrue(needs_hf_tokenizer(job.training))


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

    def test_empty_string_rendered_as_quoted_not_bare(self):
        # An empty-string maxtext param (e.g. profiler) must render as 'key: ""',
        # never bare 'key:' (which YAML reads as null and breaks MaxText enums).
        job, orch = _make_job()
        job.maxtext_config["profiler"] = ""
        job._write_maxtext_yaml()
        written = "\n".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertIn('profiler: ""', written)
        self.assertNotIn("profiler: \n", written)


class CheckoutMaxtextBranchTests(unittest.TestCase):
    def test_noop_when_branch_empty(self):
        # Default fixture has no maxtext_branch -> getattr returns "" -> no exec.
        job, orch = _make_job(hosts=["h0"])
        job.checkout_maxtext_branch()
        orch.exec.assert_not_called()

    def test_checks_out_and_verifies_branch(self):
        job, orch = _make_job(hosts=["h0", "h1"], maxtext_branch="feature/x", maxtext_root="/workspace/maxtext")
        orch.exec.return_value = {"h0": "feature/x", "h1": "feature/x"}
        job.checkout_maxtext_branch()  # both nodes on the branch -> no raise
        first_cmd = orch.exec.call_args_list[0].args[0]
        self.assertIn("git reset --hard", first_cmd)
        self.assertIn("git checkout", first_cmd)
        self.assertIn("feature/x", first_cmd)
        self.assertIn("/workspace/maxtext", first_cmd)

    def test_raises_when_a_node_is_on_wrong_branch(self):
        job, orch = _make_job(hosts=["h0", "h1"], maxtext_branch="feature/x")
        orch.exec.return_value = {"h0": "feature/x", "h1": "main"}  # h1 mismatch
        with self.assertRaises(RuntimeError):
            job.checkout_maxtext_branch()

    def test_install_cmd_runs_after_checkout(self):
        job, orch = _make_job(
            hosts=["h0"],
            maxtext_branch="release/v26.7",
            maxtext_install_cmd="python3 -m pip install -r rocm-requirements.txt && python3 -m pip install --no-deps -e .",
        )
        orch.exec.side_effect = [
            {"h0": "release/v26.7"},  # checkout
            {"h0": "release/v26.7"},  # verify HEAD
            {"h0": "installed\n__MAXTEXT_INSTALL_OK__"},  # install cmd
        ]
        job.checkout_maxtext_branch()  # no raise
        install_cmd = orch.exec.call_args_list[-1].args[0]
        self.assertIn("rocm-requirements.txt", install_cmd)
        self.assertIn("--no-deps -e .", install_cmd)

    def test_install_cmd_failure_raises(self):
        job, orch = _make_job(
            hosts=["h0"], maxtext_branch="release/v26.7", maxtext_install_cmd="python3 -m pip install -e ."
        )
        orch.exec.side_effect = [
            {"h0": "release/v26.7"},  # checkout
            {"h0": "release/v26.7"},  # verify HEAD
            {"h0": "ERROR: install failed"},  # no success marker
        ]
        with self.assertRaises(RuntimeError):
            job.checkout_maxtext_branch()

    def test_install_cmd_runs_without_branch(self):
        # Install recipe is decoupled from branch checkout: with an empty branch
        # but a set install_cmd (e.g. the tensorflow-cpu swap on the image's baked
        # MaxText), the recipe still runs and no git checkout is attempted.
        job, orch = _make_job(
            hosts=["h0"],
            maxtext_install_cmd="python3 -m pip install 'tensorflow-cpu>=2.20.0'",
        )
        orch.exec.return_value = {"h0": "ok\n__MAXTEXT_INSTALL_OK__"}
        job.checkout_maxtext_branch()  # no raise
        self.assertEqual(len(orch.exec.call_args_list), 1)  # only the install, no checkout/verify
        install_cmd = orch.exec.call_args_list[0].args[0]
        self.assertIn("tensorflow-cpu", install_cmd)
        self.assertNotIn("git checkout", install_cmd)


class WriteEnvScriptTests(unittest.TestCase):
    def test_distributed_exports_nccl_ib_gid_index_from_nccl_block(self):
        # nccl.ib_gid_index is now wired to export NCCL_IB_GID_INDEX (like ib_hca),
        # so the nccl block -- not env_vars -- is authoritative for the GID index.
        job, orch = _make_job(hosts=["h0", "h1"])  # distributed=True
        job._write_env_script()
        written = "\n".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertIn("export NCCL_IB_GID_INDEX=3", written)
        self.assertIn("export NCCL_IB_HCA=", written)

    def test_single_node_does_not_export_nccl_ib_gid_index(self):
        job, orch = _make_job(hosts=["h0"], distributed=False)
        job._write_env_script()
        written = "\n".join(str(c.args[0]) for c in orch.exec.call_args_list)
        self.assertNotIn("NCCL_IB_GID_INDEX", written)


class StartTrainingTests(unittest.TestCase):
    @patch("cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib.time.sleep")
    def test_launches_per_node_backgrounded(self, _sleep):
        job, orch = _make_job(hosts=["h0", "h1"])
        job.start_training()
        cmds = orch.exec_cmd_list.call_args.args[0]  # last call == launch
        self.assertEqual(len(cmds), 2)
        self.assertTrue(all("nohup bash" in c for c in cmds))
        _sleep.assert_called_once()

    @patch("cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib.time.sleep")
    def test_clears_stale_log_before_launch(self, _sleep):
        # A stale training.log with an old "completed step" marker would make
        # is_complete() pass on the first poll (fail-open). start_training must
        # rm each node's log BEFORE launching.
        job, orch = _make_job(hosts=["h0", "h1"])
        job.start_training()
        clear_cmds = orch.exec_cmd_list.call_args_list[0].args[0]
        self.assertEqual(len(clear_cmds), 2)
        self.assertTrue(all("rm -f" in c and "training.log" in c for c in clear_cmds))
        # ... and the clear precedes the launch.
        launch_cmds = orch.exec_cmd_list.call_args_list[-1].args[0]
        self.assertTrue(all("nohup bash" in c for c in launch_cmds))

    @patch("cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib.time.sleep")
    def test_captures_host_start_time(self, _sleep):
        # start_training records the host-side start time (via orch.all) so the
        # later dmesg scan can bound its window.
        job, orch = _make_job(hosts=["h0", "h1"])
        orch.all = MagicMock()
        orch.all.exec = MagicMock(return_value={"h0": "Mon Jan  2 03:04", "h1": "Mon Jan  2 03:04"})
        _wire_container_exec(orch)
        job.start_training()
        self.assertEqual(job.training_start_time, {"h0": "Mon Jan  2 03:04", "h1": "Mon Jan  2 03:04"})


class PollForCompletionTests(unittest.TestCase):
    _LIB = "cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib"

    @patch(f"{_LIB}.time.sleep")
    def test_completion_path_scans_final_drain_for_nan(self, _sleep):
        # The chunk fetched on the completion path (final "completed step" +
        # anything after) must be error-scanned before declaring success.
        job, _ = _make_job(hosts=["h0"])
        job.is_complete = MagicMock(return_value=True)
        job._drain_new_log_lines = MagicMock(
            side_effect=[
                {},  # loop-body drain: nothing new yet
                {0: "completed step: 2, TFLOP/s/device: NaN\n"},  # completion-path drain
            ]
        )
        with self.assertRaises(RuntimeError):
            job.poll_for_completion()

    @patch(f"{_LIB}.time.sleep")
    def test_completion_path_scans_worker_node_not_just_node0(self, _sleep):
        # A worker-only (non-0) error in the completion window must still raise.
        job, _ = _make_job(hosts=["h0", "h1"])
        job.is_complete = MagicMock(return_value=True)
        job._drain_new_log_lines = MagicMock(
            side_effect=[
                {},
                {1: "some log\nNCCL ERROR: boom\n"},
            ]
        )
        with self.assertRaises(RuntimeError):
            job.poll_for_completion()

    @patch(f"{_LIB}.time.sleep")
    def test_clean_completion_returns(self, _sleep):
        job, _ = _make_job(hosts=["h0"])
        job.is_complete = MagicMock(return_value=True)
        job._drain_new_log_lines = MagicMock(side_effect=[{}, {0: _log()}])
        job.poll_for_completion()  # should not raise


class ScanDmesgForErrorsTests(unittest.TestCase):
    _LIB = "cvs.lib.training.jaxmaxtext.jaxmaxtext_training_lib"

    def _job_with_host(self, **overrides):
        job, orch = _make_job(hosts=["h0", "h1"], **overrides)
        orch.all = MagicMock()
        orch.all.exec = MagicMock(return_value={"h0": "Mon Jan  2 03:04", "h1": "Mon Jan  2 03:04"})
        return job, orch

    @patch(f"{_LIB}.time.sleep")
    @patch(f"{_LIB}._verify_dmesg_for_errors")
    def test_scans_when_enabled_and_started(self, mock_verify, _sleep):
        job, orch = self._job_with_host()
        job.training_start_time = {"h0": "Mon Jan  2 03:00", "h1": "Mon Jan  2 03:00"}
        job.scan_dmesg_for_errors()
        mock_verify.assert_called_once()
        args = mock_verify.call_args.args
        self.assertIs(args[0], orch.all)  # phdl = baremetal handle
        self.assertEqual(args[1], job.training_start_time)  # start of the window

    @patch(f"{_LIB}._verify_dmesg_for_errors")
    def test_skipped_when_disabled(self, mock_verify):
        job, _ = self._job_with_host(verify_dmesg=False)
        job.training_start_time = {"h0": "t"}
        job.scan_dmesg_for_errors()
        mock_verify.assert_not_called()

    @patch(f"{_LIB}._verify_dmesg_for_errors")
    def test_skipped_when_no_start_time(self, mock_verify):
        job, _ = self._job_with_host()
        job.training_start_time = None
        job.scan_dmesg_for_errors()
        mock_verify.assert_not_called()

    @patch(f"{_LIB}.time.sleep")
    @patch(f"{_LIB}._verify_dmesg_for_errors", side_effect=RuntimeError("no passwordless sudo"))
    def test_swallows_scan_failure(self, _mock_verify, _sleep):
        job, _ = self._job_with_host()
        job.training_start_time = {"h0": "t"}
        job.scan_dmesg_for_errors()  # infra failure must not propagate


if __name__ == "__main__":
    unittest.main()
