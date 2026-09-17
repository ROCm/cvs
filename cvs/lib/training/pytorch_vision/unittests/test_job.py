import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cvs.lib.training.pytorch_vision.job import _BoundedHostHandle, PyTorchVisionJob
from cvs.lib.training.pytorch_vision.utils.metrics import ARTIFACT_METRICS


class FakeOrchestrator:
    def __init__(self, response=None):
        self.hosts = ["node0"]
        self.response = response or {"node0": {"exit_code": 0, "output": ""}}
        self.commands = []
        self.all = self

    def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return self.response


def _variant():
    sweep = SimpleNamespace(
        name="cell-w1",
        label="W1-BF16-R224-B128",
        model="resnet50",
        backend="torchvision",
        precision="BF16",
        batch_size=128,
        image_size=224,
        gradient_accumulation_steps=1,
        training_flops_per_image=24600000000,
        data_mode="synthetic",
        dataset_path="",
        rocal_device="gpu",
        augmentation="standard",
        rocal_num_threads=8,
        loader_warmup_steps=5,
        loader_benchmark_steps=20,
    )
    training = SimpleNamespace(
        enabled=True,
        distributed=False,
        master_port=29500,
        phase="performance",
        run_mode="perf",
        gpus_per_node=8,
        warmup_steps=10,
        steps=50,
        max_epochs=None,
        max_duration_seconds=None,
        eval_enabled=False,
        eval_every_epochs=1,
        eval_steps=None,
        eval_sample_count=50000,
        milestone_steps=[100, 500, 1000, 5000],
        num_classes=1000,
        channels_last=True,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        lr_schedule=SimpleNamespace(
            name="constant",
            warmup_epochs=0,
            milestones_epochs=[],
            gamma=0.1,
        ),
        timeout_s=1800,
        omp_num_threads=1,
        verify_dmesg=True,
        peak_tflops_per_gpu=1307.4,
        checkpoint_enabled=True,
        checkpoint_keep_file=False,
        checkpoint_loss_tolerance=1e-5,
        loss_curve=SimpleNamespace(sample_every_steps=1, minimum_points=2),
        accuracy=SimpleNamespace(target_top1_pct=None, target_top5_pct=None),
        scaling_baseline=SimpleNamespace(images_per_sec_total=0.0, num_nodes=1),
        convergence=SimpleNamespace(
            target_top1_pct=None,
            target_eval_loss=None,
            stop_when_reached=False,
        ),
        codecarbon=SimpleNamespace(
            enabled=False,
            required=False,
            measure_power_secs=5,
            country_iso_code=None,
        ),
        gpu_poll_interval_seconds=5,
        env_vars={"NCCL_DEBUG": "WARN"},
        error_patterns={"Process crash": "SIGSEGV"},
    )
    return SimpleNamespace(
        sweep=lambda _name: sweep,
        training=training,
        paths=SimpleNamespace(log_dir="/tmp/logs"),
        gpu_arch="MI325X",
    )


def _artifact():
    artifact = {
        "workload": "W1",
        "model": "resnet50",
        "backend": "torchvision",
        "precision": "BF16",
        "image_size": 224,
        "batch_size_per_gpu": 128,
        "gradient_accumulation_steps": 1,
        "effective_global_batch_size": 1024,
        "world_size": 8,
        "phase": "performance",
        "run_mode": "perf",
        "synthetic_data": True,
        "data_mode": "synthetic",
        "rocal_device": "gpu",
        "augmentation": "standard",
        "lr_schedule": {
            "name": "constant",
            "warmup_epochs": 0,
            "milestones_epochs": [],
            "gamma": 0.1,
        },
        "metrics": {name: index + 1.0 for index, (name, _unit) in enumerate(ARTIFACT_METRICS)},
    }
    artifact["metrics"].update(
        {
            "learning_rate_initial": 0.1,
            "learning_rate_final": 0.1,
        }
    )
    return artifact


class TestPyTorchVisionJob(unittest.TestCase):
    def test_stages_packaged_benchmark(self):
        orch = FakeOrchestrator()
        job = PyTorchVisionJob(orch, _variant(), "w1")
        job.stage_benchmark()
        command = orch.commands[-1][0]
        self.assertIn("base64 -d", command)
        self.assertIn(PyTorchVisionJob.BENCHMARK_PATH, command)

    def test_verifies_exact_gpu_count(self):
        info = {
            "torch": "2.12",
            "torchvision": "0.27",
            "hip": "7.14",
            "cuda": True,
            "gpus": 8,
            "devices": ["AMD Instinct MI325X"] * 8,
            "architectures": ["gfx942"] * 8,
        }
        response = {
            "node0": {
                "exit_code": 0,
                "output": "PYTORCH_VISION_ENV " + json.dumps(info),
            }
        }
        summary = PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").verify_environment()
        self.assertIn("gpus=8", summary)

    def test_rejects_wrong_gpu_count(self):
        info = {
            "torch": "2.12",
            "torchvision": "0.27",
            "hip": "7.14",
            "cuda": True,
            "gpus": 4,
            "devices": ["AMD Instinct MI325X"] * 4,
            "architectures": ["gfx942"] * 4,
        }
        response = {
            "node0": {
                "exit_code": 0,
                "output": "PYTORCH_VISION_ENV " + json.dumps(info),
            }
        }
        with self.assertRaisesRegex(RuntimeError, "requires 8"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").verify_environment()

    def test_rejects_wrong_gpu_model(self):
        info = {
            "torch": "2.12",
            "torchvision": "0.27",
            "hip": "7.14",
            "cuda": True,
            "gpus": 8,
            "devices": ["AMD Instinct MI300X"] * 8,
            "architectures": ["gfx942"] * 8,
        }
        response = {
            "node0": {
                "exit_code": 0,
                "output": "PYTORCH_VISION_ENV " + json.dumps(info),
            }
        }
        with self.assertRaisesRegex(RuntimeError, "requires MI325X"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").verify_environment()

    def test_builds_bounded_eight_rank_command(self):
        command = PyTorchVisionJob(FakeOrchestrator(), _variant(), "w1").build_command()
        self.assertIn("--nproc-per-node=8", command)
        self.assertIn("--precision BF16", command)
        self.assertIn("--gradient-accumulation-steps 1", command)
        self.assertIn("--checkpoint-path", command)
        self.assertIn("--peak-tflops-per-gpu 1307.4", command)
        self.assertIn("--lr-schedule constant", command)
        self.assertIn("timeout --signal=TERM", command)
        self.assertIn("export NCCL_DEBUG=WARN", command)

    def test_single_node_command_uses_standalone_rendezvous(self):
        command = PyTorchVisionJob(FakeOrchestrator(), _variant(), "w1").build_command()
        self.assertIn("--standalone", command)
        self.assertIn("--nnodes=1", command)
        self.assertNotIn("--master-addr", command)
        self.assertIn("export NNODES=1", command)
        self.assertIn("export NODE_RANK=0", command)

    def test_distributed_command_pins_rendezvous_and_per_node_rank(self):
        variant = _variant()
        variant.training.distributed = True
        orch = FakeOrchestrator()
        orch.hosts = ["node0", "node1"]
        job = PyTorchVisionJob(orch, variant, "w1")

        rank0 = job.build_command(node_rank=0)
        rank1 = job.build_command(node_rank=1)

        for command in (rank0, rank1):
            self.assertNotIn("--standalone", command)
            self.assertIn("--nnodes=2", command)
            self.assertIn("--master-addr=node0", command)
            self.assertIn("--master-port=29500", command)
            self.assertIn("export NNODES=2", command)
        self.assertIn("--node-rank=0", rank0)
        self.assertIn("export NODE_RANK=0", rank0)
        self.assertIn("--node-rank=1", rank1)
        self.assertIn("export NODE_RANK=1", rank1)

    def test_distributed_rank_overrides_configured_env_vars(self):
        """A single-node NNODES/NODE_RANK left in the config must not leak into
        a multi-node launch and silently collapse it to one rank."""
        variant = _variant()
        variant.training.distributed = True
        variant.training.env_vars = {"NNODES": "1", "NODE_RANK": "0"}
        orch = FakeOrchestrator()
        orch.hosts = ["node0", "node1"]

        command = PyTorchVisionJob(orch, variant, "w1").build_command(node_rank=1)

        self.assertIn("export NNODES=2", command)
        self.assertIn("export NODE_RANK=1", command)
        self.assertNotIn("export NNODES=1", command)

    def test_builds_target_accuracy_early_stop_command(self):
        variant = _variant()
        variant.training.run_mode = "train_to_accuracy"
        variant.training.phase = "accuracy"
        variant.training.max_epochs = 90
        variant.training.accuracy = SimpleNamespace(
            target_top1_pct=75.5,
            target_top5_pct=92.5,
        )
        variant.training.convergence = SimpleNamespace(
            target_top1_pct=75.5,
            target_eval_loss=None,
            stop_when_reached=True,
        )
        command = PyTorchVisionJob(FakeOrchestrator(), variant, "w1").build_command()
        self.assertIn("--max-epochs 90", command)
        self.assertIn("--convergence-top1 75.5", command)
        self.assertIn("--target-top5 92.5", command)
        self.assertIn("--stop-on-convergence", command)

    def test_parses_namespaced_artifact(self):
        response = {
            "node0": {
                "exit_code": 0,
                "output": json.dumps(_artifact()),
            }
        }
        results = PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").parse_results()
        self.assertEqual(results["node0"]["training.images_per_sec"], 1.0)

    def test_scaling_efficiency_absent_without_a_calibrated_baseline(self):
        """The default baseline is 0.0, so the metric must simply not appear
        rather than show up as a zero that a threshold could gate on."""
        response = {"node0": {"exit_code": 0, "output": json.dumps(_artifact())}}
        results = PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").parse_results()
        self.assertNotIn("training.scaling_efficiency_pct", results["node0"])

    def test_scaling_efficiency_uses_node_count_and_configured_baseline(self):
        variant = _variant()
        # Artifact reports 1.0 images/sec; a 0.5 single-node reference over two
        # nodes gives an ideal of 1.0, so efficiency is 100%.
        variant.training.scaling_baseline = SimpleNamespace(images_per_sec_total=0.5, num_nodes=1)
        response = {"node0": {"exit_code": 0, "output": json.dumps(_artifact())}}
        orch = FakeOrchestrator(response)
        orch.hosts = ["node0", "node1"]

        results = PyTorchVisionJob(orch, variant, "w1").parse_results()

        self.assertAlmostEqual(results["node0"]["training.scaling_efficiency_pct"], 100.0)

    def test_energy_efficiency_uses_artifact_processed_image_count(self):
        variant = _variant()
        variant.training.codecarbon.enabled = True
        artifact = _artifact()
        artifact["processed_images"] = 1024
        artifact["completed_steps"] = 1
        artifact["codecarbon"] = {
            "version": "3.2.4",
            "tracker": "amdsmi",
            "gpu_count": 8,
            "active": True,
            "emissions_kg_co2eq": 0.1,
            "energy_kwh": 0.5,
            "duration_seconds": 10,
        }
        response = {"node0": {"exit_code": 0, "output": json.dumps(artifact)}}
        results = PyTorchVisionJob(FakeOrchestrator(response), variant, "w1").parse_results()
        self.assertEqual(results["node0"]["training.images_per_kwh"], 2048)

    def test_rejects_mismatched_artifact_metadata(self):
        artifact = _artifact()
        artifact["batch_size_per_gpu"] = 64
        response = {"node0": {"exit_code": 0, "output": json.dumps(artifact)}}
        with self.assertRaisesRegex(RuntimeError, "batch_size_per_gpu"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").parse_results()

    def test_rejects_wrong_learning_rate_trajectory(self):
        artifact = _artifact()
        artifact["metrics"]["learning_rate_final"] = 0.2
        response = {"node0": {"exit_code": 0, "output": json.dumps(artifact)}}
        with self.assertRaisesRegex(RuntimeError, "learning-rate trajectory"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").parse_results()

    def test_rejects_invalid_epoch_accounting(self):
        variant = _variant()
        variant.training.max_epochs = 1
        artifact = _artifact()
        artifact.update(
            {
                "completed_epochs": 1,
                "samples_per_rank_per_epoch": 10,
                "batches_per_epoch": 0,
                "steps_per_epoch": 0,
            }
        )
        response = {"node0": {"exit_code": 0, "output": json.dumps(artifact)}}
        with self.assertRaisesRegex(RuntimeError, "epoch accounting"):
            PyTorchVisionJob(FakeOrchestrator(response), variant, "w1").parse_results()

    def test_rejects_nonzero_remote_exit(self):
        response = {"node0": {"exit_code": 7, "output": "torchrun failed"}}
        readings = [{"gpu.used_vram": 16, "gpu.max_used_vram": 2}]
        with (
            patch("cvs.lib.training.pytorch_vision.job.start_gpu_poller"),
            patch("cvs.lib.training.pytorch_vision.job.stop_and_collect_gpu_poller", return_value=readings),
            self.assertRaisesRegex(RuntimeError, "torchrun failed"),
        ):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").run_benchmark()

    def test_run_captures_host_timestamp(self):
        orch = FakeOrchestrator()
        readings = [{"gpu.used_vram": 16, "gpu.max_used_vram": 2}]
        with (
            patch("cvs.lib.training.pytorch_vision.job.start_gpu_poller"),
            patch("cvs.lib.training.pytorch_vision.job.stop_and_collect_gpu_poller", return_value=readings),
        ):
            PyTorchVisionJob(orch, _variant(), "w1").run_benchmark()
        self.assertEqual(orch.commands[0][0], "date")

    def test_monitor_records_maximum_per_device_used_memory(self):
        job = PyTorchVisionJob(FakeOrchestrator(), _variant(), "w1")
        job._set_monitor_metrics(
            [
                {"gpu.used_vram": 16, "gpu.max_used_vram": 2},
                {"gpu.used_vram": 24, "gpu.max_used_vram": 4},
            ]
        )
        self.assertEqual(job.monitor_metrics["training.peak_memory_used_mb"], 4)

    def test_dmesg_scan_can_be_disabled(self):
        variant = _variant()
        variant.training.verify_dmesg = False
        orch = FakeOrchestrator()
        PyTorchVisionJob(orch, variant, "w1").scan_dmesg_for_errors()
        self.assertEqual(orch.commands, [])

    def test_scans_dmesg_over_training_window(self):
        orch = FakeOrchestrator({"node0": "Mon Jan  2 03:04:05 UTC 2026"})
        job = PyTorchVisionJob(orch, _variant(), "w1")
        job.training_start_time = {"node0": "Mon Jan  2 03:03:05 UTC 2026"}
        with patch("cvs.lib.verify_lib.verify_dmesg_for_errors") as verify:
            job.scan_dmesg_for_errors()
        args = verify.call_args.args
        self.assertIs(args[0]._handle, orch.all)
        self.assertEqual(args[1], job.training_start_time)
        self.assertEqual(args[2], {"node0": "Mon Jan  2 03:04:05 UTC 2026"})
        self.assertFalse(verify.call_args.kwargs["till_end_flag"])

    def test_bounded_host_handle_forces_noninteractive_sudo(self):
        orch = FakeOrchestrator({"node0": {"exit_code": 0, "output": "clean"}})
        output = _BoundedHostHandle(orch, orch.hosts).exec("sudo dmesg --time-format iso -x")
        self.assertEqual(output, {"node0": "clean"})
        command, kwargs = orch.commands[-1]
        self.assertIn("sudo -n dmesg", command)
        self.assertIn("IPVS: .*no destination available", command)
        self.assertEqual(kwargs["timeout"], 60)
        self.assertTrue(kwargs["detailed"])

    def test_bounded_host_handle_rejects_remote_failure(self):
        orch = FakeOrchestrator({"node0": {"exit_code": 1, "output": "denied"}})
        with self.assertRaisesRegex(RuntimeError, "denied"):
            _BoundedHostHandle(orch, orch.hosts).exec("sudo dmesg")

    def test_rejects_invalid_environment_name(self):
        variant = _variant()
        variant.training.env_vars = {"BAD-NAME": "1"}
        with self.assertRaisesRegex(ValueError, "invalid environment"):
            PyTorchVisionJob(FakeOrchestrator(), variant, "w1").build_command()

    def test_rejects_configured_training_log_error(self):
        response = {"node0": {"exit_code": 0, "output": "worker died with SIGSEGV"}}
        with self.assertRaisesRegex(RuntimeError, "Process crash"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").verify_training_log()


if __name__ == "__main__":
    unittest.main()
