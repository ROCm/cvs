import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cvs.lib.training.pytorch_vision.job import _BoundedHostHandle, PyTorchVisionJob
from cvs.lib.training.pytorch_vision.utils.metrics import METRICS


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
    )
    training = SimpleNamespace(
        gpus_per_node=8,
        warmup_steps=10,
        steps=50,
        num_classes=1000,
        channels_last=True,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        timeout_s=1800,
        omp_num_threads=1,
        verify_dmesg=True,
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
    return {
        "workload": "W1",
        "model": "resnet50",
        "backend": "torchvision",
        "precision": "BF16",
        "image_size": 224,
        "batch_size_per_gpu": 128,
        "world_size": 8,
        "synthetic_data": True,
        "metrics": {name: index + 1.0 for index, (name, _unit) in enumerate(METRICS)},
    }


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
        self.assertIn("timeout --signal=TERM", command)
        self.assertIn("export NCCL_DEBUG=WARN", command)

    def test_parses_namespaced_artifact(self):
        response = {
            "node0": {
                "exit_code": 0,
                "output": json.dumps(_artifact()),
            }
        }
        results = PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").parse_results()
        self.assertEqual(results["node0"]["training.images_per_sec"], 1.0)

    def test_rejects_mismatched_artifact_metadata(self):
        artifact = _artifact()
        artifact["batch_size_per_gpu"] = 64
        response = {"node0": {"exit_code": 0, "output": json.dumps(artifact)}}
        with self.assertRaisesRegex(RuntimeError, "batch_size_per_gpu"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").parse_results()

    def test_rejects_nonzero_remote_exit(self):
        response = {"node0": {"exit_code": 7, "output": "torchrun failed"}}
        with self.assertRaisesRegex(RuntimeError, "torchrun failed"):
            PyTorchVisionJob(FakeOrchestrator(response), _variant(), "w1").run_benchmark()

    def test_run_captures_host_timestamp(self):
        orch = FakeOrchestrator()
        PyTorchVisionJob(orch, _variant(), "w1").run_benchmark()
        self.assertEqual(orch.commands[0][0], "date")

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
