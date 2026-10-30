'''Unit tests for the vLLM config rules in cvs.lib.inference.utils.vllm_schema.'''

import unittest

from pydantic import ValidationError

from cvs.lib.inference.utils.vllm_schema import (
    ContainerEnvTokenError,
    RunCell,
    VariantConfig,
    serialize_cli_options,
)
from cvs.unittests.support import FAKE_CRED


CELL = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
FAKE_CRED_FILE = "/home/test/.hf_token"


def _config(**overrides):
    value = {
        "enforce_thresholds": False,
        "threshold_json": "threshold.json",
        "ib_hca_devices": "auto",
        "ib_netdev": "eth0",
        "paths": {
            "shared_fs": "/home/test",
            "models_dir": "/models",
            "log_dir": "/logs",
            "hf_token_file": FAKE_CRED_FILE,
        },
        "container": {"name": "vllm", "image": "image", "runtime": {"name": "docker", "args": {}}},
        "server_params": {
            "model": "/models/DeepSeek-R1-0528-FP8",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
        },
        "benchmark_params": {"num_prompts": 25},
        "sweeps": {CELL: {"num_prompts": 50}},
        "runs": [CELL],
        "thresholds": {},
    }
    value.update(overrides)
    return value


def _with_args(args):
    config = _config()
    config["container"]["runtime"]["args"] = args
    return config


class TestRunCell(unittest.TestCase):
    def test_parses_only_canonical_cell_keys(self):
        cell = RunCell.parse(CELL)
        self.assertEqual((cell.isl, cell.osl, cell.tp, cell.pp, cell.concurrency), (1024, 1024, 8, 2, 16))
        for invalid in (
            "ISL=1024,OSL=1024,TP=8,CONC=16",
            "OSL=1024,ISL=1024,TP=8,PP=2,CONC=16",
            "ISL=1024, OSL=1024,TP=8,PP=2,CONC=16",
            "ISL=0,OSL=1024,TP=8,PP=2,CONC=16",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    RunCell.parse(invalid)


class TestRuns(unittest.TestCase):
    def test_resolves_selected_run_with_cell_overrides(self):
        variant = VariantConfig(**_config())
        (run,) = variant.resolved_runs()
        self.assertEqual(run.cell.key, CELL)
        self.assertEqual(run.benchmark_params["num_prompts"], 50)
        self.assertEqual(run.benchmark_params["random_input_len"], 1024)
        self.assertEqual(run.benchmark_params["random_output_len"], 1024)
        self.assertEqual(run.benchmark_params["max_concurrency"], 16)

    def test_rejects_unknown_root_and_legacy_shape(self):
        for kwargs in ({"unexpected": 1}, {"schema_version": 1}, {"params": {}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    VariantConfig(**_config(**kwargs))

    def test_requires_nonempty_explicit_runs(self):
        with self.assertRaisesRegex(ValidationError, "nonempty explicit"):
            VariantConfig(**_config(runs=[]))

    def test_rejects_mismatched_parallelism_and_missing_enforced_threshold(self):
        wrong_cell = "ISL=1024,OSL=1024,TP=4,PP=2,CONC=16"
        with self.assertRaisesRegex(ValidationError, "conflicts"):
            VariantConfig(**_config(sweeps={wrong_cell: {}}, runs=[wrong_cell]))
        with self.assertRaisesRegex(ValidationError, "missing threshold"):
            VariantConfig(**_config(enforce_thresholds=True))


class TestContainer(unittest.TestCase):
    def test_accepts_complete_container_network_environment(self):
        config = _config()
        config["container"]["env"] = {
            "NCCL_IB_HCA": "rdma0",
            "NCCL_SOCKET_IFNAME": "eno0",
            "GLOO_SOCKET_IFNAME": "eno0",
            "TP_SOCKET_IFNAME": "eno0",
        }
        variant = VariantConfig(**config)
        self.assertEqual(variant.container.nccl_ib_hcas, ["rdma0"])
        self.assertTrue(variant.container.socket_env_configured)

    def test_rejects_incomplete_container_socket_environment(self):
        names = ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")
        for missing in names:
            with self.subTest(missing=missing):
                config = _config()
                config["container"]["env"] = {name: "eno0" for name in names if name != missing}
                with self.assertRaisesRegex(ValidationError, "must set all socket interface"):
                    VariantConfig(**config)

    def test_rejects_empty_container_hca(self):
        config = _config()
        config["container"]["env"] = {"NCCL_IB_HCA": " , "}
        with self.assertRaisesRegex(ValidationError, "must name at least one device"):
            VariantConfig(**config)

    def test_rejects_hf_token_in_container_env_without_echoing_value(self):
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "hf_token", "Hf_Token"):
            for extra in ({}, {"NCCL_SOCKET_IFNAME": "eno0"}):
                with self.subTest(name=name, extra=extra):
                    config = _config()
                    config["container"]["env"] = {**extra, name: FAKE_CRED}
                    with self.assertRaises(ContainerEnvTokenError) as ctx:
                        VariantConfig(**config)
                    self.assertIn(f"container.env must not set ['{name}']", str(ctx.exception))
                    self.assertIn("paths.hf_token_file", str(ctx.exception))
                    self.assertNotIn(FAKE_CRED, repr(ctx.exception.args))

    def test_runtime_args_accept_docker_keys_but_not_env(self):
        variant = VariantConfig(**_with_args({"shm_size": "64g", "cap_add": ["SYS_PTRACE"], "ulimit": ["memlock=-1"]}))
        self.assertEqual(variant.container.model_dump()["runtime"]["args"]["shm_size"], "64g")
        for args in ({"env": {"NCCL_DEBUG": "INFO"}}, {"unknown_flag": True}):
            with self.subTest(args=args):
                with self.assertRaises(ValidationError):
                    VariantConfig(**_with_args(args))


class TestCommentKeysAndOptions(unittest.TestCase):
    def test_comment_keys_are_ignored_but_option_payloads_keep_theirs(self):
        config = _config(
            _comment="root",
            server_params={
                "_comment": "server",
                "model": "/models/x",
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 2,
                "json_arg": {"_comment": "upstream payload"},
            },
            sweeps={"_comment": "cells", CELL: {"_comment": "cell", "json_arg": {"_comment": "upstream payload"}}},
            accuracy={"_comment": "accuracy", "tasks": [{"_comment": "task", "id": "gsm8k", "task": "gsm8k"}]},
        )
        config["container"]["runtime"]["args"] = {"_comment_mounts": "notes", "volumes": ["/data:/data"]}
        variant = VariantConfig(**config)
        self.assertEqual(variant.server_params.extra_options(), {"json_arg": {"_comment": "upstream payload"}})
        self.assertEqual(variant.sweeps, {CELL: {"json_arg": {"_comment": "upstream payload"}}})
        self.assertEqual(variant.accuracy.tasks[0].id, "gsm8k")

    def test_serializes_documented_value_shapes(self):
        options = {
            "trust_remote_code": True,
            "max_model_len": 8192,
            "served_model_name": ["first", "second"],
            "structured_outputs": {"json": {"type": "object"}},
            "optional": None,
        }
        self.assertEqual(
            serialize_cli_options(options),
            [
                "--trust-remote-code",
                "--max-model-len",
                "8192",
                "--served-model-name",
                "first",
                "second",
                "--structured-outputs",
                '{"json":{"type":"object"}}',
            ],
        )

    def test_rejects_ambiguous_false_and_reserved_names(self):
        with self.assertRaises(ValueError):
            serialize_cli_options({"enforce_eager": False})
        with self.assertRaisesRegex(ValidationError, "cannot override harness"):
            VariantConfig(
                **_config(
                    server_params={
                        "model": "/models/x",
                        "tensor_parallel_size": 8,
                        "pipeline_parallel_size": 2,
                        "nnodes": 2,
                    }
                )
            )

        with self.assertRaisesRegex(ValidationError, "benchmark_params cannot override harness"):
            VariantConfig(**_config(benchmark_params={"num_prompts": 25, "percentile_metrics": "ttft"}))


class TestThresholds(unittest.TestCase):
    def _variant(self, thresholds, enforce=False):
        return VariantConfig(**_config(enforce_thresholds=enforce, thresholds=thresholds))

    def test_partial_and_empty_cells_load_with_enforcement_off_and_on(self):
        for thresholds in ({CELL: {"output_throughput": {"kind": "min", "value": 0}}}, {CELL: {}}):
            for enforce in (False, True):
                with self.subTest(thresholds=thresholds, enforce=enforce):
                    self.assertEqual(self._variant(thresholds, enforce=enforce).thresholds, thresholds)

    def test_enforcement_still_requires_selected_run_cell(self):
        self._variant({}, enforce=False)
        with self.assertRaisesRegex(ValidationError, "missing threshold coverage"):
            self._variant({}, enforce=True)

    def test_underscore_metrics_are_ignored_and_other_unknown_metrics_rejected(self):
        spec = {"kind": "min", "value": 0}
        variant = self._variant({CELL: {"_comment": "calibrate", "_typo": spec, "_": spec, "output_throughput": spec}})
        self.assertEqual(variant.thresholds, {CELL: {"output_throughput": spec}})
        with self.assertRaisesRegex(ValidationError, "unknown vLLM threshold metric 'notes'"):
            self._variant({CELL: {"notes": spec}})

    def test_rejects_specs_outside_the_vllm_direction_rule(self):
        for metric, spec in (
            ("output_throughput", {"kind": "max", "value": 1}),
            ("mean_ttft_ms", {"kind": "min", "value": 1}),
            ("output_throughput", {"kind": "min_tok_s", "value": 1}),
            ("output_throughput", {"kind": "within", "value": 1, "tolerance_pct": 5}),
            ("output_throughput", {"kind": "min", "value": 1, "optional": False}),
            ("output_throughput", {"kind": "min", "value": 1, "unit": "tok/s"}),
            ("output_throughput", {"kind": "min", "value": "1"}),
            ("client.output_throughput", {"kind": "min", "value": 1}),
        ):
            with self.subTest(metric=metric, spec=spec):
                with self.assertRaises(ValidationError):
                    self._variant({CELL: {metric: spec}})

    def test_accuracy_thresholds_are_not_metric_cells(self):
        accuracy = {"gsm8k": {"gsm8k.exact_match__strict-match": {"kind": "min", "value": 0, "tolerance_pct": 5}}}
        cell = {"output_throughput": {"kind": "min", "value": 0}}
        variant = self._variant({CELL: cell, "accuracy": accuracy}, enforce=True)
        self.assertEqual(variant.thresholds["accuracy"], accuracy)
        with self.assertRaisesRegex(ValidationError, "threshold cells"):
            self._variant({CELL: cell, "accuracy": accuracy, "acuracy": {}}, enforce=True)


if __name__ == "__main__":
    unittest.main()
