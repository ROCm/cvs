'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.lib.inference.utils.vllm_config_loader.
'''

import copy
import json
import logging
import tempfile
import unittest
import warnings
from pathlib import Path

from pydantic import ValidationError

from cvs.lib.inference.unittests.fake_orch import FakeOrch
from cvs.lib.inference.utils.vllm_config_loader import (
    ConfigError,
    VllmConfig,
    load_variant,
    merged_container,
    resolve_old_vllm_keys,
)
from cvs.lib.inference.utils.vllm_verification import evaluate_metric_verdicts
from cvs.lib.inference.vllm_job import VllmJob
from cvs.lib.inference.vllm_topology import EffectiveVllmTopology
from cvs.parsers.schemas import validation_problem_lines
from cvs.unittests.support import FAKE_CRED, write_filled_config

CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
SHIPPED = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"


def new_config(**overrides):
    config = {
        "enforce_thresholds": True,
        "threshold_json": "llama_threshold.json",
        "paths": {"shared_fs": "/home/u", "models_dir": "/models", "log_dir": "/home/u/LOGS", "hf_token_file": "/t"},
        "container": {"name": "vllm_llama", "image": "rocm/vllm:latest", "runtime": {"name": "docker"}},
        "server_params": {"model": "meta-llama/Llama-3.3-70B", "tp": 8, "engine_args": {"kv_cache_dtype": "fp8"}},
        "benchmark_params": {"num_prompts": 320},
        "sweeps": {CELL: {}},
        "runs": [CELL],
        "thresholds": {CELL: {"output_throughput": {"kind": "min", "value": 100}}},
    }
    config.update(overrides)
    return config


def old_config():
    return {
        "enforce_thresholds": False,
        "threshold_json": "llama_threshold.json",
        "paths": {
            "shared_fs": "/home/{user-id}",
            "models_dir": "/models",
            "log_dir": "{shared_fs}/LOGS",
            "hf_token_file": "/t",
        },
        "container": {"name": "vllm_llama", "image": "rocm/vllm:latest", "runtime": {"name": "docker"}},
        "server_params": {
            "backend": "vllm",
            "model": "meta-llama/Llama-3.3-70B",
            "tensor_parallel_size": "8",
            "pipeline_parallel_size": "1",
            "port": "8888",
            "dist_init_port": "29501",
            "server_poll_iterations": "60",
            "server_poll_wait_s": "60",
            "server_warmup_wait_s": "330",
            "distributed_executor_backend": "mp",
            "kv_cache_dtype": "fp8",
        },
        "benchmark_params": {
            "backend": "vllm",
            "dataset_name": "random",
            "num_prompts": "320",
            "client_poll_iterations": "90",
            "client_poll_wait_s": "60",
            "client_initial_wait_s": "120",
            "trust_remote_code": False,
            "ignore_eos": True,
        },
        "sweeps": {CELL: {}},
        "runs": [CELL],
        "accuracy": {"tasks": [{"id": "gsm8k_flex", "task": "gsm8k"}]},
    }


def old_thresholds():
    return {
        "_comment": "file",
        CELL: {"output_throughput": {"kind": "min", "value": 100}, "p99_ttft_ms": {"kind": "max", "value": 900}},
        "accuracy": {"gsm8k_flex": {"gsm8k.exact_match__flexible-extract": {"kind": "min", "value": 0.5}}},
    }


def validate(raw):
    return VllmConfig.model_validate(raw)


def problems(raw):
    try:
        validate(raw)
    except ValidationError as exc:
        return validation_problem_lines(exc)
    return []


def convert(raw):
    converted, message = resolve_old_vllm_keys(raw)
    return converted, message


def client_argv(config, hosts=("node0",)):
    orch = FakeOrch(hosts=list(hosts))
    run = config.resolved_runs()[0]
    job = VllmJob(
        orch=orch,
        variant=config,
        isl=run.cell.isl,
        osl=run.cell.osl,
        concurrency=run.cell.concurrency,
        num_prompts=run.benchmark_params["num_prompts"],
        benchmark_params=run.benchmark_params,
    )
    job.run_client()
    return orch.exec_on_head_commands[-1]


class TestVllmConfigModel(unittest.TestCase):
    def test_vllm_block(self):
        self.assertEqual(validate(new_config()).vllm.distributed_executor_backend, "mp")
        config = validate(new_config(vllm={"distributed_executor_backend": "ray"}))
        self.assertEqual(config.vllm.distributed_executor_backend, "ray")
        self.assertTrue(problems(new_config(vllm={"distributed_executor_backend": "local"})))

    def test_vllm_not_implemented_fields(self):
        cases = {
            "accuracy.quant_parity": {"accuracy": {"quant_parity": {"enabled": True}}},
            "accuracy.mtp_quality": {"accuracy": {"mtp_quality": {"enabled": True}}},
            "accuracy.custom_eval": {"accuracy": {"custom_eval": [{"type": "niah", "id": "n", "isl": 8192}]}},
            "benchmark_params.scaling_baseline_output_throughput": {
                "benchmark_params": {"num_prompts": 320, "scaling_baseline_output_throughput": 900.0}
            },
            "server_params.prefill_nnodes": {"server_params": {"model": "m", "tp": 8, "prefill_nnodes": 1}},
            "server_params.long_context": {
                "server_params": {"model": "m", "tp": 8, "long_context": {"chunked_prefill_size": 8192}}
            },
            "server_params.engine_args.speculative_config": {
                "server_params": {"model": "m", "tp": 8, "engine_args": {"speculative_config": {"method": "mtp"}}}
            },
            "server_params.dp": {"server_params": {"model": "m", "tp": 8, "dp": 2}},
            "server_params.ep": {"server_params": {"model": "m", "tp": 8, "ep": True}},
        }
        for path, overrides in cases.items():
            with self.subTest(path=path):
                lines = problems(new_config(**overrides))
                self.assertEqual(len(lines), 1)
                self.assertIn("not implemented for vLLM", lines[0])
                self.assertIn(path, lines[0])
        lines = problems(new_config(atom={"driver": "vllm"}))
        self.assertEqual(len(lines), 1)
        self.assertIn("atom: not implemented for vLLM", lines[0])

    def test_vllm_not_implemented_metrics(self):
        for metric in (
            "output_throughput_per_gpu",
            "mfu",
            "spec_acceptance_rate",
            "mean_spec_acceptance_length",
            "mtp_quality.chat_template_match",
            "quant_parity.probe_match",
            "scaling_efficiency_pct",
            "pass_rate",
        ):
            with self.subTest(metric=metric):
                lines = problems(new_config(thresholds={CELL: {metric: {"kind": "min", "value": 1}}}))
                self.assertEqual(len(lines), 1)
                self.assertIn("not implemented for vLLM", lines[0])
                self.assertIn(metric, lines[0])
        accuracy = {"accuracy": {"custom_eval": {"niah": {"pass_rate": {"kind": "min", "value": 0.8}}}}}
        lines = problems(new_config(thresholds={CELL: {}, **accuracy}))
        self.assertIn("thresholds accuracy.custom_eval", lines[0])

    def test_vllm_not_implemented_all_reported(self):
        raw = new_config(
            accuracy={"quant_parity": {"enabled": True}},
            thresholds={CELL: {"mfu": {"kind": "min", "value": 1}}},
        )
        raw["server_params"]["dp"] = 2
        lines = problems(raw)
        self.assertEqual(len(lines), 1)
        for name in ("server_params.dp", "accuracy.quant_parity", "mfu"):
            self.assertIn(name, lines[0])

    def test_vllm_defaults_pass(self):
        raw = new_config(
            accuracy={
                "custom_eval": [],
                "quant_parity": {"enabled": False, "probe_prompt": "The capital of France is"},
                "mtp_quality": {"enabled": False},
            }
        )
        raw["server_params"].update({"dp": 1, "ep": False, "prefill_nnodes": None, "long_context": {}})
        raw["benchmark_params"]["scaling_baseline_output_throughput"] = None
        self.assertEqual(problems(raw), [])

    def test_vllm_checks(self):
        bad = {
            "harness flag in engine_args": {"server_params": {"model": "m", "tp": 8, "engine_args": {"port": 9}}},
            "vLLM flag in engine_args": {
                "server_params": {"model": "m", "tp": 8, "engine_args": {"tensor_parallel_size": 8}}
            },
            "harness flag in client_args": {
                "benchmark_params": {"num_prompts": 1, "client_args": {"max_concurrency": 4}}
            },
            "false engine flag": {"server_params": {"model": "m", "tp": 8, "engine_args": {"enforce_eager": False}}},
            "client wait flag in client_args": {
                "benchmark_params": {"num_prompts": 1, "client_args": {"client_poll_iterations": 4}}
            },
            "one socket variable": {
                "container": {"name": "c", "image": "i", "env": {"NCCL_SOCKET_IFNAME": "eth0"}},
            },
            "empty NCCL_IB_HCA": {"container": {"name": "c", "image": "i", "env": {"NCCL_IB_HCA": " , "}}},
            "threshold cell not in sweeps": {
                "thresholds": {CELL: {}, "ISL=1,OSL=1,TP=8,PP=1,CONC=1": {}},
            },
        }
        for name, overrides in bad.items():
            with self.subTest(name=name):
                self.assertTrue(problems(new_config(**overrides)))
        for nnodes in (None, 2):
            with self.subTest(nnodes=nnodes):
                lines = problems(new_config(server_params={"model": "m", "tp": 8, "nnodes": nnodes}))
                self.assertEqual(lines, ["server_params: server_params cannot override harness fields: ['nnodes']"])
        sockets = {name: "eth0" for name in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")}
        raw = new_config(container={"name": "c", "image": "i", "env": sockets})
        raw["benchmark_params"]["client_args"] = {"ignore_eos": False, "temperature": 0.0}
        self.assertEqual(problems(raw), [])

    def test_vllm_threshold_metrics(self):
        self.assertEqual(problems(new_config(thresholds={CELL: {"p99_ttft_ms": {"kind": "max", "value": 1}}})), [])
        bad = {
            "unknown metric": {"made_up_metric": {"kind": "min", "value": 1}},
            "wrong direction": {"p99_ttft_ms": {"kind": "min", "value": 1}},
            "optional": {"p99_ttft_ms": {"kind": "max", "value": 1, "optional": False}},
            "info": {"output_throughput": {"kind": "info", "value": 1}},
            "underscore metric": {"_output_throughput": {"kind": "min", "value": 1}},
        }
        for name, cell in bad.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = _write(tmp, new_config(), {CELL: cell})
                    with self.assertRaises(ValueError):
                        load_variant(path, {}, warn=False)


class TestVerdicts(unittest.TestCase):
    def test_vllm_missing_value_as_on_main(self):
        config = validate(new_config())
        verdicts = evaluate_metric_verdicts(
            {"node0": {"p99_ttft_ms": 10.0}}, config.thresholds.get(CELL), enforce_thresholds=True
        )
        row = next(verdict for verdict in verdicts if verdict["metric"] == "output_throughput")
        self.assertEqual(row["status"], "fail")
        self.assertTrue(row["enforced"])

    def test_vllm_verdict_same_after_conversion(self):
        old = {CELL: {"p99_ttft_ms": {"kind": "max", "value": 5}}}
        actuals = {"node0": {"p99_ttft_ms": 7.0}}
        raw = old_config()
        raw["thresholds"] = old
        converted, _ = convert(raw)
        config = validate(converted)
        verdicts = evaluate_metric_verdicts(actuals, config.thresholds.get(CELL), enforce_thresholds=True)
        expected = evaluate_metric_verdicts(actuals, old[CELL], enforce_thresholds=True)
        self.assertEqual(verdicts, expected)
        self.assertEqual([(row["metric"], row["status"]) for row in verdicts], [("p99_ttft_ms", "fail")])

    def test_vllm_verdict_same_after_rename(self):
        old = {CELL: {"queue_time_p50_ms": {"kind": "max", "value": 5}}}
        actuals = {"node0": {"p50_queue_time_ms": 7.0}}
        raw = old_config()
        raw["thresholds"] = old
        converted, _ = convert(raw)
        config = validate(converted)
        verdicts = evaluate_metric_verdicts(actuals, config.thresholds.get(CELL), enforce_thresholds=True)
        expected = evaluate_metric_verdicts(
            actuals, {"p50_queue_time_ms": {"kind": "max", "value": 5}}, enforce_thresholds=True
        )
        self.assertEqual(verdicts, expected)
        self.assertEqual([(row["metric"], row["status"]) for row in verdicts], [("p50_queue_time_ms", "fail")])


class TestShippedConfigs(unittest.TestCase):
    def test_vllm_shipped_configs_load(self):
        templates = sorted(path for path in SHIPPED.glob("mi*_vllm_*.json") if "threshold" not in path.name)
        self.assertEqual(len(templates), 33)
        for template in templates:
            with self.subTest(config=template.name), tempfile.TemporaryDirectory() as tmp:
                config = load_variant(write_filled_config(template, tmp), {"username": "u"}, warn=False)
                self.assertTrue(config.runs)


class TestClientFlags(unittest.TestCase):
    def test_vllm_client_flags_kept(self):
        def argv_for(**benchmark):
            raw = old_config()
            raw["benchmark_params"].update(benchmark)
            converted, _ = convert(raw)
            return client_argv(validate(converted))

        default = argv_for()
        self.assertIn(" --dataset-name random ", default)
        self.assertIn(" --ignore-eos", default)
        self.assertNotIn("--trust-remote-code", default)
        self.assertNotIn("--ignore-eos", argv_for(ignore_eos=False))
        self.assertIn(" --dataset-name sharegpt ", argv_for(dataset_name="sharegpt"))
        self.assertIn(" --ignore-eos --trust-remote-code ", argv_for(trust_remote_code=True))
        self.assertEqual(argv_for(trust_remote_code=False), default)
        self.assertIn(" --temperature 0.0", argv_for(temperature=0.0))


class TestOldKeys(unittest.TestCase):
    def test_old_key_rows(self):
        converted, message = convert(old_config())
        server = converted["server_params"]
        self.assertEqual((server["tp"], server["pp"], server["master_port"]), ("8", "1", "29501"))
        self.assertEqual(server["engine_args"], {"kv_cache_dtype": "fp8"})
        self.assertEqual(converted["vllm"], {"distributed_executor_backend": "mp"})
        self.assertEqual(converted["accuracy"], {"lm_eval": {"tasks": [{"id": "gsm8k_flex", "task": "gsm8k"}]}})
        self.assertEqual(converted["benchmark_params"], {"num_prompts": "320"})
        for name in (
            "server_params.tensor_parallel_size -> server_params.tp",
            "server_params.dist_init_port -> server_params.master_port",
            "server_params.kv_cache_dtype -> server_params.engine_args.kv_cache_dtype",
            "server_params.backend (dropped)",
            "benchmark_params.ignore_eos (dropped)",
            "accuracy.tasks -> accuracy.lm_eval.tasks",
        ):
            self.assertIn(name, message)
        config = validate(converted)
        self.assertEqual(config.server_params.tp, 8)

        raw = old_config()
        raw["benchmark_params"].update({"temperature": 0.0, "trust_remote_code": True})
        converted, _ = convert(raw)
        self.assertEqual(converted["benchmark_params"]["client_args"], {"trust_remote_code": True, "temperature": 0.0})

        raw = old_config()
        raw["server_params"]["max_model_len"] = 2056
        self.assertEqual(convert(raw)[0]["server_params"]["max_model_length"], 2056)

        raw = old_config()
        del raw["benchmark_params"]
        self.assertEqual(convert(raw)[0]["benchmark_params"], {"num_prompts": 3200})

        for section, key, value in (
            ("server_params", "backend", "sglang"),
            ("benchmark_params", "backend", "openai"),
            ("server_params", "max_model_len", True),
        ):
            with self.subTest(key=f"{section}.{key}"):
                raw = old_config()
                raw[section][key] = value
                with self.assertRaisesRegex(ValueError, f"{section}.{key}") as caught:
                    convert(raw)
                self.assertNotIn(str(value), str(caught.exception))

        raw = old_config()
        other = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32"
        raw["sweeps"] = {CELL: {"client_poll_iterations": "200", "temperature": 0.0}, other: {"client_poll_wait_s": 1}}
        converted, message = convert(raw)
        self.assertEqual(converted["timeouts"]["client_s"], 120 + 200 * 60)
        self.assertEqual(converted["sweeps"], {CELL: {"temperature": 0.0}, other: {}})
        self.assertIn(f"sweeps.{CELL}.client_poll_iterations -> timeouts.client_s", message)
        raw["sweeps"][CELL] = {"client_poll_iterations": "2"}
        self.assertEqual(convert(raw)[0]["timeouts"]["client_s"], 5520)

        raw = old_config()
        raw["benchmark_params"].update({"ignore_eos": "false", "trust_remote_code": "true"})
        self.assertEqual(
            convert(raw)[0]["benchmark_params"]["client_args"], {"ignore_eos": False, "trust_remote_code": True}
        )
        raw["benchmark_params"].update({"ignore_eos": "true", "trust_remote_code": "false"})
        self.assertNotIn("client_args", convert(raw)[0]["benchmark_params"])

        raw = old_config()
        raw["server_params"]["master_port"] = 29501
        with self.assertRaisesRegex(ValueError, r"^server_params cannot override harness fields: \['master_port'\]$"):
            convert(raw)

        raw = old_config()
        raw["server_params"]["tp"] = 4
        with self.assertRaisesRegex(ValueError, "server_params.tensor_parallel_size and server_params.tp"):
            convert(raw)

    def test_reference_config_stem_dropped_with_warning(self):
        raw = new_config(accuracy={"quant_parity": {"probe_prompt": "p", "reference_config_stem": "llama_fp16"}})
        converted, message = convert(raw)
        self.assertEqual(converted["accuracy"], {"quant_parity": {"probe_prompt": "p"}})
        self.assertIn("accuracy.quant_parity.reference_config_stem (dropped)", message)
        raw = new_config(accuracy={"quant_parity": {"reference_config_stem": "llama_fp16"}})
        self.assertNotIn("accuracy", convert(raw)[0])

    def test_old_threshold_rows(self):
        converted, message = convert({**old_config(), "thresholds": old_thresholds()})
        self.assertEqual(
            converted["thresholds"]["accuracy"],
            {"lm_eval": {"gsm8k_flex": {"gsm8k.exact_match__flexible-extract": {"kind": "min", "value": 0.5}}}},
        )
        self.assertIn("thresholds.accuracy.gsm8k_flex -> thresholds.accuracy.lm_eval.gsm8k_flex", message)
        thresholds = old_thresholds()
        thresholds["accuracy"]["gsm8k_flex"]["gsm8k.exact_match__flexible-extract"]["kind"] = "min_tok_s"
        converted, _ = convert({**old_config(), "thresholds": thresholds})
        spec = converted["thresholds"]["accuracy"]["lm_eval"]["gsm8k_flex"]["gsm8k.exact_match__flexible-extract"]
        self.assertEqual(spec, {"kind": "min", "value": 0.5})
        current = {CELL: {}, "accuracy": {"lm_eval": {"gsm8k_flex": {}}}}
        self.assertEqual(
            convert({**new_config(), "thresholds": current}), ({**new_config(), "thresholds": current}, None)
        )

    def test_old_metric_names(self):
        renames = {
            "queue_time_p50_ms": "p50_queue_time_ms",
            "queue_time_p95_ms": "p95_queue_time_ms",
            "prefill_time_p50_ms": "p50_prefill_time_ms",
            "prefill_time_p95_ms": "p95_prefill_time_ms",
            "decode_throughput_p50": "p50_decode_throughput",
        }
        for old, new in renames.items():
            with self.subTest(old=old):
                spec = {"kind": "min" if "throughput" in old else "max", "value": 3}
                converted, message = convert({**new_config(), "thresholds": {CELL: {old: spec}}})
                self.assertEqual(converted["thresholds"][CELL], {new: spec})
                self.assertIn(f"{old} -> thresholds.{CELL}.{new}", message)
                self.assertEqual(validate(converted).thresholds.get(CELL), {new: spec})
        both = {
            CELL: {"queue_time_p50_ms": {"kind": "max", "value": 3}, "p50_queue_time_ms": {"kind": "max", "value": 3}}
        }
        with self.assertRaisesRegex(ValueError, "queue_time_p50_ms and p50_queue_time_ms"):
            convert({**new_config(), "thresholds": both})

    def test_timeout_conversion_same_timing(self):
        converted, _ = convert(old_config())
        config = validate(converted)
        self.assertEqual((config.timeouts.server_ready_s, config.timeouts.client_s), (3930, 5520))
        job = VllmJob(FakeOrch(), config, 1024, 1024, 16, 320)
        self.assertEqual((job._warmup_wait, job._server_poll_count, job._server_poll_wait), (330, 60, 60))
        self.assertEqual((job._client_initial_wait, job._client_poll_count, job._client_poll_wait), (120, 90, 60))

        default = validate(new_config())
        self.assertEqual((default.timeouts.server_ready_s, default.timeouts.client_s), (None, None))
        job = VllmJob(FakeOrch(), default, 1024, 1024, 16, 320)
        self.assertEqual((job._warmup_wait, job._server_poll_count, job._server_poll_wait), (330, 60, 60))
        self.assertEqual((job._client_initial_wait, job._client_poll_count, job._client_poll_wait), (120, 20, 60))

        raw = old_config()
        raw["server_params"].update(
            {
                "pipeline_parallel_size": 2,
                "server_warmup_wait_s": 30,
                "server_poll_iterations": 540,
                "server_poll_wait_s": 10,
            }
        )
        raw["benchmark_params"].update(
            {"client_initial_wait_s": 10, "client_poll_iterations": 720, "client_poll_wait_s": 10}
        )
        raw["sweeps"] = {"ISL=1024,OSL=1024,TP=8,PP=2,CONC=16": {}}
        raw["runs"] = list(raw["sweeps"])
        config = validate(convert(raw)[0])
        config.bind_effective_topology(EffectiveVllmTopology("distributed", ("n0", "n1"), 2), suite="distributed")
        job = VllmJob(FakeOrch(hosts=["n0", "n1"]), config, 1024, 1024, 16, 320)
        self.assertEqual((job._warmup_wait, job._server_poll_count, job._server_poll_wait), (30, 540, 10))
        self.assertEqual((job._client_initial_wait, job._client_poll_count, job._client_poll_wait), (10, 720, 10))


def _write(tmp, config, thresholds):
    root = Path(tmp)
    path = root / "llama.json"
    path.write_text(json.dumps(config))
    if thresholds is not None:
        (root / config["threshold_json"]).write_text(json.dumps(thresholds))
    return path


def _capture_logs(action):
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = action()
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    return result, [record.getMessage() for record in records], [str(w.message) for w in caught]


class TestLoadVariant(unittest.TestCase):
    def test_load_variant_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = old_config()
            old["thresholds"] = "ignored, replaced by the threshold file"
            path = _write(tmp, old, old_thresholds())
            loaded, logs, caught = _capture_logs(lambda: load_variant(path, {"username": "jdoe"}))
        self.assertEqual(loaded.paths.shared_fs, "/home/jdoe")
        self.assertEqual(loaded.paths.log_dir, "/home/jdoe/LOGS")
        self.assertEqual(len(caught), 1)
        self.assertIn("server_params.tensor_parallel_size -> server_params.tp", caught[0])
        self.assertEqual([line for line in logs if line == caught[0]], [caught[0]])

        current = copy.deepcopy(loaded.model_dump(exclude={"thresholds"}))
        del current["server_params"]["nnodes"]
        current["paths"]["shared_fs"] = "/home/{user-id}"
        current["paths"]["log_dir"] = "{shared_fs}/LOGS"
        thresholds = {CELL: old_thresholds()[CELL], "accuracy": {"lm_eval": old_thresholds()["accuracy"]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, current, thresholds)
            reloaded, _logs, caught = _capture_logs(lambda: load_variant(path, {"username": "jdoe"}))
        self.assertEqual(caught, [])
        self.assertEqual(reloaded, loaded)

    def test_load_variant_quiet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, old_config(), old_thresholds())
            loaded, logs, caught = _capture_logs(lambda: load_variant(path, {}, warn=False))
        self.assertEqual(loaded.server_params.tp, 8)
        self.assertEqual(caught, [])
        self.assertFalse([line for line in logs if "Deprecated" in line])

    def test_old_kinds_in_a_cell_fail_at_load_as_on_main(self):
        for metric, kind in (("p99_ttft_ms", "max_ms"), ("output_throughput", "min_tok_s")):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    path = _write(tmp, new_config(), {CELL: {metric: {"kind": kind, "value": 1}}})
                    with self.assertRaises(ConfigError) as caught:
                        load_variant(path, {}, warn=False)
                self.assertIn(f"{metric}.kind", str(caught.exception))

    def test_quoted_accuracy_value_loads_as_a_number_with_the_warning(self):
        metric = "gsm8k.exact_match__flexible-extract"
        thresholds = old_thresholds()
        thresholds["accuracy"]["gsm8k_flex"][metric]["value"] = "0.75"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, new_config(), thresholds)
            loaded, _logs, caught = _capture_logs(lambda: load_variant(path, {}))
        value = loaded.thresholds.accuracy.lm_eval["gsm8k_flex"][metric].value
        self.assertEqual((value, type(value)), (0.75, float))
        self.assertEqual(len(caught), 1)
        self.assertIn(f"thresholds.accuracy.gsm8k_flex.{metric}.value", caught[0])

        bad = {
            "non-finite": {"kind": "min", "value": "nan"},
            "not a number": {"kind": "min", "value": "high"},
            "within": {"kind": "within", "value": "0.75", "tolerance_pct": 5},
            "min_ratio": {"kind": "min_ratio", "value": 0.75, "reference": "gsm8k.exact_match__strict-match"},
        }
        for name, spec in bad.items():
            with self.subTest(name=name):
                thresholds["accuracy"]["gsm8k_flex"][metric] = spec
                with tempfile.TemporaryDirectory() as tmp:
                    path = _write(tmp, new_config(), thresholds)
                    with self.assertRaises(ConfigError) as caught:
                        load_variant(path, {}, warn=False)
                self.assertIn(f"gsm8k_flex.{metric}", str(caught.exception))

    def test_empty_threshold_json_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, new_config(threshold_json=""), None)
            loaded, logs, _caught = _capture_logs(lambda: load_variant(path, {}, warn=False))
        self.assertFalse(loaded.enforce_thresholds)
        self.assertEqual(loaded.thresholds.cells, {})
        self.assertEqual(len([line for line in logs if "names no threshold_json" in line]), 1)
        self.assertIn(str(path), logs[0])

    def test_load_variant_errors_without_values(self):
        raw = new_config()
        raw["server_params"]["tp"] = FAKE_CRED
        raw["container"]["env"] = {"HF_TOKEN": FAKE_CRED}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, raw, {CELL: {}})
            with self.assertRaises(ConfigError) as caught:
                load_variant(path, {}, warn=False)
        message = str(caught.exception)
        self.assertIn("server_params.tp", message)
        self.assertIn("HF_TOKEN", message)
        self.assertNotIn(FAKE_CRED, message)

    def test_merged_container(self):
        cluster = {"container": {"runtime": {"args": {"registry": {"username": "jdoe"}, "volumes": ["/a:/a"]}}}}
        config = validate(new_config())
        merged = merged_container(cluster, config)
        self.assertEqual(merged["runtime"]["args"]["registry"], {"username": "jdoe"})
        self.assertEqual(merged["runtime"]["args"]["volumes"], [])
        self.assertEqual(merged["image"], "rocm/vllm:latest")
        merged["runtime"]["args"]["registry"]["username"] = "changed"
        self.assertEqual(cluster["container"]["runtime"]["args"]["registry"], {"username": "jdoe"})


if __name__ == "__main__":
    unittest.main()
