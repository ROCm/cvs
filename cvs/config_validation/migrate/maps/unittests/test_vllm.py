'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.migrate.maps.vllm, run through
``engine.migrate`` and ``cvs config migrate`` with the real maps package, on the
old fixture in ``migrate/unittests/old/``, synthetic old configs and every
shipped vLLM config.
'''

import copy
import io
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


import cvs.main as main
from cvs.cli_plugins.config_plugin import ConfigPlugin
from cvs.config_validation.loaders.suite_config import UNMAPPED_FIELD_MESSAGE
from cvs.config_validation.migrate.engine import migrate, migrate_many, old_paths
from cvs.config_validation.migrate.maps import vllm
from cvs.config_validation.schemas.inference.engines import (
    CLIENT_RESERVED,
    SERVER_RESERVED,
    VllmBenchmarkParams,
    VllmConfig,
    VllmServerParams,
)
from cvs.lib.conf_lib.config_blocks import has_placeholder
from cvs.unittests.support import FAKE_CRED

OLD_DIR = Path(__file__).resolve().parents[2] / "unittests" / "old"
FIXTURE = OLD_DIR / "mi3xx_vllm_llama33-70b_fp8_distributed.json"
SHIPPED_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
GOLDEN = Path(__file__).resolve().parents[4] / "tests" / "inference" / "vllm" / "unittests" / "golden_commands.json"

KEY = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
OTHER_KEY = "ISL=8192,OSL=1024,TP=8,PP=2,CONC=64"
PATHS = {
    "shared_fs": "/home/{user-id}",
    "models_dir": "/models",
    "log_dir": "{shared_fs}/LOGS",
    "hf_token_file": "{shared_fs}/.hf_token",  # nosec B105 - a file path, not a secret
}
CONTAINER = {
    "name": "vllm",
    "image": "rocm/vllm:latest",
    "runtime": {"name": "docker", "args": {"devices": ["/dev/kfd"]}},
}

# Serve flags as today's configs write them: every JSON type, a quoted number, an empty object.
SERVE_FLAGS = {
    "kv_cache_dtype": "fp8",
    "max_model_len": 2056,
    "gpu_memory_utilization": "0.5",
    "trust_remote_code": True,
    "limit_mm_per_prompt": {},
    "max_num_seqs": None,
    "my_own_flag": ["a", 1],
}


def _old(**overrides):
    old = {
        "enforce_thresholds": False,
        "threshold_json": "",
        "paths": PATHS,
        "container": CONTAINER,
        "server_params": {
            "backend": "vllm",
            "model": "org/model",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 2,
            "dist_init_port": 29600,
        },
        "benchmark_params": {"backend": "vllm", "num_prompts": 64},
        "sweeps": {KEY: {}},
        "runs": [KEY],
    }
    old.update(copy.deepcopy(overrides))
    return old


NEW = {
    "enforce_thresholds": False,
    "threshold_json": "",
    "paths": PATHS,
    "container": CONTAINER,
    "server_params": {"model": "org/model", "tp": 8, "pp": 2, "nnodes": 2, "master_port": 29600},
    "benchmark_params": {"num_prompts": 64},
    "sweeps": {KEY: {}},
    "runs": [KEY],
    "vllm": {},
}


def _placeholders(node):
    """Every string in ``node`` that holds a ``<changeme>`` form, counted."""
    if isinstance(node, dict):
        return sum((_placeholders(value) for value in node.values()), Counter())
    if isinstance(node, list):
        return sum((_placeholders(value) for value in node), Counter())
    return Counter([node]) if has_placeholder(node) else Counter()


def _strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def _vllm_flags(command):
    """Each ``vllm serve`` or ``vllm bench serve`` in ``command``, with the snake_case name of each flag it passes."""
    for match in re.finditer(r"vllm (serve|bench serve) (.*?)(?= [<>]|$)", command):
        yield match[1], {flag.replace("-", "_") for flag in re.findall(r"(?:^| )--([a-z][a-z0-9-]*)", match[2])}


def _validate(config, threshold_file):
    """Validate a converted config as ``VllmConfig`` with its converted threshold file attached, placeholders allowed."""
    return VllmConfig.model_validate({**config, "thresholds": json.loads(threshold_file.read_text())})


def _with_threshold_file(config_path):
    """``config_path`` and the threshold file it names beside it, as ``cvs config migrate`` takes them together."""
    name = json.loads(config_path.read_text()).get("threshold_json")
    return [config_path, config_path.parent / name] if name else [config_path]


class _MapCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def _migrate(self, data, name="old.json"):
        """Migrate ``data``, written as ``name``, or the file ``data`` names, as the vllm suite.

        Returns the exit code, the lines after ``wrote ...`` (or the one line
        saying why nothing was written), and the output, or None.
        """
        if isinstance(data, Path):
            source = data
        else:
            source = self.tmp / name
            source.write_text(json.dumps(data, indent=2))
        output = self.tmp / f"new_{source.name}"
        output.unlink(missing_ok=True)
        lines = []
        code = migrate("vllm", source, output, out=lines.append)
        self.assertNotIn(FAKE_CRED, "\n".join(lines))
        if not output.exists():
            return code, lines, None
        self.assertTrue(lines[0].startswith(f"wrote {output}"))
        return code, [line.strip() for line in lines[1:]], json.loads(output.read_text())

    def _migrate_with_thresholds(self, config_path):
        """Migrate a config with its threshold file into one folder; return the code, the lines and both outputs."""
        folder = self.tmp / "converted"
        folder.mkdir(exist_ok=True)
        lines = []
        code = migrate_many("vllm", _with_threshold_file(config_path), folder, force=True, out=lines.append)
        new = json.loads((folder / config_path.name).read_text())
        return code, lines, new, folder / new["threshold_json"]


class TestMapModule(unittest.TestCase):
    def test_the_old_keys_are_the_renamed_and_dropped_ones_and_a_converted_config_has_none(self):
        old = json.loads(FIXTURE.read_text())
        self.assertEqual(
            old_paths(old, vllm.CONFIG),
            [
                "server_params.backend",
                "server_params.tensor_parallel_size",
                "server_params.pipeline_parallel_size",
                "server_params.dist_init_port",
                "server_params.distributed_executor_backend",
                "server_params.max_model_len",
                "benchmark_params.backend",
                "benchmark_params.dataset_name",
                "benchmark_params.trust_remote_code",
                "benchmark_params.ignore_eos",
                "accuracy.tasks",
            ],
        )
        self.assertEqual(
            old_paths({"ib_hca_devices": "auto", "ib_netdev": None}, vllm.CONFIG), ["ib_hca_devices", "ib_netdev"]
        )
        self.assertEqual(old_paths(NEW, vllm.CONFIG), [])


class TestOldFixture(_MapCase):
    def test_the_fixture_converts_clean_and_validates_with_its_threshold_file(self):
        old = json.loads(FIXTURE.read_text())
        code, lines, new, threshold_file = self._migrate_with_thresholds(FIXTURE)
        self.assertEqual(code, 0, lines)
        self.assertNotIn("_unmapped", new)
        self.assertEqual(_placeholders(new), _placeholders(old))
        server, bench = old["server_params"], old["benchmark_params"]
        self.assertEqual(
            new["server_params"],
            {
                "_comment": server["_comment"],
                "model": "<changeme>",
                "tp": 8,
                "pp": 2,
                "nnodes": 2,
                "port": "8888",
                "master_port": "29501",
                "server_poll_iterations": 360,
                "server_poll_wait_s": 10,
                "server_warmup_wait_s": 30,
                "engine_args": {"kv_cache_dtype": "fp8", "max_num_seqs": 16, "gpu_memory_utilization": 0.5},
                "max_model_length": 2056,
                "_comment_context": server["_comment_context"],
                "_comment_startup": server["_comment_startup"],
            },
        )
        self.assertEqual(new["vllm"], {"distributed_executor_backend": "mp"})
        dropped = ("backend", "dataset_name", "trust_remote_code", "ignore_eos", "temperature")
        self.assertEqual(
            new["benchmark_params"],
            {
                **{key: value for key, value in bench.items() if key not in dropped},
                "client_args": {"temperature": 0.0},
            },
        )
        self.assertEqual((new["sweeps"], new["runs"]), (old["sweeps"], old["runs"]))
        self.assertEqual(new["accuracy"], {"lm_eval": old["accuracy"]})
        moved = ("server_params", "benchmark_params", "accuracy", "vllm")
        self.assertEqual(
            {key: value for key, value in new.items() if key not in moved},
            {key: value for key, value in old.items() if key not in moved},
        )

        config = _validate(new, threshold_file)
        self.assertEqual((config.server_params.tp, config.server_params.port), (8, 8888))
        self.assertEqual(list(config.thresholds.model_extra), [KEY])
        self.assertEqual(list(config.thresholds.accuracy.lm_eval), [task["id"] for task in old["accuracy"]["tasks"]])

    def test_a_converted_config_is_already_in_the_new_format(self):
        new = self._migrate_with_thresholds(FIXTURE)[2]
        source = self.tmp / "again.json"
        self.assertEqual(
            self._migrate(new, name=source.name),
            (0, [f"{source} is already in the new format; nothing to convert"], None),
        )


class TestRenames(_MapCase):
    def test_each_renamed_key_takes_its_new_name_and_both_backends_are_left_out(self):
        self.assertEqual(self._migrate(_old()), (0, [], NEW))

    def test_a_backend_other_than_vllm_goes_to_unmapped(self):
        old = _old()
        old["server_params"]["backend"] = "sglang"
        code, lines, new = self._migrate(old)
        self.assertEqual((code, lines), (1, [f"_unmapped.server_params.backend: {UNMAPPED_FIELD_MESSAGE}"]))
        self.assertEqual(new, {"_unmapped": {"server_params.backend": "sglang"}, **NEW})

    def test_a_hugging_face_token_in_container_env_stops_naming_the_variable_and_writes_nothing(self):
        old = _old(container={**CONTAINER, "env": {"NCCL_DEBUG": "WARN", "HF_TOKEN": FAKE_CRED}})
        self.assertEqual(
            self._migrate(old),
            (
                2,
                [
                    "container.env must not set ['HF_TOKEN']: container.env values become docker run -e arguments. "
                    "Put the Hugging Face token in the file named by paths.hf_token_file instead."
                ],
                None,
            ),
        )


class TestFlags(_MapCase):
    def test_every_other_server_key_moves_into_engine_args_unchanged_and_fields_and_comments_stay(self):
        fields = {
            "port": 8010,
            "master_port": 29700,
            "server_poll_iterations": 360,
            "server_poll_wait_s": 10,
            "server_warmup_wait_s": 30,
        }
        server = {
            "_comment": "kept",
            "model": "org/model",
            "tensor_parallel_size": 8,
            "pp": 2,
            "distributed_executor_backend": "ray",
            **fields,
            **SERVE_FLAGS,
        }
        code, lines, new = self._migrate(_old(server_params=server))
        self.assertEqual((code, lines), (0, []))
        flags = {name: value for name, value in SERVE_FLAGS.items() if name != "max_model_len"}
        expected = {"_comment": "kept", "model": "org/model", "tp": 8, "pp": 2, **fields}
        self.assertEqual(new["server_params"], {**expected, "engine_args": flags, "max_model_length": 2056})
        self.assertEqual(json.dumps(new["server_params"]["engine_args"]), json.dumps(flags))
        self.assertEqual(new["vllm"], {"distributed_executor_backend": "ray"})

    def test_a_pipeline_above_one_sets_nnodes_and_a_false_trust_remote_code_passes_no_flag(self):
        server = {**_old()["server_params"], "pipeline_parallel_size": 1, "trust_remote_code": False}
        new = self._migrate(_old(server_params=server))[2]
        self.assertEqual(new["server_params"], {"model": "org/model", "tp": 8, "pp": 1, "master_port": 29600})

    def test_every_other_benchmark_key_moves_into_client_args_and_fields_stay(self):
        bench = {
            "backend": "vllm",
            "dataset_name": "random",
            "num_prompts": 64,
            "seed": 7,
            "ignore_eos": True,
            "trust_remote_code": True,
            "temperature": 0.0,
            "goodput": ["ttft:500"],
        }
        server = {**_old()["server_params"], "trust_remote_code": True}
        code, lines, new = self._migrate(_old(server_params=server, benchmark_params=bench))
        self.assertEqual((code, lines), (0, []))
        self.assertEqual(
            new["benchmark_params"],
            {"num_prompts": 64, "seed": 7, "client_args": {"temperature": 0.0, "goodput": ["ttft:500"]}},
        )
        self.assertEqual(new["server_params"]["engine_args"], {"trust_remote_code": True})

    def test_a_client_field_that_differs_from_what_the_harness_now_passes_goes_to_unmapped(self):
        for field, value in (("dataset_name", "sharegpt"), ("ignore_eos", False), ("trust_remote_code", True)):
            with self.subTest(field=field):
                bench = {"num_prompts": 64, field: value}
                code, lines, new = self._migrate(_old(benchmark_params=bench))
                self.assertEqual((code, lines), (1, [f"_unmapped.benchmark_params.{field}: {UNMAPPED_FIELD_MESSAGE}"]))
                self.assertEqual(new["_unmapped"], {f"benchmark_params.{field}": value})


class TestNicFields(_MapCase):
    def test_a_value_that_acts_as_the_default_today_is_left_out(self):
        for key, value in (("ib_hca_devices", "auto"), ("ib_netdev", "")):
            with self.subTest(key=key, value=value):
                self.assertEqual(self._migrate(_old(**{key: value})), (0, [], NEW))

    def test_other_values_go_to_unmapped(self):
        for key, value in (("ib_netdev", FAKE_CRED), ("ib_netdev", "auto")):
            with self.subTest(key=key, value=value):
                code, lines, new = self._migrate(_old(**{key: value}))
                self.assertEqual((code, lines), (1, [f"_unmapped.{key}: {UNMAPPED_FIELD_MESSAGE}"]))
                self.assertEqual(new, {"_unmapped": {key: value}, **NEW})


class TestSweeps(_MapCase):
    def test_sweeps_and_runs_keep_their_shape_and_comments(self):
        sweeps = {"_comment": "cells", KEY: {}, OTHER_KEY: {"_comment": "not run"}}
        self.assertEqual(self._migrate(_old(sweeps=sweeps)), (0, [], {**NEW, "sweeps": sweeps}))

    def test_a_per_cell_client_flag_is_kept_and_a_harness_flag_is_named(self):
        sweeps = {KEY: {}, OTHER_KEY: {"num_prompts": 32, "temperature": 0.5}}
        code, lines, new = self._migrate(_old(sweeps=sweeps, runs=[KEY, OTHER_KEY]))
        self.assertEqual((code, lines), (0, []))
        self.assertEqual((new["sweeps"], new["runs"]), (sweeps, [KEY, OTHER_KEY]))
        sweeps[OTHER_KEY] = {"ignore_eos": True}
        code, lines, new = self._migrate(_old(sweeps=sweeps, runs=[KEY, OTHER_KEY]))
        self.assertEqual(
            (code, lines), (1, [f"(config): sweeps.{OTHER_KEY} cannot set harness fields: ['ignore_eos']"])
        )


class TestThresholdFile(_MapCase):
    def test_old_kinds_are_renamed_and_the_file_validates_with_vllms_rules(self):
        old = {
            "_comment": "x",
            KEY: {
                "mean_ttft_ms": {"kind": "max_ms", "value": 80},
                "output_throughput": {"kind": "min_tok_s", "value": 900},
            },
            "accuracy": {"gsm8k_flex": {"exact_match,flexible-extract": {"kind": "min", "value": 0.5}}},
        }
        code, lines, new = self._migrate(old, name="old_threshold.json")
        self.assertEqual((code, lines), (0, []))
        self.assertEqual(
            new[KEY], {"mean_ttft_ms": {"kind": "max", "value": 80}, "output_throughput": {"kind": "min", "value": 900}}
        )
        self.assertEqual(new["accuracy"], {"lm_eval": old["accuracy"]})
        self.assertEqual(list(_validate(NEW, self.tmp / "new_old_threshold.json").thresholds.model_extra), [KEY])

    def test_an_old_kind_against_the_metrics_direction_is_named_and_one_with_no_new_kind_goes_to_unmapped(self):
        old = {
            KEY: {
                "output_throughput": {"kind": "max_ms", "value": 900},
                "mean_ttft_ms": {"kind": "within", "value": 80, "tolerance_pct": 5},
            }
        }
        code, lines, new = self._migrate(old, name="old_threshold.json")
        self.assertEqual(code, 1)
        self.assertEqual(
            lines,
            [
                f"_unmapped.{KEY}.mean_ttft_ms: {UNMAPPED_FIELD_MESSAGE}",
                f"{KEY}: output_throughput threshold kind must be 'min' or 'info'",
            ],
        )
        self.assertEqual(new[KEY], {"output_throughput": {"kind": "max", "value": 900}})


class TestShippedConfigs(_MapCase):
    def test_every_shipped_config_converts_clean_and_validates_with_its_threshold_file(self):
        configs = sorted(path for path in SHIPPED_DIR.glob("*.json") if not path.name.endswith("threshold.json"))
        self.assertEqual(len(configs), 33)
        for path in configs:
            with self.subTest(config=path.name):
                old = json.loads(path.read_text())
                if "server_params" in old and "tensor_parallel_size" not in old["server_params"]:
                    self.skipTest("shipped configs are already in the new format")
                code, lines, new, threshold_file = self._migrate_with_thresholds(path)
                self.assertEqual(code, 0, lines)
                self.assertNotIn("_unmapped", new)
                self.assertEqual(_placeholders(new), _placeholders(old))
                config = _validate(new, threshold_file)
                self.assertEqual(config.runs, old["runs"])
                pp = old["server_params"].get("pipeline_parallel_size", 1)
                self.assertEqual(config.server_params.nnodes, int(pp) if int(pp) > 1 else None)

    def test_the_reserved_names_are_the_fields_their_old_names_and_each_flag_the_harness_sets(self):
        golden = json.loads(GOLDEN.read_text())
        harness = {"serve": set(), "bench serve": set()}
        for name, recorded in golden.items():
            old = json.loads((SHIPPED_DIR / name).read_text())
            if "tensor_parallel_size" not in old.get("server_params", {}):
                self.skipTest("shipped configs are already in the new format")
            new = self._migrate_with_thresholds(SHIPPED_DIR / name)[2]
            own = {
                "serve": set(new["server_params"].get("engine_args", {})),
                "bench serve": set(new.get("benchmark_params", {}).get("client_args", {})),
            }
            for command in _strings(recorded):
                for kind, flags in _vllm_flags(command):
                    harness[kind] |= flags - own[kind]
        for kind, block, model, reserved in (
            ("serve", "server_params", VllmServerParams, SERVER_RESERVED),
            ("bench serve", "benchmark_params", VllmBenchmarkParams, CLIENT_RESERVED),
        ):
            with self.subTest(command=f"vllm {kind}"):
                old_names = {path.removeprefix(f"{block}.") for path in vllm.CONFIG if path.startswith(f"{block}.")}
                # Every flag the harness passes and every old field name is reserved; nothing else but fields is.
                self.assertLessEqual(harness[kind] | old_names, reserved)
                fields = set(model.model_fields) - {"engine_args", "client_args"}
                self.assertLessEqual(reserved - harness[kind] - old_names, fields | {"engine", "headless", "node_rank"})


class TestCli(unittest.TestCase):
    def _cvs(self, *argv):
        """Run ``cvs <argv>`` through ``cvs.main`` with the real maps package; return the exit code, stdout and stderr."""
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", ["cvs", *argv]), redirect_stdout(out), redirect_stderr(err):
            try:
                main.main([ConfigPlugin()])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_cvs_config_migrate_vllm_lists_the_suite_and_converts_the_old_fixture_into_a_folder(self):
        code, out, _ = self._cvs("config", "migrate", "--help")
        self.assertEqual(code, 0)
        self.assertRegex(" ".join(out.split()), r"\(available: [^)]*\bvllm\b")
        before = FIXTURE.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            written = Path(tmp) / FIXTURE.name
            config, threshold = _with_threshold_file(FIXTURE)
            self.assertEqual(
                self._cvs("config", "migrate", "vllm", str(config), str(threshold), tmp),
                (0, f"wrote {Path(tmp) / threshold.name}\nwrote {written}\n", ""),
            )
            new = json.loads(written.read_text())
        self.assertEqual(FIXTURE.read_bytes(), before)
        self.assertEqual(new["vllm"], {"distributed_executor_backend": "mp"})
        self.assertEqual(new["runs"], json.loads(before)["runs"])
        self.assertNotIn("_unmapped", new)


if __name__ == "__main__":
    unittest.main()
