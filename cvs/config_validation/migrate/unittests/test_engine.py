'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs.config_validation.migrate.engine, on a synthetic "demo"
suite map and schema defined here, with single, distributed and
disaggregated-shaped old configs and an old threshold file.
'''

import copy
import errno
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Dict, List, Literal, Optional
from unittest import mock

from pydantic import BeforeValidator, Field, model_validator

from cvs.config_validation.loaders.suite_config import (
    MISSING_THRESHOLD_MESSAGE,
    UNMAPPED_FIELD_MESSAGE,
    load_suite_config,
)
from cvs.config_validation.migrate import engine, maps
from cvs.config_validation.migrate.engine import (
    MISSING_MESSAGE,
    TO_UNMAPPED,
    flags_from_list,
    migrate,
    old_paths,
    suite_maps,
    suite_names,
)
from cvs.config_validation.schemas.base import OpenBlock, StrictBlock, SuiteConfig, drop_comment_keys
from cvs.config_validation.schemas.inference.llm_serving import LlmServingConfig, ServingServerParams
from cvs.config_validation.schemas.sweeps import Axis
from cvs.config_validation.schemas.thresholds import Threshold
from cvs.lib.conf_lib.config_blocks import PLACEHOLDER_MESSAGE
from cvs.lib.conf_lib.loader import ConfigError
from cvs.unittests.support import FAKE_CRED

CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
CELL2 = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=32"
DCELL = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64"
CLUSTER = {"username": "jdoe", "priv_key_file": "/home/jdoe/.ssh/id_rsa", "node_dict": {"node0": {}}}


class _DemoServer(ServingServerParams):
    port: int = 8000
    master_port: int = 29500


class _Disaggregation(StrictBlock):
    prefill: str
    decode: str


class _Router(StrictBlock):
    policy: str = "round_robin"


class _DemoBlock(StrictBlock):
    add_flags: List[str] = Field(default_factory=list)
    disaggregation: Optional[_Disaggregation] = None
    router: Optional[_Router] = None


class _DemoConfig(LlmServingConfig):
    suite = "demo"
    axes = (Axis("ISL", "isl"), Axis("OSL", "osl"), Axis("TP", "tp"), Axis("PP", "pp"), Axis("CONC", "concurrency"))
    server_params: _DemoServer
    demo: _DemoBlock = Field(default_factory=_DemoBlock)

    @classmethod
    def old_keys(cls, raw):
        return old_paths(raw, DEMO_MAP.CONFIG)


def _auto_or_unmapped(value, config):
    return {} if value in ("auto", None) else TO_UNMAPPED


def _every_sweep_when_empty(value, config):
    return {"runs": value or [key for key in config.get("sweeps", {}) if not key.startswith("_")]}


def _fewshot_count(value, config):
    return {"accuracy.lm_eval.tasks.*.num_fewshot": int(value)}


DEMO_MAP = SimpleNamespace(
    SCHEMA=_DemoConfig,
    CONFIG={
        "gpu_name": None,
        "ib_netdev": None,
        "server_params.backend": None,
        "server_params.tensor_parallel_size": "server_params.tp",
        "server_params.pipeline_parallel_size": "server_params.pp",
        "server_params.dist_init_port": "server_params.master_port",
        "server_params.kv_cache_dtype": "server_params.engine_args.kv_cache_dtype",
        "server_params.add_flags": "demo.add_flags",
        "benchmark_params.backend": None,
        "benchmark_params.temperature": "benchmark_params.client_args.temperature",
        "sweep.runs.*.combo": "runs.*",
        "accuracy.tasks.*.*": "accuracy.lm_eval.tasks.*.*",
        "policies.*": "demo.disaggregation.*",
        "router": "demo.router",
    },
    THRESHOLDS={"max_ms": "max", "min_tok_s": "min"},
    FIELDS={
        "ib_hca_devices": _auto_or_unmapped,
        "ib_netdev": _auto_or_unmapped,
        "runs": _every_sweep_when_empty,
        "accuracy.tasks.*.fewshot": _fewshot_count,
    },
    FILL_THRESHOLD_FROM_SIBLING=True,
)


class _MaxSpec(Threshold):
    kind: Literal["max"]


class _MaxOnlyFile(OpenBlock):
    __pydantic_extra__: Dict[str, Annotated[Dict[str, _MaxSpec], BeforeValidator(drop_comment_keys)]]


class _MaxOnlyConfig(_DemoConfig):
    thresholds: _MaxOnlyFile = Field(default_factory=_MaxOnlyFile)


class _CoveredConfig(_DemoConfig):
    @model_validator(mode="after")
    def _enforced_cells_need_threshold_cells(self, info):
        if not (info.context or {}).get("config_path"):
            raise ValueError("validated without the config's path")
        if self.enforce_thresholds and any(cell not in self.thresholds.model_extra for cell in self.runs):
            raise ValueError("with enforce_thresholds, every cell needs a threshold cell")
        return self


class _OtherConfig(SuiteConfig):
    suite = "other"


OTHER_MAP = SimpleNamespace(
    SCHEMA=_OtherConfig,
    CONFIG={"server_params.tensor_parallelism": "server_params.tp", "server_params.serve_port": "server_params.port"},
    THRESHOLDS={},
)

PATHS = {
    "_comment": "Host paths; {user-id} is filled at load.",
    "shared_fs": "/home/{user-id}",
    "models_dir": "/models",
    "log_dir": "{shared_fs}/LOGS",
    "hf_token_file": "{shared_fs}/.hf_token",
}
ACCURACY = {"lm_eval": {"tasks": [{"id": "gsm8k_flex", "task": "gsm8k", "num_fewshot": 5}]}}
RUNTIME = {"name": "docker", "args": {"volumes": ["/home/{user-id}:/home/{user-id}"], "devices": []}}
SOCKET_ENV = {
    "NCCL_SOCKET_IFNAME": "eth0",
    "GLOO_SOCKET_IFNAME": "eth0",
    "TP_SOCKET_IFNAME": "eth0",
    "NCCL_IB_HCA": "mlx5_0",
}

OLD_SINGLE = {
    "_comment": "Demo single-node config.",
    "enforce_thresholds": False,
    "threshold_json": "demo_threshold.json",
    "gpu_name": "<changeme> mi355",
    "ib_hca_devices": "auto",
    "paths": PATHS,
    "container": {"name": "demo_{user-id}", "image": "registry.example/demo:1", "runtime": RUNTIME, "env": {}},
    "server_params": {
        "_comment_ports": "port serves the clients.",
        "backend": "demo",
        "model": "org/model",
        "tensor_parallel_size": 8,
        "port": 8010,
        "kv_cache_dtype": "fp8",
    },
    "benchmark_params": {"backend": "demo", "num_prompts": 320, "temperature": 0.0},
    "sweeps": {"_comment": "Two cells.", CELL: {}, CELL2: {}},
    "runs": [],
    "accuracy": {"tasks": [{"id": "gsm8k_flex", "task": "gsm8k", "fewshot": "5"}]},
}
NEW_SINGLE = {
    "_comment": "Demo single-node config.",
    "enforce_thresholds": False,
    "threshold_json": "demo_threshold.json",
    "paths": PATHS,
    "container": {"name": "demo_{user-id}", "image": "registry.example/demo:1", "runtime": RUNTIME},
    "server_params": {
        "_comment_ports": "port serves the clients.",
        "model": "org/model",
        "tp": 8,
        "port": 8010,
        "engine_args": {"kv_cache_dtype": "fp8"},
    },
    "benchmark_params": {"num_prompts": 320, "client_args": {"temperature": 0.0}},
    "sweeps": {"_comment": "Two cells.", CELL: {}, CELL2: {}},
    "runs": [CELL, CELL2],
    "accuracy": ACCURACY,
    "demo": {},
}

OLD_DISTRIBUTED = {
    "_comment": "Demo two-node config.",
    "threshold_json": "",
    "ib_netdev": None,
    "paths": PATHS,
    "container": {"name": "demo", "image": "registry.example/demo:1", "runtime": {"name": "docker"}, "env": SOCKET_ENV},
    "server_params": {
        "backend": "demo",
        "model": "org/model",
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 2,
        "dist_init_port": 29600,
    },
    "benchmark_params": {"num_prompts": 640},
    "sweeps": {DCELL: {}},
    "runs": [DCELL],
}
NEW_DISTRIBUTED = {
    "_comment": "Demo two-node config.",
    "threshold_json": "",
    "paths": PATHS,
    "container": {"name": "demo", "image": "registry.example/demo:1", "runtime": {"name": "docker"}, "env": SOCKET_ENV},
    "server_params": {"model": "org/model", "tp": 8, "pp": 2, "master_port": 29600},
    "benchmark_params": {"num_prompts": 640},
    "sweeps": {DCELL: {}},
    "runs": [DCELL],
    "demo": {},
}

OLD_DISAGGREGATED = {
    "_comment": "Demo prefill and decode config.",
    "threshold_json": "",
    "paths": PATHS,
    "server_params": {"model": "org/model", "tensor_parallel_size": 8, "add_flags": ["--attention-backend aiter"]},
    "benchmark_params": {"num_prompts": 64},
    "sweeps": {CELL: {}},
    "sweep": {"_comment": "Cells to run.", "runs": [{"combo": CELL}]},
    "policies": {"prefill": "cache_aware", "decode": "round_robin"},
    "router": {"policy": "cache_aware"},
}
NEW_DISAGGREGATED = {
    "_comment": "Demo prefill and decode config.",
    "threshold_json": "demo_disaggregated_threshold.json",
    "paths": PATHS,
    "server_params": {"model": "org/model", "tp": 8},
    "demo": {
        "add_flags": ["--attention-backend aiter"],
        "disaggregation": {"prefill": "cache_aware", "decode": "round_robin"},
        "router": {"policy": "cache_aware"},
    },
    "benchmark_params": {"num_prompts": 64},
    "sweeps": {CELL: {}},
    "runs": [CELL],
}

OLD_THRESHOLDS = {
    "_comment": "Demo thresholds.",
    CELL: {
        "output_throughput": {"kind": "min_tok_s", "value": 900},
        "mean_ttft_ms": {"kind": "max_ms", "value": 80},
        "p99_itl_ms": {"kind": "max", "value": 40},
        "_old": {"kind": "max_ms", "value": 1},
    },
    "accuracy": {"_comment": "By task id.", "gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.5}}},
}
NEW_THRESHOLDS = {
    "_comment": "Demo thresholds.",
    CELL: {
        "output_throughput": {"kind": "min", "value": 900},
        "mean_ttft_ms": {"kind": "max", "value": 80},
        "p99_itl_ms": {"kind": "max", "value": 40},
        "_old": {"kind": "max_ms", "value": 1},
    },
    "accuracy": {"_comment": "By task id.", "lm_eval": {"gsm8k_flex": {"exact_match": {"kind": "min", "value": 0.5}}}},
}


def _map_with(suite_map, **attrs):
    return SimpleNamespace(**{**vars(suite_map), **attrs})


class _MigrateCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        found = mock.patch.object(engine, "suite_maps", return_value={"demo": DEMO_MAP, "other": OTHER_MAP})
        found.start()
        self.addCleanup(found.stop)

    def _write(self, name, data):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data, indent=2))
        return path

    def _files(self):
        return {path: path.read_bytes() if path.is_file() else None for path in self.tmp.rglob("*")}

    def _run(self, data, output="new.json", suite="demo", force=False):
        """Migrate ``data``, written as ``old.json`` unless None; return the exit code and printed lines.

        The input never changes, and exit 2 leaves every file and folder as it was.
        """
        source = self.tmp / "old.json" if data is None else self._write("old.json", data)
        before = self._files()
        lines = []
        code = migrate(suite, source, self.tmp / output, force=force, out=lines.append)
        if source.is_file():
            self.assertEqual(source.read_bytes(), before[source])
        if code == 2:
            self.assertEqual(self._files(), before)
        return code, lines

    def _output(self, name="new.json"):
        return json.loads((self.tmp / name).read_text())

    def _load(self, name="new.json", schema=_DemoConfig):
        cluster = self._write("cluster.json", CLUSTER)
        return load_suite_config(schema, str(self.tmp / name), str(cluster), mock.Mock())

    def _refused(self, name="new.json", schema=_DemoConfig):
        with self.assertRaises(ConfigError) as ctx:
            self._load(name, schema)
        return ctx.exception.problems


class TestFlagsFromList(unittest.TestCase):
    def test_each_flag_takes_the_words_after_it_as_its_value(self):
        self.assertEqual(
            flags_from_list(
                [
                    "--attention-backend aiter",
                    "--disable-custom-all-reduce",
                    "--kv-cache-dtype",
                    "fp8_e4m3",
                    "--mem-fraction-static=0.9",
                    "--chat-template 'a b'",
                    "--lora-paths a b",
                ]
            ),
            {
                "attention_backend": "aiter",
                "disable_custom_all_reduce": True,
                "kv_cache_dtype": "fp8_e4m3",
                "mem_fraction_static": "0.9",
                "chat_template": "a b",
                "lora_paths": ["a", "b"],
            },
        )
        self.assertEqual(flags_from_list([]), {})

    def test_a_word_before_any_flag_and_a_repeated_flag_are_refused(self):
        with self.assertRaisesRegex(ValueError, "'aiter' comes before any --flag"):
            flags_from_list(["aiter --attention-backend"])
        with self.assertRaisesRegex(ValueError, "--tp is given more than once"):
            flags_from_list(["--tp 8", "--tp=4"])


class TestSuiteMaps(unittest.TestCase):
    def test_each_public_module_in_the_maps_package_is_a_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "zeta.py").write_text("CONFIG = {'a': 'b'}\n")
            (root / "alpha.py").write_text("CONFIG = {}\n")
            (root / "_helpers.py").write_text("raise AssertionError('a private module is not a map')\n")
            (root / "unittests").mkdir()
            (root / "unittests" / "__init__.py").write_text("raise AssertionError('a package is not a map')\n")
            self.addCleanup(self._forget, "alpha", "zeta")
            with mock.patch.object(maps, "__path__", [str(root)]):
                found = suite_maps()
                names = suite_names()
        self.assertEqual(names, ["alpha", "zeta"])
        self.assertEqual(sorted(found), ["alpha", "zeta"])
        self.assertEqual(found["zeta"].CONFIG, {"a": "b"})

    @staticmethod
    def _forget(*names):
        for name in names:
            sys.modules.pop(f"{maps.__name__}.{name}", None)


class TestOldPaths(unittest.TestCase):
    def test_literal_and_star_paths_match_in_file_order_skipping_comments_and_never_below_a_match(self):
        raw = {
            "server_params": {"_comment": "x", "tensor_parallel_size": 8, "model": "m"},
            "policies": {"_note": "kept", "prefill": "a", "decode": "b"},
            "sweep": {"runs": [{"combo": CELL}, {"weight": 1}, {"combo": CELL2}]},
            "router": {"policy": "x"},
        }
        renames = {
            "router": "demo.router",
            "router.policy": "demo.policy",
            "policies.*": "demo.disaggregation.*",
            "sweep.runs.*.combo": "runs.*",
            "server_params.tensor_parallel_size": "server_params.tp",
        }
        self.assertEqual(
            old_paths(raw, renames),
            [
                "server_params.tensor_parallel_size",
                "policies.prefill",
                "policies.decode",
                "sweep.runs.0.combo",
                "sweep.runs.2.combo",
                "router",
            ],
        )

    def test_a_new_format_config_matches_nothing_and_a_digit_is_not_a_list_item(self):
        self.assertEqual(old_paths(NEW_SINGLE, DEMO_MAP.CONFIG), [])
        self.assertEqual(old_paths({"runs": ["a"]}, {"runs.0": None}), [])


class TestOldPathsAtLoad(_MigrateCase):
    def test_a_schema_whose_old_keys_hook_uses_old_paths_refuses_an_old_config_naming_the_command(self):
        path = self._write("old.json", OLD_DISTRIBUTED)
        self.assertEqual(
            self._refused("old.json"),
            [
                "old config format: ib_netdev, server_params.backend, server_params.tensor_parallel_size, "
                "server_params.pipeline_parallel_size, server_params.dist_init_port; "
                f"run cvs config migrate demo {path} <new config>"
            ],
        )


class TestConvertsCleanly(_MigrateCase):
    def _converts(self, old, new):
        code, lines = self._run(old)
        self.assertEqual((code, lines), (0, [f"wrote {self.tmp / 'new.json'}"]))
        self.assertEqual(self._output(), new)
        return self._load().config

    def test_a_single_config_converts_and_loads_with_its_named_threshold_file(self):
        self._write("demo_threshold.json", NEW_THRESHOLDS)
        config = self._converts(OLD_SINGLE, NEW_SINGLE)
        self.assertEqual((config.demo.add_flags, config.server_params.tp), ([], 8))
        self.assertEqual(config.runs, [CELL, CELL2])
        self.assertEqual(config.accuracy.lm_eval.tasks[0].num_fewshot, 5)
        self.assertEqual(config.thresholds.model_extra[CELL]["mean_ttft_ms"].kind, "max")

    def test_a_renamed_key_takes_the_old_keys_place_and_the_engine_block_comes_last(self):
        self._write("demo_threshold.json", NEW_THRESHOLDS)
        self.assertEqual(self._run(OLD_SINGLE)[0], 0)
        new = self._output()
        self.assertEqual(list(new), list(NEW_SINGLE))
        self.assertEqual(list(new)[-1], "demo")
        self.assertEqual(list(new["server_params"]), ["_comment_ports", "model", "tp", "port", "engine_args"])

    def test_a_distributed_config_converts_and_loads_record_only_without_a_threshold_file_beside_it(self):
        config = self._converts(OLD_DISTRIBUTED, NEW_DISTRIBUTED)
        self.assertEqual((config.server_params.pp, config.server_params.master_port), (2, 29600))
        self.assertIs(config.enforce_thresholds, False)

    def test_a_disaggregated_config_converts_and_gets_the_one_threshold_file_beside_it(self):
        self._write("demo_disaggregated_threshold.json", NEW_THRESHOLDS)
        config = self._converts(OLD_DISAGGREGATED, NEW_DISAGGREGATED)
        self.assertEqual((config.demo.disaggregation.decode, config.demo.router.policy), ("round_robin", "cache_aware"))
        self.assertEqual(list(config.thresholds.model_extra), [CELL])

    def test_an_empty_threshold_json_stays_empty_with_no_threshold_file_or_several_beside_the_input(self):
        for siblings in ((), ("a_threshold.json", "b_threshold.json")):
            with self.subTest(siblings=siblings):
                for name in siblings:
                    self._write(name, NEW_THRESHOLDS)
                code, _ = self._run(OLD_DISAGGREGATED, force=True)
                self.assertEqual(code, 0)
                self.assertEqual(self._output()["threshold_json"], "")

    def test_a_map_that_leaves_the_sibling_fill_off_keeps_an_empty_threshold_json_recording_only(self):
        self._write("demo_disaggregated_threshold.json", NEW_THRESHOLDS)
        unset = SimpleNamespace(**{k: v for k, v in vars(DEMO_MAP).items() if k != "FILL_THRESHOLD_FROM_SIBLING"})
        with mock.patch.object(engine, "suite_maps", return_value={"demo": unset}):
            code, _ = self._run(OLD_DISAGGREGATED, force=True)
        self.assertEqual(code, 0)
        self.assertEqual(self._output(), {**NEW_DISAGGREGATED, "threshold_json": ""})
        config = self._load().config
        self.assertEqual((config.enforce_thresholds, config.thresholds.model_extra), (False, {}))

    def test_a_new_format_config_or_threshold_file_exits_0_and_writes_nothing(self):
        # NEW_THRESHOLDS keeps an old kind under a comment key, which is not converted.
        self._write("new.json", {"keep": FAKE_CRED})
        for data in (NEW_SINGLE, NEW_THRESHOLDS):
            source = self._write("old.json", data)
            before = self._files()
            # The input is checked first, so an existing output or the input itself as output is never reached.
            for output in ("new.json", "old.json"):
                with self.subTest(data=list(data), output=output):
                    self.assertEqual(
                        self._run(None, output=output),
                        (0, [f"{source} is already in the new format; nothing to convert"]),
                    )
                    self.assertEqual(self._files(), before)

    def test_a_schema_check_that_reads_the_threshold_file_sees_it_as_load_does(self):
        covered = _map_with(DEMO_MAP, SCHEMA=_CoveredConfig)
        record_only = {**OLD_SINGLE, "threshold_json": "", "enforce_thresholds": True}
        covering = {**OLD_SINGLE, "enforce_thresholds": True}
        with mock.patch.object(engine, "suite_maps", return_value={"demo": covered}):
            self.assertEqual(self._run(record_only)[0], 0)
            self.assertIs(self._load(schema=_CoveredConfig).config.enforce_thresholds, False)
            self._write("demo_threshold.json", {**NEW_THRESHOLDS, CELL2: NEW_THRESHOLDS[CELL]})
            self.assertEqual(self._run(covering, force=True), (0, [f"wrote {self.tmp / 'new.json'}"]))
        self.assertEqual(self._output(), {**NEW_SINGLE, "enforce_thresholds": True})
        config = self._load(schema=_CoveredConfig).config
        self.assertEqual((config.enforce_thresholds, list(config.thresholds.model_extra)), (True, [CELL, CELL2]))

    def test_an_empty_unmapped_block_in_the_input_is_left_out(self):
        self.assertEqual(self._run({**OLD_DISTRIBUTED, "_unmapped": {}})[0], 0)
        self.assertEqual(self._output(), NEW_DISTRIBUTED)

    def test_a_star_over_list_items_keeps_each_item_at_its_index_in_the_list_it_moves_to(self):
        self._write("demo_threshold.json", NEW_THRESHOLDS)
        old = copy.deepcopy(OLD_SINGLE)
        old["accuracy"]["tasks"] = [{"id": "a", "task": "gsm8k"}, {"id": "b", "task": "mmlu", "fewshot": "3"}]
        self.assertEqual(self._run(old)[0], 0)
        self.assertEqual(
            self._output()["accuracy"]["lm_eval"]["tasks"],
            [{"id": "a", "task": "gsm8k"}, {"id": "b", "task": "mmlu", "num_fewshot": 3}],
        )

    def test_a_threshold_file_converts_its_old_kinds_and_loads(self):
        code, lines = self._run(OLD_THRESHOLDS, output="new_threshold.json")
        self.assertEqual((code, lines), (0, [f"wrote {self.tmp / 'new_threshold.json'}"]))
        self.assertEqual(self._output("new_threshold.json"), NEW_THRESHOLDS)
        self._write("new.json", {**NEW_SINGLE, "threshold_json": "new_threshold.json"})
        cell = self._load().config.thresholds.model_extra[CELL]
        self.assertEqual(
            {metric: spec.kind for metric, spec in cell.items()},
            {"output_throughput": "min", "mean_ttft_ms": "max", "p99_itl_ms": "max"},
        )

    def test_comments_and_placeholders_are_kept_as_written_even_in_a_typed_field(self):
        old = copy.deepcopy(OLD_SINGLE)
        old["threshold_json"] = "<changeme>"
        old["container"]["image"] = "<changeme>"
        old["server_params"]["tensor_parallel_size"] = "<changeme-tp>"
        old["paths"]["_comment"] = "Replace <changeme> and {user-id}."
        code, _ = self._run(old)
        new = self._output()
        self.assertEqual(code, 0)
        self.assertEqual(
            (new["threshold_json"], new["container"]["image"], new["server_params"]["tp"]),
            ("<changeme>", "<changeme>", "<changeme-tp>"),
        )
        self.assertEqual(new["paths"], old["paths"])

    def test_an_empty_object_or_list_is_left_out_only_when_its_new_default_is_the_same(self):
        old = copy.deepcopy(OLD_DISTRIBUTED)
        old["container"]["env"] = {}
        old["container"]["runtime"]["args"] = {"devices": []}
        old["benchmark_params"]["client_args"] = {}
        old["accuracy"] = {}
        code, _ = self._run(old)
        new = self._output()
        self.assertEqual(code, 0)
        self.assertNotIn("env", new["container"])
        self.assertEqual(new["benchmark_params"], {"num_prompts": 640})
        # The accuracy block's default holds its four sections, so an empty one is kept as written.
        self.assertEqual(new["accuracy"], {})
        # runtime args default to unset, not to an empty list.
        self.assertEqual(new["container"]["runtime"]["args"], {"devices": []})

    def test_an_old_and_a_new_key_with_the_same_value_merge(self):
        old = copy.deepcopy(OLD_DISTRIBUTED)
        old["server_params"]["tp"] = 8
        code, _ = self._run(old)
        self.assertEqual(code, 0)
        self.assertEqual(self._output(), NEW_DISTRIBUTED)

    def test_the_first_written_path_that_matches_applies(self):
        first = _map_with(DEMO_MAP, CONFIG={"policies.decode": None, **DEMO_MAP.CONFIG})
        with mock.patch.object(engine, "suite_maps", return_value={"demo": first}):
            code, _ = self._run(OLD_DISAGGREGATED)
        self.assertEqual(code, 1)
        self.assertEqual(self._output()["demo"]["disaggregation"], {"prefill": "cache_aware", "decode": "<changeme>"})


class TestFieldsToFix(_MigrateCase):
    def _fix(self, old, output="new.json"):
        code, lines = self._run(old, output=output)
        self.assertEqual(code, 1)
        self.assertEqual(lines[0], f"wrote {self.tmp / output}; fix these fields before using it:")
        self.assertNotIn(FAKE_CRED, "\n".join(lines))
        return [line.strip() for line in lines[1:]]

    def test_unmapped_top_level_and_nested_fields_go_first_under_their_old_paths(self):
        self._write("demo_threshold.json", NEW_THRESHOLDS)
        old = copy.deepcopy(OLD_SINGLE)
        old["server_params"]["max_model_len"] = FAKE_CRED
        old["notes"] = {"owner": FAKE_CRED}
        lines = self._fix(old)
        expected = [
            f"_unmapped.server_params.max_model_len: {UNMAPPED_FIELD_MESSAGE}",
            f"_unmapped.notes: {UNMAPPED_FIELD_MESSAGE}",
        ]
        self.assertCountEqual(lines, expected)
        new = self._output()
        self.assertEqual(list(new)[0], "_unmapped")
        self.assertEqual(new["_unmapped"], {"server_params.max_model_len": FAKE_CRED, "notes": {"owner": FAKE_CRED}})
        self.assertEqual({key: value for key, value in new.items() if key != "_unmapped"}, NEW_SINGLE)
        self.assertCountEqual(self._refused(), expected)

    def test_a_refused_key_inside_a_moved_block_goes_under_its_old_path(self):
        old = copy.deepcopy(OLD_DISAGGREGATED)
        old["policies"]["bogus"] = FAKE_CRED
        old["router"]["bogus"] = FAKE_CRED
        lines = self._fix(old)
        expected = [
            f"_unmapped.policies.bogus: {UNMAPPED_FIELD_MESSAGE}",
            f"_unmapped.router.bogus: {UNMAPPED_FIELD_MESSAGE}",
        ]
        self.assertCountEqual(lines, expected)
        self.assertEqual(self._output()["_unmapped"], {"policies.bogus": FAKE_CRED, "router.bogus": FAKE_CRED})
        self.assertCountEqual(self._refused(), expected)

    def test_a_field_function_sends_a_field_to_unmapped_and_runs_before_config(self):
        # CONFIG would drop ib_netdev; its field function runs first and keeps a real value.
        old = copy.deepcopy(OLD_DISTRIBUTED)
        old["ib_netdev"] = FAKE_CRED
        old["sweeps"][DCELL] = {"client_timeout_s": 60}
        old["ib_hca_devices"] = ["mlx5_0", FAKE_CRED]
        lines = self._fix(old)
        unmapped = {"ib_netdev": FAKE_CRED, "ib_hca_devices": ["mlx5_0", FAKE_CRED]}
        expected = [f"_unmapped.{path}: {UNMAPPED_FIELD_MESSAGE}" for path in unmapped]
        self.assertEqual(lines, expected)
        new = self._output()
        self.assertEqual(new["_unmapped"], unmapped)
        # A per-cell client flag stays with its cell.
        self.assertEqual((new["sweeps"], new["runs"]), ({DCELL: {"client_timeout_s": 60}}, [DCELL]))
        self.assertEqual(self._refused(), expected)

    def test_a_key_left_beside_moved_keys_keeps_its_block_its_comment_and_its_list_position(self):
        old = copy.deepcopy(OLD_DISAGGREGATED)
        old["sweep"]["runs"] = [{"combo": CELL}, {"combo": CELL2, "weight": FAKE_CRED}]
        old["sweeps"][CELL2] = {}
        self.assertEqual(self._fix(old), [f"_unmapped.sweep: {UNMAPPED_FIELD_MESSAGE}"])
        new = self._output()
        self.assertEqual(
            new["_unmapped"], {"sweep": {"_comment": "Cells to run.", "runs": [{}, {"weight": FAKE_CRED}]}}
        )
        self.assertEqual(new["runs"], [CELL, CELL2])

    def test_a_missing_required_field_is_written_as_changeme(self):
        old = copy.deepcopy(OLD_DISTRIBUTED)
        del old["server_params"]["model"]
        del old["benchmark_params"]
        self.assertCountEqual(
            self._fix(old), [f"server_params.model: {MISSING_MESSAGE}", f"benchmark_params: {MISSING_MESSAGE}"]
        )
        new = self._output()
        self.assertEqual((new["server_params"]["model"], new["benchmark_params"]), ("<changeme>", "<changeme>"))
        self.assertCountEqual(
            self._refused(), [f"server_params.model: {PLACEHOLDER_MESSAGE}", f"benchmark_params: {PLACEHOLDER_MESSAGE}"]
        )

    def test_a_wrong_type_is_named_and_kept_for_the_user_to_fix(self):
        old = copy.deepcopy(OLD_DISTRIBUTED)
        old["server_params"]["tensor_parallel_size"] = "eight"
        old["benchmark_params"]["num_prompts"] = FAKE_CRED
        lines = self._fix(old)
        self.assertEqual([line.split(":")[0] for line in lines], ["server_params.tp", "benchmark_params.num_prompts"])
        self.assertEqual(self._output()["server_params"]["tp"], "eight")
        self.assertEqual(self._refused(), lines)

    def test_an_unknown_threshold_section_a_spec_with_no_new_kind_and_an_extra_spec_field_go_to_unmapped(self):
        old = copy.deepcopy(OLD_THRESHOLDS)
        old[CELL]["mean_ttft_ms"]["unit"] = FAKE_CRED
        old[CELL]["p99_ttft_ms"] = {"kind": "within", "value": 100, "tolerance_pct": 5}
        old[CELL]["request_throughput"] = {"kind": "min_ratio", "value": 0.9, "reference": FAKE_CRED}
        old["gpu"] = FAKE_CRED
        paths = [f"{CELL}.p99_ttft_ms", f"{CELL}.request_throughput", "gpu", f"{CELL}.mean_ttft_ms.unit"]
        expected = [f"_unmapped.{path}: {UNMAPPED_FIELD_MESSAGE}" for path in paths]
        self.assertEqual(self._fix(old, output="new_threshold.json"), expected)
        new = self._output("new_threshold.json")
        self.assertEqual(list(new)[0], "_unmapped")
        self.assertEqual(list(new["_unmapped"]), paths)
        self.assertEqual(new["_unmapped"][f"{CELL}.p99_ttft_ms"], old[CELL]["p99_ttft_ms"])
        self.assertEqual({key: value for key, value in new.items() if key != "_unmapped"}, NEW_THRESHOLDS)
        self._write("new.json", {**NEW_DISTRIBUTED, "threshold_json": "new_threshold.json"})
        self.assertEqual(self._refused(), expected)

    def test_a_threshold_file_is_checked_as_the_schemas_own_thresholds_type(self):
        # A ThresholdFile takes min; this schema's thresholds type takes only max, as its loader will.
        max_only = _map_with(DEMO_MAP, SCHEMA=_MaxOnlyConfig)
        old = {"_comment": "x", CELL: {k: v for k, v in OLD_THRESHOLDS[CELL].items() if k != "p99_itl_ms"}}
        with mock.patch.object(engine, "suite_maps", return_value={"demo": max_only}):
            lines = self._fix(old, output="new_threshold.json")
        self.assertEqual(lines, [f"{CELL}.output_throughput.kind: Input should be 'max'"])
        self._write("new.json", {**NEW_DISTRIBUTED, "threshold_json": "new_threshold.json"})
        self.assertEqual(self._refused(schema=_MaxOnlyConfig), lines)

    def test_a_threshold_file_that_is_missing_still_old_or_not_covering_is_named_as_load_names_it(self):
        covered = _map_with(DEMO_MAP, SCHEMA=_CoveredConfig)
        old = {**OLD_SINGLE, "enforce_thresholds": True}
        threshold_path = self.tmp / "demo_threshold.json"
        missing = [f"threshold_json: {MISSING_THRESHOLD_MESSAGE}"]
        still_old = [
            f"old config format: accuracy.gsm8k_flex, {CELL}.output_throughput, {CELL}.mean_ttft_ms; "
            f"run cvs config migrate demo {threshold_path} <new config>"
        ]
        not_covering = [f"(config): runs missing threshold coverage: [{CELL2!r}]"]
        for thresholds, at_load, prefix in (
            (None, missing, ""),
            (OLD_THRESHOLDS, still_old, "demo_threshold.json: "),
            (NEW_THRESHOLDS, not_covering, ""),
        ):
            with self.subTest(at_load=at_load):
                if thresholds is not None:
                    self._write("demo_threshold.json", thresholds)
                with mock.patch.object(engine, "suite_maps", return_value={"demo": covered}):
                    code, lines = self._run(old, force=True)
                self.assertEqual(code, 1)
                self.assertEqual([line.strip() for line in lines[1:]], [prefix + line for line in at_load])
                self.assertEqual(self._refused(schema=_CoveredConfig), at_load)

    def test_an_unmapped_block_in_the_input_is_listed_and_written_first_before_new_entries(self):
        old = {**copy.deepcopy(OLD_DISTRIBUTED), "_unmapped": {"server_params.max_model_len": FAKE_CRED}}
        lines = self._fix(old)
        self.assertEqual(lines, [f"_unmapped.server_params.max_model_len: {UNMAPPED_FIELD_MESSAGE}"])
        self.assertEqual(self._output(), {"_unmapped": old["_unmapped"], **NEW_DISTRIBUTED})
        old["notes"] = FAKE_CRED
        code, lines = self._run(old, force=True)
        self.assertEqual(code, 1)
        new = self._output()
        self.assertEqual(new["_unmapped"], {"server_params.max_model_len": FAKE_CRED, "notes": FAKE_CRED})
        self.assertEqual(list(new)[0], "_unmapped")
        self.assertEqual(self._refused(), [line.strip() for line in lines[1:]])

    def test_a_star_over_list_items_fills_another_list_in_order_without_gaps(self):
        old = copy.deepcopy(OLD_DISAGGREGATED)
        old["sweep"]["runs"] = [{"combo": CELL}, {"weight": FAKE_CRED}, {"combo": CELL2}]
        self.assertEqual(self._fix(old), [f"_unmapped.sweep: {UNMAPPED_FIELD_MESSAGE}"])
        new = self._output()
        self.assertEqual(new["runs"], [CELL, CELL2])
        self.assertEqual(new["_unmapped"]["sweep"]["runs"], [{}, {"weight": FAKE_CRED}])


class TestNothingWritten(_MigrateCase):
    def _stopped(self, data, **kwargs):
        code, lines = self._run(data, **kwargs)
        self.assertEqual((code, len(lines)), (2, 1))
        self.assertNotIn(FAKE_CRED, lines[0])
        return lines[0]

    def test_an_old_and_a_new_key_with_different_values_collide(self):
        old = copy.deepcopy(OLD_DISTRIBUTED)
        old["server_params"]["tp"] = FAKE_CRED
        self.assertEqual(
            self._stopped(old),
            "server_params.tp is set by both server_params.tensor_parallel_size and server_params.tp "
            "with different values; keep one, then run cvs config migrate again",
        )

    def test_a_value_where_a_moved_key_needs_a_block_collides_in_either_order(self):
        server = OLD_SINGLE["server_params"]
        cases = (
            ({"engine_args": FAKE_CRED, **server}, "server_params.engine_args and server_params.kv_cache_dtype"),
            ({**server, "engine_args": FAKE_CRED}, "server_params.kv_cache_dtype and server_params.engine_args"),
        )
        for params, origins in cases:
            with self.subTest(origins=origins):
                self.assertEqual(
                    self._stopped({**OLD_SINGLE, "server_params": params}),
                    f"server_params.engine_args is set by both {origins} with different values; "
                    "keep one, then run cvs config migrate again",
                )

    def test_a_refused_key_whose_old_path_the_inputs_unmapped_block_holds_with_another_value_stops_it(self):
        old = {**OLD_DISTRIBUTED, "_unmapped": {"notes": "kept"}, "notes": FAKE_CRED}
        self.assertEqual(
            self._stopped(old),
            "notes is both in the input's _unmapped block and set in the input, with different values; "
            "keep one, then run cvs config migrate again",
        )

    def test_a_threshold_task_id_both_old_and_under_lm_eval_stops_it(self):
        old = copy.deepcopy(OLD_THRESHOLDS)
        old["accuracy"]["lm_eval"] = {"gsm8k_flex": {}}
        self.assertEqual(
            self._stopped(old, output="new_threshold.json"),
            "accuracy.lm_eval and accuracy hold the same task id: gsm8k_flex",
        )

    def test_an_unknown_suite_lists_the_suites_or_none(self):
        self.assertEqual(
            self._stopped(OLD_SINGLE, suite="vllm"), "unknown suite vllm; the suites with a map are: demo, other"
        )
        with mock.patch.object(engine, "suite_maps", return_value={}):
            self.assertEqual(self._stopped(OLD_SINGLE), "unknown suite demo; the suites with a map are: none")

    def test_another_suites_file_is_named(self):
        source = self.tmp / "old.json"
        self.assertEqual(
            self._stopped({"server_params": {"model": "m", "tensor_parallelism": 8}}),
            f"{source} looks like other, not demo; run cvs config migrate other {source} <output>",
        )

    def test_a_missing_unreadable_empty_or_non_object_input_stops_it(self):
        source = self.tmp / "old.json"
        cases = (
            (None, f"cannot read {source}: No such file or directory"),
            ("", f"{source} is empty"),
            ("[1]", f"{source} must hold a JSON object"),
        )
        for data, message in cases:
            with self.subTest(data=data):
                self.assertEqual(self._stopped(data), message)
        source.write_bytes(b'{"model": "\xff"}')
        self.assertEqual(self._stopped(None), f"{source} is not UTF-8 text")
        source.unlink()
        source.mkdir()
        self.assertEqual(self._stopped(None), f"cannot read {source}: Is a directory")

    def test_invalid_json_names_the_line_and_column(self):
        self.assertEqual(
            self._stopped('{\n  "server_params": {\n    "model" "m"\n  }\n}\n'),
            f"{self.tmp / 'old.json'} is not valid JSON: Expecting ':' delimiter at line 3, column 13",
        )

    def test_a_map_target_with_more_stars_than_its_old_path_raises(self):
        bad = _map_with(DEMO_MAP, CONFIG={"server_params.tensor_parallel_size": "server_params.*.tp"})
        with mock.patch.object(engine, "suite_maps", return_value={"demo": bad}):
            with self.assertRaises(ValueError) as ctx:
                self._run(OLD_DISTRIBUTED)
        self.assertIn("server_params.*.tp", str(ctx.exception))
        self.assertFalse((self.tmp / "new.json").exists())


class TestOutputPath(_MigrateCase):
    def test_an_existing_directory_gets_the_input_file_name(self):
        (self.tmp / "out").mkdir()
        code, lines = self._run(OLD_DISTRIBUTED, output="out")
        self.assertEqual((code, lines), (0, [f"wrote {self.tmp / 'out' / 'old.json'}"]))
        self.assertEqual(self._output("out/old.json"), NEW_DISTRIBUTED)

    def test_an_existing_output_is_replaced_only_with_force(self):
        for output, target in (("new.json", "new.json"), ("out", "out/old.json")):
            with self.subTest(output=output):
                existing = self._write(target, {"keep": FAKE_CRED})
                self.assertEqual(
                    self._run(OLD_DISTRIBUTED, output=output),
                    (2, [f"{existing} already exists; use --force to overwrite it"]),
                )
                self.assertEqual(self._run(OLD_DISTRIBUTED, output=output, force=True)[0], 0)
                self.assertEqual(self._output(target), NEW_DISTRIBUTED)

    def test_the_output_may_not_be_the_input_even_through_its_folder_a_symlink_or_a_hard_link(self):
        source = self._write("old.json", OLD_DISTRIBUTED)
        (self.tmp / "link.json").symlink_to(source)
        os.link(source, self.tmp / "hard.json")
        for output in ("old.json", ".", "link.json", "hard.json"):
            named = source if output == "." else self.tmp / output
            with self.subTest(output=output):
                self.assertEqual(
                    self._run(OLD_DISTRIBUTED, output=output, force=True),
                    (2, [f"{named} is the input; write the new file elsewhere"]),
                )

    def test_missing_folders_are_created_and_an_unwritable_output_stops_it(self):
        self.assertEqual(self._run(OLD_DISTRIBUTED, output="a/b/new.json")[0], 0)
        self.assertEqual(self._output("a/b/new.json"), NEW_DISTRIBUTED)
        self._write("blocker", "a file, not a folder")
        code, lines = self._run(OLD_DISTRIBUTED, output="blocker/new.json")
        self.assertEqual((code, lines), (2, [f"cannot write {self.tmp / 'blocker' / 'new.json'}: File exists"]))

    def test_a_failed_write_keeps_the_old_output_and_leaves_no_temporary_file_or_new_folder(self):
        self._write("new.json", {"keep": FAKE_CRED})
        with mock.patch.object(engine.os, "replace", side_effect=OSError(errno.EIO, os.strerror(errno.EIO))):
            for output in ("new.json", "a/b/new.json"):
                with self.subTest(output=output):
                    self.assertEqual(
                        self._run(OLD_DISTRIBUTED, output=output, force=True),
                        (2, [f"cannot write {self.tmp / output}: {os.strerror(errno.EIO)}"]),
                    )
        self.assertEqual(self._output(), {"keep": FAKE_CRED})


if __name__ == "__main__":
    unittest.main()
