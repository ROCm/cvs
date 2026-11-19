'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for the ATOM section of cvs.parsers.schemas: AtomConfig and the
conversion of older ATOM config layouts.
'''

import unittest
import warnings

from pydantic import ValidationError

from cvs.lib.inference.atom.atom_parsing import GATED_METRICS
from cvs.parsers.schemas import (
    AtomConfig,
    atom_flag_tokens_to_args,
    resolve_old_atom_config,
    validation_problem_lines,
)

CELL = "ISL=1024,OSL=1024,TP=8,PP=1,CONC=16"
PP2_CELL = "ISL=1024,OSL=1024,TP=8,PP=2,CONC=16"
PATHS = {"shared_fs": "/fs", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/fs/.hf"}
CONTAINER = {"name": "atom", "image": "rocm/atom:latest", "runtime": {"name": "docker", "args": {}}}
GATED = {name: {"kind": "min", "value": 0} for name in GATED_METRICS}


def _atom_dict(server=None, cell=CELL, **overrides):
    value = {
        "paths": PATHS,
        "container": CONTAINER,
        "server_params": {"model": "m", "tp": 8, **(server or {})},
        "benchmark_params": {"num_prompts": 32},
        "sweeps": {cell: {}},
        "runs": [cell],
        "atom": {"driver": "atom"},
    }
    value.update(overrides)
    return value


def _problems(data):
    try:
        AtomConfig.model_validate(data)
    except ValidationError as exc:
        return "\n".join(validation_problem_lines(exc))
    raise AssertionError("expected a validation error")


def _old_flat(params=None, roles=None, **overrides):
    value = {
        "schema_version": 1,
        "framework": "atom",
        "paths": PATHS,
        "model": {"id": "m", "remote": 0},
        "container": CONTAINER,
        "roles": {"server": {"atom_args": ["-tp", "8", "--kv_cache_dtype", "fp8"], **(roles or {})}},
        "params": {"driver": "atom", "tensor_parallelism": "8", **(params or {})},
        "sweeps": {CELL: {}},
        "runs": [],
    }
    value.update(overrides)
    return value


def _old_serving(server=None, bench=None, **overrides):
    value = {
        "paths": PATHS,
        "model": {"id": "m", "remote": 0},
        "container": CONTAINER,
        "server_params": {"backend": "vllm", "tensor_parallelism": "8", **(server or {})},
        "benchmark_params": {"num_prompts": "100", **(bench or {})},
        "sweeps": {CELL: {}},
        "runs": [],
    }
    value.update(overrides)
    return value


class TestAtomConfig(unittest.TestCase):
    def test_minimal_config_validates(self):
        config = AtomConfig.model_validate(_atom_dict())
        self.assertEqual(config.atom.driver, "atom")

    def test_atom_block_is_required(self):
        data = _atom_dict()
        del data["atom"]
        self.assertIn("atom: Field required", _problems(data))

    def test_each_refused_field_fails_naming_atom(self):
        cases = (
            ({"dp": 2}, "server_params.dp"),
            ({"ep": 2}, "server_params.ep"),
            ({"prefill_nnodes": 1}, "server_params.prefill_nnodes"),
            ({"long_context": {"chunked_prefill_size": 8192}}, "server_params.long_context"),
        )
        for server, path in cases:
            with self.subTest(path=path):
                self.assertIn(f"{path} is not implemented for AtomConfig", _problems(_atom_dict(server=server)))

    def test_per_cell_overrides_are_refused(self):
        problems = _problems(_atom_dict(sweeps={CELL: {"temperature": 0.0}}))
        self.assertIn("per-cell overrides are not implemented for AtomConfig", problems)

    def test_atom_driver_refuses_multinode_pipeline_parallel(self):
        data = _atom_dict(server={"pp": 2, "nnodes": 2}, cell=PP2_CELL)
        self.assertIn("standalone ATOM cannot run pipeline parallel", _problems(data))

    def test_pp_drivers_need_pp_and_nnodes_together(self):
        multinode_pp1 = _atom_dict(server={"nnodes": 2}, atom={"driver": "vllm_atom"})
        self.assertIn("needs server_params.pp > 1", _problems(multinode_pp1))
        single_pp2 = _atom_dict(server={"pp": 2}, cell=PP2_CELL, atom={"driver": "sglang"})
        self.assertIn("needs server_params.nnodes > 1", _problems(single_pp2))
        ray = _atom_dict(
            server={"nnodes": 2, "engine_args": {"distributed_executor_backend": "ray"}}, atom={"driver": "vllm"}
        )
        AtomConfig.model_validate(ray)

    def test_multinode_scaling_cells_need_efficiency_when_enforced(self):
        data = _atom_dict(
            server={"pp": 2, "nnodes": 2},
            cell=PP2_CELL,
            atom={"driver": "vllm_atom"},
            benchmark_params={"num_prompts": 32, "scaling_baseline_output_throughput": 1500},
            threshold_json="t.json",
            thresholds={PP2_CELL: GATED},
        )
        self.assertIn("scaling.efficiency_pct in every run cell", _problems(data))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            AtomConfig.model_validate({**data, "enforce_thresholds": False})
        self.assertEqual(len(caught), 1)
        self.assertIn("scaling.efficiency_pct", str(caught[0].message))
        covered = {PP2_CELL: {**GATED, "scaling.efficiency_pct": {"kind": "min", "value": 50}}}
        AtomConfig.model_validate({**data, "thresholds": covered})

    def test_gated_metric_gaps_fail_when_enforced(self):
        data = _atom_dict(
            threshold_json="t.json", thresholds={CELL: {"output_throughput": {"kind": "min", "value": 1}}}
        )
        self.assertIn("cells missing gated-metric specs", _problems(data))


class TestFlagTokens(unittest.TestCase):
    def test_tokens_become_snake_case_args(self):
        args = atom_flag_tokens_to_args(
            ["--kv-cache-dtype", "fp8", "--trust-remote-code", "--x=1", "--d", "a", "--d", "b"], "flags"
        )
        self.assertEqual(args, {"kv_cache_dtype": "fp8", "trust_remote_code": True, "x": "1", "d": ["a", "b"]})

    def test_a_token_that_is_not_a_flag_fails_by_position(self):
        with self.assertRaises(ValueError) as caught:
            atom_flag_tokens_to_args(["secret-value"], "server_params.add_flags")
        self.assertIn("server_params.add_flags[0]", str(caught.exception))
        self.assertNotIn("secret-value", str(caught.exception))


class TestOldFlatLayout(unittest.TestCase):
    def test_params_and_roles_move_to_the_shared_blocks(self):
        config, _, message = resolve_old_atom_config(
            _old_flat(params={"port_no": "8001", "bench_extra_args": "--use-chat-template"}), {}, "mi3xx"
        )
        self.assertEqual(config["atom"], {"driver": "atom"})
        self.assertEqual(config["server_params"]["port"], 8001)
        self.assertEqual(config["server_params"]["tp"], 8)
        self.assertEqual(config["server_params"]["engine_args"], {"kv_cache_dtype": "fp8"})
        self.assertEqual(config["benchmark_params"]["client_args"], {"use_chat_template": True})
        self.assertEqual(config["benchmark_params"]["random_range_ratio"], 0.8)
        self.assertEqual(config["runs"], [CELL])
        for name in ("params", "roles", "model", "schema_version", "framework"):
            self.assertNotIn(name, config)
        self.assertIn("params.port_no -> server_params.port", message)
        self.assertIn("params.bench_extra_args -> benchmark_params.client_args", message)
        AtomConfig.model_validate(config)

    def test_tp_flag_that_disagrees_with_tensor_parallelism_fails(self):
        raw = _old_flat(roles={"atom_args": ["-tp", "4"]})
        with self.assertRaises(ValueError) as caught:
            resolve_old_atom_config(raw, {}, "mi3xx")
        self.assertIn("roles.server.atom_args sets -tp 4", str(caught.exception))

    def test_tp_flag_forms_and_missing_value(self):
        config, _, _ = resolve_old_atom_config(
            _old_flat(roles={"atom_args": ["--tensor-parallel-size=8", "--x"]}), {}, "mi3xx"
        )
        self.assertEqual(config["server_params"]["engine_args"], {"x": True})
        with self.assertRaises(ValueError) as caught:
            resolve_old_atom_config(_old_flat(roles={"atom_args": ["--x", "-tp"]}), {}, "mi3xx")
        self.assertIn("roles.server.atom_args[1] sets the tensor parallel size without a value", str(caught.exception))

    def test_false_serve_arg_has_no_place(self):
        raw = _old_flat(params={"driver": "vllm_atom"}, roles={"atom_args": [], "serve_args": {"enforce-eager": False}})
        with self.assertRaises(ValueError) as caught:
            resolve_old_atom_config(raw, {}, "mi3xx")
        self.assertIn("roles.server.serve_args.enforce-eager", str(caught.exception))

    def test_poll_keys_compute_the_timeouts(self):
        params = {"client_poll_count": "80", "server_poll_count": "120"}
        config, _, _ = resolve_old_atom_config(_old_flat(params=params), {}, "mi3xx")
        self.assertEqual(config["timeouts"], {"server_ready_s": 30 + 330 + 120 * 60, "client_s": 120 + 80 * 60})

    def test_env_sources_merge_into_container_env(self):
        raw = _old_flat(roles={"env": {"A": "1"}}, model={"id": "m", "remote": 0, "precision": "mxfp4"})
        raw["container"] = {**CONTAINER, "runtime": {"name": "docker", "args": {"env": {"B": "2"}}}}
        gfx942, _, message = resolve_old_atom_config(raw, {}, "mi3xx")
        self.assertEqual(
            gfx942["container"]["env"], {"A": "1", "B": "2", "ATOM_USE_TRITON_MOE": "1", "ATOM_USE_TRITON_GEMM": "1"}
        )
        self.assertNotIn("env", gfx942["container"]["runtime"]["args"])
        self.assertIn("model.precision -> container.env", message)
        gfx950, _, message = resolve_old_atom_config(raw, {}, "mi355x")
        self.assertEqual(gfx950["container"]["env"], {"A": "1", "B": "2"})
        self.assertIn("model.precision (dropped)", message)

    def test_explicit_ib_devices_become_nccl_env(self):
        raw = _old_flat(roles={"ib_hca_devices": ["mlx5_0", "mlx5_1"], "ib_netdev": "eth0"})
        config, _, _ = resolve_old_atom_config(raw, {}, "mi3xx")
        env = config["container"]["env"]
        self.assertEqual(env["NCCL_IB_HCA"], "mlx5_0,mlx5_1")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "eth0")
        auto, _, _ = resolve_old_atom_config(_old_flat(roles={"ib_hca_devices": "auto"}), {}, "mi3xx")
        self.assertNotIn("env", auto["container"])

    def test_param_with_one_working_value_fails_when_changed(self):
        with self.assertRaises(ValueError) as caught:
            resolve_old_atom_config(_old_flat(params={"metric_percentiles": "50,90"}), {}, "mi3xx")
        self.assertIn("params.metric_percentiles", str(caught.exception))
        self.assertNotIn("50,90", str(caught.exception))
        config, _, _ = resolve_old_atom_config(_old_flat(params={"metric_percentiles": "95,99"}), {}, "mi3xx")
        self.assertNotIn("params", config)

    def test_dropped_suite_switches_are_named(self):
        raw = _old_flat(functional={"api_smoke": True}, platform={"dmesg_scan": False})
        config, _, message = resolve_old_atom_config(raw, {}, "mi3xx")
        self.assertNotIn("functional", config)
        self.assertIn("functional (dropped)", message)
        self.assertIn("platform (dropped)", message)


class TestOldServingLayout(unittest.TestCase):
    def test_backend_port_and_bench_args_move(self):
        raw = _old_serving(
            server={"port_no": "8000", "dist_init_port": "29600", "serve_args": {"kv-cache-dtype": "fp8"}},
            bench={"seed": "5", "client_poll_count": "150"},
        )
        config, _, message = resolve_old_atom_config(raw, {}, "mi3xx")
        server = AtomConfig.model_validate(config).server_params
        self.assertEqual(config["atom"], {"driver": "vllm_atom"})
        self.assertEqual((server.model, server.tp, server.port, server.master_port), ("m", 8, 8000, 29600))
        self.assertEqual(server.engine_args, {"kv_cache_dtype": "fp8"})
        self.assertNotIn("seed", config["benchmark_params"])
        self.assertEqual(config["benchmark_params"]["num_prompts"], "100")
        self.assertEqual(config["timeouts"], {"client_s": 120 + 150 * 60})
        self.assertFalse(config["enforce_thresholds"])
        self.assertIn("server_params.backend -> atom.driver", message)
        self.assertIn("benchmark_params.seed (dropped)", message)
        AtomConfig.model_validate(config)

    def test_router_port_wins_and_sglang_flags_parse(self):
        raw = _old_serving(
            server={
                "backend": "sglang",
                "port_no": "8000",
                "proxy_router_serv_port": "9000",
                "add_flags": ["--mem-fraction-static", "0.9", "--disable-cuda-graph"],
            }
        )
        config, _, _ = resolve_old_atom_config(raw, {}, "mi3xx")
        self.assertEqual(config["server_params"]["port"], 9000)
        self.assertEqual(
            config["server_params"]["engine_args"], {"mem_fraction_static": "0.9", "disable_cuda_graph": True}
        )

    def test_empty_runs_take_the_sweeps_keys(self):
        other = "ISL=128,OSL=32,TP=8,PP=1,CONC=1"
        config, _, _ = resolve_old_atom_config(_old_serving(sweeps={CELL: {}, other: {}}), {}, "mi3xx")
        self.assertEqual(config["runs"], [CELL, other])

    def test_unsupported_backend_fails(self):
        with self.assertRaises(ValueError):
            resolve_old_atom_config(_old_serving(server={"backend": "trtllm"}), {}, "mi3xx")


class TestOldProfilesAndThresholds(unittest.TestCase):
    def _profiles(self):
        return {
            "schema_version": 2,
            "framework": "atom",
            "default_profile": "perf",
            "paths": PATHS,
            "model": {"id": "m", "remote": 0},
            "profiles": {
                "perf": {"container": CONTAINER, "params": {"driver": "atom"}, "sweeps": {CELL: {}}, "runs": []},
                "mtp3": {
                    "container": CONTAINER,
                    "params": {"driver": "atom", "bench_extra_args": "--use-chat-template"},
                    "sweeps": {CELL: {}},
                    "runs": [],
                    "threshold_profile": "mtp",
                },
            },
        }

    def test_profile_is_selected_and_its_threshold_profile_applied(self):
        thresholds = {"profiles": {"perf": {CELL: GATED}, "mtp": {CELL: {"failed": {"kind": "max", "value": 0}}}}}
        config, converted, message = resolve_old_atom_config(self._profiles(), thresholds, "mi3xx", "mtp3")
        self.assertEqual(config["benchmark_params"]["client_args"], {"use_chat_template": True})
        self.assertEqual(converted, {CELL: {"failed": {"kind": "max", "value": 0}}})
        self.assertIn("profiles.mtp3.params -> params", message)
        default, _, _ = resolve_old_atom_config(self._profiles(), thresholds, "mi3xx")
        self.assertNotIn("client_args", default["benchmark_params"])

    def test_unknown_profile_fails(self):
        with self.assertRaises(ValueError) as caught:
            resolve_old_atom_config(self._profiles(), {}, "mi3xx", "missing")
        self.assertIn("unknown config profile 'missing'", str(caught.exception))

    def test_old_threshold_names_and_kinds_convert(self):
        unrun = "ISL=8,OSL=8,TP=8,PP=2,CONC=1"
        thresholds = {
            CELL: {
                "mean_ttft_ms": {"kind": "max_ms", "value": 5},
                "output_throughput": {"kind": "min_tok_s", "value": 1},
            },
            unrun: {"failed": {"kind": "max", "value": 0}},
            "accuracy": {"gsm8k_flex": {"gsm8k.exact_match__flexible-extract": {"kind": "min", "value": 0.9}}},
            "long_context_accuracy": {"niah_8k": {"accuracy.niah_pass_rate__niah_8k": {"kind": "min", "value": 0.8}}},
            "quant_parity": {"quant_parity.probe_match": {"kind": "min", "value": 1}},
        }
        _, converted, message = resolve_old_atom_config(_old_flat(), thresholds, "mi3xx")
        self.assertEqual(
            converted[CELL],
            {"mean_ttft_ms": {"kind": "max", "value": 5}, "output_throughput": {"kind": "min", "value": 1}},
        )
        self.assertNotIn(unrun, converted)
        self.assertEqual(set(converted["accuracy"]), {"lm_eval", "custom_eval", "quant_parity"})
        self.assertIn("gsm8k_flex", converted["accuracy"]["lm_eval"])
        self.assertIn("niah_8k", converted["accuracy"]["custom_eval"])
        self.assertIn("threshold file:", message)

    def test_config_without_atom_block_or_old_keys_is_left_to_validation(self):
        config = _atom_dict()
        del config["atom"]
        same, _, message = resolve_old_atom_config(config, {}, "mi3xx")
        self.assertIs(same, config)
        self.assertIsNone(message)
        self.assertIn("atom: Field required", _problems(same))

    def test_current_config_refuses_a_profiled_threshold_file(self):
        with self.assertRaises(ValueError) as caught:
            resolve_old_atom_config(_atom_dict(), {"profiles": {"perf": {CELL: GATED}}}, "mi3xx")
        self.assertIn("own threshold file", str(caught.exception))

    def test_current_config_converts_only_an_old_threshold_file(self):
        config = _atom_dict()
        same, thresholds, message = resolve_old_atom_config(config, {CELL: GATED}, "mi3xx")
        self.assertIs(same, config)
        self.assertIsNone(message)
        _, thresholds, message = resolve_old_atom_config(
            config, {CELL: {"failed": {"kind": "max_ms", "value": 0}}}, "mi3xx"
        )
        self.assertEqual(thresholds[CELL]["failed"]["kind"], "max")
        self.assertIn("threshold file:", message)
        with self.assertRaises(ValueError):
            resolve_old_atom_config(config, {}, "mi3xx", "perf")


if __name__ == "__main__":
    unittest.main()
