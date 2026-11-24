'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

vLLM config model and loader.

`VllmConfig` is the shared `InferenceVariantConfig` plus the `vllm` block and
the checks only vLLM needs. `load_variant` loads a config through the shared
inference load path with vLLM's old-key tables below, so a 0.3.0 config and
its threshold file still load, with one warning.
'''

import copy
import json
from collections import namedtuple
from typing import List, Literal, Optional, Union

from pydantic import Field, PrivateAttr, TypeAdapter, ValidationError, create_model, field_validator, model_validator

from cvs.lib.inference.utils.vllm_metrics import validate_threshold_spec
from cvs.lib.utils.config_loader import ConfigError  # noqa: F401 - re-exported for vLLM callers
from cvs.lib.utils.config_loader import load_variant as load_inference_variant
from cvs.parsers.schemas import (
    OLD_KEY_DROPPED,
    OLD_KEY_UNMAPPED,
    InferenceVariantConfig,
    OldKey,
    ServingBenchmarkParams,
    ServingServerParams,
    _Forbid,
    check_container_network_env,
    check_flags,
    check_not_implemented,
    keep_sweep_cells,
    not_implemented_names,
    old_accuracy_rule,
    old_timeout_rule,
    parse_cell_key,
    resolve_old_keys,
)

# Server flags the vLLM job writes itself from the shared fields and the vllm block.
_HARNESS_SERVER_FLAGS = frozenset(
    {"headless", "tensor_parallel_size", "pipeline_parallel_size", "max_model_len", "distributed_executor_backend"}
)
# Client flags that take false: the job always sends --ignore-eos unless told not to.
_BOOLEAN_CLIENT_FLAGS = frozenset({"ignore_eos", "trust_remote_code"})
# Client flags the job sets from timeouts.client_s.
_CLIENT_WAIT_FLAGS = frozenset({"client_initial_wait_s", "client_poll_iterations", "client_poll_wait_s"})
_SPECULATIVE_ENGINE_ARGS = frozenset({"speculative_config", "speculative_model", "num_speculative_tokens"})
# Shared fields vLLM does not run yet, in the order the error names them.
_NOT_IMPLEMENTED_FIELDS = (
    "server_params.dp",
    "server_params.ep",
    "server_params.prefill_nnodes",
    "server_params.long_context",
    *(f"server_params.engine_args.{name}" for name in sorted(_SPECULATIVE_ENGINE_ARGS)),
    "benchmark_params.scaling_baseline_output_throughput",
    "accuracy.custom_eval",
    "accuracy.quant_parity",
    "accuracy.mtp_quality",
)


def serialize_cli_options(options):
    """Serialize a snake_case option map into vLLM command-line flags; true writes a bare flag."""
    check_flags(options, "option", ())
    _refuse_false(options, "option")
    argv = []
    for name, value in options.items():
        if value is None:
            continue
        flag = f"--{name.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *(str(item) for item in value)])
        elif isinstance(value, dict):
            argv.extend([flag, json.dumps(value, separators=(",", ":"))])
        else:
            argv.extend([flag, str(value)])
    return argv


RunCell = namedtuple("RunCell", "key isl osl tp pp concurrency")
ResolvedRun = namedtuple("ResolvedRun", "cell benchmark_params")


def parse_run_cell(key):
    return RunCell(key, **parse_cell_key(key))


def _refuse_false(flags, section, allowed=frozenset()):
    for name, value in flags.items():
        if name in allowed:
            if not isinstance(value, bool):
                raise ValueError(f"{section}.{name} must be true or false")
        elif value is False:
            raise ValueError(f"{section}.{name}=false is ambiguous; omit it or use the option's negative flag")


def _check_vllm(self):
    server = self.server_params
    check_not_implemented("vLLM", not_implemented_names(self, fields=_NOT_IMPLEMENTED_FIELDS))

    check_flags(server.engine_args, "server_params.engine_args", _HARNESS_SERVER_FLAGS)
    _refuse_false(server.engine_args, "server_params.engine_args")
    check_flags(self.benchmark_params.client_args, "benchmark_params.client_args", _CLIENT_WAIT_FLAGS)
    _refuse_false(self.benchmark_params.client_args, "benchmark_params.client_args", _BOOLEAN_CLIENT_FLAGS)
    for key, overrides in self.sweeps.items():
        check_flags(overrides, f"sweeps.{key}", _CLIENT_WAIT_FLAGS)
        _refuse_false(overrides, f"sweeps.{key}", _BOOLEAN_CLIENT_FLAGS)

    check_container_network_env(self.container)

    unknown = sorted(set(self.thresholds.cells) - set(self.sweeps))
    if unknown:
        raise ValueError(f"threshold cells have no matching sweep: {unknown}")
    for cell_key, cell in self.thresholds.cells.items():
        for metric, spec in cell.items():
            if "optional" in spec.model_fields_set:
                raise ValueError(f"{cell_key}.{metric}: vLLM thresholds do not support optional")
            validate_threshold_spec(metric, spec.model_dump())
    self._run_cells = {key: parse_run_cell(key) for key in self.sweeps}
    return self


def _refuse_nnodes(cls, value):
    if isinstance(value, dict) and "nnodes" in value:
        raise ValueError("server_params cannot override harness fields: ['nnodes']")
    return value


def _refuse_atom_block(cls, data):
    if isinstance(data, dict) and "atom" in data:
        raise ValueError("atom: not implemented for vLLM; a vLLM config holds the vllm block only")
    return data


class _VllmVariant(InferenceVariantConfig):
    _effective_topology = PrivateAttr(default=None)
    # The suite module that runs the config; vllm_distributed on one host runs in single mode.
    _suite = PrivateAttr(default="")
    _run_cells = PrivateAttr(default_factory=dict)

    @property
    def model_id(self):
        return self.server_params.model

    def bind_effective_topology(self, topology, suite=""):
        self._effective_topology = topology
        self._suite = suite

    def cell(self, key):
        return self._run_cells[key]

    def expected_cells(self):
        return list(self.runs)

    def cell_key(self, isl, osl, concurrency, **_unused):
        return f"ISL={isl},OSL={osl},TP={self.server_params.tp},PP={self.server_params.pp},CONC={concurrency}"

    def client_values(self, overrides=None):
        """The shared client fields, then ``client_args``, then ``overrides``, as one flat map."""
        values = self.benchmark_params.model_dump(exclude={"client_args", "scaling_baseline_output_throughput"})
        values.update(self.benchmark_params.client_args)
        values.update(overrides or {})
        return values

    def resolved_runs(self):
        resolved = []
        for key in self.runs:
            cell = self.cell(key)
            values = self.client_values(self.sweeps[key])
            values.update(
                {"random_input_len": cell.isl, "random_output_len": cell.osl, "max_concurrency": cell.concurrency}
            )
            resolved.append(ResolvedRun(cell=cell, benchmark_params=values))
        return resolved


VllmBlock = create_model(
    "VllmBlock",
    __base__=_Forbid,
    distributed_executor_backend=(Literal["mp", "ray"], "mp"),
)

VllmConfig = create_model(
    "VllmConfig",
    __base__=_VllmVariant,
    __validators__={
        "_refuse_atom_block": model_validator(mode="before")(_refuse_atom_block),
        "_refuse_nnodes": field_validator("server_params", mode="before")(_refuse_nnodes),
        "_check_vllm": model_validator(mode="after")(_check_vllm),
    },
    vllm=(VllmBlock, Field(default_factory=VllmBlock)),
    # "auto" discovers the devices.
    ib_hca_devices=(Union[Literal["auto"], List[str], None], "auto"),
    ib_netdev=(Optional[str], None),
)


# ---------- old vLLM keys ----------

_OLD_KEYS = {
    "server_params.tensor_parallel_size": "server_params.tp",
    "server_params.pipeline_parallel_size": "server_params.pp",
    "server_params.dist_init_port": "server_params.master_port",
    "server_params.distributed_executor_backend": "vllm.distributed_executor_backend",
    "accuracy.tasks": "accuracy.lm_eval.tasks",
    "accuracy.quant_parity.reference_config_stem": OLD_KEY_DROPPED,
}
_OLD_SERVER_FIELDS = frozenset(
    {
        "backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "dist_init_port",
        "max_model_len",
        "server_poll_iterations",
        "server_poll_wait_s",
        "server_warmup_wait_s",
        "distributed_executor_backend",
    }
)
_OLD_CLIENT_FIELDS = frozenset(
    {
        "backend",
        "dataset_name",
        "client_poll_iterations",
        "client_poll_wait_s",
        "client_initial_wait_s",
        "trust_remote_code",
        "ignore_eos",
    }
)
# The old wait and poll keys, each with the old schema's default for a key the config leaves out.
_OLD_SERVER_WAITS = (
    ("server_params", "server_warmup_wait_s", 330),
    ("server_params", "server_poll_iterations", 60),
    ("server_params", "server_poll_wait_s", 60),
)
_OLD_CLIENT_WAITS = (
    ("benchmark_params", "client_initial_wait_s", 120),
    ("benchmark_params", "client_poll_iterations", 20),
    ("benchmark_params", "client_poll_wait_s", 60),
)
# Old client keys that move to client_args only when they change the client command.
_OLD_CLIENT_DEFAULTS = {"dataset_name": "random", "ignore_eos": True, "trust_remote_code": False}
_OLD_CELL_TIMEOUT_KEYS = _CLIENT_WAIT_FLAGS
_OLD_DEFAULT_NUM_PROMPTS = 3200
_OLD_THRESHOLD_KINDS = {"max_ms": "max", "min_tok_s": "min"}
_NEW_SERVER_FIELDS = frozenset(ServingServerParams.model_fields)
_NEW_CLIENT_FIELDS = frozenset(ServingBenchmarkParams.model_fields)


def _section(config, name):
    value = config.get(name) if isinstance(config, dict) else None
    return value if isinstance(value, dict) else {}


def _is_old_config(config):
    return "tensor_parallel_size" in _section(config, "server_params")


def _first_wait_plus_polls(first_wait, polls, per_poll):
    return first_wait + polls * per_poll


def _fixed_value_row(value, expected):
    return OLD_KEY_DROPPED if value == expected else OLD_KEY_UNMAPPED


def _client_rule(config):
    rows = {}
    server = _section(config, "server_params")
    benchmark = _section(config, "benchmark_params")
    if "backend" in server:
        rows["server_params.backend"] = _fixed_value_row(server["backend"], "vllm")
    if "backend" in benchmark:
        rows["benchmark_params.backend"] = _fixed_value_row(benchmark["backend"], "vllm")
    for key, default in _OLD_CLIENT_DEFAULTS.items():
        if key in benchmark:
            value = _old_value(benchmark[key], default)
            if value == default:
                rows[f"benchmark_params.{key}"] = OLD_KEY_DROPPED
            else:
                rows[f"benchmark_params.{key}"] = OldKey(f"benchmark_params.client_args.{key}", value=value)
    if _is_old_config(config) and "master_port" in server:
        raise ValueError("server_params cannot override harness fields: ['master_port']")
    return rows


def _old_value(value, default):
    """An old value as the old schema read it, so a quoted boolean keeps its meaning."""
    if isinstance(default, bool):
        try:
            return TypeAdapter(bool).validate_python(value)
        except ValidationError:
            return value
    return value


def _flags_rule(config):
    """Move the native flags an old config wrote beside the harness fields."""
    if not _is_old_config(config):
        return {}
    rows = {}
    server = _section(config, "server_params")
    if "max_model_len" in server:
        usable = not isinstance(server["max_model_len"], bool)
        rows["server_params.max_model_len"] = "server_params.max_model_length" if usable else OLD_KEY_UNMAPPED
    for section, old_fields, new_fields, target in (
        ("server_params", _OLD_SERVER_FIELDS, _NEW_SERVER_FIELDS, "engine_args"),
        ("benchmark_params", _OLD_CLIENT_FIELDS, _NEW_CLIENT_FIELDS, "client_args"),
    ):
        for key in _section(config, section):
            if key not in old_fields and key not in new_fields and not key.startswith("_"):
                rows[(section, key)] = OldKey((section, target, key))
    if "num_prompts" not in _section(config, "benchmark_params"):
        rows["benchmark_params.num_prompts"] = OldKey("benchmark_params.num_prompts", value=_OLD_DEFAULT_NUM_PROMPTS)
    return rows


def resolve_old_vllm_keys(raw):
    """Map an old vLLM config, with its attached ``thresholds``, to the current names.

    Returns the converted config and one warning message, or the input and
    ``None`` when it holds no old key.
    """
    converted, message = resolve_old_keys(
        raw,
        _OLD_KEYS,
        rules=(
            _client_rule,
            old_timeout_rule(
                "timeouts.server_ready_s", _OLD_SERVER_WAITS, _first_wait_plus_polls, force=_is_old_config
            ),
            old_timeout_rule(
                "timeouts.client_s",
                _OLD_CLIENT_WAITS,
                _first_wait_plus_polls,
                force=_is_old_config,
                sweep_keys=_OLD_CELL_TIMEOUT_KEYS,
            ),
            _flags_rule,
            old_accuracy_rule(_OLD_THRESHOLD_KINDS),
        ),
        message="Deprecated vLLM config keys: {keys}. Use the new names; old keys will be removed in a future release.",
        label="vLLM config",
    )
    return keep_sweep_cells(raw, converted), message


def load_variant(config_path, cluster_dict, warn=True):
    """Load and validate one vLLM config against its resolved cluster dict.

    Runs the shared inference load path with `VllmConfig` and vLLM's old-key
    tables: placeholders, the threshold file it names (none records every
    metric without gating), one warning naming each old key unless ``warn`` is
    false, then validation. A config that does not validate raises
    `ConfigError` naming each field, never a value.
    """
    return load_inference_variant(
        config_path, cluster_dict, VllmConfig, "vLLM", resolve_old=resolve_old_vllm_keys, warn=warn
    )


def _deep_merge(base, override):
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for key, value in override.items():
        out[key] = _deep_merge(base[key], value) if key in base else value
    return out


def merged_container(cluster_dict, config):
    """The cluster's container block with the config's merged over it, as a copy.

    Dicts merge by key; lists and scalars from the config replace the cluster's.
    """
    return copy.deepcopy(_deep_merge(cluster_dict.get("container") or {}, config.container.model_dump()))
