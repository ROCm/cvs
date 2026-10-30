'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Turns a config file and a cluster file into a validated config, the resolved
cluster, and the merged container block.

Substitution and threshold loading come from
``cvs.lib.utils.config_loader.substitute_config``, which older suite loaders
still call directly.
'''

import copy
import json
from collections import namedtuple

from pydantic import ValidationError

from cvs.lib.utils.config_loader import substitute_config
from cvs.lib.utils_lib import resolve_cluster_config_placeholders


class ConfigError(ValueError):
    """A config file that does not match its suite's schema.

    The message has one ``location: problem`` line per error and never the
    offending values, which can be credentials.
    """

    def __init__(self, config_path, problems):
        self.problems = problems
        super().__init__(f"{config_path} is not a valid config:\n" + "\n".join(f"  {p}" for p in problems))


LoadedConfig = namedtuple("LoadedConfig", "config cluster container")


def _deep_merge(base, override):
    """Merge `override` onto `base`: dicts merge by key, lists and scalars are replaced."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base else v
    return out


def _problem_lines(exc):
    lines = []
    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "(config)"
        lines.append(f"{location}: {error['msg'].removeprefix('Value error, ')}")
    return lines


def load_config(model_cls, config_path, cluster):
    """Load and validate a suite config against its cluster.

    `cluster` is a cluster file path or an already-parsed cluster dict. A
    `<changeme>` left in the cluster file exits, as for every suite; one left in
    the config raises `ConfigError` listing each field that still has one.

    Returns `LoadedConfig(config, cluster, container)`. `container` is the
    cluster file's container block with the config's `container.model_dump()`
    merged onto it, the config winning on shared keys, or None when the model
    has no container. Cluster list values such as volumes are replaced here;
    the container orchestrator adds its own defaults to them. The merged
    container is a copy, so changing it does not change the cluster.
    """
    if not isinstance(cluster, dict):
        with open(cluster) as fp:
            cluster = json.load(fp)
    cluster = resolve_cluster_config_placeholders(cluster)
    raw, thresholds = substitute_config(config_path, cluster)
    if "thresholds" in model_cls.model_fields:
        raw["thresholds"] = thresholds
    try:
        config = model_cls.model_validate(raw, context={"placeholders": "reject"})
    except ValidationError as exc:
        raise ConfigError(config_path, _problem_lines(exc)) from None
    spec = getattr(config, "container", None)
    if spec is None:
        container = None
    else:
        container = copy.deepcopy(_deep_merge(cluster.get("container") or {}, spec.model_dump()))
    return LoadedConfig(config, cluster, container)
