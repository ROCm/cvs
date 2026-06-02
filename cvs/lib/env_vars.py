'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Central registry of CVS environment variables.

Single source of truth for every CVS_* (and related) environment variable:
name, default, type, and description. All call sites should read through
``get()`` rather than calling ``os.environ`` directly, and the ``cvs env``
command renders this registry so the supported set never drifts from the docs.

Keep this module dependency-free (stdlib only) so it can be imported very early
(e.g. from cvs.extension) without risk of circular imports.
'''

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional


def _as_bool(value: str) -> bool:
    """Parse a truthy string value the same way across all CVS env vars."""
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class EnvVar:
    """Declarative description of a single supported environment variable."""

    name: str
    default: Any
    description: str
    category: str = 'general'
    cast: Callable[[str], Any] = str
    secret: bool = False

    def is_set(self) -> bool:
        return self.name in os.environ

    def raw(self) -> Optional[str]:
        return os.environ.get(self.name)

    def get(self) -> Any:
        """Return the casted value, or the default if unset/invalid."""
        raw = os.environ.get(self.name)
        if raw is None:
            return self.default
        try:
            return self.cast(raw)
        except (ValueError, TypeError):
            return self.default


ENV_VARS = [
    EnvVar(
        'CLUSTER_FILE',
        None,
        'Path to cluster configuration JSON file (fallback when --cluster_file is not given).',
        category='core',
    ),
    EnvVar(
        'CVS_HOSTS_PER_SHARD',
        32,
        'Hosts processed per parallel shard. 0 disables process sharding (single process).',
        category='parallel',
        cast=int,
    ),
    EnvVar(
        'CVS_WORKERS_PER_CPU',
        4,
        'Worker processes per CPU core (total workers = cpu_count * this).',
        category='parallel',
        cast=int,
    ),
    EnvVar(
        'CVS_PERSISTENT_SHARDS',
        False,
        'Reuse shard worker processes across operations instead of respawning.',
        category='parallel',
        cast=_as_bool,
    ),
    EnvVar(
        'CVS_JUMP_HOST',
        None,
        'SSH jump/bastion host to tunnel target connections through.',
        category='jump host',
    ),
    EnvVar(
        'CVS_JUMP_USER',
        None,
        'Username for the jump host.',
        category='jump host',
    ),
    EnvVar(
        'CVS_JUMP_PASSWORD',
        None,
        'Password for the jump host. Prefer CVS_JUMP_PKEY (key auth) where possible.',
        category='jump host',
        secret=True,
    ),
    EnvVar(
        'CVS_JUMP_PKEY',
        None,
        'Path to a private key file used to authenticate to the jump host.',
        category='jump host',
    ),
    EnvVar(
        'CVS_JUMP_PORT',
        22,
        'SSH port on the jump host.',
        category='jump host',
        cast=int,
    ),
    EnvVar(
        'CVS_EXTENSION_PKG_NAMES',
        None,
        'Comma-separated extension package names to load.',
        category='extensions',
    ),
]

ENV_VARS_BY_NAME = {e.name: e for e in ENV_VARS}


def get(name: str) -> Any:
    """Read + cast a registered env var by name.

    Raises KeyError if ``name`` is not a registered variable, which keeps
    call sites honest: every variable cvs reads must be declared in ENV_VARS.
    """
    return ENV_VARS_BY_NAME[name].get()
