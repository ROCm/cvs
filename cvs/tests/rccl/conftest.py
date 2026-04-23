"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Shared fixtures for the RCCL test suite. Scoped to cvs/tests/rccl/ only so
the legacy non-RCCL test suites (health, ibperf, training, inference, ...)
are unaffected -- they keep their per-module fixtures and continue to load
JSON / build Pssh handles directly, exactly as on main.

What lives here:
  * cluster_file / config_file - thin pytest CLI accessors.
  * cluster_dict               - the raw cluster JSON (post-placeholder
                                 resolution), as cfg.raw -> resolve. Provides
                                 the legacy escape-hatch consumed by
                                 cvs/lib/utils_lib + verify_lib.
  * orch_config                - the parsed OrchestratorConfig from load_config.
  * orch                       - the new Orchestrator built from orch_config.
                                 Calls orch.setup() before yielding (which
                                 brings up the runtime AND runs PREPARE_PIPELINE,
                                 i.e. MultinodeSshPhase if applicable) and
                                 orch.cleanup() afterwards.

What does NOT live here:
  * config_dict - per-suite config files have a top-level wrapper key
                  ({"rccl": {...}}). Each test module pulls its own key in a
                  one-line per-module fixture; cleaning up the wrapper-key
                  schema is a separate PR's problem.
"""

from __future__ import annotations

import pytest

from cvs.core import create_orchestrator, load_config
from cvs.lib import globals
from cvs.lib.utils_lib import resolve_cluster_config_placeholders


log = globals.log


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def orch_config(cluster_file, config_file):
    """Parsed OrchestratorConfig. Single canonical loader for new code."""
    return load_config(cluster_file, config_file)


@pytest.fixture(scope="module")
def cluster_dict(orch_config):
    """Raw cluster JSON (with placeholders resolved) for legacy library
    helpers (cvs/lib/utils_lib, cvs/lib/verify_lib). Resolution stays in the
    fixture (working agreement: loader does not silently rewrite user input).
    """
    raw = dict(orch_config.raw)
    raw.pop("__testsuite__", None)
    resolved = resolve_cluster_config_placeholders(raw)
    log.info("%s", resolved)
    return resolved


@pytest.fixture(scope="module")
def orch(orch_config):
    """Build the Orchestrator and run its prepare pipeline. cleanup() rolls
    back PREPARE_PIPELINE (e.g. stops the in-namespace sshd) and tears down
    the runtime (e.g. removes the docker container)."""
    log.info("Creating orchestrator from cluster + testsuite config files")
    o = create_orchestrator(orch_config, log)
    o.setup()
    yield o
    o.cleanup()
