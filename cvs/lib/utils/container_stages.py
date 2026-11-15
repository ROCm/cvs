'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Container launch, sshd and teardown stages written as pytest tests.

A suite that launches its container as tests imports these into its test module so pytest
collects them, and orders them with its own collection hook. They need module-scoped ``orch``,
``variant_config`` and ``lifecycle`` fixtures; ``lifecycle`` carries ``failed``, ``torn_down``
and ``record(nodeid, label, value, unit="s")``, as ``InferenceLifecycle`` does.
'''

import time

import pytest


def test_launch_container(orch, variant_config, lifecycle, request):
    """Stage 1: launch the container."""
    t = time.monotonic()
    ok = orch.setup_containers()
    launch_s = time.monotonic() - t
    lifecycle.record(request.node.nodeid, "container_launch", launch_s)
    lifecycle.record(request.node.nodeid, "server.container_launch_s", launch_s, "s")
    if not ok:
        lifecycle.failed = True
        name = orch.get_container_name(orch.container_config, orch.container_config["image"])
        pytest.fail(f"setup_containers() returned False for {name}")
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    if not orch.verify_containers_running(name):
        lifecycle.failed = True
        pytest.fail(f"container {name} not running after setup_containers()")


def test_setup_sshd(orch, lifecycle, request):
    if lifecycle.failed:
        pytest.skip("a prior lifecycle stage failed")
    from cvs.core.orchestrators.container import sshd_port_listen_ok, sshd_port_listen_probe_cmd

    t = time.monotonic()
    ok = orch.setup_sshd()
    lifecycle.record(request.node.nodeid, "sshd_setup", time.monotonic() - t)
    if not ok:
        lifecycle.failed = True
        pytest.fail("setup_sshd() returned False")
    # Spur skips sshd on purpose (srun, not port 2224). Probing here fails every multinode run.
    from cvs.core.scheduler import is_managed_compute

    if is_managed_compute():
        return
    if len(orch.hosts) > 1:
        probe = orch.exec(sshd_port_listen_probe_cmd(getattr(orch, "ssh_port", 2224)))
        if not any(sshd_port_listen_ok(v) for v in (probe or {}).values()):
            lifecycle.failed = True
            pytest.fail("sshd not listening on 2224 after setup_sshd()")


def test_teardown(orch, lifecycle, request):
    name = orch.get_container_name(orch.container_config, orch.container_config["image"])
    t = time.monotonic()
    orch.teardown_containers()
    lifecycle.record(request.node.nodeid, "teardown", time.monotonic() - t)
    if orch.verify_containers_running(name):
        pytest.fail(f"container {name} still running after teardown_containers()")
    lifecycle.torn_down = True
