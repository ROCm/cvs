'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os
import warnings

from cvs.lib.utils_lib import fail_test


def looks_like_docker_image_reference(ref):
    """
    True if ``ref`` resembles a registry/repo:tag **image**, not a container ID.

    Container IDs are hex (no ``/``). Image references almost always contain ``/``.
    """
    if not ref or not isinstance(ref, str):
        return False
    s = ref.strip()
    return '/' in s


def resolve_transferbench_container_id(config_dict, cluster_dict=None):
    """
    Resolve default container **ID** (same value applied to all hosts unless overridden per node).

    Precedence: ``transferbench.container_id``, cluster ``transferbench_container_id``,
    env ``CVS_TRANSFERBENCH_CONTAINER_ID``.
    """
    cd = config_dict or {}
    cl = cluster_dict or {}
    candidates = [
        cd.get('container_id'),
        cl.get('transferbench_container_id'),
        os.environ.get('CVS_TRANSFERBENCH_CONTAINER_ID'),
    ]
    for c in candidates:
        if c is None:
            continue
        s = str(c).strip()
        if s:
            return s
    return ''


def build_transferbench_container_id_map(host_list, config_dict, cluster_dict=None):
    """
    Build ``{ssh_host: container_id}``. Each machine runs its own Docker daemon — IDs almost
    always **differ per node**. Use ``transferbench.container_id_by_node`` and/or
    ``node_dict[host].transferbench_container_id`` to override the default from
    ``resolve_transferbench_container_id``.
    """
    default = resolve_transferbench_container_id(config_dict, cluster_dict)
    cd = config_dict or {}
    raw = cd.get('container_id_by_node')
    by_node = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is None:
                continue
            ks, vs = str(k).strip(), str(v).strip()
            if vs:
                by_node[ks] = vs
    nd = (cluster_dict or {}).get('node_dict') or {}
    out = {}
    for host in host_list:
        hid = ''
        hs = str(host)
        if host in by_node:
            hid = by_node[host]
        elif hs in by_node:
            hid = by_node[hs]
        elif isinstance(nd.get(host), dict):
            hid = (nd[host].get('transferbench_container_id') or '').strip()
        out[host] = hid or default
    return out


def resolve_rvs_container_id(config_dict, cluster_dict=None):
    """
    Default container **ID** for RVS (same semantics as TransferBench, RVS-specific cluster/env keys).

    Precedence: ``rvs.container_id``, cluster ``rvs_container_id``, env ``CVS_RVS_CONTAINER_ID``.
    """
    cd = config_dict or {}
    cl = cluster_dict or {}
    candidates = [
        cd.get('container_id'),
        cl.get('rvs_container_id'),
        os.environ.get('CVS_RVS_CONTAINER_ID'),
    ]
    for c in candidates:
        if c is None:
            continue
        s = str(c).strip()
        if s:
            return s
    return ''


def build_rvs_container_id_map(host_list, config_dict, cluster_dict=None):
    """
    Build ``{ssh_host: container_id}`` for RVS. Use ``rvs.container_id_by_node`` and/or
    ``node_dict[host].rvs_container_id``; default from ``resolve_rvs_container_id``.
    """
    default = resolve_rvs_container_id(config_dict, cluster_dict)
    cd = config_dict or {}
    raw = cd.get('container_id_by_node')
    by_node = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is None:
                continue
            ks, vs = str(k).strip(), str(v).strip()
            if vs:
                by_node[ks] = vs
    nd = (cluster_dict or {}).get('node_dict') or {}
    out = {}
    for host in host_list:
        hid = ''
        hs = str(host)
        if host in by_node:
            hid = by_node[host]
        elif hs in by_node:
            hid = by_node[hs]
        elif isinstance(nd.get(host), dict):
            hid = (nd[host].get('rvs_container_id') or '').strip()
        out[host] = hid or default
    return out


def docker_spec_for_ssh_hosts(spec, host_list):
    """Subset ``spec`` to ``host_list`` keys when ``spec`` is a per-host map."""
    if isinstance(spec, dict):
        return {h: spec[h] for h in host_list}
    return spec


def resolve_transferbench_executable(config_dict):
    """
    Absolute path to the TransferBench **binary** used by tests.

    Precedence:

    - ``transferbench.executable``: if set, use as-is when absolute; otherwise relative to
      ``transferbench.path`` (e.g. ``build/TransferBench`` after a CMake build).
    - Else ``{path}/TransferBench`` (typical Makefile install in the repo root).

    CMake builds usually emit ``build/TransferBench``. If ``ls`` shows ``TransferBench`` at repo root,
    run ``ls -la`` / ``file TransferBench`` — it may be a directory or a binary built for another
    environment. Bash reports ``No such file or directory`` when the ELF dynamic linker is missing
    (rebuild inside the runtime container; check with ``file`` and ``ldd``).
    """
    cd = config_dict or {}
    base = (cd.get('path') or '').strip().rstrip('/')
    if not base:
        fail_test('transferbench.path is empty or missing')
    ex = (cd.get('executable') or '').strip()
    if not ex:
        return '%s/TransferBench' % base
    if ex.startswith('/'):
        return ex
    return '%s/%s' % (base, ex.lstrip('/'))


def coerce_bool(value, default=False):
    """Interpret JSON booleans and common string forms ('True', 'false', '1')."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


def remote_exec(phdl, inner_cmd, use_docker, container_id=None, timeout=None, print_console=True):
    """
    Run ``inner_cmd`` on each host in ``phdl``'s fan-out.

    When ``use_docker`` is true, ``phdl`` must use ``docker_container_id`` set to the same string,
    or the same per-host map, as configured for TransferBench.
    """
    kwargs = {'print_console': print_console}
    if timeout is not None:
        kwargs['timeout'] = timeout
    if use_docker:
        dc = getattr(phdl, 'docker_container_id', None)
        if isinstance(dc, dict) or isinstance(container_id, dict):
            pass
        elif dc is not None and container_id is not None and dc != container_id:
            fail_test(
                'remote_exec(use_docker=True) requires Pssh(..., docker_container_id=%r); got %r.'
                % (container_id, dc)
            )
    return phdl.exec(inner_cmd, **kwargs)


def assert_container_running(phdl, container_id):
    """Fail fast if the container is not running on every node (``docker inspect`` on the host)."""
    hosts = phdl.reachable_hosts
    if isinstance(container_id, dict):
        cmds = [
            'docker inspect -f "{{.State.Running}}" %s 2>/dev/null || echo not_running' % container_id[h]
            for h in hosts
        ]
        use_host_exec = getattr(phdl, 'docker_container_id', None) and hasattr(phdl, 'exec_host')
        if use_host_exec and hasattr(phdl, 'exec_host_cmd_list'):
            out_dict = phdl.exec_host_cmd_list(cmds)
        else:
            out_dict = phdl.exec_cmd_list(cmds)
    else:
        cmd = 'docker inspect -f "{{.State.Running}}" %s 2>/dev/null || echo not_running' % container_id
        use_host_exec = getattr(phdl, 'docker_container_id', None) and hasattr(phdl, 'exec_host')
        runner = phdl.exec_host if use_host_exec else phdl.exec
        out_dict = runner(cmd)

    for node, out in out_dict.items():
        if 'true' not in out.lower():
            hint = ''
            if 'no such container' in out.lower():
                hint = (
                    ' On multi-node clusters each host has its own container ID — set '
                    'transferbench.container_id_by_node (map host -> id from docker ps on that host).'
                )
            fail_test(f'Docker container not running on node {node}: {out}{hint}')


def docker_opts_from_config(config_dict, cluster_dict=None, host_list=None):
    """
    Returns ``(use_docker, container_spec)``. ``container_spec`` is a **dict** ``host -> id``
    when ``host_list`` is provided; otherwise a **string** default ID.

    See ``build_transferbench_container_id_map`` for per-node IDs.
    """
    cfg_ud = coerce_bool(config_dict.get('use_docker'), False)
    cluster_ud = coerce_bool((cluster_dict or {}).get('transferbench_use_docker'), False)
    use_docker = cfg_ud or cluster_ud

    env_mode = (os.environ.get('CVS_TRANSFERBENCH_USE_DOCKER') or '').strip().lower()
    if env_mode in ('1', 'true', 'yes', 'y'):
        use_docker = True
    elif env_mode in ('0', 'false', 'no', 'n'):
        use_docker = False

    if use_docker and host_list:
        container_spec = build_transferbench_container_id_map(host_list, config_dict, cluster_dict)
        missing = [h for h, v in container_spec.items() if not str(v).strip()]
        if missing:
            fail_test(
                'TransferBench Docker mode needs a container ID for every node. Missing for host(s) %r. '
                'Set transferbench.container_id (default for all), container_id_by_node, and/or '
                'node_dict[host].transferbench_container_id.'
                % (missing,)
            )
        for cid in container_spec.values():
            if looks_like_docker_image_reference(cid):
                fail_test(
                    'container_id value looks like a Docker **image** (%r), not a container ID.' % (cid,)
                )
        if len(host_list) > 1 and len(set(container_spec.values())) == 1:
            warnings.warn(
                UserWarning(
                    'All nodes use the same container_id %r. Each Docker daemon has its own IDs — '
                    'if docker exec fails with No such container on some hosts, set '
                    'transferbench.container_id_by_node with each host\'s ID from docker ps on that host.'
                    % (next(iter(container_spec.values())),)
                ),
                stacklevel=2,
            )
        return use_docker, container_spec

    container_spec = resolve_transferbench_container_id(config_dict, cluster_dict)

    if use_docker and not container_spec:
        fail_test(
            'TransferBench Docker mode is enabled but container_id is empty. '
            'Set transferbench.container_id, cluster transferbench_container_id, or CVS_TRANSFERBENCH_CONTAINER_ID.'
        )
    if use_docker and looks_like_docker_image_reference(container_spec):
        fail_test(
            'transferbench.container_id looks like a Docker **image** (%r), not a container ID.'
            % (container_spec,)
        )
    return use_docker, container_spec


def docker_opts_from_rvs_config(config_dict, cluster_dict=None, host_list=None):
    """
    Returns ``(use_docker, container_spec)`` for RVS install/tests (dict per host when
    ``host_list`` is set). Same layout as ``docker_opts_from_config`` but uses
    ``rvs_use_docker`` / ``CVS_RVS_USE_DOCKER`` and ``build_rvs_container_id_map``.
    """
    cfg_ud = coerce_bool(config_dict.get('use_docker'), False)
    cluster_ud = coerce_bool((cluster_dict or {}).get('rvs_use_docker'), False)
    use_docker = cfg_ud or cluster_ud

    env_mode = (os.environ.get('CVS_RVS_USE_DOCKER') or '').strip().lower()
    if env_mode in ('1', 'true', 'yes', 'y'):
        use_docker = True
    elif env_mode in ('0', 'false', 'no', 'n'):
        use_docker = False

    if use_docker and host_list:
        container_spec = build_rvs_container_id_map(host_list, config_dict, cluster_dict)
        missing = [h for h, v in container_spec.items() if not str(v).strip()]
        if missing:
            fail_test(
                'RVS Docker mode needs a container ID for every node. Missing for host(s) %r. '
                'Set rvs.container_id, rvs.container_id_by_node, and/or node_dict[host].rvs_container_id.'
                % (missing,)
            )
        for cid in container_spec.values():
            if looks_like_docker_image_reference(cid):
                fail_test(
                    'container_id value looks like a Docker **image** (%r), not a container ID.' % (cid,)
                )
        if len(host_list) > 1 and len(set(container_spec.values())) == 1:
            warnings.warn(
                UserWarning(
                    'All nodes use the same RVS container_id %r. Each Docker daemon has its own IDs — '
                    'if docker exec fails on some hosts, set rvs.container_id_by_node with each host\'s ID from '
                    'docker ps on that host.'
                    % (next(iter(container_spec.values())),)
                ),
                stacklevel=2,
            )
        return use_docker, container_spec

    container_spec = resolve_rvs_container_id(config_dict, cluster_dict)

    if use_docker and not container_spec:
        fail_test(
            'RVS Docker mode is enabled but container_id is empty. '
            'Set rvs.container_id, cluster rvs_container_id, or CVS_RVS_CONTAINER_ID.'
        )
    if use_docker and looks_like_docker_image_reference(container_spec):
        fail_test(
            'rvs.container_id looks like a Docker **image** (%r), not a container ID.' % (container_spec,)
        )
    return use_docker, container_spec


def log_transferbench_docker_resolution(
    log, label, use_docker, container_spec, config_dict, cluster_dict, *, info_always=False
):
    """Log whether Docker mode is active."""
    if use_docker:
        if isinstance(container_spec, dict):
            target = 'container_id_by_host=%r' % (container_spec,)
        else:
            target = 'container_id=%r' % (container_spec,)
    else:
        target = 'docker_exec=off (host install via SSH)'
    msg = (
        '%s: TransferBench Docker active=%s %s | health use_docker=%s cluster transferbench_use_docker=%s',
        label,
        use_docker,
        target,
        config_dict.get('use_docker'),
        (cluster_dict or {}).get('transferbench_use_docker'),
    )
    if use_docker or info_always:
        log.info(*msg)
    else:
        log.debug(*msg)


def log_rvs_docker_resolution(
    log, label, use_docker, container_spec, config_dict, cluster_dict, *, info_always=False
):
    """Log whether RVS Docker mode is active."""
    if use_docker:
        if isinstance(container_spec, dict):
            target = 'container_id_by_host=%r' % (container_spec,)
        else:
            target = 'container_id=%r' % (container_spec,)
    else:
        target = 'docker_exec=off (host install via SSH)'
    msg = (
        '%s: RVS Docker active=%s %s | health use_docker=%s cluster rvs_use_docker=%s',
        label,
        use_docker,
        target,
        config_dict.get('use_docker'),
        (cluster_dict or {}).get('rvs_use_docker'),
    )
    if use_docker or info_always:
        log.info(*msg)
    else:
        log.debug(*msg)
