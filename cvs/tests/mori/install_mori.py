'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import shlex
import uuid

import pytest

from cvs.core.scheduler import is_managed_compute
from cvs.lib import globals, mori_lib
from cvs.lib.utils_lib import fail_test, update_test_result
from cvs.tests.mori.conftest import launch_mori_container, teardown_mori_container

log = globals.log

CHECK_TIMEOUT = 300
APT_TIMEOUT = 900
CHECKOUT_TIMEOUT = 900
BUILD_TIMEOUT = 3600

# Defaults for the keys MORI configs from earlier releases don't have.
DEFAULT_GIT_URL = 'https://github.com/ROCm/mori'
DEFAULT_GIT_REF = 'main'


def _install_system_packages(mori_dict):
    """``install_system_packages`` as a bool: JSON true/false or the strings "True"/"False"; true when absent."""
    return str(mori_dict.get('install_system_packages', True)).lower() == 'true'


NOT_APT_HINT = (
    'the ROCm dev stack and torch must come from the image or host; CVS never installs them, pip or the NIC libraries'
)


def _apt_prefix(orch, install_system_packages):
    """Return ``(prefix, hint)``: the sudo prefix for apt, or None when apt may not run, and what to tell
    the user about missing items that apt can install."""
    if not install_system_packages:
        return None, 'install them, or set install_system_packages to true'
    if is_managed_compute():
        return None, 'a Slurm or SPUR job step never runs apt; install them in the node image or before allocating'
    if orch.orchestrator_type == 'container':
        return '', 'apt did not install them; its output is in the CVS log'
    sudo = orch.sudo_prefix()
    if not sudo:
        return None, 'apt needs passwordless sudo on the hosts; install them yourself'
    return sudo, 'apt did not install them; its output is in the CVS log'


def _check_deps(orch, install_system_packages):
    """Fail for every node still missing a build dependency, after apt-installing what each node misses where allowed."""
    missing = mori_lib.parse_missing(orch.exec(mori_lib.deps_check_cmd(), timeout=CHECK_TIMEOUT))
    sudo, hint = _apt_prefix(orch, install_system_packages)
    if sudo is not None:
        nodes_by_packages = {}
        for node, items in missing.items():
            packages = mori_lib.apt_packages(items or [])
            if packages:
                nodes_by_packages.setdefault(tuple(packages), []).append(node)
        for packages, nodes in nodes_by_packages.items():
            log.info('apt-installing %s on %s', ' '.join(packages), ', '.join(nodes))
            orch.exec(
                f'{sudo}apt-get update -q && '
                f'{sudo}env DEBIAN_FRONTEND=noninteractive apt-get install -y -q {" ".join(packages)}',
                hosts=nodes,
                timeout=APT_TIMEOUT,
            )
        if nodes_by_packages:
            missing = mori_lib.parse_missing(orch.exec(mori_lib.deps_check_cmd(), timeout=CHECK_TIMEOUT))
    for node, items in missing.items():
        if items is None:
            fail_test(f'could not check the MoRI build dependencies on node {node}')
        elif items:
            apt_can_install = all(mori_lib.apt_packages([item]) for item in items)
            fail_test(
                f'MoRI build dependencies missing on node {node}: {", ".join(items)}; '
                f'{hint if apt_can_install else NOT_APT_HINT}'
            )


def _run(orch, step, cmd, timeout, head_only):
    """Run ``cmd`` on the head node or on every node, and fail the test for any node where it exits non-zero."""
    run = orch.exec_on_head if head_only else orch.exec
    for node, res in run(cmd, timeout=timeout, detailed=True).items():
        if res.get('exit_code') != 0:
            fail_test(f'{step} failed on node {node} with exit code {res.get("exit_code")}')
    update_test_result()


def test_launch_mori_container(orch, lifecycle):
    if orch.orchestrator_type == 'container' and orch.container_config.get('lifetime', 'per_run') == 'per_run':
        lifecycle.failed = True
        pytest.fail(
            'container.lifetime is per_run, so the container and everything installed in it are removed '
            'when install_mori ends; set it to persistent or no_launch'
        )
    launch_mori_container(orch, lifecycle)


def test_install_mori(orch, mori_dict, lifecycle):
    if lifecycle.failed:
        pytest.skip('mori container launch failed')
    globals.error_list = []
    mori_dir = mori_dict['mori_dir']
    md = shlex.quote(mori_dir)

    for node, out in orch.exec(mori_lib.env_mori_check_cmd(mori_dir), timeout=CHECK_TIMEOUT).items():
        if mori_lib.MORI_IN_DIR in (out or ''):
            fail_test(
                f'the mori package python3 finds on node {node} is under mori_dir {mori_dir}; '
                'building there would replace it, so give mori_dir a directory of its own'
            )
    update_test_result()

    _check_deps(orch, _install_system_packages(mori_dict))
    update_test_result()

    marker = shlex.quote(f'{mori_dir}/.cvs_storage_probe')
    token = uuid.uuid4().hex
    orch.exec_on_head(f'mkdir -p {md} && echo {token} > {marker}', timeout=CHECK_TIMEOUT)
    try:
        out_dict = orch.exec(f'cat {marker} 2>/dev/null', timeout=CHECK_TIMEOUT)
        try:
            layout = mori_lib.storage_layout(
                {node: token in (out or '') for node, out in out_dict.items()}, orch.head_node
            )
        except ValueError as e:
            fail_test(str(e))
            update_test_result()
        head_only = layout == 'shared'
        log.info(
            'mori_dir %s is %s storage: building on %s',
            mori_dir,
            layout,
            'the head node' if head_only else 'every node',
        )

        git_ref = mori_dict.get('git_ref', DEFAULT_GIT_REF)
        checkout = mori_lib.checkout_cmd(mori_dir, mori_dict.get('git_url', DEFAULT_GIT_URL), git_ref)
        _run(orch, f'checkout of {git_ref}', checkout, CHECKOUT_TIMEOUT, head_only)
        build = mori_lib.build_cmd(mori_dir, mori_dict['log_dir'])
        _run(orch, f'MoRI build (see {mori_dict["log_dir"]}/install_mori_<host>.log)', build, BUILD_TIMEOUT, head_only)
        if orch.orchestrator_type != 'container':
            pip = f'python3 -m pip install --target {md}/site prettytable'
            _run(orch, 'prettytable install into mori_dir/site', pip, CHECK_TIMEOUT, head_only)

        out_dict = orch.exec(mori_lib.verify_build_cmd(mori_dir), timeout=CHECK_TIMEOUT)
        for node, items in mori_lib.parse_missing(out_dict).items():
            if items is None:
                fail_test(f'could not check the MoRI build on node {node}')
            elif items:
                fail_test(f'MoRI build on node {node} is missing {", ".join(items)}')
        log.info('MoRI commit %s is built in %s', mori_lib.built_commit(orch, mori_dir), mori_dir)
    finally:
        orch.exec_on_head(f'rm -f {marker}', timeout=CHECK_TIMEOUT)
    update_test_result()


def test_teardown(orch, lifecycle):
    teardown_mori_container(orch, lifecycle)
