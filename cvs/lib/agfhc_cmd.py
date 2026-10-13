'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import shlex

# ./install without -o exits unless uid is 0. Paths are remote POSIX paths,
# so this does not use os.path (Windows normpath rewrites /opt to \opt).
_COMPANION_KEYS = ('TXB_PATH', 'MINIHPL_PATH', 'MEMROCKER_PATH', 'FIREXS_PATH', 'AFIDSAG_PATH')


def rootless_install(orchestrator_type, sudo_prefix):
    """True when ./install must use -o.

    Container payloads already run as root inside the container, so an empty
    sudo prefix there is not this case.
    """
    return orchestrator_type != 'container' and not sudo_prefix


def rootless_output_dir(path):
    """Parent of path. ./install -o writes the agfhc tool directory there."""
    trimmed = path.rstrip('/')
    parent, _sep, _name = trimmed.rpartition('/')
    return parent or '/'


def uses_output_dir(orchestrator_type, sudo_prefix, path):
    """True when ./install must pass -o and must not run as root.

    Root-owned files under an NFS home fail with root squash, so -o drops sudo
    even when passwordless sudo exists.
    """
    return rootless_install(orchestrator_type, sudo_prefix) or rootless_output_dir(path) != '/opt/amd'


def derived_agfhc_env(path):
    """Companion locations next to a -o prefix. Used when agfhc.env omits a key."""
    root = rootless_output_dir(path)
    agfhc_dir = path.rstrip('/')
    return {
        'TXB_PATH': f'{root}/transferbench',
        'MINIHPL_PATH': f'{root}/minihpl',
        'MEMROCKER_PATH': f'{root}/memrocker',
        'FIREXS_PATH': f'{root}/firexs2',
        'AFIDSAG_PATH': f'{agfhc_dir}/sag',
    }


def resolve_agfhc_env(path, env):
    """agfhc.env values, with a derived path for any missing or empty key."""
    derived = derived_agfhc_env(path)
    supplied = env or {}
    resolved = {}
    for key in _COMPANION_KEYS:
        value = supplied.get(key)
        resolved[key] = value if value else derived[key]
    return resolved


def install_command(sudo_prefix, orchestrator_type, install_dir, path):
    flags = '--rocm-tar'
    prefix = sudo_prefix
    if uses_output_dir(orchestrator_type, sudo_prefix, path):
        flags = '--rocm-tar -o ' + shlex.quote(rootless_output_dir(path))
        prefix = ''
    inner = f'cd {shlex.quote(install_dir)} && ./install {flags}'
    return f'{prefix}bash -c {shlex.quote(inner)}'


def agfhc_command(sudo_prefix, path, args, env=None):
    """AGFHC CLI. Companion paths come from agfhc.env, including a /opt/amd install."""
    tool = f'{path.rstrip("/")}/agfhc'
    resolved = resolve_agfhc_env(path, env)
    firexs = resolved['FIREXS_PATH'].rstrip('/')
    lines = [f'export {key}={shlex.quote(resolved[key])}' for key in _COMPANION_KEYS]
    lines.append(f'export FIREXS2_APP_PATH={shlex.quote(firexs + "/usr/lib/firexs2")}')
    lines.append('export LD_LIBRARY_PATH=' + shlex.quote(firexs + '/usr/lib') + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}')
    lines.append(f'exec {shlex.quote(tool)} {args}')
    return f'{sudo_prefix}bash -c {shlex.quote("; ".join(lines))}'


def writable_dir_commands(directory, sudo_prefix=''):
    """mkdir -p, then a probe whose stdout is ``writable`` or ``not-writable``."""
    quoted = shlex.quote(directory)
    return (
        f'{sudo_prefix}mkdir -p {quoted}',
        f'{sudo_prefix}test -w {quoted} && echo writable || echo not-writable',
    )


def probe_is_writable(text):
    body = str(text)
    return 'not-writable' not in body and 'writable' in body


def companion_probe_command(path, env):
    """One shell that names each companion directory that is not a directory."""
    resolved = resolve_agfhc_env(path, env)
    checks = [
        f'test -d {shlex.quote(directory)} || echo missing {key} {directory}' for key, directory in resolved.items()
    ]
    return 'bash -c ' + shlex.quote('; '.join(checks))
