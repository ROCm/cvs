'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# ./install without -o exits unless uid is 0. These parents are not writable
# by the SSH user, so a rootless install cannot target them. Paths are remote
# POSIX paths, so this does not use os.path.
_ROOT_INSTALL_PARENTS = ('/opt/amd', '/opt', '/usr', '/')


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


def path_requires_root(path):
    return rootless_output_dir(path) in _ROOT_INSTALL_PARENTS


def install_command(sudo_prefix, orchestrator_type, install_dir, path):
    flags = '--rocm-tar'
    if rootless_install(orchestrator_type, sudo_prefix) or rootless_output_dir(path) != '/opt/amd':
        flags = f'--rocm-tar -o {rootless_output_dir(path)}'
    return f"{sudo_prefix}bash -c 'cd {install_dir} && ./install {flags}'"


def agfhc_command(sudo_prefix, path, args):
    """AGFHC CLI. A non-/opt prefix needs the companion-tool env the installer prints."""
    tool = f'{path.rstrip("/")}/agfhc'
    if rootless_output_dir(path) == '/opt/amd':
        return f'{sudo_prefix}{tool} {args}'
    root = rootless_output_dir(path)
    agfhc_dir = path.rstrip('/')
    script = (
        f'root={root}; '
        'if [ -d "$root/firexs2/usr/lib" ]; then fx="$root/firexs2"; '
        'else fx="$root/firexs2-external"; fi; '
        'export FIREXS_PATH="$fx"; '
        'export FIREXS2_APP_PATH="$fx/usr/lib/firexs2"; '
        'export LD_LIBRARY_PATH="$fx/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
        f'export TXB_PATH="$root/transferbench"; '
        f'export MINIHPL_PATH="$root/minihpl"; '
        f'export MEMROCKER_PATH="$root/memrocker"; '
        f'export AFIDSAG_PATH="{agfhc_dir}/sag"; '
        f'exec {tool} {args}'
    )
    return f"{sudo_prefix}bash -c '{script}'"
