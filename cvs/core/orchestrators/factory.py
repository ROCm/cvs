'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import os

from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.core.orchestrators.container import ContainerOrchestrator


VALID_CONTAINER_LIFETIMES = ("external", "per_run", "persistent")

# Packaged default provisioning script, run inside each freshly-launched
# container when container.setup_script is not set. Installs openssh-server so
# the in-container sshd can start on port 2224. Resolved __file__-relative so it
# works in both editable and wheel installs.
DEFAULT_CONTAINER_SETUP_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scripts", "default_container_setup.sh"
)


def _resolve_container_lifetime(container):
    """Normalize a container config block to a single resolved ``lifetime`` key.

    The legacy two-axis schema (``enabled`` + ``launch``) is removed in favor of
    one tri-valued ``container.lifetime``. Mutates and returns the passed dict.

    Resolution rules (first match wins):
      - ``enabled`` present (any value) -> ``ValueError`` (removed field).
      - ``launch`` present (any value)  -> ``ValueError`` (removed field). Both
        removed fields fail loudly rather than being silently mapped, so a stale
        flag can never quietly override an explicit ``lifetime``.
      - ``lifetime`` present            -> validated, kept as-is.
      - none of the above               -> default ``per_run``.

    An empty/absent container block is returned untouched (baremetal path).
    """
    if not container:
        return container

    if 'enabled' in container:
        raise ValueError(
            "container.enabled is removed; delete the field and set "
            "container.lifetime to one of 'external', 'per_run', 'persistent'"
        )

    if 'launch' in container:
        raise ValueError(
            "container.launch is removed; delete the field and set "
            "container.lifetime ('launch: true' -> 'per_run', "
            "'launch: false' -> 'external')"
        )

    if 'lifetime' in container:
        lifetime = container['lifetime']
        if lifetime not in VALID_CONTAINER_LIFETIMES:
            raise ValueError(
                f"container.lifetime must be one of {VALID_CONTAINER_LIFETIMES}, "
                f"got {lifetime!r}"
            )
        return container

    container['lifetime'] = 'per_run'
    return container


def _resolve_container_setup_script(container):
    """Resolve ``container.setup_script`` to a concrete, existing file path.

    The script is run inside each freshly-launched container (see
    ``ContainerOrchestrator._provision_container``) to install packages on top
    of the base image. Resolution rules:

      - empty/absent container block (baremetal path) -> returned untouched.
      - ``setup_script`` set                          -> validated to exist on
        the control host; ``ValueError`` (fail fast at config load) if missing.
      - ``setup_script`` absent                       -> defaults to the
        packaged ``default_container_setup.sh`` (installs openssh-server).

    Relative paths are resolved against the current working directory of the
    process running ``cvs``. Mutates and returns the passed dict.
    """
    if not container:
        return container

    setup_script = container.get('setup_script')
    if setup_script:
        resolved = os.path.abspath(os.path.expanduser(setup_script))
        if not os.path.isfile(resolved):
            raise ValueError(
                f"container.setup_script not found: {setup_script!r} "
                f"(resolved to {resolved!r})"
            )
        container['setup_script'] = resolved
        return container

    # No user-supplied script: fall back to the packaged default. Validate it
    # exists here (same fail-fast as a user path) so a broken/incomplete install
    # surfaces at config load rather than as an OSError mid-run inside
    # _provision_container.
    if not os.path.isfile(DEFAULT_CONTAINER_SETUP_SCRIPT):
        raise ValueError(
            f"packaged default container setup script is missing: "
            f"{DEFAULT_CONTAINER_SETUP_SCRIPT!r} (broken CVS install?)"
        )
    container['setup_script'] = DEFAULT_CONTAINER_SETUP_SCRIPT
    return container


class OrchestratorConfig:
    """
    Configuration for orchestrator creation.

    Contains only the configuration keys required by orchestrators.

    Expected keys in cluster_config.json or <testsuite>_config.json:
    - orchestrator: Type of orchestrator ('baremetal', 'container', etc.) [optional, default: 'baremetal']
    - node_dict: List/dict of cluster nodes [required, no default]
    - username: SSH username for node access [required, no default]
    - priv_key_file: Path to SSH private key file [required, no default]
    - password: Optional SSH password (if not using key-based auth) [default: None]
    - head_node_dict: Optional head node configuration with 'mgmt_ip' key [default: {}]
    - container: Optional container configuration for docker orchestrator [default: {}]
          Example container configuration:
          ```json
          "container": {
            "lifetime": "per_run",
            "runtime": {
              "name": "docker",
              "args": {
                "volumes": ["/host/path:/container/path:ro"],
                "devices": ["/dev/kfd", "/dev/dri"],
                "env": {"GPUS": "8", "MULTINODE": "true"},
                "cap_add": ["SYS_PTRACE"],
                "security_opt": ["apparmor=unconfined", "seccomp=unconfined"],
                "group_add": ["video"],
                "network": "host",
                "ipc": "host",
                "ulimit": ["memlock=-1"],
                "privileged": true
              }
            },
            "image": "rocm/cvs:latest",
            "name": "myuser_rocm_cvs_latest",
            "setup_script": "/path/to/setup.sh"
          }
          ```
          lifetime: Container lifecycle policy [default: 'per_run']
            - 'external'   : containers managed outside CVS. Setup verifies they
                             are running; teardown is a no-op.
            - 'per_run'    : start at setup, remove at teardown (the default).
            - 'persistent' : start if absent / attach if present; never torn down
                             by the run. Pin container.name explicitly under this
                             mode (the default <user>_<image> name shifts on tag bumps).
          setup_script: Optional path to a shell script run inside each freshly
            launched container (per_run, and persistent when cold-started) before
            sshd setup, to install packages on top of the base image. Omit to use
            the packaged default that installs openssh-server. A non-existent path
            fails at config load.
    """

    def __init__(self, **kwargs):
        """
        Initialize orchestrator configuration.

        Args:
            **kwargs: Required orchestrator configuration keys

        Raises:
            ValueError: If any required key (orchestrator, node_dict, username,
                priv_key_file) is missing.
        """
        required = ('orchestrator', 'node_dict', 'username', 'priv_key_file')
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"OrchestratorConfig missing required keys: {missing}")

        self.orchestrator = kwargs['orchestrator']
        self.node_dict = kwargs['node_dict']
        self.username = kwargs['username']
        self.priv_key_file = kwargs['priv_key_file']
        self.password = kwargs.get('password')
        self.head_node_dict = kwargs.get('head_node_dict', {})
        # Normalize the container block. This is the single chokepoint:
        # from_configs constructs via cls(**required_config), so both file-driven
        # and direct programmatic construction hit the same normalization (and the
        # same enabled-removed / launch-deprecated errors, and the same
        # setup_script validation / default injection).
        container = _resolve_container_lifetime(kwargs.get('container', {}))
        self.container = _resolve_container_setup_script(container)

    def get(self, key, default=None):
        """Get configuration value with default."""
        return getattr(self, key, default)

    @classmethod
    def from_configs(cls, cluster_config, testsuite_config=None):
        """
        Create config from multiple configuration sources.

        Merges cluster_config.json and <testsuite>_config.json configurations, with testsuite_config
        taking precedence for overlapping keys. Before merging, the cluster_config is run through
        cvs.lib.utils_lib.resolve_cluster_config_placeholders to substitute {user-id} and enforce
        the <changeme> guard on cluster-portion fields (username, priv_key_file, container.*) --
        unresolved <changeme> tokens in the cluster portion trigger sys.exit(1) at this boundary.
        testsuite_config is then merged in verbatim; testsuite subsections (transferbench, rvs,
        agfhc, ...) are resolved later by per-test fixtures (resolve_test_config_placeholders),
        scoped to the subsection the test actually consumes.

        Args:
            cluster_config: Cluster configuration (dict or path to cluster_config.json)
                           Required keys: orchestrator, node_dict, username, priv_key_file
                           Optional keys: container,
                           head_node_dict, password (defaults provided for missing optional keys)
                           Container structure: {lifetime: 'external'|'per_run'|'persistent', runtime: {name: str, args: dict}, image: str, name: str, setup_script: str, ...}
            testsuite_config: Test suite specific configuration (dict or path to <testsuite>_config.json)
                            Can override any keys from cluster_config

        Returns:
            OrchestratorConfig instance with extracted orchestrator keys
        """
        # Handle file paths or dicts
        if isinstance(cluster_config, str):
            import json  # Lazy import - only loaded when reading config files

            with open(cluster_config) as f:
                cluster_config = json.load(f)

        if isinstance(testsuite_config, str):
            import json  # Lazy import - only loaded when reading config files

            with open(testsuite_config) as f:
                testsuite_config = json.load(f)

        # Resolve {user-id} on the CLUSTER PORTION ONLY, before merging. from_configs only
        # consumes cluster-portion keys (orchestrator, node_dict, username, priv_key_file,
        # password, head_node_dict, container). Testsuite subsections (transferbench, rvs,
        # agfhc, ...) belong to per-test fixtures which do their own subsection-scoped
        # resolution via resolve_test_config_placeholders. Resolving the merged dict here
        # would walk testsuite subsections that use <changeme> as a legitimate auto-detect
        # sentinel (e.g. transferbench.rocm_path), causing cross-subsection guard trips on
        # tests that don't even use that subsection.
        from cvs.lib.utils_lib import (
            resolve_cluster_config_placeholders,
        )  # Lazy: avoids core->lib import at module load

        cluster_config = resolve_cluster_config_placeholders(cluster_config)

        merged_config = cluster_config.copy()
        if testsuite_config:
            merged_config.update(testsuite_config)

        # Extract only required keys for orchestrators
        required_config = {
            'orchestrator': merged_config.get('orchestrator', 'baremetal'),
            'node_dict': merged_config.get('node_dict'),
            'username': merged_config.get('username'),
            'priv_key_file': merged_config.get('priv_key_file'),
            'password': merged_config.get('password'),
            'head_node_dict': merged_config.get('head_node_dict', {}),
            'container': merged_config.get('container', {}),
        }

        # Validate required keys
        if required_config['node_dict'] is None:
            raise ValueError("node_dict must be provided in configuration")
        if required_config['username'] is None:
            raise ValueError("username must be provided in configuration")
        if required_config['priv_key_file'] is None:
            raise ValueError("priv_key_file must be provided in configuration")

        return cls(**required_config)


class OrchestratorFactory:
    """
    Factory for creating orchestrator instances based on configuration.

    Provides pluggable architecture for adding new backends without
    modifying existing test code.
    """

    @staticmethod
    def create_orchestrator(log, config, stop_on_errors=False):
        """
        Create orchestrator instance from configuration.

        Args:
            log: Logger instance
            config: OrchestratorConfig instance
            stop_on_errors: Whether to stop execution on first error

        Returns:
            Orchestrator instance

        Raises:
            ValueError: If orchestrator type is unsupported
            TypeError: If config is not OrchestratorConfig instance
        """
        if not isinstance(config, OrchestratorConfig):
            raise TypeError("config must be OrchestratorConfig instance")

        orchestrator_type = config.orchestrator.lower()

        if orchestrator_type == 'baremetal':
            return BaremetalOrchestrator(log, config, stop_on_errors)
        elif orchestrator_type == 'container':
            return ContainerOrchestrator(log, config, stop_on_errors)
        else:
            raise ValueError(f"Unsupported orchestrator type: {orchestrator_type}")

    @staticmethod
    def get_supported_backends():
        """
        Get list of supported orchestrator backends.

        Returns:
            List of backend names
        """
        return ['baremetal', 'container']
