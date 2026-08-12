'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest
import json
import re

from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


# Importing additional cmd line args to script ..
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Return the path to the cluster configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --cluster_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the cluster configuration file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Return the path to the test configuration JSON file passed via pytest CLI.

    Expects:
      - pytest to be invoked with: --config_file <path>

    Args:
      pytestconfig: Built-in pytest config object used to access CLI options.

    Returns:
      str: Filesystem path to the test configuration file.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and expose full cluster configuration for the test module.

    Behavior:
      - Opens the JSON at cluster_file and parses it into a Python dict.
      - Logs the parsed dictionary for visibility and debugging.
      - Returns the entire cluster configuration (node list, credentials, etc.).

    Args:
      cluster_file (str): Path to the cluster configuration JSON.

    Returns:
      dict: Parsed cluster configuration. Expected keys include:
            - 'node_dict': Map of node name -> node metadata
            - 'username': SSH username
            - 'priv_key_file': Path to SSH private key
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    # Resolve path placeholders like {user-id} in cluster config
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and return the RCCL-specific configuration dictionary for the test module.

    Expected rccl_config.json structure:
    {
        "rccl": {
            "installation_params": {
                "nfs_install":            "True" | "False",
                "rccl_lib_install":       "True" | "False",
                "rccl_lib_install_dir":   "/path/to/rccl/install",
                "rccl_tests_install_dir": "/path/to/rccl-tests",
                "rccl_repository":        "https://github.com/ROCm/rocm-systems.git",
                "rccl_git_tag":           "rocm-6.x.y",        (optional)
                "rccl_tests_repository":  "same as rccl_repository if omitted",
                "rccl_tests_git_tag":     "rocm-6.x.y",        (optional)
                "rccl_sparse_path":       "projects/rccl",     (optional, auto for rocm-systems)
                "rccl_tests_sparse_path": "projects/rccl-tests",
                "ompi_install_dir":       "/opt/ompi/build",
                "rocm_path":              "/opt/rocm"          (or "<changeme>"),
                "rccl_tests_use_amdclang": "True" | "False"    (optional, default True)
            }
        }
    }
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)

    rccl_install_cfg = config_dict_t['rccl']['installation_params']
    if not rccl_install_cfg.get("rccl_tests_repository"):
        rccl_install_cfg["rccl_tests_repository"] = rccl_install_cfg["rccl_repository"]
    log.info("%s", rccl_install_cfg)
    return rccl_install_cfg


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for all cluster nodes.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing:
        - node_dict: dict of node_name -> node_details
        - username: SSH username
        - priv_key_file: path to SSH private key

    Returns:
      Pssh: Handle configured for all nodes (for broadcast/parallel operations).

    Notes:
      - Prints the cluster_dict for quick debugging; consider replacing with log.debug.
      - Module-scoped so a single shared handle is used across all tests in the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
      - Assumes Pssh(log, node_list, user=..., pkey=...) is available in scope.
    """
    log.info("%s", cluster_dict)
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    if len(node_list) < 2:
        raise ValueError('At least 2 nodes are required to run this test')
    if len(node_list) % 2 != 0:
        log.info(
            f'Odd number of nodes ({len(node_list)}) detected; popping last node from the cluster to make the count even'
        )
        node_list.pop()
    phdl = Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return phdl


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    """
    Build and return a parallel SSH handle (Pssh) for the head node only.

    Args:
      cluster_dict (dict): Cluster metadata fixture (see phdl docstring).

    Returns:
      Pssh: Handle configured for the first node (head node) in node_dict.

    Notes:
      - Useful when commands should be executed only from a designated head node.
      - Module scope ensures a single connection context for the duration of the module.
      - nhdl_dict is currently unused; it can be removed unless used elsewhere.
    """
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    shdl = Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)
    return shdl


@pytest.fixture(scope="module")
def vpc_node_list(cluster_dict):
    """
    Collect and return a list of VPC IPs for all nodes in the cluster.

    Args:
      cluster_dict (dict): Cluster metadata fixture containing node_dict with vpc_ip per node.

    Returns:
      list[str]: List of VPC IP addresses in the cluster, ordered by node_dict iteration.

    Notes:
      - Iteration order depends on the insertion order of node_dict.
      - Consider validating that each node entry contains a 'vpc_ip' key.
    """
    vpc_node_list = []
    node_list = list(cluster_dict['node_dict'].keys())

    if len(node_list) < 2:
        raise ValueError('At least 2 nodes are required to run this test')

    if len(node_list) % 2 != 0:
        log.info(
            f'Odd number of nodes ({len(node_list)}) detected; popping last node from the cluster to make the count even'
        )
        node_list.pop()
    for node in node_list:
        vpc_node_list.append(cluster_dict['node_dict'][node]['vpc_ip'])
    return vpc_node_list


def detect_rocm_path(phdl, config_rocm_path):
    if config_rocm_path and config_rocm_path != '<changeme>':
        out_dict = phdl.exec(
            f'test -d {config_rocm_path}/lib && ls {config_rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'
        )
        for node, output in out_dict.items():
            if output.strip() and 'libamdhip64.so' in output:
                log.info(f'Using configured ROCm path: {config_rocm_path} (validated)')
                return config_rocm_path
            else:
                log.warning(
                    f'Configured ROCm path {config_rocm_path} does not contain required libraries, will auto-detect'
                )

    log.info('Auto-detecting ROCm path...')

    out_dict = phdl.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for node, output in out_dict.items():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            validate_dict = phdl.exec(
                f'test -d {rocm_path}/lib && ls {rocm_path}/lib/libamdhip64.so* 2>/dev/null | head -1'
            )
            for _, lib_output in validate_dict.items():
                if lib_output.strip() and 'libamdhip64.so' in lib_output:
                    log.info(f'Detected ROCm path (new layout): {rocm_path}')
                    return rocm_path

    out_dict = phdl.exec('test -d /opt/rocm/lib && ls /opt/rocm/lib/libamdhip64.so* 2>/dev/null | head -1')
    for node, output in out_dict.items():
        if output.strip() and 'libamdhip64.so' in output:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    log.warning('Could not detect ROCm path with required libraries, defaulting to /opt/rocm')
    return '/opt/rocm'


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sparse_checkout_path(config_dict, for_tests=False):
    """
    Return the git sparse-checkout path for rocm-systems, or None for a full clone.

    Explicit rccl_sparse_path / rccl_tests_sparse_path in config take precedence.
    When the repository URL contains 'rocm-systems', defaults to projects/rccl or
    projects/rccl-tests.
    """
    if for_tests:
        explicit = config_dict.get("rccl_tests_sparse_path", "").strip()
        repo = config_dict.get("rccl_tests_repository", config_dict["rccl_repository"])
        default = "projects/rccl-tests"
    else:
        explicit = config_dict.get("rccl_sparse_path", "").strip()
        repo = config_dict["rccl_repository"]
        default = "projects/rccl"

    if explicit:
        return explicit
    if "rocm-systems" in repo:
        return default
    return None


def _rccl_install_prefix(config_dict):
    """
    Workspace root for RCCL source/build trees (sparse clone at this prefix).

    rccl_lib_install_dir may point at the install root or at .../lib when reusing
    the ROCm layout.
    """
    install_dir = config_dict["rccl_lib_install_dir"].rstrip("/")
    if install_dir.endswith("/lib"):
        return install_dir[: -len("/lib")]
    return install_dir


def _rccl_source_layout(config_dict):
    """Return install prefix, RCCL project dir, and cmake build dir for rccl_lib_install_dir."""
    install_prefix = _rccl_install_prefix(config_dict)
    sparse_path = _sparse_checkout_path(config_dict, for_tests=False)
    if sparse_path:
        rccl_project_dir = f"{install_prefix}/{sparse_path}"
    else:
        rccl_project_dir = f"{install_prefix}/src"
    rccl_build_dir = f"{rccl_project_dir}/build"
    return install_prefix, rccl_project_dir, rccl_build_dir


def _find_rccl_shared_lib(hdl, search_roots):
    """
    Locate librccl.so or libnccl.so under search_roots on every node reached by hdl.

    Returns the path from the first node, or "" if not found on any node.
    """
    roots = " ".join(search_roots)
    out_dict = hdl.exec(
        f"bash -c 'for root in {roots}; do "
        f"find \"$root\" -maxdepth 5 "
        f"\\( -name librccl.so -o -name \"libnccl.so\" -o -name \"libnccl.so.*\" \\) "
        f"2>/dev/null; done | head -1'",
        timeout=60,
    )
    for node, output in out_dict.items():
        lib_path = output.strip().splitlines()[0] if output.strip() else ""
        if not lib_path:
            return ""
        log.info("Node %s: resolved RCCL shared library: %s", node, lib_path)
    return next(iter(out_dict.values())).strip().splitlines()[0]


def _resolve_rccl_shared_lib(hdl, search_roots):
    """
    Locate librccl.so or libnccl.so after a build-only (make -j) RCCL build.
    """
    lib_path = _find_rccl_shared_lib(hdl, search_roots)
    if not lib_path:
        roots = " ".join(search_roots)
        fail_test(f"RCCL shared library not found under {roots} after make -j")
    return lib_path


def _rocm_amdclang_pp(rocm_path):
    """ROCm HIP C++ driver used by rccl-tests (avoids hipcc -x hip link issues with .a archives)."""
    return f"{rocm_path.rstrip('/')}/llvm/bin/amdclang++"


def _rocm_hipcc(rocm_path):
    return f"{rocm_path.rstrip('/')}/bin/hipcc"


def _config_bool_str(value, default=True):
    """Parse installation_params flags stored as \"True\" / \"False\" strings."""
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() == "true"


def _rccl_tests_hip_compiler(rocm_path, use_amdclang):
    return _rocm_amdclang_pp(rocm_path) if use_amdclang else _rocm_hipcc(rocm_path)


def _build_env_exports(ompi_install_dir, rocm_path, *, use_amdclang=False):
    """
    Shell exports used before RCCL / rccl-tests builds (ROCm + Open MPI on PATH/LD_LIBRARY_PATH).

    rccl-tests sets use_amdclang=True so HIPCC/CXX match upstream install.sh (amdclang++).
    RCCL cmake builds keep the default hipcc wrapper.
    """
    ompi = ompi_install_dir.rstrip("/")
    rocm = rocm_path.rstrip("/")
    hip_compiler = _rocm_amdclang_pp(rocm) if use_amdclang else f"{rocm}/bin/hipcc"
    return (
        f"export OMPI_PREFIX={ompi}; "
        f"export ROCM_PREFIX={rocm}; "
        f"export ROCM_PATH={rocm}; "
        f"export HIPCC={hip_compiler}; "
        f"export CXX={hip_compiler}; "
        f"export PATH=${{OMPI_PREFIX}}/bin:${{ROCM_PREFIX}}/bin:${{PATH}}; "
        f"export LD_LIBRARY_PATH=${{OMPI_PREFIX}}/lib:${{ROCM_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}; "
    )


def _git_clone_source(hdl, repository, dest_dir, sparse_path=None, git_tag=""):
    """
    Clone repository at dest_dir. When sparse_path is set, use partial clone +
    sparse-checkout for a single rocm-systems project path.
    """
    hdl.exec(f"rm -rf {dest_dir}", timeout=60)
    branch_arg = f"-b {git_tag} " if git_tag else ""
    if sparse_path:
        log.info(
            "Sparse clone %s -> %s (path=%s)",
            repository,
            dest_dir,
            sparse_path,
        )
        hdl.exec(
            f"bash -c '"
            f"git clone --depth 1 --filter=blob:none {branch_arg} --sparse {repository} {dest_dir} && "
            f"cd {dest_dir} && git sparse-checkout set {sparse_path}"
            f"'",
            timeout=600,
        )
    else:
        log.info("Full clone %s -> %s", repository, dest_dir)
        hdl.exec(f"git clone {repository} {dest_dir}", timeout=300)

        if git_tag:
            out_dict = hdl.exec(
                f"bash -c 'cd {dest_dir} && git checkout {git_tag}'",
                timeout=120,
            )
            for node, output in out_dict.items():
                if re.search(r"error:|fatal:", output, re.I):
                    fail_test(f"git checkout {git_tag} failed on node {node}: {output.strip()}")


def _check_ompi_installed(hdl, config_dict):
    """
    Verify the user-supplied OMPI installation is present and functional.

    Checks performed (all run on every node reached by hdl):
      1. ompi_install_dir directory exists.
      2. ompi_install_dir/bin/mpirun is present and executable.
      3. mpirun --version produces output that contains the word 'mpi'.

    Args:
      hdl:         Pssh handle (shdl or phdl, already resolved by the caller).
      config_dict: Test configuration dict; must contain 'ompi_install_dir'.

    Returns:
      True  – OMPI is present and mpirun responds correctly on every node.
      False – any check failed on any node (details logged at ERROR level).
    """
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip('/')
    mpirun_path = f"{ompi_install_dir}/bin/mpirun"

    # --- 1. Directory exists ---
    log.info("Checking OMPI install directory: %s", ompi_install_dir)
    out_dict = hdl.exec(
        f"test -d {ompi_install_dir} && echo 'DIR_OK' || echo 'DIR_MISSING'",
        timeout=30,
    )
    for node, output in out_dict.items():
        if "DIR_MISSING" in output:
            log.error(
                "OMPI install directory not found on node %s: %s",
                node,
                ompi_install_dir,
            )
            return False

    # --- 2. mpirun binary is executable ---
    log.info("Checking mpirun binary: %s", mpirun_path)
    out_dict = hdl.exec(
        f"test -x {mpirun_path} && echo 'MPIRUN_OK' || echo 'MPIRUN_MISSING'",
        timeout=30,
    )
    for node, output in out_dict.items():
        if "MPIRUN_MISSING" in output:
            log.error(
                "mpirun not found or not executable on node %s: %s",
                node,
                mpirun_path,
            )
            return False

    # --- 3. mpirun --version produces recognisable output ---
    log.info("Running mpirun --version to validate binary on all nodes")
    out_dict = hdl.exec(
        f"{mpirun_path} --version",
        timeout=30,
    )
    for node, output in out_dict.items():
        if not output or "mpi" not in output.lower():
            log.error(
                "mpirun --version did not produce valid output on node %s: %s",
                node,
                output.strip() if output else "<empty>",
            )
            return False

    log.info("OMPI install directory and binaries are good: %s", ompi_install_dir)
    return True


def _install_rccl_lib(hdl, config_dict):
    """
    Clone and build the RCCL library from source (cmake + make -j in build/).

    For the rocm-systems monorepo, only projects/rccl is fetched (sparse checkout).

    Returns:
      tuple[str, str]: (NCCL_HOME for rccl-tests, path to built librccl.so / libnccl.so)
    """
    rccl_repository = config_dict["rccl_repository"]
    rccl_git_tag = config_dict.get("rccl_git_tag", "").strip()
    install_prefix = _rccl_install_prefix(config_dict)
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip("/")

    sparse_path = _sparse_checkout_path(config_dict, for_tests=False)
    rocm_path = detect_rocm_path(hdl, config_dict.get("rocm_path", "<changeme>"))
    build_env = _build_env_exports(ompi_install_dir, rocm_path)

    if sparse_path:
        repo_dir = install_prefix
        rccl_project_dir = f"{repo_dir}/{sparse_path}"
    else:
        repo_dir = f"{install_prefix}/src"
        rccl_project_dir = repo_dir

    log.info("Building RCCL library from source (cmake + make -j)")
    log.info("  repository : %s", rccl_repository)
    log.info("  tag        : %s", rccl_git_tag or "<none, using default branch>")
    log.info("  sparse     : %s", sparse_path or "<full clone>")
    log.info("  source dir : %s", rccl_project_dir)
    log.info("  workspace  : %s", install_prefix)
    log.info("  ROCm path  : %s", rocm_path)
    log.info("  OMPI prefix: %s", ompi_install_dir)

    _git_clone_source(
        hdl,
        rccl_repository,
        repo_dir,
        sparse_path=sparse_path,
        git_tag=rccl_git_tag,
    )

    rccl_build_dir = f"{rccl_project_dir}/build"
    hdl.exec(
        f"bash -c '{build_env}cd {rccl_project_dir} && mkdir -p build && cd build && cmake .. && make -j $(nproc)'",
        timeout=14400,
    )
    nccl_home = rccl_build_dir
    search_roots = [rccl_build_dir, rccl_project_dir]

    custom_lib = _resolve_rccl_shared_lib(hdl, search_roots)
    log.info("RCCL library build complete. NCCL_HOME=%s CUSTOM_RCCL_LIB=%s", nccl_home, custom_lib)
    return nccl_home, custom_lib


def _install_rccl_tests(hdl, config_dict, rccl_lib_prefix, use_custom_rccl_lib, custom_rccl_lib_path=""):
    """
    Clone and build rccl-tests against the resolved RCCL and OMPI installations.

    For rocm-systems, only projects/rccl-tests is sparse-cloned. When RCCL was built
    from source, the build passes CUSTOM_RCCL_LIB=<prefix>/lib/librccl.so.

    Args:
      hdl:                   Pssh handle (shdl or phdl, already resolved by caller).
      config_dict:           installation_params from rccl_config.json.
      rccl_lib_prefix:       RCCL install prefix or ROCm root for bundled librccl.
      use_custom_rccl_lib:   True when rccl-tests must link against a custom build.

    Returns:
      str: Path to the rccl-tests build directory.
    """
    rccl_tests_repository = config_dict["rccl_tests_repository"]
    rccl_tests_git_tag = config_dict.get("rccl_tests_git_tag", "").strip()
    rccl_tests_install_dir = config_dict["rccl_tests_install_dir"].rstrip("/")
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip("/")

    sparse_path = _sparse_checkout_path(config_dict, for_tests=True)
    rocm_path = detect_rocm_path(hdl, config_dict.get("rocm_path", "<changeme>"))
    use_amdclang = _config_bool_str(config_dict.get("rccl_tests_use_amdclang"), default=True)
    hip_compiler = _rccl_tests_hip_compiler(rocm_path, use_amdclang)
    build_env = _build_env_exports(ompi_install_dir, rocm_path, use_amdclang=use_amdclang)

    if sparse_path:
        repo_dir = rccl_tests_install_dir
        rccl_tests_srcdir = f"{repo_dir}/{sparse_path}"
    else:
        repo_dir = rccl_tests_install_dir
        rccl_tests_srcdir = rccl_tests_install_dir

    log.info("Building rccl-tests from source")
    log.info("  repository      : %s", rccl_tests_repository)
    log.info("  tag             : %s", rccl_tests_git_tag or "<none, using default branch>")
    log.info("  sparse          : %s", sparse_path or "<full clone>")
    log.info("  source dir      : %s", rccl_tests_srcdir)
    log.info("  RCCL lib prefix : %s", rccl_lib_prefix)
    log.info("  custom RCCL lib : %s", use_custom_rccl_lib)
    log.info("  MPI_HOME        : %s", ompi_install_dir)
    log.info("  ROCm path       : %s", rocm_path)
    log.info("  rccl_tests_use_amdclang : %s", use_amdclang)
    log.info("  HIP compiler    : %s", hip_compiler)

    _git_clone_source(
        hdl,
        rccl_tests_repository,
        repo_dir,
        sparse_path=sparse_path,
        git_tag=rccl_tests_git_tag,
    )

    make_cmd = f"make -j $(nproc) MPI=1 MPI_HOME={ompi_install_dir} ROCM_PATH={rocm_path} HIPCC={hip_compiler} "
    if use_custom_rccl_lib:
        make_cmd += f"CUSTOM_RCCL_LIB={custom_rccl_lib_path} NCCL_HOME={rccl_lib_prefix} "
    else:
        make_cmd += f"RCCL_HOME={rccl_lib_prefix} "

    out_dict = hdl.exec(
        f"bash -c '{build_env}cd {rccl_tests_srcdir} && {make_cmd}'",
        timeout=1800,
    )
    scan_test_results(out_dict)
    for node, output in out_dict.items():
        if re.search(r'\berror:', output, re.I):
            fail_test(f"rccl-tests build failed on node {node}: {output.strip()}")

    build_dir = f"{rccl_tests_srcdir}/build"
    log.info("rccl-tests build complete. Build dir: %s", build_dir)
    return build_dir


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


def test_install_rccl_tests(phdl, shdl, config_dict):
    """
    Build and install rccl-tests (and optionally the RCCL library) according
    to rccl_config.json installation_params.

    Steps:
      1. Resolve the orchestrator handle: shdl (head node only) when
         nfs_install=True, phdl (every node) otherwise.
      2. Verify the user-supplied OMPI installation is present and functional.
         Bail out immediately if the check fails.
      3. If rccl_lib_install=True, clone and build the RCCL library from
         source for rccl-tests. If rccl_lib_install=False, use bundled ROCm
         RCCL from rocm_path/lib (rccl_lib_install_dir is ignored).
      4. Clone and build rccl-tests against the resolved RCCL and OMPI paths.
      5. Verify the build artifacts exist on every node reached by the handle.
    """
    globals.error_list = []

    log.info("Testcase: install rccl-tests")

    nfs_install = config_dict["nfs_install"]
    rccl_lib_install = config_dict["rccl_lib_install"]

    rccl_lib_install_dir = config_dict["rccl_lib_install_dir"].rstrip('/')
    rccl_tests_install_dir = config_dict["rccl_tests_install_dir"].rstrip('/')
    rccl_repository = config_dict["rccl_repository"]
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip('/')

    # Resolve the orchestrator handle
    if nfs_install == "True":
        hdl = shdl
    else:
        hdl = phdl

    log.info("NFS install        : %s", nfs_install)
    log.info("RCCL lib install   : %s", rccl_lib_install)
    log.info("OMPI install dir   : %s", ompi_install_dir)
    log.info("RCCL tests dir     : %s", rccl_tests_install_dir)
    log.info("RCCL repository    : %s", rccl_repository)
    if rccl_lib_install == "True":
        log.info("RCCL lib dir       : %s", rccl_lib_install_dir)
    else:
        log.info(
            "RCCL lib dir       : bundled ROCm under %s/lib (rccl_lib_install_dir ignored)",
            config_dict.get("rocm_path", "<changeme>"),
        )

    # Verify OMPI is present and functional before doing any work
    ompi_installed = _check_ompi_installed(hdl, config_dict)
    if not ompi_installed:
        log.error(
            "OMPI check failed. Cannot build rccl-tests without a functional MPI. "
            "Verify 'ompi_install_dir' in config: %s",
            ompi_install_dir,
        )
        fail_test("OMPI not installed or not functional – aborting rccl-tests installation")
        update_test_result()
        return

    # Resolve RCCL library prefix
    custom_rccl_lib_path = ""
    if rccl_lib_install == "True":
        rccl_lib_prefix, custom_rccl_lib_path = _install_rccl_lib(hdl, config_dict)
        use_custom_rccl_lib = True
    else:
        rccl_lib_prefix = detect_rocm_path(hdl, config_dict.get("rocm_path", "<changeme>"))
        use_custom_rccl_lib = False
        log.info("Using bundled RCCL from ROCm at: %s/lib", rccl_lib_prefix)

    # Clone and build rccl-tests
    rccl_tests_build_dir = _install_rccl_tests(
        hdl,
        config_dict,
        rccl_lib_prefix,
        use_custom_rccl_lib,
        custom_rccl_lib_path,
    )

    # Verify build artifacts are present on every node
    log.info("Verifying rccl-tests build artifacts at: %s", rccl_tests_build_dir)
    out_dict = hdl.exec(f"ls {rccl_tests_build_dir}", timeout=30)
    for node, output in out_dict.items():
        if not output or re.search(r'No such file|cannot access', output, re.I):
            fail_test(f"rccl-tests build artifacts not found on node {node} at {rccl_tests_build_dir}")
        else:
            log.info(
                "Node %s: rccl-tests artifacts present:\n%s",
                node,
                output.strip(),
            )

    update_test_result()
