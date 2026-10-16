'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest
import json
import re
import shlex

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    """
    Retrieve the --cluster_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the cluster configuration JSON file.
    """
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    """
    Retrieve the --config_file CLI option provided to pytest.

    Args:
      pytestconfig: Built-in pytest fixture exposing command-line options.

    Returns:
      str: Path to the test configuration JSON file.

    Notes:
      - Ensure your pytest invocation includes: --config_file=/path/to/config.json
      - Module scope ensures this is resolved once per module.
    """
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    """
    Load and return the entire cluster configuration.

    Args:
      cluster_file (str): Path to the cluster configuration JSON file.

    Returns:
      dict: Parsed cluster configuration (nodes, credentials, etc).
    """
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)

    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    """
    Load and return the RCCL install configuration dictionary for the test module.

    Expected rccl_tests_install_config.json structure:
    {
        "rccl_install": {
            "nfs_install":            "True" | "False",
            "rccl_lib_install":       "True" | "False",
            "rccl_lib_install_dir":   "/path/to/rccl/install",
            "rccl_tests_install_dir": "/path/to/rccl-tests",
            "rccl_repository":        "https://github.com/ROCm/rocm-systems.git",
            "rccl_git_tag":           "rocm-6.x.y",
            "rccl_tests_repository":  "same as rccl_repository if omitted",
            "rccl_tests_git_tag":     "rocm-6.x.y",
            "rccl_sparse_path":       "projects/rccl",
            "rccl_tests_sparse_path": "projects/rccl-tests",
            "ompi_install_dir":       "/opt/ompi/build",
            "rocm_path":              "/opt/rocm",
            "gpu_arch":               ""                   (or e.g. "gfx942"; empty = auto-detect),
            "rccl_tests_use_amdclang": "True" | "False"    (optional, default True),
            "rccl_lib_build_timeout": "10800"              (optional, default 14400)
        }
    }
    """
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)

    rccl_install_cfg = config_dict_t['rccl_install']

    rccl_install_cfg = resolve_test_config_placeholders(rccl_install_cfg, cluster_dict)
    if not rccl_install_cfg.get("rccl_tests_repository"):
        rccl_install_cfg["rccl_tests_repository"] = rccl_install_cfg["rccl_repository"]
    log.info("%s", rccl_install_cfg)
    return rccl_install_cfg


_UBUNTU_BUILD_DEPS = [
    "pkg-config",
    "libdrm-dev",
    "libdrm-amdgpu1",
    "libnuma-dev",
    "libpci-dev",
    "build-essential",
    "cmake",
    "git",
]


def _check_install_build_deps(hdl):
    """
    On Ubuntu nodes: detect missing build dependencies and install them with apt.
    Non-Ubuntu distros: no-op. Any failure logs a warning but does not abort the build.
    """
    out_dict = hdl.exec("grep -i ubuntu /etc/os-release 2>/dev/null && echo IS_UBUNTU || echo NOT_UBUNTU", timeout=30)
    is_ubuntu = any("IS_UBUNTU" in str(v) for v in out_dict.values())
    if not is_ubuntu:
        log.info("Non-Ubuntu distro detected; skipping build dependency check")
        return

    query_cmd = "dpkg-query -W -f='${Package} ${Status}\\n' " + " ".join(_UBUNTU_BUILD_DEPS) + " 2>/dev/null || true"
    out_dict = hdl.exec(query_cmd, timeout=30)

    missing_per_node = {}
    for node, output in out_dict.items():
        missing = []
        installed_pkgs = set()
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[3] == "installed":
                installed_pkgs.add(parts[0])
        for pkg in _UBUNTU_BUILD_DEPS:
            if pkg not in installed_pkgs:
                missing.append(pkg)
        if missing:
            missing_per_node[node] = missing

    if not missing_per_node:
        log.info("All build dependencies already installed")
        return

    for node, pkgs in missing_per_node.items():
        log.info("Node %s: missing build deps: %s", node, pkgs)

    # Collect the union of all missing packages across nodes
    all_missing = sorted({p for pkgs in missing_per_node.values() for p in pkgs})
    install_cmd = f"sudo -n apt-get install -y {' '.join(all_missing)}"
    log.info("Installing missing build deps: %s", all_missing)
    try:
        out_dict = hdl.exec(install_cmd, timeout=300)
        for node, output in out_dict.items():
            if re.search(r'\berror\b', output, re.I):
                log.warning("Build dep install may have issues on node %s: %s", node, output.strip()[:300])
    except Exception as exc:
        log.warning("Could not install build dependencies (non-fatal): %s", exc)


def _detect_gpu_arch(hdl, config_dict):
    """
    Return the GPU architecture string (e.g. 'gfx942') for RCCL/rccl-tests builds.

    Uses config gpu_arch when set. Otherwise runs rocminfo on every node reached
    by hdl, warns if architectures differ across nodes, and returns the detected arch.
    Returns empty string on failure so callers can skip the flag gracefully.
    """
    config_arch = str(config_dict.get("gpu_arch") or "").strip()
    if config_arch:
        log.info("Using configured gpu_arch: %s", config_arch)
        return config_arch

    rocm_path = config_dict.get("rocm_path", "/opt/rocm").rstrip("/")
    rocminfo_cmd = f"{rocm_path}/bin/rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | sort -u"
    log.info("Auto-detecting GPU architecture via rocminfo")
    try:
        out_dict = hdl.exec(rocminfo_cmd, timeout=60, print_console=False)
    except Exception as exc:
        log.warning("rocminfo failed (gpu_arch will not be set): %s", exc)
        return ""

    arch_per_node = {}
    for node, output in out_dict.items():
        archs = [a.strip() for a in output.strip().splitlines() if a.strip()]
        if archs:
            arch_per_node[node] = archs[0]
        else:
            log.warning("No GPU arch detected on node %s via rocminfo", node)

    if not arch_per_node:
        log.warning("Could not detect GPU arch on any node; AMDGPU_TARGETS will not be set")
        return ""

    unique_archs = set(arch_per_node.values())
    if len(unique_archs) > 1:
        log.warning(
            "Heterogeneous GPU architectures detected across nodes: %s; using %s",
            arch_per_node,
            sorted(unique_archs)[0],
        )

    detected = sorted(unique_archs)[0]
    log.info("Detected GPU arch: %s", detected)
    return detected


def detect_rocm_path(orch, config_rocm_path):
    """
    Detect the ROCm installation path, supporting both old (/opt/rocm) and new (/opt/rocm/core-X.Y) layouts.

    Args:
        orch: Orchestrator instance
        config_rocm_path (str): Configured ROCm path from config file (empty string for auto-detect)

    Returns:
        str: Detected ROCm path
    """
    if config_rocm_path:
        log.info(f'Using configured ROCm path: {config_rocm_path}')
        return config_rocm_path

    log.info('Auto-detecting ROCm path...')

    out_dict = orch.exec('ls -d /opt/rocm/core-* 2>/dev/null | sort -V | tail -1')
    for output in out_dict.values():
        if output and '/opt/rocm/core-' in output:
            rocm_path = output.strip()
            log.info(f'Detected ROCm path (new layout): {rocm_path}')
            return rocm_path

    out_dict = orch.exec('test -d /opt/rocm', detailed=True)
    for info in out_dict.values():
        if info.get('exit_code') == 0:
            log.info('Detected ROCm path (legacy layout): /opt/rocm')
            return '/opt/rocm'

    log.warning('Could not detect ROCm path, defaulting to /opt/rocm')
    return '/opt/rocm'


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


def _find_rccl_shared_lib(hdl, search_roots):
    """
    Locate librccl.so or libnccl.so under search_roots on every node reached by hdl.

    Returns the path from the first node, or "" if not found on any node.
    """
    roots = " ".join(shlex.quote(r) for r in search_roots)
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


def _config_bool_str(value, default=False):
    """Parse rccl_install flags stored as \"True\" / \"False\" strings."""
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
        f"export OMPI_PREFIX={shlex.quote(ompi)}; "
        f"export ROCM_PREFIX={shlex.quote(rocm)}; "
        f"export ROCM_PATH={shlex.quote(rocm)}; "
        f"export HIPCC={shlex.quote(hip_compiler)}; "
        f"export CXX={shlex.quote(hip_compiler)}; "
        f"export PATH=${{OMPI_PREFIX}}/bin:${{ROCM_PREFIX}}/bin:${{PATH}}; "
        f"export LD_LIBRARY_PATH=${{OMPI_PREFIX}}/lib:${{ROCM_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}; "
    )


def _git_clone_source(hdl, repository, dest_dir, sparse_path=None, git_tag=""):
    """
    Clone repository at dest_dir. When sparse_path is set, use partial clone +
    sparse-checkout for a single rocm-systems project path.
    """
    result = hdl.exec(f"rm -rf {shlex.quote(dest_dir)}", timeout=60, detailed=True)
    failed = [node for node, res in result.items() if res.get('exit_code') != 0]
    if failed:
        fail_test(f"rm -rf {dest_dir} failed on nodes {failed}")
    if sparse_path:
        log.info(
            "Sparse clone %s -> %s (path=%s)",
            repository,
            dest_dir,
            sparse_path,
        )
        branch_flag = f"-b {shlex.quote(git_tag)} " if git_tag else ""
        # Pass config-derived values as env vars — shlex.quote cannot be used
        # inside single-quoted bash -c strings directly.
        env = (
            f"GIT_REPO={shlex.quote(repository)} "
            f"DEST_DIR={shlex.quote(dest_dir)} "
            f"SPARSE_PATH={shlex.quote(sparse_path)} "
        )
        result = hdl.exec(
            f"{env}bash -c '"
            f"git clone --depth 1 --filter=blob:none {branch_flag}--sparse \"$GIT_REPO\" \"$DEST_DIR\" && "
            f"cd \"$DEST_DIR\" && git sparse-checkout set \"$SPARSE_PATH\""
            f"'",
            timeout=600,
            detailed=True,
        )
        failed = [node for node, res in result.items() if res.get('exit_code') != 0]
        if failed:
            fail_test(f"Sparse clone of {repository} failed on nodes {failed}: {result[failed[0]].get('output', '')}")
    else:
        log.info("Full clone %s -> %s", repository, dest_dir)
        result = hdl.exec(f"git clone {shlex.quote(repository)} {shlex.quote(dest_dir)}", timeout=300, detailed=True)
        failed = [node for node, res in result.items() if res.get('exit_code') != 0]
        if failed:
            fail_test(f"git clone {repository} failed on nodes {failed}: {result[failed[0]].get('output', '')}")

        if git_tag:
            result = hdl.exec(
                f"DEST_DIR={shlex.quote(dest_dir)} GIT_TAG={shlex.quote(git_tag)} "
                f"bash -c 'cd \"$DEST_DIR\" && git checkout \"$GIT_TAG\"'",
                timeout=120,
                detailed=True,
            )
            failed = [node for node, res in result.items() if res.get('exit_code') != 0]
            if failed:
                fail_test(f"git checkout {git_tag} failed on nodes {failed}: {result[failed[0]].get('output', '')}")


def _check_ompi_installed(hdl, config_dict):
    """
    Verify the user-supplied OMPI installation is present and functional.

    Checks performed (all run on every node reached by hdl):
      1. ompi_install_dir directory exists.
      2. ompi_install_dir/bin/mpirun is present and executable.
      3. mpirun --version produces output that contains the word 'mpi'.

    Args:
      hdl:         orch handle.
      config_dict: Test configuration dict; must contain 'ompi_install_dir'.

    Returns:
      True  – OMPI is present and mpirun responds correctly on every node.
      False – any check failed on any node (details logged at ERROR level).
    """
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip('/')
    mpirun_path = f"{ompi_install_dir}/bin/mpirun"

    # --- 1. Directory exists ---
    log.info("Checking OMPI install directory: %s", ompi_install_dir)
    result = hdl.exec(f"test -d {shlex.quote(ompi_install_dir)}", timeout=30, detailed=True)
    failed = [node for node, res in result.items() if res.get('exit_code') != 0]
    if failed:
        for node in failed:
            log.error("OMPI install directory not found on node %s: %s", node, ompi_install_dir)
        return False

    # --- 2. mpirun binary is executable ---
    log.info("Checking mpirun binary: %s", mpirun_path)
    result = hdl.exec(f"test -x {shlex.quote(mpirun_path)}", timeout=30, detailed=True)
    failed = [node for node, res in result.items() if res.get('exit_code') != 0]
    if failed:
        for node in failed:
            log.error("mpirun not found or not executable on node %s: %s", node, mpirun_path)
        return False

    # --- 3. mpirun --version produces recognisable output ---
    log.info("Running mpirun --version to validate binary on all nodes")
    result = hdl.exec(f"{shlex.quote(mpirun_path)} --version", timeout=30, detailed=True)
    for node, res in result.items():
        output = res.get('output', '')
        if res.get('exit_code') != 0 or not output or "mpi" not in output.lower():
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
    install_prefix = config_dict["rccl_lib_install_dir"].rstrip("/")
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip("/")

    sparse_path = _sparse_checkout_path(config_dict, for_tests=False)
    rocm_path = detect_rocm_path(hdl, config_dict.get("rocm_path", ""))
    build_env = _build_env_exports(ompi_install_dir, rocm_path)
    raw_timeout = str(config_dict.get("rccl_lib_build_timeout", "")).strip()
    rccl_lib_build_timeout = int(raw_timeout) if raw_timeout else 14400

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
    log.info("  build timeout: %s", rccl_lib_build_timeout)

    _git_clone_source(
        hdl,
        rccl_repository,
        repo_dir,
        sparse_path=sparse_path,
        git_tag=rccl_git_tag,
    )

    rccl_build_dir = f"{rccl_project_dir}/build"
    gpu_arch = _detect_gpu_arch(hdl, config_dict)
    if not gpu_arch:
        fail_test(
            "gpu_arch could not be determined. "
            "Set 'gpu_arch' explicitly in rccl_tests_install_config.json (e.g. 'gfx942' for MI300X, 'gfx1250' for MI455X). "
            "Building RCCL without a target GPU architecture takes hours or fails on some hardware."
        )
        update_test_result()
        return None, ""
    log.info("  gpu_arch   : %s", gpu_arch)
    hipcc = f"{rocm_path.rstrip('/')}/bin/hipcc"
    cmake_flags = (
        f"-DCMAKE_PREFIX_PATH={shlex.quote(rocm_path)} "
        f"-DCMAKE_CXX_COMPILER={shlex.quote(hipcc)} "
        f"-DGPU_TARGETS={shlex.quote(gpu_arch)} "
        f"-DAMDGPU_TARGETS={shlex.quote(gpu_arch)}"
    )
    result = hdl.exec(
        f"BUILD_DIR={shlex.quote(rccl_project_dir)} "
        f"bash -c '{build_env}cd \"$BUILD_DIR\" && mkdir -p build && cd build && cmake {cmake_flags} .. && make -j $(nproc)'",
        timeout=rccl_lib_build_timeout,
        detailed=True,
    )
    failed = [node for node, res in result.items() if res.get('exit_code') != 0]
    if failed:
        fail_test(f"RCCL library build (cmake/make) failed on nodes {failed}: {result[failed[0]].get('output', '')}")
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
      hdl:                   orch handle.
      config_dict:           rccl_install object from rccl_tests_install_config.json.
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
    rocm_path = detect_rocm_path(hdl, config_dict.get("rocm_path", ""))
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

    # rccl-tests build for all arch doesn't break or take too long
    # so even if the gpu arch is not given or couldn't be detected
    # the rccl-tests build can proceed with building it for all
    # architectures.
    gpu_arch = _detect_gpu_arch(hdl, config_dict)
    log.info("  gpu_arch            : %s", gpu_arch or "<auto/not set>")

    make_cmd = (
        f"make -j $(nproc) MPI=1 "
        f"MPI_HOME={shlex.quote(ompi_install_dir)} "
        f"ROCM_PATH={shlex.quote(rocm_path)} "
        f"HIPCC={shlex.quote(hip_compiler)} "
    )
    if gpu_arch:
        make_cmd += f"AMDGPU_TARGETS={shlex.quote(gpu_arch)} GPU_TARGETS={shlex.quote(gpu_arch)} "
    if use_custom_rccl_lib:
        make_cmd += f"CUSTOM_RCCL_LIB={shlex.quote(custom_rccl_lib_path)} NCCL_HOME={shlex.quote(rccl_lib_prefix)} "
    else:
        make_cmd += f"RCCL_HOME={shlex.quote(rccl_lib_prefix)} "

    result = hdl.exec(
        f"SRC_DIR={shlex.quote(rccl_tests_srcdir)} bash -c '{build_env}cd \"$SRC_DIR\" && {make_cmd}'",
        timeout=1800,
        detailed=True,
    )
    out_dict = {node: res.get('output', '') for node, res in result.items()}
    scan_test_results(out_dict)
    failed = [node for node, res in result.items() if res.get('exit_code') != 0]
    if failed:
        fail_test(f"rccl-tests build failed on nodes {failed}: {out_dict[failed[0]]}")

    build_dir = f"{rccl_tests_srcdir}/build"
    log.info("rccl-tests build complete. Build dir: %s", build_dir)
    return build_dir


def test_install_rccl_tests(orch, config_dict):
    """
    Build and install rccl-tests (and optionally the RCCL library) according
    to rccl_tests_install_config.json.

    Steps:
      1. Resolve the build handle: orch.head (head node only) when
         nfs_install=True (NFS propagates to all nodes), orch.all (every node)
         otherwise. Works with a single-node cluster either way.
      2. Check and install Ubuntu build prerequisites (non-fatal on failure or
         non-Ubuntu distros).
      3. Verify the user-supplied OMPI installation is present and functional.
         Bail out immediately if the check fails.
      4. If rccl_lib_install=True, clone and build the RCCL library from
         source for rccl-tests. If rccl_lib_install=False, use bundled ROCm
         RCCL from rocm_path/lib (rccl_lib_install_dir is ignored).
      5. Clone and build rccl-tests against the resolved RCCL and OMPI paths.
         GPU architecture is passed explicitly (AMDGPU_TARGETS / GPU_TARGETS)
         either from config gpu_arch or auto-detected via rocminfo.
      6. Verify the build artifacts exist on every node reached by the handle.
    """
    globals.error_list = []

    log.info("Testcase: install rccl-tests")

    nfs_install = _config_bool_str(config_dict["nfs_install"])
    rccl_lib_install = _config_bool_str(config_dict["rccl_lib_install"])

    rccl_lib_install_dir = config_dict["rccl_lib_install_dir"].rstrip('/')
    rccl_tests_install_dir = config_dict["rccl_tests_install_dir"].rstrip('/')
    rccl_repository = config_dict["rccl_repository"]
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip('/')

    if nfs_install:
        hdl = orch.head
    else:
        hdl = orch.all

    # Guard always-required path parameters against empty strings
    _required_paths = {
        "rccl_tests_install_dir": rccl_tests_install_dir,
        "ompi_install_dir": ompi_install_dir,
        "rccl_repository": rccl_repository,
    }
    for param, value in _required_paths.items():
        if not value:
            log.error("Config parameter '%s' is empty. Update it in rccl_tests_install_config.json", param)
            fail_test(f"Required config parameter '{param}' is empty – aborting rccl-tests installation")
            update_test_result()
            return

    log.info("NFS install        : %s", config_dict["nfs_install"])
    log.info("RCCL lib install   : %s", config_dict["rccl_lib_install"])
    log.info("OMPI install dir   : %s", ompi_install_dir)
    log.info("RCCL tests dir     : %s", rccl_tests_install_dir)
    log.info("RCCL repository    : %s", rccl_repository)
    log.info("RCCL lib dir       : %s", rccl_lib_install_dir)

    # When building a custom RCCL library, the install dir must not be the default
    # ROCm lib path — building into /opt/rocm/lib would corrupt the system ROCm install.
    if rccl_lib_install and rccl_lib_install_dir == "/opt/rocm/lib":
        log.error(
            "rccl_lib_install is True but rccl_lib_install_dir is still '/opt/rocm/lib'. "
            "Set rccl_lib_install_dir to a writable build directory in rccl_tests_install_config.json"
        )
        fail_test("rccl_lib_install_dir must be changed from the default when rccl_lib_install=True")
        update_test_result()
        return

    # OS packages are node-local, not NFS-shared — always install on every node
    _check_install_build_deps(orch.all)

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
    if rccl_lib_install:
        rccl_lib_prefix, custom_rccl_lib_path = _install_rccl_lib(hdl, config_dict)
        use_custom_rccl_lib = True
    else:
        rccl_lib_prefix = detect_rocm_path(hdl, config_dict.get("rocm_path", ""))
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
    result = hdl.exec(f"ls {shlex.quote(rccl_tests_build_dir)}", timeout=30, detailed=True)
    for node, res in result.items():
        if res.get('exit_code') != 0 or not res.get('output', '').strip():
            fail_test(f"rccl-tests build artifacts not found on node {node} at {rccl_tests_build_dir}")
        else:
            log.info("Node %s: rccl-tests artifacts present:\n%s", node, res['output'].strip())

    update_test_result()
