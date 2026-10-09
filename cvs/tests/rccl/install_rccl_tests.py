'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import pytest
import json
import re

from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals
from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory

log = globals.log


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
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
            "rccl_git_tag":           "rocm-6.x.y",        (optional)
            "rccl_tests_repository":  "same as rccl_repository if omitted",
            "rccl_tests_git_tag":     "rocm-6.x.y",        (optional)
            "rccl_sparse_path":       "projects/rccl",     (optional, auto for rocm-systems)
            "rccl_tests_sparse_path": "projects/rccl-tests",
            "ompi_install_dir":       "/opt/ompi/build",
            "rocm_path":              "/opt/rocm"          (or "<changeme>"),
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


@pytest.fixture(scope="module")
def orch(cluster_file, config_file):
    if not cluster_file or not config_file:
        pytest.fail("orch fixture requires --cluster_file and --config_file")
    cfg = OrchestratorConfig.from_configs(cluster_file, config_file)
    orchestrator = OrchestratorFactory.create_orchestrator(log, cfg)
    yield orchestrator
    orchestrator.close()


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
    log.info("  gpu_arch   : %s", gpu_arch or "<auto/not set>")
    hipcc = f"{rocm_path.rstrip('/')}/bin/hipcc"
    cmake_flags = f"-DCMAKE_PREFIX_PATH={rocm_path} -DCMAKE_CXX_COMPILER={hipcc}"
    if gpu_arch:
        cmake_flags += f" -DGPU_TARGETS={gpu_arch} -DAMDGPU_TARGETS={gpu_arch}"
    hdl.exec(
        f"bash -c '{build_env}cd {rccl_project_dir} && mkdir -p build && cd build && cmake {cmake_flags} .. && make -j $(nproc)'",
        timeout=rccl_lib_build_timeout,
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

    gpu_arch = _detect_gpu_arch(hdl, config_dict)
    log.info("  gpu_arch            : %s", gpu_arch or "<auto/not set>")

    make_cmd = f"make -j $(nproc) MPI=1 MPI_HOME={ompi_install_dir} ROCM_PATH={rocm_path} HIPCC={hip_compiler} "
    if gpu_arch:
        make_cmd += f"AMDGPU_TARGETS={gpu_arch} GPU_TARGETS={gpu_arch} "
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

    nfs_install = config_dict["nfs_install"]
    rccl_lib_install = config_dict["rccl_lib_install"]

    rccl_lib_install_dir = config_dict["rccl_lib_install_dir"].rstrip('/')
    rccl_tests_install_dir = config_dict["rccl_tests_install_dir"].rstrip('/')
    rccl_repository = config_dict["rccl_repository"]
    ompi_install_dir = config_dict["ompi_install_dir"].rstrip('/')

    # orch.head = single head-node handle (Pssh); orch.all = all-nodes handle (Pssh).
    # Both share the same .exec(cmd, timeout=...) interface used by all helpers below.
    if nfs_install == "True":
        hdl = orch.head
    else:
        hdl = orch.all

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

    # Check and install Ubuntu build prerequisites (non-fatal)
    _check_install_build_deps(hdl)

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
