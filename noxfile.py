#!/usr/bin/env python3
"""
Nox configuration for CVS project.

This file provides cross-platform build automation, replacing the Linux-specific Makefile
while maintaining backward compatibility. All Makefile targets are available as Nox sessions.

Usage:
    nox -s <session>        # Run specific session
    nox -l                  # List all available sessions
    nox                     # Run default sessions

Examples:
    nox -s ut               # Run unit tests
    nox -s lint             # Run linting
    nox -s fmt              # Format code
    nox -s docs             # Build and serve docs
"""

import os
import shutil
import socket
from pathlib import Path

import nox

# Version constants (matching Makefile)
RUFF_VERSION = "0.14.8"
PYLINT_VERSION = "4.0.5"
DOC_PORT = int(os.environ.get("DOC_PORT", "8080"))

# Directory constants
TEST_VENV_DIR = ".test_venv"
CVS_VENV_DIR = ".cvs_venv"
RUFF_VENV_DIR = ".ruff_venv"
DOC_VENV_DIR = ".doc_venv"
DOC_BUILD_DIR = "docs/_build/html"

# Configure Nox
nox.options.sessions = ["ut", "lint", "fmt-check"]


# =============================================================================
# Core Build & Test Sessions
# =============================================================================


@nox.session(name="sdist")
def build_sdist(session: nox.Session) -> None:
    """Build source distribution."""
    session.install("setuptools")
    session.log("Building source distribution...")
    clean_sdist(session)
    session.run("python", "setup.py", "sdist")


@nox.session(name="build")
def build(session: nox.Session) -> None:
    """Format check, lint, then build source distribution."""
    session.log("Running build pipeline: fmt-check -> lint -> sdist")
    fmt_check(session)
    lint(session)
    build_sdist(session)


@nox.session(name="ut", venv_backend="none")
def unit_tests(session: nox.Session) -> None:
    """Run unit tests using .test_venv (matches make ut)."""
    # Equivalent to make installtest (creates .test_venv and installs package)
    session.log("Running installtest first...")
    session.run("python", "-m", "nox", "-s", "installtest", external=True)

    # Run tests using .test_venv python (equivalent to $(TEST_VENV_DIR)/bin/python)
    session.log("Unit Testing cvs...")
    import platform

    test_venv = Path(".test_venv")
    if platform.system() == "Windows":
        python_path = test_venv / "Scripts" / "python.exe"
    else:
        python_path = test_venv / "bin" / "python"

    session.run(str(python_path), "run_all_unittests.py", external=True)


@nox.session(name="test")
def test_all(session: nox.Session) -> None:
    """Run unit tests and CLI tests."""
    session.log("Running unit tests...")
    unit_tests(session)

    session.log("Running CLI tests...")
    # Set CVS environment variable to point to installed cvs
    cvs_path = session.bin / "cvs" if os.name != "nt" else session.bin / "cvs.exe"
    session.env["CVS"] = str(cvs_path)
    session.run("bash", "./test_cli.sh", external=True)


@nox.session(name="install", venv_backend="none")
def install_package(session: nox.Session) -> None:
    """Install package from built distribution into .cvs_venv (matches make install)."""
    # Build source distribution first by calling the separate sdist session
    session.log("Building source distribution using sdist session...")
    session.run("python", "-m", "nox", "-s", "sdist", external=True)

    # Create .cvs_venv if it doesn't exist (equivalent to make cvs-venv)
    cvs_venv = Path(".cvs_venv")
    if not cvs_venv.exists():
        session.log("Creating .cvs_venv...")
        session.run("python", "-m", "venv", ".cvs_venv", external=True)

    # Determine pip path based on platform
    import platform

    if platform.system() == "Windows":
        pip_path = cvs_venv / "Scripts" / "pip.exe"
    else:
        pip_path = cvs_venv / "bin" / "pip"

    # Install from built distribution (equivalent to $(CVS_PIP) install dist/*.tar.gz)
    session.log("Installing from built distribution into .cvs_venv...")
    dist_files = list(Path("dist").glob("*.tar.gz"))
    if not dist_files:
        session.error("No distribution files found. Run 'nox -s sdist' first.")

    # Install the most recent distribution file
    latest_dist = max(dist_files, key=os.path.getctime)
    session.run(str(pip_path), "install", str(latest_dist), external=True)


# =============================================================================
# Code Quality Sessions
# =============================================================================


@nox.session(name="lint")
def lint(session: nox.Session) -> None:
    """Run ruff linter and pylint logging checks."""
    session.install(f"ruff=={RUFF_VERSION}", f"pylint=={PYLINT_VERSION}")

    session.log("Running ruff linter...")
    try:
        session.run("ruff", "check", ".", "--unsafe-fixes")
    except nox.command.CommandFailed:
        session.error("\n\nLinting failed. Run 'nox -s lint_fix' to auto-fix issues.\n")

    session.log("Running pylint logging checks (E1205/E1206 on cvs/)...")
    try:
        session.run(
            "pylint",
            "cvs",
            "--disable=all",
            "--enable=logging-too-many-args,logging-too-few-args",
            "-j",
            "0",
            "--recursive=y",
        )
    except nox.command.CommandFailed:
        session.error("\n\nPylint logging checks failed. Fix logging call argument counts (see E1205/E1206).\n")


@nox.session(name="fmt")
def format_code(session: nox.Session) -> None:
    """Format code with ruff."""
    session.install(f"ruff=={RUFF_VERSION}")
    session.log("Running ruff formatter...")
    session.run("ruff", "format", ".")


@nox.session(name="fmt-check")
def fmt_check(session: nox.Session) -> None:
    """Check code formatting without modification."""
    session.install(f"ruff=={RUFF_VERSION}")
    session.log("Checking ruff formatting...")
    try:
        session.run("ruff", "format", "--check", ".")
    except nox.command.CommandFailed:
        session.error("\n\nFormatting check failed. Run 'nox -s fmt' to auto-fix formatting issues.\n")


@nox.session(name="lint-fix")
def lint_fix(session: nox.Session) -> None:
    """Auto-fix linting issues."""
    session.install(f"ruff=={RUFF_VERSION}")
    session.log("Running ruff linter with auto-fix...")
    session.run("ruff", "check", ".", "--fix")


@nox.session(name="unsafe-lint-fix")
def unsafe_lint_fix(session: nox.Session) -> None:
    """Interactive unsafe lint fixes with user confirmation."""
    session.install(f"ruff=={RUFF_VERSION}")

    session.log("")
    session.log(
        "WARNING: This will apply unsafe fixes that may remove unused variables or make other potentially breaking changes."
    )
    session.log("You can fix these issues manually after careful review, or proceed with per-file confirmation.")
    session.log("")

    # Get list of files with unsafe fixes
    session.log("Getting list of files with unsafe fixes...")
    try:
        result = session.run("ruff", "check", ".", "--unsafe-fixes", silent=True, success_codes=[0, 1])
        output = result if isinstance(result, str) else ""
    except Exception:
        output = ""

    # Parse files from output (simplified version of awk logic)
    files = []
    for line in output.split('\n'):
        if ' --> ' in line:
            # Extract filename from "path/file.py:line:col: --> ..."
            file_part = line.split(' --> ')[0].strip()
            if ':' in file_part:
                filename = file_part.split(':')[0]
                if filename not in files:
                    files.append(filename)

    if not files:
        session.log("No unsafe fixes needed.")
        return

    session.log("Files with unsafe fixes:")
    for file in files:
        session.log(f"  - {file}")
    session.log("")

    # Process each file interactively
    for file in files:
        session.log(f"File: {file} has unsafe fixes.")
        session.log("=== DIFF for {} ===".format(file))

        # Show diff
        try:
            session.run("ruff", "check", file, "--unsafe-fixes", "--diff")
            session.log("=== END DIFF ===")

            # Prompt user
            confirm = input("Apply fixes to this file? (y/N): ").strip().lower()
            if confirm in ('y', 'yes'):
                session.log(f"Applying unsafe fixes to {file}...")
                session.run("ruff", "check", file, "--fix", "--unsafe-fixes")
            else:
                session.log(f"Skipping {file}.")
        except nox.command.CommandFailed:
            session.log("No unsafe fixes available for this file.")
            session.log(f"If you want to fix issues manually, run: ruff check {file} --unsafe-fixes")
            session.log("=== END DIFF ===")

    session.log("Running formatter...")
    session.run("ruff", "format", ".")


# =============================================================================
# Documentation Session
# =============================================================================


@nox.session(name="docs")
def build_docs(session: nox.Session) -> None:
    """Build and serve Sphinx documentation with live reload."""
    session.install("-r", "docs/sphinx/requirements.txt")

    # Install sphinx-autobuild if not already available
    try:
        session.run("python", "-c", "import sphinx_autobuild", silent=True)
    except nox.command.CommandFailed:
        session.log("Installing sphinx-autobuild...")
        session.install("sphinx-autobuild")

    # Get local IP for cross-platform compatibility
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "localhost"

    session.log(f"Watching docs for changes. Open http://{local_ip}:{DOC_PORT} in your browser.")
    session.log("Press Ctrl+C to stop.")
    session.log("")

    # Set environment variable to disable GitHub activity reading
    session.env["ROCM_DOCS_CORE_READ_GITHUB_ACTIVITY"] = "false"

    session.run("sphinx-autobuild", "docs", DOC_BUILD_DIR, "--port", str(DOC_PORT), "--host", "0.0.0.0", "-q")


# =============================================================================
# Utility Sessions
# =============================================================================


@nox.session(name="gen-anc-suites")
def gen_anc_suites(session: nox.Session) -> None:
    """Generate per-group ANC suite files from anc_lib group lists."""
    session.log("Generating per-group ANC suite files from anc_lib group lists...")

    script = "build_tools/gen_anc_suites.py"

    # Try different Python environments (replicating Makefile fallback logic)
    for venv_dir in [CVS_VENV_DIR, TEST_VENV_DIR]:
        venv_path = Path(venv_dir)
        if os.name == "nt":
            python_path = venv_path / "Scripts" / "python.exe"
        else:
            python_path = venv_path / "bin" / "python"

        if python_path.exists() and python_path.is_file():
            session.log(f"Using Python from {python_path}")
            session.run(str(python_path), script, external=True)
            return

    # Fallback to session Python - install CVS package first
    session.log("Using session Python - installing CVS package...")
    session.install("-e", ".")
    session.run("python", script)


@nox.session(name="clean")
def clean_all(session: nox.Session) -> None:
    """Remove all virtual environments, build artifacts, and Python cache files."""
    session.log("Cleaning all build artifacts and virtual environments...")

    # Clean virtual environments
    for venv_dir in [TEST_VENV_DIR, CVS_VENV_DIR, RUFF_VENV_DIR, DOC_VENV_DIR]:
        if Path(venv_dir).exists():
            session.log(f"Removing {venv_dir}...")
            shutil.rmtree(venv_dir, ignore_errors=True)

    # Clean build artifacts
    for artifact_dir in ["dist", "build"]:
        if Path(artifact_dir).exists():
            session.log(f"Removing {artifact_dir}...")
            shutil.rmtree(artifact_dir, ignore_errors=True)

    # Clean egg-info directories
    for egg_info in Path(".").glob("*.egg-info"):
        session.log(f"Removing {egg_info}...")
        shutil.rmtree(egg_info, ignore_errors=True)

    for egg_info in Path("src").glob("*.egg-info") if Path("src").exists() else []:
        session.log(f"Removing {egg_info}...")
        shutil.rmtree(egg_info, ignore_errors=True)

    # Clean doc build directory
    if Path(DOC_BUILD_DIR).exists():
        session.log(f"Removing {DOC_BUILD_DIR}...")
        shutil.rmtree(DOC_BUILD_DIR, ignore_errors=True)

    # Clean docs/_toc.yml
    toc_file = Path("docs/sphinx/_toc.yml")
    if toc_file.exists():
        session.log(f"Removing {toc_file}...")
        toc_file.unlink()

    # Clean Python cache files
    session.log("Removing Python cache files...")
    _clean_pycache(".")


def _clean_pycache(root_dir: str) -> None:
    """Recursively remove Python cache files and directories."""
    root = Path(root_dir)

    # Remove __pycache__ directories
    for pycache_dir in root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir, ignore_errors=True)

    # Remove .pyc and .pyo files
    for pyc_file in root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
        except (OSError, PermissionError):
            pass

    for pyo_file in root.rglob("*.pyo"):
        try:
            pyo_file.unlink()
        except (OSError, PermissionError):
            pass


# =============================================================================
# Helper Sessions (for compatibility)
# =============================================================================


@nox.session(name="clean-sdist")
def clean_sdist(session: nox.Session) -> None:
    """Remove build artifacts (helper for sdist)."""
    # This session doesn't need setuptools - it just removes files
    session.log("Removing build artifacts...")
    for artifact_dir in ["dist", "build"]:
        if Path(artifact_dir).exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)

    for egg_info in Path(".").glob("*.egg-info"):
        shutil.rmtree(egg_info, ignore_errors=True)

    for egg_info in Path("src").glob("*.egg-info") if Path("src").exists() else []:
        shutil.rmtree(egg_info, ignore_errors=True)


@nox.session(name="help")
def show_help(session: nox.Session) -> None:
    """Show available targets (equivalent to make help)."""
    help_text = """
Available Nox sessions:

Core Build & Test:
  sdist           - Build source distribution
  build           - Format/lint check and then build source distribution  
  ut              - Execute all unit tests
  test            - Execute all unit tests and CLI tests
  install         - Install from built distribution

Code Quality:
  lint            - Run ruff + pylint (logging E1205/E1206 on cvs/)
  fmt             - Run ruff formatter
  fmt-check       - Check ruff formatting without modifying files
  lint-fix        - Run ruff linter with auto-fix (fixes code quality issues, not formatting)
  unsafe-lint-fix - Interactive unsafe lint fixes

Documentation:
  docs            - Build and serve docs with live-reload at http://localhost:{port}
                    Override port: DOC_PORT=9090 nox -s docs

Utilities:
  gen-anc-suites  - Regenerate per-group ANC suite files from anc_lib group lists
  clean           - Remove virtual environment, build artifacts, and Python cache files
  help            - Show this help message

Usage:
  nox -s <session>    # Run specific session
  nox -l              # List all sessions
  nox                 # Run default sessions (ut, lint, fmt-check)

Examples:
  nox -s ut           # Run unit tests
  nox -s lint fmt     # Run linting then formatting
  DOC_PORT=9090 nox -s docs  # Serve docs on port 9090
""".format(port=DOC_PORT)

    session.log(help_text)


# =============================================================================
# Session Aliases (for exact Makefile compatibility)
# =============================================================================


# Create session aliases to match exact Makefile target names
@nox.session(name="all")
def build_all(session: nox.Session) -> None:
    """Run build, test-venv, installtest, and test (equivalent to make all)."""
    session.log("Running full build pipeline...")
    build(session)
    unit_tests(session)  # Replaces installtest + ut
    # Note: test-venv creation is handled automatically by Nox


# Additional compatibility sessions
@nox.session(name="installtest", venv_backend="none")
def install_test(session: nox.Session) -> None:
    """Install package and prepare for testing using .test_venv (matches make installtest)."""
    # Build the package first (equivalent to make build dependency)
    session.log("Building package...")
    session.run("python", "-m", "nox", "-s", "build", external=True)

    # Create .test_venv if it doesn't exist (equivalent to make test-venv)
    test_venv = Path(".test_venv")
    if not test_venv.exists():
        session.log("Creating .test_venv...")
        session.run("python", "-m", "venv", ".test_venv", external=True)

    # Determine pip path based on platform
    import platform

    if platform.system() == "Windows":
        pip_path = test_venv / "Scripts" / "pip.exe"
    else:
        pip_path = test_venv / "bin" / "pip"

    # Install from built distribution (equivalent to $(PIP) install dist/*.tar.gz)
    session.log("Installing from built distribution into .test_venv...")
    dist_files = list(Path("dist").glob("*.tar.gz"))
    if not dist_files:
        session.error("No distribution files found. Run 'nox -s build' first.")

    # Install the most recent distribution file
    latest_dist = max(dist_files, key=os.path.getctime)
    session.run(str(pip_path), "install", str(latest_dist), external=True)


@nox.session(name="html-doc")
def html_doc(session: nox.Session) -> None:
    """Alias for docs session (equivalent to make html-doc)."""
    build_docs(session)
