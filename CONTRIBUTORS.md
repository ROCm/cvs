# Contributors Guide

Welcome to the CVS (ROCm Cluster Validation Suite) project! This guide will help you get started with contributing to the codebase.

## Prerequisites

- Python 3.9 or later
- Git

**Debian/Ubuntu Systems:** On Debian and Ubuntu distributions, install the `venv` module:
```bash
sudo apt install python3-venv
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ROCm/cvs.git
   cd cvs
   ```

2. Set up the development environment:
   ```bash
   make test-venv
   source .test_venv/bin/activate  # On Linux/macOS
   # or
   .test_venv\Scripts\activate     # On Windows
   ```

3. Install the package in development mode:
   ```bash
   make installtest
   ```

   This installs `requirements-test.txt` alongside CVS. Its `jsonschema`
   dependency validates the RCCL Run Deck JSON profile against its schema in
   unit tests; it is not required by end-user installs.

4. Enable the repository git hooks (run once per fresh clone):
   ```bash
   make hooks
   ```

   This points `core.hooksPath` at `.githooks/`, enabling:
   - `commit-msg` - checks the message has a title, a blank 2nd line, a body,
     and a trailing `Signed-off-by` (warns, without blocking, on lines over 80
     characters).
   - `pre-commit` - runs `ruff format --check` and `ruff check` plus the
     pylint logging checks (the same gates as `make fmt-check` and `make lint`),
     then a **security scan** of the staged diff, and **blocks the commit** if
     any fail; also warns on trailing whitespace in staged changes. Needs the
     ruff venv (`make ruff-venv`, created by `make installtest`).
   - `pre-push` - runs the **security scan** over the commits being pushed, then
     the full unit-test suite (`run_all_unittests.py`, the same tests as
     `make test`), and **blocks the push** if either fails. Needs a test venv
     (`make test-venv installtest`).

   The security scan (`build_tools/security_scan.py`) inspects only the added
   lines and blocks internal-infrastructure or secret leaks: internal
   `*.amd.com` endpoints (public docs domains such as `rocm.docs.amd.com` stay
   allowed), real cluster hostnames (a family name followed by rack/node
   coordinates, e.g. `heliosr-<rack>-<node>`; the short family names `heliosr`,
   `heliosp`, `heliosm-rck` stay allowed), https URLs that expose a non-standard
   listening port, and committed tokens/keys. For a verified false positive,
   append the `cvs-security-allow` pragma to that line.

   This infra/secret scanner runs **only in these local hooks** -- CI does not
   invoke it. CI runs the unit tests (`make test`) and a separate security
   workflow (gitleaks, bandit, trivy, CodeQL, zizmor); those catch committed
   secrets independently, but the internal-host, cluster-hostname, and port
   checks are enforced locally only. So `--no-verify` skips them with no CI
   backstop -- use it sparingly.

   Git never auto-installs repository hooks, so each clone must run `make hooks`
   once. Bypass a single commit with `git commit --no-verify`, or a single push
   with `git push --no-verify`; the ruff/pylint/test gates are still re-run in
   CI, but the infra/secret scanner above is not.

## Running Tests

Before submitting changes, ensure all tests pass:

```bash
make test
```

This command will:
- Run all unit tests
- Execute CLI command tests
- Validate that your changes don't break existing functionality

For detailed information on testing procedures and guidelines, see [UNIT_TESTING_GUIDE.md](UNIT_TESTING_GUIDE.md).

## Code Quality

We use Ruff for linting and formatting Python code. Always check and fix code quality issues:

### Check Formatting
```bash
make fmt-check
```

This will check for formatting issues without modifying files.

### Format Code
```bash
make fmt
```

This will format all Python files according to our style guide.

### Check Code Quality
```bash
make lint
```

This will check for:
- Linting issues (code style, potential bugs)

### Auto-fix Issues
```bash
make lint-fix
```

This will automatically fix safe linting issues.

### Advanced Linting
For unsafe fixes (like removing unused variables), use:
```bash
make unsafe-lint-fix
```

This provides interactive confirmation for each file with potentially breaking changes.

## Documentation Preview

Before committing documentation changes, preview them locally in your browser to verify formatting and structure:

```bash
make html-doc
```

This will:
- Create a dedicated Python virtual environment (`.doc_venv/`) with all Sphinx dependencies
- Build the HTML documentation
- Start a live-reload server accessible at the URL printed in the terminal
- Automatically rebuild whenever you save a `.rst` or `.md` file

Press `Ctrl+C` to stop the server. To override the default port (8080):

```bash
make html-doc DOC_PORT=9090
```

To clean up the documentation environment and build artifacts:

```bash
make clean
```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run quality checks:
   ```bash
   make fmt-check
   make lint
   make test
   ```

4. Fix any issues:
   ```bash
   make fmt
   make lint-fix
   make unsafe-lint-fix
   ```

5. Run tests again:
   ```bash
   make test
   ```

6. Commit your changes:
   ```bash
   git commit -ms "Description of changes"
   ```

7. Create a pull request

## Building for Distribution

To build the package for distribution:

```bash
make build
```

This creates a source distribution in the `dist/` directory.

## Available Make Targets

- `make help` - Show all available targets
- `make hooks` - Enable the repo git hooks (commit-msg format, pre-commit ruff + security gate, pre-push security scan + unit tests)
- `make test-venv` - Create test virtual environment
- `make installtest` - Install package in development mode
- `make test` - Run all tests
- `make lint` - Check code quality (linting only) (linting only)
- `make fmt` - Format code
- `make fmt-check` - Check formatting without modifying files
- `make lint-fix` - Auto-fix safe linting issues
- `make unsafe-lint-fix` - Interactive unsafe fixes
- `make build` - Build distribution
- `make html-doc` - Build and serve docs with live-reload (preview in browser)
- `make clean` - Clean build artifacts and environments

## Code Style Guidelines

- Use Ruff for consistent formatting
- Follow PEP 8 style guidelines
- Write descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

## Getting Help

If you have questions:
- Check the existing issues and documentation
- Ask in the ROCm community forums
- Contact the maintainers

Thank you for contributing to CVS!
