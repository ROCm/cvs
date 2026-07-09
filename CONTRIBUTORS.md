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

We use [pre-commit](https://pre-commit.com/) hooks with [Ruff](https://docs.astral.sh/ruff/) to enforce linting and formatting. The hooks run automatically on every commit.

### Setup (one-time)

```bash
make pre-commit
```

This installs `pre-commit` and sets up the git hooks. From this point on, every `git commit` will automatically:
- Lint Python files and auto-fix issues (including unsafe fixes)
- Format Python files
- Strip trailing whitespace
- Fix missing end-of-file newlines
- Validate YAML syntax

If a hook modifies a file, the commit will be aborted — just re-stage the changes and commit again.

### Manual checks

You can also run the checks explicitly using Make targets.

### Advanced Linting
For unsafe fixes (like removing unused variables), use:
```bash
make unsafe-lint-fix
```

This provides interactive confirmation for each file with potentially breaking changes.

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Set up pre-commit hooks (first time only):
   ```bash
   make pre-commit
   ```

3. Make your changes

4. Run tests:
   ```bash
   make test
   ```

5. Commit your changes — pre-commit hooks will automatically check and fix lint/formatting:
   ```bash
   git commit -ms "Description of changes"
   ```
   If the hooks modify any files, re-stage and commit again.

6. Create a pull request

## Building for Distribution

To build the package for distribution:

```bash
make build
```

This creates a source distribution in the `dist/` directory.

## Available Make Targets

- `make help` - Show all available targets
- `make test-venv` - Create test virtual environment
- `make installtest` - Install package in development mode
- `make test` - Run all tests
- `make lint` - Check code quality (linting only) (linting only)
- `make fmt` - Format code
- `make fmt-check` - Check formatting without modifying files
- `make lint-fix` - Auto-fix safe linting issues
- `make unsafe-lint-fix` - Interactive unsafe fixes
- `make build` - Build distribution
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
