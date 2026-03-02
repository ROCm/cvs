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