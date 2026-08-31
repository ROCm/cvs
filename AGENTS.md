# CVS (Cluster Validation Suite)

CVS is a collection of test suites that can help customers qualify AMD's ROCm cluster from a functionality and performance point of view and certify its readiness for production workloads.

**Key Value Propositions:**
- **Deterministic qualification**: Make AI infrastructure qualification more deterministic, repeatable and time bound through an automated validation process
- **Parallel execution**: Offer ability to run single node burn-in, health tests in parallel across a cluster and collate test results
- **Issue detection**: Quickly spot out configuration inconsistencies, hardware issues/degradation that can potentially impact AI workload performance
- **Troubleshooting tools**: Provide troubleshooting and visibility tools to narrow down problems when failures, performance issues are encountered

## Architecture

- **Validation Framework**: PyTest-based customer-facing test suites under `cvs/tests/`
- **Orchestrator Layer**: `orch` fixture from `cvs/tests/conftest.py` — primary execution abstraction (baremetal SSH or container backends via `cvs/core/orchestrators/`)
- **Cluster Management**: JSON-configured cluster topology, SSH credentials, and node connectivity
- **Test Suites**: health, rccl, platform, preflight, ibperf, training, inference, benchmark, and others — see `cvs list`
- **Reporting**: HTML reports, structured logging, and real-time monitoring

## Development Setup

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for complete development workflow, setup instructions, and quality checks.

Quick start:
```bash
git clone https://github.com/ROCm/cvs.git
cd cvs
make install
source .cvs_venv/bin/activate  # Activate CVS environment
cvs --version  # Validate installation
```

Before declaring a change done, from the repo root:
```bash
make fmt-check   # ruff format --check
make lint        # ruff check + pylint E1205/E1206 (logging arg counts)
make ut          # sdist -> .test_venv -> run_all_unittests.py
```

`make test` = `make ut` plus `test_cli.sh`. `make build` = `fmt-check lint sdist`.

## CVS Development Conventions

### Test Structure
- Test functions use `test_` prefix (PyTest requirement)
- Customer test suites live under `cvs/tests/`; library code under `cvs/lib/`
- Configuration files in `cvs/input/config_file/` (JSON-structured, explicit parameters). Sample configs mark mandatory user fields with `<changeme>` in value strings — users must replace every `<changeme>` before running tests; unresolved placeholders hard-exit at startup via `_resolve_placeholders_in_dict` (`cvs/lib/utils_lib.py`), reached by `resolve_test_config_placeholders`, which nearly every test module calls. The `<changeme>` validators in `cvs/schema/config_file/` (aorta and pytorch_xdit) cover only those config types.
- Tests require `--cluster_file` and `--config_file` CLI args (wired via `cvs/conftest.py`; the `orch` fixture is in `cvs/tests/conftest.py`)

### Orchestrator Patterns (Recommended)
- Use the `orch` fixture for new test development — see `cvs/tests/conftest.py`
- Execute cluster commands with `orch.exec(cmd, timeout=30)`
- Supports baremetal SSH and container orchestration backends
- Handle node failures gracefully; inspect per-node results in returned dicts

### Legacy Parallel SSH Patterns
- **Legacy**: Direct `phdl`/`shdl` usage — avoid in new code; many existing suites still use it
- `phdl.exec(cmd)` and `phdl.exec_cmd_list(cmd_list)` remain in older test files

### Error Handling & Logging
- Use `fail_test(msg)` for test failures with descriptive context
- Time-bounded dmesg scanning (current pattern in RCCL/ibperf suites):
  ```python
  start_time = handle.exec('date +"%a %b %e %H:%M:%S"')
  # ... run test ...
  end_time = handle.exec('date +"%a %b %e %H:%M:%S"')
  verify_dmesg_for_errors(handle, start_time, end_time, till_end_flag=False)
  ```
  `handle` is `phdl` today in most suites; `orch` works if it exposes `.exec()`. Use `till_end_flag=False` to bound scans to the test window.
- Dmesg parsing: `full_dmesg_scan` honors `CVS_DMESG_PARSER` (default `node-scraper`; set `legacy` for the regex path). Note that `verify_dmesg_for_errors` — the time-bounded scan shown above — always uses the `err_patterns_dict` regex path and ignores this variable.
- Use `err_patterns_dict` in `cvs/lib/verify_lib.py` for failure patterns

### Custom HTML Reports
- Generate test-specific HTML reports (performance charts, detailed results) as separate files
- Integrate with pytest HTML report using `request.config._html_report_manager.add_html_to_report()`:
  ```python
  # Generate custom report
  html_file = f'/tmp/custom_report_{os.getpid()}.html'
  # ... build HTML content ...

  # Add to pytest report bundle with clickable link
  copied_path = request.config._html_report_manager.add_html_to_report(
      html_file, link_name="Custom Performance Report", request=request
  )
  ```
- See `cvs/tests/rccl/rccl_perf.py`, `cvs/tests/preflight/preflight_checks.py` for examples

### Configuration Management
- Cluster files: node topology, SSH credentials, head-node settings (`cvs/input/cluster_file/`)
- Config files: test-specific parameters (RDMA interfaces, GPU settings, thresholds)
- Use `<changeme>` in sample/default JSON values for any field the user must customize; do not ship cluster-specific values as defaults
- Use `cvs config` and `cvs generate cluster_json` for setup automation
- Tune parallel SSH via `CVS_HOSTS_PER_SHARD` and `CVS_WORKERS_PER_CPU`

### Performance & Scalability
- Prefer time-bounded log collection: `journalctl -k -o short-iso --since="$start" --until="$end"`
- Prefer shell-level filtering on large clusters before transferring full journals
- Balance network I/O vs. parsing accuracy when changing dmesg/journalctl collection

## Git Workflow

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for complete development workflow including branch naming, commit style, and quality checks.

Key CVS-specific considerations:
- **Performance impact**: Assess cluster scalability for changes affecting parallel operations
- **Cross-platform testing**: Validate on Ubuntu (primary supported distro)
- **Configuration compatibility**: Ensure changes maintain backward compatibility with existing cluster/config files

## Code Quality Boundaries

### Unit Testing Standards
See [UNIT_TESTING_GUIDE.md](UNIT_TESTING_GUIDE.md) for comprehensive testing procedures.

- **All library functions and classes** must have corresponding unit tests in the tested module's own `unittests/` dir — e.g. `cvs/lib/preflight/unittests/`, not a central `cvs/lib/unittests/`. `run_all_unittests.py` discovers them recursively from the repo root.
- `cvs/core/orchestrators/` is the model: every concrete module has a same-named test beside it (`baremetal.py` → `unittests/test_baremetal.py`, and so on; `base.py` is an ABC and exempt).
- This applies to every library package, not just `cvs/lib/` — also `cvs/core/`, `cvs/runners/`, `cvs/parsers/`, `cvs/reports/`, `cvs/schema/`, `cvs/cli_plugins/`. The middle four have no `unittests/` dir yet; new code there should create one.
- Use `unittest` framework for all library unit tests
- Mock parallel SSH / orchestrator calls in unit tests — do not require live cluster access
- Fast, deterministic tests with no external service dependencies

### Security & Reliability
- Never hardcode credentials in test files
- Use SSH key-based authentication exclusively
- Validate cluster configuration before test execution
- Handle node failures without cascading test suite failures
- Report security issues to AMD security team following responsible disclosure

## Common Patterns

### Orchestrator Test Pattern (Recommended)
See `cvs/tests/health/rvs_cvs.py` for a real example:
```python
def test_example(orch, config_dict):
    out_dict = orch.exec(f'{config_dict["path"]}/rvs --version', timeout=30)
    for node, output in out_dict.items():
        if "expected" not in output:
            fail_test(f"Validation failed on {node}: {output}")
    update_test_result()
```

## Do Not

### Security & Credentials
- **Never commit credentials or SSH keys** to the repository
- **Never hardcode cluster-specific values** in test code (IP addresses, hostnames, interface names)
- **Always report security issues** to the user immediately, even if unrelated to current task

### Code Quality & Comments
- **Never add comments that explain what code does** - comments are only for why (non-obvious intent, trade-offs, constraints)
- **Never reference issue/PR numbers in code comments** - that context belongs in git history
- **Never add review feedback comments** ("changed per review", "moved to fix X") - code should stand alone
- **Never apply band-aid fixes** - if a cleaner refactor improves maintainability, do the refactor

### Unit Testing Standards
- **Never write tests that simulate behavior** instead of calling actual code - tests must exercise real functions
- **Never write environment-dependent unit tests** - use explicit fixtures for consistent results
- **Never add timeouts to unit tests** - unit tests should be fast and deterministic (remote `orch.exec(..., timeout=...)` is fine)
- **Never add external service dependencies** to unit tests - mock parallel SSH / orchestrator calls
- **Never mix unittest and pytest patterns** - use `unittest.TestCase` for library unit tests only
- **Never add library functions without unit tests** - all library code requires corresponding tests in that module's own `unittests/` dir

### CVS-Specific Don'ts
- **Never bypass time-bounded log scanning** - always use proper start/end timestamps for dmesg/journalctl
- **Never ignore orchestrator failures** - handle node connectivity issues gracefully
- **Never use legacy phdl/shdl in new tests** - use `orch` fixture for new development
- **Never run full log collection on large clusters** without time bounds or filtering
- **Never assume uniform hardware topology** - require explicit configuration per cluster
- **Never skip dmesg error scanning** after hardware-intensive tests (GPU burn-in, RCCL)
- **Never mix test categories in single test files** - maintain separation (health vs. rccl vs. training)
- **Never block the entire test suite** on single node failures - implement proper isolation
- **Never use `make installtest` in production** - use `make install` for end-user installs; `make installtest` is for dev only
- **Never use bare except clauses** - handle specific exceptions with proper error context

## Further Reading

### CVS Documentation
- [README.md](README.md) - Installation, usage, and getting started
- [CONTRIBUTORS.md](CONTRIBUTORS.md) - Development workflow and setup
- [UNIT_TESTING_GUIDE.md](UNIT_TESTING_GUIDE.md) - Testing procedures and guidelines
- [docs/](docs/) - Published documentation at https://rocm.docs.amd.com/projects/cvs/

### Test-Specific Guides
- [cvs/tests/health/README.md](cvs/tests/health/README.md) - Single-node health validation
- [cvs/tests/rccl/README.md](cvs/tests/rccl/README.md) - Multi-node RCCL networking tests
- [cvs/tests/preflight/README.md](cvs/tests/preflight/README.md) - Pre-flight cluster validation
- [cvs/input/config_file/preflight/README_preflight_config.md](cvs/input/config_file/preflight/README_preflight_config.md) - Preflight configuration guide

### Configuration References
- [cvs/input/cluster_file/README.md](cvs/input/cluster_file/README.md) - Cluster configuration format
- Sample configs: `cvs config list-dirs` to browse categories; `cvs config list` for file paths

### AMD/ROCm Resources
- [ROCm Documentation](https://rocm.docs.amd.com/) - ROCm platform documentation
- [RCCL Documentation](https://rccl.readthedocs.io/) - Multi-node communication library
- [AMD Instinct Documentation](https://instinct.docs.amd.com/) - GPU hardware and drivers