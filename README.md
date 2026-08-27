# Cluster Validation Suite

> [!NOTE]
> **Documentation:** Full documentation is at [rocm.docs.amd.com/projects/cvs](https://rocm.docs.amd.com/projects/cvs/en/latest/) (search, table of contents, and how-to guides). Source files are in the `docs/` folder. To contribute, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

CVS is a collection of test suites that validate AMD AI clusters end to end — from single-node burn-in health checks to cluster-wide distributed training and inference. CVS requires only SSH connectivity to cluster nodes (no Slurm or Kubernetes).

## Quick install

```bash
git clone https://github.com/ROCm/cvs
cd cvs
make install
source .cvs_venv/bin/activate
cvs --version
cvs list
```

## Where to go next

| Task | Documentation |
|------|----------------|
| First run in ~15 minutes | [Quickstart](docs/getting-started/quickstart.rst) |
| Install and upgrade | [Install](docs/getting-started/install.rst) · [Upgrade](docs/getting-started/upgrade.rst) |
| Set up cluster and config files | [Set up test configs](docs/how-to/configure/test-suite-config/index.rst) · [Set up cluster file](docs/how-to/configure/cluster-config.rst) |
| Run test suites | [Run tests](docs/how-to/run-tests/index.rst) |
| JSON schemas | [Configuration files](docs/reference/configuration-files/index.rst) |
| CLI options | [CLI reference](docs/reference/cli/cvs-run.rst) |

Repository layout: `tests/` (pytest suites), `lib/` (Python utilities), `input/` (sample JSON configs), `utils/` (standalone scripts).

Public repository: https://github.com/ROCm/cvs
