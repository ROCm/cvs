The burn-in health tests are single node diagnostic tests which validate the hardware and firmware version's functionality and performance. For the performance validations, it will use the reference bandwidth or latency numbers provided as part of the input config_file for the relevant test. Currently we support the following burn-in tests. CVS does not package any of them along with CVS repo in order to avoid any dependencies between versions to ensure that CVS can work with any version of the burn-in tests. Some tests like AGFHC need NDA at this point.

Following are the currently supported test suites

1. AGFHC
2. transferbench
3. RVS (ROCmValidationSuite)


# How to run the tests

These are Pytest scripts and can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [user@host]~/cvs:(main)$
(myenv) [user@host]~/cvs:(main)$pwd
/home/user/cvs/cvs
(myenv) [user@host]~/cvs:(main)$pytest -vvv -log-file=/tmp/agfhc_test.log -s ./tests/health/agfhc_cvs.py --cluster_file ./input/cluster.json --config_file ./input/health/mi300_config.json --html=/var/www/html/cvs/agfhc_health_report.html --capture=tee-sys --self-contained-html
```
## Example: Running AGFHC Tests

```

In the config file, cvs/input/config_file/health/mi300_health_config.json, change the value of parameters(/root/cache/):         "package_tar_ball": "/root/cache/PACKAGES/agfhc-mi300x_1.22.0_ub2204.tar.bz2",
"install_dir": "/root/cache/INSTALL/agfhc/", to the desired location.

pytest -vvv --log-file=/tmp/test.log -s ./tests/health/install/install_agfhc.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html

pytest -vvv -log-file=/tmp/agfhc_test.log -s ./tests/health/agfhc_cvs.py --cluster_file ./input/cluster.json --config_file ./input/health/mi300_config.json --html=/var/www/html/cvs/agfhc_health_report.html --capture=tee-sys --self-contained-html
```

## Example: Running Transferbench Tests

```
In the config file, cvs/input/config_file/health/mi300_health_config.json, change the value of parameters(/tmp/cvs):
"path": "/tmp/cvs/INSTALL/TransferBench",
"example_tests_path": "/tmp/cvs/INSTALL/TransferBench/examples",
"git_install_path": "/tmp/cvs/INSTALL/", to desired location.

pytest -vvv --log-file=/tmp/test.log -s ./tests/health/install/install_agfhc.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/agfhc.html --capture=tee-sys --self-contained-html

pytest -vvv -log-file=/tmp/agfhc_test.log -s ./tests/health/agfhc_cvs.py --cluster_file ./input/cluster.json --config_file ./input/health/mi300_config.json --html=/var/www/html/cvs/agfhc_health_report.html --capture=tee-sys --self-contained-html
```


## Example: Running RVS Tests

```bash

In the config file, cvs/input/config_file/health/mi300_health_config.json, change the value of parameters(/tmp/rvs):
"git_install_path": "/tmp/rvs/INSTALL", to desired location.

pytest -vvv --log-file=/tmp/test.log -s ./tests/health/install/install_rvs.py --cluster_file input/cluster_file/cluster.json  --config_file input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/rvs.html --capture=tee-sys --self-contained-html

pytest -vvv --log-file=/tmp/test.log -s ./tests/health/rvs_cvs.py --cluster_file ./input/cluster_file/cluster.json --config_file ./input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/rvs_health_report.html --capture=tee-sys --self-contained-html
```

### RVS (ROCmValidationSuite) Test Suite

RVS provides comprehensive GPU validation through multiple test modules. The test suite intelligently adapts based on RVS version and GPU hardware detected.

#### Supported Test Modules

Individual test modules include:
- **MEM** (Memory Test) - Validates GPU memory functionality and integrity
- **GST** (GPU Stress Test) - Validates GPU functionality and performance under load
- **IET** (Peak Power Test) - Validates power consumption and thermal behavior
- **PEBB** (PCI Express Bandwidth Benchmark) - Measures and validates PCIe bandwidth performance
- **PBQT** (P2P Benchmark and Qualification Tool) - Validates peer-to-peer communication between GPUs
- **BABEL** (BABEL Benchmark) - GPU memory bandwidth validation using BABEL streaming benchmark
- **GPU Enumeration** - Basic GPU detection test

#### Automatic Device-Specific Configuration

The RVS test suite automatically detects the GPU device present in your system and uses the appropriate configuration files:

- Detects GPU model using `amd-smi` (e.g., MI300X, MI308X, MI300XHF)
- Searches for device-specific configuration folders (e.g., `/opt/rocm/share/rocm-validation-suite/conf/MI300X/`)
- Falls back to default configuration if device-specific configs are not available
- Works with any MI300 series variant

This ensures optimal test configuration for your specific hardware without manual intervention.

#### RVS Version-Based Test Execution

The test suite adapts its execution strategy based on the RVS version installed:

**RVS Version >= 1.3.0:**
- Runs **LEVEL-based configuration test** by default (collective execution of all modules)
- Skips individual module tests to reduce overall execution time
- Uses `-r <level>` option for comprehensive validation in a single run

**RVS Version < 1.3.0:**
- Runs **individual module tests** (LEVEL configs not supported)
- Each test module executes separately

**Note:** Test skipping based on RVS version is expected behavior and improves efficiency.

#### RVS Test Level Configuration

Configure the test depth and comprehensiveness using the `rvs_test_level` parameter in your config file:

```json
{
  "rvs": {
    "rvs_test_level": 4
  }
}
```

**Available Test Levels:**

- **Level 0**: Force individual module tests (skip LEVEL test regardless of RVS version)
- **Level 1**: Quick basic checks - Fast validation
- **Level 2**: Standard validation - Recommended for routine testing
- **Level 3**: Extended validation - Thorough testing
- **Level 4**: Comprehensive testing - **Default** (recommended for qualification)
- **Level 5**: Maximum stress testing - Most extensive validation


#### Future Direction

In future CVS releases, individual RVS module test functions will be removed in favor of the more efficient LEVEL-based configuration tests. The current implementation maintains backward compatibility while encouraging migration to LEVEL-based testing.

#### Expected Test Behavior

When running RVS tests, you may see some tests being skipped. This is **expected behavior** based on:

1. **RVS Version**: Tests skip based on version capabilities
   - Example: `SKIPPED [test_rvs_gpup_single: RVS version 1.3.0 >= 1.3.0: Running LEVEL-4 test instead]`

2. **Test Level Configuration**: Setting `rvs_test_level=0` skips LEVEL tests
   - Example: `SKIPPED [test_rvs_level_config: rvs_test_level=0: Running individual tests instead]`

This intelligent test selection reduces execution time while maintaining comprehensive validation coverage.

