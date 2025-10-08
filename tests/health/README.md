The burn-in health tests are single node diagnostic tests which validate the hardware and firmware version's functionality and performance. For the performance validations, it will use the reference bandwidth or latency numbers provided as part of the input config_file for the relevant test. Currently we support the following burn-in tests. CVS does not package any of them along with CVS repo in order to avoid any dependencies between versions to ensure that CVS can work with any version of the burn-in tests. Some tests like AGFHC need NDA at this point.

Following are the currently supported test suites

1. AGFHC
2. Babelstream
3. rocBLAS
4. rocHPL
5. transferbench
6. RVS (ROCmValidationSuite)


# How to run the tests

These are PYtest scripts and can be run in the following fashion (for the details on arguments and their purpose, please refer the main README under the CVS parent folder

```
(myenv) [venksrin@ubuntu-host1]~/cvs:(main)$
(myenv) [venksrin@ubuntu-host1]~/cvs:(main)$pwd
/home/venksrin/cvs
(myenv) [venksrin@ubuntu-host1]~/cvs:(main)$pytest -vvv -log-file=/tmp/agfhc_test.log -s ./tests/health/agfhc_cvs.py --cluster_file ./input/cluster.json --config_file ./input/health/mi300_config.json --html=/var/www/html/cvs/agfhc_health_report.html --capture=tee-sys --self-contained-html
```


## Example: Running RVS Tests

```bash
pytest -vvv --log-file=/tmp/rvs_test.log -s ./tests/health/rvs_cvs.py --cluster_file ./input/cluster_file/cluster.json --config_file ./input/config_file/health/mi300_health_config.json --html=/var/www/html/cvs/rvs_health_report.html --capture=tee-sys --self-contained-html
```

RVS (ROCmValidationSuite) tests include:
- GST (GPU Stress Test) - Single GPU validation
- IET (Input EDPp Test) - Power and thermal validation
- PEBB (PCI Express Bandwidth Benchmark) - PCIe performance validation
- GPU Enumeration - Basic GPU detection
- Memory Test - GPU memory functionality validation

The RVS tests automatically detect and use MI300X-specific configuration files when available, falling back to default configurations otherwise.
