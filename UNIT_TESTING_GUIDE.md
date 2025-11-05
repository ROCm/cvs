
# Unit Test Organization

This project separates **unit tests** and **cluster validation tests** to maintain clarity and modularity.

---

## 📁 Directory Structure

```
cvs/
├── lib/
│   ├── docker_lib.py
│   ├── utils_lib.py
│   ├── verify_lib.py
│   └── unittests/              # Unit tests for lib modules
│       ├── __init__.py
│       ├── test_verify_lib.py
│       └── test_html_lib.py
├── tests/                       # cluster validation tests (Pytest)
│   ├── health
│   ├── ibperf
│   ├── rccl
├── conftest.py
├── pytest.ini
├── README.md
```

---

## ✅ Unit Tests (using `unittest`)

- Use Python's built-in `unittest` framework
- Each test file should be named `test_*.py`
- Each test class should inherit from `unittest.TestCase`

### 🔧 How to Run Unit Tests

From the project root (`cvs/`):

```bash
python -m unittest discover -s lib/unittests
```

This will discover and run all unit tests under `lib/unittests/`.

---

## 🛠️ Run All Unit Tests from Multiple Folders

If you have multiple unit test folders (e.g., `lib/unittests`, `utils/unittests`), create a script like:

Run it with:

```bash
python run_all_unittests.py
test_missing_gpu_key_raises_keyerror (test_html_lib.TestBuildHtmlMemUtilizationTable.test_missing_gpu_key_raises_keyerror) ... Build HTML mem utilization table
ok
test_multiple_nodes (test_html_lib.TestBuildHtmlMemUtilizationTable.test_multiple_nodes) ... Build HTML mem utilization table
ok
test_rocm7_style_gpu_data (test_html_lib.TestBuildHtmlMemUtilizationTable.test_rocm7_style_gpu_data) ... Build HTML mem utilization table
ok
test_single_node_valid_input (test_html_lib.TestBuildHtmlMemUtilizationTable.test_single_node_valid_input) ... Build HTML mem utilization table
ok
test_bytes_only (test_html_lib.TestNormalizeBytes.test_bytes_only) ... ok
test_gigabytes (test_html_lib.TestNormalizeBytes.test_gigabytes) ... ok
test_kilobytes_binary (test_html_lib.TestNormalizeBytes.test_kilobytes_binary) ... ok
test_kilobytes_decimal (test_html_lib.TestNormalizeBytes.test_kilobytes_decimal) ... ok
test_megabytes (test_html_lib.TestNormalizeBytes.test_megabytes) ... ok
test_negative_bytes (test_html_lib.TestNormalizeBytes.test_negative_bytes) ... ok
test_precision (test_html_lib.TestNormalizeBytes.test_precision) ... ok
test_type_error (test_html_lib.TestNormalizeBytes.test_type_error) ... ok
test_invalid_bus_speed (test_verify_lib.TestVerifyGpuPcieBusWidth.test_invalid_bus_speed) ... ok
test_valid_bus_width (test_verify_lib.TestVerifyGpuPcieBusWidth.test_valid_bus_width) ... ok
test_threshold_exceeded (test_verify_lib.TestVerifyGpuPcieErrors.test_threshold_exceeded) ... ok
test_valid_error_metrics (test_verify_lib.TestVerifyGpuPcieErrors.test_valid_error_metrics) ... ok

----------------------------------------------------------------------
Ran 16 tests in 0.026s

OK
```

---

## 🧪 Tips for Organizing Tests

- Keep unit tests close to the code they test (e.g., `lib/unittests/`)
- Use `__init__.py` in all test directories to make them importable
- Use `unittest.mock` to isolate unit tests from external dependencies

---
