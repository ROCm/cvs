# PyTorch Vision Training Suite (single-node and distributed)

The documentation for the PyTorch Vision **test suites** lives in the CVS docs
and is the single source of truth:

* **Running the suites (lifecycle, metrics, reports):** docs/how-to/test-suites/training/pytorch_vision.rst
* **Configuration & threshold reference:** docs/reference/configuration-files/training/pytorch_vision.rst

Suite entrypoints in this directory:

| Suite (`cvs run <name>`) | File | Use with |
|---|---|---|
| `pytorch_vision_single` | `pytorch_vision_single.py` | single-node config (`distributed: false`) |
| `pytorch_vision_distributed` | `pytorch_vision_distributed.py` | multi-node config (`distributed: true`) |

Both delegate every stage to `_common.py`; fixtures and sweep parametrization
live in `conftest.py`. Neither helper is a runnable suite.

The container image is built from `build_tools/pytorch_vision/`, because the
public `rocm/pytorch` images do not ship rocAL.
