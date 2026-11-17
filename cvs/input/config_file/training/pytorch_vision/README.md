# PyTorch Vision Training — Config and Threshold Files

The documentation for the PyTorch Vision **configuration and threshold files**
lives in the CVS docs and is the single source of truth:

* **Configuration & threshold reference:** docs/reference/configuration-files/training/pytorch_vision.rst
* **Running the suites (lifecycle, metrics, reports):** docs/how-to/test-suites/training/pytorch_vision.rst

The config files and their sibling `_threshold.json` files live in this
directory, named
`<gpu>_pytorch_vision_<model>_<mode>_<profile>_config.json`.

Replace `<changeme-imagenet-host-path>` with an ImageNet root laid out as
`train/<label>/*.JPEG` and `val/<label>/*.JPEG` before use. For distributed
runs that path must resolve identically on every node.
