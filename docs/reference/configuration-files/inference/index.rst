.. meta::
  :description: Index of CVS inference test configuration schemas, covering ATOM, vLLM, SGLang, and xDiT benchmark suites for AMD GPU clusters.
  :keywords: CVS, inference, vLLM, ATOM, SGLang, xDiT, ROCm, GPU, AMD Instinct, LLM, benchmark, JSON, configuration

************************************************************
Cluster Validation Suite (CVS) inference test configuration schemas
************************************************************

JSON configuration schemas for inference and serving benchmarks under ``cvs/input/config_file/inference/``.

- :doc:`ATOM </reference/configuration-files/inference/atom>` — ATOM inference benchmarks
- :doc:`vLLM Inference </reference/configuration-files/inference/vllm>` — vLLM serving throughput and latency
- :doc:`SGLang Inference </reference/configuration-files/inference/sglang>` — single-node, distributed, and disaggregated LLM serving
- :doc:`xDiT Inference </reference/configuration-files/inference/xdit>` — FLUX.1/FLUX.2 text-to-image and WAN 2.2 image-to-video

How to run these suites: :doc:`/how-to/test-suites/inference/index`.

Convert old configuration files
===============================

``cvs config migrate`` converts a suite's configuration files and threshold files from the old format to the new one:

.. code:: bash

  cvs config migrate <suite> <input.json> <output> [--force]
  cvs config migrate <suite> <input.json> [<input.json> ...] <output folder> [--force]

- ``<suite>`` is the suite the file belongs to. A suite becomes available when it moves to the new format; ``cvs config migrate --help`` lists the available suites.
- ``<input.json>`` is an old configuration or threshold file. You can pass both kinds in one run; threshold files are converted first, so a configuration is checked against its converted threshold file. The command never changes an input.
- ``<output>`` is the file to write, or a directory, where the file gets the input's name. With more than one input, it must be an existing directory, and each file is written there under its own name. An existing file is replaced only with ``--force``.

A suite in the new format refuses an old file and prints the ``cvs config migrate`` command to run.

The command prints one result line for each file, and exits with the highest status of any file:

- ``0``: it wrote the new file, or wrote nothing because the input is already in the new format.
- ``1``: it wrote the new file with fields to fix, and printed one line for each.
- ``2``: it wrote nothing, and printed why: for example, the input is not valid JSON or belongs to another suite, or the output exists and ``--force`` is not given.

Before you use a file written with exit status ``1``, fix these fields. CVS refuses to load the file until you do.

- ``_unmapped``, at the top of the file, holds each old field that has no place in the new format, under its old dotted path, such as ``"server_params.foo": 1``. Move each field to its new place or delete it, then delete the ``_unmapped`` block.
- ``"<changeme>"`` marks a required field the old file did not set. Replace it with the value for your setup.
- Any other field the command printed, such as a value of the wrong type, needs a valid value.
