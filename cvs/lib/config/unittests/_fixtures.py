"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

# Shared fixtures for the cvs/lib/config unit tests. Not a test module (no
# ``test_`` prefix) so unittest discovery ignores it.
#
# ``BASES`` is a framework-keyed registry of minimal *valid* v2 configs. The
# generic spine suites parametrize over ``iter_bases()`` so every invariant that
# lives on base.py / loader.py / thresholds.py is re-proven for every framework;
# a framework-specific suite (frameworks/unittests/test_<fw>.py) uses
# ``make_base("<fw>")`` to probe that framework's own schema rules.
#
# Adding a framework = one entry here + one test_<fw>.py. Do NOT add an entry for
# a framework whose config class is not yet registered (@register_config).
import copy

BASES = {
    "vllm": {
        "framework": "vllm",
        "model": "meta-llama/Llama-3.1-70B",
        "topology": {"nnodes": 1},
        "container": {"image": "rocm/vllm-dev:nightly", "env": {"HF_TOKEN": "hf_secret_abc"}},
        "params": {},
    },
}


def make_base(framework="vllm", **overrides):
    """Return an independent deep copy of one framework's minimal-valid config,
    with top-level keys replaced by ``overrides``."""
    cfg = copy.deepcopy(BASES[framework])
    cfg.update(overrides)
    return cfg


def iter_bases():
    """Yield ``(framework, base)`` for every framework that ships a fixture.

    Each ``base`` is an independent deep copy, so a subTest may mutate it freely
    without bleeding state into sibling cases.
    """
    for framework, base in BASES.items():
        yield framework, copy.deepcopy(base)
