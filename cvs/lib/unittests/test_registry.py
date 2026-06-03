"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import unittest


class TestImportSmoke(unittest.TestCase):
    """G5a surface: the three contract modules must import cleanly."""

    def test_imports_resolve(self) -> None:
        from cvs.lib.adapter_protocol import WorkloadAdapter  # noqa: F401
        from cvs.lib.base_adapter import BaseWorkloadAdapter  # noqa: F401
        from cvs.lib.registry import get_adapter, register_adapter  # noqa: F401


class TestRegistry(unittest.TestCase):
    """G5a registry contract: dup-reject and unknown-reject."""

    def setUp(self) -> None:
        # Isolate from any global registry state populated by adapter imports.
        from cvs.lib import registry

        self._saved_inference = dict(registry.INFERENCE_REGISTRY)
        self._saved_training = dict(registry.TRAINING_REGISTRY)
        registry.INFERENCE_REGISTRY.clear()
        registry.TRAINING_REGISTRY.clear()

    def tearDown(self) -> None:
        from cvs.lib import registry

        registry.INFERENCE_REGISTRY.clear()
        registry.INFERENCE_REGISTRY.update(self._saved_inference)
        registry.TRAINING_REGISTRY.clear()
        registry.TRAINING_REGISTRY.update(self._saved_training)

    def test_register_duplicate_framework_raises(self) -> None:
        from cvs.lib.registry import register_adapter

        @register_adapter("vllm")
        class _First:
            pass

        with self.assertRaises(ValueError):

            @register_adapter("vllm")
            class _Second:
                pass

    def test_get_adapter_unknown_framework_raises(self) -> None:
        from cvs.lib.registry import get_adapter

        with self.assertRaises(ValueError):
            get_adapter("nonexistent_framework_xyz")


if __name__ == "__main__":
    unittest.main()
