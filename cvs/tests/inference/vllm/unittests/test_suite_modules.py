'''Unit tests for the vLLM suite module layout.'''

import inspect
import unittest

from cvs.tests.inference.vllm import _common, _shared, vllm_distributed, vllm_single

EXPECTED_TESTS = [
    "test_launch_container",
    "test_setup_sshd",
    "test_discover_topology",
    "test_model_fetch",
    "test_openai_compatible_smoke",
    "test_vllm_inference",
    "test_accuracy_eval",
    "test_print_results_table",
    "test_teardown",
]


def _test_functions(module):
    return [
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_") and obj.__module__ == module.__name__
    ]


class TestVllmSuiteModules(unittest.TestCase):
    def test_suites_define_every_test_in_their_own_file(self):
        for module in (vllm_single, vllm_distributed):
            with self.subTest(module=module.__name__):
                self.assertEqual(sorted(_test_functions(module)), sorted(EXPECTED_TESTS))
                imported = [
                    name
                    for name, obj in vars(module).items()
                    if name.startswith("test_") and inspect.isfunction(obj) and obj.__module__ != module.__name__
                ]
                self.assertEqual(imported, [])

    def test_shared_modules_expose_no_collectable_tests(self):
        for module in (_common, _shared):
            with self.subTest(module=module.__name__):
                self.assertEqual([name for name in vars(module) if name.startswith("test_")], [])


if __name__ == "__main__":
    unittest.main()
