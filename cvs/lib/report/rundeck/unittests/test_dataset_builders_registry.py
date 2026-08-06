'''Unit tests for Run Deck dataset builder registry.'''

import unittest

from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets, register_dataset_builder


class TestDatasetBuildersRegistry(unittest.TestCase):
    def test_build_datasets_returns_empty_when_builder_missing(self):
        self.assertEqual(build_datasets("missing_builder_xyz", {}, {}), {})

    def test_register_dataset_builder(self):
        @register_dataset_builder("demo_ut_builder")
        def _demo_builder(sources, profile):
            return {"demo": True}

        self.assertEqual(build_datasets("demo_ut_builder", {}, {}), {"demo": True})


if __name__ == "__main__":
    unittest.main()
