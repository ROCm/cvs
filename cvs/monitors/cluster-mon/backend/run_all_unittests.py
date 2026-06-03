"""
Discover and run the cluster-mon backend unit tests.

Mirrors the repo's run_all_unittests.py, but is rooted at ``backend/`` so the
top-level ``app`` package (and the per-module ``app/**/unittests`` packages)
import correctly. cluster-mon is a standalone app whose ``app.*`` imports cannot
be resolved from the repo root, hence this dedicated runner.

Usage (from backend/): python run_all_unittests.py
"""

import os
import sys
import unittest


def main() -> int:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(backend_dir, "app"),
        pattern="test_*.py",
        top_level_dir=backend_dir,
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
