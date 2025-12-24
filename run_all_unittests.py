# run_all_unittests.py
import sys

import unittest


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all unit tests recursively from the cvs package
    suite.addTests(loader.discover(start_dir=".", pattern="test_*.py"))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return 0 if successful, 1 if failed
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
