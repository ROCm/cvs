# run_all_unittests.py
import sys
import os

import unittest

def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all unit test directories - look in installed package location
    import cvs.lib.unittests
    test_dir = os.path.dirname(cvs.lib.unittests.__file__)
    suite.addTests(loader.discover(start_dir=test_dir))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return 0 if successful, 1 if failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(main())
