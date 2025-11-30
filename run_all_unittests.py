# run_all_unittests.py
import sys
import os

# Add lib directory to sys.path for absolute imports
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
sys.path.insert(0, lib_path)

import unittest

def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all unit test directories
    for test_dir in ['lib/unittests']:
        suite.addTests(loader.discover(start_dir=test_dir))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main()