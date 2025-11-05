# run_all_unittests.py
import unittest

loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add all unit test directories
for test_dir in ['lib/unittests']:
    suite.addTests(loader.discover(start_dir=test_dir))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)