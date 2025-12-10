# cvs/lib/unittests/test_jax_training_lib.py
import unittest
import cvs.lib.jax_training_lib as jax_training_lib


class TestJaxTrainingLib(unittest.TestCase):
    def test_parse_training_output(self):
        # Mock output with TFLOPS
        output = "TFLOPS/s/device: 1.5"
        result = jax_training_lib.parse_training_output(output)
        self.assertIsInstance(result, dict)
        # Assuming it parses and returns a dict


if __name__ == '__main__':
    unittest.main()
