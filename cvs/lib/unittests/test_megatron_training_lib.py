# cvs/lib/unittests/test_megatron_training_lib.py
import unittest
from unittest.mock import MagicMock
import cvs.lib.megatron_training_lib as megatron_training_lib


class TestMegatronTrainingLib(unittest.TestCase):
    def setUp(self):
        self.mock_phdl = MagicMock()

    def test_launch_training(self):
        # Test launch_training with unused out_dict removed
        megatron_training_lib.launch_training(self.mock_phdl, {})
        self.mock_phdl.exec_cmd_list.assert_called()


if __name__ == '__main__':
    unittest.main()
