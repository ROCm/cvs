import tempfile
import unittest
from pathlib import Path

from cvs.core.agent.lifecycle import Rank0State, Rank0StateStore


class TestRank0StateStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Rank0StateStore(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_update_and_get_state(self):
        self.store.update_state(Rank0State.STARTING)
        self.assertEqual(self.store.get_state(), Rank0State.STARTING)

        self.store.update_state(Rank0State.RUNNING)
        self.assertEqual(self.store.get_state(), Rank0State.RUNNING)

    def test_get_state_rejects_unknown_value(self):
        self.store.path.write_text("UNKNOWN\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "invalid rank-0 state"):
            self.store.get_state()


if __name__ == "__main__":
    unittest.main()
