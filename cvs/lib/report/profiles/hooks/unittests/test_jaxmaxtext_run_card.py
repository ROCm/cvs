'''Unit tests for the JAX MaxText Run Deck run card.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report.profiles.hooks.jaxmaxtext_run_card import jaxmaxtext_run_card_display


class TestJaxMaxTextRunCard(unittest.TestCase):
    def test_displays_model_gpu_arch_and_node_count(self):
        variant = SimpleNamespace(
            model=SimpleNamespace(id="llama3.1-70b"),
            gpu_arch="mi325x",
            nnodes=4,
        )

        self.assertEqual(
            jaxmaxtext_run_card_display(variant, {}),
            [
                ("Model", "llama3.1-70b", False),
                ("GPU arch", "mi325x", False),
                ("nnodes", "4", False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
