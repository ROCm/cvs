'''Unit tests for cvs.lib.inference.atom.atom_quant_parity.'''

import unittest

from cvs.lib.inference.atom.atom_quant_parity import (
    compare_quant_fingerprints,
    completion_fingerprint,
    run_quant_parity_probe,
)


class TestAtomQuantParity(unittest.TestCase):
    def test_run_quant_parity_probe(self):
        out = run_quant_parity_probe(probe_text="Paris")
        self.assertEqual(out["quant_parity.probe_chars"], 5)
        self.assertEqual(out["quant_parity.probe_sha256"], completion_fingerprint("Paris"))

    def test_compare_quant_fingerprints_match(self):
        cur = run_quant_parity_probe(probe_text="same")
        ref = run_quant_parity_probe(probe_text="same")
        cmp = compare_quant_fingerprints(cur, ref)
        self.assertEqual(cmp["quant_parity.probe_match"], 1.0)

    def test_compare_quant_fingerprints_mismatch(self):
        cur = run_quant_parity_probe(probe_text="a")
        ref = run_quant_parity_probe(probe_text="b")
        cmp = compare_quant_fingerprints(cur, ref)
        self.assertEqual(cmp["quant_parity.probe_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
