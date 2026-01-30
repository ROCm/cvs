# cvs/lib/unittests/test_inference_max.py
import unittest
from cvs.lib.inference.base import textwrap_for_yml


class TestInferenceMaxLib(unittest.TestCase):
    def test_textwrap_for_yml(self):
        msg_string = "  line1\n    line2\n  line3"
        result = textwrap_for_yml(msg_string)
        expected = "line1\nline2\nline3"
        self.assertEqual(result, expected)

    def test_textwrap_for_yml_empty(self):
        msg_string = ""
        result = textwrap_for_yml(msg_string)
        self.assertEqual(result, "")

    def test_textwrap_for_yml_no_leading(self):
        msg_string = "line1\nline2"
        result = textwrap_for_yml(msg_string)
        self.assertEqual(result, "line1\nline2")


if __name__ == '__main__':
    unittest.main()
