'''Unit tests for the vLLM re-exports in cvs.lib.conf_lib.schemas.vllm.'''

import unittest

from cvs.lib.conf_lib.schemas import vllm
from cvs.parsers import schemas


class TestReExports(unittest.TestCase):
    def test_old_names_are_the_parser_schema_objects(self):
        self.assertIs(vllm.VariantConfig, schemas.VllmConfig)
        for name in ("ContainerEnvTokenError", "ResolvedRun", "RunCell", "VllmConfig", "serialize_cli_options"):
            with self.subTest(name=name):
                self.assertIs(getattr(vllm, name), getattr(schemas, name))


if __name__ == "__main__":
    unittest.main()
