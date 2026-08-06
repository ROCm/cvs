'''Unit tests for Run Deck pytest session hooks.'''

import unittest
from types import SimpleNamespace

from cvs.lib.report import pytest_hooks
from cvs.lib.report.registry import clear_session_results, get_session_results
from cvs.lib.report.rundeck.config_builder import make_inference_report_config


class _FixtureLookupError(LookupError):
    pass


class TestPytestHooks(unittest.TestCase):
    def setUp(self):
        clear_session_results()

    def tearDown(self):
        clear_session_results()

    def test_cvs_rundeck_bind_module_fixture_resolves_inf_res_dict_alias(self):
        cfg = make_inference_report_config(
            suite_id="demo",
            results_columns=(),
            metric_units={},
            tier_metric_specs=lambda _c, _t: {},
        )
        request = SimpleNamespace(
            config=SimpleNamespace(_suite_report_config=cfg),
            _finalizers=[],
        )

        def fake_getfixturevalue(name):
            if name == "inf_res_dict":
                return {"k": 1}
            if name == "variant_config":
                return object()
            if name == "lifecycle":
                return SimpleNamespace(report={})
            raise _FixtureLookupError(name)

        request.getfixturevalue = fake_getfixturevalue
        request.addfinalizer = request._finalizers.append

        gen = pytest_hooks.cvs_rundeck_bind_module_fixture(request, None)
        next(gen)
        for fn in request._finalizers:
            fn()
        with self.assertRaises(StopIteration):
            next(gen)

        store = get_session_results()
        self.assertEqual(store["cvs_results_dict"], {"k": 1})


if __name__ == "__main__":
    unittest.main()
