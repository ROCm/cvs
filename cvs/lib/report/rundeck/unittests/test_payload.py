'''Tests for the identity fields every new Run Deck records: suite_stem, area, and generated_at_utc.'''

import unittest
from datetime import datetime

from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.payload import GENERATED_AT_UTC_FORMAT, build_rundeck_payload
from cvs.lib.report.testing.fixtures import generic_sweep_profile, generic_variant, two_cell_inf_res


def _sweep_store(**extra):
    store = {"inf_res_dict": two_cell_inf_res(), "variant_config": generic_variant(), "lifecycle_report": {}}
    store.update(extra)
    return store


class TestDeckIdentity(unittest.TestCase):
    def test_new_decks_record_their_stem_area_and_utc_time(self):
        profile = dict(generic_sweep_profile(), area="Inference")
        payload = build_rundeck_payload(profile=profile, store=_sweep_store(suite_stem="vllm_single"), cvs_version="1")
        self.assertEqual(payload["suite_stem"], "vllm_single")
        self.assertEqual(payload["area"], "Inference")
        generated = datetime.strptime(payload["generated_at_utc"], GENERATED_AT_UTC_FORMAT)
        self.assertEqual(generated.strftime("%Y-%m-%d %H:%M UTC"), payload["generated_at"])

    def test_child_profile_inherits_the_area_of_its_base(self):
        store = {"cvs_results_dict": {"groups": {"g": {"nodes": {"n1": {"status": "pass"}}}}}, "suite_stem": "rvs_cvs"}
        payload = build_rundeck_payload(profile=load_json_profile("rvs_cvs"), store=store)
        self.assertEqual((payload["suite_stem"], payload["area"]), ("rvs_cvs", "Health"))

    def test_unknown_stem_and_area_are_left_out(self):
        payload = build_rundeck_payload(profile=generic_sweep_profile(), store=_sweep_store(), cvs_version="1")
        self.assertNotIn("suite_stem", payload)
        self.assertNotIn("area", payload)
        self.assertIn("generated_at_utc", payload)


if __name__ == "__main__":
    unittest.main()
