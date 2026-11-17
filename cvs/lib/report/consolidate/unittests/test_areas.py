'''Tests for the consolidated report's area table.'''

import json
import unittest
from pathlib import Path

import cvs.lib.report as report_package
from cvs.lib.report.consolidate.areas import (
    AREA_BY_SUITE_ID,
    AREAS,
    UNCATEGORIZED,
    ordered_areas,
    resolve_area,
    suite_id_for_stem,
)

# Shared base profiles that other profiles extend; no deck reports them as its own suite_id.
BASE_PROFILE_SUITE_IDS = {"anc", "arc", "health_status"}


class TestAreaTable(unittest.TestCase):
    def test_every_profile_suite_id_has_an_area(self):
        profiles = Path(report_package.__file__).with_name("profiles")
        checked = 0
        for path in sorted(profiles.glob("*.json")):
            suite_id = json.loads(path.read_text(encoding="utf-8")).get("suite_id")
            if suite_id is None or suite_id in BASE_PROFILE_SUITE_IDS:
                continue
            checked += 1
            with self.subTest(profile=path.name):
                self.assertIn(AREA_BY_SUITE_ID.get(suite_id), AREAS)
        self.assertGreater(checked, 30)

    def test_suites_from_appendix_b_map_to_their_area(self):
        self.assertEqual(AREA_BY_SUITE_ID["rvs_cvs"], "Burn-in / Diag")
        self.assertEqual(AREA_BY_SUITE_ID["arc_test_disruptive_flows"], "Burn-in / Diag")
        self.assertEqual(AREA_BY_SUITE_ID["mori_benchmark_test"], "Network")
        self.assertEqual(AREA_BY_SUITE_ID["jaxmaxtext"], "Training")
        self.assertEqual(AREA_BY_SUITE_ID["xdit"], "Inference")


class TestSuiteIdForStem(unittest.TestCase):
    def test_variant_stems_resolve_to_their_suite_id(self):
        self.assertEqual(suite_id_for_stem("rccl_perf"), "rccl")
        self.assertEqual(suite_id_for_stem("sglang_disagg_distributed"), "sglang")
        self.assertEqual(suite_id_for_stem("xdit_wan22_14b_diffusers_single"), "xdit")

    def test_exact_suite_id_wins_over_a_shorter_prefix(self):
        self.assertEqual(suite_id_for_stem("arc_test_disruptive"), "arc_test_disruptive")
        self.assertEqual(suite_id_for_stem("arc_test_disruptive_flows"), "arc_test_disruptive_flows")

    def test_unknown_or_empty_stem(self):
        self.assertIsNone(suite_id_for_stem("my_new_suite"))
        self.assertIsNone(suite_id_for_stem(""))
        self.assertIsNone(suite_id_for_stem(None))


class TestResolveArea(unittest.TestCase):
    def test_declared_area_wins(self):
        self.assertEqual(resolve_area({"suite_id": "rvs_cvs", "area": " Health "}), "Health")

    def test_older_deck_uses_the_table(self):
        self.assertEqual(resolve_area({"suite_id": "vllm"}, "vllm_single"), "Inference")

    def test_unknown_suite_id_is_not_guessed_from_the_stem(self):
        self.assertIsNone(resolve_area({"suite_id": "brand_new"}, "rccl_perf"))

    def test_unreadable_deck_uses_its_stem(self):
        self.assertEqual(resolve_area(None, "rccl_pairwise"), "Network")
        self.assertIsNone(resolve_area(None, "unknown_stem"))


class TestOrderedAreas(unittest.TestCase):
    def test_standard_areas_then_declared_ones_then_uncategorized(self):
        used = ["Network", UNCATEGORIZED, "Health", "Inference", "Health"]
        self.assertEqual(ordered_areas(used), list(AREAS) + ["Health", UNCATEGORIZED])

    def test_uncategorized_only_when_used(self):
        self.assertEqual(ordered_areas(["Training"]), list(AREAS))


if __name__ == "__main__":
    unittest.main()
