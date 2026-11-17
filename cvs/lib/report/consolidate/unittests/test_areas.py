'''Tests for the consolidated report's area table.'''

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
from cvs.lib.report.profile import load_json_profile

# Every suite_id that wrote a Run Deck in releases 0.2.0 to 0.3.0, before decks recorded their own area.
RELEASED_SUITE_IDS = (
    "agfhc_cvs",
    "csp_qual_agfhc",
    "rvs_cvs",
    "transferbench_cvs",
    "host_configs_cvs",
    "preflight_checks",
    "anc_test_babel",
    "anc_test_basic",
    "anc_test_computerocker",
    "anc_test_cpu",
    "anc_test_gemm",
    "anc_test_gpu",
    "anc_test_memrocker",
    "anc_test_oblex",
    "anc_test_pcie",
    "anc_test_ualink",
    "anc_test_xgmi",
    "arc_test_anc",
    "arc_test_diag",
    "arc_test_disruptive",
    "arc_test_disruptive_flows",
    "arc_test_examples",
    "arc_test_flows",
    "arc_test_telemetry",
    "arc_test_workload",
    "rccl",
    "ib_perf_bw_test",
    "mori_benchmark_test",
    "aorta",
    "jaxmaxtext",
    "megatron",
    "torchtitan",
    "atom",
    "sglang",
    "vllm",
    "xdit",
)


def _deck_profiles():
    profiles = Path(report_package.__file__).with_name("profiles")
    return {
        path.stem: load_json_profile(path.stem) for path in sorted(profiles.glob("*.json")) if path.stem != "schema"
    }


class TestAreaTable(unittest.TestCase):
    def test_every_deck_profile_declares_an_area(self):
        profiles = _deck_profiles()
        self.assertGreater(len(profiles), 30)
        for stem, profile in profiles.items():
            with self.subTest(profile=stem):
                self.assertIn(profile.get("area"), AREAS)

    def test_table_covers_exactly_the_suites_that_wrote_decks_before_decks_had_an_area(self):
        self.assertEqual(set(AREA_BY_SUITE_ID), set(RELEASED_SUITE_IDS))
        for suite_id in RELEASED_SUITE_IDS:
            with self.subTest(suite_id=suite_id):
                self.assertIn(AREA_BY_SUITE_ID[suite_id], AREAS)

    def test_older_and_newer_decks_of_a_suite_land_in_the_same_area(self):
        for stem, profile in _deck_profiles().items():
            if profile["suite_id"] in AREA_BY_SUITE_ID:
                with self.subTest(profile=stem):
                    self.assertEqual(profile["area"], AREA_BY_SUITE_ID[profile["suite_id"]])


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
