import unittest
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from cvs.lib.man_lib import find_parameters, iter_parameters, render_json, render_text


class Leaf(BaseModel):
    model_config = ConfigDict(extra="allow")

    plain: str = Field(default="hello", description="A plain string.")
    bounded: int = Field(default=4, ge=1, le=8, description="A bounded integer.")
    needed: str = Field(description="A required value.")
    aliased: str = Field(alias="32_cu_local_read", default="1650", description="A key that is not an identifier.")
    undescribed: str = Field(default="x")


class Section(BaseModel):
    model_config = ConfigDict(extra="allow")

    nested: Leaf = Field(default_factory=lambda: Leaf(needed="n"), description="A grouping section.")
    listed: List[Leaf] = Field(default_factory=list, description="A list of models.")
    mapped: Optional[Dict[str, Dict[str, Leaf]]] = Field(default=None, description="Two mappings deep.")
    toggle: bool = Field(default=False, description="A boolean.", examples=[True])


class TestIterParameters(unittest.TestCase):
    def setUp(self):
        self.params = iter_parameters(Leaf)
        self.by_path = {p.path: p for p in self.params}

    def test_reports_type_default_and_description(self):
        plain = self.by_path["plain"]
        self.assertEqual("string", plain.type_name)
        self.assertEqual("hello", plain.default)
        self.assertFalse(plain.required)
        self.assertEqual("A plain string.", plain.description)

    def test_reports_constraints(self):
        self.assertEqual([">= 1", "<= 8"], self.by_path["bounded"].constraints)

    def test_marks_required_fields(self):
        self.assertTrue(self.by_path["needed"].required)

    def test_uses_alias_as_the_documented_key(self):
        self.assertIn("32_cu_local_read", self.by_path)
        self.assertNotIn("aliased", self.by_path)

    def test_missing_description_is_empty_not_absent(self):
        self.assertEqual("", self.by_path["undescribed"].description)

    def test_prefix_is_applied(self):
        prefixed = iter_parameters(Leaf, prefix="rccl.cvs_params")
        self.assertIn("rccl.cvs_params.plain", {p.path for p in prefixed})


class TestNesting(unittest.TestCase):
    def setUp(self):
        self.params = iter_parameters(Section)
        self.paths = {p.path for p in self.params}

    def test_nested_model_is_a_section_and_recursed_into(self):
        section = next(p for p in self.params if p.path == "nested")
        self.assertTrue(section.is_section)
        self.assertIn("nested.plain", self.paths)

    def test_list_of_models_records_the_element_level(self):
        self.assertIn("listed[].plain", self.paths)

    def test_dict_of_dict_of_models_records_both_levels(self):
        self.assertIn("mapped.<key>.<key>.plain", self.paths)

    def test_container_fields_are_not_sections(self):
        listed = next(p for p in self.params if p.path == "listed")
        self.assertFalse(listed.is_section)

    def test_section_default_is_suppressed(self):
        section = next(p for p in self.params if p.path == "nested")
        self.assertIsNone(section.default)


class TestFindParameters(unittest.TestCase):
    def setUp(self):
        self.params = iter_parameters(Section)

    def test_exact_leaf_name_wins_over_substring(self):
        matches = find_parameters(self.params, "plain")
        self.assertTrue(matches)
        self.assertTrue(all(p.name == "plain" for p in matches))

    def test_falls_back_to_substring(self):
        matches = find_parameters(self.params, "bound")
        self.assertTrue(matches)
        self.assertTrue(all("bound" in p.path for p in matches))

    def test_no_match_returns_empty(self):
        self.assertEqual([], find_parameters(self.params, "nothing_matches_this"))

    def test_query_matching_a_section_name_resolves_to_its_children(self):
        # "nested" is a section (a grouping model with no value of its own), so
        # matching it exactly must not return *only* the empty section stub --
        # it should fall through to substring matching and find its leaf
        # children too, the same way an unfiltered "cvs man <test>" would.
        matches = find_parameters(self.params, "nested")
        leaf_matches = [p for p in matches if not p.is_section]
        self.assertTrue(leaf_matches, "expected leaf children of the 'nested' section, got only the section stub")
        self.assertIn("nested.plain", {p.path for p in leaf_matches})

    def test_query_matching_a_list_of_models_field_includes_its_elements(self):
        # "listed" is not a section (its own default, [], is meaningful), but
        # it is a List[Leaf] -- an exact match on it should still pull in the
        # documented element fields, not just the empty-list stub.
        matches = find_parameters(self.params, "listed")
        paths = {p.path for p in matches}
        self.assertIn("listed", paths)
        self.assertIn("listed[].plain", paths)

    def test_query_matching_a_dict_of_dict_of_models_field_includes_its_leaves(self):
        matches = find_parameters(self.params, "mapped")
        paths = {p.path for p in matches}
        self.assertIn("mapped", paths)
        self.assertIn("mapped.<key>.<key>.plain", paths)


class TestRenderText(unittest.TestCase):
    def setUp(self):
        self.text = render_text(iter_parameters(Section), title="cvs man demo", summary="A demo.")

    def test_includes_title_and_summary(self):
        self.assertIn("cvs man demo", self.text)
        self.assertIn("A demo.", self.text)

    def test_sections_are_not_repeated(self):
        # Declaration order interleaves a section's own fields with its
        # children, which previously emitted the same heading twice.
        heading = "\n  nested\n"
        self.assertEqual(1, self.text.count(heading))

    def test_counts_only_leaf_parameters(self):
        leaves = [p for p in iter_parameters(Section) if not p.is_section]
        self.assertIn(f"Total: {len(leaves)} parameters", self.text)

    def test_marks_required_and_defaults(self):
        self.assertIn("required", self.text)
        self.assertIn("default", self.text)

    def test_flags_undocumented_parameters(self):
        self.assertIn("(undocumented)", self.text)

    def test_singular_parameter_count(self):
        one = [p for p in iter_parameters(Leaf) if p.path == "plain"]
        self.assertIn("Total: 1 parameter\n", render_text(one))


class TestRenderJson(unittest.TestCase):
    def test_emits_parseable_payload(self):
        import json

        payload = json.loads(render_json(iter_parameters(Leaf), test="demo", config_files=["a.json"]))
        self.assertEqual("demo", payload["test"])
        self.assertEqual(["a.json"], payload["config_files"])

        by_path = {entry["path"]: entry for entry in payload["parameters"]}
        self.assertEqual("hello", by_path["plain"]["default"])
        self.assertTrue(by_path["needed"]["required"])
        self.assertNotIn("default", by_path["needed"])

    def test_sections_are_flagged_and_carry_no_default(self):
        import json

        payload = json.loads(render_json(iter_parameters(Section)))
        nested = next(e for e in payload["parameters"] if e["path"] == "nested")
        self.assertTrue(nested["section"])
        self.assertNotIn("default", nested)


if __name__ == "__main__":
    unittest.main()
