import unittest

from cvs.parsers.schemas import OLD_KEY_DROPPED, OLD_KEY_UNMAPPED, OldKey, resolve_old_keys


def _switch_rule(config):
    if config.get("switch"):
        return {"switch": OLD_KEY_DROPPED, "x": "new.x"}
    return {"switch": OLD_KEY_DROPPED, "x": OLD_KEY_DROPPED}


class TestResolveOldKeys(unittest.TestCase):
    def test_a_moved_key_lands_at_its_new_path_and_is_named(self):
        converted, message = resolve_old_keys({"a": 1}, {"a": "b.c"})
        self.assertEqual(converted, {"b": {"c": 1}})
        self.assertIn("a -> b.c", message)

    def test_a_dropped_key_is_removed_and_named(self):
        converted, message = resolve_old_keys({"a": 1, "keep": 2}, {"a": OLD_KEY_DROPPED})
        self.assertEqual(converted, {"keep": 2})
        self.assertIn("a (dropped)", message)

    def test_two_old_keys_give_one_message_naming_both(self):
        config = {"a": 1, "s": {"d": 2}}
        converted, message = resolve_old_keys(config, {"a": "b", "s.d": OLD_KEY_DROPPED})
        self.assertEqual(converted, {"b": 1, "s": {}})
        self.assertEqual(message.count("a -> b"), 1)
        self.assertIn("s.d (dropped)", message)
        self.assertEqual(config, {"a": 1, "s": {"d": 2}})

    def test_no_old_keys_return_the_input_and_no_message(self):
        config = {"b": 1}
        converted, message = resolve_old_keys(config, {"a": "b", "x.y": OLD_KEY_DROPPED})
        self.assertIs(converted, config)
        self.assertIsNone(message)

    def test_a_rule_moves_x_when_the_switch_is_on_and_drops_it_otherwise(self):
        table = {("switch", "x"): _switch_rule}
        on, on_message = resolve_old_keys({"switch": True, "x": 1}, table)
        off, _ = resolve_old_keys({"switch": False, "x": 1}, table)
        missing, _ = resolve_old_keys({"x": 1}, table)
        self.assertEqual(on, {"new": {"x": 1}})
        self.assertIn("switch (dropped)", on_message)
        self.assertEqual(off, {})
        self.assertEqual(missing, {})

    def test_a_rule_computes_one_value_from_two_old_keys(self):
        def total(config):
            return {
                "polls": OldKey("timeouts.total_s", value=config["polls"] * config.get("wait_s", 5)),
                "wait_s": OLD_KEY_DROPPED,
            }

        converted, message = resolve_old_keys({"polls": 3, "wait_s": 10}, {("polls", "wait_s"): total})
        self.assertEqual(converted, {"timeouts": {"total_s": 30}})
        self.assertIn("polls -> timeouts.total_s", message)
        self.assertIn("wait_s (dropped)", message)

    def test_an_unmapped_key_fails_naming_the_key_and_never_its_value(self):
        with self.assertRaises(ValueError) as caught:
            resolve_old_keys({"backend": "FAKE_VALUE"}, {"backend": OLD_KEY_UNMAPPED}, prefix="suite.")
        self.assertIn("suite.backend", str(caught.exception))
        self.assertNotIn("FAKE_VALUE", str(caught.exception))

    def test_a_different_value_at_the_new_path_fails_naming_both_paths(self):
        with self.assertRaisesRegex(ValueError, "Conflicting x options: p.a and p.b"):
            resolve_old_keys({"a": 1, "b": 2}, {"a": "b"}, label="x", prefix="p.")
        converted, _ = resolve_old_keys({"a": 1, "b": 1}, {"a": "b"})
        self.assertEqual(converted, {"b": 1})

    def test_a_replace_copy_overwrites_the_new_path_and_keeps_the_old_key(self):
        converted, _ = resolve_old_keys(
            {"a": {"k": 1}, "b": {"k": None}}, {"a": OldKey("b", keep_old=True, conflict="replace")}
        )
        self.assertEqual(converted, {"a": {"k": 1}, "b": {"k": 1}})
        self.assertIsNot(converted["a"], converted["b"])

    def test_a_new_path_through_a_non_object_fails(self):
        with self.assertRaisesRegex(ValueError, "b must be an object to hold b.c"):
            resolve_old_keys({"a": 1, "b": 2}, {"a": "b.c"})

    def test_the_message_template_and_prefix_are_used(self):
        _, message = resolve_old_keys({"a": 1}, {"a": "b"}, message="Old: {keys}.", prefix="p.")
        self.assertEqual(message, "Old: p.a -> p.b.")


if __name__ == "__main__":
    unittest.main()
