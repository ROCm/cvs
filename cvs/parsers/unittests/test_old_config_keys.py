import logging
import unittest
import warnings

from cvs.parsers.schemas import (
    OLD_KEY_DROPPED,
    OLD_KEY_UNMAPPED,
    OldKey,
    normalize_legacy_preflight_node_smoke_sections,
    normalize_legacy_preflight_rdma_config,
    resolve_old_keys,
    warn_old_keys,
)


def _logged_messages(action):
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("cvs.parsers.schemas")
    logger.addHandler(handler)
    try:
        action()
    finally:
        logger.removeHandler(handler)
    return [record.getMessage() for record in records]


def _switch_rule(config):
    if config.get("switch"):
        return {"switch": OLD_KEY_DROPPED, "x": "new.x"}
    return {"switch": OLD_KEY_DROPPED, "x": OLD_KEY_DROPPED}


def _total_rule(config):
    total = config.get("polls", 2) * config.get("wait_s", 5)
    return {"polls": OldKey("timeouts.total_s", value=total), "wait_s": OLD_KEY_DROPPED}


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
        self.assertEqual(converted, {"b": 1})
        self.assertEqual(message.count("a -> b"), 1)
        self.assertIn("s.d (dropped)", message)
        self.assertEqual(config, {"a": 1, "s": {"d": 2}})

    def test_no_old_keys_return_the_input_and_no_message(self):
        config = {"b": 1}
        converted, message = resolve_old_keys(config, {"a": "b", "x.y": OLD_KEY_DROPPED}, rules=(_switch_rule,))
        self.assertIs(converted, config)
        self.assertIsNone(message)
        self.assertEqual(resolve_old_keys([1], {"a": "b"}), ([1], None))

    def test_a_rule_moves_x_when_the_switch_is_on_and_drops_it_otherwise(self):
        on, on_message = resolve_old_keys({"switch": True, "x": 1}, {}, rules=(_switch_rule,))
        off, _ = resolve_old_keys({"switch": False, "x": 1}, {}, rules=(_switch_rule,))
        missing, _ = resolve_old_keys({"x": 1}, {}, rules=(_switch_rule,))
        self.assertEqual(on, {"new": {"x": 1}})
        self.assertIn("switch (dropped)", on_message)
        self.assertEqual(off, {})
        self.assertEqual(missing, {})

    def test_a_rule_computes_one_value_from_two_old_keys_even_when_one_is_absent(self):
        both, message = resolve_old_keys({"polls": 3, "wait_s": 10}, {}, rules=(_total_rule,))
        wait_only, _ = resolve_old_keys({"wait_s": 10}, {}, rules=(_total_rule,))
        self.assertEqual(both, {"timeouts": {"total_s": 30}})
        self.assertIn("polls -> timeouts.total_s", message)
        self.assertIn("wait_s (dropped)", message)
        self.assertEqual(wait_only, {"timeouts": {"total_s": 20}})

    def test_a_computed_value_is_not_written_when_no_old_key_is_present(self):
        config = {"timeouts": {"total_s": 9}}
        self.assertEqual(resolve_old_keys(config, {}, rules=(_total_rule,)), (config, None))

    def test_a_computed_value_is_copied_not_shared_with_the_input(self):
        config = {"a": [1]}
        converted, _ = resolve_old_keys(config, {}, rules=(lambda c: {"a": OldKey("b", value=c["a"])},))
        converted["b"].append(2)
        self.assertEqual(config, {"a": [1]})

    def test_rows_may_drop_a_parent_and_move_its_child(self):
        config = {"sp": {"tp": 8, "old": 1}}
        converted, _ = resolve_old_keys(config, {"sp": OLD_KEY_DROPPED, "sp.tp": "server.tp"})
        self.assertEqual(converted, {"server": {"tp": 8}})

    def test_chained_rows_read_every_value_before_moving_any(self):
        converted, _ = resolve_old_keys({"a": 1, "b": 2}, {"a": "b", "b": "c"})
        self.assertEqual(converted, {"b": 1, "c": 2})

    def test_emptied_parent_dropped(self):
        converted, _ = resolve_old_keys({"old": {"a": 1}, "keep": {}}, {"old.a": "new.a"})
        self.assertEqual(converted, {"new": {"a": 1}, "keep": {}})
        converted, _ = resolve_old_keys({"s": {"t": {"a": 1}}}, {"s.t.a": OLD_KEY_DROPPED})
        self.assertEqual(converted, {})
        converted, _ = resolve_old_keys({"s": {"a": 1, "b": 2}}, {"s.a": "s.c"})
        self.assertEqual(converted, {"s": {"b": 2, "c": 1}})
        rule = (lambda c: {"s.x": OldKey("n", value=5)},)
        converted, _ = resolve_old_keys({"old": 1, "s": {}}, {"old": OLD_KEY_DROPPED}, rules=rule)
        self.assertEqual(converted, {"s": {}, "n": 5})
        converted, _ = resolve_old_keys({"old": {"a": 1}}, {"old.a": "new.a"}, drop_emptied=False)
        self.assertEqual(converted, {"old": {}, "new": {"a": 1}})

    def test_dotted_path_matching_nothing_fails(self):
        config = {"accuracy": {"gsm8k.flex": {"kind": "min"}}}
        with self.assertRaisesRegex(ValueError, "accuracy.gsm8k.flex.kind matches nothing; write it as a tuple"):
            resolve_old_keys(config, {"accuracy.gsm8k.flex.kind": "x"})
        with self.assertRaisesRegex(ValueError, "matches nothing"):
            resolve_old_keys(config, {}, rules=(lambda c: {"accuracy.gsm8k.flex": OLD_KEY_DROPPED},))
        self.assertEqual(resolve_old_keys(config, {"accuracy.other.kind": "x"}), (config, None))
        shadowed = {"a": {"b": {}, "b.c": {"d": 1}}}
        with self.assertRaisesRegex(ValueError, "a.b.c.d matches nothing"):
            resolve_old_keys(shadowed, {"a.b.c.d": "x"})
        converted, _ = resolve_old_keys({"a": {"b": {"c": 1}, "b.c": 2}}, {"a.b.c": "x"})
        self.assertEqual(converted, {"a": {"b.c": 2}, "x": 1})
        self.assertEqual(
            resolve_old_keys(config, {"accuracy.gsm8k.flex.kind": "x"}, check_dotted=False), (config, None)
        )

    def test_a_tuple_path_names_a_key_that_holds_a_dot(self):
        converted, message = resolve_old_keys(
            {"accuracy": {"gsm8k.flex": 1}}, {("accuracy", "gsm8k.flex"): ("accuracy", "lm_eval", "gsm8k.flex")}
        )
        self.assertEqual(converted, {"accuracy": {"lm_eval": {"gsm8k.flex": 1}}})
        self.assertIn("accuracy.gsm8k.flex -> accuracy.lm_eval.gsm8k.flex", message)

    def test_an_unmapped_key_fails_naming_the_key_and_never_its_value(self):
        with self.assertRaises(ValueError) as caught:
            resolve_old_keys({"backend": "FAKE_VALUE"}, {"backend": OLD_KEY_UNMAPPED}, prefix="suite.")
        self.assertIn("suite.backend", str(caught.exception))
        self.assertNotIn("FAKE_VALUE", str(caught.exception))

    def test_two_different_rows_for_one_path_fail(self):
        with self.assertRaisesRegex(ValueError, "rows disagree for a"):
            resolve_old_keys({"a": 1}, {"a": "b"}, rules=(lambda c: {"a": OLD_KEY_DROPPED},))
        converted, _ = resolve_old_keys({"a": 1}, {"a": "b"}, rules=(lambda c: {"a": "b"},))
        self.assertEqual(converted, {"b": 1})

    def test_a_row_of_unknown_kind_fails_naming_its_path(self):
        with self.assertRaisesRegex(ValueError, "row for a is not"):
            resolve_old_keys({"a": 1}, {"a": None})
        with self.assertRaisesRegex(ValueError, "Unknown old-key conflict rule"):
            OldKey("b", conflict="keep")

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

    def test_converting_old_keys_logs_nothing(self):
        self.assertEqual(_logged_messages(lambda: resolve_old_keys({"a": 1}, {"a": "b"})), [])

    def test_the_message_template_and_prefix_are_used(self):
        _, message = resolve_old_keys({"a": 1}, {"a": "b"}, message='Old: {keys}; e.g. {"b": 1}.', prefix="p.")
        self.assertEqual(message, 'Old: p.a -> p.b; e.g. {"b": 1}.')


class TestWarnOldKeys(unittest.TestCase):
    def test_one_future_warning_and_one_log_line_carry_the_same_text(self):
        with self.assertLogs("cvs.parsers.schemas", level="WARNING") as logs:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warn_old_keys("Deprecated config keys: a -> b.")
        self.assertEqual(
            [(w.category, str(w.message)) for w in caught], [(FutureWarning, "Deprecated config keys: a -> b.")]
        )
        self.assertEqual([r.getMessage() for r in logs.records], ["Deprecated config keys: a -> b."])

    def test_no_message_neither_warns_nor_logs(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            logged = _logged_messages(lambda: warn_old_keys(None))
        self.assertEqual(caught, [])
        self.assertEqual(logged, [])


class TestPreflightOldKeys(unittest.TestCase):
    def test_rdma_keys_move_with_the_preflight_warning_text(self):
        converted, message = normalize_legacy_preflight_rdma_config(
            {"node_check": {"rdma_interfaces": ["a"], "gid_index": "3"}}
        )
        self.assertEqual(
            converted, {"node_check": {}, "connectivity_check": {"rdma": {"gid_index": "3", "interfaces": ["a"]}}}
        )
        self.assertEqual(
            message,
            "Deprecated preflight RDMA configuration detected: "
            "preflight.node_check.gid_index -> preflight.connectivity_check.rdma.gid_index, "
            "preflight.node_check.rdma_interfaces -> preflight.connectivity_check.rdma.interfaces. "
            "Use the preflight.connectivity_check.rdma paths; legacy paths will be removed in a future release.",
        )

    def test_rdma_without_old_keys_returns_the_input(self):
        config = {"node_check": {"enabled": True}}
        self.assertEqual(normalize_legacy_preflight_rdma_config(config), (config, None))
        self.assertIs(normalize_legacy_preflight_rdma_config(config)[0], config)

    def test_rdma_ignores_a_dotted_key(self):
        config = {"node_check.gid_index": "3"}
        self.assertEqual(normalize_legacy_preflight_rdma_config(config), (config, None))

    def test_rdma_fails_when_the_target_block_is_not_an_object(self):
        with self.assertRaisesRegex(ValueError, r"^preflight\.connectivity_check\.rdma must be an object when"):
            normalize_legacy_preflight_rdma_config(
                {"node_check": {"gid_index": "3"}, "connectivity_check": {"rdma": 1}}
            )

    def test_node_smoke_copies_a_block_and_keeps_the_old_name(self):
        converted, message = normalize_legacy_preflight_node_smoke_sections(
            {"node_smoke": {"a": 1}, "node_smoke_tier1": {"a": None}}
        )
        self.assertEqual(converted, {"node_smoke": {"a": 1}, "node_smoke_tier1": {"a": 1}})
        self.assertEqual(
            message,
            "Deprecated preflight Node Smoke section name(s) detected: preflight.node_smoke -> "
            "preflight.node_smoke_tier1. Prefer node_smoke_tier1 and node_smoke_tier3 in new configs.",
        )

    def test_preflight_converters_leave_logging_to_their_caller(self):
        def convert():
            normalize_legacy_preflight_rdma_config({"node_check": {"gid_index": "3"}})
            normalize_legacy_preflight_node_smoke_sections({"node_smoke": {"a": 1}})

        self.assertEqual(_logged_messages(convert), [])

    def test_node_smoke_leaves_a_non_object_block_and_returns_the_input(self):
        config = {"tier3_info": "x"}
        converted, message = normalize_legacy_preflight_node_smoke_sections(config)
        self.assertIs(converted, config)
        self.assertIsNone(message)


if __name__ == "__main__":
    unittest.main()
