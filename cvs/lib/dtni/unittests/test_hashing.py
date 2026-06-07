"""workload_hash: canonicalization, order-independence, fail-loud on missing digest."""

import pytest

from cvs.lib.dtni.hashing import canonical_json, workload_hash


def test_key_order_independent():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert canonical_json(a) == canonical_json(b)


def test_whitespace_in_string_changes_hash():
    # spaces inside a value DO change the hash (they're not whitespace formatting)
    h1 = workload_hash(image_digest="sha256:abc", workload={"k": "a b"}, thresholds={})
    h2 = workload_hash(image_digest="sha256:abc", workload={"k": "ab"}, thresholds={})
    assert h1 != h2


def test_reorder_workload_keys_stable():
    h1 = workload_hash(image_digest="sha256:abc", workload={"a": 1, "b": 2}, thresholds={})
    h2 = workload_hash(image_digest="sha256:abc", workload={"b": 2, "a": 1}, thresholds={})
    assert h1 == h2


def test_image_digest_changes_hash():
    h1 = workload_hash(image_digest="sha256:aaa", workload={"k": 1}, thresholds={})
    h2 = workload_hash(image_digest="sha256:bbb", workload={"k": 1}, thresholds={})
    assert h1 != h2


def test_missing_digest_raises():
    with pytest.raises(ValueError, match="image_digest is required"):
        workload_hash(image_digest="", workload={}, thresholds={})


def test_hash_length():
    h = workload_hash(image_digest="sha256:abc", workload={"k": 1}, thresholds={})
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)
