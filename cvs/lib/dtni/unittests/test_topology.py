"""topology.resolve_bindings: host allocation math, overflow, ordering."""

import pytest

from cvs.lib.dtni.topology import resolve_bindings


def test_single_role_single_host():
    out = resolve_bindings(
        node_dict={"n1": {}},
        roles={"server": {"count": 1, "gpus_per_node": 8}},
    )
    assert out == {"server": ["n1"]}


def test_multi_role_partitions_distinctly():
    out = resolve_bindings(
        node_dict={"n1": {}, "n2": {}, "n3": {}},
        roles={
            "prefill": {"count": 1, "gpus_per_node": 8},
            "decode": {"count": 2, "gpus_per_node": 8},
        },
    )
    assert out == {"prefill": ["n1"], "decode": ["n2", "n3"]}


def test_overflow_rejected():
    with pytest.raises(ValueError, match="demands 3 hosts"):
        resolve_bindings(
            node_dict={"n1": {}},
            roles={
                "a": {"count": 1, "gpus_per_node": 8},
                "b": {"count": 2, "gpus_per_node": 8},
            },
        )


def test_zero_count_rejected():
    with pytest.raises(ValueError, match="count=0"):
        resolve_bindings(
            node_dict={"n1": {}},
            roles={"server": {"count": 0, "gpus_per_node": 8}},
        )


def test_node_dict_order_preserved():
    """Bindings honor insertion order of node_dict — important for reproducibility."""
    out = resolve_bindings(
        node_dict={"zeta": {}, "alpha": {}},
        roles={"server": {"count": 2, "gpus_per_node": 8}},
    )
    assert out["server"] == ["zeta", "alpha"]
