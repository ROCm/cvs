"""Token substitution: known/unknown tokens, recursion, scalar coercion."""

import pytest

from cvs.lib.substitution import build_context, resolve_paths_block, substitute


def test_simple_token():
    assert substitute("{a}/x", {"a": "foo"}) == "foo/x"


def test_two_tokens_one_string():
    assert substitute("{a}/{b}", {"a": "x", "b": "y"}) == "x/y"


def test_recursive_resolve_via_paths_block():
    paths = {"root": "/r", "models": "{root}/m", "weights": "{models}/w"}
    out = resolve_paths_block(paths)
    assert out["weights"] == "/r/m/w"


def test_unknown_token_fails_loudly():
    with pytest.raises(ValueError, match="unknown token"):
        substitute("{missing}", {"a": "1"})


def test_non_scalar_token_rejected():
    with pytest.raises(ValueError, match="non-scalar"):
        substitute("{x}", {"x": [1, 2]})


def test_list_dict_walked():
    out = substitute(
        {"k": ["{a}", {"nested": "{b}"}]},
        {"a": "A", "b": "B"},
    )
    assert out == {"k": ["A", {"nested": "B"}]}


def test_no_token_passthrough():
    assert substitute("no tokens here", {}) == "no tokens here"


def test_user_id_token():
    ctx = build_context(paths={"shared_fs": "/x"}, run_id="r1", user_id="alice")
    assert ctx["user-id"] == "alice"
    assert substitute("/home/{user-id}", ctx) == "/home/alice"


def test_params_token():
    ctx = build_context(paths={}, run_id="r", user_id="u", params={"tp": 8})
    assert substitute("--tp {params.tp}", ctx) == "--tp 8"


def test_circular_paths_caught():
    # Either "circular" (resolve_paths_block) or "depth" (_substitute_str)
    # is acceptable — both surface the cycle.
    with pytest.raises(ValueError, match="circular|depth"):
        resolve_paths_block({"a": "{b}", "b": "{a}"})
