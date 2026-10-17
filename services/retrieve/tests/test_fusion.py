"""融合单测：RRF 数值、加权融合、sparse_boost 乘法加权。"""

import pytest

from kf_retrieve import fusion


def test_rrf_numeric_values():
    rankings = [["a", "b"], ["b", "a"]]
    scores = fusion.rrf(rankings, k=60)
    # a: 1/61 + 1/62；b: 1/61 + 1/62
    expected = 1 / 61 + 1 / 62
    assert scores["a"] == pytest.approx(expected)
    assert scores["b"] == pytest.approx(expected)


def test_rrf_rank_starts_at_one():
    scores = fusion.rrf([["solo"]], k=60)
    assert scores["solo"] == pytest.approx(1 / 61)


def test_rrf_orders_consistent_winner_first():
    # a 在两个列表都靠前，应高于只在单列表靠前的 c
    scores = fusion.rrf([["a", "c"], ["a", "b"]], k=60)
    assert fusion.ranked_ids(scores)[0] == "a"


def test_weighted_uses_weights():
    lists = {"dense": ["a"], "sparse": ["b"]}
    scores = fusion.weighted(lists, {"dense": 0.6, "sparse": 0.4}, k=60)
    assert scores["a"] == pytest.approx(0.6 * (1 / 61))
    assert scores["b"] == pytest.approx(0.4 * (1 / 61))


def test_weighted_missing_channel_weight_is_zero():
    scores = fusion.weighted({"graph": ["a"]}, {"dense": 0.6, "sparse": 0.4}, k=60)
    assert "a" not in scores


def test_combine_rrf_boost_only_sparse():
    lists = {"dense": ["d"], "sparse": ["s"]}
    plain = fusion.combine(lists, "rrf", k=60, boost=1.0)
    boosted = fusion.combine(lists, "rrf", k=60, boost=2.0)
    assert boosted["d"] == pytest.approx(plain["d"])          # dense 不受 boost 影响
    assert boosted["s"] == pytest.approx(plain["s"] * 2.0)    # sparse 乘 2


def test_combine_weighted_exact_lookup_weights():
    lists = {"dense": ["d"], "sparse": ["s"]}
    scores = fusion.combine(
        lists, "weighted", k=60, weights={"dense": 0.3, "sparse": 0.7}, boost=2.0
    )
    assert scores["d"] == pytest.approx(0.3 * (1 / 61))
    assert scores["s"] == pytest.approx(0.7 * 2.0 * (1 / 61))


def test_ranked_ids_tie_breaks_by_id():
    scores = {"b": 1.0, "a": 1.0}
    assert fusion.ranked_ids(scores) == ["a", "b"]
