"""档位单测：读自配置文件、默认值合并、safe_default 回退、融合权重与预算。"""

from kf_retrieve.config import load_retrieval_config
from kf_retrieve.profile import (
    budget_of,
    fusion_weights_for,
    profile_names,
    profiles_summary,
    resolve_profile,
)

EXPECTED = {
    "naive",
    "hybrid_rerank",
    "exact_lookup",
    "hyde",
    "multi_query",
    "decompose_agentic",
    "graph_global",
    "strict_refuse",
}


def test_profile_names_read_from_config():
    names = set(profile_names(load_retrieval_config()))
    assert names == EXPECTED


def test_global_budget_read_from_config():
    budget = budget_of()
    assert budget["max_context_tokens"] == 8000
    assert budget["max_latency_ms"] == 3000
    assert budget["max_agentic_rounds"] == 3


def test_unknown_profile_falls_back_to_safe_default():
    prof = resolve_profile("no_such_profile")
    assert prof["known"] is False
    assert prof["name"] == "hybrid_rerank"
    assert prof["requested"] == "no_such_profile"


def test_hybrid_profile_merged_defaults():
    prof = resolve_profile("hybrid_rerank")
    assert prof["fusion"] == "rrf"
    assert prof["fusion_k"] == 60
    assert prof["rerank"] is True
    # 重排窗口从 12 收到 4：cross-encoder 开销随「候选数 × 序列长度²」增长，
    # 真机 4GB 卡上 12 个候选 ×512 token 要 9.4s，4 个 ×256 token 只要 1.1s。
    # 这个断言锁住"窗口必须小"，防止有人改回大值又不测延迟。
    assert prof["rerank_top_n"] == 4
    assert prof["sparse_boost"] == 1.0          # 缺省补齐
    assert prof["query_expansion"] is None       # 缺省补齐
    assert prof["channels"] == ["dense", "sparse"]


def test_exact_lookup_weights_and_boost():
    prof = resolve_profile("exact_lookup")
    assert prof["sparse_boost"] == 2.0
    assert prof["fusion"] == "weighted"
    # structured 必须有权重：weighted 模式下权重缺失的通道会被记 0 分，
    # 那样「召回成功但不进证据」——真机回归正是踩到过这一条。
    assert prof["fusion_weights"] == {"dense": 0.3, "sparse": 0.7, "structured": 0.5}
    assert prof["filters"] == ["sku", "price_range", "spec"]


def test_weighted_profile_channels_all_have_weight():
    """凡以 weighted 融合的档位，其 channels 中每个通道都必须有权重。"""
    for name in ("exact_lookup", "hybrid_rerank"):
        prof = resolve_profile(name)
        if prof["fusion"] != "weighted":
            continue
        weights = prof["fusion_weights"]
        for channel in prof["channels"]:
            assert weights.get(channel, 0.0) > 0, f"{name}.{channel} 缺权重会被静默丢弃"


def test_naive_single_channel_no_rerank():
    prof = resolve_profile("naive")
    assert prof["channels"] == ["dense"]
    assert prof["rerank"] is False
    assert prof["fusion"] == "none"


def test_strict_refuse_threshold():
    prof = resolve_profile("strict_refuse")
    assert prof["answerability_threshold"] == 0.55
    assert prof["on_low_score"] == "refuse"


def test_agentic_and_graph_flags():
    assert resolve_profile("decompose_agentic")["agentic"]["enabled"] is True
    assert resolve_profile("graph_global")["graph"]["enabled"] is True
    assert resolve_profile("hybrid_rerank")["agentic"]["enabled"] is False


def test_fusion_weights_default_for_hybrid():
    assert fusion_weights_for("hybrid_rerank") == {"dense": 0.6, "sparse": 0.4}


def test_profiles_summary_shape():
    summary = profiles_summary()
    assert set(summary["profiles"]) == EXPECTED
    assert summary["global_budget"]["max_context_tokens"] == 8000
    assert summary["safe_default"] == "hybrid_rerank"
