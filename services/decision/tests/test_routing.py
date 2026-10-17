"""决策路由回归测试：特征派生 + heuristic 档位路由 + 契约字段。

重点保护两件容易悄悄退化的事：
  1. `derive_features` 必须能从查询文本补出特征 —— 否则 `exact_lookup` /
     `strict_refuse` 分支永远不触发，所有查询都掉进安全默认档；
  2. 路由优先级必须是「意图优先于表面特征」—— 对比类查询即使含型号，
     也应走多跳而非单点精确查找。
"""

import pytest

from app import (
    CONFIDENCE_FLOOR,
    PROFILES,
    QueryFeatures,
    SAFE_DEFAULT,
    build_agentic,
    build_graph,
    derive_features,
    detect_lang,
    escalate_for,
    heuristic_decide,
    runtime_state,
)


def route(query: str, **overrides):
    feats = derive_features(query, QueryFeatures(**overrides))
    return heuristic_decide(query, feats), feats


# ------------------------------------------------------------------ 语言判定
@pytest.mark.parametrize("query,lang", [
    ("MX-200 的价格是多少", "zh"),
    ("What is the price of MX-200?", "en"),
    ("製品のインストール手順を教えてください", "ja"),
    ("제품 설치 방법을 알려주세요", "ko"),
])
def test_detect_lang(query, lang):
    assert detect_lang(query) == lang


# ------------------------------------------------------------------ 特征派生
def test_derives_identifier_and_price_from_text():
    _, feats = route("MX-200 的价格是多少")
    assert feats.has_code_or_identifier is True
    assert feats.has_number_or_price is True
    assert feats.length == len("MX-200 的价格是多少")


def test_derives_features_when_payload_leaves_them_empty():
    """调用方只给 query、特征全默认时，仍必须得到可用路由。"""
    decision, feats = route("MX-200 的价格是多少")
    assert decision["profile"] == "exact_lookup"
    assert feats.has_code_or_identifier


def test_explicit_features_are_not_overwritten():
    feats = derive_features("毫无特征的一句话", QueryFeatures(lang="fr", has_code_or_identifier=True))
    assert feats.lang == "fr"
    assert feats.has_code_or_identifier is True


def test_sku_pattern_matches_various_forms():
    for query in ("A1000 参数", "查 AB_12 库存", "X/200 兼容性", "看下 sku 编码"):
        _, feats = route(query)
        assert feats.has_code_or_identifier, query


# ------------------------------------------------------------------ 路由优先级
def test_price_query_routes_to_exact_lookup():
    decision, _ = route("MX-200 的价格是多少")
    assert decision["profile"] == "exact_lookup"


def test_comparison_beats_identifier_surface_feature():
    """对比类查询含型号时，意图（多跳）优先于表面特征（精确查找）。"""
    decision, feats = route("对比 MX-100 与 MX-200 的差异并说明各自价格")
    assert feats.has_code_or_identifier is True, "该查询确实含型号"
    assert decision["profile"] == "decompose_agentic", "但意图是多跳对比"


def test_aggregation_query_routes_to_agentic():
    decision, _ = route("产品有哪些型号")
    assert decision["profile"] == "decompose_agentic"


def test_short_query_routes_to_hyde():
    decision, _ = route("?")
    assert decision["profile"] == "hyde"
    assert len("?") <= 6


def test_plain_query_falls_back_to_safe_default():
    decision, _ = route("这个产品应该怎么用才不会出问题呢")
    assert decision["profile"] == SAFE_DEFAULT


def test_graph_enabled_routes_aggregation_to_graph():
    decision, _ = route("汇总所有产品的差异", graph_enabled=True)
    assert decision["profile"] == "graph_global"


def test_high_no_answer_rate_routes_to_strict_refuse():
    decision, _ = route("这个功能如何配置", historical_no_answer_rate=0.9)
    assert decision["profile"] == "strict_refuse"


# ------------------------------------------------------------------ 契约与护栏
def test_all_profiles_are_known_names():
    for query in ("MX-200 价格", "?", "对比 A 与 B", "普通问题描述一下",
                  "汇总一下", "MX-100 参数"):
        decision, _ = route(query)
        assert decision["profile"] in PROFILES, f"{query} -> {decision['profile']}"
        assert 0.0 <= decision["confidence"] <= 1.0


def test_agentic_and_graph_default_to_disabled():
    assert build_agentic("hybrid_rerank", None)["enabled"] is False
    assert build_graph("hybrid_rerank", None)["enabled"] is False


def test_escalation_mapping():
    assert escalate_for("decompose_agentic", 0.9) == "agentic"
    assert escalate_for("graph_global", 0.9) == "agentic"
    assert escalate_for("strict_refuse", 0.9) == "none"
    assert escalate_for("hybrid_rerank", 0.9) == "rerank"


def test_runtime_state_before_load_does_not_trigger_download():
    """导入模块与读取健康状态都不应触发模型下载。"""
    state = runtime_state()
    assert state["mode"] == "heuristic-fallback"
    assert "model_id" in state and "device" in state
    assert CONFIDENCE_FLOOR > 0
