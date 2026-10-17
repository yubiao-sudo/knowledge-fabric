"""六项缺口 v1 的检索侧测试：structured / graph / OpenSearch / Agentic 及其降级分支。

覆盖契约第 0 节第 3 条「缺条件必须显式跳过并记 note」的各条路径：
  - structured 无结构化条件 → 跳过 + note，不退化成 BM25；
  - graph 库未开启 / 存储缺失 → 跳过 + note；
  - OpenSearch 未配置 / 连不上 → 回退 + note；
  - Agentic 无 gen → 只跑 1 轮，agentic_stop="no_generator"。
"""

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kf_retrieve.backends import MemoryBackend  # noqa: E402
from kf_retrieve.profile import resolve_profile  # noqa: E402
from kf_retrieve.service import SearchRequest, run_search  # noqa: E402
from kf_retrieve.structured import (  # noqa: E402
    has_conditions,
    matches_conditions,
    parse_price,
    structured_conditions,
)

_COMMON = {"tenant_id": "default", "kb_id": "demo", "status": "active", "lang": "zh"}

SEED = [
    {**_COMMON, "chunk_id": "c1", "doc_id": "d1", "page": 1,
     "content": "产品 X 的保修期为 24 个月，自购买之日起计算。"},
    {**_COMMON, "chunk_id": "c3", "doc_id": "d2", "page": 3,
     "content": "产品 X 型号 MX-200 售价 1999 元，含一年质保。",
     "skus": ["MX-200"], "price": 1999.0, "currency": "CNY"},
    {**_COMMON, "chunk_id": "c4", "doc_id": "d2", "page": 4,
     "content": "另一型号 AB-100 售价 1999 元。",
     "skus": ["AB-100"], "price": 1999.0, "currency": "CNY"},
]


def _backend(chunks=None) -> MemoryBackend:
    backend = MemoryBackend(collection="kf_default_demo")
    backend.add_chunks(chunks or SEED)
    return backend


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """清空所有外部依赖变量，保证降级分支可复现。"""
    for name in ("OPENSEARCH_URL", "DATABASE_APP_URL", "VLLM_URL", "OPENAI_BASE_URL",
                 "AGENTIC_MAX_ROUNDS"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ 1) structured
def test_structured_conditions_only_from_exact_terms():
    cond = structured_conditions({"skus": ["MX-200"], "prices": ["1999 元"]})
    assert cond == {"skus": ["MX-200"], "price": {"gte": 1999.0, "lte": 1999.0}, "currency": "CNY"}
    assert has_conditions(cond) is True
    # 没有任何 exact 词 → 不得凭空生成条件
    assert has_conditions(structured_conditions({})) is False
    assert has_conditions(structured_conditions({"skus": [], "prices": []})) is False


def test_parse_price_variants():
    assert parse_price("¥1,999.00") == (1999.0, "CNY")
    assert parse_price("$1999") == (1999.0, "USD")
    assert parse_price("1999元") == (1999.0, "CNY")
    assert parse_price("abc") == (None, None)


def test_matches_conditions_by_payload_fields():
    chunk = {"skus": ["MX-200"], "price": 1999.0, "currency": "CNY"}
    assert matches_conditions(chunk, {"skus": ["MX-200"], "price": {"gte": 1999.0, "lte": 1999.0},
                                      "currency": "CNY"}) is True
    assert matches_conditions(chunk, {"skus": ["AB-100"], "price": None, "currency": None}) is False
    assert matches_conditions({"skus": [], "price": 1999.0, "currency": "CNY"},
                              {"skus": ["MX-200"]}) is False


def test_structured_channel_skips_without_conditions():
    """查询无结构化条件 → 空 + note，且 trace 标注 skipped（不退化成 BM25）。"""
    req = SearchRequest(query="保修期是多久", profile="exact_lookup", tenant="default", kb="demo")
    body = run_search(req, _backend())
    trace = body["trace"]
    assert trace["structured"] == {"skipped": "查询无结构化条件"}
    assert any("structured 通道：查询无结构化条件，已跳过" == n for n in trace["notes"])
    # 没有结构化条件的命中：sparse 仍工作，但没有任何块被标成 structured 通道
    assert trace["channels"].get("structured", 0) == 0


def test_structured_channel_matches_payload_precisely():
    """有结构化条件 → 按 payload 精确过滤（skus 交集 + price 区间 + currency）。"""
    cond = structured_conditions({"skus": ["MX-200"], "prices": ["1999 元"]})
    hits = _backend().search(
        "structured", "MX-200 售价 1999 元",
        filters={"tenant_id": "default", "kb_id": "demo", "status": "active",
                 "structured": cond},
        top_k=10,
    )
    ids = [h["chunk"]["chunk_id"] for h in hits]
    assert ids == ["c3"], "结构化通道只能召回 SKU/价格/币种三者都命中的块"

    req = SearchRequest(query="MX-200 售价 1999 元", profile="exact_lookup",
                        tenant="default", kb="demo")
    body = run_search(req, _backend())
    trace = body["trace"]
    assert trace["structured"]["matched"] == 1
    assert trace["structured"]["filters"]["skus"] == ["MX-200"]
    assert "c3" in [e["chunk_id"] for e in body["evidence"]]


def test_structured_trace_when_profile_lacks_channel():
    req = SearchRequest(query="保修期是多久", profile="hybrid_rerank", tenant="default", kb="demo")
    body = run_search(req, _backend())
    assert body["trace"]["structured"] == {"skipped": "档位未启用 structured 通道"}


# ------------------------------------------------------------------ 2) graph
def test_memory_backend_graph_multihop_and_community():
    """MemoryBackend 同语义：实体 → 多跳扩展 → 关联块 + 社区摘要 evidence。"""
    chunks = [
        {**_COMMON, "chunk_id": "g1", "doc_id": "d1", "page": 1,
         "content": "Alpha 产品说明。",
         "graph": {"entities": ["Alpha", "Beta"], "edges": [["Alpha", "Beta"]]}},
        {**_COMMON, "chunk_id": "g2", "doc_id": "d2", "page": 2,
         "content": "Gamma 配件说明。",
         "graph": {"entities": ["Gamma"], "edges": [["Beta", "Gamma"]]}},
        {**_COMMON, "chunk_id": "g3", "doc_id": "d3", "page": 3,
         "content": "无关内容。", "graph": {"entities": ["Delta"]}},
    ]
    backend = _backend(chunks)
    res = backend.graph_search("Alpha Gamma", filters={"status": "active"}, max_hops=2)
    ids = [h["chunk"]["chunk_id"] for h in res["hits"]]
    assert "g1" in ids and "g3" not in ids

    # 社区摘要：mode=community_summary
    chips = [
        {**_COMMON, "chunk_id": "h1", "doc_id": "d1", "page": 1, "content": "Alpha 产品说明。",
         "graph": {"entities": ["Alpha"],
                   "communities": [{"id": "comm1", "summary": "关于 Alpha 的汇总。",
                                    "entities": ["Alpha"], "chunk_ids": ["h1"]}]}},
    ]
    b2 = _backend(chips)
    res2 = b2.graph_search("Alpha", filters={"status": "active"}, max_hops=1,
                           mode="community_summary")
    comm = [h for h in res2["hits"] if h["chunk"]["chunk_id"].startswith("community:")]
    assert comm and comm[0]["chunk"]["chunk_type"] == "community_summary"
    assert res2["communities_used"] == 1


def test_graph_channel_skipped_when_kb_not_enabled():
    """profile 要求 graph 但库开关关闭 → 跳过 + note。"""
    req = SearchRequest(query="Alpha", profile="graph_global", tenant="default", kb="demo")
    body = run_search(req, _backend())
    trace = body["trace"]
    assert trace["graph"]["enabled"] is False
    assert any("graph_enabled=false" in n for n in trace["notes"])


class _NoGraphBackend:
    """没有 `graph_search` 的后端：图通道只能走 PG（GraphStore）路径。"""

    def describe(self):
        return {"backend": "memory", "collection": "kf_default_demo"}

    def search(self, *args, **kwargs):
        return []

    def get_chunk(self, *args, **kwargs):
        return None


def test_graph_channel_skipped_when_store_missing(monkeypatch):
    """库已开启但 DATABASE_APP_URL 缺失 → 跳过 + note，不得报错。"""
    req = SearchRequest(query="Alpha", profile="graph_global", tenant="default", kb="demo",
                        graph_enabled=True)
    body = run_search(req, _NoGraphBackend())
    trace = body["trace"]
    assert trace["graph"]["enabled"] is True  # 通道被启用
    assert trace["graph"]["entities_matched"] == []
    assert any("Graph 存储未配置（DATABASE_APP_URL 缺失）" in n for n in trace["notes"])


def test_graph_channel_runs_via_memory_when_enabled():
    """库开启 + seed 带 graph 字段 → graph 通道真实产出实体与社区证据。"""
    chunks = [
        {**_COMMON, "chunk_id": "g1", "doc_id": "d1", "page": 1, "content": "Alpha 产品总览与规格。",
         "graph": {"entities": ["Alpha"],
                   "communities": [{"id": "comm1", "summary": "Alpha 汇总摘要。",
                                    "entities": ["Alpha"], "chunk_ids": ["g1"]}]}},
    ]
    req = SearchRequest(query="Alpha", profile="graph_global", tenant="default", kb="demo",
                        graph_enabled=True)
    body = run_search(req, _backend(chunks), gen=None)
    trace = body["trace"]
    assert trace["graph"]["enabled"] is True
    assert "alpha" in trace["graph"]["entities_matched"]
    assert trace["graph"]["communities_used"] == 1
    comm = [e for e in body["evidence"] if e["chunk_id"] == "community:comm1"]
    assert comm and comm[0]["chunk_type"] == "community_summary"
    assert comm[0]["community_chunk_ids"] == ["g1"]  # 可回溯到原文块


# ------------------------------------------------------------------ 3) OpenSearch
def test_sparse_backend_reports_memory_when_opensearch_unset():
    req = SearchRequest(query="保修期是多久", profile="hybrid_rerank", tenant="default", kb="demo")
    body = run_search(req, _backend())
    assert body["trace"]["sparse_backend"] == "memory"
    assert not any("OpenSearch" in n for n in body["trace"]["notes"])


def test_sparse_falls_back_with_note_when_opensearch_unreachable(monkeypatch):
    """配置了 OPENSEARCH_URL 但连不上 → 自动回退 + note，稀疏仍可用。"""
    monkeypatch.setenv("OPENSEARCH_URL", "http://127.0.0.1:1")
    req = SearchRequest(query="保修期是多久", profile="hybrid_rerank", tenant="default", kb="demo")
    body = run_search(req, _backend())
    trace = body["trace"]
    assert trace["sparse_backend"] == "memory"
    assert any("OpenSearch" in n for n in trace["notes"])
    assert any(e["chunk_id"] == "c1" for e in body["evidence"])


def test_opensearch_module_index_name_and_analyzer():
    from kf_retrieve.opensearch import analyzer_for_lang, build_query, index_name

    assert index_name("ac-me", "de-mo", "zh") == "kf_ac_me_de_mo_zh"
    assert analyzer_for_lang("zh") == "cjk"
    assert analyzer_for_lang("ja") == "cjk"
    assert analyzer_for_lang("en") == "english"
    assert analyzer_for_lang("ko") == "english"
    assert analyzer_for_lang("de") == "standard"
    dsl = build_query("保修期", version=2, top_k=7)["query"]["bool"]
    assert {"term": {"status": "active"}} in dsl["filter"]
    assert {"term": {"version": 2}} in dsl["filter"]


# ------------------------------------------------------------------ 4) Agentic
def test_agentic_without_generator_runs_single_round():
    """无 gen 时不得伪造多轮：只跑 1 轮，stop=no_generator，并记 note。"""
    req = SearchRequest(query="产品 X 的保修期", profile="decompose_agentic",
                        tenant="default", kb="demo")
    body = run_search(req, _backend(), gen=None)
    trace = body["trace"]
    assert trace["agentic_rounds"] == 1
    assert trace["agentic_stop"] == "no_generator"
    assert len(trace["agentic_rounds_detail"]) == 1
    assert trace["agentic_rounds_detail"][0]["round"] == 1
    assert any("Agentic 多轮跳过" in n for n in trace["notes"])


def test_agentic_stops_on_saturation_with_generator():
    """有 gen：第 2 轮无新增证据 → stop="saturation"。"""
    def gen(_prompt: str) -> str:
        return "保修政策"

    req = SearchRequest(query="产品 X 的保修期", profile="decompose_agentic",
                        tenant="default", kb="demo")
    body = run_search(req, _backend(), gen=gen)
    trace = body["trace"]
    assert trace["agentic_rounds"] == 2
    assert trace["agentic_stop"] == "saturation"
    assert trace["agentic_rounds_detail"][1]["new_chunks"] == 0


def test_agentic_profile_disabled_keeps_zero_rounds():
    req = SearchRequest(query="保修期是多久", profile="hybrid_rerank", tenant="default", kb="demo")
    body = run_search(req, _backend())
    assert body["trace"]["agentic_rounds"] == 0
    assert body["trace"]["agentic_stop"] == "disabled"


def test_agentic_hard_limit_overridable(monkeypatch):
    from kf_retrieve.agentic import global_hard_limit, round_limit

    assert global_hard_limit() == 3
    assert round_limit(5) == 3
    assert round_limit(2) == 2
    monkeypatch.setenv("AGENTIC_MAX_ROUNDS", "2")
    assert global_hard_limit() == 2
    assert round_limit(5) == 2


# ------------------------------------------------------------------ 5) profile 默认值
def test_profile_defaults_for_agentic_and_graph():
    prof = resolve_profile("hybrid_rerank")
    assert prof["min_new_evidence"] == 1
    assert prof["budget_tokens"] == 8000
    assert prof["agentic"] == {"enabled": False, "max_rounds": 3}
    assert prof["graph"]["enabled"] is False
    assert prof["graph"]["max_hops"] == 2

    gg = resolve_profile("graph_global")
    assert gg["graph"]["enabled"] is True
    assert gg["graph"]["mode"] == "community_summary"
    assert gg["graph"]["max_hops"] == 2


def test_request_graph_enabled_defaults_false():
    assert SearchRequest(query="x").graph_enabled is False
