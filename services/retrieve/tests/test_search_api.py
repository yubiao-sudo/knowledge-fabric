"""端到端单测：FastAPI TestClient + MemoryBackend 打 /health /v1/profiles /v1/search。

全部不依赖 Qdrant / TEI / vLLM：MemoryBackend 用稀疏 BM25，重排不可用则复用融合分。
"""

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from kf_retrieve.app import app, reset_backend, set_backend  # noqa: E402
from kf_retrieve.backends import MemoryBackend  # noqa: E402

_COMMON = {
    "tenant_id": "default",
    "kb_id": "demo",
    "status": "active",
    "source_uri": "minio://kf-raw/doc.pdf",
    "lang": "zh",
    "chunk_type": "text",
}

SEED = [
    {**_COMMON, "chunk_id": "p1", "parent_chunk_id": None, "doc_id": "d1", "page": 1,
     "content": "【父块】本手册介绍产品 X 的规格与保修政策，保修期为 24 个月。"},
    {**_COMMON, "chunk_id": "c1", "parent_chunk_id": "p1", "doc_id": "d1", "page": 1,
     "content": "产品 X 的保修期为 24 个月，自购买之日起计算。",
     "section_path": ["1.2 产品概述"], "char_start": 120, "char_end": 260},
    {**_COMMON, "chunk_id": "c2", "parent_chunk_id": "p1", "doc_id": "d1", "page": 1,
     "content": "更换零部件不额外收费，具体以门店政策为准。"},
    {**_COMMON, "chunk_id": "c3", "parent_chunk_id": None, "doc_id": "d2", "page": 3,
     "content": "产品 X 型号 MX-200 售价 1999 元，含一年质保。",
     "source_uri": "minio://kf-raw/price.pdf", "char_start": 5, "char_end": 40},
    {**_COMMON, "chunk_id": "o1", "parent_chunk_id": None, "doc_id": "d9", "kb_id": "other",
     "page": 1, "content": "其他知识库的保修说明，不应被 demo 查询命中。"},
]


@pytest.fixture()
def client():
    backend = MemoryBackend(collection="kf_default_demo")
    backend.add_chunks(SEED)
    set_backend(backend)
    yield TestClient(app)
    reset_backend()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "kf-retrieve"
    assert set(body["index"]) >= {"configured", "collection", "collection_note"}
    # reranker 只暴露"URL 是否配置"，并明示这不等于端点可达
    assert "url_configured" in body["reranker"]
    assert "note" in body["reranker"]


def test_profiles(client):
    resp = client.get("/v1/profiles")
    assert resp.status_code == 200
    body = resp.json()
    assert "hybrid_rerank" in body["profiles"]
    assert body["global_budget"]["max_context_tokens"] == 8000


def test_search_end_to_end_hybrid(client):
    resp = client.post("/v1/search", json={
        "profile": "hybrid_rerank", "query": "保修期是多久", "tenant": "default",
        "kb": "demo", "top_k": 50, "max_context_tokens": 8000,
    })
    assert resp.status_code == 200
    body = resp.json()

    # 命中最相关的保修子块。父块**不再替换** content：
    # content 必须是命中子块自己的内容，offset 才与引用区间同源。
    assert body["evidence"], "应命中保修相关块"
    top = body["evidence"][0]
    assert top["chunk_id"] == "c1"
    assert top["parent_chunk_id"] == "p1"
    assert top["content"] == "产品 X 的保修期为 24 个月，自购买之日起计算。"
    assert top["char_range"] == [120, 260], "content 与 char_range 必须同源"
    assert top["offset_chunk_id"] == "c1"
    assert top["retrieval_channel"] == "sparse"
    assert isinstance(top["page"], int) and top["page"] == 1

    # 引用来自 provenance
    assert body["citations"]
    assert body["citations"][0]["char_range"] == [120, 260]

    # 重排不可用 → 复用融合分
    assert body["rerank_ms"] == 0.0
    trace = body["trace"]
    assert trace["fusion"] == "rrf"
    assert trace["reranked"] is False
    assert trace["profile"] == "hybrid_rerank"
    assert trace["channels"].get("sparse", 0) > 0
    assert set(trace["stages_ms"]) >= {"understand", "retrieve", "rerank", "assemble"}


def test_search_respects_kb_filter(client):
    resp = client.post("/v1/search", json={
        "profile": "hybrid_rerank", "query": "保修说明", "tenant": "default", "kb": "demo",
    })
    ids = [e["chunk_id"] for e in resp.json()["evidence"]]
    assert "o1" not in ids


def test_search_exact_lookup_hits_price_chunk(client):
    resp = client.post("/v1/search", json={
        "profile": "exact_lookup", "query": "MX-200 售价 1999 元", "tenant": "default", "kb": "demo",
    })
    assert resp.status_code == 200
    ids = [e["chunk_id"] for e in resp.json()["evidence"]]
    assert "c3" in ids
    assert resp.json()["trace"]["fusion"] == "weighted"


def test_search_unknown_profile_falls_back(client):
    resp = client.post("/v1/search", json={"query": "保修期", "profile": "not_a_profile"})
    assert resp.status_code == 200
    trace = resp.json()["trace"]
    assert trace["profile"] == "hybrid_rerank"
    assert any("回退" in n for n in trace["notes"])


def test_search_naive_dense_only_returns_empty_without_tei(client):
    resp = client.post("/v1/search", json={"query": "保修期", "profile": "naive", "tenant": "default", "kb": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["evidence"] == []          # 无 TEI → dense 为空，但不报错
    assert body["trace"]["fusion"] == "none"


def test_search_strict_refuse_on_low_score(client):
    # 用一个与库内容几乎无关的查询，验证拒答路径
    resp = client.post("/v1/search", json={
        "profile": "strict_refuse", "query": "zzz", "tenant": "default", "kb": "demo",
    })
    assert resp.status_code == 200
    assert resp.json()["trace"]["refused"] is True
    assert resp.json()["evidence"] == []


def test_search_never_500_on_empty_backend():
    set_backend(MemoryBackend(collection="empty"))
    try:
        client = TestClient(app)
        resp = client.post("/v1/search", json={"query": "任意查询", "profile": "hyde"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["evidence"] == []
        # HyDE/扩展需要生成端点，未配置时应记录 note 而非报错
        assert any("生成端点" in n for n in body["trace"]["notes"])
    finally:
        reset_backend()
