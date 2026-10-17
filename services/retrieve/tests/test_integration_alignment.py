"""跨服务一致性测试：摄入侧与检索侧的「集合命名 / 点 ID / 语言过滤」必须对齐。

这三者一旦不一致，线上表现是**静默召不回**（没有任何报错，只是结果为空），
是最难排查也最致命的一类集成缺陷，因此必须用测试锁死。
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RETRIEVE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_RETRIEVE, "..", ".."))
_INGEST = os.path.join(_REPO, "services", "ingest")
for path in (_RETRIEVE, _INGEST):
    if path not in sys.path:
        sys.path.insert(0, path)

from kf_retrieve.backends import MemoryBackend, matches_filters  # noqa: E402
from kf_retrieve.service import SearchRequest, run_search  # noqa: E402


def _ingest_module():
    """kf_ingest 依赖 PyMuPDF 等重依赖，缺失时跳过依赖它的断言。"""
    try:
        import kf_ingest.clients as clients
        import kf_ingest.index as index
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"无法导入 kf_ingest: {exc}")
    return clients, index


# ------------------------------------------------------------------ 集合命名
def test_collection_name_is_a_single_collection_per_tenant_kb():
    from kf_retrieve.backends import default_collection as retrieve_naming

    assert retrieve_naming("acme", "demo") == "kf_acme_demo"
    # 破折号需归一为下划线，避免非法集合名
    assert retrieve_naming("ac-me", "de-mo") == "kf_ac_me_de_mo"


def test_collection_naming_matches_ingest_side():
    """两侧命名函数必须给出同名集合，否则检索永远命中不到摄入的数据。"""
    clients, _ = _ingest_module()
    from kf_retrieve.backends import default_collection as retrieve_naming

    for tenant, kb in (("acme", "demo"), ("t1", "kb-2"), ("default", "default")):
        assert retrieve_naming(tenant, kb) == clients.default_collection(tenant, kb)


def test_collection_is_not_split_by_language():
    """多语言必须靠集合内 lang 过滤，而不是按语言拆集合。

    拆集合会让稠密通道只能看到单一语言，从而丧失跨语言召回能力。
    """
    clients, _ = _ingest_module()
    from kf_retrieve.backends import default_collection as retrieve_naming

    assert clients.default_collection("acme", "demo") == "kf_acme_demo"
    assert retrieve_naming("acme", "demo") == "kf_acme_demo"
    # 不应存在语言后缀形式
    assert not clients.default_collection("acme", "demo").endswith(("_zh", "_en", "_ja"))


# ------------------------------------------------------------------ 点 ID
def test_point_id_matches_ingest_side():
    from kf_retrieve.backends import point_id as retrieve_point_id

    _, index = _ingest_module()
    from kf_ingest.models import Chunk

    chunk = Chunk(chunk_id="d:v1:C1-abcd", chunk_type="text", content="x", lang="zh",
                  page=0, order=1)
    assert retrieve_point_id("d:v1:C1-abcd") == index.point_id(chunk)


# ------------------------------------------------------------------ 语言过滤
def test_matches_filters_enforces_language():
    zh = {"tenant_id": "t", "kb_id": "k", "lang": "zh", "status": "active"}
    en = {"tenant_id": "t", "kb_id": "k", "lang": "en", "status": "active"}

    assert matches_filters(zh, {"lang": "zh", "status": "active"}) is True
    assert matches_filters(en, {"lang": "zh", "status": "active"}) is False
    # lang=None 表示不按语言过滤（稠密通道用）
    assert matches_filters(en, {"lang": None, "status": "active"}) is True
    assert matches_filters(zh, None) is True


def test_sparse_channel_filters_by_language_but_dense_does_not():
    """稀疏通道按查询语言过滤；稠密通道不过滤（跨语言召回）。

    这是多语言检索正确性的核心：BM25 本质单语言，稠密向量跨语言对齐。
    """
    seen = []

    class RecordingBackend:
        def search(self, channel, query, *, filters=None, top_k=20, collection=None):
            seen.append((channel, dict(filters or {})))
            return []

        def get_chunk(self, chunk_id, collection=None):
            return None

    req = SearchRequest(query="保修期是多久", profile="hybrid_rerank",
                        tenant="default", kb="demo")
    run_search(req, RecordingBackend())

    channels = {channel: filters for channel, filters in seen}
    assert "sparse" in channels, f"应走过稀疏通道，实际 {list(channels)}"
    assert channels["sparse"]["lang"] == "zh", "稀疏通道必须按查询语言过滤"
    assert channels["sparse"]["status"] == "active"

    if "dense" in channels:
        assert channels["dense"]["lang"] is None, "稠密通道不得按语言过滤（跨语言召回）"


def test_sparse_filter_language_tracks_query_language():
    """查询语言变化时，稀疏过滤条件必须随之变化，而不是写死。"""
    seen = []

    class RecordingBackend:
        def search(self, channel, query, *, filters=None, top_k=20, collection=None):
            seen.append((channel, (filters or {}).get("lang")))
            return []

        def get_chunk(self, chunk_id, collection=None):
            return None

    for query in ("保修期是多久", "What is the warranty period?"):
        seen.clear()
        run_search(SearchRequest(query=query, profile="hybrid_rerank",
                                 tenant="t", kb="k"), RecordingBackend())
        sparse_langs = [lang for channel, lang in seen if channel == "sparse"]
        assert sparse_langs, f"查询 {query!r} 未走稀疏通道"
        expected = "zh" if "保修" in query else "en"
        assert set(sparse_langs) == {expected}, f"{query!r} -> {sparse_langs}"


# ------------------------------------------------------------------ 后端行为
def test_memory_backend_excludes_other_language_for_sparse_query():
    """端到端行为：中文查询的稀疏召回不应带出纯英文块。"""
    common = {"tenant_id": "t", "kb_id": "k", "status": "active", "source_uri": "s"}
    backend = MemoryBackend(collection="kf_t_k")
    backend.add_chunks([
        {**common, "chunk_id": "zh1", "lang": "zh",
         "content": "产品保修期为二十四个月，自购买之日起计算。"},
        {**common, "chunk_id": "en1", "lang": "en",
         "content": "The warranty period is twenty four months from purchase."},
    ])
    hits = backend.search("sparse", "保修期是多久", filters={"lang": "zh", "status": "active"})
    ids = [h["chunk"]["chunk_id"] for h in hits]
    assert "zh1" in ids
    assert "en1" not in ids, "中文稀疏查询不应召回英文块"
