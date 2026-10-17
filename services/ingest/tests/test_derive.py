"""摄入侧四缺口的新增测试：结构化双轨 / GraphRAG / ACL / OpenSearch。

既有 54 个测试不动；本文件只做增量覆盖，全部**离线可跑**（不连真实 Qdrant /
Postgres / OpenSearch），用最小 FakeConn 承载落库断言。
"""

import json

import pytest

from kf_ingest.acl import parse_acl
from kf_ingest.graph import (
    aggregate_edges,
    build_graph,
    connected_components,
    edge_pair,
    extract_entities,
    normalize_entity,
    summarize_community,
)
from kf_ingest.index import _payload
from kf_ingest.models import Chunk, IngestContext
from kf_ingest.opensearch import (
    analyzer_for_lang,
    build_bulk_body,
    build_mapping,
    index_chunks_opensearch,
    index_name,
)
from kf_ingest.pipeline import run_ingest
from kf_ingest.structured import extract_from_table, record_key, write_structured_records

TABLE_MD = (
    "| 型号 | 名称 | 价格 | 币种 | 颜色 | 尺寸 |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
    "| MX-100 | 智能网关 | ¥1,299.00 | CNY | 黑色 | 120x80 |\n"
    "| KX2000 | 边缘服务器 | 2500 元 | CNY | 银色 | 2U |\n"
)


def make_chunk(chunk_id="c1", content="正文", chunk_type="text", lang="zh", **kw):
    return Chunk(chunk_id=chunk_id, chunk_type=chunk_type, content=content,
                 lang=lang, page=1, order=0, **kw)


def make_ctx(**kw):
    base = dict(tenant_id="t1", kb_id="kb1", doc_id="d1", source_uri="s://d1")
    base.update(kw)
    return IngestContext(**base)


# --------------------------------------------------------------- FakeConn
class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._one = None
        self._many = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        s = " ".join(str(sql).split())
        self._one, self._many = None, []
        if isinstance(params, dict):  # 命名参数（结构化的 ON CONFLICT / 图写入）
            if "INSERT INTO structured_records" in s:
                key = (params["chunk_id"], params["schema_id"], params["sku"] or "")
                self.store.setdefault("structured", {})[key] = params
            elif "INSERT INTO graph_entities" in s and "graph_entity_chunks" not in s:
                self.store.setdefault("entities", {})[params["norm_name"]] = (
                    params["entity_id"], params["norm_name"],
                )
                self._one = (params["entity_id"],)
            elif "INSERT INTO graph_entity_chunks" in s:
                self.store.setdefault("entity_chunks", []).append(params)
            elif "INSERT INTO graph_edges" in s:
                self.store.setdefault("edges", {})[
                    (params["src_entity_id"], params["dst_entity_id"])
                ] = params
            elif "INSERT INTO graph_communities" in s:
                self.store["communities"] = self.store.get("communities", 0) + 1
        else:
            args = tuple(params or ())
            if "INSERT INTO chunk_acl" in s:
                self.store.setdefault("acl", []).append(args)
            elif s.startswith("SELECT entity_id, norm_name FROM graph_entities"):
                self._many = list(self.store.get("entities", {}).values())
            elif s.startswith("SELECT src_entity_id, dst_entity_id FROM graph_edges"):
                self._many = list(self.store.get("edges", {}).keys())
        return self

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class FakeConn:
    def __init__(self):
        self.store = {}
        self.commits = 0

    def cursor(self):
        return FakeCursor(self.store)

    def commit(self):
        self.commits += 1


# --------------------------------------------------------------- 结构化
def test_extract_from_table_maps_headers_to_roles():
    info = extract_from_table(TABLE_MD)
    assert info["skus"] == ["MX-100", "KX2000"]
    assert info["price"] == 1299.0            # 首行价格（符号+数值）
    assert info["currency"] == "CNY"
    assert info["attrs"] == {"颜色": "黑色", "尺寸": "120x80"}   # 列头 → 单元格映射
    assert len(info["records"]) == 2
    assert info["records"][1]["price"] == 2500.0             # 「2500 元」按单位识别
    assert info["records"][1]["sku"] == "KX2000"


def test_structured_records_idempotent_on_rerun():
    chunk = make_chunk(content=TABLE_MD, chunk_type="table")
    conn = FakeConn()
    n1 = write_structured_records(conn, make_ctx(), [chunk])
    n2 = write_structured_records(conn, make_ctx(), [chunk])   # 重跑同一文档
    assert n1 == n2 == 2
    assert len(conn.store["structured"]) == 2                 # 无重复行
    row = conn.store["structured"][("c1", "table:型号|名称|价格|币种|颜色|尺寸", "MX-100")]
    # 契约必需列齐全
    for col in ("tenant_id", "kb_id", "chunk_id", "schema_id", "sku", "name",
                "price", "currency", "attrs", "payload", "doc_id", "version"):
        assert col in row
    assert row["tenant_id"] == "t1" and row["doc_id"] == "d1"


def test_record_key_matches_unique_key():
    assert record_key({"chunk_id": "c", "schema_id": "s", "sku": None}) == ("c", "s", "")


def test_payload_has_fixed_structured_and_entity_keys():
    chunk = make_chunk(content=TABLE_MD, chunk_type="table")
    chunk.meta["structured"] = {"skus": ["MX-100"], "price": 1.0,
                                "currency": "CNY", "attrs": {"a": "b"}}
    chunk.meta["entities_norm"] = ["mx-100"]
    payload = _payload(make_ctx(), chunk)
    assert payload["skus"] == ["MX-100"]
    assert payload["price"] == 1.0
    assert payload["currency"] == "CNY"
    assert payload["attrs"] == {"a": "b"}
    assert payload["entities"] == ["mx-100"]


# --------------------------------------------------------------- 图 / 实体
def test_normalize_entity_fullwidth_lowercase_whitespace():
    assert normalize_entity("  BGE-M3  ") == "bge-m3"
    assert normalize_entity("ＡＢＣ　１２３") == "abc 123"     # NFKC + 压缩空白
    assert normalize_entity("　") == ""


def test_extract_entities_rule_based_types():
    ents = extract_entities("华为技术有限公司的 MX-100 与 KX2000 协同工作")
    by_norm = {e["norm"]: e["entity_type"] for e in ents}
    assert by_norm.get("mx-100") == "sku"
    assert by_norm.get("kx2000") == "sku"
    assert by_norm.get("华为技术有限公司") == "org"
    assert "协同工作" in by_norm                 # 连续 CJK 术语（2–8）


def test_edge_pair_is_undirected_normalized():
    assert edge_pair("z", "a") == ("a", "z")
    assert edge_pair("a", "z") == ("a", "z")     # 双向都归一到同一对


def test_aggregate_edges_dedupes_direction():
    chunk_entities = {
        "c1": [{"norm": "kx2000"}, {"norm": "mx-100"}],   # 顺序相反
        "c2": [{"norm": "mx-100"}, {"norm": "kx2000"}],
    }
    edges = aggregate_edges(chunk_entities)
    assert list(edges.keys()) == [("kx2000", "mx-100")]   # 只留一条无向边
    assert edges[("kx2000", "mx-100")]["weight"] == 2     # 共现 2 个块


def test_connected_components_groups():
    groups = connected_components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert groups == [{"a", "b", "c"}]                     # 只返回 size ≥ 2


def test_summarize_community_extractive_without_generator():
    summary, method = summarize_community(
        ["mx-100", "kx2000"], ["MX-100 是智能网关。第二句。"], generator=None
    )
    assert method == "extractive"                          # 无 gen 如实标注
    assert "mx-100" in summary and "MX-100 是智能网关" in summary


def test_summarize_community_uses_llm_when_generator_present():
    summary, method = summarize_community(
        ["mx-100"], ["正文"], generator=lambda prompt: "  这是 LLM 摘要  "
    )
    assert method == "llm" and summary == "这是 LLM 摘要"


def test_build_graph_writes_undirected_edges_and_communities(monkeypatch):
    for name in ("VLLM_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)            # 无生成端点 → 抽取式
    chunks = [
        make_chunk("c1", "MX-100 与 KX2000 协同工作"),
        make_chunk("c2", "KX2000 单独出现"),
    ]
    conn = FakeConn()
    stats = build_graph(conn, make_ctx(), chunks)
    assert stats["entities"] >= 2
    assert stats["edges"] >= 1
    for src, dst in conn.store["edges"]:
        assert src < dst                                   # 无向规范化存储
    assert stats["communities"] >= 1
    assert conn.store.get("communities", 0) >= 1           # 已写社区


def test_graph_opt_in_defaults_off():
    assert make_ctx().graph is False                   # 默认关闭（设计约束）
    assert make_ctx(graph=True).graph is True


# --------------------------------------------------------------- ACL
def test_parse_acl_with_deny_prefix():
    rules = parse_acl("user:u1,group:g1,!user:u9")
    assert rules == [
        {"principal_type": "user", "principal_id": "u1", "effect": "allow"},
        {"principal_type": "group", "principal_id": "g1", "effect": "allow"},
        {"principal_type": "user", "principal_id": "u9", "effect": "deny"},
    ]


def test_parse_acl_deny_wins_over_allow():
    rules = parse_acl("user:u1,!user:u1")
    assert rules == [{"principal_type": "user", "principal_id": "u1", "effect": "deny"}]


def test_parse_acl_empty_and_default_principal_type():
    assert parse_acl("") == []
    assert parse_acl(" alice ") == [
        {"principal_type": "user", "principal_id": "alice", "effect": "allow"}
    ]


def test_chunk_acl_insert_carries_tenant_id():
    conn = FakeConn()
    chunk = make_chunk()
    chunk.acl = parse_acl("user:u1,!user:u9")
    # 直接驱动 write_postgres，检查 ACL 行写入了 tenant_id
    from kf_ingest.index import write_postgres

    write_postgres(conn, make_ctx(), [chunk], dim=8, collection="kf_t1_kb1")
    acl_rows = conn.store.get("acl", [])
    assert len(acl_rows) == 2
    assert all(row[1] == "t1" for row in acl_rows)          # 第 2 列 = tenant_id
    assert {row[4] for row in acl_rows} == {"allow", "deny"}


# --------------------------------------------------------------- OpenSearch
def test_analyzer_and_index_name_by_lang():
    assert analyzer_for_lang("zh") == "cjk"
    assert analyzer_for_lang("ja") == "cjk"
    assert analyzer_for_lang("en") == "english"
    assert analyzer_for_lang("ko") == "english"
    assert analyzer_for_lang("fr") == "standard"
    assert index_name("acme", "demo", "zh") == "kf_acme_demo_zh"


def test_build_bulk_body_uses_chunk_id_and_active_status():
    ctx = make_ctx()
    chunk = make_chunk("c1", content="正文", lang="zh")
    body = build_bulk_body(ctx, [chunk]).strip().split("\n")
    action, document = json.loads(body[0]), json.loads(body[1])
    assert action["index"]["_id"] == "c1"                   # 幂等：_id = chunk_id
    assert action["index"]["_index"] == "kf_t1_kb1_zh"
    assert document["status"] == "active"
    assert document["kb_id"] == "kb1" and document["chunk_id"] == "c1"


def test_opensearch_disabled_records_note_not_error():
    ctx = make_ctx()
    stats = index_chunks_opensearch(ctx, [make_chunk()], url="")
    assert stats["indexed"] == 0
    assert stats["notes"] and "OPENSEARCH_URL" in stats["notes"][0]   # 降级 note


def test_opensearch_mapping_uses_analyzer():
    mapping = build_mapping("cjk")
    assert mapping["mappings"]["properties"]["content"]["analyzer"] == "cjk"


def test_opensearch_pipeline_stage_notes_without_url(monkeypatch):
    """真机 run_ingest（dry_run=False）在未配置 OPENSEARCH_URL 时只记 note，不报错。"""
    for name in ("OPENSEARCH_URL", "QDRANT_URL", "DATABASE_URL", "TEI_EMBED_URL",
                 "VLLM_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    ctx = IngestContext(tenant_id="t1", kb_id="kb1", doc_id="d1", source_uri="s://d1")
    ctx.parsed = _stub_parsed()
    run_ingest(ctx, dry_run=False, with_enrich=False)
    assert ctx.errors == []                                  # 未中断摄入
    assert ctx.stage_stats["opensearch"]["notes"]            # 记了降级 note


def _stub_parsed():
    from kf_ingest.models import Block, ParsedDoc

    blocks = [Block(block_type="paragraph", text="MX-100 设备安装说明。", page=1, order=0)]
    return ParsedDoc(doc_id="d1", version=1, lang="zh", doc_type="manual",
                     blocks=blocks, raw_pages=["MX-100 设备安装说明。"])
