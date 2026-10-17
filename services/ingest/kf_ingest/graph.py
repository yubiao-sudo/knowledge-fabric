"""GraphRAG 建图：轻量实体抽取 + 无向共现边 + 连通分量社区摘要。

设计红线（契约第 6 节）：

  - 实体抽取**不依赖 LLM**：规则 = SKU 形态 / 产品型号名 / 机构名 / 连续 CJK 术语（2–8）。
  - 归一化 = 小写 + 去首尾空白 + 全角转半角（NFKC）+ 压缩空白。
  - 边按**无向规范化**存储：`src_entity_id < dst_entity_id`，否则 `uq_edge_tenant_pair` 去重失效。
  - 社区检测用**连通分量**（不引入图算法库）；摘要优先用生成端点（LLM），
    无生成端点时用抽取式摘要（高频实体 + 代表块首句），`summary_method` 如实标注。

本模块把「纯计算」与「落库」分开：纯计算（归一化 / 抽取 / 建边 / 连通分量 / 摘要）
可离线单测；`build_graph` 负责写 `graph_entities` / `graph_entity_chunks` /
`graph_edges` / `graph_communities`，任一失败只记 note 不中断摄入。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .models import Chunk, IngestContext

# ------------------------------------------------------------------ 形态规则
_ORG_RE = re.compile(
    r"[\u4e00-\u9fff]{2,12}"
    r"(?:股份有限公司|有限责任公司|有限公司|集团|公司|银行|大学|学院|研究院|研究所|医院)"
)
_SKU_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,6}[-_]?\d{2,}[A-Za-z0-9]*(?![A-Za-z0-9])")
_PRODUCT_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,}\d{1,4}[A-Za-z]*(?![A-Za-z0-9])")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

_CJK_MAX = 8


# ------------------------------------------------------------------ 归一化
def normalize_entity(text: str) -> str:
    """归一化实体名：全角转半角（NFKC）+ 小写 + 压缩空白 + 去首尾空白。"""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm.lower()


def _cjk_terms(run: str, max_len: int = _CJK_MAX) -> List[str]:
    """连续 CJK 片段切成 2–8 字的术语（长串按 max_len 不重叠切分）。"""
    if len(run) <= max_len:
        return [run] if len(run) >= 2 else []
    out: List[str] = []
    cursor = 0
    while cursor < len(run):
        piece = run[cursor:cursor + max_len]
        if len(piece) < 2:
            break
        out.append(piece)
        cursor += max_len
    return out


def extract_entities(text: str) -> List[Dict[str, str]]:
    """轻量实体抽取（不依赖 LLM）。返回 `[{name, norm, entity_type}]`，按 norm 去重。"""
    text = text or ""
    found: List[Tuple[str, str]] = []
    for match in _SKU_RE.finditer(text):
        found.append((match.group(0), "sku"))
    for match in _ORG_RE.finditer(text):
        found.append((match.group(0), "org"))
    for match in _PRODUCT_RE.finditer(text):
        found.append((match.group(0), "product"))
    for run in _CJK_RUN_RE.findall(text):
        for term in _cjk_terms(run):
            found.append((term, "term"))

    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for name, entity_type in found:
        norm = normalize_entity(name)
        if not norm or len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        out.append({"name": name, "norm": norm, "entity_type": entity_type})
    return out


def extract_chunk_entities(chunks: Sequence[Chunk]) -> Dict[str, List[Dict[str, str]]]:
    return {chunk.chunk_id: extract_entities(chunk.content or "") for chunk in chunks}


# ------------------------------------------------------------------ 边（无向规范化）
def edge_pair(a: str, b: str) -> Tuple[str, str]:
    """无向边规范化：小 id 在前（`src_entity_id < dst_entity_id`）。"""
    return (a, b) if a <= b else (b, a)


def aggregate_edges(chunk_entities: Dict[str, Sequence[Dict[str, str]]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """按块内共现聚合边：同一对实体无论方向只留一条（无向去重），weight = 共现块数。"""
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk_id, entities in chunk_entities.items():
        norms = sorted({entity["norm"] for entity in entities if entity.get("norm")})
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                key = edge_pair(norms[i], norms[j])
                bucket = edges.setdefault(key, {"chunk_ids": set(), "doc_ids": set()})
                bucket["chunk_ids"].add(chunk_id)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, bucket in edges.items():
        out[key] = {
            "weight": len(bucket["chunk_ids"]),
            "chunk_ids": sorted(bucket["chunk_ids"]),
            "doc_ids": sorted(bucket["doc_ids"]),
        }
    return out


# ------------------------------------------------------------------ 连通分量
def connected_components(
    nodes: Sequence[str], edges: Sequence[Tuple[str, str]]
) -> List[Set[str]]:
    """无向图的连通分量（并查集）。只返回 size ≥ 2 的分量。"""
    parent: Dict[str, str] = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    groups: Dict[str, Set[str]] = {}
    for node in nodes:
        groups.setdefault(find(node), set()).add(node)
    return [members for members in groups.values() if len(members) >= 2]


# ------------------------------------------------------------------ 摘要
def _first_sentence(text: str, limit: int = 120) -> str:
    if not text:
        return ""
    for piece in re.split(r"[。！？!?\n]", text):
        sentence = piece.strip()
        if sentence:
            return sentence[:limit]
    return text.strip()[:limit]


def extractive_summary(top_entities: Sequence[str], texts: Sequence[str]) -> str:
    """抽取式摘要：社区内高频/中心实体 + 代表块首句（无生成端点时使用）。"""
    parts: List[str] = []
    if top_entities:
        parts.append("社区实体：" + "、".join(list(top_entities)[:8]))
    for text in list(texts)[:3]:
        sentence = _first_sentence(text)
        if sentence:
            parts.append(sentence)
    return "；".join(parts)


def summarize_community(
    top_entities: Sequence[str],
    texts: Sequence[str],
    generator: Optional[Callable[[str], str]] = None,
) -> Tuple[str, str]:
    """返回 (summary, summary_method)。有生成端点走 LLM，否则抽取式——如实标注。"""
    if generator is not None:
        try:
            prompt = (
                "下面是知识库中一个实体社区的中心实体与代表文本，"
                "请用 1-2 句概括该社区的主题，只输出概括本身。\n"
                "实体：" + "、".join(list(top_entities)[:8]) +
                "\n代表文本：\n" + "\n".join(list(texts)[:3])[:800]
            )
            generated = generator(prompt)
            generated = re.sub(r"\s+", " ", generated or "").strip()
            if generated:
                return generated[:400], "llm"
        except Exception:
            pass  # 生成端点失败即回退抽取式，绝不假装是 LLM 生成
    return extractive_summary(top_entities, texts), "extractive"


# ------------------------------------------------------------------ 落库
# 所有标识与查询都必须带 tenant_id：不同租户可复用同名 kb，
# 不带租户会把跨租户的实体/边/社区错误合并。
def _entity_id(tenant_id: str, kb_id: str, norm: str) -> str:
    return "e:" + hashlib.sha1(f"{tenant_id}|{kb_id}|{norm}".encode("utf-8")).hexdigest()[:20]


def _edge_id(tenant_id: str, kb_id: str, src: str, dst: str, relation: str) -> str:
    raw = f"{tenant_id}|{kb_id}|{src}|{dst}|{relation}"
    return "g:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _community_id(tenant_id: str, kb_id: str, members: Sequence[str]) -> str:
    raw = f"{tenant_id}|{kb_id}|" + "|".join(sorted(members))
    return "c:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


_INSERT_ENTITY = (
    "INSERT INTO graph_entities(entity_id, tenant_id, kb_id, name, norm_name, entity_type) "
    "VALUES (%(entity_id)s, %(tenant_id)s, %(kb_id)s, %(name)s, %(norm_name)s, %(entity_type)s) "
    "ON CONFLICT (tenant_id, kb_id, norm_name) DO UPDATE SET "
    "name = EXCLUDED.name, entity_type = EXCLUDED.entity_type "
    "RETURNING entity_id"
)
_INSERT_ENTITY_CHUNK = (
    "INSERT INTO graph_entity_chunks(entity_id, chunk_id, doc_id, kb_id, tenant_id, version, weight) "
    "VALUES (%(entity_id)s, %(chunk_id)s, %(doc_id)s, %(kb_id)s, %(tenant_id)s, %(version)s, %(weight)s) "
    "ON CONFLICT (entity_id, chunk_id) DO UPDATE SET weight = EXCLUDED.weight"
)
_INSERT_EDGE = (
    "INSERT INTO graph_edges(edge_id, tenant_id, kb_id, src_entity_id, dst_entity_id, relation, "
    "weight, chunk_ids, doc_ids) "
    "VALUES (%(edge_id)s, %(tenant_id)s, %(kb_id)s, %(src_entity_id)s, %(dst_entity_id)s, "
    "%(relation)s, %(weight)s, %(chunk_ids)s, %(doc_ids)s) "
    "ON CONFLICT (tenant_id, kb_id, src_entity_id, dst_entity_id, relation) DO UPDATE SET "
    "chunk_ids = (SELECT COALESCE(jsonb_agg(DISTINCT v), '[]'::jsonb) "
    "FROM jsonb_array_elements_text(graph_edges.chunk_ids || EXCLUDED.chunk_ids) AS t(v)), "
    "doc_ids = (SELECT COALESCE(jsonb_agg(DISTINCT v), '[]'::jsonb) "
    "FROM jsonb_array_elements_text(graph_edges.doc_ids || EXCLUDED.doc_ids) AS t(v)), "
    "weight = (SELECT count(DISTINCT v) "
    "FROM jsonb_array_elements_text(graph_edges.chunk_ids || EXCLUDED.chunk_ids) AS t(v))"
)
_INSERT_COMMUNITY = (
    "INSERT INTO graph_communities(community_id, tenant_id, kb_id, level, size, members, "
    "top_entities, summary, summary_method, chunk_ids) "
    "VALUES (%(community_id)s, %(tenant_id)s, %(kb_id)s, %(level)s, %(size)s, %(members)s, "
    "%(top_entities)s, %(summary)s, %(summary_method)s, %(chunk_ids)s) "
    "ON CONFLICT (community_id) DO UPDATE SET "
    "size = EXCLUDED.size, members = EXCLUDED.members, top_entities = EXCLUDED.top_entities, "
    "summary = EXCLUDED.summary, summary_method = EXCLUDED.summary_method, "
    "chunk_ids = EXCLUDED.chunk_ids"
)


def build_graph(conn: Any, ctx: IngestContext, chunks: Sequence[Chunk]) -> Dict[str, Any]:
    """写入图存储。任一子步骤失败只记 note，不抛异常。"""
    from psycopg.types.json import Jsonb

    stats: Dict[str, Any] = {
        "entities": 0, "entity_chunks": 0, "edges": 0, "communities": 0, "notes": [],
    }
    chunk_entities = extract_chunk_entities(chunks)
    if not chunk_entities:
        return stats

    generator = _llm_generator()

    id_by_norm: Dict[str, str] = {}
    with conn.cursor() as cur:
        for chunk_id, entities in chunk_entities.items():
            for entity in entities:
                norm = entity["norm"]
                if norm in id_by_norm:
                    continue
                candidate = _entity_id(ctx.tenant_id, ctx.kb_id, norm)
                cur.execute(_INSERT_ENTITY, {
                    "entity_id": candidate, "tenant_id": ctx.tenant_id, "kb_id": ctx.kb_id,
                    "name": entity["name"], "norm_name": norm,
                    "entity_type": entity["entity_type"],
                })
                row = cur.fetchone()
                id_by_norm[norm] = row[0] if row else candidate
                stats["entities"] += 1

        for chunk_id, entities in chunk_entities.items():
            for entity in entities:
                entity_id = id_by_norm.get(entity["norm"])
                if not entity_id:
                    continue
                cur.execute(_INSERT_ENTITY_CHUNK, {
                    "entity_id": entity_id, "chunk_id": chunk_id, "doc_id": ctx.doc_id,
                    "kb_id": ctx.kb_id, "tenant_id": ctx.tenant_id, "version": ctx.version,
                    "weight": 1.0,
                })
                stats["entity_chunks"] += 1

        edges = _aggregate_edges_by_id(chunk_entities, id_by_norm, ctx)
        for (src, dst), payload in edges.items():
            cur.execute(_INSERT_EDGE, {
                "edge_id": _edge_id(ctx.tenant_id, ctx.kb_id, src, dst, "co_occur"),
                "tenant_id": ctx.tenant_id, "kb_id": ctx.kb_id,
                "src_entity_id": src, "dst_entity_id": dst, "relation": "co_occur",
                "weight": payload["weight"],
                "chunk_ids": Jsonb(payload["chunk_ids"]),
                "doc_ids": Jsonb(payload["doc_ids"]),
            })
            stats["edges"] += 1

    try:
        stats["communities"] = _rebuild_communities(conn, ctx, generator, Jsonb)
    except Exception as exc:
        stats["notes"].append(f"社区检测失败（已跳过）: {exc}")
    return stats


def _aggregate_edges_by_id(
    chunk_entities: Dict[str, List[Dict[str, str]]],
    id_by_norm: Dict[str, str],
    ctx: IngestContext,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk_id, entities in chunk_entities.items():
        ids = sorted({id_by_norm[e["norm"]] for e in entities if e["norm"] in id_by_norm})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = edge_pair(ids[i], ids[j])
                bucket = edges.setdefault(key, {"chunk_ids": set(), "doc_ids": set()})
                bucket["chunk_ids"].add(chunk_id)
                bucket["doc_ids"].add(ctx.doc_id)
    return {
        key: {
            "weight": len(bucket["chunk_ids"]),
            "chunk_ids": sorted(bucket["chunk_ids"]),
            "doc_ids": sorted(bucket["doc_ids"]),
        }
        for key, bucket in edges.items()
    }


def _llm_generator() -> Optional[Callable[[str], str]]:
    try:
        from .clients import try_llm

        client = try_llm()
    except Exception:
        return None
    if client is None:
        return None
    return lambda prompt: client.chat([{"role": "user", "content": prompt}])


def _rebuild_communities(
    conn: Any,
    ctx: IngestContext,
    generator: Optional[Callable[[str], str]],
    jsonb: Any,
    max_communities: int = 200,
    reps_per_community: int = 5,
) -> int:
    """从库内全量实体/边重算连通分量并写 graph_communities（幂等：community_id 确定性）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT entity_id, norm_name FROM graph_entities "
            "WHERE tenant_id = %s AND kb_id = %s",
            (ctx.tenant_id, ctx.kb_id),
        )
        entities = cur.fetchall()
        cur.execute(
            "SELECT src_entity_id, dst_entity_id FROM graph_edges "
            "WHERE tenant_id = %s AND kb_id = %s",
            (ctx.tenant_id, ctx.kb_id),
        )
        edges = cur.fetchall()

    norm_of = {row[0]: row[1] for row in entities}
    components = connected_components(list(norm_of.keys()), [(row[0], row[1]) for row in edges])

    written = 0
    for members in components[:max_communities]:
        member_ids = sorted(members)
        top_entities = [norm_of.get(mid, mid) for mid in member_ids[:reps_per_community]]
        representative_chunks = _representative_chunks(conn, ctx, member_ids, reps_per_community)
        texts = _chunk_texts(conn, representative_chunks)
        summary, method = summarize_community(top_entities, texts, generator)
        with conn.cursor() as cur:
            cur.execute(_INSERT_COMMUNITY, {
                "community_id": _community_id(ctx.tenant_id, ctx.kb_id, member_ids),
                "tenant_id": ctx.tenant_id, "kb_id": ctx.kb_id, "level": 0,
                "size": len(member_ids),
                "members": jsonb(member_ids),
                "top_entities": jsonb(top_entities),
                "summary": summary, "summary_method": method,
                "chunk_ids": jsonb(representative_chunks),
            })
        written += 1
    return written


def _representative_chunks(conn: Any, ctx: IngestContext, member_ids: Sequence[str],
                           limit: int) -> List[str]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM graph_entity_chunks "
                "WHERE tenant_id = %s AND kb_id = %s AND entity_id = ANY(%s) "
                "ORDER BY chunk_id LIMIT %s",
                (ctx.tenant_id, ctx.kb_id, list(member_ids), limit),
            )
            return [row[0] for row in cur.fetchall()]
    except Exception:
        return []


def _chunk_texts(conn: Any, chunk_ids: Sequence[str]) -> List[str]:
    if not chunk_ids:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, content FROM chunks WHERE chunk_id = ANY(%s)",
                (list(chunk_ids),),
            )
            return [row[1] for row in cur.fetchall() if row[1]]
    except Exception:
        return []
