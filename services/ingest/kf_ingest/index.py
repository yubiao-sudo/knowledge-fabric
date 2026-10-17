"""索引写入：Qdrant（向量）+ Postgres（元数据 / ACL / 溯源）。

两条独立的写入路径，各自可降级：

- **Qdrant**：命名向量 `dense` + `sparse`，payload 携带 tenant/kb/lang/version
  等过滤字段，让"相关性 ∩ 可见性"在向量层就能收窄。稠密向量缺席时只写稀疏位。
- **Postgres**：`document_versions` / `chunks` / `chunk_provenance` /
  `chunk_acl` / `chunk_vectors`。幂等以 `content_hash` 与
  `(doc_id, version)` 主键保证。

版本切换策略：新版本写入后把旧版本块标记 `stale`（不物理删除），检索侧过滤。
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .models import Chunk, IngestContext, ParsedDoc

_POINT_NS = uuid.UUID("6f1a3c9e-2b47-5d8e-9a10-1c2d3e4f5a6b")


def point_id(chunk: Chunk) -> str:
    """Qdrant 点 ID 必须是 uuid 或 uint64，因此由 chunk_id 派生确定性 UUID。"""
    return str(uuid.uuid5(_POINT_NS, chunk.chunk_id))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload(ctx: IngestContext, chunk: Chunk) -> Dict[str, Any]:
    prov = chunk.provenance
    meta = chunk.meta or {}
    structured = meta.get("structured") or {}
    return {
        "tenant_id": ctx.tenant_id,
        "kb_id": ctx.kb_id,
        "workspace_id": None,
        "doc_id": ctx.doc_id,
        "version": ctx.version,
        "chunk_id": chunk.chunk_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "chunk_type": chunk.chunk_type,
        "lang": chunk.lang,
        "page": chunk.page,
        "order": chunk.order,
        "section_path": chunk.section_path,
        "keywords": chunk.keywords,
        "content": chunk.content,
        "status": "active",
        "source_uri": prov.source_uri if prov else None,
        "char_start": prov.char_start if prov else None,
        "char_end": prov.char_end if prov else None,
        "acl": chunk.acl,
        # 契约第 2 节：结构化双轨 + GraphRAG 的固定 payload 键名（不许改）
        "skus": list(structured.get("skus") or []),
        "price": structured.get("price"),
        "currency": structured.get("currency"),
        "attrs": dict(structured.get("attrs") or {}),
        "entities": list(meta.get("entities_norm") or []),
    }


# ------------------------------------------------------------------ Qdrant
def ensure_collection(client: Any, name: str, dim: int) -> None:
    from qdrant_client.models import (
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
    )


def mark_stale_qdrant(client: Any, collection: str, doc_id: str, version: int,
                      tenant_id: str = "") -> int:
    """版本切换：把同一文档**旧版本**的 Qdrant 点标为 stale（不物理删除）。

    与 Postgres 侧的 `UPDATE chunks SET status='stale'` 对称。检索侧按 payload
    `status == active` 过滤，因此只标 Postgres 而不标 Qdrant 时，旧版本点仍会被
    召回——agent 就会拿到已作废的旧版本文档内容。返回值是受影响点数。

    `tenant_id` 非空时一并作为过滤条件：集合按 `kf_<tenant>_<kb>` 隔离，
    正常情况下集合内只有一个租户，但显式过滤可防止未来共用集合时跨租户误标。
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    if tenant_id:
        must.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))
    only_older = Filter(
        must=must,
        must_not=[FieldCondition(key="version", match=MatchValue(value=version))],
    )
    try:
        stale = client.count(
            collection_name=collection, count_filter=only_older, exact=True
        ).count
    except Exception:
        stale = 0
    if stale:
        client.set_payload(
            collection_name=collection, payload={"status": "stale"},
            points=only_older, wait=True,
        )
    return stale


def upsert_qdrant(
    client: Any,
    collection: str,
    ctx: IngestContext,
    chunks: Sequence[Chunk],
    dim: int,
) -> int:
    from qdrant_client.models import PointStruct, SparseVector

    ensure_collection(client, collection, dim)

    points: List[Any] = []
    for chunk in chunks:
        vectors = (chunk.meta or {}).get("vectors") or {}
        dense = vectors.get("dense")
        sparse = vectors.get("sparse") or {}
        vector: Dict[str, Any] = {}
        if dense:
            vector["dense"] = dense
        if sparse.get("indices"):
            vector["sparse"] = SparseVector(
                indices=list(sparse["indices"]), values=list(sparse["values"])
            )
        if not vector:
            continue
        points.append(
            PointStruct(id=point_id(chunk), vector=vector, payload=_payload(ctx, chunk))
        )

    if not points:
        return 0
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


# ------------------------------------------------------------------ Postgres
def _exec(conn: Any, sql: str, params: Sequence[Any] = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def bootstrap_entities(conn: Any, ctx: IngestContext) -> None:
    """确保 tenant / kb / document 存在，满足外键约束（幂等）。

    **冲突目标必须带 tenant_id**（004 迁移后主键已改为租户维度）：
    旧实现用 `ON CONFLICT (kb_id)` / `ON CONFLICT (doc_id)`，只按业务 id 判重，
    导致第二个租户摄入同名 kb / 同名 doc 时命中第一个租户的行：
      · kb 用 DO NOTHING → 第二个租户连自己的库都建不出来（真机实测：
        documents 有 demoB/graphdemo 而 knowledge_bases 里只有 demoA/graphdemo）；
      · doc 用 DO UPDATE → 只改版本号，tenant_id/kb_id 保持首写租户，
        新租户归属被静默丢弃，接口却仍返回 ok=true。
    改为租户复合键后，两个租户可以各自持有同名 kb / doc，互不干扰。
    """
    _exec(
        conn,
        "INSERT INTO tenants(tenant_id, name) VALUES (%s, %s) "
        "ON CONFLICT (tenant_id) DO NOTHING",
        (ctx.tenant_id, ctx.tenant_id),
    )
    _exec(
        conn,
        "INSERT INTO knowledge_bases(kb_id, tenant_id, name, chunking_strategy) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (tenant_id, kb_id) DO UPDATE "
        "SET name = EXCLUDED.name, chunking_strategy = EXCLUDED.chunking_strategy",
        (ctx.kb_id, ctx.tenant_id, ctx.kb_id, ctx.strategy),
    )
    _exec(
        conn,
        "INSERT INTO documents(doc_id, kb_id, tenant_id, title, source_uri, lang, doc_type, "
        "current_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (tenant_id, doc_id) DO UPDATE SET kb_id = EXCLUDED.kb_id, "
        "current_version = EXCLUDED.current_version, updated_at = now()",
        (ctx.doc_id, ctx.kb_id, ctx.tenant_id, ctx.doc_id, ctx.source_uri,
         ctx.lang, ctx.doc_type, ctx.version),
    )


def write_postgres(conn: Any, ctx: IngestContext, chunks: Sequence[Chunk],
                   dim: int, collection: str) -> int:
    """写入版本、块、溯源、ACL、向量映射。调用方负责事务与提交。"""
    from psycopg.types.json import Jsonb

    parsed: Optional[ParsedDoc] = ctx.parsed
    _exec(
        conn,
        "INSERT INTO document_versions(tenant_id, doc_id, version, content_hash, parse_meta, "
        "cleaning_meta, chunking_meta, embed_model, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (tenant_id, doc_id, version) DO UPDATE SET parse_meta = EXCLUDED.parse_meta, "
        "cleaning_meta = EXCLUDED.cleaning_meta, chunking_meta = EXCLUDED.chunking_meta, "
        "embed_model = EXCLUDED.embed_model, status = EXCLUDED.status",
        (
            ctx.tenant_id,
            ctx.doc_id,
            ctx.version,
            content_hash("\n".join(c.chunk_id for c in chunks)),
            Jsonb({
                "scanner": parsed.scanner if parsed else None,
                "pages": parsed.page_count() if parsed else 0,
                "warnings": parsed.warnings if parsed else [],
            }),
            Jsonb({
                "removed_blocks": ctx.cleaning.removed_blocks,
                "removed_by_type": ctx.cleaning.removed_by_type,
                "deduped_blocks": ctx.cleaning.deduped_blocks,
                "boilerplate_keys": ctx.cleaning.boilerplate_keys[:50],
            }),
            Jsonb({"strategy": ctx.strategy}),
            ctx.embedded_model,
            "ready",
        ),
    )

    # 旧版本块标记 stale（不物理删除，检索侧按 status 过滤）。
    # **必须带 tenant_id**：doc_id 只在本租户内唯一，两个租户可以有同名 doc，
    # 不过滤会把其他租户的同名文档块一起标成作废（真机缺陷，与主键缺租户同源）。
    _exec(
        conn,
        "UPDATE chunks SET status = 'stale' WHERE tenant_id = %s AND doc_id = %s "
        "AND version <> %s AND status = 'active'",
        (ctx.tenant_id, ctx.doc_id, ctx.version),
    )

    written = 0
    for chunk in chunks:
        prov = chunk.provenance
        vec = (chunk.meta or {}).get("vectors") or {}
        _exec(
            conn,
            "INSERT INTO chunks(chunk_id, tenant_id, kb_id, doc_id, version, parent_chunk_id, "
            "chunk_type, content, context_prefix, content_hash, lang, section_path, page, "
            "order_no, entities, keywords, status) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'active') "
            "ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET content = EXCLUDED.content, "
            "context_prefix = EXCLUDED.context_prefix, parent_chunk_id = EXCLUDED.parent_chunk_id, "
            "status = 'active'",
            (
                chunk.chunk_id, ctx.tenant_id, ctx.kb_id, ctx.doc_id, ctx.version,
                chunk.parent_chunk_id, chunk.chunk_type, chunk.content,
                chunk.context_prefix, content_hash(chunk.content), chunk.lang,
                chunk.section_path or [], chunk.page, chunk.order,
                Jsonb(chunk.entities or []), chunk.keywords or [],
            ),
        )
        written += 1

        if prov is not None:
            _exec(
                conn,
                "INSERT INTO chunk_provenance(chunk_id, tenant_id, source_uri, page, char_start, "
                "char_end, bbox, cleaning_ops, original_text) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET "
                "source_uri = EXCLUDED.source_uri, "
                "page = EXCLUDED.page, char_start = EXCLUDED.char_start, "
                "char_end = EXCLUDED.char_end, cleaning_ops = EXCLUDED.cleaning_ops, "
                "original_text = EXCLUDED.original_text",
                (
                    chunk.chunk_id, ctx.tenant_id, prov.source_uri, prov.page, prov.char_start,
                    prov.char_end, prov.bbox, prov.cleaning_ops or [], prov.original_text,
                ),
            )

        for rule in chunk.acl or []:
            _exec(
                conn,
                "INSERT INTO chunk_acl(chunk_id, tenant_id, principal_type, principal_id, effect) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (tenant_id, chunk_id, principal_type, principal_id) DO UPDATE SET "
                "effect = EXCLUDED.effect",
                (chunk.chunk_id, ctx.tenant_id, rule.get("principal_type"),
                 rule.get("principal_id"), rule.get("effect", "allow")),
            )

        if vec.get("dense") or (vec.get("sparse") or {}).get("indices"):
            _exec(
                conn,
                "INSERT INTO chunk_vectors(tenant_id, chunk_id, collection, embed_model, "
                "dense_point_id, sparse_point_id, colbert_point_id, dims) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET collection = EXCLUDED.collection, "
                "embed_model = EXCLUDED.embed_model, dims = EXCLUDED.dims",
                (
                    ctx.tenant_id, chunk.chunk_id, collection, vec.get("model") or "",
                    point_id(chunk) if vec.get("dense") else None,
                    point_id(chunk) if (vec.get("sparse") or {}).get("indices") else None,
                    point_id(chunk) if vec.get("colbert") else None,
                    dim,
                ),
            )

    return written


def _guarded_derived(conn: Any, fn) -> Any:
    """在 SAVEPOINT 内运行派生写入：任一失败只回滚该步，不污染外层事务。"""
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT kf_derived")
        result = fn()
        with conn.cursor() as cur:
            cur.execute("RELEASE SAVEPOINT kf_derived")
        return result
    except Exception:
        try:
            with conn.cursor() as cur:
                cur.execute("ROLLBACK TO SAVEPOINT kf_derived")
        except Exception:
            pass
        raise


def write_derived_postgres(conn: Any, ctx: IngestContext, chunks: Sequence[Chunk]) -> Dict[str, Any]:
    """结构化记录 + GraphRAG 建图。两者各自可降级，失败只记 note。"""
    stats: Dict[str, Any] = {"structured_records": 0, "graph": None, "notes": []}
    try:
        from .structured import write_structured_records

        stats["structured_records"] = _guarded_derived(
            conn, lambda: write_structured_records(conn, ctx, chunks)
        )
    except Exception as exc:
        stats["notes"].append(f"结构化记录写入失败（已跳过）: {exc}")

    if getattr(ctx, "graph", False):
        try:
            from .graph import build_graph

            graph_stats = _guarded_derived(conn, lambda: build_graph(conn, ctx, chunks))
            stats["graph"] = graph_stats
            stats["notes"].extend(graph_stats.get("notes") or [])
        except Exception as exc:
            stats["notes"].append(f"建图失败（已跳过）: {exc}")
    return stats


def index_chunks(
    ctx: IngestContext,
    chunks: Sequence[Chunk],
    *,
    qdrant_client: Any = None,
    collection: str = "",
    dim: int = 1024,
    pg_conn: Any = None,
    commit: bool = True,
) -> Dict[str, Any]:
    """写入 Qdrant 与 Postgres。任一路径不可用只记录 note，不抛异常。"""
    stats: Dict[str, Any] = {"qdrant": 0, "qdrant_stale": 0, "postgres": 0,
                             "collection": collection, "notes": []}

    if qdrant_client is not None and collection:
        try:
            stats["qdrant"] = upsert_qdrant(qdrant_client, collection, ctx, chunks, dim)
            # 版本切换必须在写入之后：旧版本点标 stale，检索侧才能滤掉作废证据
            stats["qdrant_stale"] = mark_stale_qdrant(
                qdrant_client, collection, ctx.doc_id, ctx.version, ctx.tenant_id
            )
        except Exception as exc:
            stats["notes"].append(f"Qdrant 写入失败: {exc}")

    if pg_conn is not None:
        try:
            bootstrap_entities(pg_conn, ctx)
            # 建图 opt-in：--graph 同时把 knowledge_bases.graph_enabled 置 TRUE
            if getattr(ctx, "graph", False):
                _exec(
                    pg_conn,
                    "UPDATE knowledge_bases SET graph_enabled = TRUE "
                    "WHERE tenant_id = %s AND kb_id = %s",
                    (ctx.tenant_id, ctx.kb_id),
                )
            stats["postgres"] = write_postgres(pg_conn, ctx, chunks, dim, collection)
            stats.update(write_derived_postgres(pg_conn, ctx, chunks))
            if commit:
                pg_conn.commit()
        except Exception as exc:
            stats["notes"].append(f"Postgres 写入失败: {exc}")
            try:
                pg_conn.rollback()
            except Exception:
                pass

    return stats
