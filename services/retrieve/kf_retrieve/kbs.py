"""知识库目录：列出/读取某个租户下有哪些库、各有多少文档与切片。

## 为什么需要这个模块

此前底座**没有**"列知识库"的 HTTP 接口：
* 网关 `GET /v1/kb/{id}` 是占位实现（直接返回 `status: not_provisioned`，不读库）；
* 唯一能列库的是 `examples/kb_navigator.py`，而它**直连 Postgres**。

让 agent 拿着数据库密码去连 Postgres 是不可接受的——所以补上这层只读查询，
让客户端只需要一个 HTTP 地址 + 令牌。

## 为什么走数据库函数而不是直接查表

`chunk_vectors` 启用了 RLS（`USING (tenant_id = current_setting('app.tenant_id', true))`），
服务角色 `kf_app` 不设置该变量就**一行都看不到**——实测直接 JOIN 返回 0 行，
目录整个是空的。而"列出有哪些库"天然是跨租户动作，靠 set_config 只能解决单租户。

所以走 `db/006_kb_catalog.sql` 里的 `kf_kb_catalog()` / `kf_kb_docs()`：
SECURITY DEFINER、只读、只返回计数（不含任何正文）。**绕过 RLS 的范围被收窄到一个
不可注入的聚合函数**，比给 `kf_app` 加 BYPASSRLS（整库级）安全得多。

数据来源是 `chunks` 实时聚合（而不是 `knowledge_bases.stats`）：
`stats` 是摄入时写死的快照，会随重摄入/清理漂移；实时聚合才是真的。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def db_dsn() -> str:
    return (os.environ.get("DATABASE_APP_URL")
            or os.environ.get("DATABASE_URL") or "").strip()


def _connect():
    import psycopg

    return psycopg.connect(db_dsn(), connect_timeout=5)


def list_kbs(tenant: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """列出有活跃切片的知识库。`tenant=None` 表示全部租户（仅本机控制台会用到）。"""
    if not db_dsn():
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM kf_kb_catalog(%s)", (tenant,))
            rows = cur.fetchall()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for tenant_id, kb_id, documents, chunks, with_dense, last_at in rows[:limit]:
        out.append({
            "tenant": tenant_id,
            "kb": kb_id,
            "documents": int(documents),
            "chunks": int(chunks),
            "chunks_with_dense": int(with_dense),
            # 稠密向量覆盖率：<1 通常意味着含重摄入前写入的旧切片，
            # 检索会退化成纯稀疏通道——直接暴露出来，省得别人猜。
            "dense_ratio": round(int(with_dense) / max(1, int(chunks)), 4),
            "last_ingested_at": last_at.isoformat() if last_at else None,
        })
    return out


def get_kb(tenant: str, kb: str) -> Optional[Dict[str, Any]]:
    """单个知识库的元信息 + 档位/策略 + 文档明细。"""
    if not db_dsn():
        return None
    meta: Dict[str, Any] = {}
    docs: List[Dict[str, Any]] = []
    try:
        with _connect() as conn, conn.cursor() as cur:
            # knowledge_bases 没有 RLS，可以直接查
            cur.execute(
                """
                SELECT kb_id, tenant_id, name, chunking_strategy, retrieval_profile,
                       graph_enabled, languages, created_at
                  FROM knowledge_bases
                 WHERE tenant_id = %s AND kb_id = %s
                """,
                (tenant, kb),
            )
            row = cur.fetchone()
            if row:
                meta = {
                    "kb": row[0], "tenant": row[1], "name": row[2],
                    "chunking_strategy": row[3], "retrieval_profile": row[4],
                    "graph_enabled": bool(row[5]),
                    "languages": list(row[6] or []),
                    "created_at": row[7].isoformat() if row[7] else None,
                }
            cur.execute("SELECT * FROM kf_kb_docs(%s, %s)", (tenant, kb))
            docs = [
                {"doc_id": r[0], "chunks": int(r[1]), "chunks_with_dense": int(r[2]),
                 "last_ingested_at": r[3].isoformat() if r[3] else None}
                for r in cur.fetchall()
            ]
    except Exception:
        return None

    if not meta and not docs:
        return None
    total_chunks = sum(d["chunks"] for d in docs)
    with_dense = sum(d["chunks_with_dense"] for d in docs)
    return {
        **meta,
        "tenant": tenant,
        "kb": kb,
        "documents": len(docs),
        "chunks": total_chunks,
        "chunks_with_dense": with_dense,
        "dense_ratio": round(with_dense / max(1, total_chunks), 4),
        "doc_list": docs[:50],
    }
