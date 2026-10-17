"""GraphRAG 检索：PG 图读取（实体 → 扩展 → 关联块 → 回 backend 打分）。

契约第 3 节 `graph` 通道：

  - 依赖 PG 图存储，连接用环境变量 `DATABASE_APP_URL`（**非属主角色 `kf_app`**，
    受 RLS 约束）。连接后**必须先** `set_config('app.tenant_id', <租户>, true)`（等价
    `SET LOCAL app.tenant_id`），否则 RLS 默认拒绝全部访问。
  - `DATABASE_APP_URL` 缺失 / psycopg 不可用 / 连不上 → 通道**跳过 + note**，绝不报错。
  - 流程：查询切词 → 规范化词 + 相邻二元组 匹配 `graph_entities.norm_name`
    → 命中实体沿 `graph_edges` 扩展 `max_hops` 跳 → 取 `graph_entity_chunks.chunk_id`
    → 回 backend 取 payload 打分；命中实体名记进 trace。
  - 聚合档位（`mode=community_summary`）额外产出社区摘要 evidence：
    `chunk_id="community:<id>"`、`chunk_type="community_summary"`、`content=summary`，
    并带 `community_chunk_ids` 以可回溯原文。

  - **psycopg 可选导入**：缺失时视为「图存储未配置」。
  - MemoryBackend 走同语义的 `graph_search`（从 seed 的 `graph` 字段），使无依赖也能测。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .backends import bm25_rank, graph_query_terms, normalize_entity

try:  # psycopg 可选：缺失不阻断检索，只是图通道跳过
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - 环境相关
    psycopg = None


def app_database_url() -> str:
    """应用角色连接串（`kf_app`，受 RLS 约束）。未配置返回空串。"""
    return (os.environ.get("DATABASE_APP_URL") or "").strip()


class GraphStore:
    """只读 PG 图访问（每次 search 独立连接，RLS 会话级生效）。"""

    def __init__(self, dsn: Optional[str] = None, *, timeout: float = 5.0) -> None:
        self._dsn = dsn
        self.timeout = timeout

    def dsn(self) -> str:
        return (self._dsn if self._dsn is not None else app_database_url()).strip()

    def available(self) -> bool:
        return bool(self.dsn()) and psycopg is not None

    def search(self, *, tenant: str, kb: str, query: str, max_hops: int = 2,
               mode: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """读图并返回 `{chunk_ids, entities_matched, communities, entities_reached}`。

        任何异常都收敛进 `error` 字段，交由上层「跳过 + note」，绝不向上抛。
        """
        result: Dict[str, Any] = {
            "chunk_ids": [], "entities_matched": [], "entities_reached": [],
            "communities": [], "error": None,
        }
        dsn = self.dsn()
        if not dsn or psycopg is None:
            result["error"] = "Graph 存储未配置（DATABASE_APP_URL 缺失）"
            return result
        try:
            with psycopg.connect(dsn, connect_timeout=self.timeout) as conn:
                with conn.cursor() as cur:
                    # RLS 会话：必须显式声明租户，否则策略默认拒绝全部行
                    cur.execute("SELECT set_config('app.tenant_id', %s, true)", (str(tenant),))
                    matched_ids, matched_norms = self._match_entities(cur, kb, query)
                    if not matched_ids:
                        return result
                    reached_ids = self._expand(cur, kb, matched_ids, max_hops)
                    chunk_ids = self._chunk_ids(cur, kb, matched_ids + reached_ids)
                    norm_of = {eid: norm for eid, norm in zip(matched_ids, matched_norms)}
                    result["chunk_ids"] = chunk_ids
                    result["entities_matched"] = matched_norms
                    result["entities_reached"] = [
                        eid for eid in reached_ids if eid not in norm_of
                    ]
                    if mode == "community_summary":
                        result["communities"] = self._communities(cur, kb, set(matched_ids + reached_ids))
        except Exception as exc:  # 连不上 / SQL 失败：跳过
            result["error"] = f"Graph 存储不可用: {exc}"
        return result

    # ---- 内部查询 ----
    def _match_entities(self, cur: Any, kb: str, query: str):
        """规范化词 + 相邻二元组 + 子串 匹配 `graph_entities.norm_name`。"""
        cands = graph_query_terms(query)
        qnorm = normalize_entity(query)
        cur.execute(
            "SELECT entity_id, norm_name FROM graph_entities "
            "WHERE kb_id = %s AND (norm_name = ANY(%s) OR position(norm_name in %s) > 0)",
            (kb, cands, qnorm),
        )
        rows = cur.fetchall()
        ids = [row[0] for row in rows]
        norms = [row[1] for row in rows]
        return ids, norms

    def _expand(self, cur: Any, kb: str, seeds: List[str], max_hops: int) -> List[str]:
        reached: List[str] = []
        seen = set(seeds)
        frontier = set(seeds)
        for _ in range(max(0, int(max_hops))):
            if not frontier:
                break
            frontier_list = list(frontier)
            cur.execute(
                "SELECT src_entity_id, dst_entity_id FROM graph_edges "
                "WHERE kb_id = %s AND (src_entity_id = ANY(%s) OR dst_entity_id = ANY(%s))",
                (kb, frontier_list, frontier_list),
            )
            nxt = set()
            for src, dst in cur.fetchall():
                if src in frontier and dst not in seen:
                    nxt.add(dst)
                if dst in frontier and src not in seen:
                    nxt.add(src)
            if not nxt:
                break
            seen |= nxt
            reached.extend(sorted(nxt))
            frontier = nxt
        return reached

    def _chunk_ids(self, cur: Any, kb: str, entity_ids: List[str]) -> List[str]:
        if not entity_ids:
            return []
        cur.execute(
            "SELECT DISTINCT chunk_id FROM graph_entity_chunks "
            "WHERE kb_id = %s AND entity_id = ANY(%s)",
            (kb, entity_ids),
        )
        return [row[0] for row in cur.fetchall()]

    def _communities(self, cur: Any, kb: str, entity_ids: set) -> List[Dict[str, Any]]:
        cur.execute(
            "SELECT community_id, summary, chunk_ids, members FROM graph_communities WHERE kb_id = %s",
            (kb,),
        )
        out: List[Dict[str, Any]] = []
        for community_id, summary, chunk_ids, members in cur.fetchall():
            member_ids = set(members or [])
            if entity_ids and not (member_ids & entity_ids):
                continue
            out.append({
                "id": community_id,
                "summary": summary or "",
                "chunk_ids": list(chunk_ids or []),
            })
        return out


def graph_recall(backend: Any, *, tenant: str, kb: str, query: str,
                 filters: Optional[Dict[str, Any]] = None, top_k: int = 20,
                 collection: Optional[str] = None, max_hops: int = 2,
                 mode: Optional[str] = None) -> Dict[str, Any]:
    """图通道统一入口：Memory 走 backend.graph_search，其余走 PG。

    返回 `{"hits": [...], "entities_matched": [...], "communities_used": n, "note": str|None}`。
    """
    if hasattr(backend, "graph_search"):
        result = backend.graph_search(query, filters=filters, top_k=top_k,
                                      collection=collection, max_hops=max_hops, mode=mode)
        result.setdefault("note", None)
        return result

    store = GraphStore()
    if not store.available():
        return {"hits": [], "entities_matched": [], "communities_used": 0,
                "note": "Graph 存储未配置（DATABASE_APP_URL 缺失）"}

    found = store.search(tenant=tenant, kb=kb, query=query,
                         max_hops=max_hops, mode=mode, limit=top_k)
    if found.get("error"):
        return {"hits": [], "entities_matched": [], "communities_used": 0,
                "note": str(found["error"])}

    docs: List[Dict[str, Any]] = []
    for cid in found.get("chunk_ids") or []:
        chunk = backend.get_chunk(cid, collection=collection)
        if chunk:
            docs.append(chunk)
    hits = bm25_rank(query, docs, top_k, "graph")

    communities_used = 0
    if mode == "community_summary":
        for comm in found.get("communities") or []:
            hits.append({
                "chunk": {
                    "chunk_id": "community:" + str(comm.get("id")),
                    "doc_id": "community:" + str(comm.get("id")),
                    "chunk_type": "community_summary",
                    "content": str(comm.get("summary") or ""),
                    "community_chunk_ids": list(comm.get("chunk_ids") or []),
                    "score": 1.0,
                    "retrieval_channel": "graph",
                },
                "score": 1.0,
                "channel": "graph",
            })
            communities_used += 1

    hits.sort(key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
    return {
        "hits": hits,
        "entities_matched": list(found.get("entities_matched") or []),
        "communities_used": communities_used,
        "note": None,
    }
