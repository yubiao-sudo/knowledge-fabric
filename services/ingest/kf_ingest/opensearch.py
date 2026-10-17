"""OpenSearch 分语言 BM25 稀疏索引（摄入侧写入）。

契约要点（第 3 / 6 节）：

  - 仅当 `OPENSEARCH_URL` 已配置时才写入；**未配置 / 连不上只记 note，不报错、不中断摄入**。
  - 索引名 `kf_{tenant}_{kb}_{lang}`（lang 用块的语言标签）。
  - 分析器：zh/ja → `cjk`；en/ko → `english`；其它 → `standard`。
  - 写入 `chunk_id` / `doc_id` / `version` / `status` / `kb_id` / `content`，
    以 `chunk_id` 作 `_id` 保证幂等。
  - 换版本时把旧版本文档的 `status` 置 `stale`（检索侧按 status=active + version 过滤）。

检索侧按**同一索引名与 schema**去查，故键名必须与此处一致。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .models import Chunk, IngestContext

_CJK_LANGS = frozenset({"zh", "ja"})
_EN_LANGS = frozenset({"en", "ko"})


def analyzer_for_lang(lang: str) -> str:
    code = (lang or "").lower()
    if code in _CJK_LANGS:
        return "cjk"
    if code in _EN_LANGS:
        return "english"
    return "standard"


def index_name(tenant: str, kb: str, lang: str) -> str:
    code = (lang or "unknown") or "unknown"
    return f"kf_{tenant}_{kb}_{code}".replace("-", "_")


def open_search_url() -> str:
    return (os.environ.get("OPENSEARCH_URL") or "").strip()


def build_mapping(analyzer: str) -> Dict[str, Any]:
    return {
        "settings": {"analysis": {}},
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "version": {"type": "integer"},
                "status": {"type": "keyword"},
                "kb_id": {"type": "keyword"},
                "lang": {"type": "keyword"},
                "content": {"type": "text", "analyzer": analyzer},
            }
        },
    }


def build_bulk_body(ctx: IngestContext, chunks: Sequence[Chunk]) -> str:
    """ndjson bulk 体：`_id = chunk_id`（同块重写即幂等覆盖）。"""
    lines: List[str] = []
    for chunk in chunks:
        action = {
            "index": {
                "_index": index_name(ctx.tenant_id, ctx.kb_id, chunk.lang),
                "_id": chunk.chunk_id,
            }
        }
        document = {
            "chunk_id": chunk.chunk_id,
            "doc_id": ctx.doc_id,
            "version": ctx.version,
            "status": "active",
            "kb_id": ctx.kb_id,
            "lang": chunk.lang,
            "content": chunk.content,
        }
        lines.append(json.dumps(action, ensure_ascii=False))
        lines.append(json.dumps(document, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def index_chunks_opensearch(
    ctx: IngestContext,
    chunks: Sequence[Chunk],
    *,
    url: Optional[str] = None,
    timeout: float = 10.0,
    http: Any = None,
) -> Dict[str, Any]:
    """按语言把块写入 OpenSearch。未配置 / 连不上只记 note。`http` 可注入以便测试。"""
    stats: Dict[str, Any] = {"indexed": 0, "stale": 0, "indices": [], "notes": []}
    base = (url if url is not None else open_search_url()).strip()
    if not base:
        stats["notes"].append(
            "未配置 OPENSEARCH_URL，跳过分语言稀疏索引（检索侧走 Qdrant 稀疏）"
        )
        return stats
    if not chunks:
        return stats

    if http is None:
        try:
            import httpx as http  # noqa: F811
        except Exception as exc:
            stats["notes"].append(f"httpx 不可用，跳过 OpenSearch 索引: {exc}")
            return stats

    base = base.rstrip("/")
    try:
        indices: Dict[str, str] = {}
        for chunk in chunks:
            name = index_name(ctx.tenant_id, ctx.kb_id, chunk.lang)
            indices.setdefault(name, analyzer_for_lang(chunk.lang))
        for name, analyzer in indices.items():
            _ensure_index(http, base, name, analyzer, stats, timeout)
        stats["indices"] = sorted(indices.keys())

        body = build_bulk_body(ctx, chunks)
        if body:
            resp = http.post(
                f"{base}/_bulk",
                content=body.encode("utf-8"),
                headers={"Content-Type": "application/x-ndjson"},
                timeout=timeout,
            )
            stats["indexed"] = _count_bulk_success(resp)
            if getattr(resp, "status_code", 200) >= 400:
                stats["notes"].append(f"OpenSearch _bulk 返回 {resp.status_code}")

        stats["stale"] = _mark_stale(http, base, ctx, sorted(indices.keys()), stats, timeout)
    except Exception as exc:
        stats["notes"].append(f"OpenSearch 索引失败（已跳过）: {exc}")
    return stats


def _ensure_index(http: Any, base: str, name: str, analyzer: str,
                  stats: Dict[str, Any], timeout: float) -> None:
    try:
        resp = http.get(f"{base}/{name}", timeout=timeout)
        if getattr(resp, "status_code", 200) == 404:
            http.put(f"{base}/{name}", json=build_mapping(analyzer), timeout=timeout)
    except Exception as exc:
        stats["notes"].append(f"OpenSearch 建索引 {name} 失败: {exc}")


def _mark_stale(http: Any, base: str, ctx: IngestContext, names: Sequence[str],
                stats: Dict[str, Any], timeout: float) -> int:
    """把同一文档旧版本置 stale（不物理删除，检索侧按 status 过滤）。"""
    total = 0
    for name in names:
        try:
            resp = http.post(
                f"{base}/{name}/_update_by_query",
                json={
                    "query": {
                        "bool": {
                            "must": [{"term": {"doc_id": ctx.doc_id}}],
                            "must_not": [{"term": {"version": ctx.version}}],
                        }
                    },
                    "script": {"source": "ctx._source.status = 'stale'"},
                },
                timeout=timeout,
            )
            total += int((getattr(resp, "json", lambda: {})() or {}).get("updated", 0) or 0)
        except Exception as exc:
            stats["notes"].append(f"OpenSearch 置 stale 失败 {name}: {exc}")
    return total


def _count_bulk_success(resp: Any) -> int:
    try:
        payload = resp.json() or {}
    except Exception:
        return 0
    items = payload.get("items") or []
    count = 0
    for item in items:
        outcome = (item or {}).get("index") or {}
        if not outcome.get("error"):
            count += 1
    return count
