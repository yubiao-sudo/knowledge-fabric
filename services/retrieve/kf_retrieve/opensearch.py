"""OpenSearch 分语言 BM25 稀疏检索（契约第 3 节 `sparse` 通道）。

  - `OPENSEARCH_URL` 已配置且连得通 → 稀疏通道走 OpenSearch BM25：
    索引名 `kf_{tenant}_{kb}_{lang}`（与摄入侧 `kf_ingest/opensearch.py:index_name` 同名，
    lang 由 `query.detect_lang` 决定）；分析器 zh/ja→`cjk`、en/ko→`english`、其它→`standard`；
    查询加 `status=active` 与 `version` 过滤；返回 `chunk_id + score` 后由上层回 backend 取 payload。
  - 未配置 → 不走本模块（上层回退 Qdrant 稀疏，保留 payload lang 过滤）。
  - 配置了但连不上 → 返回 note，由上层自动回退 Qdrant 稀疏（不得报错）。

用 httpx 直连 REST（不引入 opensearch-py 依赖），与摄入侧写入实现保持一致。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_CJK_LANGS = frozenset({"zh", "ja"})
_EN_LANGS = frozenset({"en", "ko"})


def open_search_url() -> str:
    return (os.environ.get("OPENSEARCH_URL") or "").strip()


def analyzer_for_lang(lang: str) -> str:
    code = (lang or "").lower()
    if code in _CJK_LANGS:
        return "cjk"
    if code in _EN_LANGS:
        return "english"
    return "standard"


def index_name(tenant: str, kb: str, lang: str) -> str:
    """索引名：`kf_{tenant}_{kb}_{lang}`（破折号归一为下划线，与摄入侧一致）。"""
    code = (lang or "unknown") or "unknown"
    return f"kf_{tenant}_{kb}_{code}".replace("-", "_")


def build_query(query: str, *, version: Optional[int] = None, top_k: int = 20) -> Dict[str, Any]:
    """构造 BM25 查询 DSL：match content + status=active（+ version，若已知）。"""
    filters: List[Dict[str, Any]] = [{"term": {"status": "active"}}]
    if version is not None:
        filters.append({"term": {"version": int(version)}})
    return {
        "size": max(1, int(top_k)),
        "_source": ["chunk_id"],
        "query": {
            "bool": {
                "must": [{"match": {"content": query}}],
                "filter": filters,
            }
        },
    }


def search(tenant: str, kb: str, lang: str, query: str, *, top_k: int = 20,
           version: Optional[int] = None, url: Optional[str] = None,
           timeout: float = 3.0, http: Any = None) -> Dict[str, Any]:
    """查 OpenSearch，返回 `{"hits": [{"chunk_id", "score"}], "note": str|None}`。"""
    base = (url if url is not None else open_search_url()).strip()
    if not base:
        return {"hits": [], "note": "未配置 OPENSEARCH_URL"}
    if http is None:
        try:
            import httpx as http  # noqa: F811
        except Exception as exc:  # pragma: no cover - 环境相关
            return {"hits": [], "note": f"httpx 不可用，回退 Qdrant 稀疏: {exc}"}

    name = index_name(tenant, kb, lang)
    try:
        resp = http.post(
            f"{base.rstrip('/')}/{name}/_search",
            json=build_query(query, version=version, top_k=top_k),
            timeout=timeout,
        )
        if getattr(resp, "status_code", 200) >= 400:
            return {"hits": [], "note": f"OpenSearch 返回 {resp.status_code}，回退 Qdrant 稀疏"}
        payload = resp.json() or {}
    except Exception as exc:
        return {"hits": [], "note": f"OpenSearch 不可用，回退 Qdrant 稀疏: {exc}"}

    hits: List[Dict[str, Any]] = []
    for item in (payload.get("hits") or {}).get("hits") or []:
        source = item.get("_source") or {}
        chunk_id = source.get("chunk_id")
        if chunk_id:
            hits.append({"chunk_id": str(chunk_id), "score": float(item.get("_score") or 0.0)})
    return {"hits": hits, "note": None}
