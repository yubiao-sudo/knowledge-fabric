"""知识库导航：告诉 Agent / 使用者「有什么库、问题该问哪个库」。

## 为什么需要它

Agent 的拒答**本身是正确的**（我们实测过：问 `acme/final` 一个只有 kf_ab.pdf 的库
「灯塔的设计目标」，可达分 0.327 < 0.45 → 拒答），但**拒答时不说"去哪找"**，
使用者只会觉得"这系统坏了"。真实场景下库会越来越多，这是可用性上的硬需求。

本模块只做两件务实的事：
  1. `list_kbs()` —— 列出有哪些知识库、各有多少文档与块（人对齐认知）；
  2. `discover()` —— 拿一个问题在**所有库**上探一遍，按可达分排序，
     告诉你"这个问题该问哪个库"（Agent 对齐认知）。

数据来源优先用 Postgres（一次查询拿全量，最可靠）；拿不到时退回逐库调检索服务。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class KBInfo:
    """一个知识库的概况。"""

    tenant: str
    kb: str
    docs: int = 0
    chunks: int = 0
    titles: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.tenant}/{self.kb}"

    def as_dict(self) -> Dict[str, Any]:
        return {"tenant": self.tenant, "kb": self.kb, "docs": self.docs,
                "chunks": self.chunks, "titles": self.titles}


@dataclass
class DiscoveryHit:
    """在某个库上探测一次的结果。"""

    tenant: str
    kb: str
    evidence: int = 0
    answerability: float = 0.0
    refused: bool = False
    note: str = ""
    top_snippet: str = ""
    answerability_available: bool = True   # False = 算不出来，**不是** 0 分

    @property
    def key(self) -> str:
        return f"{self.tenant}/{self.kb}"

    def as_dict(self) -> Dict[str, Any]:
        return {"tenant": self.tenant, "kb": self.kb, "evidence": self.evidence,
                "answerability": (round(self.answerability, 4)
                                  if self.answerability_available else None),
                "refused": self.refused, "note": self.note,
                "top_snippet": self.top_snippet}

    @property
    def answerability_display(self) -> str:
        """算不出来时显示 N/A——把"无法计算"画成 0.0000 会误导排序与判断。"""
        return f"{self.answerability:.4f}" if self.answerability_available else "N/A"


def list_kbs(dsn: Optional[str] = None) -> List[KBInfo]:
    """列出所有有活跃块的知识库（含文档数与前几个文档名）。"""
    dsn = dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        return []
    try:
        import psycopg
    except Exception:
        return []
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.tenant_id, c.kb_id,
                       count(DISTINCT c.doc_id) AS docs,
                       count(*) AS chunks
                  FROM chunks c
                 WHERE c.status = 'active'
              GROUP BY 1, 2
              ORDER BY 4 DESC
                """
            )
            rows = cur.fetchall()
            out: List[KBInfo] = []
            for tenant, kb, docs, chunks in rows:
                cur.execute(
                    """
                    SELECT DISTINCT doc_id FROM chunks
                     WHERE status='active' AND tenant_id=%s AND kb_id=%s
                     ORDER BY 1 LIMIT 6
                    """,
                    (tenant, kb),
                )
                titles = [r[0] for r in cur.fetchall()]
                out.append(KBInfo(tenant, kb, int(docs), int(chunks), titles))
            return out
    except Exception:
        return []


def discover(question: str, kbs: List[KBInfo], *,
             retrieve_url: str = "http://127.0.0.1:8082",
             profile: str = "", top: int = 5, timeout: float = 300.0) -> List[DiscoveryHit]:
    """拿一个问题在所有库上探一遍，按可达分降序返回。

    用**归一化可达分**（`trace.answerability`）排序，而不是融合分——后者对
    "能否回答"没有区分度（RRF 上界 1/(k+1)，可答与不可答都是 0.0164）。
    """
    hits: List[DiscoveryHit] = []
    url = f"{retrieve_url.rstrip('/')}/v1/search"
    for info in kbs:
        body: Dict[str, Any] = {"query": question, "tenant": info.tenant,
                                "kb": info.kb, "top_k": top}
        if profile:
            body["profile"] = profile
        try:
            resp = httpx.post(url, json=body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            hits.append(DiscoveryHit(info.tenant, info.kb, note=f"检索失败: {exc}"))
            continue
        trace = data.get("trace") or {}
        evidence = data.get("evidence") or []
        raw = trace.get("answerability")
        hit = DiscoveryHit(
            tenant=info.tenant, kb=info.kb,
            evidence=len(evidence),
            answerability=float(raw or 0.0),
            answerability_available=raw is not None,
            refused=bool(trace.get("refused")),
            note="；".join(trace.get("notes") or []) if trace.get("refused") else "",
        )
        if evidence:
            hit.top_snippet = (evidence[0].get("content") or "")[:120].replace("\n", " ")
        hits.append(hit)
    # 排序：先按"有解"（拒答排后），再按可达分降序；算不出分的排在算出分的后面
    hits.sort(key=lambda h: (h.refused, not h.answerability_available, -h.answerability))
    return hits
