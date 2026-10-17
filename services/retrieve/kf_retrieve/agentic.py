"""Agentic RAG：多轮检索 + 饱和检测 + 预算硬限（契约第 4 节）。

  - 在 `run_search` 外层做多轮循环，**硬限** = `min(全局 3, 档位 max_rounds)`；
    全局 3 可被环境变量 `AGENTIC_MAX_ROUNDS` 覆盖。
  - 每轮后判定停止条件（满足即停）：
      1. 新增块数 < `min_new_evidence`（默认 1）→ `stop="saturation"`；
      2. 累计 token > `budget_tokens` → `stop="budget"`；
      3. 轮数达硬限 → `stop="max_rounds"`。
  - 下一轮查询**必须由生成端点（VLLM_URL / OPENAI_BASE_URL 派生的 `gen`）产出**；
    **无 `gen` 时不得伪造多轮**：只跑 1 轮，`agentic_rounds=1`、`agentic_stop="no_generator"`。

本模块只负责循环控制，单轮检索由注入的 `execute(query)` 完成，便于单测。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

DEFAULT_HARD_LIMIT = 3
DEFAULT_MIN_NEW = 1
DEFAULT_BUDGET_TOKENS = 8000


def global_hard_limit() -> int:
    """全局硬限：默认 3，可被 `AGENTIC_MAX_ROUNDS` 覆盖（非法值回落 3）。"""
    raw = (os.environ.get("AGENTIC_MAX_ROUNDS") or str(DEFAULT_HARD_LIMIT)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_HARD_LIMIT
    return max(1, value)


def round_limit(profile_max_rounds: Optional[int]) -> int:
    """实际硬限 = `min(全局硬限, 档位 max_rounds)`。"""
    hard = global_hard_limit()
    try:
        want = int(profile_max_rounds) if profile_max_rounds is not None else hard
    except (TypeError, ValueError):
        want = hard
    return max(1, min(hard, want))


def next_queries(gen: Callable[[str], str], query: str,
                 evidence: List[Dict[str, Any]]) -> List[str]:
    """由生成端点产出下一轮查询（基于已获证据指出「还缺什么」）。"""
    snippet = "\n".join(
        f"- {str(e.get('content') or '')[:200]}" for e in evidence[:5]
    )
    prompt = (
        "下面是已经检索到的证据摘要。请判断还缺少哪些关键信息，并给出 1 个最可能补全它们的"
        "检索查询，只输出查询本身，不要解释、不要编号。\n\n"
        f"原始问题：{query}\n已有证据：\n{snippet}"
    )
    try:
        text = (gen(prompt) or "").strip()
    except Exception:
        return []
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln][:1]


def run_agentic(
    execute: Callable[[str], Dict[str, Any]],
    *,
    query: str,
    gen: Optional[Callable[[str], str]],
    max_rounds: int,
    min_new_evidence: int = DEFAULT_MIN_NEW,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> Dict[str, Any]:
    """跑多轮检索，返回合并后的结果与轮次统计。

    `execute(q)` 返回单轮响应 dict（含 `evidence` / `citations` / `trace`）。
    """
    detail: List[Dict[str, Any]] = []
    merged: Dict[str, Dict[str, Any]] = {}
    citations: List[Dict[str, Any]] = []
    seen_cite: set = set()
    total_tokens = 0
    rounds = 0
    stop: Optional[str] = None
    last: Optional[Dict[str, Any]] = None
    current = query

    while True:
        result = execute(current)
        rounds += 1
        last = result

        new_chunks = 0
        for ev in result.get("evidence") or []:
            cid = str(ev.get("chunk_id") or "")
            if cid and cid not in merged:
                merged[cid] = ev
                new_chunks += 1
        for cite in result.get("citations") or []:
            key = json.dumps(cite, sort_keys=True, ensure_ascii=False, default=str)
            if key not in seen_cite:
                seen_cite.add(key)
                citations.append(cite)

        tokens = int((result.get("trace") or {}).get("tokens_used") or 0)
        total_tokens += tokens
        detail.append({"round": rounds, "queries": [current], "new_chunks": new_chunks, "tokens": tokens})

        if new_chunks < max(1, int(min_new_evidence)):
            stop = "saturation"
            break
        if total_tokens > int(budget_tokens):
            stop = "budget"
            break
        if rounds >= max(1, int(max_rounds)):
            stop = "max_rounds"
            break
        if gen is None:
            stop = "no_generator"
            break
        nxt = next_queries(gen, current, list(merged.values()))
        if not nxt:
            stop = "no_generator"
            break
        current = nxt[0]

    evidence = sorted(
        merged.values(),
        key=lambda e: (-float(e.get("score", 0.0) or 0.0), str(e.get("chunk_id") or "")),
    )
    return {
        "result": last,
        "evidence": evidence,
        "citations": citations,
        "rounds": rounds,
        "stop": stop,
        "detail": detail,
        "tokens": total_tokens,
    }
