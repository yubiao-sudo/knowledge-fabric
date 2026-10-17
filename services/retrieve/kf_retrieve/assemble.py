"""组装：父子结构处理 → 去重 → 预算硬限 → 抗「中间遗忘」排序 → 引用 → 拒答。

设计要点：

- **父块默认不替换 content**（`backfill_parent_content=False`）：命中子块的 content
  与 `char_range` 始终同源，引用自洽。父块改以 `parent_context` 字段按需附带，
  上限由 `parent_context_tokens` 控制（0 表示不带）。旧实现用父块全文替换 content，
  实测放大 3~15 倍且偏移仍指向子块，造成「读到的内容 ≠ 引用区间」。
- **去重按 chunk_id，不按父块**：多个不同子块命中同一父块时是不同信息，
  旧实现按父块去重会把多条命中压成 1 条。
- **近似去重**：64 位 SimHash 汉明距离（复刻 `clean.py`）剔除，只与已保留项比较。
- **预算硬限**：累计 token 不得超过 `max_context_tokens`；估算用启发式
  （CJK 1 token/字、拉丁 1.3 token/词、标点 0.5），不引入 transformers。
- **抗中间遗忘**：证据按分数降序后重排成「首尾强、中间弱」的形态
  （下标序列形如 `[1,3,5,...,6,4,2]`），缓解长上下文中段的注意力衰减。
- **ACL**：`principal` 为空则不过滤；提供时 deny 优先。`acl_default_allow`
  控制"无 ACL 规则"时的默认（默认 True 保持兼容，多租户应传 False）。
- **字符级引用**：evidence 除 content 外带上 `char_range` / `char_start` / `char_end`
  与 `source_uri`，并用 `offset_chunk_id` 显式标注偏移归属。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .sparse import hamming, simhash

# ---- token 启发式估算（复刻 kf_ingest/tokenize.py:heuristic_count）----
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z0-9\u00c0-\u024f]+")
_PUNCT = re.compile(r"[^\sA-Za-z0-9\u00c0-\u024f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

SIMHASH_MAX_DISTANCE = 3
MIN_LEN_NEAR = 120


def estimate_tokens(text: str) -> int:
    """启发式 token 估算：CJK 每字 ≈ 1；拉丁每词 ≈ 1.3；标点 ≈ 0.5。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    punct = len(_PUNCT.findall(text))
    return int(cjk + words * 1.3 + punct * 0.5) + 1


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（向量已 L2 归一，退化也安全）。"""
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def answerability_scores(query_vec: Optional[Sequence[float]],
                         candidates: Sequence[Dict[str, Any]],
                         top_n: int = 5) -> List[float]:
    """用**归一化余弦**给出答案可达分（降序）。

    为什么必须另算：`fusion` 后的分数（RRF / 加权）**只反映排名、不反映相似度**。
    真机实测（acme/demo，520 活跃块，hybrid_rerank）：

    | 查询 | RRF 融合分 |
    |---|---|
    | 可答「MX-100 的功率是多少」 | 0.0164 |
    | 不可答「今天天气怎么样，帮我订张机票」 | 0.0164 |

    两者**完全相同**，因为 RRF 的理论上界只有 1/(k+1)≈0.0164——排名第一就饱和。
    于是"按分数拒答"这条路在 RRF 档位下根本不成立（`strict_refuse` 要求 0.55，
    永远达不到 → 退回默认档），越界查询畅通无阻。

    同一批数据换成**稠密余弦**则分离干净：可答 0.64~0.77，不可答 0.29~0.41。

    取分优先用候选自带的 `answerability`（检索侧直接取 Qdrant 的 Cosine 分），
    其次才用 `query_vec` 与候选向量现算。
    """
    out: List[float] = []
    for cand in candidates[:top_n]:
        score = cand.get("answerability")
        if score is not None:
            out.append(float(score))
        elif query_vec is not None:
            vec = cand.get("dense_vector") or cand.get("query_vector")
            if vec:
                out.append(cosine(query_vec, vec))
    return sorted(out, reverse=True)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """按比例粗略截断（启发式估算，用于单块就超预算时的兜底）。"""
    if max_tokens <= 0:
        return ""
    total = estimate_tokens(text)
    if total <= max_tokens:
        return text
    ratio = max_tokens / total
    return text[: max(1, int(len(text) * ratio))]


# ------------------------------------------------------------------ 排序
def interleave_extremes(items: Sequence[Any]) -> List[Any]:
    """抗「中间遗忘」排序：把最相关的放首尾、次相关的放中间。

    输入为**按分数降序**的序列，输出重排后序列，其原下标顺序形如
    `[0,2,4,...,5,3,1]`（1 基即 `[1,3,5,...,6,4,2]`），让高相关项落在上下文两端。
    """
    n = len(items)
    if n <= 2:
        return list(items)
    last_odd = n - 1 if (n - 1) % 2 == 1 else n - 2
    order = list(range(0, n, 2)) + list(range(last_odd, 0, -2))
    return [items[i] for i in order]


# ------------------------------------------------------------------ ACL
def normalize_principal(principal: Any) -> Set[Tuple[str, str]]:
    """把多种 principal 写法归一成 `{(principal_type, principal_id)}` 集合。"""
    pairs: Set[Tuple[str, str]] = set()

    def add(item: Any) -> None:
        if isinstance(item, dict):
            t = item.get("principal_type") or item.get("type") or item.get("kind")
            i = item.get("principal_id") or item.get("id")
            if t and i:
                pairs.add((str(t), str(i)))
        elif isinstance(item, str):
            if ":" in item:
                t, i = item.split(":", 1)
                pairs.add((t.strip(), i.strip()))
            elif item:
                pairs.add(("*", item))

    if isinstance(principal, (list, tuple, set)):
        for item in principal:
            add(item)
    else:
        add(principal)
    return pairs


def acl_allows(acl: Optional[Sequence[Dict[str, Any]]], principal: Any,
               default_allow: bool = True) -> bool:
    """判断块的 ACL 是否放行给定 principal。

    规则：deny 优先；命中 allow（含 `tenant_public`、`principal_id=*`）则放行；
    完全没有 ACL 规则时按 `default_allow` 处理；有规则但无匹配 allow 则拒绝。

    `default_allow` 默认 True 保持既有行为；**多租户场景应传 False**
    （fail-closed：没声明 ACL 的块不因"没规则"而被公开）。
    """
    pairs = normalize_principal(principal)
    if not pairs:
        return True
    rules = list(acl or [])
    if not rules:
        return bool(default_allow)

    allowed = False
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        t = str(rule.get("principal_type") or rule.get("type") or "")
        i = str(rule.get("principal_id") or rule.get("id") or "")
        effect = str(rule.get("effect") or "allow").lower()
        matched = (t, i) in pairs or (t, "*") in pairs or i == "*" or t == "tenant_public"
        if not matched:
            continue
        if effect == "deny":
            return False
        allowed = True
    return allowed


def _clip_text(text: str, max_tokens: int) -> str:
    """按 token 预算裁剪父块上下文（用于 backfill 场景）。"""
    if max_tokens <= 0:
        return ""
    return truncate_to_tokens(text, max_tokens)


# ------------------------------------------------------------------ 组装主流程
def assemble(
    candidates: Sequence[Dict[str, Any]],
    *,
    parents: Optional[Dict[str, Dict[str, Any]]] = None,
    budget_tokens: int = 8000,
    threshold: Optional[float] = None,
    principal: Any = None,
    max_items: Optional[int] = None,
    acl_default_allow: bool = True,
    backfill_parent_content: bool = False,
    parent_context_tokens: int = 0,
    query_vec: Optional[Sequence[float]] = None,
    answerability_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """把融合/重排后的候选块组装成对外 evidence + citations。

    candidates 每项为 dict，字段：chunk_id / parent_chunk_id / content / score /
    source / page / chunk_type / retrieval_channel / doc_id / section /
    char_start / char_end / acl。

    **父块回填默认关闭**（`backfill_parent_content=False`）：
    回填的语义是"用父块全文替换命中子块的 content"，但字符偏移仍属于子块，
    于是出现「读到的内容 ≠ 引用指向的区间」，且实测放大 3~15 倍、把精准的
    320 token 子块换成上千 token 的整节。父块改以 `parent_context` 字段
    按需附带（受 `parent_context_tokens` 限制，0 表示不带）。
    """
    notes: List[str] = []
    parents = parents or {}

    items = sorted(candidates, key=lambda c: (-float(c.get("score", 0.0)), str(c.get("chunk_id"))))

    # 1) ACL 过滤（principal 为空则不过滤）
    if principal is not None and normalize_principal(principal):
        before = len(items)
        items = [c for c in items
                 if acl_allows(c.get("acl"), principal, default_allow=acl_default_allow)]
        hidden = before - len(items)
        if hidden:
            notes.append(f"ACL 过滤 {hidden} 块")

    # 2) 去重（同一 chunk_id 只留最高分；**不按父块合并**——多个不同子块命中同一
    #    父块时它们是不同的信息，旧实现按父块去重会把 3 条命中压成 1 条）
    merged: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for cand in items:
        key = str(cand.get("chunk_id") or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(dict(cand))

    # 3) 近似去重（SimHash 汉明距离；只与**已保留**项比较）
    merged = _dedupe_near(merged)

    # 4) 拒答：**优先用归一化答案可达分**（稠密余弦），而不是融合分。
    #    融合分（RRF）只反映排名、上界 1/(k+1)，对「能否回答」没有区分度
    #    ——真机实测可答与不可答的 RRF 分完全相同（都是 0.0164）。
    #    余弦分则能干净分离（可答 0.64~0.77 / 不可答 0.29~0.41）。
    a_scores = answerability_scores(query_vec, merged)
    normalized_top = a_scores[0] if a_scores else None
    top_score = float(merged[0].get("score", 0.0)) if merged else 0.0

    if answerability_threshold is not None and normalized_top is not None:
        if not merged or normalized_top < float(answerability_threshold):
            return {
                "evidence": [],
                "citations": [],
                "tokens_used": 0,
                "refused": True,
                "top_score": round(top_score, 6),
                "answerability": round(normalized_top, 6),
                "threshold_basis": "normalized_cosine",
                "notes": notes + [
                    f"归一化答案可达分 {normalized_top:.4f} 低于阈值 "
                    f"{float(answerability_threshold)}，按档位拒答"
                ],
            }
    else:
        # 走到这里有两种情况：① 档位本就用融合分判拒答；② 配置要求用余弦，
        # 但稠密通道不可用（无嵌入端点 / 无稠密向量）导致算不出分。
        # **两种情况都必须继续用融合分兜住**，绝不能因为"新判据算不出来"
        # 就静默放行——真机教训：融合分没有区分度，放行即等于不拒答。
        if answerability_threshold is not None and normalized_top is None:
            notes.append(
                "配置要求按归一化余弦判拒答，但稠密分数不可用（无嵌入端点或无稠密向量）；"
                "已退回融合分判定，该基准区分度低，拒答可能失效"
            )
        if threshold is not None and (not merged or top_score < float(threshold)):
            return {
                "evidence": [],
                "citations": [],
                "tokens_used": 0,
                "refused": True,
                "top_score": round(top_score, 6),
                "answerability": None,
                "threshold_basis": "fusion_score",
                "notes": notes + ["最高分低于答案可达阈值，按档位拒答"],
            }

    # 5) 预算硬限：贪婪装入，超预算的跳过（同分排序下先装高分）
    kept: List[Dict[str, Any]] = []
    used = 0
    budget = max(0, int(budget_tokens))
    for cand in merged:
        content = str(cand.get("content") or "")
        cost = estimate_tokens(content)
        if cost > budget:
            if not kept:
                trimmed = truncate_to_tokens(content, budget)
                if not trimmed:
                    continue
                cand = dict(cand)
                cand["content"] = trimmed
                cost = estimate_tokens(trimmed)
                kept.append(cand)
                used += cost
            continue
        if used + cost > budget:
            continue
        kept.append(cand)
        used += cost

    # 6) 抗「中间遗忘」排序（在分数降序基础上重排首尾）
    kept = interleave_extremes(kept)
    if max_items is not None:
        kept = kept[:max_items]

    # 补充：可选的父块上下文。默认只把父块作为 `parent_context` 附带（不替换 content），
    # 这样 content 与 char_range 始终同源、引用自洽；只有显式开启 backfill 才替换。
    if parent_context_tokens and parent_context_tokens > 0:
        for cand in kept:
            pid = cand.get("parent_chunk_id")
            parent = parents.get(str(pid)) if pid else None
            if parent is None:
                continue
            parent_text = str(parent.get("content") or "")
            if not parent_text or parent_text == str(cand.get("content") or ""):
                continue
            clipped = _clip_text(parent_text, int(parent_context_tokens))
            if not clipped:
                continue
            if backfill_parent_content:
                cand["content"] = clipped
            else:
                cand["parent_context"] = clipped

    evidence = [_to_evidence(c) for c in kept]
    citations = _to_citations(kept)
    return {
        "evidence": evidence,
        "citations": citations,
        "tokens_used": used,
        "refused": False,
        "top_score": round(top_score, 6),
        "answerability": round(normalized_top, 6) if normalized_top is not None else None,
        "threshold_basis": "normalized_cosine" if normalized_top is not None else "fusion_score",
        "notes": notes,
    }


def _exact_key(content: str) -> str:
    """把内容归一成"逐字比较键"：去掉**所有**空白。

    为什么需要这一层：SimHash 那条路径有 `MIN_LEN_NEAR=120` 的长度下限
    （太短的文本 SimHash 噪声大，容易误判），于是**短证据的重复永远不被剔除**
    —— 真机上同一段 40 字的规则因为来自两份不同的 doc_id，在结果里出现了两遍。
    逐字归一后比较是零误判的，短文本也该走。
    """
    return "".join(str(content or "").split())


def _dedupe_near(items: Sequence[Dict[str, Any]],
                 max_distance: int = SIMHASH_MAX_DISTANCE) -> List[Dict[str, Any]]:
    """重复剔除，两级：

    1. **逐字相同**（仅空白差异）——对所有长度生效，零误判；
    2. **近似重复**（SimHash 汉明距离 ≤ max_distance）——只对长文本生效。

    输入已按分数降序，因此只需与**已保留**项比较——已丢弃项没有比较价值，
    旧实现把已丢弃的哈希也留在比较集里，会让后续不同内容被误判为重复。
    """
    kept: List[Dict[str, Any]] = []
    hashes: List[int] = []
    seen_exact: set = set()
    for cand in items:
        content = str(cand.get("content") or "")
        exact = _exact_key(content)
        if exact:
            if exact in seen_exact:
                continue
            seen_exact.add(exact)
        if len(content) < MIN_LEN_NEAR:
            kept.append(cand)
            continue
        sh = simhash(content)
        if any(hamming(sh, prev) <= max_distance for prev in hashes):
            continue
        hashes.append(sh)
        kept.append(cand)
    return kept


def _to_evidence(cand: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "chunk_id": str(cand.get("chunk_id") or ""),
        "content": str(cand.get("content") or ""),
        "score": round(float(cand.get("score", 0.0)), 6),
        "source": str(cand.get("source") or ""),
        "page": int(cand.get("page") or 0),
    }
    pid = cand.get("parent_chunk_id")
    if pid:
        out["parent_chunk_id"] = str(pid)
    # 父块上下文（非替换语义）：content 与 char_range 仍属于命中子块，二者同源
    if cand.get("parent_context"):
        out["parent_context"] = str(cand["parent_context"])
    if cand.get("chunk_type"):
        out["chunk_type"] = str(cand["chunk_type"])
    if cand.get("retrieval_channel"):
        out["retrieval_channel"] = str(cand["retrieval_channel"])
    # 社区摘要 evidence（graph_global）：带 community_chunk_ids 以可回溯原文
    if cand.get("community_chunk_ids"):
        out["community_chunk_ids"] = [str(c) for c in cand["community_chunk_ids"]]

    # 字符级引用（可高亮 / 一键还原原文）。语义要点：偏移永远属于**实际命中的
    # 子块**——父块回填只替换 content，偏移不随之改变。因此必须显式标注归属：
    # `offset_chunk_id` 指出 `char_range` 是哪个 chunk 的偏移。父块回填时
    # evidence.chunk_id 仍是命中子块，与 offset_chunk_id 一致；content 则来自
    # `parent_chunk_id`。三者齐备才不含糊。
    source_uri = cand.get("source_uri") or cand.get("source")
    if source_uri:
        out["source_uri"] = str(source_uri)
    start = cand.get("char_start")
    end = cand.get("char_end")
    if start is not None:
        out["char_start"] = int(start)
        out["offset_chunk_id"] = str(cand.get("chunk_id") or "")
    if start is not None and end is not None:
        out["char_end"] = int(end)
        out["char_range"] = [int(start), int(end)]
    return out


def _to_citations(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int, Optional[int]]] = set()
    for cand in items:
        doc_id = str(cand.get("doc_id") or "")
        page = int(cand.get("page") or 0)
        start = cand.get("char_start")
        key = (doc_id, page, None if start is None else int(start))
        if key in seen:
            continue
        seen.add(key)
        cite: Dict[str, Any] = {"doc_id": doc_id, "page": page}
        if cand.get("section"):
            cite["section"] = str(cand["section"])
        if cand.get("source"):
            cite["url"] = str(cand["source"])
        if start is not None and cand.get("char_end") is not None:
            cite["char_range"] = [int(start), int(cand["char_end"])]
        citations.append(cite)
    return citations
