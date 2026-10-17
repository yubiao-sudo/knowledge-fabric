"""增强：上下文前缀 + 假设问题 + 关键词。

三件事，全部**可选且有成本护栏**——生成类调用按模型注册表用 vLLM 承载，
未配置或超预算时静默跳过（只做零成本的词法关键词抽取）。

1. **上下文前缀**（`context_prefix`）：为每个子块生成 1–2 句文档级上下文。
   这是解决"小块脱离上下文后语义漂移"的关键——检索命中小块、喂给 LLM 时
   带上所属章节与文档背景，答案质量提升显著。
2. **假设问题**（`chunk_type=question`）：每块生成 3–5 个可回答问题，
   与正文一同索引。它让"用户提问的措辞"与"文档的措辞"之间的鸿沟被填平。
3. **关键词**（`keywords`）：零成本词法抽取，喂稀疏侧，补向量对型号 / 专名的弱项。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .clients import LLMClient
from .embed import extract_keywords, tokenize_for_sparse
from .models import Chunk

_CONTEXT_PROMPT = (
    "你是文档索引助手。下面给出一个文档片段的上下文（章节路径）与片段正文。"
    "请用 1-2 句简洁的中文说明【这个片段在整个文档中讲的是什么、属于哪个部分】，"
    "只输出这两句话，不要复述原文，不要加任何前缀或解释。"
)

_QUESTION_PROMPT = (
    "你是文档索引助手。请针对下面的文档片段，生成 {n} 个用户可能会提出的、"
    "且该片段能够回答的问题。要求：与用户真实提问措辞接近（口语化、含具体型号/参数），"
    "每行一个问题，不要编号，不要输出任何其他内容。"
)

# 每块生成问题数为 3–5
DEFAULT_QUESTIONS_PER_CHUNK = 3
# 大库成本护栏：单文档生成类调用的最大块数
DEFAULT_MAX_GENERATED_CHUNKS = 400


def _section_hint(chunk: Chunk) -> str:
    path = [p for p in (chunk.section_path or []) if p and p != "(根)"]
    return " > ".join(path) if path else ""


def build_context_prefix(chunk: Chunk, llm: LLMClient) -> Optional[str]:
    hint = _section_hint(chunk)
    user = f"章节路径：{hint or '(无)'}\n片段正文：\n{chunk.content[:1200]}"
    try:
        text = llm.chat(
            [{"role": "system", "content": _CONTEXT_PROMPT}, {"role": "user", "content": user}],
            max_tokens=120,
            temperature=0.2,
        )
    except Exception:
        return None
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:400] or None


def build_hypothetical_questions(
    chunk: Chunk, llm: LLMClient, n: int = DEFAULT_QUESTIONS_PER_CHUNK
) -> List[str]:
    user = f"片段正文：\n{chunk.content[:1200]}"
    try:
        text = llm.chat(
            [{"role": "system", "content": _QUESTION_PROMPT.format(n=n)},
             {"role": "user", "content": user}],
            max_tokens=200,
            temperature=0.5,
        )
    except Exception:
        return []
    out: List[str] = []
    for line in (text or "").split("\n"):
        q = re.sub(r"^\s*(?:[-*·•]|\d+[.)、．])\s*", "", line).strip()
        if len(q) >= 4 and q not in out:
            out.append(q)
        if len(out) >= n:
            break
    return out


def enrich_chunks(
    chunks: Sequence[Chunk],
    *,
    llm: Optional[LLMClient] = None,
    with_context: bool = True,
    with_questions: bool = True,
    questions_per_chunk: int = DEFAULT_QUESTIONS_PER_CHUNK,
    max_generated_chunks: int = DEFAULT_MAX_GENERATED_CHUNKS,
    doc_id: str = "",
    version: int = 1,
) -> Dict[str, Any]:
    """就地增强 chunks，并返回新产出的 question 子块列表与统计。

    返回::

        {"stats": {...}, "question_chunks": [Chunk, ...]}

    question 子块的 `parent_chunk_id` 指向被增强块的父块，保证 small-to-big
    回填时与正文同源；`meta["derived_from"]` 记录来源块，便于溯源与去重。
    """
    stats: Dict[str, Any] = {
        "chunks": len(chunks),
        "context_prefix": 0,
        "questions": 0,
        "keywords": 0,
        "skipped_llm": False,
        "notes": [],
    }
    question_chunks: List[Chunk] = []

    # 零成本：关键词抽取始终执行（稀疏侧受益，且不花生成预算）
    for chunk in chunks:
        if not chunk.keywords:
            chunk.keywords = extract_keywords(chunk.content, chunk.lang)
            if chunk.keywords:
                stats["keywords"] += 1

    use_llm = llm is not None and (with_context or with_questions)
    if not use_llm:
        stats["skipped_llm"] = True
        stats["notes"].append("未配置生成端点，仅执行词法关键词抽取")
        return {"stats": stats, "question_chunks": question_chunks}

    budget = max_generated_chunks if max_generated_chunks > 0 else len(chunks)
    if len(chunks) > budget:
        stats["notes"].append(f"超出生成护栏（{len(chunks)} > {budget}），仅前 {budget} 块做生成增强")

    for index, chunk in enumerate(chunks):
        if index >= budget:
            break
        # 原子块（表格/公式/图片）不做上下文改写，避免破坏结构完整性
        if chunk.chunk_type in ("table", "formula", "figure"):
            continue

        if with_context and not chunk.context_prefix:
            prefix = build_context_prefix(chunk, llm)
            if prefix:
                chunk.context_prefix = prefix
                stats["context_prefix"] += 1

        if with_questions:
            questions = build_hypothetical_questions(chunk, llm, questions_per_chunk)
            for q_index, question in enumerate(questions):
                stats["questions"] += 1
                question_chunks.append(
                    Chunk(
                        chunk_id=f"{chunk.chunk_id}:Q{q_index + 1}",
                        chunk_type="question",
                        content=question,
                        lang=chunk.lang,
                        page=chunk.page,
                        order=chunk.order,
                        parent_chunk_id=chunk.parent_chunk_id,
                        context_prefix=chunk.context_prefix,
                        section_path=list(chunk.section_path),
                        acl=list(chunk.acl),
                        provenance=chunk.provenance,
                        keywords=extract_keywords(question, chunk.lang),
                        meta={
                            "role": "child",
                            "derived_from": chunk.chunk_id,
                            "doc_id": doc_id,
                            "version": version,
                        },
                    )
                )

    return {"stats": stats, "question_chunks": question_chunks}
