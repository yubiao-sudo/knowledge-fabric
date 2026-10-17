"""
Knowledge Fabric · 切片策略评测（自查询 recall）

目的：把「切片方案」从玄学变成可 A/B 的工程参数。
方法：从每个 chunk 反向生成若干「它自己能回答的问题」，用这些问题去检索，
      看该 chunk 是否被召回。指标越高，说明该切片策略对检索越友好。

用法：
    # 对同一份文档跑多个切片策略，直接比较（无需数据库）
    python scripts/chunk_eval.py --path 手册.pdf \
        --strategies doc_structured,semantic,table_centric

    # 用已有的 chunk 集合评测（JSON: [{"chunk_id": "...", "content": "..."}, ...]）
    python scripts/chunk_eval.py --from-json chunks.json

    # 把评测结果写入 Postgres eval_runs（需 DATABASE_URL），供选参留痕
    python scripts/chunk_eval.py --path 手册.pdf --kb demo --doc d1 --persist

口径说明（重要）：
  检索器使用**与索引侧完全相同的稀疏编码**（kf_ingest.embed.tokenize_for_sparse），
  这样评测得分与线上检索行为同源，而不是各算一套。

  自查询问题生成有两种模式：
    - 有生成端点（VLLM_URL）：用 LLM 生成贴近真实用户措辞的问题，评测口径可信；
    - 无生成端点：退回模板式启发（取首句 + 关键词）。此时问题与原文存在较强词面
      重叠，得分会**偏乐观**，应视为上界参考，而非绝对质量。控制台会明确标注。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# 让脚本能直接 import 摄入侧的切片与编码实现（保证评测口径与线上一致）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INGEST_DIR = os.path.join(_REPO_ROOT, "services", "ingest")
if _INGEST_DIR not in sys.path:
    sys.path.insert(0, _INGEST_DIR)


# ---------------------------------------------------------------- 指标


def recall_at_k(ranked_ids: Sequence[str], relevant: set, k: int) -> float:
    """单查询 recall@k：前 k 命中里覆盖了多少相关项。"""
    if not relevant:
        return 0.0
    hit = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hit / len(relevant)


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(ranked_ids[:k])
        if cid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------- 检索器
def build_lexical_retriever(
    chunks: Sequence[Tuple[str, str]], k1: float = 1.2, b: float = 0.75
) -> Callable[[str, int], List[str]]:
    """BM25 词法检索器。分词复用索引侧的稀疏编码，保证口径同源。"""
    from kf_ingest.embed import tokenize_for_sparse

    doc_tokens: List[List[str]] = []
    doc_lengths: List[int] = []
    df: Counter = Counter()

    for _, content in chunks:
        tokens = tokenize_for_sparse(content)
        doc_tokens.append(tokens)
        doc_lengths.append(len(tokens))
        df.update(set(tokens))

    total = max(1, len(chunks))
    avg_len = (sum(doc_lengths) / total) or 1.0
    chunk_ids = [cid for cid, _ in chunks]

    def retrieve(query: str, top_k: int = 20) -> List[str]:
        query_tokens = tokenize_for_sparse(query)
        if not query_tokens:
            return []
        scores: List[float] = []
        for index, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            length = doc_lengths[index] or 1
            score = 0.0
            for term in set(query_tokens):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (total - df[term] + 0.5) / (df[term] + 0.5))
                denom = freq + k1 * (1 - b + b * length / avg_len)
                score += idf * (freq * (k1 + 1)) / denom
            scores.append(score)
        ranked = sorted(range(len(chunk_ids)), key=lambda i: scores[i], reverse=True)
        return [chunk_ids[i] for i in ranked[:top_k]]

    return retrieve


# ---------------------------------------------------------------- 自查询生成
_SPLIT_SENT = re.compile(r"(?<=[。！？；.!?])\s*")


def _template_questions(chunk_text: str, n: int):
    """无生成端点时的模板式启发。

    刻意用不同的句式与不同的锚点（首句 / 关键词），避免 n 个问题高度雷同
    ——雷同问题会互相抬高分数，让评测失去区分度。
    """
    from kf_ingest.embed import extract_keywords

    text = (chunk_text or "").strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SPLIT_SENT.split(text) if s.strip()]
    anchor = sentences[0][:48] if sentences else text[:48]
    keywords = [k for k in extract_keywords(text, top_k=6) if len(k) > 1]

    out: List[str] = []
    if anchor:
        out.append(f"{anchor} 是什么？")
    for keyword in keywords:
        if len(out) >= n:
            break
        if keyword in anchor:
            continue
        out.append(f"{keyword} 的说明是什么？")
    return out[:n]


def generate_self_questions(
    chunk_text: str, n: int = 3, llm: Optional[object] = None, lang: str = ""
) -> List[str]:
    """从 chunk 反向生成 n 个可回答问题。

    传入 llm（kf_ingest.clients.LLMClient）时走 LLM 生成，否则走模板启发。
    """
    if llm is not None:
        from kf_ingest.enrich import build_hypothetical_questions
        from kf_ingest.models import Chunk

        stub = Chunk(chunk_id="eval", chunk_type="text", content=chunk_text,
                     lang=lang, page=0, order=0)
        questions = build_hypothetical_questions(stub, llm, n)
        if questions:
            return questions
    return _template_questions(chunk_text, n)


# ---------------------------------------------------------------- 评测主体
@dataclass
class StrategyResult:
    strategy: str
    recall10: float
    recall20: float
    ndcg10: float
    sampled: int
    avg_chunk_tokens: float = 0.0
    # 语料规模不足以区分 recall@10 时为 False（此时 recall@10 恒为 1.0，指标饱和无区分力）
    informative: bool = True

    def as_dict(self, params: Optional[Dict] = None) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "params": params or {},
            "metrics": {
                "recall@10": self.recall10,
                "recall@20": self.recall20,
                "ndcg@10": self.ndcg10,
                "avg_chunk_tokens": self.avg_chunk_tokens,
                "informative": self.informative,
            },
            "sampled_chunks": self.sampled,
        }


def evaluate_strategy(
    chunks: List[Tuple[str, str]],           # [(chunk_id, content), ...]
    strategy: str,
    retriever: Optional[Callable[[str, int], List[str]]] = None,
    questions_per_chunk: int = 3,
    llm: Optional[object] = None,
    lang: str = "",
) -> StrategyResult:
    """对一组 chunk 做自查询评测。retriever 缺省时在**本组 chunk 上**建词法索引。"""
    if not chunks:
        return StrategyResult(strategy, 0.0, 0.0, 0.0, 0)
    retriever = retriever or build_lexical_retriever(chunks)

    r10 = r20 = n10 = 0.0
    n = 0
    for chunk_id, content in chunks:
        for question in generate_self_questions(content, questions_per_chunk, llm, lang):
            ranked = retriever(question, 20)
            relevant = {chunk_id}
            r10 += recall_at_k(ranked, relevant, 10)
            r20 += recall_at_k(ranked, relevant, 20)
            n10 += ndcg_at_k(ranked, relevant, 10)
            n += 1
    n = max(n, 1)
    total_tokens = 0
    try:
        from kf_ingest.tokenize import heuristic_count

        total_tokens = sum(heuristic_count(c) for _, c in chunks)
    except Exception:
        pass
    return StrategyResult(
        strategy=strategy,
        recall10=round(r10 / n, 4),
        recall20=round(r20 / n, 4),
        ndcg10=round(n10 / n, 4),
        sampled=len(chunks),
        avg_chunk_tokens=round(total_tokens / max(1, len(chunks)), 1),
        # recall@10 只有在语料大于 10 块时才可能低于 1.0，否则该指标恒为 1.0
        informative=len(chunks) > 10,
    )


def compare(results: List[StrategyResult], min_improvement: float = 0.03) -> str:
    """按 recall@20 排序，并给出是否值得切换策略的建议。"""
    if not results:
        return "无评测结果"
    ordered = sorted(results, key=lambda r: r.recall20, reverse=True)
    best, current = ordered[0], results[0]
    header = (f"{'strategy':<18}{'recall@10':>11}{'recall@20':>11}"
              f"{'ndcg@10':>10}{'avg_tok':>9}{'sampled':>9}")
    lines = [header]
    for r in ordered:
        lines.append(
            f"{r.strategy:<18}{r.recall10:>11.4f}{r.recall20:>11.4f}"
            f"{r.ndcg10:>10.4f}{r.avg_chunk_tokens:>9.1f}{r.sampled:>9}"
        )
    if best.strategy != current.strategy and (best.recall20 - current.recall20) >= min_improvement:
        lines.append(
            f"建议：切换到 {best.strategy}（recall@20 提升 "
            f"{best.recall20 - current.recall20:+.4f} ≥ 阈值 {min_improvement}）"
        )
    else:
        lines.append("建议：保持现状（收益未达阈值，避免无谓变更）")

    # 指标饱和时必须明确提示，否则会基于恒为 1.0 的数字做出错误选参决策
    if not all(r.informative for r in results):
        smallest = min(r.sampled for r in results)
        lines.append(
            f"警告：语料仅 {smallest} 块（≤10），recall@10 恒为 1.0、无区分力。"
            "该结果不能用于选参，请改用块数显著大于 10 的语料重跑。"
        )
    # 各策略产出完全一致时，说明差异参数尚未生效，A/B 无意义
    signatures = {(r.sampled, r.avg_chunk_tokens, r.recall20, r.ndcg10) for r in results}
    if len(results) > 1 and len(signatures) == 1:
        lines.append(
            "警告：所有策略产出完全相同（块数与粒度一致），说明各策略的差异参数"
            "在当前实现下尚未生效，本次 A/B 无效。"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 数据装载
def chunks_from_pdf(path: str, strategy: str, doc_id: str = "eval", lang: str = ""
                    ) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    """按指定策略对同一份文档切片，返回 [(chunk_id, content)] 与切片统计。"""
    from kf_ingest.chunk import chunk_document
    from kf_ingest.clean import clean_document
    from kf_ingest.config import load_chunking_config, strategy_params, strategy_pipeline
    from kf_ingest.parse import parse_pdf
    from kf_ingest.tokenize import TokenCounter

    cfg = load_chunking_config()
    params = strategy_params(cfg, strategy)
    counter = TokenCounter(params.get("tokenizer", "heuristic"))

    parsed = parse_pdf(path, doc_id, lang=lang)
    kept, _ = clean_document(parsed)
    result = chunk_document(
        kept, doc_id=doc_id, version=1,
        lang=lang or parsed.lang, params=params, counter=counter,
        strategy=strategy, pipeline=strategy_pipeline(cfg, strategy),
    )
    pairs = [(c.chunk_id, c.content) for c in result.children]
    return pairs, result.stats


def chunks_from_json(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    pairs: List[Tuple[str, str]] = []
    for index, item in enumerate(data):
        chunk_id = item.get("chunk_id") or f"c{index}"
        content = item.get("content") or item.get("text") or ""
        if content.strip():
            pairs.append((chunk_id, content))
    return pairs


def persist_eval(kb_id: str, doc_id: str, payloads: List[Dict[str, object]]) -> str:
    """把评测结果写入 eval_runs（需 DATABASE_URL + psycopg）。失败只返回原因，不抛。"""
    from kf_ingest.clients import database_url

    dsn = database_url()
    if not dsn:
        return "未配置 DATABASE_URL，跳过写入 eval_runs"
    try:
        import psycopg
        from psycopg.types.json import Jsonb

        with psycopg.connect(dsn) as conn:
            for payload in payloads:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO eval_runs(kb_id, doc_id, strategy, params, metrics, "
                        "sampled_chunks) VALUES (%s,%s,%s,%s,%s,%s)",
                        (kb_id, doc_id, payload["strategy"],
                         Jsonb(payload.get("params") or {}),
                         Jsonb(payload.get("metrics") or {}),
                         payload.get("sampled_chunks")),
                    )
            conn.commit()
        return f"已写入 eval_runs：{len(payloads)} 条"
    except Exception as exc:
        return f"写入 eval_runs 失败（已跳过）：{exc}"


# ---------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="切片策略评测（自查询 recall）")
    ap.add_argument("--path", help="对同一份文档比较多个切片策略（PDF）")
    ap.add_argument("--from-json", dest="from_json",
                    help="从 JSON 装载 chunk 集合进行评测")
    ap.add_argument("--strategies", default="doc_structured,semantic,table_centric",
                    help="逗号分隔的策略名，取自 configs/chunking_strategies.yaml")
    ap.add_argument("--questions-per-chunk", dest="qpc", type=int, default=3)
    ap.add_argument("--min-improvement", dest="min_improvement", type=float, default=0.03)
    ap.add_argument("--lang", default="")
    ap.add_argument("--limited", action="store_true",
                    help="只评测前 200 个 chunk（大文档快速试跑）")
    ap.add_argument("--kb", default="", help="写入 eval_runs 用的 kb_id")
    ap.add_argument("--doc", default="", help="写入 eval_runs 用的 doc_id")
    ap.add_argument("--persist", action="store_true", help="把结果写入 Postgres eval_runs")
    args = ap.parse_args(argv)

    if not args.path and not args.from_json:
        ap.error("请至少提供 --path 或 --from-json")

    # 有生成端点时用 LLM 生成自查询问题；否则模板启发（得分偏乐观，会标注）
    llm = None
    try:
        from kf_ingest.clients import try_llm

        llm = try_llm()
    except Exception:
        llm = None
    if llm is None:
        print("提示：未检测到生成端点（VLLM_URL/OPENAI_BASE_URL），自查询问题使用模板启发，"
              "得分偏乐观，仅作上界参考。")

    if args.from_json:
        pairs = chunks_from_json(args.from_json)
        if args.limited:
            pairs = pairs[:200]
        retriever = build_lexical_retriever(pairs)
        result = evaluate_strategy(pairs, "from_json", retriever=retriever,
                                   questions_per_chunk=args.qpc, llm=llm, lang=args.lang)
        print(compare([result], args.min_improvement))
        return 0

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    results: List[StrategyResult] = []
    payloads: List[Dict[str, object]] = []

    for strategy in strategies:
        try:
            pairs, chunk_stats = chunks_from_pdf(args.path, strategy, lang=args.lang)
        except Exception as exc:
            print(f"[{strategy}] 切片失败，已跳过：{exc}")
            continue
        if args.limited:
            pairs = pairs[:200]
        if not pairs:
            print(f"[{strategy}] 无子块产出，已跳过")
            continue
        retriever = build_lexical_retriever(pairs)
        result = evaluate_strategy(pairs, strategy, retriever=retriever,
                                  questions_per_chunk=args.qpc, llm=llm, lang=args.lang)
        results.append(result)
        payloads.append(result.as_dict(chunk_stats))
        print(f"[{strategy}] 子块 {len(pairs)}，平均 {chunk_stats.get('avg_child_tokens')} token")
        for note in (chunk_stats.get("notes") or []):
            print(f"    · {note}")

    if not results:
        print("所有策略均未产出可评测结果")
        return 1

    print()
    print(compare(results, args.min_improvement))

    if args.persist:
        if not (args.kb and args.doc):
            print("--persist 需要同时提供 --kb 与 --doc")
        else:
            print(persist_eval(args.kb, args.doc, payloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
