"""融合：倒数排名融合（RRF）与加权融合。

输入统一为「按分数降序的 id 列表」（每个通道一份），输出 `{id: fused_score}`。
两种模式：

- `rrf(rankings, k=60)`：`score = Σ 1/(k + rank)`，rank 从 1 起。对分数量纲不敏感，
  是混合检索的稳健基线。
- `weighted(lists, weights)`：按通道权重对**排名归一化分**（1/(k+rank)）做线性加权，
  权重读自 `retrieval_profiles.yaml` 的 `fusion_weights`。

`combine(...)` 是编排层用的薄封装：选定模式 + 施加 `sparse_boost` 乘法加权。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    """倒数排名融合。rank 从 1 起：`score = Σ 1/(k + rank)`。"""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def weighted(lists: Mapping[str, Sequence[str]], weights: Mapping[str, float],
             k: int = 60) -> Dict[str, float]:
    """加权融合：每个通道按 `1/(k+rank)` 归一化后乘以其权重再求和。

    `lists` 形如 `{"dense": [id...], "sparse": [id...]}`（各自按分数降序），
    `weights` 形如 `{"dense": 0.6, "sparse": 0.4}`。权重缺失的通道记 0 分。
    """
    scores: Dict[str, float] = defaultdict(float)
    for channel, ranking in lists.items():
        w = float(weights.get(channel, 0.0))
        if w == 0.0:
            continue
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += w * (1.0 / (k + rank))
    return dict(scores)


def combine(
    lists: Mapping[str, Sequence[str]],
    mode: str = "rrf",
    *,
    k: int = 60,
    weights: Optional[Mapping[str, float]] = None,
    boost: float = 1.0,
) -> Dict[str, float]:
    """按档位融合策略组合多通道排名。

    - `mode="rrf"`：标准 RRF；`boost` 只作用在稀疏通道（对稀疏分做乘法加权）。
    - `mode="weighted"`：按 `weights` 加权；`boost` 同样只乘稀疏通道。
    - 其它（含 `none`）：返回空——调用方应直接采用单通道原始分。
    """
    lists = {ch: list(r) for ch, r in (lists or {}).items() if r}
    if not lists:
        return {}
    if mode == "weighted":
        w = dict(weights or {})
        if boost != 1.0 and "sparse" in lists:
            w["sparse"] = float(w.get("sparse", 0.0)) * boost
        return weighted(lists, w, k=k)
    if mode == "rrf":
        if boost == 1.0:
            return rrf(list(lists.values()), k=k)
        base = rrf([r for ch, r in lists.items() if ch != "sparse"], k=k)
        sparse = rrf([lists["sparse"]], k=k) if "sparse" in lists else {}
        for doc_id, score in sparse.items():
            base[doc_id] = base.get(doc_id, 0.0) + score * boost
        return base
    return {}


def ranked_ids(scores: Mapping[str, float], limit: Optional[int] = None) -> List[str]:
    """按分数降序返回 id；同分时按 id 升序，保证结果稳定可测。"""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ids = [doc_id for doc_id, _ in ordered]
    return ids[:limit] if limit else ids


def top_scores(scores: Mapping[str, float], limit: Optional[int] = None) -> Dict[str, float]:
    return {doc_id: scores[doc_id] for doc_id in ranked_ids(scores, limit)}


def _iter_ids(rankings: Iterable[Sequence[str]]) -> Iterable[str]:
    for ranking in rankings:
        yield from ranking
