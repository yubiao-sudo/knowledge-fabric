"""嵌入编码：一份权重出三向量的**降级友好**实现。

- **dense**：TEI 调 BGE-M3（1024 维，多语言）。不可用时跳过 dense，
  但**不阻断**稀疏路径——检索侧仍可用词法稀疏 + 结构化过滤工作。
- **sparse**：优先 TEI `/embed_sparse`（学习式词权重）；TEI 未部署该能力时
  回退到**词法稀疏编码**（BM25 风格的 tf 权重 + 稳定哈希，可直接写入 Qdrant
  sparse 向量位）。词法稀疏必须**按语言分片**——BM25 本质单语言。
- **colbert**：多向量（图文混排场景），本模块保留接口，未启用时留空。

入库形态（写入 `chunk.meta["vectors"]`）::

    {"dense": [...1024 floats...],
     "sparse": {"indices": [...uint32...], "values": [...floats...]},
     "colbert": null,
     "model": "BAAI/bge-m3"}
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clients import TEIClient, embed_model_id
from .models import Chunk

_WORD = re.compile(r"[a-z0-9\u00c0-\u024f]+")
_CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

# 稀疏向量索引位宽（Qdrant 要求 uint32）
_INDEX_MASK = (1 << 31) - 1


def tokenize_for_sparse(text: str, lang: str = "") -> List[str]:
    """语言感知的分词：CJK 用单字 + 相邻双字（bigram），拉丁用词。

    中文不引入 jieba 这类词典依赖，用 bigram 即可获得接近分词的检索效果，
    且对未见词与专名更稳健（型号 / SKU 往往是未登录词）。
    """
    text = (text or "").lower()
    tokens: List[str] = [m.group(0) for m in _WORD.finditer(text)]

    cjk = _CJK_CHAR.findall(text)
    tokens.extend(cjk)
    # bigram 只在 CJK 连续片段内生成，避免跨标点拼出无意义组合
    for run in re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+", text):
        if len(run) >= 2:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _hash_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & _INDEX_MASK


def lexical_sparse(text: str, lang: str = "") -> Tuple[List[int], List[float]]:
    """BM25 风格词法稀疏编码：tf 权重 + L2 归一 + 稳定哈希索引。"""
    tokens = tokenize_for_sparse(text, lang)
    if not tokens:
        return [], []

    counts = Counter(tokens)
    weights: Dict[int, float] = {}
    for term, tf in counts.items():
        idx = _hash_index(term)
        weights[idx] = weights.get(idx, 0.0) + (1.0 + math.log(tf))

    norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
    indices = sorted(weights.keys())
    values = [round(weights[i] / norm, 6) for i in indices]
    return indices, values


def extract_keywords(text: str, lang: str = "", top_k: int = 8) -> List[str]:
    """轻量关键词抽取（喂稀疏侧与引用展示），不引入词典依赖。"""
    tokens = tokenize_for_sparse(text, lang)
    if not tokens:
        return []
    counts = Counter(tokens)
    # 丢弃单字 CJK 与长度 1 的拉丁词，优先 bigram / 词
    ranked = sorted(
        counts.items(),
        key=lambda kv: (len(kv[0]) > 1, kv[1], kv[0]),
        reverse=True,
    )
    out: List[str] = []
    for term, _ in ranked:
        if len(term) < 2:
            continue
        out.append(term)
        if len(out) >= top_k:
            break
    return out


def _content_of(chunk: Chunk) -> str:
    """送入嵌入的文本 = 上下文前缀 + 章节路径 + 正文（与检索侧查询对齐）。"""
    parts: List[str] = []
    if chunk.context_prefix:
        parts.append(chunk.context_prefix)
    if chunk.section_path:
        parts.append(" > ".join(p for p in chunk.section_path if p and p != "(根)"))
    parts.append(chunk.content)
    prefix = "\n".join(p for p in parts[:-1] if p)
    return f"{prefix}\n{parts[-1]}" if prefix else parts[-1]


def encode_chunks(
    chunks: Sequence[Chunk],
    *,
    tei: Optional[TEIClient] = None,
    model_id: Optional[str] = None,
    batch_size: int = 0,
) -> Dict[str, Any]:
    """为每个 chunk 写入 `meta["vectors"]`。返回统计。

    `batch_size=0` 表示**交给客户端决定**：本地嵌入器会读 `KF_EMBED_BATCH`（缺省 16），
    TEI 客户端则用其自身默认。4GB 卡上 32 会把显存撑爆，故不再在此硬编码 32。

    降级顺序：TEI dense + TEI sparse → TEI dense + 词法 sparse →
    仅词法 sparse（dense 缺席）。任一步失败都不抛异常。
    """
    model = model_id or embed_model_id()
    texts = [_content_of(c) for c in chunks]
    stats: Dict[str, Any] = {
        "chunks": len(chunks),
        "dense": 0,
        "sparse": 0,
        "sparse_backend": "lexical",
        "model": model,
        "notes": [],
    }

    dense_vectors: List[Optional[List[float]]] = [None] * len(texts)
    if tei is not None and texts:
        try:
            vectors = tei.embed(texts, batch_size=batch_size)
            if len(vectors) == len(texts):
                dense_vectors = vectors
                stats["dense"] = len(vectors)
                stats["dim"] = len(vectors[0]) if vectors else 0
                stats["dense_backend"] = getattr(tei, "backend", "tei")
            else:
                stats["notes"].append(f"嵌入返回 {len(vectors)} 条，与输入 {len(texts)} 不匹配，已忽略")
        except Exception as exc:
            stats["notes"].append(f"稠密嵌入不可用，跳过: {exc}")

    sparse_vectors: List[Optional[Dict[str, List]]] = [None] * len(texts)
    # 只有 TEI 具备**学习式稀疏**端点；本地嵌入客户端没有这个能力，
    # 用 hasattr 判能力而不是靠 except 吞异常（旧写法会让 AttributeError 被静默吞掉，
    # 日志上看不出"为什么走了词法稀疏"）。
    if tei is not None and texts and hasattr(tei, "embed_sparse"):
        try:
            learned = tei.embed_sparse(texts, batch_size=batch_size)
            if len(learned) == len(texts):
                sparse_vectors = learned
                stats["sparse_backend"] = "tei_learned"
        except Exception as exc:
            stats["notes"].append(f"学习式稀疏不可用，回退词法稀疏: {exc}")

    if stats["sparse_backend"] == "lexical":
        for i, chunk in enumerate(chunks):
            indices, values = lexical_sparse(texts[i], chunk.lang)
            sparse_vectors[i] = {"indices": indices, "values": values}

    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        chunk.meta["vectors"] = {
            "dense": dense,
            "sparse": sparse,
            "colbert": None,
            "model": model,
        }
        if sparse and sparse.get("indices"):
            stats["sparse"] += 1

    if not stats["dense"]:
        stats["notes"].append(
            "dense 向量缺席：检索将退化为稀疏 + 结构化路径"
            "（TEI 未配置时会在进程内回退本地嵌入；两者都不可用才会走到这里）"
        )
    return stats
