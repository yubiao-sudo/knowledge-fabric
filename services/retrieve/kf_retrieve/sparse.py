"""稀疏编码：与摄入侧**逐行等价**的复刻。

本文件必须与 `services/ingest/kf_ingest/embed.py` 的
`tokenize_for_sparse` / `lexical_sparse`（及其 `_hash_index` / `_INDEX_MASK`）
保持等价：检索侧与索引侧必须使用完全相同的编码，否则稀疏通道永远匹配不上。

等价性由 `tests/test_sparse_golden.py` 兜底：用同一批中英混合样本分别调用
两份实现，断言 `indices` / `values` 完全相同（该测试用 sys.path 注入
`services/ingest` 来 import kf_ingest，import 失败则 skip）。

检索服务刻意**不 import kf_ingest**（会拖入 PyMuPDF 等重依赖），故在此复刻。
另附 `clean.py` 的 64 位 SimHash（`blake2b digest_size=8` + 3-shingle），供近似去重。
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter
from typing import Dict, List, Tuple

# ---- 与 kf_ingest/embed.py 完全一致的分词与哈希 ----
_WORD = re.compile(r"[a-z0-9\u00c0-\u024f]+")
_CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

# 稀疏向量索引位宽（Qdrant 要求 uint32）
_INDEX_MASK = (1 << 31) - 1


def tokenize_for_sparse(text: str, lang: str = "") -> List[str]:
    """语言感知的分词：CJK 用单字 + 相邻双字（bigram），拉丁用词。

    与 `kf_ingest.embed.tokenize_for_sparse` **逐行等价**。
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
    """BM25 风格词法稀疏编码：tf 权重 + L2 归一 + 稳定哈希索引。

    与 `kf_ingest.embed.lexical_sparse` **逐行等价**。
    """
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


def sparse_vector(text: str, lang: str = "") -> Dict[str, List]:
    """打包成 Qdrant `/query_points` 需要的 `{indices, values}` 形态。"""
    indices, values = lexical_sparse(text, lang)
    return {"indices": indices, "values": values}


# ---- 近似去重：复刻 clean.py 的 64 位 SimHash ----
def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def _sim_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", (text or "").lower())


def simhash(text: str, shingle: int = 3, bits: int = 64) -> int:
    """字符/词混元 shingle 的 64 位 SimHash，用于近似重复检测。"""
    toks = _sim_tokens(text)
    if not toks:
        return 0
    grams = [" ".join(toks[i:i + shingle]) for i in range(max(1, len(toks) - shingle + 1))]
    vector = [0] * bits
    for gram in grams:
        h = _hash64(gram)
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
