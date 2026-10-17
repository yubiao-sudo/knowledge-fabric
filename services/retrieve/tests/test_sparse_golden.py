"""Golden 测试：检索侧稀疏编码必须与摄入侧**完全相同**。

用同一批中英混合样本分别调用 `kf_retrieve.sparse` 与 `kf_ingest.embed`，
断言 `indices` / `values` 完全一致。若 `kf_ingest` 因依赖缺失无法 import，
则 skip（不把环境问题当失败）。
"""

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_INGEST = os.path.abspath(os.path.join(_ROOT, "..", "ingest"))
for _p in (_ROOT, _INGEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kf_retrieve.sparse import (  # noqa: E402
    hamming,
    lexical_sparse,
    simhash,
    tokenize_for_sparse,
)

SAMPLES = [
    "产品 X 的保修期是多久？",
    "MX-200 的售价为 1999 元，支持 5G 网络。",
    "How long is the warranty for the BGE-M3 embedder?",
    "混合 Mixed 文本 123 与 some identifiers XPS-13_2024。",
    "日本語のドキュメントと 한국어 문서 もテストします。",
    "",
]


def _load_ingest_embed():
    try:
        import kf_ingest.embed as embed  # type: ignore
    except Exception as exc:  # 依赖缺失（PyMuPDF 等）时跳过
        pytest.skip(f"无法 import kf_ingest.embed: {exc}")
    return embed


@pytest.mark.parametrize("text", SAMPLES)
def test_tokenize_matches_ingest(text):
    embed = _load_ingest_embed()
    assert tokenize_for_sparse(text) == embed.tokenize_for_sparse(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_lexical_sparse_matches_ingest(text):
    embed = _load_ingest_embed()
    idx_a, val_a = lexical_sparse(text)
    idx_b, val_b = embed.lexical_sparse(text)
    assert idx_a == idx_b, f"indices 漂移: {idx_a} != {idx_b}"
    assert val_a == val_b, f"values 漂移: {val_a} != {val_b}"


def test_simhash_matches_ingest_clean():
    try:
        from kf_ingest.clean import hamming as ih, simhash as ish  # type: ignore
    except Exception as exc:
        pytest.skip(f"无法 import kf_ingest.clean: {exc}")
    for text in SAMPLES:
        assert simhash(text) == ish(text)
    assert hamming(123, 456) == ih(123, 456)
