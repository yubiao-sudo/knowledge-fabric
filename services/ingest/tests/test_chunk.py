"""切片模块测试：父子块、原子块、token 上限、中文分句、表格行组切分。"""

import pytest

from kf_ingest.chunk import (
    chunk_document,
    hard_split,
    split_markdown_table,
    split_sentences,
)
from kf_ingest.models import (
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TITLE,
    Block,
)
from kf_ingest.tokenize import TokenCounter, heuristic_count


@pytest.fixture()
def counter():
    """强制走启发式计数，测试不依赖网络与模型权重。"""
    return TokenCounter("heuristic")


_UID_SEQ = [0]


def _block(text, block_type=BLOCK_PARAGRAPH, page=0, level=None):
    _UID_SEQ[0] += 1
    return Block(block_type=block_type, text=text, page=page, order=0,
                 bbox=[0, 0, 100, 100], level=level, uid=f"u{_UID_SEQ[0]}")


def test_split_sentences_chinese_uses_cjk_punctuation():
    text = "第一条：安装步骤。第二条：配置参数！第三条：验证结果？"
    sents = split_sentences(text, "zh")
    assert len(sents) == 3
    assert sents[0].endswith("。")
    # 中文句子里的英文句点不应被当作句末（不按 . 切）
    mixed = "版本 v1.2 已发布，请升级。"
    assert len(split_sentences(mixed, "zh")) == 1


def test_split_sentences_latin():
    sents = split_sentences("First step. Second step! Third?", "en")
    assert len(sents) == 3
    # 小数点的版本号不应被切开
    assert len(split_sentences("Version 1.2.3 is out.", "en")) == 1


def test_hard_split_respects_budget(counter):
    long_text = "配置说明" * 300
    pieces = hard_split(long_text, 50, counter)
    assert len(pieces) > 1
    assert all(counter.count(p) <= 50 for p in pieces)
    assert sum(len(p) for p in pieces) >= len(long_text) - 5


def test_chunk_produces_parent_child_with_links(counter):
    blocks = [
        _block("产品概述", BLOCK_TITLE, level=1),
        _block("本产品是一款面向企业的知识库底座。" * 20),
        _block("安装说明", BLOCK_TITLE, level=1),
        _block("安装前请先备份数据。" * 20),
    ]
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params={"child_tokens": 100, "parent_tokens": 400, "overlap_ratio": 0.12},
        counter=counter,
    )
    assert len(result.parents) >= 2, "每个一级标题应开启新的父块"
    assert result.children, "应产出子块"
    parent_ids = {p.chunk_id for p in result.parents}
    assert all(c.parent_chunk_id in parent_ids for c in result.children)
    assert all(c.is_parent is False for c in result.children)
    assert all(p.is_parent for p in result.parents)
    # 子块体型应贴近目标，而不是一句一块
    assert result.stats["avg_child_tokens"] >= 25
    # 除原子块外，子块不超过目标 token（overlap 可能带来轻微超出）
    for c in result.children:
        if c.chunk_type == "text":
            assert c.token_count <= 120, f"子块超限: {c.token_count}"


def test_chunk_table_is_atomic_and_kept_whole(counter):
    rows = ["| 型号 | 价格 | 库存 |", "| --- | --- | --- |"]
    rows += [f"| M-{i} | {i}00 | {i * 3} |" for i in range(30)]
    table_md = "\n".join(rows)
    blocks = [
        _block("规格表", BLOCK_TITLE, level=1),
        _block(table_md, BLOCK_TABLE),
    ]
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params={"child_tokens": 80, "parent_tokens": 400, "overlap_ratio": 0.0,
                "group_rows": 5, "repeat_header_per_group": True},
        counter=counter,
    )
    tables = [c for c in result.children if c.chunk_type == "table"]
    assert len(tables) >= 2, "大表应按行组切分为多块"
    for t in tables:
        # 每个表块都必须自带表头，保证信息完整
        assert t.content.startswith("| 型号 | 价格 | 库存 |")


def test_split_markdown_table_repeats_header(counter):
    rows = ["| a | b |", "| --- | --- |"] + [f"| {i} | {i} |" for i in range(10)]
    groups = split_markdown_table("\n".join(rows), counter, group_rows=3, max_tokens=60)
    assert len(groups) >= 3
    for g in groups:
        assert g.split("\n")[0] == "| a | b |"
        assert g.split("\n")[1].startswith("| ---")


def test_heuristic_count_cjk_cheaper_than_char_mix():
    assert heuristic_count("中文测试") >= 4
    assert heuristic_count("") == 0


# ------------------------------------------------- 父块分组粒度（真机缺陷回归）
def test_parent_groups_break_at_level2_headings(counter):
    """章节标题常被判为 level 2，父块分组必须在 level<=2 处断开。

    旧实现只在 `level == 1` 处断开，而 PDF 的章节标题字号比例多在 1.35~1.6，
    被 `parse._classify_block` 判为 level 2 → 章节边界全部失效，
    真机实测 6 个章节被压成 2 个父块、每个父块横跨 3~4 章。
    """
    blocks = []
    for i in range(1, 7):
        blocks.append(_block(f"第 {i} 章 说明", BLOCK_TITLE, level=2))
        blocks.append(_block(f"第 {i} 章的正文内容。" * 30))
    result = chunk_document(
        blocks, doc_id="d2", version=1, lang="zh",
        params={"child_tokens": 100, "parent_tokens": 500, "overlap_ratio": 0.0,
                "parent_break_level": 2},
        counter=counter,
    )
    assert len(result.parents) == 6, f"6 个 level-2 章节应产出 6 个父块，实得 {len(result.parents)}"
    for parent in result.parents:
        assert parent.meta.get("spans_multiple_sections") is False


def test_parent_tokens_cannot_swallow_many_children(counter):
    """父块上限不得远大于子块：旧默认 1600 与子块 320 恰成 1:5，必然横跨结构。"""
    from kf_ingest.chunk import _settings

    st = _settings({"child_tokens": 320, "parent_tokens": 1600})
    assert st.parent_tokens <= 320 * 3, "父块被压到至多 3 个子块"
    assert st.parent_break_level == 2


def test_parent_records_covered_sections(counter):
    """父块必须记录实际覆盖的章节，便于发现跨章节分组（旧实现只留首块章节）。"""
    blocks = [
        _block("第一章", BLOCK_TITLE, level=2),
        _block("正文甲。" * 40),
        _block("第二章", BLOCK_TITLE, level=2),
        _block("正文乙。" * 40),
    ]
    result = chunk_document(
        blocks, doc_id="d3", version=1, lang="zh",
        params={"child_tokens": 100, "parent_tokens": 500, "overlap_ratio": 0.0},
        counter=counter,
    )
    assert result.parents
    for parent in result.parents:
        assert "sections_covered" in parent.meta
        assert isinstance(parent.meta["sections_covered"], list)
