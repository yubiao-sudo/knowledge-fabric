"""章节树质量断言与编号式标题识别（回归测试）。

背景（真机缺陷）：章节标题 16pt / 正文 11pt → 字号比 1.4545 落在 [1.35,1.6) → level 2；
而父块分组旧实现只在 `level == 1` 处断开，于是 6 个章节被压成 2 个父块、
每个父块横跨 3~4 章，检索返回整节而非目标模块。这类退化此前完全静默。
"""
from kf_ingest.models import BLOCK_PARAGRAPH, BLOCK_TITLE, Block, ParsedDoc
from kf_ingest.parse import (
    _level_from_numbering,
    section_tree_quality,
    structured_heading_cues,
)

PAGE_W, PAGE_H = 600.0, 800.0


def _block(text, *, font=11.0, block_type=BLOCK_PARAGRAPH, level=None, uid=""):
    return Block(block_type=block_type, text=text, page=0, order=0,
                 bbox=[0.0, 200.0, PAGE_W, 260.0], max_font_size=font,
                 level=level, uid=uid, meta={"page_height": PAGE_H})


def _doc(blocks):
    parsed = ParsedDoc(doc_id="d", version=1, lang="zh", doc_type="manual", blocks=blocks,
                       raw_pages=["x"], page_sizes=[[PAGE_W, PAGE_H]])
    for i, b in enumerate(parsed.blocks):
        b.order = i
        b.uid = b.uid or f"p0-b{i}"
    return parsed


# ------------------------------------------------------------------ 编号层级推断
def test_level_from_numbering_infers_depth():
    assert _level_from_numbering("第 3 章 运维规范", 11.0, 16.0) == 1
    assert _level_from_numbering("第三章 运维规范", 11.0, 16.0) == 1
    assert _level_from_numbering("3. 部署步骤", 11.0, 11.0) == 1
    assert _level_from_numbering("3.1 环境准备", 11.0, 11.0) == 2
    assert _level_from_numbering("3.1.2 依赖安装", 11.0, 11.0) == 3
    assert _level_from_numbering("（三）注意事项", 11.0, 11.0) == 3


def test_level_from_numbering_font_promotes_one_level():
    # 用了章级字号的 "3.1" 上提为 1 级，但不会把 3.1.2 提成章级
    assert _level_from_numbering("3.1 环境准备", 11.0, 17.5) == 1
    assert _level_from_numbering("3.1.2 依赖安装", 11.0, 17.5) == 2


# ------------------------------------------------------------------ 标题线索统计
def test_structured_heading_cues_counts_numbered_lines_only():
    blocks = [
        _block("第 1 章 总则"),
        _block("1.1 适用范围"),
        _block("本规定适用于全部内部系统，其中包括若干历史遗留系统与外包系统。"),  # 正文，不算
        _block("这段话没有任何编号且非常长" * 6),  # 过长，不算
        _block("这是正文句子。"),  # 有句末标点，不算
        _block("已成为正式标题", block_type=BLOCK_TITLE, level=2),  # 已是标题，不算线索
    ]
    assert structured_heading_cues(blocks) == 2


# ------------------------------------------------------------------ 章节树质量
def test_section_quality_flags_under_detected_headings():
    """大量编号线索但几乎没识别出标题 → 必须报 heading_under_detected。"""
    blocks = [_block(f"第 {i} 章 章节标题") for i in range(1, 8)]
    blocks += [_block(f"第 {i} 章的正文内容，讲述具体规范与操作步骤。") for i in range(1, 8)]
    quality = section_tree_quality(_doc(blocks))
    assert quality["numbered_cues"] >= 3
    assert quality["titles"] == 0
    assert "heading_under_detected" in quality["issues"]
    # 父块无法按章节切分——这是检索退化的直接判据
    assert quality["parents_will_split_by_section"] is False


def test_section_quality_ok_when_headings_detected():
    blocks = [_block(f"第 {i} 章 章节标题", block_type=BLOCK_TITLE, level=1)
              for i in range(1, 5)]
    blocks += [_block(f"第 {i} 章的正文内容。") for i in range(1, 5)]
    quality = section_tree_quality(_doc(blocks))
    assert quality["titles"] == 4
    assert quality["issues"] == []
    assert quality["parents_will_split_by_section"] is True


def test_section_quality_reports_levels_and_roots():
    blocks = [
        _block("第 1 章 甲", block_type=BLOCK_TITLE, level=1),
        _block("1.1 乙", block_type=BLOCK_TITLE, level=2),
        _block("正文。"),
    ]
    quality = section_tree_quality(_doc(blocks))
    assert quality["levels"] == [1, 2]
    assert quality["titles"] == 2


def test_numbered_heading_detected_without_font_advantage():
    """编号式标题即使字号与正文相同也必须被识别为 title（不赌字号）。"""
    from kf_ingest.parse import _classify_block

    block = _block("第 3 章 运维规范", font=11.0)
    _classify_block(block, body_size=11.0)
    assert block.block_type == BLOCK_TITLE, "编号是排版无关的强特征，不该漏"
    assert block.level == 1
    assert block.meta.get("title_source") == "numbering"
