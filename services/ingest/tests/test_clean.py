"""清洗模块测试：跨页页眉页脚移除、文本规范化、SimHash、溯源偏移。"""

from kf_ingest.clean import (
    boilerplate_key,
    clean_document,
    compute_provenance,
    dedupe_blocks,
    detect_running_boilerplate,
    hamming,
    normalize_text,
    simhash,
)
from kf_ingest.models import (
    BLOCK_FOOTER,
    BLOCK_HEADER,
    BLOCK_PARAGRAPH,
    BLOCK_PAGE_NUMBER,
    Block,
    ParsedDoc,
)

PAGE_H = 800.0
PAGE_W = 600.0


def _block(text, page, y0, y1, block_type=BLOCK_PARAGRAPH, page_height=PAGE_H):
    return Block(
        block_type=block_type,
        text=text,
        page=page,
        order=0,
        bbox=[0.0, y0, PAGE_W, y1],
        meta={"page_height": page_height},
    )


def _doc(blocks, pages):
    parsed = ParsedDoc(
        doc_id="d1",
        version=1,
        lang="zh",
        doc_type="manual",
        blocks=blocks,
        raw_pages=pages,
        page_sizes=[[PAGE_W, PAGE_H] for _ in pages],
    )
    for order, b in enumerate(parsed.blocks):
        b.order = order
        b.uid = f"p{b.page}-b{order}"
    return parsed


def test_normalize_merges_hyphen_break():
    assert normalize_text("exam-\nple") == "example"
    assert normalize_text("multi    space") == "multi space"


def test_normalize_fullwidth_to_halfwidth():
    # NFKC：全角字母/数字统一为半角
    assert normalize_text("ＡＢＣ１２３") == "ABC123"


def test_boilerplate_key_normalizes_digits():
    assert boilerplate_key("Page 12 of 34") == boilerplate_key("Page 99 of 100")
    assert "#" in boilerplate_key("第 12 页")


def test_simhash_distance_near_duplicates():
    a = "企业知识库的检索质量取决于切片方案与重排模型的选择是否合理"
    b = "企业知识库的检索质量取决于切片方案与重排模型的选择是否恰当"
    c = "苹果香蕉橘子葡萄西瓜草莓柚子菠萝蜜桃柠檬石榴"
    assert hamming(simhash(a), simhash(b)) < hamming(simhash(a), simhash(c))


def test_detect_running_header_footer_removed():
    blocks = []
    pages = []
    for page in range(4):
        pages.append(f"正文内容第 {page} 页，讲述产品规格与安装步骤。")
        blocks.append(_block("企业产品手册 v2.0", page, 10, 30, BLOCK_HEADER))
        blocks.append(_block("第 %d 页" % (page + 1), page, 780, 795, BLOCK_PAGE_NUMBER))
        blocks.append(_block(f"正文内容第 {page} 页，讲述产品规格与安装步骤。", page, 200, 260))
    for order, b in enumerate(blocks):
        b.uid = f"p{b.page}-b{order}"

    remove, hit = detect_running_boilerplate(blocks, [PAGE_H] * 4)
    header_uids = {b.uid for b in blocks if b.block_type == BLOCK_HEADER}
    page_num_uids = {b.uid for b in blocks if b.block_type == BLOCK_PAGE_NUMBER}
    assert header_uids <= remove, "跨页重复的页眉应被移除"
    assert page_num_uids <= remove, "页码应被移除"
    assert len(hit) >= 1
    # 正文块绝不能被误删
    body_uids = {b.uid for b in blocks if b.block_type == BLOCK_PARAGRAPH}
    assert not (body_uids & remove)


def test_clean_document_keeps_body_and_records_ops():
    body = "这是正文段落，包含足够多的字符用于保证清洗、去重与溯源映射的有效性，长度必须超过精确去重的最小阈值。"
    blocks = []
    pages = []
    for page in range(3):
        pages.append(f"第 {page} 页原始文本。" + body)
        blocks.append(_block("公司机密 · 内部资料", page, 10, 28, BLOCK_HEADER))
        blocks.append(_block(f"第 {page} 页正文。" + body, page, 200, 300))
    # 再来一份与第 0 页完全相同（忽略空白/换行）的正文，用于验证精确去重
    blocks.append(_block("第 0 页正文。" + body, 1, 400, 500))
    parsed = _doc(blocks, pages)
    kept, report = clean_document(parsed)

    kept_types = {b.block_type for b in kept}
    assert BLOCK_HEADER not in kept_types
    assert len(kept) >= 3
    assert report.removed_blocks >= 3
    assert report.deduped_blocks >= 1, "重复正文应被精确去重"
    # 保留块必须记录原始文本，保证可逆
    assert all("raw_text" in b.meta for b in kept)


def test_dedupe_skips_titles_and_tables():
    from kf_ingest.models import BLOCK_TABLE, BLOCK_TITLE

    blocks = [
        _block("重复标题" * 20, 0, 0, 10, BLOCK_TITLE),
        _block("重复标题" * 20, 1, 0, 10, BLOCK_TITLE),
        _block("| a | b |\n| --- | --- |\n| 1 | 2 |", 0, 0, 0, BLOCK_TABLE),
        _block("| a | b |\n| --- | --- |\n| 1 | 2 |", 1, 0, 0, BLOCK_TABLE),
    ]
    kept, removed = dedupe_blocks(blocks)
    assert removed == 0
    assert len(kept) == 4


def test_compute_provenance_exact_offset():
    page_text = "第一段正文。\n\n第二段正文，包含更多内容用于验证偏移。\n\n第三段。"
    block = Block(
        block_type=BLOCK_PARAGRAPH,
        text="第二段正文，包含更多内容用于验证偏移。",
        page=0,
        order=0,
        bbox=[0, 0, 100, 100],
    )
    prov = compute_provenance(block, [page_text], "file:///doc.pdf")
    assert prov.char_start == page_text.index("第二段正文")
    assert prov.char_end == prov.char_start + len("第二段正文，包含更多内容用于验证偏移。")
    assert prov.original_text == "第二段正文，包含更多内容用于验证偏移。"
    assert "provenance_approx" not in prov.cleaning_ops


def test_compute_provenance_falls_back_with_marker():
    page_text = "完全不相关的内容。"
    block = Block(
        block_type=BLOCK_PARAGRAPH,
        text="这个块在原文中找不到。",
        page=0,
        order=0,
        bbox=[0, 0, 100, 100],
    )
    prov = compute_provenance(block, [page_text], "file:///doc.pdf")
    assert prov.char_start == 0 and prov.char_end == len(page_text)
    assert "provenance_approx" in prov.cleaning_ops


# ---------------------------------------------------------------- 块级溯源（切片后）
def _chunk(content: str, page: int = 0):
    from kf_ingest.models import Chunk

    return Chunk(chunk_id="c1", chunk_type="text", content=content, lang="zh",
                 page=page, order=1)


def test_chunk_provenance_locates_multi_block_text():
    """跨块拼接的切片内容必须能定位（旧实现按单块定位，86% 退化成整页区间）。"""
    from kf_ingest.clean import compute_chunk_provenance

    page = "标题行\n第一段正文内容。\n第二段正文内容，继续补充说明。"
    content = "第一段正文内容。\n第二段正文内容，继续补充说明。"
    prov = compute_chunk_provenance(_chunk(content), [page], "file:///d.pdf")
    assert "provenance_approx" not in prov.cleaning_ops
    assert prov.char_start == page.index("第一段正文内容。")
    assert prov.original_text.startswith("第一段正文内容。")


def test_chunk_provenance_survives_nfkc_divergence():
    """清洗会做 NFKC 归一，块文本与原文在字符层面已不同；bigram 锚点必须兜住。"""
    from kf_ingest.clean import compute_chunk_provenance

    # 原文用全角标点，切片内容已被 normalize_text 转为半角
    page = "第 3 章 运维规范\n系统默认采用混合召回策略，稠密向量负责跨语言语义匹配。"
    content = "系统默认采用混合召回策略,稠密向量负责跨语言语义匹配。"
    prov = compute_chunk_provenance(_chunk(content), [page], "file:///d.pdf")
    assert "provenance_approx" not in prov.cleaning_ops, "bigram 锚点应能定位"
    assert prov.char_start == page.index("系统默认采用混合召回策略")


def test_chunk_provenance_marks_partial_when_anchor_is_short():
    """锚点只覆盖局部时如实标注 provenance_partial，不假装覆盖整块。"""
    from kf_ingest.clean import compute_chunk_provenance

    page = "第一段正文内容。"
    content = "第一段正文内容。" + "完全不存在的后续内容。" * 30
    prov = compute_chunk_provenance(_chunk(content), [page], "file:///d.pdf")
    ops = " ".join(prov.cleaning_ops)
    assert "provenance_partial" in ops or "provenance_approx" in ops


def test_chunk_provenance_falls_back_with_marker_when_unlocatable():
    """页面与切片完全无共同字符时，必须如实退化并标 provenance_approx。"""
    from kf_ingest.clean import compute_chunk_provenance

    page = "ABQZXWVUTSRPONMLKJIHGFEDCBA9876543210"
    prov = compute_chunk_provenance(_chunk("中文内容在这里完全找不到对应位置的一句话"),
                                    [page], "file:///d.pdf")
    assert "provenance_approx" in prov.cleaning_ops
    assert prov.char_start == 0 and prov.char_end == len(page)


def test_span_union_same_page_covers_first_to_last_child():
    from kf_ingest.clean import span_union
    from kf_ingest.models import ProvenanceSpan

    a = ProvenanceSpan(source_uri="u", page=0, char_start=10, char_end=20)
    b = ProvenanceSpan(source_uri="u", page=0, char_start=100, char_end=140)
    merged = span_union([a, b], "u")
    assert merged is not None
    assert (merged.char_start, merged.char_end) == (10, 140)
    assert "provenance_union" in merged.cleaning_ops


def test_span_union_cross_page_keeps_first_page_range():
    """跨页父块不得产生倒置区间（真机出现过 [653, 97] 这种非法数据）。"""
    from kf_ingest.clean import span_union
    from kf_ingest.models import ProvenanceSpan

    a = ProvenanceSpan(source_uri="u", page=4, char_start=653, char_end=900)
    b = ProvenanceSpan(source_uri="u", page=5, char_start=17, char_end=97)
    merged = span_union([a, b], "u")
    assert merged is not None
    assert merged.page == 4
    assert merged.char_end >= merged.char_start, f"区间倒置: {merged.char_start}..{merged.char_end}"
    ops = " ".join(merged.cleaning_ops)
    assert "provenance_end_page=5" in ops
    assert "provenance_spans_pages" in ops
