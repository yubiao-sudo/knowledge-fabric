"""端到端测试：生成真实 PDF → 解析 → 清洗 → 切片（dry-run 全链路）。

覆盖用户最关心的三件事：
  1. 跨页页眉 / 页脚被洗掉，正文不受影响；
  2. 每个子块都能溯源回原文（provenance 映射）；
  3. 子块 token 贴近目标粒度，父块聚合多个子块。
"""

import pytest

from kf_ingest.embed import lexical_sparse, tokenize_for_sparse
from kf_ingest.index import point_id
from kf_ingest.models import Chunk, IngestContext
from kf_ingest.pipeline import document_features, run_ingest
from kf_ingest.tokenize import TokenCounter

HEADER = "企业产品手册 v2.1 · 内部资料"
# 每页正文必须不同：跨页完全相同的段落会被精确去重正确丢弃（这是预期行为）
BODY_TMPL = (
    "第 {n} 章说明 MX-{n}00 型设备的安装与配置流程。本产品是一款面向企业的知识库底座，"
    "支持多语言语料的解析、清洗、切片与混合检索。安装前请先备份现有数据，"
    "并确认磁盘剩余空间不少于二十吉字节。配置参数包括嵌入模型、重排模型与向量库地址，"
    "均通过环境变量注入。验证阶段建议先执行 dry-run，确认切片粒度与召回效果后再正式落库。"
)


def _make_pdf(path):
    import pymupdf as fitz

    doc = fitz.open()
    for page_index in range(4):
        page = doc.new_page()  # A4: 595 x 842
        page.insert_textbox(fitz.Rect(60, 20, 535, 45), HEADER, fontsize=9, fontname="china-s")
        page.insert_textbox(fitz.Rect(60, 795, 535, 820), f"第 {page_index + 1} 页",
                            fontsize=9, fontname="china-s", align=1)
        page.insert_textbox(fitz.Rect(60, 90, 535, 120),
                            f"1.{page_index + 1} 产品概述", fontsize=18,
                            fontname="china-s")
        page.insert_textbox(fitz.Rect(60, 140, 535, 600),
                            BODY_TMPL.format(n=page_index + 1), fontsize=11,
                            fontname="china-s")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def counter():
    return TokenCounter("heuristic")


def _make_table_pdf(path):
    """一页带框线表格 + 两页正文，用于验证表格的抽取与原子切片。"""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(60, 20, 535, 45), "产品规格书 v1.0",
                        fontsize=9, fontname="china-s")
    page.insert_textbox(fitz.Rect(60, 90, 535, 125), "1.1 规格对照表",
                        fontsize=18, fontname="china-s")
    page.insert_textbox(fitz.Rect(60, 140, 535, 175), "下表列出各型号的关键参数：",
                        fontsize=11, fontname="china-s")

    x0, y0, row_h = 60, 190, 26
    cols = [0, 130, 240, 340, 470]
    header = ["型号", "价格", "库存", "交期"]
    data = [["MX-100", "1200", "35", "3 天"], ["MX-200", "1999", "12", "7 天"],
            ["MX-300", "2450", "8", "14 天"], ["MX-400", "3100", "0", "预售"],
            ["MX-500", "4200", "5", "21 天"]]
    n_rows = len(data) + 1
    for r in range(n_rows + 1):
        y = y0 + r * row_h
        page.draw_line(fitz.Point(x0, y), fitz.Point(x0 + cols[-1], y))
    for c in cols:
        page.draw_line(fitz.Point(x0 + c, y0), fitz.Point(x0 + c, y0 + n_rows * row_h))
    for c, name in enumerate(header):
        page.insert_textbox(fitz.Rect(x0 + cols[c] + 4, y0 + 4,
                                      x0 + cols[c + 1] - 4, y0 + 24),
                            name, fontsize=10, fontname="china-s")
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            top = y0 + row_h * (r + 1)
            page.insert_textbox(fitz.Rect(x0 + cols[c] + 4, top + 4,
                                          x0 + cols[c + 1] - 4, top + 24),
                                value, fontsize=10)

    for i in range(2):
        extra = doc.new_page()
        extra.insert_textbox(fitz.Rect(60, 20, 535, 45), "产品规格书 v1.0",
                             fontsize=9, fontname="china-s")
        extra.insert_textbox(fitz.Rect(60, 795, 535, 820), f"第 {i + 2} 页",
                             fontsize=9, fontname="china-s", align=1)
        extra.insert_textbox(
            fitz.Rect(60, 100, 535, 400),
            f"第 {i + 2} 节 补充说明：该型号的安装与维护要求如下所述，"
            f"请务必遵照执行以避免质保失效，否则将影响后续的售后服务与保修认定。",
            fontsize=11, fontname="china-s")
    doc.save(path)
    doc.close()
    return path


def test_table_is_single_atomic_chunk_with_header(tmp_path, counter):
    """表格必须整块入库且带表头，不能被切开、丢失表头或被后续正文抢占。"""
    pdf = _make_table_pdf(str(tmp_path / "spec.pdf"))
    ctx = IngestContext(tenant_id="acme", kb_id="demo", doc_id="spec",
                        source_uri=str(pdf), source_path=str(pdf), doc_type="manual")
    run_ingest(ctx, counter=counter, dry_run=True)

    tables = [c for c in ctx.chunks if c.chunk_type == "table"]
    assert len(tables) == 1, f"应为一个完整表块，实际 {len(tables)} 个"
    table = tables[0]
    assert table.content.startswith("| 型号 | 价格 | 库存 | 交期 |"), table.content[:60]
    assert "MX-500" in table.content and "MX-100" in table.content

    # 表块不得被后续正文污染：正文块里不能出现表格行
    prose = [c for c in ctx.chunks if c.chunk_type == "text"]
    for chunk in prose:
        assert "| MX-" not in chunk.content, f"正文块混入表格行: {chunk.content[:80]}"


def test_table_cells_are_not_duplicated_as_fragments(tmp_path):
    """表格区域内的单元格会被 get_text 拆成碎片，必须被剔除，否则内容重复索引。"""
    from kf_ingest.parse import parse_pdf

    pdf = _make_table_pdf(str(tmp_path / "spec2.pdf"))
    parsed = parse_pdf(pdf, "spec2", doc_type="manual")

    tables = [b for b in parsed.blocks if b.block_type == "table"]
    assert len(tables) == 1

    # 不应存在「把单元格竖着堆叠」的碎片段落（形如 型号\\n价格\\n库存）
    for block in parsed.blocks:
        if block.block_type == "table":
            continue
        assert "型号\n价格" not in block.text, f"检测到表格碎片: {block.text!r}"
        # 表格区域内的行碎片也一并被剔除
        assert block.text.strip() not in ("MX-100\n1200\n35\n3 天",), block.text


def test_atomic_block_type_survives_classification(tmp_path):
    """表格/图片的类型由抽取阶段确定，不能被字号与位置规则改写。"""
    from kf_ingest.parse import parse_pdf

    pdf = _make_table_pdf(str(tmp_path / "spec3.pdf"))
    parsed = parse_pdf(pdf, "spec3", doc_type="manual")
    assert any(b.block_type == "table" for b in parsed.blocks)
    # 页眉仍然是页眉（表格不会被误判成页眉，页眉也不会被误判成正文）
    assert any(b.block_type == "header" for b in parsed.blocks)
    assert parsed.doc_type == "manual"


def test_end_to_end_dry_run(tmp_path, counter):
    pdf = _make_pdf(str(tmp_path / "manual.pdf"))

    ctx = IngestContext(
        tenant_id="acme", kb_id="demo", doc_id="manual_v1",
        source_uri=str(pdf), source_path=str(pdf), version=1, doc_type="manual",
    )
    run_ingest(ctx, counter=counter, dry_run=True)

    assert not ctx.errors, f"链路不应报错: {ctx.errors}"

    parse_stats = ctx.stage_stats["parse"]
    assert parse_stats["pages"] == 4
    assert parse_stats["blocks"] > 0

    # 1) 跨页页眉 + 页脚被移除
    clean_stats = ctx.stage_stats["clean"]
    assert clean_stats["removed_blocks"] >= 8, clean_stats
    assert clean_stats["removed_by_type"].get("header", 0) >= 4

    # 2) 子块可溯源：每块 provenance 指向原文且能取回原句
    assert ctx.parents and ctx.chunks
    traced = [c for c in ctx.chunks if c.provenance is not None]
    assert traced, "子块必须带 provenance"
    for chunk in traced[:5]:
        prov = chunk.provenance
        assert prov.source_uri == str(pdf)
        assert prov.original_text, "应能一键还原原文"

    # 3) 粒度：子块贴近目标，父块聚合多个子块
    chunk_stats = ctx.stage_stats["chunk"]
    assert chunk_stats["child_tokens_target"] == 320
    assert chunk_stats["max_child_tokens"] <= 320 * 1.2
    assert chunk_stats["avg_child_tokens"] > 60
    assert len(ctx.parents) <= len(ctx.chunks)

    # 4) 策略路由：有章节结构 → doc_structured
    assert ctx.strategy == "doc_structured"
    assert chunk_stats["features"]["has_section_hierarchy"] is True


def test_document_features_routing_on_titles(tmp_path, counter):
    pdf = _make_pdf(str(tmp_path / "manual2.pdf"))
    ctx = IngestContext(tenant_id="t", kb_id="k", doc_id="d",
                        source_uri=str(pdf), source_path=str(pdf))
    run_ingest(ctx, counter=counter, dry_run=True)
    features = document_features(ctx.parsed)
    assert features["scanned"] is False
    assert features["table_density"] == "low"
    assert features["has_section_hierarchy"] is True


def test_lexical_sparse_and_keywords():
    indices, values = lexical_sparse("产品型号 MX-200 的价格是 1999 元", "zh")
    assert indices and len(indices) == len(values)
    assert all(isinstance(i, int) and 0 <= i for i in indices)
    assert all(isinstance(v, float) for v in values)
    # L2 归一
    assert abs(sum(v * v for v in values) - 1.0) < 1e-5
    # CJK bigram + 拉丁词都应进入词表
    tokens = tokenize_for_sparse("产品型号 MX-200", "zh")
    assert "mx" in tokens
    assert any(len(t) == 2 for t in tokens)


def test_point_id_is_deterministic_uuid():
    chunk = Chunk(chunk_id="d:v1:C1-abcd", chunk_type="text", content="x",
                  lang="zh", page=0, order=1)
    first = point_id(chunk)
    assert first == point_id(chunk)
    assert len(first) == 36  # uuid 字符串


class _FakeQdrantPayloadStore:
    """最小 Qdrant 替身：用 `doc_id` / `version` 两个 payload 字段求值 Filter。

    只实现 `count` / `set_payload` 与 must / must_not 判定，足以验证
    「版本切换只标旧版本」这一语义（真机联调曾实测到旧版本未被标记）。
    """

    def __init__(self, points):
        self.fields = {key: dict(value) for key, value in points.items()}

    def _match(self, flt, key):
        for cond in (flt.must or []):
            if self.fields[key].get(cond.key) != cond.match.value:
                return False
        for cond in (flt.must_not or []):
            if self.fields[key].get(cond.key) == cond.match.value:
                return False
        return True

    def count(self, collection_name, count_filter=None, exact=True):
        from types import SimpleNamespace

        return SimpleNamespace(count=sum(1 for k in self.fields if self._match(count_filter, k)))

    def set_payload(self, collection_name, payload, points=None, wait=True):
        for key in self.fields:
            if self._match(points, key):
                self.fields[key].update(payload)


def test_mark_stale_qdrant_marks_only_older_versions():
    pytest.importorskip("qdrant_client")  # 可选依赖，未安装时跳过落库相关测试
    from kf_ingest.index import mark_stale_qdrant

    client = _FakeQdrantPayloadStore({
        "a": {"doc_id": "d1", "version": 1, "status": "active"},
        "b": {"doc_id": "d1", "version": 1, "status": "active"},
        "c": {"doc_id": "d1", "version": 2, "status": "active"},
        "x": {"doc_id": "d2", "version": 1, "status": "active"},
    })

    assert mark_stale_qdrant(client, "kf_acme_demo", "d1", 2) == 2
    # 同文档旧版本置 stale；当前版本与其它文档不受影响
    assert {k: v["status"] for k, v in client.fields.items()} == {
        "a": "stale", "b": "stale", "c": "active", "x": "active"
    }
    # 幂等：重复标记不再产生新变更
    assert mark_stale_qdrant(client, "kf_acme_demo", "d1", 2) == 2
