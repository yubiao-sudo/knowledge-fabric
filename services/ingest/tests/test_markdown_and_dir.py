"""Markdown 解析 + 目录遍历测试。

覆盖新能力的两条主线：
  1. `parse_markdown`：把 Markdown 结构化（标题 / 代码 / 表格 / 列表），
     无分页时如实记警告，解码走 utf-8-sig → gb18030 回退；
  2. `parse_document` / `discover_documents`：按后缀分流、目录遍历与「不支持格式如实列出」。
"""

import os

import pytest

from kf_ingest.parse import (
    MARKDOWN_SCANNER,
    SUPPORTED_SUFFIXES,
    discover_documents,
    parse_document,
    parse_markdown,
)
from kf_ingest.pipeline import run_ingest
from kf_ingest.models import IngestContext

MD_SAMPLE = """---
title: 测试手册
tags: [a, b]
---

# 平台总览

这是一个**测试**段落，用于验证段落成块。

## 功能清单

- 求职安全互助
- 合同审核提醒
- 企业资质核验

## 关键参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| port | 服务端口 | 8083 |
| kb | 知识库名 | demo |

> 注意：以上参数需通过环境变量注入。

```python
def health():
    return {"status": "ok"}
```

### 验收要求

1. 页面加载不超过两秒
2. 证据可回溯到原文
"""


def _write(tmp_path, name, text, encoding="utf-8"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "wb") as fh:
        fh.write(text.encode(encoding))
    return path


# ------------------------------------------------------------------ Markdown 解析
def test_markdown_headings_become_title_blocks_with_level(tmp_path):
    path = _write(tmp_path, "doc.md", MD_SAMPLE)
    parsed = parse_markdown(path, "doc")
    titles = [(b.text, b.level) for b in parsed.blocks if b.block_type == "title"]
    assert titles == [
        ("平台总览", 1),
        ("功能清单", 2),
        ("关键参数", 2),
        ("验收要求", 3),
    ]
    assert parsed.scanner == MARKDOWN_SCANNER


def test_markdown_fenced_code_is_atomic_candidate(tmp_path):
    path = _write(tmp_path, "doc.md", MD_SAMPLE)
    parsed = parse_markdown(path, "doc")
    code_blocks = [b for b in parsed.blocks if (b.meta or {}).get("is_code")]
    assert len(code_blocks) == 1
    assert "def health():" in code_blocks[0].text
    # 围栏本身不进正文（否则会污染检索内容）
    assert "```" not in code_blocks[0].text
    assert code_blocks[0].block_type == "paragraph"


def test_markdown_table_is_structured_block(tmp_path):
    path = _write(tmp_path, "doc.md", MD_SAMPLE)
    parsed = parse_markdown(path, "doc")
    tables = [b for b in parsed.blocks if b.block_type == "table"]
    assert len(tables) == 1
    assert tables[0].meta["rows"] == 3       # 表头 + 2 数据行；分隔行不计入
    assert tables[0].meta["cols"] == 3
    assert "| 参数 | 说明 | 默认值 |" in tables[0].text


def test_markdown_list_block_counts_items(tmp_path):
    path = _write(tmp_path, "doc.md", MD_SAMPLE)
    parsed = parse_markdown(path, "doc")
    lists = [b for b in parsed.blocks if b.block_type == "list"]
    assert len(lists) == 2
    assert lists[0].meta["items"] == 3
    assert lists[1].meta["items"] == 2


def test_markdown_front_matter_is_stripped(tmp_path):
    path = _write(tmp_path, "doc.md", MD_SAMPLE)
    parsed = parse_markdown(path, "doc")
    assert "front_matter_stripped" in parsed.warnings
    assert all("tags:" not in b.text for b in parsed.blocks)


def test_markdown_has_no_pages_and_says_so(tmp_path):
    """Markdown 没有页概念：page 恒为 0，且必须记 `no_page_layout` 警告，
    不能假装有分页（否则下游页码引用会指向错误位置）。"""
    path = _write(tmp_path, "doc.md", MD_SAMPLE)
    parsed = parse_markdown(path, "doc")
    assert all(b.page == 0 for b in parsed.blocks)
    assert parsed.page_sizes == []
    assert "no_page_layout" in parsed.warnings


def test_markdown_bom_does_not_break_heading(tmp_path):
    """带 BOM 的文件若按 utf-8 读，`\\ufeff#` 会挡住标题识别 → 退化成段落。
    以 utf-8-sig 读可避免，这里锁住该行为。"""
    path = _write(tmp_path, "bom.md", "# 带 BOM 的标题\n\n正文。\n", encoding="utf-8-sig")
    parsed = parse_markdown(path, "bom")
    assert parsed.blocks[0].block_type == "title"
    assert parsed.blocks[0].text == "带 BOM 的标题"


def test_markdown_gb18030_fallback_is_reported(tmp_path):
    """GBK 写的中文文件不能变成乱码：退回 gb18030 解码，并如实记 warning。"""
    path = _write(tmp_path, "gbk.md", "# 中文标题\n\n这是一段中文正文。\n", encoding="gb18030")
    parsed = parse_markdown(path, "gbk")
    assert "中文标题" in parsed.blocks[0].text
    assert "decoded_gb18030" in parsed.warnings


def test_markdown_empty_file_reports_no_blocks(tmp_path):
    path = _write(tmp_path, "empty.md", "\n\n   \n")
    parsed = parse_markdown(path, "empty")
    assert parsed.blocks == []
    assert "no_blocks_extracted" in parsed.warnings


# ------------------------------------------------------------------ 格式分流
@pytest.mark.parametrize("name,expected", [
    ("a.md", "markdown"),
    ("a.markdown", "markdown"),
    ("a.MD", "markdown"),       # 大小写不敏感
    ("a.pdf", "pdf"),
])
def test_parse_document_dispatches_by_suffix(tmp_path, monkeypatch, name, expected):
    called = {}

    def fake_pdf(path, doc_id, **kwargs):
        called["pdf"] = path
        return "PDF_PARSED"

    def fake_md(path, doc_id, **kwargs):
        called["markdown"] = path
        return "MD_PARSED"

    monkeypatch.setattr("kf_ingest.parse.parse_pdf", fake_pdf)
    monkeypatch.setattr("kf_ingest.parse.parse_markdown", fake_md)
    path = _write(tmp_path, name, "# t\n")
    got = parse_document(path, "d")
    assert got == ("PDF_PARSED" if expected == "pdf" else "MD_PARSED")
    assert expected in called


def test_parse_document_rejects_unsupported_format(tmp_path):
    """不支持的格式必须**明确报错**，不能退化成「当纯文本硬塞」——
    否则用户会以为 .docx 已经建好库了。"""
    path = _write(tmp_path, "a.docx", "whatever")
    with pytest.raises(ValueError, match="不支持的文档格式"):
        parse_document(path, "d")


def test_supported_suffixes_include_markdown():
    assert ".pdf" in SUPPORTED_SUFFIXES
    assert ".md" in SUPPORTED_SUFFIXES
    assert ".docx" not in SUPPORTED_SUFFIXES


# ------------------------------------------------------------------ 目录遍历
def test_discover_documents_separates_supported_from_skipped(tmp_path):
    _write(tmp_path, "a.md", "# a\n")
    _write(tmp_path, "b.pdf", "%PDF-1.4")
    _write(tmp_path, "c.docx", "x")
    _write(tmp_path, "d.txt", "x")
    todo, skipped = discover_documents(str(tmp_path))
    assert sorted(os.path.basename(p) for p in todo) == ["a.md", "b.pdf"]
    assert sorted(os.path.basename(p) for p in skipped) == ["c.docx", "d.txt"]


def test_discover_documents_skips_hidden_and_temp_files(tmp_path):
    _write(tmp_path, "a.md", "# a\n")
    _write(tmp_path, ".hidden.md", "# h\n")
    _write(tmp_path, "~$draft.md", "# t\n")
    todo, skipped = discover_documents(str(tmp_path))
    assert [os.path.basename(p) for p in todo] == ["a.md"]
    assert skipped == []


def test_discover_documents_recursive_flag(tmp_path):
    _write(tmp_path, "top.md", "# t\n")
    nested = os.path.join(str(tmp_path), "nested")
    os.makedirs(nested)
    _write(nested, "deep.md", "# d\n")

    shallow, _ = discover_documents(str(tmp_path), recursive=False)
    assert [os.path.basename(p) for p in shallow] == ["top.md"]

    deep, _ = discover_documents(str(tmp_path), recursive=True)
    assert sorted(os.path.basename(p) for p in deep) == ["deep.md", "top.md"]


def test_discover_documents_match_filters_by_substring(tmp_path):
    _write(tmp_path, "灯塔需求.md", "# 灯塔\n")
    _write(tmp_path, "其他.md", "# 其他\n")
    todo, _ = discover_documents(str(tmp_path), match="灯塔")
    assert [os.path.basename(p) for p in todo] == ["灯塔需求.md"]


def test_discover_documents_hidden_dir_not_entered(tmp_path):
    hidden = os.path.join(str(tmp_path), ".git")
    os.makedirs(hidden)
    _write(hidden, "secret.md", "# s\n")
    _write(tmp_path, "ok.md", "# o\n")
    todo, _ = discover_documents(str(tmp_path), recursive=True)
    assert [os.path.basename(p) for p in todo] == ["ok.md"]


# ------------------------------------------------------------------ Markdown 走完整流水线
def test_markdown_runs_through_pipeline_and_keeps_provenance(tmp_path):
    """Markdown 必须能走完整条流水线（解析→清洗→切片），且子块保留字符级溯源。"""
    path = _write(tmp_path, "manual.md", MD_SAMPLE)
    ctx = IngestContext(
        tenant_id="t", kb_id="k", doc_id="manual",
        source_uri=path, source_path=path, version=1, lang="zh",
    )
    run_ingest(ctx, dry_run=True, collection="kf_t_k")

    assert not ctx.errors, ctx.errors
    children = [c for c in ctx.chunks if c.parent_chunk_id is not None]
    assert children, "Markdown 应切出子块"
    # 表格 / 代码作为原子块不得被跨块切开
    kinds = {c.chunk_type for c in children}
    assert "table" in kinds or "code" in kinds

    raw_len = len(ctx.parsed.raw_pages[0])
    for child in children:
        prov = child.provenance
        assert prov is not None, f"子块 {child.chunk_id} 缺少 provenance"
        assert 0 <= prov.char_start <= prov.char_end <= raw_len
