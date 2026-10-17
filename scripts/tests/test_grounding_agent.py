"""接地引用 Agent 的单测：配置文件加载、密钥打码、引用校验。

这些行为直接决定"配置错在哪能不能自查"与"引用是否可信"，因此必须锁住。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples")))

from grounding_agent import (  # noqa: E402
    CitationCheck,
    _strip_quotes,
    load_config_files,
    load_env_file,
    llm_config,
    mask,
    verify_citation,
)


# ------------------------------------------------------------------ 配置文件解析
def test_load_env_file_basic(tmp_path):
    f = tmp_path / "c.env"
    f.write_text("A=1\nB=two\n", encoding="utf-8")
    for k in ("A", "B"):
        os.environ.pop(k, None)
    applied = load_env_file(str(f))
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "two"
    assert set(applied) == {"A", "B"}


def test_env_var_wins_over_file(tmp_path):
    """环境变量优先：便于临时覆盖文件里的配置。"""
    f = tmp_path / "c.env"
    f.write_text("KF_TEST_KEY=from_file\n", encoding="utf-8")
    os.environ["KF_TEST_KEY"] = "from_env"
    try:
        applied = load_env_file(str(f))
        assert os.environ["KF_TEST_KEY"] == "from_env"
        assert "KF_TEST_KEY" not in applied, "已存在的环境变量不应被计入写入"
    finally:
        os.environ.pop("KF_TEST_KEY", None)


def test_export_prefix_and_quotes(tmp_path):
    f = tmp_path / "c.env"
    f.write_text(
        "export KF_Q1='single quoted'\n"
        'KF_Q2="double quoted"\n'
        "KF_Q3=plain\n",
        encoding="utf-8",
    )
    for k in ("KF_Q1", "KF_Q2", "KF_Q3"):
        os.environ.pop(k, None)
    load_env_file(str(f))
    assert os.environ["KF_Q1"] == "single quoted"
    assert os.environ["KF_Q2"] == "double quoted"
    assert os.environ["KF_Q3"] == "plain"
    for k in ("KF_Q1", "KF_Q2", "KF_Q3"):
        os.environ.pop(k, None)


def test_line_comments_and_blank_lines_skipped(tmp_path):
    f = tmp_path / "c.env"
    f.write_text("# 注释行\n\n   \nKF_C1=ok\n# 尾部注释行\n", encoding="utf-8")
    os.environ.pop("KF_C1", None)
    load_env_file(str(f))
    assert os.environ["KF_C1"] == "ok"
    os.environ.pop("KF_C1", None)


def test_inline_hash_is_preserved(tmp_path):
    """行内 # 不算注释——密钥里可能含 #，截断会导致鉴权失败。"""
    f = tmp_path / "c.env"
    f.write_text("KF_HASH=abc#def\n", encoding="utf-8")
    os.environ.pop("KF_HASH", None)
    load_env_file(str(f))
    assert os.environ["KF_HASH"] == "abc#def"
    os.environ.pop("KF_HASH", None)


def test_empty_value_is_written(tmp_path):
    """空值要真的写入（用于显式置空，例如让 VLLM_URL= 失效）。"""
    f = tmp_path / "c.env"
    f.write_text("KF_EMPTY=\n", encoding="utf-8")
    os.environ["KF_EMPTY"] = "previous"
    try:
        load_env_file(str(f), override=True)
        assert os.environ["KF_EMPTY"] == ""
    finally:
        os.environ.pop("KF_EMPTY", None)


def test_missing_file_is_silent(tmp_path):
    missing = str(tmp_path / "nope.env")
    assert load_env_file(missing) == []


def test_strip_quotes_helper():
    assert _strip_quotes("'a'") == "a"
    assert _strip_quotes('"b"') == "b"
    assert _strip_quotes("c") == "c"
    assert _strip_quotes("'") == "'"          # 不完整的引号原样保留
    assert _strip_quotes("") == ""


def test_load_config_files_reports_existence(tmp_path):
    f = tmp_path / "exists.env"
    f.write_text("KF_EXIST=1\n", encoding="utf-8")
    os.environ.pop("KF_EXIST", None)
    report = load_config_files(str(f))
    assert report, "应至少包含指定的文件"
    assert report[0]["exists"] is True
    assert report[0]["path"] == os.path.abspath(str(f))
    # 不存在的默认路径也要如实出现在报告里（exists=False），便于 --show-config 展示
    assert any(item["exists"] is False for item in report) or len(report) == 1
    os.environ.pop("KF_EXIST", None)


# ------------------------------------------------------------------ 密钥打码
def test_mask_hides_secret():
    assert mask("") == "(未设置)"
    short = mask("sk-abc")
    assert "sk-abc" not in short and "长度 6" in short
    # 用**明显的假密钥**做样本。此前这里放的是仓库作者的真实 DeepSeek key，
    # 一旦入 Git 历史就等于把密钥公开——测试数据必须是假的。
    fake_key = "sk-0123456789abcdef0123456789abcdef"
    masked = mask(fake_key)
    assert fake_key not in masked, "完整密钥绝不能出现在输出里"
    assert masked.startswith("sk-0") and "cdef" in masked and "长度 35" in masked


# ------------------------------------------------------------------ 引用校验
def _ev(chunk_id="c1", content="产品 X 的保修期为 24 个月。", char_range=(0, 20),
        page=1, offset_chunk_id=None):
    d = {"chunk_id": chunk_id, "content": content, "char_range": list(char_range), "page": page}
    if offset_chunk_id is not None:
        d["offset_chunk_id"] = offset_chunk_id
    return d


def test_citation_verified_when_content_matches_authoritative():
    auth = {"c1": {"content": "产品 X 的保修期为 24 个月。", "page": 1}}
    check = verify_citation(_ev(offset_chunk_id="c1"), auth)
    assert check.status == "verified"
    assert check.matched_chars > 0


def test_citation_unverified_on_inverted_range():
    auth = {"c1": {"content": "x", "page": 1}}
    check = verify_citation(_ev(char_range=(100, 10), offset_chunk_id="c1"), auth)
    assert check.status == "unverified"
    assert "倒置" in check.detail


def test_citation_unverified_when_offset_belongs_to_other_chunk():
    """父块回填曾导致 content 与偏移不同源——这是必须被抓住的失效模式。"""
    auth = {"c1": {"content": "产品 X 的保修期为 24 个月。", "page": 1}}
    check = verify_citation(_ev(offset_chunk_id="parent_p1"), auth)
    assert check.status == "unverified"
    assert "偏移归属不一致" in check.detail


def test_citation_unverified_when_content_differs_from_authoritative():
    auth = {"c1": {"content": "完全不相干的另一段内容，讲的是安装步骤与注意事项。", "page": 1}}
    check = verify_citation(_ev(content="产品 X 的保修期为 24 个月。", offset_chunk_id="c1"), auth)
    assert check.status == "unverified"
    assert "重合" in check.detail


def test_citation_self_consistent_without_authoritative_source():
    """取不到权威内容时只报"自洽"，不谎称 verified。"""
    check = verify_citation(_ev(offset_chunk_id="c1"), {})
    assert check.status == "self_consistent"


def test_citation_check_as_dict_roundtrip():
    check = CitationCheck("c1", 1, [0, 5], "verified", "ok", 5)
    d = check.as_dict()
    assert d["chunk_id"] == "c1" and d["status"] == "verified" and d["matched_chars"] == 5
