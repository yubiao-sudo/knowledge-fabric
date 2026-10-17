"""语言判定回归测试。

背景（真机复现的缺陷）：`detect_lang` 原按**原始字符数**比较 `han >= latin`，
把中英混排的技术文档判成 `en`；而稀疏通道按文档语言标签过滤，于是**中文查询
召回为 0**。这里锁住「汉字按 5 倍权重」的修正，防止回归。
"""

import pytest

from kf_ingest.tokenize import CJK_LATIN_WEIGHT, detect_lang


def test_pure_chinese_is_zh():
    assert detect_lang("这是一段纯中文的正文，用于验证语言判定。") == "zh"


def test_pure_english_is_en():
    assert detect_lang("This is a plain English paragraph about retrieval.") == "en"


def test_mixed_technical_chinese_is_zh():
    """真机原型：开头 4000 字符内 汉字 714 / 拉丁 2286（旧规则判 en）。"""
    text = "数据库设计方案_合并完整版 " + "abc_def " * 350 + "数据库表结构设计说明。" * 200
    assert detect_lang(text) == "zh"


def test_code_heavy_chinese_markdown_is_zh():
    """代码占比很高的中文技术文档仍判 zh（汉字绝对量大）。

    真机原型：全文汉字 6733 / 拉丁 73940（比例仅 0.09）——加权比例仍不够，
    靠 `HAN_ABSOLUTE_FLOOR` 兜住。
    """
    text = "增量优化方案_合并完整版\n\n" + "本节说明增量更新的实现思路与落地步骤，" * 40 \
        + "\n```python\n" + "def compute_metrics(values): return sum(values) / len(values)\n" * 60 \
        + "```\n"
    assert detect_lang(text) == "zh"


def test_english_doc_with_stray_chinese_is_still_en():
    """偶发少量汉字不应把英文文档判成中文。"""
    text = "Overview of the retrieval pipeline. " * 40 + "机器人"
    assert detect_lang(text) == "en"


def test_japanese_kana_wins_over_kanji():
    assert detect_lang("これは日本語のテキストです。") == "ja"


def test_korean_hangul_wins():
    assert detect_lang("이것은 한국어 텍스트입니다.") == "ko"


def test_empty_and_symbol_only():
    assert detect_lang("") == "unknown"
    assert detect_lang("12345 !@#$%^&*()") == "unknown"


def test_weight_is_documented_value():
    """权重是显式常量，不是魔法数字散在代码里——这里锁住它的取值，便于将来调参时被看见。"""
    assert CJK_LATIN_WEIGHT == 5


@pytest.mark.parametrize("han,latin,expected", [
    (1, 1000, "en"),      # 英文文档里一个汉字
    (50, 100, "zh"),      # 中文占比已明确
    (0, 10, "en"),
])
def test_weighted_boundary_cases(han, latin, expected):
    text = "汉" * han + "a" * latin
    assert detect_lang(text) == expected
