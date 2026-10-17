"""去重单测：短文本的"逐字相同"必须被合并（SimHash 那条路管不到）。

## 为什么单独测这个

`_dedupe_near` 原本只走 SimHash，而 SimHash 有 `MIN_LEN_NEAR=120` 的长度下限
（短文本 SimHash 噪声大、容易误判）。于是**短证据的重复永远不被剔除**：

真机上同一段 40 字的规则因为来自两份不同 doc_id（同内容、不同文件名各传了一次），
在检索结果里原样出现了两遍，用户看到的是"同一条证据重复"。

修法是加一层"去空白后逐字比较"，它对所有长度生效且零误判。
"""

from __future__ import annotations

from kf_retrieve.assemble import _dedupe_near, _exact_key


def cand(content: str, score: float = 1.0, chunk_id: str = "", **kw):
    out = {"chunk_id": chunk_id or content[:8], "content": content, "score": score}
    out.update(kw)
    return out


# --------------------------------------------------------------- 精确键
def test_exact_key_strips_all_whitespace():
    assert _exact_key("a b\n\tc  d") == "abcd"
    assert _exact_key("   ") == ""
    assert _exact_key("") == ""


def test_exact_key_keeps_punctuation():
    """标点不同说明是不同内容，不能被归一成同一个键。"""
    assert _exact_key("甲，乙") != _exact_key("甲。乙")


# --------------------------------------------------------------- 短文本去重
def test_short_duplicates_differing_only_by_whitespace_are_merged():
    """这就是真机上的形态：同一条规则，一个 `\\n` 一个 `\\n\\n`。"""
    a = "第五百零三条 校验口令\n接入验收口令是「白泽青鸟」,仅用于本次测试。"
    b = "第五百零三条 校验口令\n\n接入验收口令是「白泽青鸟」,仅用于本次测试。"
    assert len(a) < 120, "这条样本必须短于 SimHash 的长度下限，否则测不到目标路径"
    out = _dedupe_near([cand(a, 0.97), cand(b, 0.97)])
    assert len(out) == 1
    assert out[0]["content"] == a, "应当保留分更高（或先出现）的那条"


def test_short_but_different_content_is_kept():
    a = "第三条 密码长度不得少于十二位。"
    b = "第四条 违规操作导致数据泄露的,将暂停账号权限。"
    assert len(_dedupe_near([cand(a), cand(b)])) == 2


def test_high_score_wins_when_identical():
    a = "同样的短句内容"
    out = _dedupe_near([cand(a, 0.9, chunk_id="hi"), cand(a, 0.5, chunk_id="lo")])
    assert len(out) == 1
    assert out[0]["chunk_id"] == "hi"


def test_three_way_duplicate_collapses_to_one():
    a = "重复三次的同一段短文本内容"
    out = _dedupe_near([cand(a), cand(a), cand(a.replace("同一段", "同一段 "))])
    assert len(out) == 1


# --------------------------------------------------------------- 边界
def test_empty_content_is_not_dropped():
    """空/纯空白没有内容可比，但也不该被""当重复"删掉——交给下游预算处理。"""
    out = _dedupe_near([cand(""), cand("   "), cand("abc")])
    assert len(out) == 3


def test_order_is_preserved():
    items = [cand("第一段内容"), cand("第二段内容"), cand("第一段内容")]
    out = _dedupe_near(items)
    assert [c["content"] for c in out] == ["第一段内容", "第二段内容"]


def test_long_near_duplicates_still_use_simhash():
    """长文本仍走 SimHash 近似判重（精确层只是补上短文本的缺口）。

    注意阈值的实际尺度：`SIMHASH_MAX_DISTANCE=3` 很紧，改几个字就可能超出。
    所以这里用"只多一点标点"的真实近重复样本；改 4 个字（如 运维→运营）不该被合并
    —— 那是不同内容，误合并才是灾难。
    """
    base = "这是一段足够长的运维规范文本,用于触发 SimHash 分支。" * 4
    near = base + "。"
    out = _dedupe_near([cand(base), cand(near)])
    assert len(out) == 1, "仅差一个标点的长文本应被合并"


def test_long_text_with_real_word_change_is_kept():
    """改了几个实词的长文本**不该**被合并：误删正确证据比留一份重复更糟。"""
    base = "这是一段足够长的运维规范文本,用于触发 SimHash 分支。" * 4
    changed = base.replace("运维", "运营")
    out = _dedupe_near([cand(base), cand(changed)])
    assert len(out) == 2
