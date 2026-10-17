"""知识库导航（kb_navigator）单测：只测纯逻辑，不依赖真实服务。

可达分"算不出来"与"真的是 0"必须被区分——把前者显示成 0.0000 会误导排序与判断，
这正是本项目反复出现的"静默降级"模式。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples")))

from kb_navigator import DiscoveryHit, KBInfo  # noqa: E402


def test_kb_key_format():
    info = KBInfo("acme", "demo", 3, 100, ["a.md"])
    assert info.key == "acme/demo"
    assert info.as_dict()["docs"] == 3
    assert info.as_dict()["titles"] == ["a.md"]


def test_answerability_display_distinguishes_unavailable_from_zero():
    zero = DiscoveryHit("t", "k", evidence=5, answerability=0.0, answerability_available=True)
    na = DiscoveryHit("t", "k", evidence=5, answerability=0.0, answerability_available=False)
    assert zero.answerability_display == "0.0000", "真的是 0 分要显示 0.0000"
    assert na.answerability_display == "N/A", "算不出来必须显示 N/A，不能画成 0"
    assert zero.as_dict()["answerability"] == 0.0
    assert na.as_dict()["answerability"] is None, "机读输出同样要用 null 区分"


def test_discovery_hit_dict_shape():
    hit = DiscoveryHit("acme", "demo", evidence=8, answerability=0.58347,
                       answerability_available=True, top_snippet="片段")
    d = hit.as_dict()
    assert d["tenant"] == "acme" and d["kb"] == "demo"
    assert d["answerability"] == 0.5835          # 四舍五入到 4 位
    assert d["evidence"] == 8 and d["top_snippet"] == "片段"


def test_sort_order_prefers_answerable_then_scored():
    """排序：未拒答优先 → 可达分可算的优先 → 分数降序。"""
    hits = [
        DiscoveryHit("a", "refused", evidence=0, refused=True),
        DiscoveryHit("a", "na", evidence=5, answerability_available=False),
        DiscoveryHit("a", "low", evidence=5, answerability=0.30, answerability_available=True),
        DiscoveryHit("a", "high", evidence=5, answerability=0.58, answerability_available=True),
    ]
    hits.sort(key=lambda h: (h.refused, not h.answerability_available, -h.answerability))
    assert [h.kb for h in hits] == ["high", "low", "na", "refused"]
