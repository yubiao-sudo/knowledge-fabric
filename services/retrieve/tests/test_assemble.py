"""组装单测：抗中间遗忘排序、父块回填、去重、预算硬限、ACL、拒答、引用。"""

from kf_retrieve.assemble import (
    acl_allows,
    assemble,
    estimate_tokens,
    interleave_extremes,
)


def cand(chunk_id: str, score: float, content: str, **kw):
    base = {
        "chunk_id": chunk_id,
        "content": content,
        "score": score,
        "source": "minio://bucket/doc.pdf",
        "page": 1,
        "doc_id": "d1",
        "chunk_type": "text",
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ 抗中间遗忘
def test_interleave_extremes_six():
    assert interleave_extremes([0, 1, 2, 3, 4, 5]) == [0, 2, 4, 5, 3, 1]


def test_interleave_extremes_odd_and_small():
    assert interleave_extremes([0, 1, 2, 3, 4]) == [0, 2, 4, 3, 1]
    assert interleave_extremes([0, 1, 2]) == [0, 2, 1]
    assert interleave_extremes([0, 1]) == [0, 1]
    assert interleave_extremes([0]) == [0]


def test_interleave_keeps_highest_at_head():
    items = [{"id": i} for i in range(6)]
    ordered = interleave_extremes(items)
    assert ordered[0]["id"] == 0        # 最相关在首位
    assert ordered[-1]["id"] == 1        # 次相关在末位


# ------------------------------------------------------------------ 父子结构处理
def test_children_under_same_parent_are_not_collapsed():
    """多个不同子块命中同一父块时是不同信息，不得按父块去重压成一条。"""
    parents = {"p1": {"chunk_id": "p1", "content": "PARENT-CONTENT", "source": "minio://p"}}
    result = assemble(
        [
            cand("c1", 0.9, "child1", parent_chunk_id="p1"),
            cand("c2", 0.5, "child2", parent_chunk_id="p1"),
            cand("c3", 0.7, "standalone"),
        ],
        parents=parents,
        budget_tokens=8000,
    )
    chunk_ids = [e["chunk_id"] for e in result["evidence"]]
    assert sorted(chunk_ids) == ["c1", "c2", "c3"], "同一父块下的不同子块都应保留"
    # 默认不替换 content：content 必须仍是命中子块自己的内容
    by_id = {e["chunk_id"]: e for e in result["evidence"]}
    assert by_id["c1"]["content"] == "child1"
    assert by_id["c1"]["parent_chunk_id"] == "p1"
    assert "parent_context" not in by_id["c1"], "默认不带父块上下文"


def test_parent_context_is_attached_without_replacing_content():
    """父块可作为附加上下文附带，但绝不替换 content（否则引用与内容错位）。"""
    parents = {"p1": {"chunk_id": "p1", "content": "PARENT-CONTENT", "source": "minio://p"}}
    result = assemble(
        [cand("c1", 0.9, "child1", parent_chunk_id="p1", char_start=10, char_end=16)],
        parents=parents,
        budget_tokens=8000,
        parent_context_tokens=200,
    )
    top = result["evidence"][0]
    assert top["content"] == "child1"
    assert top["parent_context"] == "PARENT-CONTENT"
    assert top["char_range"] == [10, 16], "偏移必须仍指向命中子块"


def test_backfill_flag_restores_legacy_replace_behaviour():
    """显式开启 backfill 时保持旧行为（用父块替换 content），供确有需要的调用方。"""
    parents = {"p1": {"chunk_id": "p1", "content": "PARENT-CONTENT", "source": "minio://p"}}
    result = assemble(
        [cand("c1", 0.9, "child1", parent_chunk_id="p1")],
        parents=parents,
        budget_tokens=8000,
        parent_context_tokens=200,
        backfill_parent_content=True,
    )
    assert result["evidence"][0]["content"] == "PARENT-CONTENT"


# ------------------------------------------------------------------ 去重
def test_near_duplicate_dropped_by_simhash():
    long_text = "产品保修期为二十四个月，" * 14      # >= 120 字，触发 SimHash
    result = assemble(
        [
            cand("hi", 0.9, long_text),
            cand("dup", 0.6, long_text),             # 完全近似重复
            cand("other", 0.5, "另一条完全不同的短内容"),
        ],
        budget_tokens=8000,
    )
    ids = [e["chunk_id"] for e in result["evidence"]]
    assert "dup" not in ids
    assert "hi" in ids


# ------------------------------------------------------------------ 预算硬限
def test_budget_hard_limit_skips_overflow():
    # 每块 "你好世界" ≈ 5 token
    per = estimate_tokens("你好世界")
    assert per == 5
    result = assemble(
        [cand("a", 0.9, "你好世界"), cand("b", 0.8, "你好世界"), cand("c", 0.7, "你好世界")],
        budget_tokens=12,
    )
    assert len(result["evidence"]) == 2
    assert result["tokens_used"] <= 12


def test_single_oversized_chunk_is_truncated():
    result = assemble([cand("big", 0.9, "你好世界中国")], budget_tokens=3)
    assert len(result["evidence"]) == 1
    assert len(result["evidence"][0]["content"]) < len("你好世界中国")
    assert result["tokens_used"] <= 3


# ------------------------------------------------------------------ 拒答
def test_strict_refuse_below_threshold():
    result = assemble([cand("a", 0.2, "低相关")], threshold=0.55)
    assert result["refused"] is True
    assert result["evidence"] == []


def test_threshold_pass_keeps_evidence():
    result = assemble([cand("a", 0.9, "高相关")], threshold=0.55)
    assert result["refused"] is False
    assert len(result["evidence"]) == 1


# ------------------------------------------------------------------ ACL
def test_acl_allows_with_matching_allow():
    acl = [{"principal_type": "group", "principal_id": "g1", "effect": "allow"}]
    assert acl_allows(acl, {"principal_type": "group", "principal_id": "g1"}) is True
    assert acl_allows(acl, {"principal_type": "user", "principal_id": "u1"}) is False


def test_acl_deny_overrides_allow():
    acl = [
        {"principal_type": "group", "principal_id": "g1", "effect": "allow"},
        {"principal_type": "user", "principal_id": "u9", "effect": "deny"},
    ]
    assert acl_allows(acl, {"principal_type": "user", "principal_id": "u9"}) is False


def test_acl_empty_is_public():
    assert acl_allows([], {"principal_type": "user", "principal_id": "u1"}) is True


def test_assemble_filters_acl_when_principal_given():
    candidates = [
        cand("open", 0.9, "公开块"),
        cand("secret", 0.8, "受限块", acl=[{"principal_type": "group", "principal_id": "g1", "effect": "allow"}]),
    ]
    result = assemble(candidates, principal={"principal_type": "user", "principal_id": "u1"})
    ids = [e["chunk_id"] for e in result["evidence"]]
    assert ids == ["open"]


def test_assemble_no_principal_skips_acl():
    candidates = [
        cand("open", 0.9, "公开块"),
        cand("secret", 0.8, "受限块", acl=[{"principal_type": "group", "principal_id": "g1", "effect": "allow"}]),
    ]
    result = assemble(candidates, principal=None)
    assert len(result["evidence"]) == 2


# ------------------------------------------------------------------ 引用
def test_citations_from_provenance():
    result = assemble(
        [cand("c1", 0.9, "内容", char_start=120, char_end=260, section="1.2 产品概述")],
        budget_tokens=8000,
    )
    cite = result["citations"][0]
    assert cite["doc_id"] == "d1"
    assert cite["page"] == 1
    assert cite["char_range"] == [120, 260]
    assert cite["section"] == "1.2 产品概述"
    assert cite["url"] == "minio://bucket/doc.pdf"
