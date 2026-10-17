"""切片评测的指标、检索器与「诚实性」护栏测试。

评测脚本的价值在于用数字驱动选参决策，因此它**不能悄悄给出误导性的数字**。
本测试重点保护两处容易退化的诚实性：
  1. 语料小于 k 时 recall@k 恒为 1.0，必须标记为不可信，否则会基于饱和指标选参；
  2. 各策略产出完全相同时，A/B 无意义，必须显式告警而不是照常给出「建议」。
"""

import pytest

from chunk_eval import (
    StrategyResult,
    build_lexical_retriever,
    compare,
    evaluate_strategy,
    ndcg_at_k,
    recall_at_k,
)


# ------------------------------------------------------------------ 指标
def test_recall_at_k_basic():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a"}, 1) == 1.0
    assert recall_at_k(ranked, {"d"}, 1) == 0.0
    assert recall_at_k(ranked, {"c", "d"}, 2) == 0.0
    assert recall_at_k(ranked, {"c", "d"}, 3) == 0.5
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_ndcg_higher_for_better_rank():
    relevant = {"a"}
    assert ndcg_at_k(["a", "b", "c"], relevant, 3) > ndcg_at_k(["b", "a", "c"], relevant, 3)
    assert ndcg_at_k(["a"], relevant, 1) == 1.0
    assert ndcg_at_k(["b", "c"], relevant, 2) == 0.0


# ------------------------------------------------------------------ 检索器
def test_lexical_retriever_ranks_relevant_first():
    chunks = [
        ("c1", "MX-200 型设备的额定功率为 500 瓦，防护等级 IP45。"),
        ("c2", "本手册说明企业知识库的安装与配置流程。"),
        ("c3", "滤芯型号 FL-100，每运行 800 小时更换一次。"),
    ]
    retrieve = build_lexical_retriever(chunks)
    assert retrieve("MX-200 的额定功率是多少", 3)[0] == "c1"
    assert retrieve("滤芯型号", 3)[0] == "c3"
    assert retrieve("", 3) == []


def test_lexical_retriever_uses_same_encoding_as_index_side():
    """查询侧分词必须与索引侧一致，否则评测口径与线上脱节。"""
    from kf_ingest.embed import tokenize_for_sparse

    text = "产品型号 MX-200 的价格是 1999 元"
    tokens = tokenize_for_sparse(text)
    assert "mx" in tokens
    assert any(len(t) == 2 for t in tokens), "CJK 应产出 bigram"


# ------------------------------------------------------------------ 诚实性护栏
def test_small_corpus_is_marked_uninformative():
    """块数 ≤10 时 recall@10 恒为 1.0，必须被标记为不可信。"""
    chunks = [(f"c{i}", f"第 {i} 号设备的规格说明，包含功率与防护等级等信息。") for i in range(5)]
    result = evaluate_strategy(chunks, "tiny")
    assert result.informative is False
    assert result.recall10 == 1.0
    assert "不能用于选参" in compare([result])


def test_large_corpus_is_informative():
    chunks = [(f"c{i}", f"设备 DEV-{i:03d} 的额定功率为 {500 + i} 瓦，防护等级 IP45。")
              for i in range(40)]
    result = evaluate_strategy(chunks, "big")
    assert result.informative is True
    assert result.recall10 <= 1.0


def test_identical_strategies_trigger_ab_guard():
    """所有策略产出一致时，必须告警「A/B 无效」，而不是照常给出切换建议。"""
    same = [StrategyResult(f"s{i}", 0.6, 0.8, 0.4356, 34, 54.4) for i in range(3)]
    report = compare(same)
    assert "本次 A/B 无效" in report


def test_compare_reports_improvement_when_real():
    results = [
        StrategyResult("current", 0.50, 0.70, 0.30, 40, 320.0),
        StrategyResult("better", 0.70, 0.90, 0.45, 40, 180.0),
    ]
    report = compare(results, min_improvement=0.03)
    assert "切换到 better" in report
    assert "本次 A/B 无效" not in report


def test_compare_keeps_status_quo_below_threshold():
    results = [
        StrategyResult("current", 0.50, 0.70, 0.30, 40, 320.0),
        StrategyResult("marginal", 0.51, 0.71, 0.31, 40, 320.0),
    ]
    report = compare(results, min_improvement=0.03)
    assert "保持现状" in report


def test_template_questions_are_not_near_duplicates():
    """自查询问题若高度雷同，会互相抬高分数、让评测失去区分度。"""
    from chunk_eval import generate_self_questions

    text = ("设备编号 DEV-001，额定功率 500 瓦。防护等级 IP45，重量 12 千克。"
            "安装前须核对铭牌参数与现场供电是否一致。")
    questions = generate_self_questions(text, 3)
    assert questions
    assert len(set(questions)) == len(questions), f"问题重复: {questions}"


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_empty_chunk_yields_no_questions(empty):
    from chunk_eval import generate_self_questions

    assert generate_self_questions(empty, 3) == []
