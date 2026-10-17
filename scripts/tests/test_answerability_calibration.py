"""按库 + 按显存档位标定的答案可达阈值：只测纯逻辑（不连服务/不加载模型）。

为什么值得单独锁住：拒答阈值一旦按库或按档位错配，表现是"大库过度拒答、
小库放行越界问题"或"换显卡后拒答行为漂移"——都是静默错误，很难从单条响应看出来。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "services", "retrieve")))

import pytest

from kf_retrieve import config as cfgmod


def _write(tmp_path, data) -> None:
    (tmp_path / "answerability_calibration.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


@pytest.fixture()
def calibration_file(tmp_path, monkeypatch):
    """带 tiers 的新格式（v3）。"""
    data = {
        "version": 3,
        "calibrated_tiers": ["compact", "wide"],
        "kb": {
            "acme/demo": {
                "threshold": 0.5793,          # cosine（与档位无关）
                "tiers": {
                    "compact": {"rerank_top_n": 4, "max_length": 256,
                                "rerank_threshold": 0.1389},
                    "wide": {"rerank_top_n": 20, "max_length": 512,
                             "rerank_threshold": 0.2100},
                },
            },
            "alpha/kbalpha": {
                "threshold": 0.3796,
                "tiers": {
                    "compact": {"rerank_top_n": 4, "max_length": 256,
                                "rerank_threshold": 0.0301},
                },
            },
        },
    }
    _write(tmp_path, data)
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    yield tmp_path
    cfgmod.load_answerability_calibration.cache_clear()


def test_cosine_threshold_is_tier_independent(calibration_file):
    """cosine 是 embedding 层面的分，不该随重排档位变化。"""
    assert cfgmod.answerability_threshold_for("acme", "demo", basis="cosine") == \
        pytest.approx(0.5793)
    assert cfgmod.answerability_threshold_for("acme", "demo", basis="cosine",
                                              tier="wide") == pytest.approx(0.5793)


def test_rerank_threshold_differs_by_tier(calibration_file):
    """同一个库在不同档位下必须拿到各自的阈值——窗口变了分布就变了。"""
    compact = cfgmod.answerability_threshold_for("acme", "demo", basis="rerank",
                                                tier="compact")
    wide = cfgmod.answerability_threshold_for("acme", "demo", basis="rerank",
                                             tier="wide")
    assert compact == pytest.approx(0.1389)
    assert wide == pytest.approx(0.2100)
    assert compact != wide


def test_missing_tier_returns_none_not_flat_value(calibration_file):
    """未标定的档位必须返回 None，**不能**悄悄用别的档位的扁平阈值。

    这是本模块最重要的一条：跨档位复用会让拒答行为在换显卡后系统性偏移，
    而且完全无声——比"没有阈值"危险得多。
    """
    # standard 档位没标定
    assert cfgmod.answerability_threshold_for("acme", "demo", basis="rerank",
                                              tier="standard") is None
    # 不传 tier 时也不该读 tiers 里的值
    assert cfgmod.answerability_threshold_for("acme", "demo", basis="rerank") is None


def test_legacy_flat_format_still_readable(tmp_path, monkeypatch):
    """旧版（v2，无 tiers 字段）在**未指定档位**时仍可用，便于平滑升级。"""
    _write(tmp_path, {"version": 2,
                      "kb": {"x/y": {"threshold": 0.5, "rerank_threshold": 0.11}}})
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.answerability_threshold_for("x", "y", basis="cosine") == \
            pytest.approx(0.5)
        assert cfgmod.answerability_threshold_for("x", "y", basis="rerank") == \
            pytest.approx(0.11)
        # 指定档位后，旧格式没有档位信息 → None
        assert cfgmod.answerability_threshold_for("x", "y", basis="rerank",
                                                  tier="compact") is None
    finally:
        cfgmod.load_answerability_calibration.cache_clear()


def test_uncalibrated_kb_returns_none(calibration_file):
    assert cfgmod.answerability_threshold_for("nope", "nothing", basis="rerank",
                                              tier="compact") is None


def test_big_kb_threshold_higher_than_small_kb(calibration_file):
    """语料越大噪声底越高，阈值也应更高——这是按库标定存在的理由。"""
    big = cfgmod.answerability_threshold_for("acme", "demo", basis="rerank",
                                             tier="compact")
    small = cfgmod.answerability_threshold_for("alpha", "kbalpha", basis="rerank",
                                               tier="compact")
    assert big > small


def test_missing_file_falls_back_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.load_answerability_calibration() == {}
        assert cfgmod.answerability_threshold_for("acme", "demo", basis="rerank",
                                                  tier="compact") is None
    finally:
        cfgmod.load_answerability_calibration.cache_clear()


def test_utf8_bom_tolerated(tmp_path, monkeypatch):
    """PowerShell 等写出的 JSON 常带 BOM，读的时候必须容忍。"""
    raw = '{"kb": {"x/y": {"threshold": 0.5}}}'
    (tmp_path / "answerability_calibration.json").write_bytes(
        b"\xef\xbb\xbf" + raw.encode("utf-8"))
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.answerability_threshold_for("x", "y", basis="cosine") == \
            pytest.approx(0.5)
    finally:
        cfgmod.load_answerability_calibration.cache_clear()


def test_malformed_entry_returns_none(tmp_path, monkeypatch):
    _write(tmp_path, {"kb": {"x/y": {"tiers": {"compact": {"rerank_threshold":
                                                          "not-a-number"}}}}})
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.answerability_threshold_for("x", "y", basis="rerank",
                                                  tier="compact") is None
    finally:
        cfgmod.load_answerability_calibration.cache_clear()
