"""按显存自适应裁剪重排窗口：纯逻辑测试（不加载模型、不碰 GPU）。

锁住三件事：
1. 档位边界正确（4GB 走 compact，8GB 走 standard，24GB 走 wide）；
2. 无 GPU 时有保守档位，不能假装能用 20 个候选；
3. `effective_top_n` 的语义：档位是上限，配置可以更小但不能被抬高。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "services", "retrieve")))

import pytest

from kf_retrieve import adaptive


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    monkeypatch.delenv("KF_RERANK_TOP_N", raising=False)
    adaptive.reset_cache()
    yield
    adaptive.reset_cache()


@pytest.mark.parametrize("total_mb,expected", [
    (4096, "compact"),
    (6144, "standard"),
    (8192, "standard"),
    (10240, "wide"),
    (24576, "wide"),
])
def test_tier_boundaries(total_mb, expected):
    tier, _ = adaptive.detect(total_mb=total_mb, cuda_available=True)
    assert tier.name == expected


def test_cpu_tier_is_conservative():
    tier, reason = adaptive.detect(cuda_available=False)
    assert tier.name == "cpu"
    assert tier.rerank_top_n <= 2, "无 GPU 时不能排太多候选（CPU 上约 3.4s/对）"
    assert "CUDA" in reason


def test_enum_override_wins(monkeypatch):
    monkeypatch.setenv("KF_RERANK_TOP_N", "8")
    adaptive.reset_cache()
    tier, reason = adaptive.detect(total_mb=4096, cuda_available=True)
    assert tier.rerank_top_n == 8
    assert "KF_RERANK_TOP_N" in reason


def test_invalid_enum_override_ignored(monkeypatch):
    monkeypatch.setenv("KF_RERANK_TOP_N", "abc")
    adaptive.reset_cache()
    tier, _ = adaptive.detect(total_mb=4096, cuda_available=True)
    assert tier.name == "compact"


def test_effective_top_n_uses_tier(monkeypatch):
    """档位值必须压过配置里的旧经验值，否则自适应形同虚设。"""
    monkeypatch.setattr(adaptive, "current", lambda: (
        adaptive.TIERS[2], "test"))          # wide: top_n=20
    n, tier_name, reason = adaptive.effective_top_n(4)
    assert (n, tier_name) == (20, "wide"), "大卡上不该被配置里的旧值压住"
    assert "覆盖" in reason


def test_enum_override_beats_tier(monkeypatch):
    """想固定窗口只能用 KF_RERANK_TOP_N，这是明确的"就要这个数"。"""
    monkeypatch.setenv("KF_RERANK_TOP_N", "6")
    adaptive.reset_cache()
    tier, _ = adaptive.detect(total_mb=24576, cuda_available=True)
    assert tier.rerank_top_n == 6


def test_effective_top_n_handles_none(monkeypatch):
    monkeypatch.setattr(adaptive, "current", lambda: (
        adaptive.TIERS[0], "test"))          # compact: top_n=4
    n, _, _ = adaptive.effective_top_n(None)
    assert n == 4


def test_effective_top_n_handles_garbage(monkeypatch):
    monkeypatch.setattr(adaptive, "current", lambda: (
        adaptive.TIERS[0], "test"))
    n, _, _ = adaptive.effective_top_n("not-a-number")
    assert n == 4


def test_status_shape(monkeypatch):
    monkeypatch.setattr(adaptive, "current", lambda: (
        adaptive.TIERS[0], "测试原因"))
    s = adaptive.status()
    for key in ("name", "rerank_top_n", "max_length", "reason"):
        assert key in s
    assert s["reason"] == "测试原因"
