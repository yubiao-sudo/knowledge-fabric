"""按库标定的答案可达阈值：只测纯逻辑（不连服务/不加载模型）。

为什么值得单独锁住：拒答阈值一旦按库错配，表现是"大库过度拒答、小库放行越界问题"
——两种都是静默错误，很难从单条响应看出来。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "services", "retrieve")))

import pytest

from kf_retrieve import config as cfgmod


@pytest.fixture()
def calibration_file(tmp_path, monkeypatch):
    """把 config_dir 指向临时目录，写一份标定文件。"""
    data = {
        "version": 1,
        "probe_count": 10,
        "percentile": 0.9,
        "margin": 0.03,
        "kb": {
            "acme/demo": {"noise_floor_p90": 0.5493, "threshold": 0.5793},
            "alpha/kbalpha": {"noise_floor_p90": 0.3496, "threshold": 0.3796},
        },
    }
    path = tmp_path / "answerability_calibration.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    yield path
    cfgmod.load_answerability_calibration.cache_clear()


def test_threshold_read_for_known_kb(calibration_file):
    assert cfgmod.answerability_threshold_for("acme", "demo") == pytest.approx(0.5793)
    assert cfgmod.answerability_threshold_for("alpha", "kbalpha") == pytest.approx(0.3796)


def test_threshold_none_for_uncalibrated_kb(calibration_file):
    """未标定的库必须返回 None，让调用方回退到全局底线——不能瞎给一个数。"""
    assert cfgmod.answerability_threshold_for("nope", "nothing") is None


def test_big_kb_threshold_higher_than_small_kb(calibration_file):
    """语料越大，无分噪声底越高，阈值也应更高——这是自适应存在的理由。"""
    big = cfgmod.answerability_threshold_for("acme", "demo")
    small = cfgmod.answerability_threshold_for("alpha", "kbalpha")
    assert big > small


def test_missing_file_falls_back_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.load_answerability_calibration() == {}
        assert cfgmod.answerability_threshold_for("acme", "demo") is None
    finally:
        cfgmod.load_answerability_calibration.cache_clear()


def test_utf8_bom_tolerated(tmp_path, monkeypatch):
    """PowerShell 等写出的 JSON 常带 BOM，读的时候必须容忍。"""
    path = tmp_path / "answerability_calibration.json"
    raw = '{"kb": {"x/y": {"threshold": 0.5}}}'
    path.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.answerability_threshold_for("x", "y") == pytest.approx(0.5)
    finally:
        cfgmod.load_answerability_calibration.cache_clear()


def test_malformed_entry_returns_none(tmp_path, monkeypatch):
    path = tmp_path / "answerability_calibration.json"
    path.write_text('{"kb": {"x/y": {"threshold": "not-a-number"}}}', encoding="utf-8")
    monkeypatch.setattr(cfgmod, "config_dir", lambda: str(tmp_path))
    cfgmod.load_answerability_calibration.cache_clear()
    try:
        assert cfgmod.answerability_threshold_for("x", "y") is None
    finally:
        cfgmod.load_answerability_calibration.cache_clear()
