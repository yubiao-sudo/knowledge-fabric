"""Laya 加载路径与模式暴露的回归测试。

保护四件事：
  1. `LAYA_MODEL_PATH`（本地目录）必须优先于 `LAYA_MODEL_ID`（联网下载）；
  2. `/health` 与 `/decide` 必须**如实**暴露当前模式（laya / heuristic-fallback）；
  3. Laya 是**判定引擎**而非分类模型——映射必须是按档位名取值，不得位置映射；
  4. 按语言选择 checkpoint（英文 typed-decisions / 其余 multilingual）。

这些测试都不得触发真实下载：本地目录缺失时必须在联网前就失败。
"""

import os

import pytest

import app


# ------------------------------------------------------------------ 权重来源
def test_model_source_prefers_local_path(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "LAYA_MODEL_PATH", str(tmp_path))
    kind, source = app.model_source()
    assert kind == "local"
    assert source == str(tmp_path)


def test_model_source_falls_back_to_hf_id(monkeypatch):
    monkeypatch.setattr(app, "LAYA_MODEL_PATH", "")
    kind, source = app.model_source()
    assert kind == "hf-download"
    assert source == app.LAYA_MODEL_ID


# ------------------------------------------------------------------ 模式暴露
def test_runtime_state_exposes_mode_and_load_error_without_loading():
    state = app.runtime_state()
    assert state["mode"] == "heuristic-fallback"
    assert "model_source" in state
    assert "load_error" in state
    # 未加载时不应有失败原因被伪造
    assert state["load_error"] is None


def test_missing_local_path_records_error_without_download(monkeypatch, tmp_path):
    missing = os.path.join(str(tmp_path), "does-not-exist")
    monkeypatch.setattr(app, "LAYA_MODEL_PATH", missing)
    rt = app.LayaRuntime()
    assert rt.ready is False
    assert rt.kind == "local"
    assert rt.load_error is not None
    assert "目录不存在" in rt.load_error


# ------------------------------------------------------------------ checkpoint 选择
def test_subfolder_for_routes_by_language():
    """英文走 typed-decisions，非英文走 multilingual（英文版在非英文会显著退化）。"""
    rt = app.LayaRuntime.__new__(app.LayaRuntime)  # 绕过 __init__，不触发真实加载
    assert rt.subfolder_for("zh") == app.LAYA_SUBFOLDER_MULTILINGUAL
    assert rt.subfolder_for("ja") == app.LAYA_SUBFOLDER_MULTILINGUAL
    assert rt.subfolder_for("en") == app.LAYA_SUBFOLDER


def test_subfolders_deduplicates_and_falls_back_to_root():
    rt = app.LayaRuntime.__new__(app.LayaRuntime)
    assert rt.subfolders() == [app.LAYA_SUBFOLDER, app.LAYA_SUBFOLDER_MULTILINGUAL]
    orig_a, orig_b = app.LAYA_SUBFOLDER, app.LAYA_SUBFOLDER_MULTILINGUAL
    try:
        app.LAYA_SUBFOLDER = ""
        app.LAYA_SUBFOLDER_MULTILINGUAL = ""
        # 两个都留空 → 直接用 LAYA_MODEL_PATH 指向的单个 checkpoint 目录
        assert rt.subfolders() == [""]
    finally:
        app.LAYA_SUBFOLDER, app.LAYA_SUBFOLDER_MULTILINGUAL = orig_a, orig_b


# ------------------------------------------------------------------ 判定映射（真实 API 口径）
def test_profile_questions_cover_every_profile():
    """choice 题的 criteria 键必须与 PROFILES 完全一致，否则映射必然漏档位。"""
    criteria = app.PROFILE_QUESTIONS["retrieval_profile"]["criteria"]
    assert set(criteria) == set(app.PROFILES)


def test_map_laya_answers_uses_named_probabilities_not_positions():
    probs = {name: 0.01 for name in app.PROFILES}
    probs["exact_lookup"] = 0.71
    raw = {
        "answers": {
            "retrieval_profile": {
                "type": "choice",
                "choice": "exact_lookup",
                "confidence": 0.71,
                "probabilities": probs,
            },
            "needs_decomposition": {"type": "noul", "noul": 0.23},
        }
    }
    got = app.map_laya_answers(raw)
    assert got["profile"] == "exact_lookup"
    assert got["confidence"] == 0.71
    assert got["probabilities"]["exact_lookup"] == 0.71
    assert "needs_decomposition=0.230" in got["reason"]


def test_map_laya_answers_falls_back_to_choice_probability():
    """模型未给 confidence 时，用该档位自身的概率，不凭空造 0.5。"""
    raw = {"answers": {"retrieval_profile": {"choice": "naive", "probabilities": {"naive": 0.62}}}}
    assert app.map_laya_answers(raw)["confidence"] == 0.62


def test_map_laya_answers_rejects_unknown_or_empty():
    assert app.map_laya_answers({"answers": {"retrieval_profile": {"choice": "nope"}}}) is None
    assert app.map_laya_answers({"answers": {}}) is None
    assert app.map_laya_answers({}) is None
    assert app.map_laya_answers(None) is None


# ------------------------------------------------------------------ 来源与参数量标注
def test_model_source_distinguishes_cache_from_download(monkeypatch, tmp_path):
    """有本地缓存时必须报 hf-cache，否则「离线复用他机权重」会被误读成又下载了一遍。"""
    monkeypatch.setattr(app, "LAYA_MODEL_PATH", "")
    monkeypatch.setattr(app, "LAYA_CACHE_DIR", str(tmp_path))
    kind, _ = app.model_source()
    assert kind == "hf-download"

    (tmp_path / "hub").mkdir()
    monkeypatch.setattr(app, "LAYA_CACHE_DIR", str(tmp_path))
    kind, _ = app.model_source()
    assert kind == "hf-cache"

    # 本地目录优先于缓存
    monkeypatch.setattr(app, "LAYA_MODEL_PATH", str(tmp_path))
    assert app.model_source()[0] == "local"


def test_checkpoint_params_are_per_checkpoint():
    """model.params 必须按实际服务的 checkpoint 取值，不能写死。"""
    assert app._CHECKPOINT_PARAMS["multilingual"] == "322M"
    assert app._CHECKPOINT_PARAMS["typed-decisions"] == "421M"
    # 未知 / 缺失 checkpoint 必须返回 None，而不是编一个参数量
    assert app._CHECKPOINT_PARAMS.get("unknown") is None
    assert app._CHECKPOINT_PARAMS.get(None) is None


def test_decide_returns_checkpoint_name():
    """decide() 必须把实际 checkpoint 名带出来，供上层标注。"""

    class _FakeAgent:
        def system_one(self, query, questions):
            return {
                "answers": {
                    "retrieval_profile": {
                        "choice": "hybrid_rerank",
                        "confidence": 0.9,
                        "probabilities": {"hybrid_rerank": 0.9},
                    }
                }
            }

    rt = app.LayaRuntime.__new__(app.LayaRuntime)
    rt.ready = True
    rt._agents = {app.LAYA_SUBFOLDER_MULTILINGUAL: _FakeAgent()}
    rt._errors = {}
    got = rt.decide("任意查询 zz", app.QueryFeatures(lang="zh"))
    assert got["checkpoint"] == app.LAYA_SUBFOLDER_MULTILINGUAL
    assert app.LAYA_SUBFOLDER_MULTILINGUAL in got["reason"]


# ------------------------------------------------------------------ 低置信护栏（阈值基准）
def test_gate_uses_laya_top1_not_uncalibrated_confidence():
    """Laya 的 confidence 未校准，判对的档位会因它低于阈值被误丢；
    因此阈值基准必须取概率分布 top-1（真机复现：exact_lookup conf=0.3895、
    top1=0.4909 → 按 top-1 应保留 exact_lookup）。"""
    probs = {name: 0.01 for name in app.PROFILES}
    probs["exact_lookup"] = 0.4909
    probs["strict_refuse"] = 0.3034
    profile, fb, basis, value = app.gate_profile("exact_lookup", 0.3895, probs, used_laya=True)
    assert profile == "exact_lookup", "top-1 越过阈值就不该被回退"
    assert fb is False
    assert basis == "laya_top1_probability"
    assert value == pytest.approx(0.4909)


def test_gate_laya_still_falls_back_when_top1_low():
    """top-1 也低于阈值时，Laya 结果同样必须回退安全默认档。"""
    probs = {name: 0.01 for name in app.PROFILES}
    probs["naive"] = 0.30
    profile, fb, basis, value = app.gate_profile("naive", 0.29, probs, used_laya=True)
    assert profile == app.SAFE_DEFAULT
    assert fb is True
    assert basis == "laya_top1_probability"
    assert value == pytest.approx(0.30)


def test_gate_heuristic_uses_own_confidence():
    """heuristic 不产出概率分布，阈值必须仍用 confidence（不得退化成 0 误回退）。"""
    profile, fb, basis, value = app.gate_profile("exact_lookup", 0.62, None, used_laya=False)
    assert profile == "exact_lookup"
    assert fb is False
    assert basis == "confidence"
    assert value == pytest.approx(0.62)

    low, fb2, basis2, _ = app.gate_profile("exact_lookup", 0.30, None, used_laya=False)
    assert low == app.SAFE_DEFAULT and fb2 is True
    assert basis2 == "confidence"


def test_gate_laya_without_probabilities_falls_back_to_confidence():
    """Laya 只回 confidence、没回概率分布时，只能退回用 confidence，不能凭空造概率。"""
    profile, fb, basis, value = app.gate_profile("naive", 0.70, {}, used_laya=True)
    assert profile == "naive" and fb is False
    assert basis == "confidence"
    assert value == pytest.approx(0.70)
