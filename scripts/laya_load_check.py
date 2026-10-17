"""
Knowledge Fabric · Laya 加载自检

目的：**如实**验证决策服务能否把 Laya 权重真正加载起来，并打印真实模式与耗时。
     成功即成功；失败即明确失败原因与退出码，绝不伪造成功（契约第 0 节）。

权重来源优先级（契约第 7 节）：
    LAYA_MODEL_PATH（本地目录）  >  LAYA_MODEL_ID（HF 缓存 / 下载）
仓库在同一 snapshot 内捆绑多个 checkpoint，由 LAYA_SUBFOLDER 与
LAYA_SUBFOLDER_MULTILINGUAL 选择（英文 / 非英文分别路由）。

用法：
    python scripts/laya_load_check.py

退出码：
    0  Laya 权重加载成功且单条推理成功（mode=laya）
    2  权重加载失败，已回退 heuristic（mode=heuristic-fallback，附原因）
    3  权重加载成功但单条推理失败
"""

from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DECISION = os.path.join(_REPO, "services", "decision")
if _DECISION not in sys.path:
    sys.path.insert(0, _DECISION)

PROBE = "MX-200 的价格是多少？"


def main() -> int:
    import app  # noqa: PLC0415  (延迟导入：确保 sys.path 已就绪)

    print("[laya-load-check] 权重来源优先级：LAYA_MODEL_PATH > LAYA_MODEL_ID")
    print(f"  LAYA_MODEL_PATH = {app.LAYA_MODEL_PATH or '(未设置)'}")
    print(f"  LAYA_MODEL_ID   = {app.LAYA_MODEL_ID}")
    print(f"  LAYA_CACHE_DIR  = {app.LAYA_CACHE_DIR}")
    print(f"  LAYA_SUBFOLDER  = {app.LAYA_SUBFOLDER or '(空=直接用目录)'}")
    print(f"  LAYA_SUBFOLDER_MULTILINGUAL = {app.LAYA_SUBFOLDER_MULTILINGUAL or '(空)'}")
    print(f"  LAYA_DEVICE     = {app.LAYA_DEVICE}")
    print(f"  HF_HUB_OFFLINE  = {os.environ.get('HF_HUB_OFFLINE', '(未设置)')}")

    t0 = time.perf_counter()
    rt = app.load_runtime()
    load_ms = (time.perf_counter() - t0) * 1000.0

    state = app.runtime_state()
    print(f"  model_source    = {state['model_source']}")
    print(f"  model_id        = {state['model_id']}")
    print(f"  mode            = {state['mode']}")
    print(f"  load_error      = {state['load_error']}")
    print(f"  checkpoints     = {json.dumps(state['laya_checkpoints'], ensure_ascii=False)}")
    print(f"  ckpt_errors     = {json.dumps(state['laya_checkpoint_errors'], ensure_ascii=False)}")
    print(f"[laya-load-check] 加载耗时 = {load_ms:.1f}ms")

    if not rt.ready:
        # 如实报告失败，并确认降级路径可用。
        print("[fail] Laya 权重未就绪，已回退 heuristic-fallback。")
        print(f"       失败原因: {rt.load_error}")
        feats = app.derive_features(PROBE, app.QueryFeatures())
        decision = app.heuristic_decide(PROBE, feats)
        print(f"       heuristic 回退仍可用: profile={decision['profile']} reason={decision['reason']}")
        return 2

    # 就绪：跑一条真实查询，打印真实档位、校准概率与耗时。
    feats = app.derive_features(PROBE, app.QueryFeatures())
    print(f"  probe features  = lang={feats.lang} "
          f"has_number_or_price={feats.has_number_or_price} length={feats.length}")
    print(f"  语言路由 checkpoint = {rt.subfolder_for(feats.lang)}")

    t1 = time.perf_counter()
    result = rt.decide(PROBE, feats)
    infer_ms = (time.perf_counter() - t1) * 1000.0

    if result is None:
        print("[fail] 权重已加载，但单条推理失败（runtime 记录见日志）。")
        return 3

    probs = sorted(result.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:3]
    print(f"[ok] Laya 加载成功；query={PROBE!r}")
    print(f"     profile={result['profile']}  confidence={result.get('confidence')}  "
          f"推理耗时={infer_ms:.1f}ms  mode=laya")
    print(f"     top-3 概率 = {json.dumps(probs, ensure_ascii=False)}")
    print(f"     reason     = {result.get('reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
