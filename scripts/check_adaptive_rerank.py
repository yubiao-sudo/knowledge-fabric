"""自适应重排窗口自检：一次跑完「档位 → 窗口 → 标定覆盖 → 实测延迟与拒答」。

## 什么时候用

* 换机器 / 换显卡后，确认自适应选对了档位；
* 改了 `rerank_top_n`、`KF_RERANK_TOP_N`、`KF_RERANK_MAX_LEN` 后，确认生效；
* 重跑标定后，确认当前档位**确实有**专属阈值（而不是悄悄回落到全局底线）。

## 用法

    python scripts\\check_adaptive_rerank.py
    python scripts\\check_adaptive_rerank.py --url http://127.0.0.1:8082
    python scripts\\check_adaptive_rerank.py --tenant acme --kb demo --quick

退出码：0 = 全部符合预期；1 = 有不符合项（会逐条列原因）。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services",
                                                "retrieve")))

import httpx

# 两条真问题 + 两条越界：一正一反，才能同时看出"该答的答了"和"该拒的拒了"
CASES = [
    ("密码长度要求是多少位", "answer"),
    ("灯塔的设计目标是什么", "answer"),
    ("今天天气怎么样，帮我订张去上海的机票", "refuse"),
    ("如何用 Rust 写一个 WebSocket 服务器", "refuse"),
]


def ok(label: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--tenant", default="acme")
    ap.add_argument("--kb", default="demo")
    ap.add_argument("--quick", action="store_true", help="只测 2 条查询")
    args = ap.parse_args()

    results: List[bool] = []
    client = httpx.Client(timeout=300.0)

    # ---------- 1) 服务可达 ----------
    print("\n=== 1. 服务与档位 ===")
    try:
        health = client.get(f"{args.url}/health").json()
    except Exception as exc:
        ok("检索服务可达", False, f"{args.url} 连不上：{exc}")
        print("\n先启动服务（scripts\\run_retrieve.cmd），再重跑本脚本。")
        return 1
    results.append(ok("检索服务可达", True, args.url))

    emb = health.get("embed") or {}
    rk = health.get("reranker") or {}
    local = rk.get("local") or {}
    adaptive = rk.get("adaptive") or {}

    print(f"  嵌入   : device={emb.get('device')} backend={emb.get('backend')}")
    print(f"  重排   : device={local.get('device')} dtype={local.get('dtype')} "
          f"max_length={local.get('max_length')} ready={local.get('ready')}")
    print(f"  档位   : {adaptive.get('name')}  reason={adaptive.get('reason')}")
    print(f"  窗口   : rerank_top_n={adaptive.get('rerank_top_n')}"
          + ("  (由 KF_RERANK_TOP_N 强制)" if adaptive.get("forced_by_env") else ""))

    results.append(ok("重排模型已就绪", bool(local.get("ready")),
                      f"device={local.get('device')}"))
    results.append(ok("档位已识别", bool(adaptive.get("name")),
                      str(adaptive.get("name"))))
    # max_length 应跟随档位（除非被环境变量强制）
    if not os.environ.get("KF_RERANK_MAX_LEN"):
        results.append(ok("序列上限与档位一致",
                          local.get("max_length") == adaptive.get("max_length"),
                          f"实际 {local.get('max_length')} vs 档位 {adaptive.get('max_length')}"))

    # ---------- 2) 标定覆盖 ----------
    print("\n=== 2. 标定覆盖（当前档位是否有专属阈值）===")
    calib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "answerability_calibration.json")
    tier = str(adaptive.get("name") or "")
    if not os.path.exists(calib_path):
        results.append(ok("标定文件存在", False, calib_path))
        print("  先运行： python scripts\\calibrate_answerability.py")
    else:
        with open(calib_path, encoding="utf-8-sig") as fh:
            calib = json.load(fh)
        key = f"{args.tenant}/{args.kb}"
        entry = (calib.get("kb") or {}).get(key) or {}
        tier_entry = (entry.get("tiers") or {}).get(tier) or {}
        thr = tier_entry.get("rerank_threshold")
        print(f"  文件里有档位: {calib.get('calibrated_tiers')}")
        print(f"  {key} 在档位 {tier} 下: "
              + (f"rerank_threshold={thr}" if thr is not None else "**未标定**"))
        results.append(ok(f"{key} 在档位 {tier} 有专属阈值", thr is not None,
                          "" if thr is not None else
                          "将回落全局底线；请重跑 calibrate_answerability.py"))

    # ---------- 3) 实测 ----------
    print("\n=== 3. 实测（延迟 + 拒答）===")
    cases = CASES[:2] + CASES[2:3] if args.quick else CASES
    latencies: List[float] = []
    for query, expect in cases:
        t0 = time.perf_counter()
        try:
            resp = client.post(f"{args.url}/v1/search",
                               json={"query": query, "tenant": args.tenant,
                                     "kb": args.kb, "top_k": 10}).json()
        except Exception as exc:
            results.append(ok(f"{query[:18]}…", False, str(exc)[:80]))
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        tr = resp.get("trace") or {}
        refused = bool(tr.get("refused"))
        got = "refuse" if refused else "answer"
        passed = got == expect
        print(f"  {query[:26]:<28} 可达分={tr.get('answerability')} "
              f"阈值={tr.get('threshold_value')} 来源={tr.get('threshold_source')} "
              f"{elapsed:.2f}s")
        results.append(ok(f"{query[:18]}… 期望 {expect}",
                          passed, f"实际 {got}"))

        results.append(ok(f"{query[:18]}… 用了当前档位",
                          tr.get("rerank_tier") in (None, tier),
                          f"trace.rerank_tier={tr.get('rerank_tier')}"))

    if latencies:
        med = statistics.median(latencies)
        print(f"\n  延迟: 中位 {med:.2f}s  最大 {max(latencies):.2f}s")
        results.append(ok("中位延迟在 3s 预算内", med <= 3.0, f"{med:.2f}s"))

    # ---------- 汇总 ----------
    passed = sum(1 for r in results if r)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"结果：{passed} / {total} 通过")
    if passed != total:
        print("有不符合项。常见原因：")
        print("  · 换过显卡 → 重跑 calibrate_answerability.py（阈值按档位存）")
        print("  · 服务没重启 → 改环境变量后必须重启才生效")
        print("  · 首次查询慢 → 预热未完成，等 health 里 reranker.ready=true")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
