"""按 db/reingest_plan.json 重新摄入文档：补 dense 向量、修溯源偏移。

背景：`acme/demo` 等库是**本地嵌入修复之前**摄入的，575 个块里 502 个没有 dense
向量——它们在稠密检索里"隐形"，只能靠稀疏命中；同时溯源退化成整页区间
（"点击回原文"因此失准）。重摄入是唯一的修法。

特性：
  * **幂等**：同一 (tenant, kb, doc_id, version) 覆盖写入，Qdrant 用确定性 point id。
  * **只重摄入需要修的**：默认读 reingest_plan.json 的 `todo`。
  * **逐篇汇报**：成功/失败/耗时/父块数/dense 数/溯源精确定位率，全部来自服务端
    返回的真实统计，不臆测。
  * 源文件缺失的文档**明确列出且不尝试**，不假装修好了。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import httpx

PLAN = r"F:\knowledge-fabric\db\reingest_plan.json"
INGEST = os.environ.get("KF_INGEST_URL", "http://127.0.0.1:8083")


def reingest_one(item: Dict[str, Any]) -> Dict[str, Any]:
    uri = item["uri"]
    name = os.path.basename(uri)
    data = {
        "tenant": item["tenant"],
        "kb": item["kb"],
        "doc_id": item["doc_id"],
        "lang": "zh",
    }
    t0 = time.perf_counter()
    try:
        with open(uri, "rb") as fh:
            resp = httpx.post(f"{INGEST}/v1/ingest",
                              files={"file": (name, fh)},
                              data=data, timeout=3600)
        elapsed = time.perf_counter() - t0
        body = resp.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elapsed": time.perf_counter() - t0,
                **{k: item[k] for k in ("tenant", "kb", "doc_id")}}
    stages = body.get("stages") or {}
    chunk = stages.get("chunk") or {}
    embed = stages.get("embed") or {}
    return {
        "ok": bool(body.get("ok")) and not body.get("errors"),
        "errors": body.get("errors") or [],
        "elapsed": elapsed,
        "parents": (body.get("counts") or {}).get("parents"),
        "children": (body.get("counts") or {}).get("children"),
        "dense": embed.get("dense"),
        "provenance": (chunk.get("provenance") or {}).get("exact_ratio"),
        "strategy": body.get("strategy"),
        "tenant": item["tenant"], "kb": item["kb"], "doc_id": item["doc_id"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（试跑用）")
    ap.add_argument("--only-kb", default="", help="只处理某个库，如 acme/demo")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="总时间上限（秒）；到点后停止并如实汇报已完成的篇数")
    args = ap.parse_args()

    # 用 utf-8-sig 读取：Windows 侧工具（PowerShell Set-Content 等）写出的 JSON
    # 常带 UTF-8 BOM，纯 utf-8 读会直接抛 JSONDecodeError。
    with open(args.plan, encoding="utf-8-sig") as fh:
        plan = json.load(fh)
    todo: List[Dict[str, Any]] = plan.get("todo") or []
    missing: List[Dict[str, Any]] = plan.get("missing") or []

    if args.only_kb:
        tenant, _, kb = args.only_kb.partition("/")
        todo = [t for t in todo if t["tenant"] == tenant and t["kb"] == kb]
    if args.limit:
        todo = todo[:args.limit]

    print(f"待重摄入 {len(todo)} 篇；源文件缺失无法重建 {len(missing)} 篇", flush=True)
    if missing:
        for m in missing:
            print(f"  [跳过·源缺失] {m['tenant']}/{m['kb']} :: {m['doc_id'][:40]}", flush=True)
    print("-" * 96, flush=True)

    results = []
    started = time.perf_counter()
    remaining = list(todo)
    for i, item in enumerate(todo, 1):
        elapsed_total = time.perf_counter() - started
        if args.max_seconds and elapsed_total > args.max_seconds:
            print(f"到达时间上限 {args.max_seconds:.0f}s，停止；"
                  f"剩余 {len(remaining)} 篇未处理", flush=True)
            break
        remaining.pop(0)
        print(f"[{i}/{len(todo)}] {item['tenant']}/{item['kb']} :: {item['doc_id'][:46]}",
              flush=True)
        r = reingest_one(item)
        results.append(r)
        if r.get("ok"):
            prov = r.get("provenance")
            print(f"        ok  用时 {r['elapsed']:.1f}s  父块={r.get('parents')} "
                  f"子块={r.get('children')} dense={r.get('dense')} "
                  f"溯源精确率={'N/A' if prov is None else format(prov, '.0%')}", flush=True)
        else:
            print(f"        FAIL 用时 {r['elapsed']:.1f}s  "
                  f"errors={r.get('errors') or r.get('error')}", flush=True)
        # 逐篇落盘：进程被中断也能看到已完成的部分
        out = r"F:\knowledge-fabric\db\reingest_result.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "remaining": remaining,
                       "ok": sum(1 for x in results if x.get("ok")),
                       "failed": sum(1 for x in results if not x.get("ok")),
                       "skipped_missing": missing}, fh, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r.get("ok"))
    print("-" * 96, flush=True)
    print(f"本次完成：{ok} 成功 / {len(results) - ok} 失败；"
          f"总耗时 {time.perf_counter()-started:.0f}s", flush=True)
    out = r"F:\knowledge-fabric\db\reingest_result.json"
    print(f"明细已写入 {out}（逐篇落盘，中断也可查）", flush=True)
    return 0 if ok == len(results) and not remaining else 1


if __name__ == "__main__":
    sys.exit(main())
