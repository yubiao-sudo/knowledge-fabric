"""按知识库标定答案可达阈值：用固定"域外探针问题"测该库的分数噪声底。

## 为什么不用分布自适应

实测否定了"相对判据"的想法（见 `scripts/calibrate_answerability.py`）：

| 信号 | 可答 min | 不可答 max | 间隙 |
|---|---|---|---|
| `top1`（绝对分） | 0.5865 | 0.5529 | **+0.0336 ✅** |
| `top1 - tail_mean` | 0.0102 | 0.0819 | −0.0718 ❌ |
| `z_top1` | 1.3791 | 3.8379 | −2.4588 ❌ |

原因：越界问题也可以有"突出的"最高分（「今天天气」z=3.73），
而可答问题也可能分数平坦（「设备工作温度」Δtail=0.0102）。
**所以只有绝对相似度可用。**

## 本脚本的做法（把固定阈值变成按库标定）

`0.57` 是在**这一个语料**上手工试出来的，换语料就得重试。改为：

1. 用一组固定的**域外探针问题**（天气 / 烹饪 / 体育 / 常识等，与任何企业库都无关）
   在每个库上跑一遍，取它们的 `top1` —— 这就是该库的**噪声底**；
2. 阈值 = 噪声底的高分位（如 90%）+ 余量（`margin`）；
3. 结果写进 `configs/answerability_calibration.json`，检索服务按库读取；
   没有标定的库回退到 `assembly.answerability_floor`。

这样语料变大、模型更换、库变密集时，只需重跑本脚本，不必改代码。

用法::

    python scripts/calibrate_answerability.py                 # 标定所有库并写文件
    python scripts/calibrate_answerability.py --tenant acme --kb demo
    python scripts/calibrate_answerability.py --dry-run       # 只打印不写文件
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, r"F:\knowledge-fabric\services\retrieve")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import httpx
from kf_retrieve.backends import try_embed_client

QDRANT = os.environ.get("QDRANT_URL", "http://localhost:6333")
OUT = r"F:\knowledge-fabric\configs\answerability_calibration.json"

PG = ["docker", "exec", "kf-postgres", "psql", "-U", "kf", "-d", "knowledge_fabric",
      "-t", "-A", "-F", "\x1f", "-c"]

# 固定探针：与**企业知识库**无关的常见问题，用来估计"无关查询在本库能拿多少分"。
#
# 挑选原则（吃过亏）：探针必须**明确离域**。曾经加过「python 的列表推导式怎么写」，
# 而本库恰好含代码，它拿到 rerank 0.3923 / cos 0.5784，把噪声底抬到阈值 0.6084
# ——**直接误拒了实测可答的「灯塔的设计目标」（0.5865）**。
# 所以剔除一切"技术/编程"主题：企业库很可能就有代码与架构文档。
#
# 保留少量**难但合理**的探针（如「如何用 Rust 写 WebSocket」）：它在两种判据下都是
# 最高分的负例（cos 0.5529 / rerank 0.0993），正是最容易漏放的那类，不能为了好调而删掉。
PROBES: List[str] = [
    "今天天气怎么样，帮我订张去上海的机票",
    "红烧肉怎么做才好吃",
    "请给出本知识库不存在的火星殖民预算表",
    "珠穆朗玛峰有多高",
    "2024 年诺贝尔物理学奖得主是谁",
    "推荐几本适合入门的机器学习书籍",
    "明天股市会涨吗",
    "猫咪一天要喂几次",
    "世界杯上一届冠军是谁",
    "如何用 Rust 写一个 WebSocket 服务器",
    "帮我写一首关于秋天的七言绝句",
    "为什么天空是蓝色的",
    "北京到上海高铁要多久",
    "怎样养好一盆多肉植物",
    "如何评价莎士比亚的悲剧作品",
    "最近有什么好看的电影推荐",
]

# 噪声底取 **max**（而不是某个分位）：
# 判据要求"不比任何已知负例更宽松"，用分位会系统性地截掉最难的负例。
# 实测（acme/demo）：探针 rerank 分的 p90 只有 0.0351，而最难负例是 0.0993
# ——按 p90+0.03=0.0651 设阈值会把那条放过去；按 max+0.03=0.1293 才与
# 实测分离点（可答 min 0.1552 / 不可答 max 0.0998）吻合。
PERCENTILE = 1.0
# 余量：阈值要高于最差探针多少。0.03 在两种判据上都与实测分离点吻合：
#   cosine: max 0.5529 + 0.03 = 0.5829（实测可答 min 0.5865）
#   rerank: max 0.0993 + 0.03 = 0.1293（实测可答 min 0.1552）
DEFAULT_MARGIN = 0.03


def sql(query: str) -> List[List[str]]:
    out = subprocess.run(PG + [query], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    rows = []
    for line in (out.stdout or "").strip().splitlines():
        if line.strip():
            rows.append(line.split("\x1f"))
    return rows


def list_collections() -> List[Tuple[str, str, int]]:
    """从库表里列出有活跃块的 (tenant, kb, chunks)。"""
    rows = sql("""
        SELECT tenant_id, kb_id, count(*) AS n
          FROM chunks WHERE status='active'
      GROUP BY 1,2 HAVING count(*) > 0
      ORDER BY 3 DESC
    """)
    return [(r[0], r[1], int(r[2])) for r in rows if len(r) >= 3]


def coll_name(tenant: str, kb: str) -> str:
    return f"kf_{tenant}_{kb}".replace("-", "_")


def top1_scores(client, query: str, tenant: str, kb: str) -> float | None:
    try:
        vec = client.embed([query])[0]
        resp = httpx.post(
            f"{QDRANT}/collections/{coll_name(tenant, kb)}/points/query",
            json={"query": vec, "using": "dense", "limit": 10, "with_payload": False},
            timeout=120,
        )
        if resp.status_code >= 400:
            return None
        pts = resp.json()["result"]["points"]
        return float(pts[0]["score"]) if pts else None
    except Exception:
        return None


def rerank_top1(reranker, query: str, tenant: str, kb: str, client,
                k: int = 4) -> float | None:
    """取 cross-encoder 的 top-1 概率（与检索链路的 rerank 判据同源）。

    `k` 必须与 `configs/retrieval_profiles.yaml` 的 `rerank_top_n` 一致（现为 4）：
    候选窗口不同，top-1 分数分布就不同，标定阈值会错位。
    真机实测 K=3/4 时延迟约 1.1~1.5s（maxlen=256），而 K=12 要 9.4s。
    """
    try:
        vec = client.embed([query])[0]
        resp = httpx.post(
            f"{QDRANT}/collections/{coll_name(tenant, kb)}/points/query",
            json={"query": vec, "using": "dense", "limit": k, "with_payload": True},
            timeout=120,
        )
        if resp.status_code >= 400:
            return None
        pts = resp.json()["result"]["points"]
        docs = [str((p.get("payload") or {}).get("content") or "") for p in pts]
        if not docs:
            return None
        scores = reranker.score_pairs(query, docs)
        return max(scores) if scores else None
    except Exception:
        return None


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="")
    ap.add_argument("--kb", default="")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-rerank", action="store_true",
                    help="只标定 cosine，不加载 cross-encoder（省时间）")
    args = ap.parse_args()

    client = try_embed_client()
    print(f"embed device = {client.describe().get('device')}")

    reranker = None
    if not args.no_rerank:
        from kf_retrieve.rerank import try_local_reranker

        reranker = try_local_reranker()
        if reranker is not None:
            d = reranker.describe()
            print(f"rerank device = {d.get('device')}  dtype={d.get('dtype')}")
        else:
            print("rerank 不可用，只标定 cosine 判据")
    print()

    targets = list_collections()
    if args.tenant:
        targets = [t for t in targets if t[0] == args.tenant]
    if args.kb:
        targets = [t for t in targets if t[1] == args.kb]

    result: Dict[str, Any] = {
        "version": 2,
        "probe_count": len(PROBES),
        "percentile": PERCENTILE,
        "margin": args.margin,
        "kb": {},
    }

    print(f"{'库':<22}{'块':>6}{'cos噪声底':>11}{'cos阈值':>9}"
          f"{'rr噪声底':>10}{'rr阈值':>8}  分布(cos)")
    print("-" * 104)
    for tenant, kb, chunks in targets:
        scores = [s for s in (top1_scores(client, p, tenant, kb) for p in PROBES)
                  if s is not None]
        if not scores:
            print(f"{tenant + '/' + kb:<22}{chunks:>6}{'N/A':>11}{'-':>9}"
                  f"{'-':>10}{'-':>8}  集合不可用")
            continue
        floor = percentile(scores, PERCENTILE)
        threshold = round(floor + args.margin, 4)

        rr_scores = []
        if reranker is not None:
            rr_scores = [s for s in (rerank_top1(reranker, p, tenant, kb, client)
                                     for p in PROBES) if s is not None]
        rr_floor = percentile(rr_scores, PERCENTILE) if rr_scores else None
        rr_threshold = round(rr_floor + args.margin, 4) if rr_floor is not None else None

        key = f"{tenant}/{kb}"
        entry: Dict[str, Any] = {
            "noise_floor_p90": round(floor, 4),
            "threshold": threshold,
            "probes": len(scores),
            "chunks": chunks,
            "probe_scores": [round(s, 4) for s in sorted(scores)],
        }
        if rr_threshold is not None:
            entry["rerank_noise_floor_p90"] = round(rr_floor, 4)
            entry["rerank_threshold"] = rr_threshold
            entry["rerank_probe_scores"] = [round(s, 4) for s in sorted(rr_scores)]
        result["kb"][key] = entry

        rng = f"{min(scores):.4f}~{max(scores):.4f}"
        rr_cols = (f"{rr_floor:>10.4f}{rr_threshold:>8.4f}" if rr_floor is not None
                   else f"{'-':>10}{'-':>8}")
        print(f"{key:<22}{chunks:>6}{floor:>11.4f}{threshold:>9.4f}{rr_cols}  {rng}")

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")
    print("检索服务会按库读取该阈值；未标定的库回退到 assembly.answerability_floor。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
