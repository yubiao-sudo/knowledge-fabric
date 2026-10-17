"""关键实验：cross-encoder（bge-reranker-v2-m3）能否比 embedding 余弦更好地
区分"可答"与"语义邻接的难负例"。

## 动机

embedding 余弦在这批查询上的分离间隙只有 +0.0336，最难的负例
「如何用 Rust 写一个 WebSocket 服务器」（0.5529）与本库技术架构文档语义相邻。
余弦是**双塔**（query 与 doc 各自独立编码），看不到两者的交互；
cross-encoder 是**单塔**（query+doc 拼接后一起过模型），能判断"这段文字是否真的
回答了这个问题"——理论上更适合做答案可达性判定。

## 本脚本只做测量，不改服务

对每条查询：从 Qdrant 取稠密 top-K 候选 → 用 reranker 给每个 (query, chunk) 打分
→ 输出 top-1 的 cross-encoder 分。对比两种判据的分离度。
"""
from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, r"F:\knowledge-fabric\services\retrieve")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import httpx
from kf_retrieve.backends import try_embed_client

QDRANT = "http://localhost:6333"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEVICE = os.environ.get("KF_RERANK_DEVICE", "cpu")
TOP_K = 10

CASES = {
    "answerable": [
        ("灯塔的设计目标是什么", "acme", "demo"),
        ("数据库表结构是怎么设计的", "acme", "demo"),
        ("密码长度要求是多少位", "acme", "demo"),
        ("用户表有哪些字段", "acme", "demo"),
        ("运维规范里对账号有什么要求", "acme", "demo"),
        ("MX-100 的额定功率是多少", "alpha", "kbalpha"),
        ("设备的工作温度范围", "alpha", "kbalpha"),
    ],
    "unanswerable": [
        ("今天天气怎么样，帮我订张去上海的机票", "acme", "demo"),
        ("请给出本知识库不存在的火星殖民预算表", "acme", "demo"),
        ("珠穆朗玛峰有多高", "acme", "demo"),
        ("红烧肉怎么做才好吃", "acme", "demo"),
        ("2024 年诺贝尔物理学奖得主是谁", "acme", "demo"),
        ("如何用 Rust 写一个 WebSocket 服务器", "acme", "demo"),
        ("推荐几本适合入门的机器学习书籍", "acme", "demo"),
    ],
}


def coll_name(tenant: str, kb: str) -> str:
    return f"kf_{tenant}_{kb}".replace("-", "_")


def dense_candidates(client, query: str, tenant: str, kb: str, k: int = TOP_K):
    """取稠密 top-K 的 (文本, 余弦分)。"""
    vec = client.embed([query])[0]
    resp = httpx.post(
        f"{QDRANT}/collections/{coll_name(tenant, kb)}/points/query",
        json={"query": vec, "using": "dense", "limit": k, "with_payload": True},
        timeout=120,
    )
    resp.raise_for_status()
    pts = resp.json()["result"]["points"]
    out = []
    for p in pts:
        payload = p.get("payload") or {}
        out.append((str(payload.get("content") or ""), float(p["score"])))
    return out


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"加载 reranker: {RERANK_MODEL} (device={DEVICE})")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(RERANK_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    model.to(DEVICE)
    print(f"加载耗时 {time.perf_counter()-t0:.1f}s")

    def rerank(query: str, docs: list[str], batch: int = 4) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(docs), batch):
            chunk = docs[i:i + batch]
            enc = tok([query] * len(chunk), chunk, padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
        return scores

    emb = try_embed_client()
    rows = {}
    for kind, items in CASES.items():
        rows[kind] = []
        for query, tenant, kb in items:
            try:
                cands = dense_candidates(emb, query, tenant, kb)
            except Exception as exc:
                print(f"  [skip] {query[:22]} -> {exc}")
                continue
            docs = [c[0] for c in cands]
            cos_top1 = cands[0][1] if cands else 0.0
            t0 = time.perf_counter()
            ce = rerank(query, docs)
            dt = time.perf_counter() - t0
            ce_top1 = max(ce) if ce else -99.0
            rows[kind].append({"cos": cos_top1, "ce": ce_top1, "sec": dt, "q": query})
            print(f"  [{kind[:4]}] cos={cos_top1:.4f}  ce={ce_top1:+7.3f}  {dt:5.2f}s  {query[:26]}")
    # 注意：cross-encoder 原始 logits 越大越相关；负数表示"不像能回答"

    print("\n" + "=" * 84)
    a_cos = [r["cos"] for r in rows["answerable"]]
    u_cos = [r["cos"] for r in rows["unanswerable"]]
    a_ce = [r["ce"] for r in rows["answerable"]]
    u_ce = [r["ce"] for r in rows["unanswerable"]]

    print(f"{'判据':<26}{'可答 min':>10}{'不可答 max':>12}{'间隙':>10}  结论")
    print(f"{'embedding 余弦':<26}{min(a_cos):>10.4f}{max(u_cos):>12.4f}"
          f"{min(a_cos)-max(u_cos):>+10.4f}  "
          f"{'✅' if min(a_cos) > max(u_cos) else '❌ 重叠'}")
    print(f"{'cross-encoder 原始分':<26}{min(a_ce):>10.3f}{max(u_ce):>12.3f}"
          f"{min(a_ce)-max(u_ce):>+10.3f}  "
          f"{'✅' if min(a_ce) > max(u_ce) else '❌ 重叠'}")

    # sigmoid 后的概率（bge-reranker 惯例用 sigmoid 归一化）
    def sig(x):
        return 1 / (1 + 2.718281828459045 ** (-x))

    a_s = [sig(x) for x in a_ce]
    u_s = [sig(x) for x in u_ce]
    print(f"{'cross-encoder sigmoid':<26}{min(a_s):>10.4f}{max(u_s):>12.4f}"
          f"{min(a_s)-max(u_s):>+10.4f}  "
          f"{'✅' if min(a_s) > max(u_s) else '❌ 重叠'}")
    print(f"\n建议 ce-sigmoid 阈值 ≈ {(min(a_s)+max(u_s))/2:.4f}")
    avg_sec = statistics.fmean([r["sec"] for k in rows for r in rows[k]])
    print(f"rerank 平均耗时 {avg_sec:.2f}s / 次（{TOP_K} 候选，{DEVICE}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
