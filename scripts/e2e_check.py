"""全链路联调：网关 → 决策(Laya) → 检索 → 证据。

验证的三层是否能真正串起来（而不是各自单测通过、链路却是断的）：
    用户查询 → 网关 /v1/retrieve 或 MCP tools/call
             → 决策服务选档位（Laya）
             → 检索服务混合召回 + 融合 + 重排 + 组装
             → 返回带引用的 evidence，供 Agent 的 LLM 组织答案

依赖：decision(8090) / retrieve(8077, 灌入 KF_RETRIEVE_SEED_JSON) / gateway(8099) 已启动。
"""

import json
import sys

import httpx

DECISION = "http://127.0.0.1:8090"
RETRIEVE = "http://127.0.0.1:8077"
GATEWAY = "http://127.0.0.1:8099"

_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    _results.append(bool(ok))
    return ok


def main() -> int:
    # ---------- 1) 三个服务的健康检查 ----------
    print("== 服务健康 ==")
    for label, url in (("decision", DECISION), ("retrieve", RETRIEVE), ("gateway", GATEWAY)):
        try:
            body = httpx.get(f"{url}/health", timeout=10).json()
            check(f"{label} /health", body.get("status") == "ok", json.dumps(body, ensure_ascii=False)[:110])
        except Exception as exc:
            check(f"{label} /health", False, f"不可达: {exc}")

    # ---------- 2) 检索服务直连：证据 + 引用齐全 ----------
    print("\n== 检索服务 /v1/search ==")
    resp = httpx.post(f"{RETRIEVE}/v1/search", timeout=20, json={
        "profile": "hybrid_rerank", "query": "MX-200 的价格是多少",
        "tenant": "acme", "kb": "demo", "top_k": 50,
    }).json()
    ids = [e["chunk_id"] for e in resp["evidence"]]
    check("命中价格相关块", "c1" in ids, f"evidence={ids[:5]}")
    check("父块回填生效",
          any(e.get("parent_chunk_id") == "p1" and e.get("content", "").startswith("【父块】")
              for e in resp["evidence"]),
          "")
    check("引用来自 provenance",
          bool(resp["citations"]) and isinstance(resp["citations"][0].get("char_range"), list),
          f"char_range={resp['citations'][0].get('char_range') if resp['citations'] else None}")
    check("kb 过滤生效", "acme_other" not in ids, "跨库块未被召回")

    # ---------- 3) 语言路由：中文稀疏查询不带出英文块 ----------
    print("\n== 多语言路由 ==")
    zh_ids = [e["chunk_id"] for e in resp["evidence"]]
    check("中文查询未召回纯英文块", "en1" not in zh_ids, f"evidence={zh_ids[:6]}")
    print(f"        （说明：稠密通道因未配置 TEI 而缺席，本轮由稀疏通道按 lang=zh 过滤）")

    # ---------- 4) 网关全链路：决策 + 检索 ----------
    print("\n== 网关 /v1/retrieve（决策→检索）==")
    for query, expect_profile in (
        ("MX-200 的价格是多少", "exact_lookup"),
        ("对比 MX-100 与 MX-200 的价格差异", "decompose_agentic"),
    ):
        body = httpx.post(f"{GATEWAY}/v1/retrieve", timeout=25, json={
            "query": query, "tenant": "acme", "kb": "demo"}).json()
        decision = body.get("decision", {})
        got = decision.get("profile")
        check(f"档位 {expect_profile}", got == expect_profile,
              f"-> {got} escalate={decision.get('escalate')} "
              f"fallback={decision.get('fallback_used')}")
        check("evidence 非空", bool(body.get("evidence")),
              f"count={len(body.get('evidence') or [])} mode={body.get('mode')}")
        check("grounding 未误拒答",
              body.get("grounding", {}).get("refused") is False,
              f"grounding={body.get('grounding')}")

    # ---------- 5) MCP 全链路 ----------
    print("\n== MCP tools/call ==")
    mcp = httpx.post(f"{GATEWAY}/mcp", timeout=25, json={
        "jsonrpc": "2.0", "id": 11, "method": "tools/call",
        "params": {"name": "retrieve", "arguments": {
            "query": "MX-200 的价格是多少", "tenant": "acme", "kb": "demo"}},
    }).json()
    inner = json.loads(mcp["result"]["content"][0]["text"])
    check("MCP 返回真实证据而非空兜底",
          bool(inner.get("evidence")),
          f"count={len(inner.get('evidence') or [])} "
          f"profile={inner.get('decision', {}).get('profile')}")
    check("MCP 与 REST 同源",
          [e["chunk_id"] for e in inner.get("evidence", [])][:1]
          == [e["chunk_id"] for e in resp["evidence"]][:1],
          "")

    print("\n== MCP tools/list ==")
    tools = [t["name"] for t in httpx.post(f"{GATEWAY}/mcp", timeout=15, json={
        "jsonrpc": "2.0", "id": 12, "method": "tools/list"}).json()["result"]["tools"]]
    check("暴露 retrieve/ingest", {"retrieve", "ingest"} <= set(tools), f"{tools}")

    passed, total = sum(_results), len(_results)
    print(f"\n== 全链路汇总：{passed}/{total} 通过 ==")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
