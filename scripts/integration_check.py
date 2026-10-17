"""联调验证：决策服务 + 网关 + MCP，全部走真实 HTTP 且保证 UTF-8 无损。

用 httpx 发送，避免 PowerShell 传中文时的编码损耗。
"""

import json
import sys

import httpx

DECISION = "http://127.0.0.1:8090"
GATEWAY = "http://127.0.0.1:8099"


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main() -> int:
    results = []

    # ---- 1) 决策服务直连：多语言 + 各种意图 ----
    probes = [
        ("MX-200 的价格是多少", "exact_lookup"),
        ("对比 MX-100 与 MX-200 的差异并说明各自价格", "decompose_agentic"),
        ("产品有哪些型号", "decompose_agentic"),
        ("What is the price of MX-200?", "exact_lookup"),
        ("製品のインストール手順を教えてください", "hybrid_rerank"),
        ("제품 설치 방법을 알려주세요", "hybrid_rerank"),
        ("?", "hyde"),
    ]
    print("== 决策服务 /decide ==")
    for query, expected in probes:
        body = {"query_id": "q1", "query": query, "tenant_id": "acme", "kb_id": "demo",
                "features": {}}
        r = httpx.post(f"{DECISION}/decide", json=body, timeout=10)
        data = r.json()
        got = data["profile"]
        results.append(check(
            f"decide '{query[:30]}'",
            got == expected and r.status_code == 200,
            f"-> {got} (期望 {expected}) lang={data['features'].get('lang')}",
        ))

    # ---- 2) 网关 /health：决策服务应可达 ----
    print("== 网关 /health ==")
    r = httpx.get(f"{GATEWAY}/health", timeout=10)
    health = r.json()
    results.append(check("health.status", health.get("status") == "ok", ""))
    results.append(check("decision_service.reachable",
                         health.get("decision_service", {}).get("reachable") is True,
                         f"latency={health.get('decision_service', {}).get('latency_ms')}ms"))

    # ---- 3) 网关 /v1/retrieve：档位来自真实决策，不再是 safe default ----
    print("== 网关 /v1/retrieve ==")
    for query, expected in probes[:4]:
        r = httpx.post(f"{GATEWAY}/v1/retrieve", timeout=15,
                       json={"query": query, "tenant": "acme", "kb": "demo"})
        data = r.json()
        decision = data.get("decision", {})
        results.append(check(
            f"retrieve '{query[:30]}'",
            decision.get("profile") == expected,
            f"-> profile={decision.get('profile')} (期望 {expected}) "
            f"fallback_used={decision.get('fallback_used')} "
            f"model={decision.get('model', {}).get('id')} "
            f"mode={data.get('mode')}",
        ))

    # ---- 4) MCP：tools/list 与 tools/call 真实路由 ----
    print("== MCP ==")
    r = httpx.post(f"{GATEWAY}/mcp", timeout=15,
                   json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = [t["name"] for t in r.json().get("result", {}).get("tools", [])]
    results.append(check("tools/list", {"retrieve", "ingest"} <= set(tools), f"-> {tools}"))

    r = httpx.post(f"{GATEWAY}/mcp", timeout=15, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "retrieve",
                   "arguments": {"query": "MX-200 的价格是多少", "tenant": "acme", "kb": "demo"}},
    })
    payload = r.json()
    inner = json.loads(payload["result"]["content"][0]["text"])
    results.append(check(
        "tools/call retrieve 真实路由",
        payload.get("result", {}).get("isError") is False
        and inner.get("decision", {}).get("profile") == "exact_lookup",
        f"-> profile={inner.get('decision', {}).get('profile')} "
        f"grounding.refused={inner.get('grounding', {}).get('refused')}",
    ))

    r = httpx.post(f"{GATEWAY}/mcp", timeout=15,
                   json={"jsonrpc": "2.0", "id": 3, "method": "no/such"})
    results.append(check("未知 method -> -32601",
                         r.json().get("error", {}).get("code") == -32601, ""))

    passed, total = sum(1 for x in results if x), len(results)
    print(f"\n== 汇总：{passed}/{total} 通过 ==")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
