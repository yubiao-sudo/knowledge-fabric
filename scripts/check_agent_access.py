"""验证 Agent 接入链路：接入码 → 令牌 → 带令牌调用 → 租户隔离 → 吊销。

这是"底座能不能被 agent 接进去"的验收脚本。全程只用 HTTP，
不连数据库、不读源码内部状态——模拟真实客户端。

    python scripts\\check_agent_access.py
    python scripts\\check_agent_access.py --base http://127.0.0.1:8082 --tenant acme

退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import httpx


def ok(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8082")
    ap.add_argument("--tenant", default="acme")
    ap.add_argument("--kb", default="demo")
    args = ap.parse_args()

    results: List[bool] = []
    c = httpx.Client(base_url=args.base, timeout=300.0)

    # ---------- 0) 服务与鉴权状态 ----------
    print("\n=== 0. 服务状态 ===")
    try:
        health = c.get("/health").json()
    except Exception as exc:
        ok("检索服务可达", False, f"{args.base} 连不上：{exc}")
        print("\n先启动服务：start.cmd")
        return 1
    results.append(ok("检索服务可达", True, args.base))
    st = health.get("auth") or {}
    print(f"  auth.mode={st.get('mode')} required={st.get('required')} "
          f"has_key={st.get('has_key')}")

    # ---------- 1) 列知识库（远程发现，不连数据库） ----------
    print("\n=== 1. 列知识库 GET /v1/kbs ===")
    try:
        r = c.get("/v1/kbs")
        results.append(ok("接口可用", r.status_code in (200, 401),
                          f"HTTP {r.status_code}"))
        if r.status_code == 200:
            data = r.json()
            n = data.get("count", 0)
            results.append(ok("返回了知识库", n > 0, f"{n} 个"))
            if n:
                first = data["items"][0]
                print(f"  示例: {first['tenant']}/{first['kb']} "
                      f"docs={first['documents']} chunks={first['chunks']} "
                      f"dense_ratio={first['dense_ratio']}")
        else:
            print(f"  已受保护（需令牌）: {r.json().get('detail', '')[:80]}")
    except Exception as exc:
        results.append(ok("接口可用", False, str(exc)[:80]))

    # ---------- 2) 生成接入码（仅本机） ----------
    print("\n=== 2. 生成接入码 POST /v1/auth/pairing ===")
    code: Optional[str] = None
    try:
        r = c.post("/v1/auth/pairing", json={"tenant": args.tenant, "name": "check-agent"})
        if r.status_code == 200:
            info = r.json()
            code = info.get("code")
            results.append(ok("拿到接入码", bool(code), str(code)))
            print(f"  有效期: {info.get('expires_at')}（{info.get('ttl_sec')} 秒）")
        elif r.status_code == 403:
            results.append(ok("签发需本机", True, "远端被拒（预期行为）"))
            print("  注意：本脚本不在本机执行时无法签发，将跳过后续签发流程")
        else:
            results.append(ok("拿到接入码", False, f"HTTP {r.status_code} {r.text[:120]}"))
    except Exception as exc:
        results.append(ok("拿到接入码", False, str(exc)[:80]))

    # ---------- 3) 兑换令牌 ----------
    print("\n=== 3. 兑换令牌 POST /v1/auth/exchange ===")
    token: Optional[str] = None
    if code:
        try:
            r = c.post("/v1/auth/exchange", json={"code": code, "name": "check-agent"})
            if r.status_code == 200:
                info = r.json()
                token = info.get("token")
                results.append(ok("拿到长期令牌", bool(token and token.startswith("kf_")),
                                  f"{token[:8]}…" if token else ""))
                results.append(ok("令牌带租户", info.get("tenant_id") == args.tenant,
                                  str(info.get("tenant_id"))))
            else:
                results.append(ok("拿到长期令牌", False, f"HTTP {r.status_code} {r.text[:120]}"))
        except Exception as exc:
            results.append(ok("拿到长期令牌", False, str(exc)[:80]))

        # ---------- 4) 接入码只能用一次 ----------
        print("\n=== 4. 接入码一次性 ===")
        try:
            r2 = c.post("/v1/auth/exchange", json={"code": code})
            results.append(ok("重复兑换被拒", r2.status_code == 401,
                              f"HTTP {r2.status_code}"))
        except Exception as exc:
            results.append(ok("重复兑换被拒", False, str(exc)[:80]))
    else:
        print("  （跳过：没有接入码）")

    # ---------- 5) 签发后自动进入受保护状态 ----------
    print("\n=== 5. 鉴权生效 ===")
    if token:
        r_no = c.post("/v1/search", json={"query": "密码长度要求是多少位",
                                          "tenant": args.tenant, "kb": args.kb})
        results.append(ok("无令牌被拒 401", r_no.status_code == 401,
                          f"HTTP {r_no.status_code}"))
        r_yes = c.post("/v1/search",
                       json={"query": "密码长度要求是多少位",
                             "tenant": args.tenant, "kb": args.kb, "top_k": 5},
                       headers={"Authorization": f"Bearer {token}"})
        results.append(ok("带令牌通过", r_yes.status_code == 200,
                          f"HTTP {r_yes.status_code}"))
        if r_yes.status_code == 200:
            body = r_yes.json()
            ev = body.get("evidence") or []
            tr = body.get("trace") or {}
            results.append(ok("返回证据", len(ev) > 0, f"{len(ev)} 条"))
            results.append(ok("带可核实引用区间",
                              all(e.get("char_range") for e in ev) if ev else False,
                              "每条 evidence 都有 char_range"))
            print(f"  refused={tr.get('refused')} answerability={tr.get('answerability')}")

        # ---------- 6) 租户隔离 ----------
        print("\n=== 6. 租户隔离 ===")
        r_cross = c.post("/v1/search",
                         json={"query": "任意", "tenant": "demo", "kb": "light"},
                         headers={"Authorization": f"Bearer {token}"})
        results.append(ok("跨租户被拒 403", r_cross.status_code == 403,
                          f"HTTP {r_cross.status_code}"))
        r_kb = c.get(f"/v1/kb/demo/light",
                     headers={"Authorization": f"Bearer {token}"})
        results.append(ok("跨租户读库被拒 403", r_kb.status_code == 403,
                          f"HTTP {r_kb.status_code}"))
        r_own = c.get("/v1/kbs", headers={"Authorization": f"Bearer {token}"})
        if r_own.status_code == 200:
            tenants = {i["tenant"] for i in r_own.json().get("items") or []}
            results.append(ok("列库只返回本租户", tenants <= {args.tenant},
                              str(sorted(tenants))))

        # ---------- 7) 吊销 ----------
        print("\n=== 7. 吊销令牌 ===")
        keys = c.get("/v1/auth/keys")
        if keys.status_code == 200:
            keylist = keys.json().get("keys") or []
            target = next((k for k in keylist if not k["revoked"]), None)
            results.append(ok("能列出令牌", target is not None,
                              f"{len(keylist)} 个"))
            if target:
                rv = c.post("/v1/auth/revoke", json={"id": target["id"]})
                results.append(ok("吊销成功", rv.status_code == 200,
                                  str(rv.json().get("revoked"))))
                r_after = c.post("/v1/search",
                                 json={"query": "密码长度", "tenant": args.tenant,
                                       "kb": args.kb},
                                 headers={"Authorization": f"Bearer {token}"})
                results.append(ok("吊销后令牌失效 401", r_after.status_code == 401,
                                  f"HTTP {r_after.status_code}"))
        else:
            results.append(ok("能列出令牌", False, f"HTTP {keys.status_code}"))
    else:
        print("  （跳过：没有令牌）")

    passed = sum(1 for r in results if r)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"结果：{passed} / {total} 通过")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
