"""真机端到端校验：所有关键修复项的断言（可重复执行）。

依赖运行中的服务与基础设施：
    * Postgres（15432）与 Qdrant（6333）
    * 摄入服务 8083、检索服务 8082（决策/网关可选，未起时对应项标 SKIP）

用法::

    python scripts/verify_real.py                 # 全部（网关未起自动 SKIP）
    python scripts/verify_real.py --no-ingest     # 跳过重新摄入，只做断言

退出码：0 全部通过 / 1 有失败。

设计原则与项目一致：**不许假装跑了**。服务未起就标 SKIP 并说明原因，
不把"没测"算成"通过"。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import httpx

INGEST = "http://127.0.0.1:8083"
RETRIEVE = "http://127.0.0.1:8082"
GATEWAY = "http://127.0.0.1:8000"
QDRANT = "http://localhost:6333"
DEFAULT_PDF = r"C:\Users\OMEN\AppData\Local\Temp\kf_ab.pdf"

PG = ["docker", "exec", "kf-postgres", "psql", "-U", "kf", "-d", "knowledge_fabric", "-t", "-A", "-c"]

# 本次运行临时兑换的令牌（结束后吊销，避免在库里留垃圾）
_TOKENS_TO_REVOK: list = []

TENANT = "acme"
KB = "final"
DOC_ID = "kf_ab_final"

_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    """ok=None 表示 SKIP（依赖未就绪），不计入失败。"""
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    _results.append((tag, name, detail))
    line = f"  [{tag}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def sql(query: str) -> str:
    try:
        out = subprocess.run(PG + [query], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=60)
        return (out.stdout or "").strip()
    except Exception as exc:  # docker 不可用等
        return f"__ERR__ {exc}"


def reachable(url: str, timeout: float = 5.0) -> bool:
    try:
        httpx.get(url, timeout=timeout)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ 各项校验
def verify_ingest(pdf: str) -> dict:
    print("\n=== A. 摄入真实多章节文档 ===")
    if not reachable(f"{INGEST}/health"):
        check("摄入服务可达（跳过摄入与摄入相关断言）", None, f"{INGEST} 未启动")
        return {}
    try:
        with open(pdf, "rb") as fh:
            resp = httpx.post(
                f"{INGEST}/v1/ingest",
                files={"file": (pdf.rsplit("\\", 1)[-1], fh, "application/pdf")},
                data={"tenant": TENANT, "kb": KB, "doc_id": DOC_ID,
                      "lang": "zh", "strategy": "doc_structured"},
                timeout=900,
            )
        data = resp.json()
    except Exception as exc:
        check("摄入请求成功", False, str(exc))
        return {}

    stages = data.get("stages", {})
    chunk = stages.get("chunk", {})
    embed = stages.get("embed", {})
    prov = chunk.get("provenance", {})

    check("摄入成功且无错误", bool(data.get("ok")) and not data.get("errors"),
          str(data.get("errors")))
    # provenance：关键是**不退化整页**。真实多章节 PDF 上 approx 必须为 0。
    check("provenance 无整页退化（approx == 0）", prov.get("approx") == 0,
          f"approx={prov.get('approx')} exact={prov.get('exact')}")
    check("provenance 定位率 >= 90%", (prov.get("exact_ratio") or 0) >= 0.9,
          f"exact_ratio={prov.get('exact_ratio')}")
    check("dense 向量全部写入（本地回退生效）",
          embed.get("dense", 0) > 0 and embed.get("dense") == embed.get("chunks"),
          f"dense={embed.get('dense')}/{embed.get('chunks')} backend={embed.get('dense_backend')}")
    check("父块数 >= 3（章节边界生效，非 token 预算硬切）",
          chunk.get("parents", 0) >= 3, f"parents={chunk.get('parents')}")
    return {"stages": stages}


def verify_qdrant(stages: dict) -> None:
    print("\n=== B. Qdrant 侧向量 ===")
    coll = f"kf_{TENANT}_{KB}"
    try:
        info = httpx.get(f"{QDRANT}/collections/{coll}", timeout=20).json()["result"]
        pts = httpx.post(f"{QDRANT}/collections/{coll}/points/scroll",
                         json={"limit": 5, "with_vector": True}, timeout=20).json()["result"]["points"]
    except Exception as exc:
        check("Qdrant 集合可读", False, str(exc))
        return
    names = sorted({k for p in pts for k in (p.get("vector") or {})})
    check("集合同时含 dense 与 sparse 向量", names == ["dense", "sparse"], str(names))
    expect = (stages.get("embed") or {}).get("chunks")
    if expect:
        check("点数与块数一致", info["points_count"] == expect,
              f"{info['points_count']} vs {expect}")


def agent_token() -> str:
    """拿一个可用的接入令牌：优先环境变量，否则当场用接入码兑换。

    底座一旦签发过令牌就会强制鉴权（`KF_REQUIRE_AUTH=auto`），
    此时不带令牌的 `/v1/search` 一律 401。验收脚本不能因此"挂掉"，
    但也不该让人手工准备令牌——本机跑就现场兑换一个，用完即吊销。
    """
    tok = (os.environ.get("KF_AGENT_TOKEN") or "").strip()
    if tok:
        return tok
    try:
        pair = httpx.post(f"{RETRIEVE}/v1/auth/pairing",
                          json={"tenant": TENANT, "name": "verify_real"},
                          timeout=20).json()
        code = pair.get("code")
        if not code:
            return ""
        ex = httpx.post(f"{RETRIEVE}/v1/auth/exchange",
                        json={"code": code, "name": "verify_real"},
                        timeout=20).json()
        tok = ex.get("token") or ""
        if tok:
            _TOKENS_TO_REVOK.append(ex.get("key_id") or "")
        return tok
    except Exception:
        return ""


def _auth_headers() -> dict:
    tok = agent_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def verify_retrieval() -> None:
    print("\n=== C. 检索精准度 ===")
    if not reachable(f"{RETRIEVE}/health"):
        check("检索服务可达", None, f"{RETRIEVE} 未启动")
        return
    # 鉴权状态要如实报出来，否则 401 会被误读成"检索坏了"
    try:
        st = httpx.get(f"{RETRIEVE}/health", timeout=20).json().get("auth") or {}
        check("鉴权状态可见", True,
              f"mode={st.get('mode')} required={st.get('required')}")
    except Exception:
        pass
    headers = _auth_headers()
    # 查询必须**确实能从该库回答**：acme/final 是 kf_ab.pdf（运维规范：密码/账号/信息安全）。
    # 早期这里用的是「保修期是多久」——那份文档里根本没有保修内容，在加入答案可达性
    # 闸门后会被正确拒答（0 证据），于是断言失败。断言应当跟着契约走，不是反过来。
    query = "密码长度要求是多少位"
    for profile in ("hybrid_rerank", "naive"):
        try:
            resp = httpx.post(f"{RETRIEVE}/v1/search",
                              json={"query": query, "tenant": TENANT, "kb": KB,
                                    "profile": profile, "top_k": 10},
                              headers=headers, timeout=180)
            if resp.status_code == 401:
                check(f"[{profile}] 检索请求成功", False,
                      "HTTP 401：未拿到接入令牌（设 KF_AGENT_TOKEN 或确认能在本机签发）")
                continue
            data = resp.json()
        except Exception as exc:
            check(f"[{profile}] 检索请求成功", False, str(exc))
            continue
        ev = data.get("evidence", [])
        ranges = [tuple(e.get("char_range") or ()) for e in ev]
        inverted = [r for r in ranges if len(r) == 2 and r[1] < r[0]]
        check(f"[{profile}] 有证据返回", len(ev) > 0, f"ev={len(ev)}")
        check(f"[{profile}] 无倒置字符区间", not inverted, str(inverted))
        check(f"[{profile}] 区间不全相同（非整篇召回）", len(set(ranges)) > 1 if ranges else False,
              str(ranges[:4]))
        check(f"[{profile}] content 与 char_range 同源",
              all(e.get("offset_chunk_id") == e.get("chunk_id")
                  for e in ev if e.get("char_range")))


def verify_isolation() -> None:
    print("\n=== D. 跨租户隔离（旧代码会把第二个租户静默并到第一个）===")
    probe = "isolation_probe_v2"
    docs = sql(f"SELECT tenant_id FROM documents WHERE doc_id='{probe}' ORDER BY tenant_id;")
    if docs.startswith("__ERR__"):
        check("Postgres 可查询", False, docs)
        return
    check("同一 doc_id 可归属多个租户", docs.replace("\n", ",") == "alpha,beta",
          docs.replace("\n", ",") or "(无数据)")

    print("\n=== E. RLS 覆盖 ===")
    uncovered = sql(
        "SELECT string_agg(c.relname, ',') FROM pg_class c "
        "JOIN pg_namespace n ON n.oid=c.relnamespace AND n.nspname='public' "
        "JOIN pg_attribute a ON a.attrelid=c.oid AND a.attname='tenant_id' AND a.attnum>0 "
        "WHERE c.relkind='r' "
        "AND NOT EXISTS (SELECT 1 FROM pg_policies p WHERE p.tablename=c.relname) "
        # 例外说明：tenants/workspaces 是全局注册表；api_keys/pairing_codes 是
        # **跨租户凭证表**（db/005 里刻意不启用 RLS，鉴权代码按 tenant_id 显式过滤）。
        "AND c.relname NOT IN ('tenants','workspaces','api_keys','pairing_codes');")
    check("除全局注册表外，带 tenant_id 的表都有 RLS 策略", not uncovered, uncovered or "无遗漏")
    check("documents 已开启 RLS",
          sql("SELECT relrowsecurity FROM pg_class WHERE relname='documents';") == "t")

    print("\n=== F. 主键租户维度 ===")
    tables = ["knowledge_bases", "documents", "document_versions", "chunks",
              "chunk_provenance", "chunk_vectors", "chunk_acl", "structured_records",
              "graph_entities", "graph_edges", "graph_entity_chunks"]
    quoted = ", ".join(f"'{t}'" for t in tables)
    bad = sql(
        "SELECT string_agg(conrelid::regclass::text, ',') FROM pg_constraint "
        "WHERE contype='p' AND connamespace='public'::regnamespace "
        f"AND conrelid::regclass::text IN ({quoted}) "
        "AND pg_get_constraintdef(oid) NOT LIKE '%tenant_id%';")
    check(f"{len(tables)} 张核心表主键都含 tenant_id", not bad, bad or "全部合规")

    print("\n=== G. 决策 trace 落库 ===")
    if not reachable(f"{GATEWAY}/health"):
        check("网关可达（跳过 trace 落库校验）", None, f"{GATEWAY} 未启动")
        return
    before = sql("SELECT count(*) FROM decision_trace;")
    try:
        httpx.post(f"{GATEWAY}/v1/retrieve",
                   json={"query": "产品有哪些型号", "tenant": TENANT, "kb": KB}, timeout=300)
        time.sleep(3)
    except Exception as exc:
        check("经网关检索成功", False, str(exc))
        return
    after = sql("SELECT count(*) FROM decision_trace;")
    try:
        grew = int(after) > int(before)
    except ValueError:
        grew = False
    check("decision_trace 行数增长（硬约束「落 trace」已落地）", grew, f"{before} -> {after}")
    check("网关 /health 报告 decision_trace 已启用",
          bool(httpx.get(f"{GATEWAY}/health", timeout=10).json().get("decision_trace")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="用于摄入的真实 PDF")
    parser.add_argument("--no-ingest", action="store_true", help="跳过重新摄入")
    args = parser.parse_args()

    stages: dict = {}
    if args.no_ingest:
        print("\n=== A. 摄入（已按 --no-ingest 跳过）===")
        check("摄入", None, "用户要求跳过")
    else:
        stages = verify_ingest(args.pdf)
    if stages:
        verify_qdrant(stages)
    verify_retrieval()
    verify_isolation()

    passed = sum(1 for tag, _, _ in _results if tag == "PASS")
    failed = sum(1 for tag, _, _ in _results if tag == "FAIL")
    skipped = sum(1 for tag, _, _ in _results if tag == "SKIP")
    print("\n" + "=" * 62)
    print(f"总计: {passed} PASS / {failed} FAIL / {skipped} SKIP")
    for tag, name, detail in _results:
        if tag == "FAIL":
            print(f"  FAILED: {name} — {detail}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    code = main()
    # 清理：吊销本次为验收临时兑换的令牌（失败不影响结论）
    for key_id in _TOKENS_TO_REVOK:
        if not key_id:
            continue
        try:
            httpx.post(f"{RETRIEVE}/v1/auth/revoke",
                       json={"id": key_id}, timeout=20)
        except Exception:
            pass
    sys.exit(code)
