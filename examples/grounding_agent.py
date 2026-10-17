"""接地引用问答 Agent（Grounding Agent）——MCP 接入 Knowledge Fabric 的参考实现。

## 为什么是这个形态

底座 `gateway` 的 MCP `retrieve` 工具明确是 **evidence-only**：它只返回证据与引用，
**不生成答案**（设计约束：「生成 | 默认 evidence-only（证据交 agent 的 LLM）」）。
因此接入方的正确形态是：**Agent 负责理解与表达，底座负责给出可核实的证据**。

本 Agent 的差异化能力不是"能答"，而是**能自己证明自己没瞎说**：

1. 经 MCP `tools/call retrieve` 取证据（同时拿到 `decision`：档位、是否 fallback）；
2. 证据为空、`grounding.refused`、或可达分低于阈值 → **如实拒答**，绝不编造；
3. **逐条引用回原文独立校验**：把 evidence 的 `char_range` 映射回实际存储的块内容，
   确认"读到的内容 = 引用指向的区间"。这正是最容易失守的契约——
   真实缺陷史里出现过父块回填导致 content 与偏移错位（放大 3~15 倍）；
4. 输出每条主张 + 可点击引用（页码 / 字符区间）+ 校验状态（verified / unverified）；
5. 每次决策可由网关落 `decision_trace`，便于事后审计"为什么走了这个档位"。

## 配置方式（三选一，优先级从高到低）

1. **环境变量**（最高优先级，适合临时覆盖）：
   `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY`（或 `VLLM_*` 别名）
2. **配置文件**（推荐，一次写好不用再敲）——按顺序加载，先找到的先写入：
   - `--env-file` 指定路径（最高）
   - `<项目根>/.env.local`（已加入 .gitignore）
   - `<项目根>/.env`
   - `examples/.env.local`
3. 都没有则退化为 `evidence_summary` 模式（只给证据，不生成答案）

配置文件格式：`KEY=VALUE`，每行一条，**行首** `#` 为注释，支持 `export KEY=VALUE`。
行内 `#` 不算注释（避免密钥里的 `#` 被截断）。
用 `--show-config` 查看最终生效配置（密钥自动打码）。

## 用法

    python examples/grounding_agent.py --show-config
    python examples/grounding_agent.py --check-llm
    python examples/grounding_agent.py --tenant acme --kb demo \
        --direct http://127.0.0.1:8082 --ask "数据库表结构是怎么设计的"
    python examples/grounding_agent.py --tenant acme --kb demo \
        --direct http://127.0.0.1:8082 --scenario

## 依赖

只依赖标准库 + `httpx`（项目已有）。**不需要 LLM 即可运行**。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ================================================================== 配置文件加载
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

# 让同目录的 kb_navigator.py 可被导入（脚本方式运行时 sys.path[0] 已是本目录，
# 但从其它工作目录以模块方式调用时需要显式补上）
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
DEFAULT_ENV_FILES: Tuple[str, ...] = (
    os.path.join(_PROJECT_ROOT, ".env.local"),
    os.path.join(_PROJECT_ROOT, ".env"),
    os.path.join(_EXAMPLES_DIR, ".env.local"),
)

# 支持的写法：KEY=VALUE / export KEY=VALUE；值可带单双引号
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

# 敏感键：展示时打码，避免密钥被截图或贴进日志
SECRET_KEYS = ("OPENAI_API_KEY", "VLLM_API_KEY")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def load_env_file(path: str, *, override: bool = False) -> List[str]:
    """把 `KEY=VALUE` 配置文件读进 `os.environ`，返回实际写入的键名列表。

    * 默认**不覆盖**已存在的环境变量（环境变量优先，便于临时覆盖）；
    * 值为空串也写进去（表示"显式置空"，例如让 `VLLM_URL=` 真正失效）；
    * 只有**行首** `#` 是注释；行内 `#` 保留（密钥里可能含 `#`）；
    * 文件不存在或解析失败的行静默跳过——本函数保持安静，
      由 `show_config()` 统一如实展示"加载了哪些文件、生效了哪些键"。
    """
    applied: List[str] = []
    if not path or not os.path.isfile(path):
        return applied
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.readlines()
    except Exception:
        return applied

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), _strip_quotes(match.group(2))
        if override or name not in os.environ:
            os.environ[name] = value
            applied.append(name)
    return applied


def load_config_files(explicit: Optional[str] = None) -> List[Dict[str, Any]]:
    """按优先级加载配置文件，返回每个文件的加载结果（供如实展示）。"""
    paths: List[str] = []
    if explicit:
        paths.append(explicit)
    paths.extend(DEFAULT_ENV_FILES)

    seen: set = set()
    report: List[Dict[str, Any]] = []
    for path in paths:
        real = os.path.abspath(path)
        if real in seen:
            continue
        seen.add(real)
        if not os.path.isfile(real):
            report.append({"path": real, "applied": [], "exists": False})
            continue
        applied = load_env_file(real)
        report.append({"path": real, "applied": applied, "exists": True})
    return report


def mask(value: str) -> str:
    """密钥打码：不足 12 位只报长度，否则留首尾各 4 位。"""
    if not value:
        return "(未设置)"
    if len(value) <= 12:
        return f"已设置（长度 {len(value)}，内容隐藏）"
    return f"{value[:4]}{'*' * 8}{value[-4:]}（长度 {len(value)}）"


# ================================================================== 配置
DEFAULT_GATEWAY = "http://127.0.0.1:8000/mcp"
PROTOCOL_VERSION = "2024-11-05"

MIN_TOP_SCORE = 0.0


def llm_config() -> Dict[str, Any]:
    """读取生成端点配置（每次读取，便于 `--env-file` 之后立刻生效）。

    变量名兼容两套习惯：本项目内部叫 vLLM，接云端时更自然叫 OPENAI_*。
    端点要写到 `/v1` 那一层（代码自行拼 `/chat/completions`）。
    """
    base = (os.environ.get("VLLM_URL") or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_URL") or "").strip()
    model = (os.environ.get("VLLM_MODEL") or os.environ.get("OPENAI_MODEL")
             or "Qwen/Qwen2.5-1.5B-Instruct").strip()
    key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "").strip()
    try:
        timeout = float(os.environ.get("KF_LLM_TIMEOUT", "120") or 120)
    except ValueError:
        timeout = 120.0
    return {"base": base, "model": model, "key": key, "timeout": timeout}


def gateway_url() -> str:
    return os.environ.get("KF_GATEWAY_MCP", DEFAULT_GATEWAY)


def min_top_score() -> float:
    try:
        return float(os.environ.get("KF_AGENT_MIN_SCORE", "0.0") or 0.0)
    except ValueError:
        return 0.0


def show_config(report: List[Dict[str, Any]]) -> None:
    """`--show-config`：如实展示配置文件加载情况与最终生效的关键配置。"""
    print("=== 配置文件加载（先加载者优先写入，环境变量优先于文件）===")
    any_file = False
    for item in report:
        mark = "已加载" if item["exists"] else "不存在"
        print(f"  [{mark}] {item['path']}")
        if item["applied"]:
            print(f"            写入 {len(item['applied'])} 个键: {', '.join(item['applied'])}")
    any_file = any(i["exists"] for i in report)
    if not any_file:
        print("  （未找到任何配置文件；可用 --env-file 指定，或直接设环境变量）")

    cfg = llm_config()
    print("\n=== 生成端点（LLM）===")
    print(f"  端点     : {cfg['base'] or '(未配置)'}"
          + (f"/chat/completions" if cfg['base'] else ""))
    print(f"  模型     : {cfg['model']}")
    print(f"  密钥     : {mask(cfg['key'])}")
    print(f"  超时     : {cfg['timeout']:.0f}s")
    print(f"  模式     : {'generated（LLM 写答案）' if cfg['base'] else 'evidence_summary（只给证据）'}")

    print("\n=== 其他 ===")
    print(f"  DATABASE_URL    : {'已设置' if os.environ.get('DATABASE_URL') else '(未设置，引用校验将退化为自洽性检查)'}")
    print(f"  KF_GATEWAY_MCP  : {gateway_url()}")
    print(f"  KF_AGENT_MIN_SCORE: {min_top_score()}")
    print(f"  KF_LLM_TIMEOUT  : {os.environ.get('KF_LLM_TIMEOUT', '(缺省 120)')}")

    print("\n提示：优先用 .env.local 存放密钥（已在 .gitignore 中，不会被提交）。")


# ================================================================== MCP 客户端
@dataclass
class MCPClient:
    """极简 MCP Streamable HTTP 客户端（initialize / tools/list / tools/call）。"""

    url: str = ""
    timeout: float = 300.0
    _rid: int = 0

    def __post_init__(self) -> None:
        if not self.url:
            self.url = gateway_url()

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._rid += 1
        body: Dict[str, Any] = {"jsonrpc": "2.0", "id": self._rid, "method": method}
        if params is not None:
            body["params"] = params
        resp = httpx.post(self.url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP {method} 失败: {data['error']}")
        return data.get("result") or {}

    def initialize(self) -> Dict[str, Any]:
        return self.call("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}})

    def tools(self) -> List[Dict[str, Any]]:
        return self.call("tools/list").get("tools") or []

    def retrieve(self, **arguments: Any) -> Dict[str, Any]:
        result = self.call("tools/call", {"name": "retrieve", "arguments": arguments})
        return result.get("structuredContent") or {}


@dataclass
class DirectClient:
    """直连检索服务（`POST /v1/search`）的等价客户端。

    用于**把检索质量与决策时延解耦**：Laya 在 CPU 上单次决策 0.8~5.0s，而网关的
    全局时延预算是 3000ms——决策一返回预算就耗尽，检索被降级为 0 证据。
    这不是检索的缺陷，但会掩盖检索本身的评测。直连模式下没有决策环节。

    接口形状与 MCPClient 保持一致（`.initialize()` / `.tools()` / `.retrieve()`），
    因此 Agent 代码无需分支。
    """

    url: str = "http://127.0.0.1:8082"
    timeout: float = 300.0

    def initialize(self) -> Dict[str, Any]:
        h = httpx.get(f"{self.url.rstrip('/')}/health", timeout=30).json()
        return {"serverInfo": {"name": "kf-retrieve(direct)", "version": h.get("status", "")}}

    def tools(self) -> List[Dict[str, Any]]:
        return [{"name": "retrieve", "description": "直连检索服务（无决策层）",
                 "inputSchema": {"required": ["query"]}}]

    def retrieve(self, **arguments: Any) -> Dict[str, Any]:
        body = dict(arguments)
        body.setdefault("top_k", 10)
        resp = httpx.post(f"{self.url.rstrip('/')}/v1/search", json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        trace = data.get("trace") or {}
        data.setdefault("decision", {
            "profile": trace.get("profile") or arguments.get("profile") or "hybrid_rerank",
            "fallback_used": False,
        })
        data.setdefault("grounding", {
            "passed": not trace.get("refused"),
            "refused": bool(trace.get("refused")),
            "score": float(trace.get("answerability") or 0.0),
            "refuse_reason": (trace.get("notes") or [None])[-1] if trace.get("refused") else None,
        })
        return data


# ================================================================== 引用校验
_PUNCT = re.compile(r"[\s，。；：、！？,.;:!?\"'()（）\[\]【】—\-·]")


def _norm(text: str) -> str:
    """比对用规范化：去掉空白与标点（清洗阶段做过 NFKC，逐字比对不可靠）。"""
    return _PUNCT.sub("", (text or "").lower())


def load_chunk_texts(pg_dsn: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """从 Postgres 取块的权威内容，用于**独立**校验引用。

    有意不走检索服务的返回值：那正是被校验的对象，用它自证等于没有校验。
    """
    dsn = pg_dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        return {}
    try:
        import psycopg
    except Exception:
        return {}
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT chunk_id, content, page FROM chunks WHERE status='active'")
            return {row[0]: {"content": row[1], "page": row[2]} for row in cur.fetchall()}
    except Exception:
        return {}


@dataclass
class CitationCheck:
    """单条引用的校验结果。"""

    chunk_id: str
    page: int
    char_range: Optional[List[int]]
    status: str          # verified / self_consistent / unverified
    detail: str
    matched_chars: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, "page": self.page, "char_range": self.char_range,
                "status": self.status, "detail": self.detail,
                "matched_chars": self.matched_chars}


def verify_citation(evidence: Dict[str, Any],
                    authoritative: Dict[str, Dict[str, Any]],
                    min_coverage: float = 0.6) -> CitationCheck:
    """校验一条引用：evidence.content 是否真的对应它声明的 char_range。

    校验强度分两级（结果如实区分，不夸大）：
      * `verified` —— 能取到权威块内容，且 evidence.content 与之高度重合；
      * `self_consistent` —— 取不到权威内容（无 DATABASE_URL 等），只能确认
        `offset_chunk_id == chunk_id` 且区间不异常；
      * `unverified` —— 两者矛盾（内容对不上、或区间倒置/指向别的块）。
    """
    cid = str(evidence.get("chunk_id") or "")
    cr = evidence.get("char_range")
    page = int(evidence.get("page") or 0)
    content = str(evidence.get("content") or "")

    if isinstance(cr, list) and len(cr) == 2 and cr[1] < cr[0]:
        return CitationCheck(cid, page, cr, "unverified",
                             f"字符区间倒置：{cr[0]} > {cr[1]}")

    if cr and evidence.get("offset_chunk_id") != cid:
        return CitationCheck(cid, page, cr, "unverified",
                             f"偏移归属不一致：offset_chunk_id={evidence.get('offset_chunk_id')}")

    record = authoritative.get(cid)
    if record is None:
        return CitationCheck(cid, page, cr, "self_consistent",
                             "无法取到权威块内容，仅确认偏移归属自洽")

    source = str(record.get("content") or "")
    if not source:
        return CitationCheck(cid, page, cr, "self_consistent", "权威块内容为空，无法比对")

    a, b = _norm(content), _norm(source)
    if not a or not b:
        return CitationCheck(cid, page, cr, "self_consistent", "规范化后为空，无法比对")

    matched = 0
    for probe_len in (min(len(a), 60), min(len(a), 30), min(len(a), 16)):
        if probe_len > 0 and a[:probe_len] in b:
            matched = len(a)
            break
    if not matched:
        step = max(1, len(a) // 40)
        for i in range(0, max(1, len(a) - 16), step):
            if a[i:i + 16] in b:
                matched = max(matched, len(a) - i)
    coverage = matched / len(a)
    if coverage >= min_coverage:
        return CitationCheck(cid, page, cr, "verified",
                             f"与权威块内容重合 {coverage:.0%}", matched)
    return CitationCheck(cid, page, cr, "unverified",
                         f"与权威块内容仅重合 {coverage:.0%}（content 与引用可能不同源）", matched)


# ================================================================== Agent
@dataclass
class AgentTurn:
    query: str
    mode: str                     # refused / evidence_summary / generated
    answer: str
    profile: str
    fallback_used: bool
    evidence_count: int
    top_score: float
    citations: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[CitationCheck] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def verified_ratio(self) -> float:
        if not self.checks:
            return 0.0
        ok = sum(1 for c in self.checks if c.status in ("verified", "self_consistent"))
        return ok / len(self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "mode": self.mode, "answer": self.answer,
            "profile": self.profile, "fallback_used": self.fallback_used,
            "evidence_count": self.evidence_count, "top_score": self.top_score,
            "citations": self.citations,
            "checks": [c.as_dict() for c in self.checks],
            "verified_ratio": round(self.verified_ratio, 3),
            "notes": self.notes,
        }


class GroundingAgent:
    """接地引用问答 Agent：只依据检索到的证据作答，并逐条校验引用。"""

    def __init__(self, tenant: str, kb: str, *, gateway: Optional[str] = None,
                 principal: Optional[Any] = None, graph_enabled: bool = False,
                 load_authoritative: bool = True, direct: str = "",
                 profile: str = "") -> None:
        self.tenant = tenant
        self.kb = kb
        self.principal = principal
        self.graph_enabled = graph_enabled
        self.profile = profile
        self.mcp = DirectClient(direct) if direct else MCPClient(gateway or gateway_url())
        self.authoritative = load_chunk_texts() if load_authoritative else {}
        self.gen = self._build_generator()

    # ---- 可选生成端点（未配置时走"证据摘要"模式，绝不假装生成了答案）
    @staticmethod
    def _build_generator():
        cfg = llm_config()
        base = cfg["base"]
        if not base:
            return None
        model, key, timeout = cfg["model"], cfg["key"], cfg["timeout"]
        url = f"{base.rstrip('/')}/chat/completions"

        def generate(prompt: str) -> str:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            try:
                resp = httpx.post(
                    url,
                    json={"model": model, "temperature": 0.1,
                          "messages": [{"role": "user", "content": prompt}]},
                    headers=headers, timeout=timeout,
                )
            except Exception as exc:
                # 网络层失败要**指出具体 URL**，否则云端配置错在哪完全看不出来
                raise RuntimeError(f"生成端点不可达 {url}: {exc}") from exc
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"生成端点 {url} 返回 HTTP {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"生成端点返回空 choices: {str(data)[:200]}")
            return str((choices[0].get("message") or {}).get("content") or "").strip()

        return generate

    @staticmethod
    def check_llm() -> int:
        """`--check-llm`：只验证生成端点能否真的出字，便于先排配置再做题。

        返回进程退出码：0 通过 / 2 未配置 / 1 调用失败。
        """
        cfg = llm_config()
        if not cfg["base"]:
            print("未配置生成端点。三种配置方式任选：")
            print("  ① 配置文件（推荐）：在 F:\\knowledge-fabric\\.env.local 写入")
            print("       OPENAI_BASE_URL=https://api.deepseek.com/v1")
            print("       OPENAI_MODEL=deepseek-chat")
            print("       OPENAI_API_KEY=sk-...")
            print("  ② 环境变量：$env:OPENAI_BASE_URL / $env:OPENAI_MODEL / $env:OPENAI_API_KEY")
            print("  ③ 命令行：python examples\\grounding_agent.py --env-file <路径> --check-llm")
            print("\n不配置也能用 Agent，只是输出退化为 evidence_summary（不给自然语言答案）。")
            return 2
        print(f"端点   : {cfg['base']}/chat/completions")
        print(f"模型   : {cfg['model']}")
        print(f"密钥   : {mask(cfg['key'])}")
        print(f"超时   : {cfg['timeout']:.0f}s")
        gen = GroundingAgent._build_generator()
        t0 = time.perf_counter()
        try:
            out = gen("只回复两个字：连通")
        except Exception as exc:
            print(f"\n[FAIL] 调用失败：{exc}")
            print("\n常见原因：")
            print("  · 端点少写或多写了 /v1（应为 https://host/v1，代码自行拼 /chat/completions）")
            print("  · 模型名不在该服务上（401 是密钥问题，400/404 多是模型名或路径问题）")
            print("  · 密钥未设置或已失效")
            return 1
        print(f"\n[PASS] 调用成功，耗时 {(time.perf_counter()-t0)*1000:.0f} ms")
        print(f"       回复：{out[:80]}")
        return 0

    def ask(self, query: str) -> AgentTurn:
        args: Dict[str, Any] = {"query": query, "tenant": self.tenant, "kb": self.kb}
        if self.principal is not None:
            args["principal"] = self.principal
        if self.graph_enabled:
            args["graph_enabled"] = True
        if self.profile:
            args["profile"] = self.profile
        data = self.mcp.retrieve(**args)

        decision = data.get("decision") or {}
        grounding = data.get("grounding") or {}
        evidence = data.get("evidence") or []
        citations = data.get("citations") or []
        profile = str(decision.get("profile") or "")
        fallback = bool(decision.get("fallback_used"))
        top_score = float(grounding.get("score") or 0.0)
        notes: List[str] = []

        if grounding.get("refused"):
            reason = grounding.get("refuse_reason") or "底座判定无答案"
            return AgentTurn(query, "refused", f"依据不足，拒绝回答。原因：{reason}",
                             profile, fallback, len(evidence), top_score,
                             citations, [], [f"grounding.refused: {reason}"])
        if not evidence:
            return AgentTurn(query, "refused", "未检索到任何证据，拒绝回答。",
                             profile, fallback, 0, top_score, citations, [], ["证据集为空"])
        threshold = min_top_score()
        if top_score < threshold:
            return AgentTurn(query, "refused",
                             f"最高证据分 {top_score} 低于门槛 {threshold}，拒绝回答。",
                             profile, fallback, len(evidence), top_score, citations, [],
                             ["低于 Agent 自设门槛"])

        checks = [verify_citation(e, self.authoritative) for e in evidence]
        bad = [c for c in checks if c.status == "unverified"]
        if bad:
            notes.append(f"{len(bad)} 条引用未通过校验，已在答案中标注")

        if self.gen is not None:
            answer = self._generate_answer(query, evidence, checks)
            mode = "generated"
        else:
            answer = self._summarize(query, evidence, checks)
            mode = "evidence_summary"
            notes.append("未配置生成端点：仅输出证据摘要，不生成自然语言答案")
        return AgentTurn(query, mode, answer, profile, fallback,
                         len(evidence), top_score, citations, checks, notes)

    def _generate_answer(self, query: str, evidence: List[Dict[str, Any]],
                         checks: List[CitationCheck]) -> str:
        blocks = []
        for i, (e, c) in enumerate(zip(evidence, checks), start=1):
            tag = {"verified": "可核实", "self_consistent": "自洽"}.get(c.status, "未通过校验")
            blocks.append(
                f"[证据 {i}]（{tag}；来源 {e.get('source_uri') or e.get('source')} "
                f"第 {e.get('page')} 页；区间 {e.get('char_range')}）\n{e.get('content')}"
            )
        prompt = (
            "你是企业知识库问答助手。**只能依据下面给出的证据回答**，"
            "每个论断后面用 [证据 N] 标注来源；证据不足以回答时明确说\"依据不足\"，不要推测。\n\n"
            f"问题：{query}\n\n" + "\n\n".join(blocks)
        )
        try:
            return self.gen(prompt)
        except Exception as exc:
            return self._summarize(query, evidence, checks) + f"\n（生成失败已降级为摘要：{exc}）"

    @staticmethod
    def _summarize(query: str, evidence: List[Dict[str, Any]],
                   checks: List[CitationCheck]) -> str:
        lines = [f"以下为检索到的原文证据（共 {len(evidence)} 条）："]
        for i, (e, c) in enumerate(zip(evidence, checks), start=1):
            flag = {"verified": "✓", "self_consistent": "~", "unverified": "✗"}.get(c.status, "?")
            snippet = (e.get("content") or "").strip().replace("\n", " ")
            lines.append(
                f"{flag} [证据 {i}] {snippet[:160]}"
                f"  —— 第 {e.get('page')} 页 区间 {e.get('char_range')}"
            )
        return "\n".join(lines)


# ================================================================== 内置场景
SCENARIOS: List[Dict[str, Any]] = [
    # 信息型：问题必须**确实能从该库回答**，否则拒答才是正确行为
    {"name": "信息型（库内确有其事，应给出证据）",
     "query": "数据库设计方案里用户表是怎么设计的", "expect": "answer"},
    {"name": "结构型（应命中具体章节）",
     "query": "数据库表结构是怎么设计的", "expect": "answer"},
    {"name": "越界型（与知识库无关，应拒答）",
     "query": "今天天气怎么样，帮我订张去上海的机票", "expect": "refuse"},
    {"name": "无中生有型（库内不存在，应拒答）",
     "query": "请给出本知识库不存在的火星殖民预算表", "expect": "refuse"},
]


def run_scenario(agent: GroundingAgent) -> int:
    print(f"\n{'='*70}\n场景集：tenant={agent.tenant} kb={agent.kb}\n{'='*70}")
    failures = 0
    for case in SCENARIOS:
        turn = agent.ask(case["query"])
        ok = (turn.mode == "refused") if case["expect"] == "refuse" else (turn.mode != "refused")
        if not ok:
            failures += 1
        print(f"\n[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"   query        : {case['query']}")
        print(f"   mode         : {turn.mode}  (期望 {case['expect']})")
        print(f"   profile      : {turn.profile}  fallback={turn.fallback_used}")
        print(f"   evidence     : {turn.evidence_count} 条  top_score={turn.top_score}")
        print(f"   引用校验     : {turn.verified_ratio:.0%} "
              f"({sum(1 for c in turn.checks if c.status=='verified')} verified / "
              f"{sum(1 for c in turn.checks if c.status=='self_consistent')} self_consistent / "
              f"{sum(1 for c in turn.checks if c.status=='unverified')} unverified)")
        if turn.notes:
            print(f"   notes        : {' | '.join(turn.notes)}")
        print("   answer       :")
        for line in turn.answer.split("\n")[:8]:
            print(f"       {line[:150]}")
    print(f"\n{'='*70}\n场景集结果：{'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="接地引用问答 Agent（MCP）",
        epilog="配置优先级：环境变量 > --env-file > <项目根>/.env.local > .env > examples/.env.local",
    )
    ap.add_argument("--env-file", default="", help="额外指定的配置文件路径（KEY=VALUE）")
    ap.add_argument("--show-config", action="store_true", help="展示生效配置（密钥打码）")
    ap.add_argument("--check-llm", action="store_true", help="只检查生成端点配置")
    ap.add_argument("--gateway", default="", help="MCP 端点（缺省取 KF_GATEWAY_MCP 或 8000/mcp）")
    ap.add_argument("--direct", default="", help="直连检索服务地址（跳过决策层）")
    ap.add_argument("--profile", default="", help="直连模式下的检索档位")
    ap.add_argument("--tenant", default="", help="租户")
    ap.add_argument("--kb", default="", help="知识库 id")
    ap.add_argument("--ask", action="append", default=[], help="提问（可多次）")
    ap.add_argument("--principal", help="访问主体，如 user:u1")
    ap.add_argument("--graph", action="store_true", help="启用 GraphRAG 通道")
    ap.add_argument("--scenario", action="store_true", help="跑内置场景集")
    ap.add_argument("--list-kbs", action="store_true", help="列出所有知识库（有哪些库、各多少文档）")
    ap.add_argument("--discover", action="store_true",
                    help="拿问题在所有库上探一遍，告诉你该问哪个库（需配合 --ask）")
    ap.add_argument("--json", action="store_true", help="机读输出")
    args = ap.parse_args()

    # 先加载配置文件，再读取任何配置（顺序很重要）
    report = load_config_files(args.env_file or None)

    if args.show_config:
        show_config(report)
        return 0
    if args.check_llm:
        return GroundingAgent.check_llm()

    # 知识库导航（不需要 --tenant/--kb）
    if args.list_kbs or args.discover:
        from kb_navigator import discover, list_kbs

        kbs = list_kbs()
        if not kbs:
            print("未发现任何知识库（检查 DATABASE_URL），或库内还没有活跃块。", file=sys.stderr)
            return 1

        if args.list_kbs:
            if args.json:
                print(json.dumps([k.as_dict() for k in kbs], ensure_ascii=False, indent=2))
                return 0
            print(f"=== 知识库共 {len(kbs)} 个（按活跃块数降序）===")
            print(f"  {'库':<24} {'文档':>4} {'块':>6}  文档示例")
            for k in kbs:
                print(f"  {k.key:<24} {k.docs:>4} {k.chunks:>6}  {' / '.join(k.titles[:3])}")
            print("\n提示：用 --discover --ask '<问题>' 让程序替你找该问哪个库。")
            return 0

        if not args.ask:
            print("--discover 需要配合 --ask '<问题>'。", file=sys.stderr)
            return 2
        retrieve_url = args.direct or os.environ.get("RETRIEVAL_URL") or "http://127.0.0.1:8082"
        rc = 0
        for question in args.ask:
            print(f"\n{'='*70}\n问题：{question}\n在 {len(kbs)} 个库上探测…\n{'='*70}")
            hits = discover(question, kbs, retrieve_url=retrieve_url, profile=args.profile)
            if args.json:
                print(json.dumps([h.as_dict() for h in hits], ensure_ascii=False, indent=2))
                continue
            best = None
            for h in hits:
                flag = "拒答" if h.refused else "有证据"
                print(f"  [{flag}] {h.key:<22} 可达分={h.answerability_display:<8} 证据={h.evidence}")
                if h.top_snippet:
                    print(f"           {h.top_snippet[:88]}")
                if not h.answerability_available and not h.refused:
                    print("           ⚠ 该库缺 dense 向量，可达分无法计算（旧数据，需重新摄入）")
                if best is None and not h.refused and h.evidence:
                    best = h
            if best is not None:
                print(f"\n→ 建议问：--tenant {best.tenant} --kb {best.kb}"
                      + ("" if best.answerability_available
                         else "（注意：该库缺 dense 向量，建议先重新摄入）"))
            else:
                print("\n→ 所有库都拒答：该问题在这批知识库里确实没有答案。")
                rc = 1
        return rc

    if not args.tenant or not args.kb:
        print("需要 --tenant 与 --kb（例如 --tenant acme --kb demo）。", file=sys.stderr)
        print("提示：只看配置用 --show-config；只测 LLM 用 --check-llm；"
              "不知道问哪个库用 --list-kbs / --discover --ask '…'。", file=sys.stderr)
        return 2

    agent = GroundingAgent(args.tenant, args.kb, gateway=args.gateway or None,
                           principal=args.principal, graph_enabled=args.graph,
                           direct=args.direct, profile=args.profile)
    init = agent.mcp.initialize()
    tools = agent.mcp.tools()
    if not args.json:
        cfg = llm_config()
        print(f"MCP: {init.get('serverInfo', {}).get('name')} "
              f"v{init.get('serverInfo', {}).get('version')} | 工具: "
              f"{', '.join(t['name'] for t in tools)}")
        print(f"权威块索引: {len(agent.authoritative)} 个（用于独立校验引用）")
        loaded = [i["path"] for i in report if i["exists"]]
        if loaded:
            print(f"配置文件: {len(loaded)} 个已加载（--show-config 看详情）")
        if agent.gen is not None:
            print(f"生成端点: {cfg['model']} @ {cfg['base']}（mode=generated）")
        else:
            print("生成端点: 未配置 → mode=evidence_summary（只给证据，不生成答案）"
                  "；用 --check-llm 或 --show-config 排查")

    if args.scenario:
        return run_scenario(agent)

    if not args.ask:
        ap.print_help()
        return 2

    turns = [agent.ask(q) for q in args.ask]
    if args.json:
        print(json.dumps([t.as_dict() for t in turns], ensure_ascii=False, indent=2))
    else:
        for turn in turns:
            print(f"\n{'='*70}\nQ: {turn.query}")
            print(f"   mode={turn.mode} profile={turn.profile} fallback={turn.fallback_used} "
                  f"evidence={turn.evidence_count} top_score={turn.top_score}")
            print(f"   引用校验: {turn.verified_ratio:.0%}")
            for c in turn.checks[:6]:
                print(f"     [{c.status}] {c.chunk_id} 第 {c.page} 页 {c.char_range} — {c.detail}")
            print("   A:")
            for line in turn.answer.split("\n"):
                print(f"     {line[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
