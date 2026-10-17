"""
Knowledge Fabric · 决策服务（Laya）

职责：策略路由 / 算力级联 / 拒答预判。
形态：非自回归单次前向，输出 typed decision + 校准概率。

注意：Laya **不是文本分类模型**，而是 System-1 结构化判定引擎。官方推理 API 为
`laya.Agent(model, subfolder=...).system_one(state, questions)`，questions 支持
`choice` / `score` / `noul` 三类题型，返回带**校准概率**的结构化判定——因此不存在
「logits → softmax → argmax 到档位」这条路。详见 configs/model_registry.yaml。

本文件把「模型推理」与「决策后处理」解耦：
  - LayaRuntime     : 真实权重推理（按语言选择 checkpoint）
  - heuristic_decide: 冷启动 / 模型不可用时的安全回退
两者输出同一份 Decision 契约（schemas/decision.schema.json）。

官方仓库 convaiinnovations/laya 在**同一 snapshot** 内捆绑多个 checkpoint，
设计上对应 model_registry 的两条注册项，用 `subfolder` 选择：
  - `typed-decisions` 421M ModernBERT-large —— 「typed decision + 校准概率」的
    policy_routing 主模型（对应 `decision-laya`）
  - `multilingual`    322M mmBERT-base     —— 100+ 语言版本，多语言库使用
    （对应 `decision-laya-multilingual`）
官方明确 typed-decisions 为**英文** ModernBERT，非英文会显著退化，故按查询语言路由。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kf.decision")

LAYA_MODEL_ID = os.getenv("LAYA_MODEL_ID", "convaiinnovations/laya")
LAYA_MODEL_MULTILINGUAL_ID = os.getenv("LAYA_MODEL_MULTILINGUAL_ID", "convaiinnovations/laya-multilingual")
# 本地权重目录：非空时**优先**从此加载，避免联网下载（契约第 7 节）。
LAYA_MODEL_PATH = os.getenv("LAYA_MODEL_PATH", "").strip()
LAYA_DEVICE = os.getenv("LAYA_DEVICE", "cpu")
LAYA_CACHE_DIR = os.getenv("LAYA_CACHE_DIR", "/models")
# Laya 官方仓库在一个 snapshot 里捆绑多个 checkpoint（设计上对应 model_registry
# 的 decision-laya / decision-laya-multilingual 两条），用 subfolder 选择：
#   typed-decisions 421M ModernBERT-large —— policy_routing 主模型（英文）
#   multilingual    322M mmBERT-base     —— 100+ 语言版本，多语言库使用
# 两个都留空表示 LAYA_MODEL_PATH 直接指向单个 checkpoint 目录。
LAYA_SUBFOLDER = os.getenv("LAYA_SUBFOLDER", "typed-decisions").strip()
LAYA_SUBFOLDER_MULTILINGUAL = os.getenv("LAYA_SUBFOLDER_MULTILINGUAL", "multilingual").strip()

# checkpoint → 参数量标注（用于 model.params）。必须按**实际服务的 checkpoint**取值，
# 不能写死：写死 421M 会把 multilingual(322M) 的响应错标成 typed-decisions。
_CHECKPOINT_PARAMS: Dict[str, str] = {
    "typed-decisions": "421M",
    "multilingual": "322M",
}

# 缓存根内已有本地快照时强制离线：避免直连 huggingface.co 超时
# （历史故障：LocalEntryNotFoundError / ConnectTimeout WinError 10060）。
# 复用其它项目已下载的缓存（如 E:\Program Files\Laya\hf_home）即可全程离线加载。
if LAYA_CACHE_DIR and os.path.isdir(os.path.join(LAYA_CACHE_DIR, "hub")):
    os.environ.setdefault("HF_HOME", LAYA_CACHE_DIR)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

PORT = int(os.getenv("DECISION_PORT", "8090"))


def model_source() -> "tuple[str, str]":
    """决定权重来源：本地目录优先于 HuggingFace 缓存/下载。

    返回 `(kind, source)`，`kind ∈ {"local", "hf-cache", "hf-download"}`：
      - `local`       : LAYA_MODEL_PATH 指向本地目录，完全离线；
      - `hf-cache`    : 未设本地目录，但 LAYA_CACHE_DIR 内已有 HF 缓存（离线复用）；
      - `hf-download` : 缓存不存在，需要联网下载。
    区分 hf-cache / hf-download 是为了**如实**说明「这次到底有没有下载」——
    只笼统报 hf-download 会让「离线复用他机权重」被误读成「又下了一遍」。
    """
    if LAYA_MODEL_PATH:
        return ("local", LAYA_MODEL_PATH)
    if os.path.isdir(os.path.join(LAYA_CACHE_DIR, "hub")):
        return ("hf-cache", LAYA_MODEL_ID)
    return ("hf-download", LAYA_MODEL_ID)


# 与 configs/retrieval_profiles.yaml 对齐
PROFILES: List[str] = [
    "naive",
    "hybrid_rerank",
    "exact_lookup",
    "hyde",
    "multi_query",
    "decompose_agentic",
    "graph_global",
    "strict_refuse",
]
ESCALATIONS = ["none", "rerank", "llm_rerank", "agentic"]

# 低置信回退档：任何异常都要落到这里
SAFE_DEFAULT = "hybrid_rerank"
CONFIDENCE_FLOOR = float(os.getenv("DECISION_CONFIDENCE_FLOOR", "0.45"))
ANSWERABLE_FLOOR = float(os.getenv("DECISION_ANSWERABLE_FLOOR", "0.55"))


# ------------------------------------------------- Laya 问题集（官方 system_one 口径）
# 档位语义严格取自 configs/retrieval_profiles.yaml，不另造一套口径；
# criteria 的**键就是档位名**，因此 Laya 返回的 probabilities 天然按档位对齐。
_PROFILE_CRITERIA: Dict[str, str] = {
    "naive": "cheapest path: dense-only vector search without reranking; simple factual lookup",
    "hybrid_rerank": "default baseline: dense plus sparse hybrid retrieval, fusion then cross-encoder rerank",
    "exact_lookup": "asks for a specific model number, SKU, part number, price, number or code identifier; needs sparse-weighted exact filtering",
    "hyde": "very short or highly abstract query where a hypothetical answer would improve recall",
    "multi_query": "ambiguous or vaguely worded query that needs several paraphrases to cover the intent",
    "decompose_agentic": "multi-hop question that must be broken into sub-questions and answered iteratively",
    "graph_global": "aggregate or summarization question about how entities, people or topics relate across documents",
    "strict_refuse": "out of scope for this knowledge base, so refusal should be preferred",
}

PROFILE_QUESTIONS: Dict[str, Any] = {
    "retrieval_profile": {
        "type": "choice",
        "instructions": "Which retrieval strategy best fits `query` for this knowledge base?",
        "criteria": _PROFILE_CRITERIA,
    },
    "needs_decomposition": {
        "type": "noul",
        "instructions": "Does answering `query` require gathering evidence in several steps?",
    },
}


def map_laya_answers(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把 Laya `system_one` 的原始返回映射成决策字段。

    纯函数：不依赖 torch / 模型，便于单测直接喂假数据。
    2026-10-17 变更要点：**不再做位置映射**。旧实现用 softmax+argmax 把 logits 第 i 位
    当作 PROFILES[i]，那是把分类模型假设硬套在判定引擎上；现在 choice 题的 criteria
    键即档位名，`probabilities` 由模型直接按档位名给出，不存在错位风险。
    """
    answers = (raw or {}).get("answers") or {}
    pick = answers.get("retrieval_profile") or {}
    choice = pick.get("choice")
    if choice not in PROFILES:
        return None

    raw_probs = pick.get("probabilities") or {}
    probabilities = {
        name: float(raw_probs[name])
        for name in PROFILES
        if isinstance(raw_probs.get(name), (int, float))
    }

    confidence = pick.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = float(probabilities.get(choice, 0.0))

    parts = [f"Laya 判定档位 {choice}（校准置信度 {float(confidence):.3f}）"]
    hop = answers.get("needs_decomposition") or {}
    if isinstance(hop.get("noul"), (int, float)):
        parts.append(f"needs_decomposition={float(hop['noul']):.3f}")
    return {
        "profile": choice,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "reason": "；".join(parts),
    }


# ---------------------------------------------------------------- 契约
class QueryFeatures(BaseModel):
    lang: Optional[str] = None
    length: Optional[int] = None
    has_code_or_identifier: bool = False
    has_number_or_price: bool = False
    kb_scale: int = 0
    session_turn: int = 0
    historical_no_answer_rate: float = 0.0
    kb_languages: List[str] = Field(default_factory=list)
    graph_enabled: bool = False


class DecideRequest(BaseModel):
    query_id: str
    query: str
    tenant_id: str = "default"
    kb_id: Optional[str] = None
    features: QueryFeatures = Field(default_factory=QueryFeatures)


class ModelInfo(BaseModel):
    id: str
    version: str = "unknown"
    params: Optional[str] = None
    latency_ms: float = 0.0


class Decision(BaseModel):
    query_id: str
    trace_id: str
    model: ModelInfo
    profile: str
    escalate: str
    agentic: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    graph: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    answerable_prob: float
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)
    features: Dict[str, Any] = Field(default_factory=dict)
    reason: str
    fallback_used: bool = False
    # 如实暴露当前决策模式：laya（权重推理）/ heuristic-fallback（规则回退）。
    mode: str = "heuristic-fallback"
    # 如实暴露低置信护栏「拿什么值去比阈值」，便于事后审计为何采纳 / 丢弃某档位：
    #   laya_top1_probability —— Laya 结果用其概率分布 top-1（因该 checkpoint 置信度未校准）
    #   confidence            —— heuristic 结果用它自己的置信度
    threshold_basis: str = "confidence"
    threshold_value: float = 0.0


# ---------------------------------------------------------------- 模型运行时
class LayaRuntime:
    """Laya 权重推理封装（官方 `laya.Agent` API）。

    权重来源优先级：`LAYA_MODEL_PATH`（本地目录）> `LAYA_MODEL_ID`（HF 缓存/下载）。
    `LAYA_MODEL_PATH` 指向 HF snapshot 根即可（由 `LAYA_SUBFOLDER` 选 checkpoint），
    也可直接指向某个 checkpoint 目录（此时两个 subfolder 变量都留空）。

    按**查询语言**选择 checkpoint：英文走 typed-decisions、其余走 multilingual。
    依据是官方说明——typed-decisions 为英文 ModernBERT-large，非英文会显著退化。

    任何加载 / 推理失败都**如实记录**，绝不让请求失败（上层走 heuristic 回退）。
    """

    def __init__(self) -> None:
        self.ready = False
        self.kind, self.model_id = model_source()
        # 失败原因：供 /health 与脚本如实暴露（None 表示尚未尝试或加载成功）。
        self.load_error: Optional[str] = None
        # 按 subfolder 缓存已加载的 Agent；逐个记录失败，互不拖累。
        self._agents: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        try:
            self._load()
            self.ready = True
        except Exception as exc:  # 冷启动阶段允许失败，走 heuristic
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning("Laya 加载失败，进入 heuristic 回退模式: %s", self.load_error)

    # ---- 只读状态（供 /health 如实暴露）
    @property
    def checkpoints(self) -> List[str]:
        """已成功加载的 checkpoint 名。"""
        return sorted(self._agents)

    @property
    def checkpoint_errors(self) -> Dict[str, str]:
        """加载失败的 checkpoint 名 → 原因（不隐藏、不美化）。"""
        return dict(self._errors)

    # ---- 选择与加载
    def subfolders(self) -> List[str]:
        """本部署需要加载的 checkpoint 列表（去重，保持声明顺序）。"""
        subs: List[str] = []
        for name in (LAYA_SUBFOLDER, LAYA_SUBFOLDER_MULTILINGUAL):
            if name and name not in subs:
                subs.append(name)
        # 两个都留空：LAYA_MODEL_PATH 直接指向单个 checkpoint 目录
        return subs or [""]

    def subfolder_for(self, lang: Optional[str]) -> str:
        """按语言选 checkpoint：英文 → typed-decisions，其余 → multilingual。

        非英文一律走多语言版，因为英文版在非英文上会显著退化而非温和降级。
        """
        if (lang or "").lower().startswith("en") and LAYA_SUBFOLDER:
            return LAYA_SUBFOLDER
        return LAYA_SUBFOLDER_MULTILINGUAL or LAYA_SUBFOLDER

    def _resolve_source(self) -> str:
        """定位权重来源：本地目录优先，且**必须在联网前**失败。"""
        if self.kind == "local":
            if not os.path.isdir(self.model_id):
                raise FileNotFoundError(f"LAYA_MODEL_PATH 目录不存在: {self.model_id}")
            return self.model_id
        return LAYA_MODEL_ID

    def _load(self) -> None:
        source = self._resolve_source()
        for sub in self.subfolders():
            try:
                self._agents[sub] = self._load_one(source, sub)
                log.info("Laya checkpoint 就绪: %s", sub)
            except Exception as exc:
                self._errors[sub] = f"{type(exc).__name__}: {exc}"
                log.warning("Laya checkpoint 加载失败 (%s): %s", sub, exc)
        if not self._agents:
            detail = "; ".join(f"{k or '(root)'} -> {v}" for k, v in self._errors.items())
            raise RuntimeError(f"Laya 无可用 checkpoint：{detail}")

    @staticmethod
    def _load_one(source: str, subfolder: str) -> Any:
        from laya import Agent

        kwargs: Dict[str, Any] = {"subfolder": subfolder or None}
        device = LAYA_DEVICE.strip()
        if device and device.lower() != "auto":  # 留空 / auto 交给 Agent 自行选择
            kwargs["device"] = device
        return Agent(source, **kwargs)

    def decide(self, query: str, feats: QueryFeatures) -> Optional[Dict[str, Any]]:
        """真实推理：返回 Decision 字段；失败返回 None（上层回退 heuristic）。"""
        if not self.ready:
            return None
        sub = self.subfolder_for(feats.lang)
        agent = self._agents.get(sub)
        if agent is None:
            # 该语言的 checkpoint 不可用时，退用任一已就绪的 checkpoint，
            # 并在 reason 中标注实际用的是哪个（不静默替换）。
            sub = next(iter(self._agents), "")
            agent = self._agents.get(sub)
        if agent is None:
            return None
        try:
            raw = agent.system_one(query, PROFILE_QUESTIONS)
        except Exception as exc:
            log.warning("Laya 推理失败 (checkpoint=%s)，回退 heuristic: %s", sub, exc)
            return None

        mapped = map_laya_answers(raw)
        if mapped is None:
            log.warning("Laya 返回无法映射为档位 (checkpoint=%s)，回退 heuristic", sub)
            return None
        # 如实带出**实际服务本次请求**的 checkpoint，供上层标注 model.params，
        # 避免把 multilingual(322M) 的响应标成 typed-decisions(421M)。
        mapped["checkpoint"] = sub or "root"
        mapped["reason"] = f"{mapped['reason']}；checkpoint={sub or 'root'}"
        return mapped


# ---------------------------------------------------------------- 特征派生
# 型号 / 料号：形如 MX-200、A1000、AB_12、X/200，或显式字段词
_SKU_PATTERN = re.compile(r"[A-Za-z]{1,6}[-_/ ]?\d{2,}[A-Za-z0-9]*")
_SKU_WORDS = ("sku", "型号", "料号", "part no", "part number", "model no", "型号是", "货号")
_PRICE_WORDS = ("价格", "多少钱", "报价", "售价", "费用", "price", "cost", "how much", "元", "¥", "$", "€")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def detect_lang(text: str) -> str:
    """极简语言判定：支撑多语言稀疏索引路由的最低要求。

    生产环境建议换成 fastText lid.176（见 configs/model_registry.yaml 的 lang-detect）。
    """
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if _CJK.search(text):
        return "zh"
    return "en"


def derive_features(query: str, feats: "QueryFeatures") -> "QueryFeatures":
    """客户端未提供时，从查询文本派生廉价词法特征。

    这些特征决定了 heuristic 回退能否真正生效：没有它们，`exact_lookup`
    与 `strict_refuse` 两个分支永远不会被触发，所有查询都会落到安全默认档。
    只填空值，绝不覆盖调用方显式给出的判断。

    注意：`historical_no_answer_rate` 无法从单条 query 得出，需要查
    `decision_trace` 的历史统计，此处保持调用方传入值。
    """
    q = query or ""
    lowered = q.lower()

    if feats.length is None:
        feats.length = len(q)
    if not feats.lang:
        feats.lang = detect_lang(q)

    if not feats.has_code_or_identifier:
        has_sku_word = any(w in lowered for w in _SKU_WORDS)
        feats.has_code_or_identifier = bool(_SKU_PATTERN.search(q)) or has_sku_word

    if not feats.has_number_or_price:
        has_digit = any(c.isdigit() for c in q)
        feats.has_number_or_price = has_digit or any(w in lowered for w in _PRICE_WORDS)

    return feats


# ---------------------------------------------------------------- 启发式回退
def heuristic_decide(query: str, f: QueryFeatures) -> Dict[str, Any]:
    """
    冷启动 / 模型不可用时的规则路由。设计目标不是最优，而是『不劣于安全默认』，
    并把样本记录下来用于后续训练 Laya（见 decision_trace.outcome）。
    """
    q = query.strip()
    length = f.length if f.length is not None else len(q)

    # 顺序即优先级：**意图优先于表面特征**。
    # 对比 / 汇总类查询即使句中出现型号（"对比 MX-100 与 MX-200 的价格"），
    # 本质仍是多跳检索，交给受限 Agentic 比当单点精确查找更合适。
    multi_hop_markers = ("以及", "并且", "分别", "有哪些", "总结", "汇总", "对比", "区别",
                         "and", "compare", "summarize")
    if any(m in q.lower() for m in multi_hop_markers):
        if f.graph_enabled:
            return {"profile": "graph_global", "confidence": 0.58, "reason": "聚合/汇总类查询，且该库已启用 Graph"}
        return {"profile": "decompose_agentic", "confidence": 0.56, "reason": "多跳/对比类查询，启用受限 Agentic"}
    if f.has_code_or_identifier or f.has_number_or_price:
        return {"profile": "exact_lookup", "confidence": 0.62,
                "reason": "含型号/价格/数字，偏稀疏 + 结构化精确检索"}
    if length <= 6:
        return {"profile": "hyde", "confidence": 0.55, "reason": "极短抽象查询，用 HyDE 提升召回"}
    if f.historical_no_answer_rate > 0.4:
        return {"profile": "strict_refuse", "confidence": 0.5, "reason": "该库历史无答案率高，收紧阈值并优先拒答"}
    return {"profile": SAFE_DEFAULT, "confidence": 0.6, "reason": "默认安全档：混合检索 + 重排"}


def build_agentic(profile: str, request: DecideRequest) -> Dict[str, Any]:
    if profile == "decompose_agentic":
        return {"enabled": True, "max_rounds": 3, "stop": "saturation", "budget_tokens": 24000}
    if profile == "graph_global":
        return {"enabled": True, "max_rounds": 2, "stop": "saturation", "budget_tokens": 16000}
    return {"enabled": False}


def build_graph(profile: str, request: DecideRequest) -> Dict[str, Any]:
    if profile == "graph_global" and request.features.graph_enabled:
        return {"enabled": True, "mode": "community_summary", "max_hops": 2}
    return {"enabled": False}


def gate_profile(
    profile: str,
    confidence: float,
    probabilities: Optional[Dict[str, float]],
    used_laya: bool,
) -> "tuple[str, bool, str, float]":
    """低置信护栏：决定是否回退安全默认档，并如实回报「拿什么值比阈值」。

    Laya 的 `confidence` **不可直接用作阈值**：该 checkpoint 随包 temperature 越界，
    模型自报 "Treat confidence from the affected entries as uncalibrated"，实测
    confidence 落在 0.06~0.47，会把**判对**的档位也一并丢掉（真机复现：
    `MX-200 的价格是多少？` 判对 exact_lookup 却因 0.3895 被丢弃）。
    因此对 Laya 结果改用其**自身概率分布的 top-1** 作阈值基准；heuristic 结果
    不产出概率分布，仍用它自己的 confidence。

    返回 `(profile, fallback_used, threshold_basis, threshold_value)`。
    """
    probs = {k: float(v) for k, v in (probabilities or {}).items()}
    top_prob = max(probs.values()) if probs else None
    use_top1 = used_laya and top_prob is not None
    gate = top_prob if use_top1 else confidence
    basis = "laya_top1_probability" if use_top1 else "confidence"
    fallback_used = False
    if gate < CONFIDENCE_FLOOR:
        profile = SAFE_DEFAULT
        fallback_used = True
    return profile, fallback_used, basis, float(gate)


def escalate_for(profile: str, confidence: float) -> str:
    if profile in ("naive",):
        return "rerank" if confidence < CONFIDENCE_FLOOR else "none"
    if profile in ("decompose_agentic", "graph_global"):
        return "agentic"
    if profile == "strict_refuse":
        return "none"
    return "rerank"


# ---------------------------------------------------------------- 应用
_runtime_lock = threading.Lock()
_runtime: Optional[LayaRuntime] = None


def load_runtime() -> LayaRuntime:
    """线程安全的懒加载：模型只在真正需要时下载，且只加载一次。

    不在 import 时加载的原因：`snapshot_download` 会拉取数百 MB 权重，
    会让导入本身变得昂贵且对外部网络强依赖；懒加载后离线环境仍能跑 heuristic。
    """
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = LayaRuntime()
        return _runtime


def runtime_state() -> Dict[str, Any]:
    """只读取当前状态，不触发加载（避免 /health 挂住等模型下载）。

    如实暴露当前模式：`laya`（权重已就绪）或 `heuristic-fallback`（含失败原因）。
    """
    rt = _runtime
    kind, source = model_source()
    return {
        "laya_ready": bool(rt and rt.ready),
        "model_id": rt.model_id if rt else source,
        "model_source": rt.kind if rt else kind,
        "device": LAYA_DEVICE,
        "mode": "laya" if (rt and rt.ready) else "heuristic-fallback",
        # 加载失败原因（未尝试 / 成功时为 None），不隐藏、不美化。
        "load_error": rt.load_error if rt else None,
        # 已就绪的 checkpoint 与逐个失败原因：暴露「哪些语言版本真的可用」。
        "laya_checkpoints": rt.checkpoints if rt else [],
        "laya_checkpoint_errors": rt.checkpoint_errors if rt else {},
    }


def warmup() -> None:
    """后台预热，避免首个真实查询承担模型加载时延。

    这一步是必要的：网关侧决策调用只有 300ms 硬超时，若把加载放在首个
    查询的请求线程里，该查询必然超时并回退安全默认档。
    """
    try:
        load_runtime()
        log.info("Laya 预热完成，ready=%s", bool(_runtime and _runtime.ready))
    except Exception as exc:
        log.warning("Laya 预热失败，继续以 heuristic 回退提供服务: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=warmup, daemon=True, name="laya-warmup").start()
    yield


app = FastAPI(title="Knowledge Fabric · Decision (Laya)", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", **runtime_state()}


@app.get("/profiles")
def profiles() -> Dict[str, List[str]]:
    return {"profiles": PROFILES, "escalations": ESCALATIONS, "safe_default": SAFE_DEFAULT}


@app.post("/decide", response_model=Decision)
def decide(req: DecideRequest) -> Decision:
    import uuid

    t0 = time.perf_counter()
    feats = derive_features(req.query, req.features)

    rt = load_runtime()
    result = rt.decide(req.query, feats)
    # 如实区分：结果真的来自 Laya 权重，还是回退到规则。
    used_laya = result is not None
    # 实际服务本次请求的 checkpoint（用于标注 model.params，400 多 M / 300 多 M 不同）
    laya_checkpoint = (result or {}).get("checkpoint")
    fallback_used = result is None
    if result is None:
        result = heuristic_decide(req.query, feats)

    profile = result.get("profile", SAFE_DEFAULT)
    if profile not in PROFILES:
        profile = SAFE_DEFAULT
        fallback_used = True
    confidence = float(result.get("confidence", 0.5))
    # ---- 低置信回退（阈值基准按来源区分，见 gate_profile）----
    profile, gate_fallback, threshold_basis, threshold_value = gate_profile(
        profile, confidence, result.get("probabilities"), used_laya
    )
    fallback_used = fallback_used or gate_fallback

    answerable = 1.0 - min(1.0, max(0.0, feats.historical_no_answer_rate))
    if profile == "strict_refuse":
        answerable = min(answerable, ANSWERABLE_FLOOR - 0.05)

    latency = (time.perf_counter() - t0) * 1000.0
    return Decision(
        query_id=req.query_id,
        trace_id=uuid.uuid4().hex,
        model=ModelInfo(
            id=rt.model_id if used_laya else "heuristic-rule-v1",
            version="local",
            params=_CHECKPOINT_PARAMS.get(laya_checkpoint) if used_laya else None,
            latency_ms=round(latency, 2),
        ),
        profile=profile,
        escalate=escalate_for(profile, confidence),
        agentic=build_agentic(profile, req),
        graph=build_graph(profile, req),
        answerable_prob=round(answerable, 4),
        confidence=round(confidence, 4),
        probabilities=result.get("probabilities", {}),
        features=feats.model_dump(),
        reason=result.get("reason", ""),
        # 如实暴露「阈值拿什么比」与比出来的值，便于事后审计为何采纳/丢弃。
        threshold_basis=threshold_basis,
        threshold_value=round(float(threshold_value), 4),
        fallback_used=fallback_used,
        mode="laya" if used_laya else "heuristic-fallback",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
