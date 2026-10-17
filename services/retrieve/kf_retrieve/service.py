"""编排：查询理解 → 多通道召回 → 融合 → 重排 → 组装（＋ Agentic 多轮外层）。

对外契约（见 `gateway/src/clients/retrieval.rs`，字段名严格对齐）：

    POST /v1/search
      请求：{ profile, query, tenant, kb, top_k, max_context_tokens, graph_enabled }
      响应：{ evidence, citations, rerank_ms, trace }

四个召回通道（契约第 3 节）：
  - `dense`      Qdrant 稠密，**不做** lang 过滤（跨语言召回）。
  - `sparse`     OpenSearch 分语言 BM25（配置且连得通）否则 Qdrant 稀疏 / Memory BM25，
                 **保留 payload lang 过滤**。
  - `structured` 结构化精确过滤（skus=MatchAny / price=Range / currency=match）；
                 识别不到结构化条件 → 空 + note，绝不退化成再跑 BM25。
  - `graph`      GraphRAG：仅当（档位要求 graph）**且**（`graph_enabled=true`）才启用；
                 否则跳过并记 note。

Agentic RAG（契约第 4 节）：在 `run_search` 外层做 ≤3 轮循环；无生成端点时
**不伪造多轮**，只跑 1 轮并标注 `agentic_stop="no_generator"`。

降级原则：Qdrant / TEI / vLLM / PG / OpenSearch 任一路径不可用都只记 note 并走替代路径，
**绝不 500**；即便编排内部意外异常，也返回空证据 + note 的合法响应。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from . import adaptive
from . import fusion as fusion_mod
from . import graph as graph_mod
from . import opensearch as osp_mod
from . import structured as structured_mod
from .agentic import round_limit as agentic_round_limit
from .agentic import run_agentic
from .assemble import assemble
from .backends import default_collection, matches_filters
from .config import answerability_threshold_for, load_retrieval_config
from .profile import resolve_profile
from .query import UnderstandResult, default_generator, understand
from .rerank import RerankClient, rerank_candidates, try_rerank_client

_ALLOWED_CHANNELS = ("dense", "sparse", "structured", "graph")


class SearchRequest(BaseModel):
    """`POST /v1/search` 请求体（字段名对齐 gateway retrieval.rs）。"""

    query: str
    profile: str = "hybrid_rerank"
    tenant: str = "default"
    kb: Optional[str] = None
    top_k: Optional[int] = None
    max_context_tokens: Optional[int] = None
    # 可选：提供则按 payload 的 ACL 过滤证据（网关默认不传）
    principal: Any = None
    # Graph 通道的库级开关（网关原样转发）；缺省 False 时 Graph 通道不启用，
    # 与既有行为逐字节等价。
    graph_enabled: bool = False


def run_search(
    req: SearchRequest,
    backend: Any,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    gen: Any = None,
    rerank_client: Optional[RerankClient] = None,
    backend_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """执行一次检索，返回对外响应 dict。任何异常都降级为「空证据 + note」。"""
    _entry = time.perf_counter()
    notes: List[str] = []
    if backend_notes:
        notes = ["后端提示：" + "；".join(backend_notes)]
    try:
        cfg = cfg or load_retrieval_config()
        prof = resolve_profile(req.profile, cfg)
        if not prof["known"]:
            notes.append(f"未知档位 {req.profile}，回退 safe_default={prof['safe_default']}")
        if gen is None:
            gen = default_generator()

        agentic_cfg = prof.get("agentic") or {}
        if not agentic_cfg.get("enabled"):
            result = _run_round(req, backend, cfg=cfg, prof=prof, gen=gen,
                                rerank_client=rerank_client, notes=notes)
            trace = result["trace"]
            trace["agentic_rounds"] = 0
            trace["agentic_stop"] = "disabled"
            trace["agentic_rounds_detail"] = []
            # 端到端耗时：`stages_ms` 只覆盖各阶段之和，实测与客户端感知的总延迟能差
            # 一倍以上（如 2.0s vs 3.7s）。没有这个数就无法判断"慢在链路还是慢在外围"。
            trace["total_ms"] = _ms(_entry)
            return result

        # ---- Agentic 多轮（契约第 4 节）----
        max_rounds = agentic_round_limit(agentic_cfg.get("max_rounds"))
        min_new = int(prof.get("min_new_evidence")
                      or agentic_cfg.get("min_new_evidence") or 1)
        budget_tokens = int(prof.get("budget_tokens")
                            or (prof.get("budget") or {}).get("max_context_tokens") or 8000)

        def execute(q: str) -> Dict[str, Any]:
            sub = req.model_copy(update={"query": q})
            return _run_round(sub, backend, cfg=cfg, prof=prof, gen=gen,
                              rerank_client=rerank_client, notes=notes)

        loop = run_agentic(
            execute, query=req.query, gen=gen, max_rounds=max_rounds,
            min_new_evidence=min_new, budget_tokens=budget_tokens,
        )
        if loop["stop"] == "no_generator":
            notes.append(
                "Agentic 多轮跳过：未配置生成端点（VLLM_URL / OPENAI_BASE_URL），已单轮降级"
            )

        final: Dict[str, Any] = loop["result"] or {
            "evidence": [], "citations": [], "rerank_ms": 0.0, "trace": {},
        }
        trace = dict(final.get("trace") or {})
        trace["agentic_rounds"] = int(loop["rounds"])
        trace["agentic_stop"] = loop["stop"]
        trace["agentic_rounds_detail"] = loop["detail"]
        trace["tokens_used"] = int(loop["tokens"])
        trace["notes"] = notes
        return {
            "evidence": loop["evidence"],
            "citations": loop["citations"],
            "rerank_ms": float(final.get("rerank_ms") or 0.0),
            "trace": trace,
        }
    except Exception as exc:  # 兜底：绝不让检索服务 500
        return {
            "evidence": [],
            "citations": [],
            "rerank_ms": 0.0,
            "trace": {
                "stages_ms": {},
                "channels": {},
                "fusion": None,
                "reranked": False,
                "refused": False,
                "agentic_rounds": 0,
                "agentic_stop": "error",
                "agentic_rounds_detail": [],
                "sparse_backend": "memory",
                "structured": {"skipped": "编排异常"},
                "graph": {"enabled": False, "hops": 0, "entities_matched": [],
                          "communities_used": 0},
                "notes": notes + [f"检索编排异常，已降级为空证据: {exc}"],
            },
        }


def _run_round(
    req: SearchRequest,
    backend: Any,
    *,
    cfg: Dict[str, Any],
    prof: Dict[str, Any],
    gen: Any,
    rerank_client: Optional[RerankClient],
    notes: List[str],
) -> Dict[str, Any]:
    """单轮检索（多轮由 `run_search` 复用本函数）。"""
    budget = prof.get("budget") or {}
    max_ctx = int(req.max_context_tokens or budget.get("max_context_tokens") or 8000)
    collection = default_collection(req.tenant, req.kb) if req.kb else None

    # ---- 1) 查询理解 ----
    t0 = time.perf_counter()
    understood: UnderstandResult = understand(req.query, prof, gen=gen, notes=notes)
    stages = {"understand": _ms(t0)}
    query_lang = understood.lang

    def build_filters(channel: str) -> Dict[str, Any]:
        """过滤条件按通道区分。

        - **稀疏通道必须按语言过滤**：BM25 词法权重本质单语言，同一词面在不同语言里
          权重不可比；不过滤会让异语言查询被高词面重叠噪声压过。
        - **稠密通道不过滤语言**：BGE-M3 跨语言对齐，不过滤才能跨语言召回。
        """
        return {
            "tenant_id": req.tenant,
            "kb_id": req.kb,
            "lang": query_lang if channel == "sparse" else None,
            "version": None,
            "status": "active",
            "chunk_type": None,
        }

    # ---- 2) 多通道召回 ----
    t1 = time.perf_counter()
    recall_k = int(req.top_k or prof.get("top_k") or 20)
    queries = understood.all_queries()
    channel_hits: Dict[str, List[Dict[str, Any]]] = {}
    by_id: Dict[str, Dict[str, Any]] = {}
    channel_of: Dict[str, str] = {}
    # 召回阶段顺带记下 dense 通道的原始**余弦分**：它同时满足两个用途——
    #   ① 融合（RRF 只用排名）；
    #   ② 答案可达性判定（拒答）。复用它可以**省掉一次额外 Qdrant 往返**，
    #      真机教训：多出来的那次往返把检索推过 3000ms 全局预算，导致所有查询被
    #      "超出时延预算"降级为 0 证据——为了修拒答反而打垮了整条链路。
    dense_scores: Dict[str, float] = {}

    # structured 通道：条件来自 `exact` 词，识别不到即跳过（不得退化成 BM25）
    structured_conditions = structured_mod.structured_conditions(understood.exact)
    structured_has = structured_mod.has_conditions(structured_conditions)
    structured_trace: Dict[str, Any] = {"skipped": "档位未启用 structured 通道"}

    backend_kind = _backend_kind(backend)
    sparse_backend = backend_kind
    graph_cfg = prof.get("graph") or {}
    graph_wanted = bool(graph_cfg.get("enabled"))
    graph_hops = int(graph_cfg.get("max_hops") or 2) if graph_wanted else 0
    graph_trace: Dict[str, Any] = {
        "enabled": False, "hops": graph_hops, "entities_matched": [], "communities_used": 0,
    }

    for channel in prof.get("channels") or []:
        if channel == "graph":
            continue  # graph 单独处理（需启用条件）
        if channel not in _ALLOWED_CHANNELS:
            notes.append(f"通道 {channel} 未接入，本轮跳过")
            continue

        best: Dict[str, Dict[str, Any]] = {}
        channel_filters = build_filters(channel)

        if channel == "structured":
            if not structured_has:
                structured_trace = {"skipped": "查询无结构化条件"}
                notes.append("structured 通道：查询无结构化条件，已跳过")
                continue
            channel_filters = {**channel_filters, "structured": structured_conditions}
            structured_trace = {"filters": structured_conditions, "matched": 0}

        if channel == "sparse":
            ordered, sparse_backend, note = _sparse_recall(
                backend, queries, channel_filters, recall_k, collection, query_lang, req
            )
            if note:
                notes.append(note)
            if ordered:
                channel_hits[channel] = ordered
                for hit in ordered:
                    cid = hit["chunk"]["chunk_id"]
                    by_id[cid] = hit["chunk"]
                    channel_of.setdefault(cid, hit["channel"])
            continue

        for q in queries:
            for hit in backend.search(channel, q, filters=channel_filters, top_k=recall_k,
                                      collection=collection):
                chunk = hit["chunk"]
                cid = chunk["chunk_id"]
                prev = best.get(cid)
                if prev is None or hit["score"] > prev["score"]:
                    best[cid] = hit
                    by_id[cid] = chunk
                    channel_of.setdefault(cid, hit["channel"])
                if channel == "dense":
                    dense_scores[cid] = max(dense_scores.get(cid, 0.0), float(hit["score"]))
        ordered = sorted(best.values(), key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
        if ordered:
            channel_hits[channel] = ordered
        else:
            # 通道返回空：把原因如实区分开，不要一律沉默。
            # 原实现把异常吞成空结果，真机上表现为某些库的 dense 恒为 0 命中却查不出原因。
            err = getattr(backend, "last_error", "")
            if err:
                notes.append(f"{channel} 通道无命中（原因：{err}）")
            elif channel == "dense":
                notes.append(
                    "dense 通道无命中：该集合内没有 dense 向量"
                    "（常见于用修复前的版本摄入的旧数据）——"
                    "本次将无法计算归一化答案可达分，请重新摄入该文档"
                )
        if channel == "structured":
            structured_trace["matched"] = len(ordered)

    # graph 通道：档位要求 + 库级开关 双条件（设计约束：默认关闭）
    if graph_wanted:
        if not req.graph_enabled:
            notes.append("该知识库未开启 Graph（graph_enabled=false）")
        else:
            graph_result = graph_mod.graph_recall(
                backend, tenant=req.tenant, kb=req.kb or "", query=req.query,
                filters=build_filters("graph"), top_k=recall_k, collection=collection,
                max_hops=graph_hops, mode=graph_cfg.get("mode"),
            )
            if graph_result.get("note"):
                notes.append(str(graph_result["note"]))
            hits = graph_result.get("hits") or []
            for hit in hits:
                cid = hit["chunk"]["chunk_id"]
                by_id[cid] = hit["chunk"]
                channel_of.setdefault(cid, hit["channel"])
            if hits:
                channel_hits["graph"] = hits
            graph_trace = {
                "enabled": True,
                "hops": graph_hops,
                "entities_matched": list(graph_result.get("entities_matched") or []),
                "communities_used": int(graph_result.get("communities_used") or 0),
            }
    stages["retrieve"] = _ms(t1)

    channel_counts = {ch: len(hits) for ch, hits in channel_hits.items()}

    # ---- 3) 融合 ----
    id_rankings = {ch: [h["chunk"]["chunk_id"] for h in hits] for ch, hits in channel_hits.items()}
    scores: Dict[str, float] = {}
    mode = prof.get("fusion")
    if mode == "none":
        for hits in channel_hits.values():
            for hit in hits:
                cid = hit["chunk"]["chunk_id"]
                scores[cid] = max(scores.get(cid, 0.0), float(hit["score"]))
    elif mode in ("rrf", "weighted"):
        scores = fusion_mod.combine(
            id_rankings, mode, k=int(prof.get("fusion_k") or 60),
            weights=prof.get("fusion_weights"), boost=float(prof.get("sparse_boost") or 1.0),
        )
    else:
        notes.append(f"未知融合模式 {mode}，降级为 RRF")
        scores = fusion_mod.combine(id_rankings, "rrf", k=int(prof.get("fusion_k") or 60))

    ranked = fusion_mod.ranked_ids(scores, limit=recall_k)
    candidates = [
        {**by_id[cid], "score": scores[cid], "retrieval_channel": channel_of.get(cid)}
        for cid in ranked if cid in by_id
    ]

    # ---- 4) 重排 ----
    t2 = time.perf_counter()
    reranked = False
    rerank_ms = 0.0
    # 重排窗口按**显存档位**决定，而不是照抄配置里的固定值：
    # 小卡（4GB）只能排 4 个候选（1.5s），大卡排 20 个才不吃亏。
    # 配置若写得比档位更小，尊重配置（那是"我就要少排"的显式表达）。
    rerank_top_n, rerank_tier, rerank_tier_reason = adaptive.effective_top_n(
        prof.get("rerank_top_n"))
    if prof.get("rerank") and candidates:
        if rerank_client is None:
            rerank_client = try_rerank_client()
        rr = rerank_candidates(
            req.query, candidates,
            top_n=rerank_top_n,
            client=rerank_client,
            # 打分窗口与返回窗口一致：cross-encoder 开销随「候选数 × 序列长度²」
            # 增长，融合后 50 个候选全送进去会让 rerank 阶段到数秒级（真机实测 3.9s）。
            candidate_limit=rerank_top_n,
        )
        candidates = rr["candidates"]
        reranked = bool(rr["reranked"])
        rerank_ms = float(rr["ms"]) if reranked else 0.0
        if rr.get("note"):
            notes.append(rr["note"])
    elif prof.get("rerank"):
        notes.append("无候选可重排")
    stages["rerank"] = _ms(t2)
    # ---- 5) 组装 ----
    t3 = time.perf_counter()
    parents = _collect_parents(candidates, backend, collection)
    threshold = prof.get("answerability_threshold")
    if prof.get("on_low_score") == "refuse" and threshold is None:
        threshold = 0.5
    # 父块处理策略（读全局配置，缺省保守）：
    #   parent_context_tokens=0    -> 完全不附带父块，evidence 就是精准命中的子块
    #   backfill_parent_content    -> True 才用父块**替换** content（旧行为，默认关）
    #   acl_default_allow=False    -> 无 ACL 规则时 fail-closed（多租户建议）
    #   answerability="cosine"     -> 拒答改用**归一化余弦**判分（融合分无区分度，见
    #                                 assemble.answerability_scores 的实测说明）
    #   answerability="rerank"     -> 改用 **cross-encoder 概率**判分。实测分离度更好：
    #                                 余弦间隙仅 +0.0336，cross-encoder 原始分间隙 +0.505。
    #                                 代价是必须真的跑了重排（否则自动回落 cosine）。
    assembly_cfg = cfg.get("assembly") or {}
    basis = str(assembly_cfg.get("answerability", "")).lower()
    norm_threshold = None
    threshold_source = ""
    answerability_method: Optional[str] = None
    if basis in ("cosine", "rerank"):
        # 拒答阈值优先级：**按库标定** > 全局底线 > 档位自声明（取较大者）。
        #
        # 为什么要按库标定：实测同一套查询在不同规模的库上分数分布差别很大——
        # 644 块的 `acme/demo` 噪声底（域外探针的 p90）是 0.5493，16 块的库只有
        # 0.3496。固定阈值要么在大库上过度拒答，要么在小库上放行越界问题。
        # 标定文件由 `scripts/calibrate_answerability.py` 生成。
        use_rerank = basis == "rerank" and reranked
        if basis == "rerank" and not reranked:
            notes.append("answerability=rerank 但重排未生效，回落 cosine 判分")
        floor = (assembly_cfg.get("answerability_floor_rerank")
                 if use_rerank else assembly_cfg.get("answerability_floor"))
        # 阈值**按档位**取：重排窗口决定分数分布，跨档位复用阈值会让拒答系统性偏移。
        calibrated = answerability_threshold_for(
            req.tenant, req.kb, basis="rerank" if use_rerank else "cosine",
            tier=rerank_tier if use_rerank else "")
        if calibrated is not None:
            base, threshold_source = calibrated, "per_kb_calibration"
        elif floor is not None:
            base, threshold_source = float(floor), "global_floor"
            if use_rerank:
                # 明确告警：这一档还没标定，阈值只是全局兜底，可能偏松/偏紧。
                notes.append(
                    f"该库在档位 {rerank_tier} 下未标定答案可达阈值，"
                    f"已回退全局底线 {float(floor)}；建议运行 "
                    f"scripts/calibrate_answerability.py"
                )
        else:
            base = None
        if base is not None:
            if threshold is not None and float(threshold) > base:
                norm_threshold, threshold_source = float(threshold), "profile_override"
            else:
                norm_threshold = base
        if use_rerank:
            # 重排已给每个候选打了 cross-encoder 概率，直接用：`answerability_scores`
            # 会优先读取候选自带的 `answerability` 字段。
            for cand in candidates:
                if cand.get("score") is not None:
                    cand["answerability"] = float(cand["score"])
            answerability_method = "rerank"
        else:
            # 答案可达分：**优先复用召回阶段已有的 dense 余弦分**（零额外往返）；
            # 档位不含 dense 通道时才去补算（会多一次往返，但只在需要判拒答时发生）。
            answerability_method = _annotate_answerability(
                backend, req, candidates, collection, queries, recall_k,
                existing=dense_scores,
            )
        if answerability_method == "unavailable":
            notes.append(
                "答案可达分不可用（无嵌入端点或无稠密向量），拒答只能退回融合分判定；"
                "该基准区分度低，越界查询可能被放行"
            )
    result = assemble(
        candidates,
        parents=parents,
        budget_tokens=max_ctx,
        # 融合阈值**照常传**：assemble 优先用归一化余弦，余弦不可用时它退回融合分兜住。
        # 早期实现这里传 None，结果是"余弦算不出来 + 融合阈值也没了"= 完全不拒答。
        threshold=threshold,
        principal=req.principal,
        max_items=recall_k,
        acl_default_allow=bool(assembly_cfg.get("acl_default_allow", True)),
        backfill_parent_content=bool(assembly_cfg.get("backfill_parent_content", False)),
        parent_context_tokens=int(assembly_cfg.get("parent_context_tokens", 0) or 0),
        query_vec=None,
        answerability_threshold=norm_threshold,
    )
    stages["assemble"] = _ms(t3)

    trace = {
        "stages_ms": stages,
        "channels": channel_counts,
        "fusion": mode,
        "reranked": reranked,
        "refused": result["refused"],
        # 归一化答案可达分与它所用的阈值基准：融合分（RRF）对"能否回答"
        # 没有区分度，如实暴露用的是哪种基准，便于事后审计为何拒答 / 为何没拒答。
        "answerability": result.get("answerability"),
        "threshold_basis": result.get("threshold_basis"),
        # 阈值来源：per_kb_calibration / global_floor / profile_override
        # —— 让"为什么这条被拒答"可追溯，不必靠猜配置。
        "threshold_source": threshold_source,
        "threshold_value": norm_threshold,
        # 判据本身也要暴露：cosine 与 rerank 的分数不可比，混看会误判。
        "answerability_method": answerability_method if norm_threshold is not None else None,
        # 重排窗口自适应：档位名 + 实际窗口，便于解释"为什么这台机器排 4 个、那台排 20 个"。
        "rerank_tier": rerank_tier,
        "rerank_top_n": rerank_top_n,
        "rerank_tier_reason": rerank_tier_reason,
        "agentic_rounds": 0,
        "agentic_stop": "disabled",
        "agentic_rounds_detail": [],
        "tokens_used": result["tokens_used"],
        "rerank_ms": round(rerank_ms, 3),
        "profile": prof["name"],
        "requested_profile": req.profile,
        "collection": collection,
        "lang": query_lang,
        "queries": len(queries),
        "exact": understood.exact,
        "sparse_backend": sparse_backend,
        "structured": structured_trace,
        "graph": graph_trace,
        "notes": notes,
    }
    if result.get("notes"):
        trace["notes"] = notes + [n for n in result["notes"] if n not in notes]

    return {
        "evidence": result["evidence"],
        "citations": result["citations"],
        "rerank_ms": round(rerank_ms, 3),
        "trace": trace,
    }


def _sparse_recall(
    backend: Any,
    queries: List[str],
    channel_filters: Dict[str, Any],
    recall_k: int,
    collection: Optional[str],
    query_lang: str,
    req: SearchRequest,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """稀疏通道：OpenSearch（配置且连得通）优先，否则回退 backend 稀疏。

    返回 `(hits, sparse_backend, note)`；`note` 非空表示发生了回退。
    """
    backend_kind = _backend_kind(backend)
    url = osp_mod.open_search_url()
    if url and req.kb:
        combined: Dict[str, Dict[str, Any]] = {}
        fallback_note: Optional[str] = None
        for q in queries:
            found = osp_mod.search(req.tenant, req.kb, query_lang, q, top_k=recall_k, url=url)
            if found.get("note"):
                fallback_note = str(found["note"])
                break
            for item in found["hits"]:
                cid = item["chunk_id"]
                if cid not in combined or item["score"] > combined[cid]["score"]:
                    combined[cid] = item
        if fallback_note is None:
            hits: List[Dict[str, Any]] = []
            for cid, item in combined.items():
                chunk = backend.get_chunk(cid, collection=collection)
                if not chunk or not matches_filters(chunk, channel_filters):
                    continue
                hits.append({"chunk": chunk, "score": float(item["score"]), "channel": "sparse"})
            hits.sort(key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
            return hits[:recall_k], "opensearch", None
        note = fallback_note
    else:
        note = None

    best: Dict[str, Dict[str, Any]] = {}
    for q in queries:
        for hit in backend.search("sparse", q, filters=channel_filters, top_k=recall_k,
                                  collection=collection):
            cid = hit["chunk"]["chunk_id"]
            prev = best.get(cid)
            if prev is None or hit["score"] > prev["score"]:
                best[cid] = hit
    ordered = sorted(best.values(), key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
    return ordered, backend_kind, note


def _backend_kind(backend: Any) -> str:
    """后端类型标签：`qdrant` / `memory`（用于 trace.sparse_backend 如实标注）。"""
    describe = getattr(backend, "describe", None)
    if callable(describe):
        try:
            kind = str((describe() or {}).get("backend") or "")
            if kind:
                return kind
        except Exception:
            pass
    return "memory"


def _annotate_answerability(backend: Any, req: SearchRequest,
                            candidates: List[Dict[str, Any]],
                            collection: Optional[str], queries: List[str],
                            recall_k: int,
                            existing: Optional[Dict[str, float]] = None) -> str:
    """给候选标注 `answerability`（归一化稠密余弦），供拒答判定使用。

    返回实际使用的取分方式：`recall_dense` / `vector_lookup` / `dense_score` /
    `unavailable`。

    取分优先级（越靠前越省）：
      1. `existing` —— 召回阶段 dense 通道已经算过的余弦分，**零额外开销**；
      2. `get_vectors` + 查询向量现算 —— 档位只跑 sparse（如 `exact_lookup`）时
         补一次往返。**必须补**：不补的话该档位永远算不出可达分，拒答静默失效
         （真机实测「火星殖民预算表」在 exact_lookup 下放行了 21 条证据）；
      3. `dense_scores` —— 后端自带的一次性查询接口。

    只对**前 5 条**候选算分：拒答只需要知道"最好的那条有多像"。
    完全不可用时返回 `unavailable`，由调用方退回融合分并**如实记 note**。
    """
    want = [str(c.get("chunk_id")) for c in candidates[:5]]
    if not want:
        return "unavailable"

    scores: Dict[str, float] = {}
    method = "unavailable"

    if existing:
        for cid in want:
            if cid in existing:
                scores[cid] = float(existing[cid])
        if scores:
            method = "recall_dense"

    if not scores:
        fn = getattr(backend, "dense_score_by_vector", None)
        getter = getattr(backend, "get_vectors", None)
        if callable(fn) and callable(getter):
            try:
                qv = fn(queries[0] if queries else req.query, collection=collection)
            except Exception:
                qv = None
            if qv is not None:
                try:
                    vecs = getter(want, collection=collection) or {}
                except Exception:
                    vecs = {}
                for cid, vec in vecs.items():
                    if vec:
                        scores[cid] = _cosine(qv, vec)
                if scores:
                    method = "vector_lookup"

    if not scores:
        fn2 = getattr(backend, "dense_scores", None)
        if callable(fn2):
            try:
                got = fn2(queries[0] if queries else req.query, want, collection=collection)
            except Exception:
                got = None
            if got:
                scores = {str(k): float(v) for k, v in got.items()}
                method = "dense_score"

    for cand in candidates:
        cid = str(cand.get("chunk_id"))
        if cid in scores:
            cand["answerability"] = float(scores[cid])
    return method if scores else "unavailable"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def _query_vector(backend: Any, query: str, collection: Optional[str]) -> Optional[List[float]]:
    """取查询的稠密向量（用于归一化答案可达分）。

    优先用后端的 `embed_query`（若实现），否则退回直接调用其 embed_client。
    任何不可用都返回 None——答案可达分届时缺失，**不得**悄悄退化成融合分。
    """
    fn = getattr(backend, "embed_query", None)
    if callable(fn):
        try:
            vec = fn(query, collection=collection)
            if vec:
                return list(vec)
        except Exception:
            pass
    client = getattr(backend, "embed_client", None)
    if client is None:
        return None
    try:
        return list(client.embed([query])[0])
    except Exception:
        return None


def _collect_parents(candidates: List[Dict[str, Any]], backend: Any,
                     collection: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """按命中子块取回父块（同一父块只取一次）。

    走后端的**批量**接口：旧实现对每个父块各发一次 retrieve，父块多时是
    明显的 N+1（真机检索耗时 4~5s 的组成部分之一）。批量接口不存在时
    逐条回退，保持对该接口的向后兼容。
    """
    wanted: List[str] = []
    seen: set = set()
    for cand in candidates:
        pid = cand.get("parent_chunk_id")
        if not pid:
            continue
        key = str(pid)
        if key in seen:
            continue
        seen.add(key)
        wanted.append(key)
    if not wanted:
        return {}

    get_many = getattr(backend, "get_chunks", None)
    if callable(get_many):
        try:
            found = get_many(wanted, collection=collection) or {}
            return {str(k): v for k, v in found.items() if v}
        except Exception:
            pass

    parents: Dict[str, Dict[str, Any]] = {}
    for pid in wanted:
        try:
            parent = backend.get_chunk(pid, collection=collection)
        except Exception:
            parent = None
        if parent:
            parents[pid] = parent
    return parents


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)
