"""摄入流水线编排。

    解析 → 清洗（可逆）→ 切片（策略引擎）→ 增强 → 嵌入 → 索引

编排原则：

1. **策略由配置路由**：文档特征（是否扫描件 / 类型 / 表格密度 / 有无章节结构）
   → `configs/chunking_strategies.yaml` 的 routing → 策略名 → 参数。
2. **每阶段可降级**：外部模型（TEI / vLLM / Qdrant / Postgres）任一路径不可用，
   只记录 note，链路继续。摄入绝不因为一个可选依赖而全盘失败。
3. **provenance 在切片前建立**：清洗后的每个块先算好原文偏移，切片阶段直接取用，
   保证每个子块都能映射回原文。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .chunk import chunk_document
from .acl import parse_acl
from .clean import clean_document, compute_chunk_provenance, span_union
from .clients import TEIClient, LLMClient, default_collection, embed_dim, embed_model_id
from .config import (
    load_chunking_config,
    route_strategy,
    strategy_params,
    strategy_pipeline,
)
from .embed import encode_chunks
from .enrich import enrich_chunks
from .graph import extract_entities
from .index import index_chunks
from .models import Block, Chunk, IngestContext, ProvenanceSpan
from .opensearch import index_chunks_opensearch
from .parse import parse_document, section_tree_quality
from .structured import extract_chunk
from .tokenize import TokenCounter, detect_lang, default_counter

log = logging.getLogger("kf.ingest")


# ------------------------------------------------------------------ 文档特征
def document_features(parsed) -> Dict[str, Any]:
    """从解析结果提取路由特征（供 route_strategy 使用）。"""
    blocks: Sequence[Block] = parsed.blocks
    total = max(1, len(blocks))
    tables = sum(1 for b in blocks if b.block_type == "table")
    figures = sum(1 for b in blocks if b.block_type == "figure")
    titles = sum(1 for b in blocks if b.block_type == "title")
    table_density = "high" if tables / total >= 0.25 else ("mid" if tables else "low")

    return {
        "scanned": parsed.needs_ocr,
        "figure_heavy": figures / total >= 0.3,
        "doc_type": parsed.doc_type,
        "table_density": table_density,
        "has_section_hierarchy": bool(parsed.section_tree) and titles > 0,
        "prose": titles == 0,
        "pages": parsed.page_count(),
    }


# ------------------------------------------------------------------ 溯源
def _reuse_embed_tokenizer(counter: Any, embed_client: Any) -> bool:
    """把嵌入客户端已加载的 tokenizer 复用给 TokenCounter。

    只在两处都具备且**尚未加载**时替换，避免覆盖调用方显式传入的计数器
    （单测会传 `TokenCounter("heuristic")`，那种情况必须尊重）。

    返回是否发生了复用。
    """
    if counter is None or embed_client is None:
        return False
    if getattr(counter, "tokenizer", None) is not None:
        return False
    tokenizer = getattr(embed_client, "tokenizer", None)
    if tokenizer is None:
        return False
    try:
        counter.tokenizer = tokenizer
        counter.model_id = getattr(embed_client, "model_id", counter.model_id)
        return True
    except Exception:
        return False


def _assign_provenance(ctx: IngestContext) -> Dict[str, Any]:
    """为父块与子块计算原文偏移，并如实统计精确定位率。

    父块的溯源由其所含子块的溯源取并集得到（父块本身不跨页定位，避免把
    一个块级区间误报成整页区间）。
    """
    raw_pages = ctx.parsed.raw_pages if ctx.parsed else []
    exact = 0
    approx = 0
    missing = 0
    # 按页缓存"规范化 + 位置映射"：不缓存时每个切片都要重新规范化整页，
    # 总开销 = 切片数 × 页字符（二次复杂度）。真机实测 145k 字符文档 164 个切片
    # 每次 142ms、合计 23s。缓存后同页只算一次。
    norm_cache: Dict[int, Any] = {}

    for chunk in ctx.chunks:
        try:
            span = compute_chunk_provenance(chunk, raw_pages, ctx.source_uri,
                                            norm_cache=norm_cache)
        except Exception:
            span = None
        chunk.provenance = span
        if span is None:
            missing += 1
        elif "provenance_approx" in span.cleaning_ops:
            approx += 1
        else:
            exact += 1

    by_parent: Dict[str, List[Optional[ProvenanceSpan]]] = {}
    for chunk in ctx.chunks:
        if chunk.parent_chunk_id:
            by_parent.setdefault(chunk.parent_chunk_id, []).append(chunk.provenance)
    for parent in ctx.parents:
        spans = by_parent.get(parent.chunk_id) or []
        parent.provenance = span_union(spans, ctx.source_uri) if spans else None
        if parent.provenance is not None and \
                "provenance_approx" not in parent.provenance.cleaning_ops:
            exact += 1
        elif parent.provenance is not None:
            approx += 1
        else:
            missing += 1

    total = exact + approx + missing
    stats = {
        "exact": exact,
        "approx": approx,
        "missing": missing,
        "exact_ratio": round(exact / total, 4) if total else 0.0,
    }
    if approx or missing:
        stats["note"] = (
            f"provenance：{approx + missing}/{total} 个切片未精确定位"
            "（已标 provenance_approx，缺偏移好过给错偏移）"
        )
    return stats


# ------------------------------------------------------------------ 主流程
def run_ingest(
    ctx: IngestContext,
    *,
    strategy: Optional[str] = None,
    dry_run: bool = False,
    chunking_cfg: Optional[Dict[str, Any]] = None,
    counter: Optional[TokenCounter] = None,
    tei: Optional[TEIClient] = None,
    llm: Optional[LLMClient] = None,
    qdrant_client: Any = None,
    pg_conn: Any = None,
    collection: Optional[str] = None,
    dim: Optional[int] = None,
    max_pages: Optional[int] = None,
    with_enrich: bool = True,
    graph: Optional[bool] = None,
    acl_spec: str = "",
) -> IngestContext:
    """执行摄入。`dry_run=True` 时只跑到切片 + 关键词，不调外部服务、不落库。"""
    import time as _time

    _t_prev = _time.perf_counter()

    def _lap(name: str) -> None:
        """累计各阶段耗时（毫秒）。用于回答"为什么这次摄入这么慢"。"""
        nonlocal _t_prev
        now = _time.perf_counter()
        ctx.stage_stats.setdefault("timing_ms", {})[name] = round((now - _t_prev) * 1000.0, 1)
        _t_prev = now

    cfg = chunking_cfg or load_chunking_config()
    if graph is not None:
        ctx.graph = graph
    acl_rules = parse_acl(acl_spec)

    # ---- 阶段 1：解析（按后缀分流：PDF / Markdown）----
    _t_prev = _time.perf_counter()
    if ctx.parsed is None:
        path = ctx.source_path or ctx.source_uri
        try:
            ctx.parsed = parse_document(
                path, ctx.doc_id, version=ctx.version,
                doc_type=ctx.doc_type, lang=ctx.lang, max_pages=max_pages,
            )
        except Exception as exc:
            ctx.errors.append(f"parse: {exc}")
            log.error("解析失败 doc=%s: %s", ctx.doc_id, exc)
            return ctx
    ctx.lang = ctx.lang or ctx.parsed.lang
    if not ctx.doc_type:
        ctx.doc_type = ctx.parsed.doc_type
    _lap("parse")
    # 章节树质量自检：标题识别失效会让父块退化成 token 硬切、横跨章节，
    # 直接损害"精确到目标模块"的检索体验——必须**如实告警**而不是静默降级。
    section_quality = section_tree_quality(ctx.parsed)
    ctx.stage_stats["parse"] = {
        "pages": ctx.parsed.page_count(),
        "blocks": len(ctx.parsed.blocks),
        "scanner": ctx.parsed.scanner,
        "warnings": ctx.parsed.warnings,
        "lang": ctx.lang,
        "section_tree": section_quality,
    }
    if section_quality["issues"]:
        for issue in section_quality["issues"]:
            log.warning(
                "章节树质量告警 doc=%s issue=%s titles=%d cues=%d —— 父块将无法按章节切分，"
                "检索可能返回整节而非目标模块",
                ctx.doc_id, issue, section_quality["titles"], section_quality["numbered_cues"],
            )

    # ---- 阶段 2：清洗（可逆）----
    try:
        kept_blocks, report = clean_document(ctx.parsed)
    except Exception as exc:
        ctx.errors.append(f"clean: {exc}")
        log.error("清洗失败 doc=%s: %s", ctx.doc_id, exc)
        return ctx
    ctx.cleaning = report
    _lap("clean")
    ctx.stage_stats["clean"] = {
        "total_blocks": report.total_blocks,
        "removed_blocks": report.removed_blocks,
        "removed_by_type": report.removed_by_type,
        "deduped_blocks": report.deduped_blocks,
    }

    # provenance：槽位在切片后填充（见下方 _assign_provenance）。
    # 旧实现在切片**前**按块定位，而切片产出的是跨块拼接文本，导致 86% 的块
    # 定位失败并退化成整页区间。现在改为拿到切片内容后沿跨页原文定位。

    # ---- 阶段 3：切片（策略路由）----
    features = document_features(ctx.parsed)
    if not strategy:
        strategy = route_strategy(cfg, features)
    ctx.strategy = strategy
    params = strategy_params(cfg, strategy)
    pipeline = strategy_pipeline(cfg, strategy)

    # 先解析嵌入客户端**再**建计数器：同一份 bge-m3 tokenizer 在摄入链路里
    # 会被加载两次（嵌入模型内部一次、TokenCounter 一次）。内存吃紧时实测
    # 重复加载要 43s（正常 3.8s）并额外占内存，而嵌入模型里本就有一份，直接复用。
    # dry_run 不触外部依赖，跳过这步（保持"零依赖也能跑通"的承诺）。
    if tei is None and not dry_run and (counter is None or getattr(counter, "tokenizer", None) is None):
        try:
            from .clients import try_embed_client

            tei = try_embed_client()
        except Exception:
            tei = None

    if counter is None:
        tokenizer_id = params.get("tokenizer") or embed_model_id()
        counter = default_counter(tokenizer_id)
        if _reuse_embed_tokenizer(counter, tei):
            ctx.stage_stats["tokenizer"] = {
                "reused_from": "embed_client",
                "note": "复用了嵌入模型已加载的 tokenizer，避免重复加载",
            }

    # 语义断点需要句向量：TEI 优先，缺 TEI 时用进程内本地嵌入；
    # 两者都没有才走 strategies 内部的词法相似度代理（并如实记入统计）。
    sentence_encoder = None
    if strategy == "semantic" and not dry_run:
        if tei is not None:
            sentence_encoder = lambda texts: tei.embed(list(texts))

    try:
        result = chunk_document(
            kept_blocks,
            doc_id=ctx.doc_id, version=ctx.version, lang=ctx.lang,
            params=params, counter=counter,
            strategy=strategy, pipeline=pipeline, sentence_encoder=sentence_encoder,
        )
    except Exception as exc:
        ctx.errors.append(f"chunk: {exc}")
        log.error("切片失败 doc=%s: %s", ctx.doc_id, exc)
        return ctx

    if strategy == "semantic":
        result.stats.setdefault("notes", []).append(
            "语义断点使用真实嵌入（TEI）" if sentence_encoder is not None
            else "语义断点使用词法相似度代理（未配置 TEI 嵌入端点）"
        )

    ctx.parents = result.parents
    ctx.chunks = list(result.children)
    # 切片后再算溯源：只有拿到切片内容（跨块拼接串）才能在原文里正确定位。
    prov_stats = _assign_provenance(ctx)
    _lap("chunk+provenance")
    ctx.stage_stats["chunk"] = {
        **result.stats,
        "features": features,
        "pipeline": pipeline,
        "provenance": prov_stats,
    }

    # ---- 阶段 4：增强（生成类可选）----
    question_chunks: List[Chunk] = []
    if with_enrich:
        try:
            enriched = enrich_chunks(
                ctx.chunks,
                llm=None if dry_run else llm,
                doc_id=ctx.doc_id, version=ctx.version,
            )
            question_chunks = enriched["question_chunks"]
            ctx.stage_stats["enrich"] = enriched["stats"]
        except Exception as exc:
            ctx.errors.append(f"enrich: {exc}")
            log.warning("增强失败（已跳过）doc=%s: %s", ctx.doc_id, exc)
            ctx.stage_stats["enrich"] = {"error": str(exc)}
    ctx.chunks.extend(question_chunks)

    # 稀疏/稠密编码与落库统一在 embed+index 处理，dry_run 时跳过
    ctx.stage_stats.setdefault("enrich", {}).setdefault("question_chunks", len(question_chunks))

    # ---- 派生元数据：ACL / 结构化 / 实体（纯计算，不触网，dry_run 也执行）----
    all_chunks = ctx.all_chunks()
    acl_applied = 0
    structured_rows = 0
    entity_norms = 0
    for chunk in all_chunks:
        if acl_rules:
            chunk.acl = [dict(rule) for rule in acl_rules]
            acl_applied += 1
        structured = extract_chunk(chunk)
        chunk.meta["structured"] = {
            "skus": structured["skus"],
            "price": structured["price"],
            "currency": structured["currency"],
            "attrs": structured["attrs"],
        }
        structured_rows += len(structured["records"])
        norms = sorted({e["norm"] for e in extract_entities(chunk.content or "")})
        chunk.meta["entities_norm"] = norms
        entity_norms += len(norms)
    _lap("enrich+derive")
    ctx.stage_stats["derive"] = {
        "acl_rules": len(acl_rules),
        "acl_chunks": acl_applied,
        "structured_records": structured_rows,
        "entity_norms": entity_norms,
    }

    if dry_run:
        ctx.stage_stats["dry_run"] = True
        return ctx

    # ---- 阶段 5：嵌入 ----
    all_chunks = ctx.all_chunks()
    model = embed_model_id()
    if tei is None:
        from .clients import try_embed_client

        tei = try_embed_client()
    try:
        embed_stats = encode_chunks(all_chunks, tei=tei, model_id=model)
        ctx.embedded = embed_stats.get("sparse", 0)
        ctx.embedded_model = model
        _lap("embed")
        ctx.stage_stats["embed"] = embed_stats
    except Exception as exc:
        ctx.errors.append(f"embed: {exc}")
        log.warning("嵌入失败 doc=%s: %s", ctx.doc_id, exc)
        ctx.stage_stats["embed"] = {"error": str(exc)}

    # ---- 阶段 6：索引 ----
    if collection is None:
        collection = default_collection(ctx.tenant_id, ctx.kb_id)
    ctx.collection = collection
    try:
        index_stats = index_chunks(
            ctx, all_chunks,
            qdrant_client=qdrant_client, collection=collection,
            dim=dim or embed_dim(), pg_conn=pg_conn,
        )
        ctx.indexed = index_stats.get("qdrant", 0)
        _lap("index")
        ctx.stage_stats["index"] = index_stats
    except Exception as exc:
        ctx.errors.append(f"index: {exc}")
        log.warning("索引失败 doc=%s: %s", ctx.doc_id, exc)
        ctx.stage_stats["index"] = {"error": str(exc)}

    # ---- 阶段 7：OpenSearch 分语言稀疏索引（可选，未配置只记 note）----
    try:
        os_stats = index_chunks_opensearch(ctx, all_chunks)
        ctx.stage_stats["opensearch"] = os_stats
    except Exception as exc:
        log.warning("OpenSearch 索引失败（已跳过）doc=%s: %s", ctx.doc_id, exc)
        ctx.stage_stats["opensearch"] = {"indexed": 0, "notes": [str(exc)]}

    return ctx


def ingestion_summary(ctx: IngestContext, *, collection: str, dry_run: bool = False) -> Dict[str, Any]:
    """机读摘要：CLI `--json` 与 HTTP `POST /v1/ingest` 共用同一口径。"""
    return {
        "doc_id": ctx.doc_id,
        "tenant": ctx.tenant_id,
        "kb": ctx.kb_id,
        "version": ctx.version,
        "lang": ctx.lang,
        "strategy": ctx.strategy,
        "dry_run": bool(dry_run),
        "collection": collection,
        "counts": {
            "parents": len(ctx.parents),
            "children": len(ctx.chunks),
            "embedded": ctx.embedded,
            "indexed": ctx.indexed,
        },
        "cleaning": {
            "removed_blocks": ctx.cleaning.removed_blocks,
            "removed_by_type": ctx.cleaning.removed_by_type,
            "deduped_blocks": ctx.cleaning.deduped_blocks,
        },
        "stages": ctx.stage_stats or {},
        "errors": ctx.errors,
    }
