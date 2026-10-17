"""摄入 HTTP 接口：`GET /health`、`POST /v1/ingest`、`POST /v1/ingest-dir`、`GET /v1/documents`。

给 Web 控制台（`console/index.html`）与任意 agent 产品提供上传即摄入的最小入口，
与 CLI 走**同一条流水线、同一份摘要口径**（`pipeline.ingestion_summary`）。

降级原则与 CLI 一致：Qdrant / Postgres / TEI 任一不可用都只记 note 并继续，
绝不因为一个可选依赖而让上传失败。端口用 `INGEST_PORT`（默认 8083）。
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .clients import (
    connect_pg,
    connect_qdrant,
    database_url,
    default_collection,
    embed_dim,
    qdrant_url,
    tei_embed_url,
)
from . import auth
from .models import IngestContext
from .parse import SUPPORTED_SUFFIXES, discover_documents
from .pipeline import ingestion_summary, run_ingest

log = logging.getLogger("kf.ingest.http")

PORT = int(os.environ.get("INGEST_PORT", "8083"))


def _warmup() -> None:
    """启动时预热 lazy-load 的重资源，避免**第一次摄入**替所有用户付这笔账。

    真机实测（i5-9300H + GTX 1650 4GB，8 页 PDF / 16 个块）：

    | 阶段 | 耗时 | 说明 |
    |---|---|---|
    | `default_counter` 加载 bge-m3 tokenizer | **21.3 s** | 首请求才触发 |
    | 本地嵌入模型加载 | 8 s(CPU) / 45 s(CUDA) | 首请求才触发 |
    | 真正的切片 `chunk_document` | 0.04 s | |
    | 真正的溯源 `compute_chunk_provenance` | 0.004 s | |

    也就是说"第一次摄入特别慢"**几乎全是 lazy load**，与切片/溯源算法无关。
    实测同一次摄入：预热前 154.7s → 预热后 5.8s。

    `KF_WARMUP=0` 可关闭（单测与纯 CLI 环境不必付这笔开销）。
    """
    import time

    if os.environ.get("KF_WARMUP", "1") == "0":
        log.info("KF_WARMUP=0，跳过预热")
        return

    t0 = time.perf_counter()
    try:
        from .clients import embed_model_id, try_embed_client
        from .tokenize import default_counter

        counter = default_counter(embed_model_id())
        log.info("预热 tokenizer: %.1fs (backend=%s)", time.perf_counter() - t0, counter.backend)

        t1 = time.perf_counter()
        client = try_embed_client()
        if client is None:
            log.warning("预热嵌入：TEI 与本地嵌入均不可用（稠密通道将为空）")
        else:
            info = client.describe() if hasattr(client, "describe") else {}
            log.info(
                "预热嵌入: %.1fs (backend=%s device=%s reason=%s)",
                time.perf_counter() - t1, info.get("backend"),
                info.get("device"), info.get("device_reason"),
            )
    except Exception as exc:  # 预热失败不影响服务可用性
        log.warning("预热失败（不影响摄入可用性，只影响首请求耗时）: %s", exc)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warmup()
    yield


app = FastAPI(title="Knowledge Fabric · Ingest", version="0.1.0", lifespan=_lifespan)

# 供本地 Web 控制台（console/index.html，file:// 打开时 Origin 为 null）直连。
# 无 Cookie / 无凭据，仅开放跨域上传与读取；部署到公网时应收紧为白名单。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestDirRequest(BaseModel):
    """目录批量摄入请求。`dir` 是**服务端本机的路径**（不是上传的文件）。

    安全提示：该端点让调用方指定服务端要读取的目录，**只适合本机 / 受信任内网**。
    部署到公网前应先加访问控制或下线该端点（与 `allow_origins=["*"]` 同一条注意）。
    """

    dir: str
    tenant: str = "default"
    kb: str = "default"
    version: int = 1
    lang: str = ""
    doc_type: str = ""
    strategy: str = ""
    dry_run: bool = False
    enrich: bool = True
    graph: bool = False
    acl: str = ""
    recursive: bool = False
    match: str = ""


def _guard_ingest(request: Request, tenant: str) -> None:
    """摄入前置鉴权：本机控制台免令牌，跨机必须带令牌且租户一致。

    此前摄入接口**完全没有鉴权**——任何能访问 8083 的进程都能往任意租户、
    任意知识库里灌文档。作为"agent 的知识库"被外部接入后这是不可接受的。
    """
    try:
        auth.require_ingest_tenant(request.headers, tenant,
                                   request.client.host if request.client else None)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@app.get("/health")
def health() -> Dict[str, Any]:
    """依赖是否配置：全为 false 时仍可 `dry_run` 摄入（零外部依赖）。

    另外如实报告**稠密嵌入实际落在哪个设备**：多服务共卡时（本机 4GB 装不下两份
    bge-m3）进程间仲裁会让后启动的服务回落 CPU，若 health 不说，运维会以为配置错了。
    摄入侧此前没有这个字段，排查"摄入为什么慢"时缺了最关键的一条线索。
    """
    from .clients import embed_model_id, local_embed_status

    deps = {
        "qdrant": bool(qdrant_url()),
        "postgres": bool(database_url()),
        "tei_embed": bool(tei_embed_url()),
    }
    info: Dict[str, Any] = {
        "status": "ok",
        "service": "kf-ingest",
        # 受支持的文档格式（CLI / 目录遍历 / 上传共用同一份判定）
        "formats": sorted(SUPPORTED_SUFFIXES),
        "deps": deps,
        "embed": local_embed_status(embed_model_id()),
    }
    return info


@app.post("/v1/ingest")
def ingest(
    request: Request,
    file: UploadFile = File(..., description="待摄入的文档（PDF / Markdown）"),
    tenant: str = Form("default"),
    kb: str = Form("default"),
    doc_id: str = Form(""),
    version: int = Form(1),
    lang: str = Form(""),
    doc_type: str = Form(""),
    strategy: str = Form(""),
    source_uri: str = Form(""),
    dry_run: bool = Form(False),
    enrich: bool = Form(True),
    graph: bool = Form(False),
    acl: str = Form(""),
) -> Dict[str, Any]:
    """上传即摄入：解析 → 清洗 → 切片 → 增强 → 嵌入 → 索引。

    返回与 CLI `--json` 同口径的摘要。`dry_run=true` 时只跑到切片，
    不触网、不落库，用于上传前预览切片效果。
    """
    _guard_ingest(request, tenant)
    filename = file.filename or "upload.pdf"
    suffix = os.path.splitext(filename)[1] or ".pdf"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        resolved_doc_id = doc_id or os.path.splitext(os.path.basename(filename))[0]
        ctx = IngestContext(
            tenant_id=tenant,
            kb_id=kb,
            doc_id=resolved_doc_id,
            source_uri=source_uri or filename,
            source_path=tmp_path,
            version=int(version or 1),
            lang=lang,
            doc_type=doc_type,
            graph=bool(graph),
        )

        qdrant = None
        pg = None
        if not dry_run:
            qdrant = connect_qdrant(qdrant_url())
            pg = connect_pg(database_url())

        collection = default_collection(tenant, kb)
        try:
            run_ingest(
                ctx,
                strategy=strategy or None,
                dry_run=bool(dry_run),
                qdrant_client=qdrant,
                pg_conn=pg,
                collection=collection,
                dim=embed_dim(),
                with_enrich=bool(enrich),
                graph=bool(graph),
                acl_spec=acl,
            )
        finally:
            _close(pg)
            _close(qdrant)

        summary = ingestion_summary(ctx, collection=collection, dry_run=bool(dry_run))
        # 如实标注哪一侧真写成功（未配置/连不上时对应计数为 0，不报错）
        summary["ok"] = not ctx.errors
        summary["notes"] = _index_notes(ctx)
        return summary
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/v1/ingest-dir")
def ingest_dir(req: IngestDirRequest, request: Request) -> Dict[str, Any]:
    """**目录批量摄入**：遍历服务端本机目录，逐个摄入其中的 PDF / Markdown。

    与 `POST /v1/ingest` 走同一条流水线；差异只有两点，都是为了「一个公司有很多文档」：
      - `doc_id` 由**文件名**派生（请求里不传），避免多个文档互相覆盖；
      - 单个文档失败**不中断**其余文档，失败数计入 `failed` 并体现在 `documents[].ok`。

    不支持的格式**如实列出**在 `skipped_unsupported`，不静默跳过。
    `dry_run=true` 时零外部依赖，只跑切片预览。
    """
    target = os.path.abspath(os.path.expanduser(req.dir))
    if not os.path.isdir(target):
        raise HTTPException(status_code=400, detail=f"目录不存在或不可读：{target}")

    _guard_ingest(request, req.tenant)

    todo, skipped = discover_documents(target, recursive=req.recursive, match=req.match or None)
    collection = default_collection(req.tenant, req.kb)
    if not todo:
        return {
            "dir": target,
            "tenant": req.tenant,
            "kb": req.kb,
            "collection": collection,
            "dry_run": bool(req.dry_run),
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_unsupported": skipped,
            "documents": [],
            "notes": [f"目录内没有受支持的文档（支持 {', '.join(sorted(SUPPORTED_SUFFIXES))}）"],
        }

    qdrant = None
    pg = None
    if not req.dry_run:
        qdrant = connect_qdrant(qdrant_url())
        pg = connect_pg(database_url())

    documents: List[Dict[str, Any]] = []
    failures = 0
    try:
        for path in todo:
            ctx = IngestContext(
                tenant_id=req.tenant,
                kb_id=req.kb,
                doc_id=os.path.splitext(os.path.basename(path))[0],
                source_uri=path,
                source_path=path,
                version=int(req.version or 1),
                lang=req.lang,
                doc_type=req.doc_type,
                graph=bool(req.graph),
            )
            run_ingest(
                ctx,
                strategy=req.strategy or None,
                dry_run=bool(req.dry_run),
                qdrant_client=qdrant,
                pg_conn=pg,
                collection=collection,
                dim=embed_dim(),
                with_enrich=bool(req.enrich),
                graph=bool(req.graph),
                acl_spec=req.acl,
            )
            summary = ingestion_summary(ctx, collection=collection, dry_run=bool(req.dry_run))
            summary["source_path"] = path
            summary["ok"] = not ctx.errors
            summary["notes"] = _index_notes(ctx)
            documents.append(summary)
            if ctx.errors:
                failures += 1
    finally:
        _close(pg)
        _close(qdrant)

    return {
        "dir": target,
        "tenant": req.tenant,
        "kb": req.kb,
        "collection": collection,
        "dry_run": bool(req.dry_run),
        "recursive": bool(req.recursive),
        "processed": len(todo),
        "succeeded": len(todo) - failures,
        "failed": failures,
        "skipped_unsupported": skipped,
        "documents": documents,
        "notes": [],
    }


@app.get("/v1/documents")
def list_documents(
    request: Request,
    tenant: str = "default",
    kb: str = "",
    limit: int = 50,
) -> Dict[str, Any]:
    """列出该租户（可按库过滤）的文档与当前版本活跃块数。

    Postgres 不可用时返回空列表 + note（控制台因此仍能正常渲染，只是列表为空）。
    """
    _guard_ingest(request, tenant)
    conn = connect_pg(database_url())
    if conn is None:
        return {"documents": [], "notes": ["Postgres 未配置或不可用，无法列出文档"]}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT d.doc_id, d.kb_id, d.title, d.source_uri, d.lang, d.doc_type,"
                "       d.current_version, d.updated_at, COALESCE(c.active_chunks, 0)"
                "  FROM documents d"
                "  LEFT JOIN ("
                "      SELECT doc_id, version, count(*) AS active_chunks"
                "        FROM chunks WHERE status = 'active' GROUP BY doc_id, version"
                "  ) c ON c.doc_id = d.doc_id AND c.version = d.current_version"
                " WHERE d.tenant_id = %s AND (%s = '' OR d.kb_id = %s)"
                " ORDER BY d.updated_at DESC LIMIT %s",
                (tenant, kb, kb, max(1, min(int(limit or 50), 500))),
            )
            rows = cur.fetchall()
    except Exception as exc:  # 查询失败同样只记 note
        return {"documents": [], "notes": [f"查询文档失败: {exc}"]}
    finally:
        _close(conn)

    documents: List[Dict[str, Any]] = [
        {
            "doc_id": r[0],
            "kb": r[1],
            "title": r[2],
            "source_uri": r[3],
            "lang": r[4],
            "doc_type": r[5],
            "current_version": r[6],
            "updated_at": r[7].isoformat() if r[7] else None,
            "active_chunks": int(r[8] or 0),
        }
        for r in rows
    ]
    return {"documents": documents, "notes": []}


def _index_notes(ctx: IngestContext) -> List[str]:
    # 仅上报索引侧 note；OpenSearch 的降级 note 保留在 stages["opensearch"]，
    # 避免在「零外部依赖」的默认模式下污染 summary.notes（既有契约：无依赖=无 note）。
    notes: List[str] = []
    index_stats = (ctx.stage_stats or {}).get("index") or {}
    if isinstance(index_stats, dict):
        notes.extend(str(n) for n in (index_stats.get("notes") or []))
    return notes


def _close(obj: Optional[Any]) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
