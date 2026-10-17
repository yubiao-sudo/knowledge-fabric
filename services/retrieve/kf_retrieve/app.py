"""FastAPI 应用：`GET /health`、`GET /v1/profiles`、`POST /v1/search`。

服务端口用环境变量 `RETRIEVAL_PORT`（默认 8080）。后端在首次请求时惰性构建
（`QDRANT_URL` 可用则用 Qdrant，否则 MemoryBackend 并记 note）；测试可用
`set_backend()` 注入 MemoryBackend。任何上游不可用都只降级，不 500。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .backends import TEIEmbedClient, build_backend, local_embed_status, try_embed_client
from . import adaptive
from .config import load_retrieval_config
from .profile import profiles_summary
from .rerank import local_rerank_status, rerank_url
from .service import SearchRequest, run_search

log = logging.getLogger("kf.retrieve.http")

PORT = int(os.environ.get("RETRIEVAL_PORT", "8080"))

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warmup()
    yield


app = FastAPI(title="Knowledge Fabric · Retrieval", version="0.1.0", lifespan=_lifespan)

# 供本地 Web 控制台（console/index.html，file:// 打开时 Origin 为 null）直连。
# 无 Cookie / 无凭据，仅开放跨域读取；部署到公网时应收紧为白名单。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: Dict[str, Any] = {"backend": None, "notes": None}


def _warmup() -> None:
    """启动时预热稠密嵌入，避免**第一次检索**替所有用户付模型加载的钱。

    真机实测：本地 bge-m3 加载（含搬运到设备）在 CPU 上约 8s、在 CUDA 上约 45s
    （首次含显存分配与 kernel 编译）。这笔开销属于进程级一次性成本，
    放在启动时付掉，检索请求就能保持线性。

    `KF_WARMUP=0` 可关闭（单测用）。
    """
    import time

    if os.environ.get("KF_WARMUP", "1") == "0":
        log.info("KF_WARMUP=0，跳过预热")
        return
    t0 = time.perf_counter()
    try:
        client = try_embed_client()
        if client is None:
            log.warning("预热嵌入：TEI 与本地嵌入均不可用（稠密通道将为空，dense/hybrid 档位会返回 0 证据）")
            return
        info = client.describe() if hasattr(client, "describe") else {}
        log.info("预热嵌入完成: %.1fs (backend=%s device=%s reason=%s)",
                 time.perf_counter() - t0, info.get("backend"),
                 info.get("device"), info.get("device_reason"))
    except Exception as exc:
        log.warning("预热失败（不影响检索可用性，只影响首请求耗时）: %s", exc)

    # 重排模型同样在启动时加载：cross-encoder 首次加载在真机约 **79 秒**
    # （2.1GB 权重 + fp16 搬运）。留到首次查询的话，第一个用户会以为服务挂了。
    t1 = time.perf_counter()
    try:
        from .rerank import try_local_reranker

        reranker = try_local_reranker()
        if reranker is None:
            log.info("预热重排：本地重排不可用")
        else:
            rinfo = reranker.describe()
            log.info("预热重排完成: %.1fs (device=%s dtype=%s max_len=%s)",
                     time.perf_counter() - t1, rinfo.get("device"),
                     rinfo.get("dtype"), rinfo.get("max_length"))
    except Exception as exc:
        log.warning("预热重排失败（不影响检索可用性）: %s", exc)


def get_backend() -> Any:
    """惰性构建召回后端（进程内单例）。"""
    if _state["backend"] is None:
        backend, notes = build_backend(embed_client=try_embed_client())
        _state["backend"] = backend
        _state["notes"] = notes
    return _state["backend"]


def backend_notes() -> List[str]:
    get_backend()
    return list(_state.get("notes") or [])


def set_backend(backend: Any, notes: Optional[List[str]] = None) -> None:
    """注入后端（测试用）。"""
    _state["backend"] = backend
    _state["notes"] = list(notes or [])


def reset_backend() -> None:
    _state["backend"] = None
    _state["notes"] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康检查。

    `index.collection` 是**后端默认集合**（由环境变量决定），而真实查询按
    `tenant/kb` 派生集合名（如 `kf_acme_demo`）——两者含义不同，运维观察时
    不可混用，因此这里两个字段都给，并显式标注命名规则。
    `reranker.url_configured` 只表示**环境变量写了 URL**，不代表端点可达；
    真实可用性由检索响应 `trace.reranked` 与 notes 体现（避免"看起来配好了"）。
    """
    backend = get_backend()
    info = backend.describe() if hasattr(backend, "describe") else {}
    url = rerank_url()
    # 稠密嵌入的**真实**可用性：TEI 与本地回退都失败时必须显式说明
    # "dense/hybrid 档位会返回 0 证据"，而不是让 health 看起来一切正常。
    embed = try_embed_client()
    if embed is None:
        embed_info: Dict[str, Any] = {
            "backend": "none",
            "ready": False,
            "note": "TEI 与本地嵌入均不可用：稠密通道为空，dense/hybrid 档位将返回 0 证据",
            "local": local_embed_status(),
        }
    else:
        described = embed.describe() if hasattr(embed, "describe") else {}
        embed_info = {
            "backend": described.get("backend")
            or ("tei" if isinstance(embed, TEIEmbedClient) else "unknown"),
            "ready": True,
            "model": described.get("model", ""),
            "device": described.get("device", ""),
            # 为什么选了这个设备（例如"GPU 已被 pid=X 占用，回落 cpu"）——
            # 不暴露它，运维看到 device=cpu 会以为是配置错误。
            "device_reason": described.get("device_reason", ""),
        }
    return {
        "status": "ok",
        "service": "kf-retrieve",
        "index": {
            "configured": info.get("backend") == "qdrant",
            "collection": str(info.get("collection") or ""),
            "collection_note": "该值为后端默认集合；真实查询按 tenant/kb 派生 kf_<tenant>_<kb>",
        },
        "embed": embed_info,
        "reranker": {
            "url_configured": bool(url),
            "url": url or "",
            # 本地 cross-encoder 的实际状态：是否就绪、落在哪个设备、用什么精度。
            # 与 embed 段同理——不报告的话，运维只能靠"重排有没有生效"去猜。
            "local": local_rerank_status(),
            # 按显存自适应出来的重排窗口。窗口大小直接决定延迟与排序质量，
            # 也让"为什么这台机器排 4 个、那台排 20 个"有据可查。
            "adaptive": adaptive.status(),
            "note": "url_configured 仅表示环境变量已配置；端点可达性见检索响应的 trace.reranked",
        },
    }


@app.get("/v1/profiles")
def profiles() -> Dict[str, Any]:
    return profiles_summary(load_retrieval_config())


@app.post("/v1/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    backend = get_backend()
    return run_search(req, backend, cfg=load_retrieval_config(), backend_notes=backend_notes())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
