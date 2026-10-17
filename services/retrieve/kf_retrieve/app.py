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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import adaptive, auth, kbs
from .backends import TEIEmbedClient, build_backend, local_embed_status, try_embed_client
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
        # 接入鉴权状态：mode=auto 时"一旦签发过令牌就自动强制"。
        # 必须暴露出来——否则客户端拿到 401 会以为地址填错了。
        "auth": auth.status(),
    }


@app.get("/v1/profiles")
def profiles() -> Dict[str, Any]:
    return profiles_summary(load_retrieval_config())


# ---------------------------------------------------------------- 接入鉴权
def _bearer(request: Request) -> str:
    raw = request.headers.get("authorization") or ""
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    # 兼容 agent 侧图省事用自定义头
    return (request.headers.get("x-kf-token") or "").strip()


def require_agent(request: Request) -> Optional[Dict[str, Any]]:
    """数据类接口的统一前置：需要时校验令牌。

    `KF_REQUIRE_AUTH=off`（或尚未签发任何令牌时的 auto）直接放行，
    保证本地开发与既有测试不受影响。
    """
    if not auth.auth_required():
        return None
    info = auth.verify_token(_bearer(request))
    if info is None:
        raise HTTPException(
            status_code=401,
            detail="缺少或无效的接入令牌。请在底座控制台生成接入码，"
                   "或在 agent 里填入令牌（Authorization: Bearer <token>）",
        )
    return info


def require_agent_or_loopback(request: Request) -> Optional[Dict[str, Any]]:
    """只读目录接口用：**本机控制台免令牌，跨机一律要令牌**。

    为什么不统一要求令牌：控制台就是发令牌的地方，如果它自己也得先有令牌，
    就变成"先有鸡还是先有蛋"。而控制台只在本机打开（file:// + 127.0.0.1），
    本机进程能读到的东西本来就比一个知识库目录敏感得多。

    带令牌时仍强制收窄到令牌所属租户（见 `/v1/kbs` 的覆写逻辑）。
    """
    if not auth.auth_required():
        return None
    info = auth.verify_token(_bearer(request))
    if info is not None:
        return info
    host = request.client.host if request.client else None
    if auth.is_loopback(host):
        # 本机且未带令牌 → 管理员视角（可看全部租户）
        return None
    raise HTTPException(
        status_code=401,
        detail="跨机访问知识库目录需要接入令牌（本机控制台免令牌）",
    )


def _require_loopback(request: Request) -> None:
    """签发接入码/管理令牌只允许本机。

    远端 agent 能**兑换**接入码，但不能**签发**——否则任何人都能给自己发令牌。
    """
    host = request.client.host if request.client else None
    if not auth.is_loopback(host):
        raise HTTPException(status_code=403,
                            detail="该操作仅允许在本机（127.0.0.1）执行")


@app.post("/v1/auth/pairing")
def create_pairing(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """生成一次性接入码（仅本机）。"""
    _require_loopback(request)
    tenant = str(payload.get("tenant") or "").strip()
    if not tenant:
        raise HTTPException(status_code=400, detail="tenant 不能为空")
    try:
        ttl = int(payload.get("ttl_sec") or auth.CODE_TTL_DEFAULT)
    except (TypeError, ValueError):
        ttl = auth.CODE_TTL_DEFAULT
    try:
        return auth.create_pairing(tenant, ttl_sec=ttl,
                                   name=str(payload.get("name") or "agent"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成接入码失败: {exc}")


@app.post("/v1/auth/exchange")
def exchange_pairing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """用接入码换长期令牌。允许远端调用（码本身就是凭证）。"""
    code = str(payload.get("code") or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="code 不能为空")
    try:
        result = auth.exchange_pairing(code, name=str(payload.get("name") or "agent"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"兑换失败: {exc}")
    if result is None:
        raise HTTPException(status_code=401, detail="接入码无效")
    if result.get("error"):
        raise HTTPException(status_code=401, detail=result["error"])
    return result


@app.get("/v1/auth/keys")
def list_keys(request: Request) -> Dict[str, Any]:
    _require_loopback(request)
    return {"keys": auth.list_keys(), "status": auth.status()}


@app.post("/v1/auth/revoke")
def revoke_key(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_loopback(request)
    key_id = str(payload.get("id") or "").strip()
    if not key_id:
        raise HTTPException(status_code=400, detail="id 不能为空")
    return {"revoked": auth.revoke_key(key_id), "id": key_id}


# ---------------------------------------------------------------- 知识库目录
@app.get("/v1/kbs")
def list_kbs(request: Request, tenant: Optional[str] = None) -> Dict[str, Any]:
    """列出知识库。**agent 靠这个接口发现可用库，不必连数据库。**"""
    info = require_agent_or_loopback(request)
    # 带令牌时只允许看自己租户的库（租户隔离不能靠调用方自觉）
    if info is not None:
        tenant = info["tenant_id"]
    items = kbs.list_kbs(tenant)
    return {"count": len(items), "items": items,
            "note": "按 chunks 实时聚合；dense_ratio<1 说明含重摄入前写入的旧切片"}


@app.get("/v1/kb/{tenant}/{kb}")
def get_kb(request: Request, tenant: str, kb: str) -> Dict[str, Any]:
    info = require_agent_or_loopback(request)
    if info is not None and info["tenant_id"] != tenant:
        raise HTTPException(status_code=403, detail="令牌无权访问该租户的知识库")
    detail = kbs.get_kb(tenant, kb)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"知识库不存在: {tenant}/{kb}")
    return detail


@app.post("/v1/search")
def search(req: SearchRequest, request: Request) -> Dict[str, Any]:
    info = require_agent(request)
    if info is not None and req.tenant != info["tenant_id"]:
        raise HTTPException(status_code=403, detail="令牌无权访问该租户的知识库")
    backend = get_backend()
    return run_search(req, backend, cfg=load_retrieval_config(), backend_notes=backend_notes())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
