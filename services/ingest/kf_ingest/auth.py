"""摄入侧鉴权：校验底座签发的接入令牌（只读子集）。

## 为什么要单独一份

底座的两级凭证（接入码 `pairing_codes` + 长期令牌 `api_keys`）实现与**签发**
都在检索服务（`services/retrieve/kf_retrieve/auth.py`）——那才是"发证机关"。
摄入服务只需要**验票**，不需要签发，所以这里只保留必要的一半：

* `bearer()`      从请求头取令牌（兼容 `Authorization: Bearer` 与 `X-KF-Token`）
* `verify_token()` 查 `api_keys`（只存 sha256，比对哈希）
* `auth_required()` 与检索侧同一套 `KF_REQUIRE_AUTH` 语义

> 为什么不跨服务复用同一个模块：两个服务是**独立部署的 Python 包**
> （各自的 PYTHONPATH、各自的容器），跨包 import 会把它们耦合成必须
> 同机同目录。宁可保留这一小段镜像逻辑，并在注释里指明"发证机关"在哪。

## 安全默认

与检索侧完全一致：`auto` 模式下**只要签发过任何令牌（含已吊销）就强制**；
连不上库时 **fail-closed**（当作已配置）。理由见检索侧 auth.py 的长注释——
简言之"吊销不该把门打开"，"数据库抖一下不该把门打开"。
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, Optional

TOKEN_PREFIX = "kf_"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

_CACHE: Dict[str, Any] = {"at": 0.0, "value": False}
_CACHE_TTL = 5.0


def db_dsn() -> str:
    return (os.environ.get("DATABASE_APP_URL")
            or os.environ.get("DATABASE_URL") or "").strip()


def auth_mode() -> str:
    return (os.environ.get("KF_REQUIRE_AUTH") or "auto").strip().lower()


def _connect():
    import psycopg

    return psycopg.connect(db_dsn(), connect_timeout=5)


def is_provisioned() -> bool:
    """是否曾经签发过令牌（含已吊销）——即"门已经装上过"。"""
    now = time.time()
    if now - float(_CACHE.get("at") or 0) < _CACHE_TTL:
        return bool(_CACHE.get("value"))
    value = False
    if db_dsn():
        try:
            with _connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM api_keys LIMIT 1")
                value = cur.fetchone() is not None
        except Exception:
            value = True          # fail-closed
    _CACHE.update({"at": now, "value": value})
    return value


def auth_required() -> bool:
    mode = auth_mode()
    if mode in ("off", "0", "false", "no"):
        return False
    if mode in ("on", "1", "true", "yes"):
        return True
    return is_provisioned()


def invalidate_cache() -> None:
    _CACHE["at"] = 0.0


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_token(raw: str) -> Optional[Dict[str, Any]]:
    """校验令牌，返回 `{key_id, tenant_id}`；无效返回 None。"""
    raw = (raw or "").strip()
    if not raw.startswith(TOKEN_PREFIX) or not db_dsn():
        return None
    digest = hash_token(raw)
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, tenant_id FROM api_keys "
                "WHERE token_hash = %s AND revoked_at IS NULL",
                (digest,),
            )
            row = cur.fetchone()
            if not row:
                return None
            key_id, tenant_id = row
            try:
                cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id = %s",
                            (key_id,))
                conn.commit()
            except Exception:
                conn.rollback()
            return {"key_id": key_id, "tenant_id": tenant_id}
    except Exception:
        return None


def bearer_from(headers: Any) -> str:
    raw = headers.get("authorization") or ""
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return (headers.get("x-kf-token") or "").strip()


def is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    h = host.strip().lower()
    if h.startswith("::ffff:"):
        h = h[7:]
    return h in LOOPBACK_HOSTS


def require_ingest_tenant(headers: Any, tenant: str, client_host: Optional[str]) -> None:
    """摄入前置检查：需要时验票，且**令牌租户必须与请求租户一致**。

    本机控制台上传免令牌（它就是发证机关，且本机进程本来就能连数据库）；
    跨机一律要令牌，防止别人往你的库里灌文档。
    """
    if not auth_required():
        return
    info = verify_token(bearer_from(headers))
    if info is None:
        if is_loopback(client_host):
            return                      # 本机控制台
        raise PermissionError("跨机摄入需要接入令牌（本机控制台免令牌）")
    if (tenant or "").strip() and info["tenant_id"] != tenant.strip():
        raise PermissionError(
            f"令牌属于租户 {info['tenant_id']}，不能写入 {tenant}"
        )
