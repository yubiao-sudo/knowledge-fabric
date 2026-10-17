"""Agent 接入认证：一次性「接入码」→ 长期令牌。

## 为什么是这套设计

底座此前 **完全没有认证**——任何能访问 8082/8083 的进程都能查询与摄入。
本地开发无所谓，但作为"agent 的知识库"被别的程序接入就不行了：
agent 可能跑在另一台机器上，同一个底座也可能被多个 agent / 多个租户共用。

两级凭证：

| 层级 | 形态 | 时效 | 用途 |
|---|---|---|---|
| 接入码 `pairing_codes` | `KF-7K3M-9QX2`（8 位，去掉易混的 I/O/0/1） | 默认 10 分钟、一次性 | 人在控制台一键生成，抄给 agent |
| 长期令牌 `api_keys` | `kf_<token_urlsafe(32)>` | 长期，可吊销 | agent 持久化保存 |

**为什么不只发长期令牌**：让人手抄 43 字符的密钥极易出错；抄 8 位随机码、
程序自己换长期令牌，是最省事又最安全的路径。

安全要点：
* 库里**只存 sha256**，明文令牌仅在兑换那一刻返回一次；
* 创建接入码与列举/吊销令牌**只允许 loopback**（127.0.0.1 / ::1）——
  即"只有本机控制台能签发"，远端 agent 只能兑换，不能签发；
* 吊销是软删（`revoked_at`），保留审计痕迹。

## 强制模式

`KF_REQUIRE_AUTH`：
* `auto`（默认）—— **库里存在任何未吊销令牌就强制**。这样"一旦签发过，
  就自动进入受保护状态"，不需要额外记得开开关；
* `on` —— 永远强制（哪怕还没签发令牌，此时所有请求都会 401，用于强制先配对）；
* `off` —— 永远不校验（本地开发/单测）。
"""

from __future__ import annotations

import hashlib
import os
import secrets
import string
import time
from typing import Any, Dict, List, Optional

# 去掉易混字符：I/O/0/1 在小屏幕上极易抄错
CODE_ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits
                        if c not in "IO01")
CODE_TTL_DEFAULT = 600          # 秒
TOKEN_PREFIX = "kf_"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

# 鉴权开关的缓存（避免每个请求都查库）
_MODE_CACHE: Dict[str, Any] = {"at": 0.0, "value": False}
_MODE_TTL = 5.0


def db_dsn() -> str:
    return (os.environ.get("DATABASE_APP_URL")
            or os.environ.get("DATABASE_URL") or "").strip()


def auth_mode() -> str:
    return (os.environ.get("KF_REQUIRE_AUTH") or "auto").strip().lower()


def _connect():
    import psycopg

    return psycopg.connect(db_dsn(), connect_timeout=5)


def is_provisioned() -> bool:
    """是否**曾经**签发过令牌（带短缓存）。

    🔴 这里查的是**任意一行**，不筛 `revoked_at IS NULL`。这是踩过的坑：
    如果只看"未吊销的令牌"，那么吊销最后一个令牌后就会判定"从未配置过"，
    `auto` 模式随之关闭鉴权——**吊销反而把门打开了**（真机复现：吊销后
    同一个旧令牌再请求返回 200）。改成看是否存在过令牌，就等于给鉴权留了
    一个**持久化的启用标记**（我们只做软删，行永远保留）。

    随之而来的问题是"吊销完最后一个令牌会把自己锁在外面"——这是刻意的：
    恢复方式是回到本机控制台重新生成接入码（签发只允许 loopback），
    或显式设 `KF_REQUIRE_AUTH=off`。**安全默认应当是关着门而不是开着门。**
    """
    now = time.time()
    if now - float(_MODE_CACHE.get("at") or 0) < _MODE_TTL:
        return bool(_MODE_CACHE.get("value"))
    value = False
    if db_dsn():
        try:
            with _connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM api_keys LIMIT 1")
                value = cur.fetchone() is not None
        except Exception:
            # 连不上库时**不要**因此放开鉴权：宁可拒绝（fail-closed）。
            value = True
    _MODE_CACHE.update({"at": now, "value": value})
    return value


def invalidate_cache() -> None:
    _MODE_CACHE["at"] = 0.0


def auth_required() -> bool:
    mode = auth_mode()
    if mode in ("off", "0", "false", "no"):
        return False
    if mode in ("on", "1", "true", "yes"):
        return True
    return is_provisioned()       # auto


def is_loopback(host: Optional[str]) -> bool:
    """只有本机控制台可以签发接入码/管理令牌。"""
    if not host:
        return False
    h = host.strip().lower()
    if h.startswith("::ffff:"):
        h = h[7:]
    return h in LOOPBACK_HOSTS


# ------------------------------------------------------------------ 令牌
def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def verify_token(raw: str) -> Optional[Dict[str, Any]]:
    """校验令牌，返回 `{tenant_id, key_id}`；无效返回 None。"""
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
            # last_used_at 按需更新，失败不影响鉴权
            try:
                cur.execute("UPDATE api_keys SET last_used_at = now() WHERE id = %s",
                            (key_id,))
                conn.commit()
            except Exception:
                conn.rollback()
            return {"key_id": key_id, "tenant_id": tenant_id}
    except Exception:
        return None


# ------------------------------------------------------------------ 接入码
def _new_code() -> str:
    def block(n: int = 4) -> str:
        return "".join(secrets.choice(CODE_ALPHABET) for _ in range(n))

    return f"KF-{block()}-{block()}"


def create_pairing(tenant_id: str, ttl_sec: int = 0, name: str = "") -> Dict[str, Any]:
    """生成一次性接入码。调用方必须已确认是 loopback。"""
    tenant_id = (tenant_id or "").strip()
    if not tenant_id:
        raise ValueError("tenant_id 不能为空")
    ttl = max(60, int(ttl_sec or CODE_TTL_DEFAULT))
    code = _new_code()
    with _connect() as conn, conn.cursor() as cur:
        # 极小概率撞码：重试几次
        for _ in range(5):
            cur.execute("SELECT 1 FROM pairing_codes WHERE code = %s", (code,))
            if cur.fetchone() is None:
                break
            code = _new_code()
        cur.execute(
            "INSERT INTO pairing_codes(code, tenant_id, expires_at) "
            "VALUES (%s, %s, now() + make_interval(secs => %s))",
            (code, tenant_id, ttl),
        )
        cur.execute(
            "SELECT expires_at FROM pairing_codes WHERE code = %s", (code,))
        expires_at = cur.fetchone()[0]
        conn.commit()
    invalidate_cache()
    return {"code": code, "tenant_id": tenant_id, "ttl_sec": ttl,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "hint": "把该接入码填进 agent 的「接入码」输入框即可自动换取令牌"}


def exchange_pairing(code: str, name: str = "") -> Optional[Dict[str, Any]]:
    """用接入码换长期令牌。码只能用一次、且必须未过期。"""
    code = (code or "").strip().upper()
    if not code.startswith("KF-") or not db_dsn():
        return None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT tenant_id, expires_at, used_at FROM pairing_codes "
            "WHERE code = %s FOR UPDATE",
            (code,),
        )
        row = cur.fetchone()
        if not row:
            return None
        tenant_id, expires_at, used_at = row
        if used_at is not None:
            return {"error": "该接入码已被使用，请重新生成"}
        cur.execute("SELECT now()")
        if expires_at is not None and cur.fetchone()[0] > expires_at:
            return {"error": "该接入码已过期，请重新生成"}

        raw = new_token()
        key_id = "key_" + secrets.token_hex(8)
        cur.execute(
            "INSERT INTO api_keys(id, tenant_id, name, token_hash, prefix) "
            "VALUES (%s, %s, %s, %s, %s)",
            (key_id, tenant_id, name or "agent", hash_token(raw), raw[:8]),
        )
        cur.execute(
            "UPDATE pairing_codes SET used_at = now(), key_id = %s WHERE code = %s",
            (key_id, code),
        )
        conn.commit()
    invalidate_cache()
    return {"token": raw, "tenant_id": tenant_id, "key_id": key_id,
            "note": "令牌只在此刻返回一次，请妥善保存"}


# ------------------------------------------------------------------ 管理
def list_keys() -> List[Dict[str, Any]]:
    """列出令牌（**只返回前缀**，绝不返回明文或哈希原文）。"""
    if not db_dsn():
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, tenant_id, name, prefix, created_at, last_used_at, revoked_at "
                "FROM api_keys ORDER BY created_at DESC LIMIT 100"
            )
            return [
                {"id": r[0], "tenant_id": r[1], "name": r[2], "prefix": r[3],
                 "created_at": r[4].isoformat() if r[4] else None,
                 "last_used_at": r[5].isoformat() if r[5] else None,
                 "revoked": r[6] is not None}
                for r in cur.fetchall()
            ]
    except Exception:
        return []


def revoke_key(key_id: str) -> bool:
    key_id = (key_id or "").strip()
    if not key_id or not db_dsn():
        return False
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE api_keys SET revoked_at = now() "
                "WHERE id = %s AND revoked_at IS NULL",
                (key_id,),
            )
            changed = cur.rowcount
            conn.commit()
        invalidate_cache()
        return changed > 0
    except Exception:
        return False


def status() -> Dict[str, Any]:
    return {
        "mode": auth_mode(),
        "required": auth_required(),
        "provisioned": is_provisioned(),
        "note": "auto 模式：签发过令牌（含已吊销）就强制鉴权；吊销最后一个不会关闭鉴权",
    }
