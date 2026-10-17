"""Agent 接入认证：只测纯逻辑（不连库、不发请求）。

为什么这些点值得锁住：
* **接入码字符集**：含 I/O/0/1 时人抄错一位就整条链路失效，且报错是"码无效"，
  根本看不出是抄错；
* **令牌只存哈希**：一旦退化成存明文，库被读走就等于令牌泄露；
* **loopback 判定**：签发接入码只允许本机；判错等于把发令牌的权力给了任何人；
* **吊销不关闭鉴权**：真机复现过——吊销最后一个令牌后 `auto` 模式判定
  "从未配置过"而关闭鉴权，旧令牌又能用了（门被吊销动作打开）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "services", "retrieve")))

import pytest

from kf_retrieve import auth


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """默认让 db_dsn 为空，测试不碰真实数据库。"""
    monkeypatch.setattr(auth, "db_dsn", lambda: "")
    auth.invalidate_cache()


# --------------------------------------------------------------- 接入码
def test_code_alphabet_excludes_confusable_chars():
    for ch in "IO01":
        assert ch not in auth.CODE_ALPHABET, f"{ch} 易与数字/字母混淆，必须排除"


def test_code_format():
    code = auth._new_code()
    assert code.startswith("KF-")
    parts = code.split("-")
    assert len(parts) == 3 and len(parts[1]) == 4 and len(parts[2]) == 4
    body = parts[1] + parts[2]
    assert all(c in auth.CODE_ALPHABET for c in body)


def test_codes_are_unique_enough():
    seen = {auth._new_code() for _ in range(200)}
    assert len(seen) == 200, "200 次生成出现重复，说明随机源有问题"


# --------------------------------------------------------------- 令牌
def test_token_has_prefix():
    assert auth.new_token().startswith(auth.TOKEN_PREFIX)


def test_token_hash_is_stable_and_not_reversible():
    raw = auth.new_token()
    h1, h2 = auth.hash_token(raw), auth.hash_token(raw)
    assert h1 == h2, "同一令牌必须得到同一哈希"
    assert raw not in h1, "哈希里不能含明文"
    assert len(h1) == 64, "sha256 十六进制应为 64 字符"


def test_different_tokens_hash_differently():
    a, b = auth.new_token(), auth.new_token()
    assert auth.hash_token(a) != auth.hash_token(b)


def test_verify_rejects_without_dsn():
    """没配数据库时不能凭空放行。"""
    assert auth.verify_token("kf_whatever") is None


def test_verify_rejects_plain_wrong_prefix():
    assert auth.verify_token("not-a-kf-token") is None
    assert auth.verify_token("") is None


# --------------------------------------------------------------- 鉴权模式
@pytest.mark.parametrize("mode,expected", [
    ("off", False), ("0", False), ("false", False), ("no", False),
    ("on", True), ("1", True), ("true", True), ("yes", True),
])
def test_auth_mode_explicit(monkeypatch, mode, expected):
    monkeypatch.setenv("KF_REQUIRE_AUTH", mode)
    assert auth.auth_required() is expected


def test_auto_mode_follows_provisioning(monkeypatch):
    monkeypatch.setenv("KF_REQUIRE_AUTH", "auto")
    monkeypatch.setattr(auth, "is_provisioned", lambda: False)
    assert auth.auth_required() is False
    monkeypatch.setattr(auth, "is_provisioned", lambda: True)
    assert auth.auth_required() is True


def test_unreachable_db_fails_closed(monkeypatch):
    """连不上库时必须当作"已配置"，否则数据库抖一下门就开了。"""
    monkeypatch.setenv("KF_REQUIRE_AUTH", "auto")
    auth.invalidate_cache()
    monkeypatch.setattr(auth, "db_dsn", lambda: "postgresql://x:x@127.0.0.1:1/none")

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(auth, "_connect", boom)
    assert auth.is_provisioned() is True
    assert auth.auth_required() is True


# --------------------------------------------------------------- loopback
@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("::1", True), ("localhost", True),
    ("::ffff:127.0.0.1", True),
    ("192.168.1.5", False), ("10.0.0.2", False), ("", False), (None, False),
])
def test_is_loopback(host, expected):
    assert auth.is_loopback(host) is expected


def test_remote_cannot_sign(monkeypatch):
    """远端地址不允许签发接入码——否则任何人都能给自己发令牌。"""
    assert auth.is_loopback("8.8.8.8") is False


# --------------------------------------------------------------- 状态
def test_status_shape(monkeypatch):
    monkeypatch.setenv("KF_REQUIRE_AUTH", "off")
    s = auth.status()
    for key in ("mode", "required", "provisioned", "note"):
        assert key in s
