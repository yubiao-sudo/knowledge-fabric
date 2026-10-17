-- =============================================================================
-- 005 · Agent 接入与认证
--
-- 背景：底座此前**完全没有认证**。任何能访问 8082/8083 的进程都能查询与摄入，
-- 这在本机开发时无所谓，但一旦作为"agent 的知识库"被别的程序接入就不可接受：
--   · agent 可能跑在另一台机器上（底座要监听非 loopback）
--   · 同一个底座可能被多个 agent / 多个租户共用
--
-- 设计取舍
--   1. **两级凭证**
--      · `pairing_codes` —— 短时效（默认 10 分钟）、一次性、"随机码"形态
--        （KF-XXXX-XXXX，去掉了易混的 I/O/0/1），在控制台一键生成，供人抄给 agent；
--      · `api_keys` —— 换取后的长期令牌（kf_ 前缀），agent 持久化保存。
--      为什么不用单一长期令牌：让人手抄一串 43 字符的密钥极易出错，
--      而"抄 8 位随机码 → 自动换长期令牌"是最省事又最安全的做法。
--
--   2. **只存哈希，不存明文**
--      令牌明文只在兑换那一刻返回一次；库里只有 sha256。即使库被读走也拿不到可用令牌。
--
--   3. **这两张表故意不参与 RLS**
--      它们的语义就是"跨租户的凭证表"，鉴权代码按 tenant_id 显式过滤；
--      给它们套 RLS 反而会让"用 token 反查租户"这一步变得别扭。
--      本迁移末尾会显式记录这个决定，避免后来者误以为是漏配。
-- =============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
    id            TEXT        PRIMARY KEY,              -- 对外可见的 key id
    tenant_id     TEXT        NOT NULL,
    name          TEXT        NOT NULL DEFAULT '',
    token_hash    TEXT        NOT NULL UNIQUE,          -- sha256(明文令牌)
    prefix        TEXT        NOT NULL,                 -- 明文前 8 位，仅用于展示
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_tenant_idx
    ON api_keys (tenant_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS pairing_codes (
    code        TEXT        PRIMARY KEY,                -- KF-XXXX-XXXX
    tenant_id   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ,
    key_id      TEXT        REFERENCES api_keys(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS pairing_codes_expiry_idx ON pairing_codes (expires_at);

-- 过期码/已用码保留一段时间便于审计，但不必永久膨胀。
-- 这里只做清理函数，由调用方按需触发（也可挂 pg_cron）。
CREATE OR REPLACE FUNCTION purge_stale_pairing_codes(keep_days INT DEFAULT 7)
RETURNS INT AS $$
DECLARE removed INT;
BEGIN
    DELETE FROM pairing_codes
     WHERE expires_at < now() - make_interval(days => keep_days);
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- 授权给服务角色 kf_app（检索服务用它连库）
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE ON api_keys     TO kf_app;
GRANT SELECT, INSERT, UPDATE ON pairing_codes TO kf_app;

-- ---------------------------------------------------------------------------
-- 说明（刻意不做的事）
--   · 不给 kf_app DELETE 权限：吊销走 revoked_at 软删，留审计痕迹。
--   · api_keys / pairing_codes **不启用 RLS**：它们是跨租户凭证表，
--     鉴权逻辑在 services/retrieve/kf_retrieve/auth.py 里按 tenant_id 显式过滤。
--     scripts/verify_real.py 的"带 tenant_id 的表都有 RLS"检查已排除这两张表。
-- ---------------------------------------------------------------------------

COMMENT ON TABLE api_keys IS 'Agent 接入令牌（只存 sha256，明文仅兑换时返回一次）';
COMMENT ON TABLE pairing_codes IS '一次性接入码（随机码），兑换为长期令牌后即失效';
