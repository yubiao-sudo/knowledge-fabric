-- =============================================================================
-- 006 · 知识库目录（跨租户只读聚合）
--
-- ## 为什么需要数据库对象而不是纯 SQL 查询
--
-- `chunk_vectors` 在 004 里启用了 RLS，策略是
--     USING (tenant_id = current_setting('app.tenant_id', true))
-- 服务角色 `kf_app` 不是超级用户，**不设置 app.tenant_id 就一行都看不到**
-- （实测：以 kf_app 身份查 chunk_vectors 返回 0 行，JOIN 后整个目录为空）。
--
-- 而"列出有哪些知识库"这个动作天然是**跨租户**的：控制台要看到全部，
-- agent 带令牌时只能看到自己租户。单靠 set_config 只能解决后者。
--
-- ## 为什么用 SECURITY DEFINER 函数，而不是给 kf_app 加 BYPASSRLS
--
-- BYPASSRLS 是**整库级**的：给了它，任何一条 SQL 都可能绕过租户隔离，代价太大。
-- 这里把绕过范围收窄到**一个只读、只返回聚合计数的函数**：
--   · 入参只有 tenant（可空）；
--   · 返回的只有 tenant/kb 名与三个计数 + 时间戳，**不含任何内容字段**；
--   · 不接收任意 SQL，无法被注入放大。
--
-- ## 租户隔离由谁保证
--
-- 函数本身不做鉴权（它是个数据层工具）。**过滤责任在 HTTP 层**：
-- `services/retrieve/kf_retrieve/app.py` 的 `/v1/kbs` 在带令牌时**强制**把
-- tenant 覆写成令牌所属租户，调用方无法指定别的租户。跨租户列表只在本机
-- 控制台（未签发令牌 / 鉴权关闭）时才可能出现。
-- =============================================================================

CREATE OR REPLACE FUNCTION kf_kb_catalog(p_tenant TEXT DEFAULT NULL)
RETURNS TABLE (
    tenant_id          TEXT,
    kb_id              TEXT,
    documents          BIGINT,
    chunks             BIGINT,
    chunks_with_dense  BIGINT,
    last_ingested_at   TIMESTAMPTZ
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT c.tenant_id,
           c.kb_id,
           count(DISTINCT c.doc_id)                                  AS documents,
           count(*)                                                  AS chunks,
           count(*) FILTER (WHERE cv.dense_point_id IS NOT NULL)      AS chunks_with_dense,
           max(c.created_at)                                         AS last_ingested_at
      FROM chunks c
      LEFT JOIN chunk_vectors cv
             ON cv.tenant_id = c.tenant_id AND cv.chunk_id = c.chunk_id
     WHERE c.status = 'active'
       AND (p_tenant IS NULL OR c.tenant_id = p_tenant)
     GROUP BY c.tenant_id, c.kb_id
     ORDER BY count(*) DESC;
$$;

-- 单个知识库的文档明细（同样跨 RLS，只返回计数与 id）
CREATE OR REPLACE FUNCTION kf_kb_docs(p_tenant TEXT, p_kb TEXT)
RETURNS TABLE (
    doc_id             TEXT,
    chunks             BIGINT,
    chunks_with_dense  BIGINT,
    last_ingested_at   TIMESTAMPTZ
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT c.doc_id,
           count(*)                                             AS chunks,
           count(*) FILTER (WHERE cv.dense_point_id IS NOT NULL) AS chunks_with_dense,
           max(c.created_at)                                    AS last_ingested_at
      FROM chunks c
      LEFT JOIN chunk_vectors cv
             ON cv.tenant_id = c.tenant_id AND cv.chunk_id = c.chunk_id
     WHERE c.status = 'active'
       AND c.tenant_id = p_tenant
       AND c.kb_id = p_kb
     GROUP BY c.doc_id
     ORDER BY count(*) DESC;
$$;

-- 只给服务角色执行权，不给 PUBLIC
REVOKE ALL ON FUNCTION kf_kb_catalog(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION kf_kb_docs(TEXT, TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION kf_kb_catalog(TEXT) TO kf_app;
GRANT EXECUTE ON FUNCTION kf_kb_docs(TEXT, TEXT) TO kf_app;

COMMENT ON FUNCTION kf_kb_catalog(TEXT) IS
    '知识库目录（跨 RLS 只读聚合）。租户过滤由 HTTP 层保证，函数只返回计数不含内容';
COMMENT ON FUNCTION kf_kb_docs(TEXT, TEXT) IS
    '单个知识库的文档明细计数（跨 RLS 只读聚合）';
