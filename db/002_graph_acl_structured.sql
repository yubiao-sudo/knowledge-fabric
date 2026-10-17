-- ============================================================
-- Knowledge Fabric · 002：GraphRAG 图存储 / 结构化双轨补强 / ACL 与 RLS 落地
--
-- 设计要点：
--   * 幂等：全部 IF NOT EXISTS / DROP POLICY IF EXISTS，可重复执行。
--   * RLS「真落地」而非骨架：006 段建了独立的**非属主角色** kf_app。
--     Postgres 的行级安全对表属主默认不生效（属主可绕过 RLS），
--     所以必须让应用以非属主身份连接，策略才真正拦得住。数据库属主
--     （compose 的 kf，供摄入/运维用）保持不受影响，不影响既有链路。
--   * 图按知识库 opt-in：`knowledge_bases.graph_enabled` 已在 001 中，本文件只补
--     存储与索引；是否建图由摄入侧按该开关与 --graph 参数决定。
-- ============================================================

-- ---------------- 1) 结构化双轨补强 ----------------
-- structured_records 已在 001 建表；这里补三件必需的东西：
--   (a) tenant_id —— RLS 策略需要直接可比对的租户列（原表只有 kb_id）
--   (b) 幂等唯一键 —— 摄入重复执行不得产生重复行（同一块同一 schema 同一 SKU）
--   (c) payload —— 存原始行/单元格，支持「精确命中 → 回溯原文行」
ALTER TABLE structured_records ADD COLUMN IF NOT EXISTS tenant_id TEXT;
ALTER TABLE structured_records ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE structured_records ADD COLUMN IF NOT EXISTS doc_id TEXT;
ALTER TABLE structured_records ADD COLUMN IF NOT EXISTS version INTEGER;

UPDATE structured_records sr
   SET tenant_id = kb.tenant_id
  FROM knowledge_bases kb
 WHERE kb.kb_id = sr.kb_id AND sr.tenant_id IS NULL;

-- 幂等唯一键带 tenant_id：chunk_id 只在本租户内唯一，不带租户会让第二个租户的
-- 同名块覆盖第一个租户的结构化记录（与主键缺租户是同一类缺陷）。
-- 004 迁移会对既有库重建这个索引。
CREATE UNIQUE INDEX IF NOT EXISTS uq_struct_tenant_chunk_schema_sku
    ON structured_records (tenant_id, chunk_id, schema_id, COALESCE(sku, ''));
CREATE INDEX IF NOT EXISTS idx_struct_tenant ON structured_records(tenant_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_struct_doc ON structured_records(doc_id, version);

-- ---------------- 2) 子表补 tenant_id（RLS 策略直接比对，避免 EXISTS 子查询） ----------------
ALTER TABLE chunk_acl ADD COLUMN IF NOT EXISTS tenant_id TEXT;
ALTER TABLE chunk_provenance ADD COLUMN IF NOT EXISTS tenant_id TEXT;

UPDATE chunk_acl a SET tenant_id = c.tenant_id FROM chunks c
 WHERE c.chunk_id = a.chunk_id AND a.tenant_id IS NULL;
UPDATE chunk_provenance p SET tenant_id = c.tenant_id FROM chunks c
 WHERE c.chunk_id = p.chunk_id AND p.tenant_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_acl_tenant ON chunk_acl(tenant_id);

-- ---------------- 3) 图存储（实体 / 实体-块 / 边 / 社区摘要） ----------------
-- 轻量实体图：不引入图数据库，实体与边落 Postgres，向量仍走 Qdrant。
CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id    TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    kb_id        TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,                       -- 原始写法
    norm_name    TEXT NOT NULL,                       -- 归一化（小写/去空白/全角转半角）
    entity_type  TEXT NOT NULL DEFAULT 'term',        -- sku / product / org / term ...
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    doc_ids      JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 同一租户同一库内同归一化名只允许一个实体（否则图会被同义写法打散）。
-- 必须带 tenant_id：不同租户可复用同名 kb，若不隔离会把 A 租户的实体并进 B 租户。
CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_tenant_kb_norm
    ON graph_entities(tenant_id, kb_id, norm_name);

CREATE TABLE IF NOT EXISTS graph_entity_chunks (
    entity_id  TEXT NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    chunk_id   TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    kb_id      TEXT NOT NULL,
    tenant_id  TEXT,
    version    INTEGER,
    weight     NUMERIC(8,4) NOT NULL DEFAULT 1.0,
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_ec_chunk ON graph_entity_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_ec_kb_doc ON graph_entity_chunks(kb_id, doc_id, version);

CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id        TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    kb_id          TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
    src_entity_id  TEXT NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    dst_entity_id  TEXT NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    relation       TEXT NOT NULL DEFAULT 'co_occur',
    weight         NUMERIC(10,4) NOT NULL DEFAULT 1.0,
    chunk_ids      JSONB NOT NULL DEFAULT '[]'::jsonb,
    doc_ids        JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 无向边按 (小 id, 大 id) 规范化存储后，此唯一键即可保证「同一对实体只有一条边」。
-- 同样必须带 tenant_id，避免跨租户边被去重合并。
CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_tenant_pair
    ON graph_edges (tenant_id, kb_id, src_entity_id, dst_entity_id, relation);

CREATE TABLE IF NOT EXISTS graph_communities (
    community_id   TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    kb_id          TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
    level          INTEGER NOT NULL DEFAULT 0,
    size           INTEGER NOT NULL DEFAULT 0,
    members        JSONB NOT NULL DEFAULT '[]'::jsonb,   -- entity_id 列表
    top_entities   JSONB NOT NULL DEFAULT '[]'::jsonb,   -- 便于展示的中心实体
    summary        TEXT,
    summary_method TEXT,                                  -- llm / extractive
    chunk_ids      JSONB NOT NULL DEFAULT '[]'::jsonb,    -- 社区代表块，可回溯原文
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_comm_tenant_kb ON graph_communities(tenant_id, kb_id, level);

-- ---------------- 4) 摄入侧作业登记补字段（图/结构化阶段可观测） ----------------
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS graph_built BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS structured_records INTEGER NOT NULL DEFAULT 0;

-- ---------------- 5) 清理图与结构的旧版本残留（换版本后重建） ----------------
-- 与 chunks 的 stale 语义对称：新版本写入后旧版本的图/结构化记录不物理删除，
-- 但检索侧按 version 过滤，因此这里只建索引供过滤使用。
CREATE INDEX IF NOT EXISTS idx_entity_chunks_version ON graph_entity_chunks(kb_id, version);

-- ============================================================
-- 6) RLS 真落地
--    · 应用角色 kf_app 是**非属主**，RLS 对其真实生效；
--    · 会话开始时 SET LOCAL app.tenant_id，策略按租户隔离；
--    · 表属主 kf（摄入/运维/DDL 用）不受 RLS 限制，既有链路零影响。
-- ============================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'kf_app') THEN
        CREATE ROLE kf_app LOGIN PASSWORD 'kf_app_password';
    END IF;
EXCEPTION WHEN OTHERS THEN
    -- 无权限建角色（非超级用户）时只告警，不中断初始化：
    -- 此时可用「以属主连接 + 应用层 ACL（assemble.acl_allows）」作为兜底。
    RAISE NOTICE 'RLS 角色 kf_app 创建失败，降级为应用层 ACL: %', SQLERRM;
END
$$;

-- 表属主开启 RLS（不加 FORCE：属主仍可绕过，便于摄入与运维）
ALTER TABLE chunks            ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_bases   ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_acl         ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_provenance  ENABLE ROW LEVEL SECURITY;
ALTER TABLE structured_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_entities    ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges       ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_communities ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_entity_chunks ENABLE ROW LEVEL SECURITY;

-- 策略：会话未设置 app.tenant_id 时（current_setting 为 NULL）拒绝一切访问，
-- 这是「默认拒绝」的正确默认值；应用必须显式声明自己代表哪个租户。
DROP POLICY IF EXISTS p_chunks_tenant ON chunks;
CREATE POLICY p_chunks_tenant ON chunks
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_kb_tenant ON knowledge_bases;
CREATE POLICY p_kb_tenant ON knowledge_bases
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_acl_tenant ON chunk_acl;
CREATE POLICY p_acl_tenant ON chunk_acl
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_prov_tenant ON chunk_provenance;
CREATE POLICY p_prov_tenant ON chunk_provenance
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_struct_tenant ON structured_records;
CREATE POLICY p_struct_tenant ON structured_records
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_entity_tenant ON graph_entities;
CREATE POLICY p_entity_tenant ON graph_entities
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_edge_tenant ON graph_edges;
CREATE POLICY p_edge_tenant ON graph_edges
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_comm_tenant ON graph_communities;
CREATE POLICY p_comm_tenant ON graph_communities
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS p_ec_tenant ON graph_entity_chunks;
CREATE POLICY p_ec_tenant ON graph_entity_chunks
    USING (tenant_id = current_setting('app.tenant_id', true));

-- 授权：kf_app 需要读写（RLS 会把它限制在自己租户内）
DO $$
BEGIN
    GRANT USAGE ON SCHEMA public TO kf_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON
        chunks, knowledge_bases, chunk_acl, chunk_provenance, structured_records,
        graph_entities, graph_edges, graph_communities, graph_entity_chunks,
        documents, document_versions, tenants, chunk_vectors
        TO kf_app;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'RLS 角色授权失败，降级为应用层 ACL: %', SQLERRM;
END
$$;

-- 「一键还原」辅助函数：给定 chunk_id 取回原文区间与原始文本（含租户校验）
CREATE OR REPLACE FUNCTION kf_restore_span(p_chunk_id TEXT)
RETURNS TABLE (source_uri TEXT, page INTEGER, char_start INTEGER, char_end INTEGER, original_text TEXT)
LANGUAGE sql STABLE AS $$
    SELECT p.source_uri, p.page, p.char_start, p.char_end, p.original_text
      FROM chunk_provenance p
      JOIN chunks c ON c.chunk_id = p.chunk_id
     WHERE p.chunk_id = p_chunk_id
       AND c.tenant_id = current_setting('app.tenant_id', true);
$$;
