-- ============================================================
-- Knowledge Fabric · 元数据 / ACL / 结构化字段 / 决策 trace
-- 多租户：Tenant → Workspace → KB → Document → Chunk
-- ACL 下沉到 chunk 级；检索侧强制『相关性 ∩ 可见性』
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 可选：若用 Postgres 做小规模向量检索才需要。默认镜像 postgres:16-alpine 不含
-- pgvector，此处**优雅降级**：装不上只记 NOTICE，不中断初始化（向量检索走 Qdrant）。
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "vector";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector 未安装，跳过 vector 扩展（向量检索走 Qdrant）: %', SQLERRM;
END
$$;

-- ---------------- 租户层级 ----------------
CREATE TABLE tenants (
    tenant_id      TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    plan           TEXT NOT NULL DEFAULT 'free',
    settings       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE workspaces (
    workspace_id   TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE knowledge_bases (
    kb_id          TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    workspace_id   TEXT REFERENCES workspaces(workspace_id) ON DELETE SET NULL,
    name           TEXT NOT NULL,
    -- 切片策略与检索档位按库配置
    chunking_strategy TEXT NOT NULL DEFAULT 'doc_structured',
    retrieval_profile TEXT NOT NULL DEFAULT 'hybrid_rerank',
    -- Graph 按库 opt-in，默认关闭（建图成本约 40x）
    graph_enabled  BOOLEAN NOT NULL DEFAULT FALSE,
    graph_config   JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- 多语言：库内语言集合，用于分语言稀疏索引
    languages      TEXT[] NOT NULL DEFAULT '{}',
    stats          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_kb_tenant ON knowledge_bases(tenant_id);

-- ---------------- 文档与版本 ----------------
CREATE TABLE documents (
    doc_id         TEXT PRIMARY KEY,
    kb_id          TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
    tenant_id      TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    title          TEXT,
    source_uri     TEXT NOT NULL,          -- 对象存储路径（MinIO）
    mime           TEXT,
    lang           TEXT,                   -- BCP-47
    doc_type       TEXT,                   -- manual / contract / regulation / code ...
    current_version INTEGER NOT NULL DEFAULT 1,
    acl            JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_doc_kb ON documents(kb_id);

CREATE TABLE document_versions (
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    version        INTEGER NOT NULL,
    content_hash   TEXT NOT NULL,
    parse_meta     JSONB NOT NULL DEFAULT '{}'::jsonb,   -- 解析器 / 版面树 / 阅读顺序
    cleaning_meta  JSONB NOT NULL DEFAULT '{}'::jsonb,   -- 清洗操作与阈值
    chunking_meta  JSONB NOT NULL DEFAULT '{}'::jsonb,   -- 切片策略与参数
    embed_model    TEXT,
    status         TEXT NOT NULL DEFAULT 'ready',        -- ingesting / ready / failed
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, version)
);

-- ---------------- 切片 ----------------
CREATE TABLE chunks (
    chunk_id       TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    kb_id          TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
    doc_id         TEXT NOT NULL,
    version        INTEGER NOT NULL,
    parent_chunk_id TEXT,
    chunk_type     TEXT NOT NULL DEFAULT 'text',
    content        TEXT NOT NULL,
    context_prefix TEXT,
    content_hash   TEXT NOT NULL,
    lang           TEXT,
    section_path   TEXT[],
    page           INTEGER,
    order_no       INTEGER,
    entities       JSONB NOT NULL DEFAULT '[]'::jsonb,
    keywords       TEXT[],
    status         TEXT NOT NULL DEFAULT 'active',       -- active / stale / deleted
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (doc_id, version) REFERENCES document_versions(doc_id, version) ON DELETE CASCADE
);
CREATE INDEX idx_chunk_kb_status ON chunks(kb_id, status);
CREATE INDEX idx_chunk_doc ON chunks(doc_id, version);
CREATE INDEX idx_chunk_hash ON chunks(content_hash);
CREATE INDEX idx_chunk_parent ON chunks(parent_chunk_id);
-- 全文兜底（真正的 BM25 走 OpenSearch 分语言索引）
CREATE INDEX idx_chunk_content_fts ON chunks USING GIN (to_tsvector('simple', content));

-- 清洗可逆：provenance 映射（引用回溯 / 点击原文 / 审计）
CREATE TABLE chunk_provenance (
    chunk_id       TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    source_uri     TEXT NOT NULL,
    page           INTEGER,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    bbox           NUMERIC[],
    cleaning_ops   TEXT[],
    original_text  TEXT                                   -- 支持一键还原原文
);

-- ACL 下沉到 chunk 级
CREATE TABLE chunk_acl (
    chunk_id       TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    principal_type TEXT NOT NULL,                          -- user / group / role / tenant_public
    principal_id   TEXT NOT NULL,
    effect         TEXT NOT NULL DEFAULT 'allow',          -- allow / deny
    PRIMARY KEY (chunk_id, principal_type, principal_id)
);
CREATE INDEX idx_acl_principal ON chunk_acl(principal_type, principal_id);

-- 向量映射（三向量：dense + sparse + colbert）
CREATE TABLE chunk_vectors (
    chunk_id        TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    collection      TEXT NOT NULL,
    embed_model     TEXT NOT NULL,
    dense_point_id  TEXT,
    sparse_point_id TEXT,
    colbert_point_id TEXT,
    dims            INTEGER
);

-- ---------------- 结构化双轨（产品 SKU / 价格 / 规格） ----------------
CREATE TABLE structured_records (
    record_id      TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    kb_id          TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
    chunk_id       TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    schema_id      TEXT NOT NULL,
    sku            TEXT,
    name           TEXT,
    price          NUMERIC(18,4),
    currency       TEXT,
    attrs          JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_struct_kb_sku ON structured_records(kb_id, sku);
CREATE INDEX idx_struct_price ON structured_records(kb_id, price);
CREATE INDEX idx_struct_attrs ON structured_records USING GIN (attrs);

-- ---------------- 摄入作业 ----------------
CREATE TABLE ingestion_jobs (
    job_id         TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    tenant_id      TEXT NOT NULL,
    kb_id          TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    version        INTEGER,
    stage          TEXT NOT NULL DEFAULT 'queued',         -- parse/clean/chunk/enrich/embed/index
    progress       NUMERIC(5,2) NOT NULL DEFAULT 0,
    error          TEXT,
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_job_status ON ingestion_jobs(stage, created_at DESC);

-- ---------------- 决策 trace（Laya 可观测 / 增量训练样本） ----------------
CREATE TABLE decision_trace (
    trace_id       TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    query_id       TEXT NOT NULL,
    tenant_id      TEXT NOT NULL,
    kb_id          TEXT,
    model_id       TEXT NOT NULL,
    model_version  TEXT,
    profile        TEXT NOT NULL,
    escalate       TEXT,
    answerable_prob NUMERIC(5,4),
    confidence     NUMERIC(5,4),
    probabilities  JSONB,
    features       JSONB,
    reason         TEXT,
    fallback_used  BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms     NUMERIC(10,2),
    -- 线上反馈：用于决策模型增量训练与净收益评估
    outcome        JSONB DEFAULT '{}'::jsonb,              -- {label, quality, cost, user_feedback}
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_trace_query ON decision_trace(query_id);
CREATE INDEX idx_trace_outcome ON decision_trace(created_at DESC);

-- ---------------- 切片评测（自查询 recall，参数选优依据） ----------------
CREATE TABLE eval_runs (
    eval_id        TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    kb_id          TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    strategy       TEXT NOT NULL,
    params         JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics        JSONB NOT NULL DEFAULT '{}'::jsonb,     -- recall@10/20, ndcg@10
    sampled_chunks INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------- 模型版本登记（影子索引 / 灰度） ----------------
CREATE TABLE model_versions (
    model_id       TEXT NOT NULL,
    version        TEXT NOT NULL,
    stage          TEXT NOT NULL,                          -- parse/clean/enrich/retrieve/decide
    backend        TEXT NOT NULL,                          -- tei/vllm/local/api
    license        TEXT,
    status         TEXT NOT NULL DEFAULT 'active',         -- active / shadow / deprecated
    traffic_ratio  NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, version)
);

-- ---------------- 行级安全（RLS）骨架 ----------------
-- 生产环境：为每个租户会话设置 app.tenant_id，由 RLS 兜底隔离。
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY chunks_tenant_isolation ON chunks
--     USING (tenant_id = current_setting('app.tenant_id', true));
