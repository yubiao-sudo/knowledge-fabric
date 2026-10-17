-- ============================================================
-- 004_tenant_scoped_keys.sql
-- 把 tenant_id 纳入主键，彻底消除「跨租户静默串数据」。
--
-- 背景（真机复现的缺陷，非推测）：
--   各表主键只有业务 id，**不含租户维度**，于是同一 id 在第二个租户下
--   必然命中第一个租户的行：
--     · knowledge_bases 用 `ON CONFLICT (kb_id) DO NOTHING` → 第二个租户
--       连自己的库都建不出来（真机实测：documents/chunks 有 demoB/graphdemo，
--       而 knowledge_bases 里只有 demoA/graphdemo）；
--     · documents 用 `ON CONFLICT (doc_id) DO UPDATE` → 只更新版本号，
--       tenant_id/kb_id 保持首次写入的租户，新租户归属被静默丢弃；
--     · chunks 的 stale 标记无租户过滤 → 会把其他租户的同名文档块一起标 stale。
--   实测复现：alpha/kbalpha 先摄入 collide_probe，再以 beta/kbbeta 摄入同一
--   doc_id，Postgres 里 documents 只有 alpha 一行、chunks 16 行全归 alpha，
--   beta 零行——接口却返回 ok=true、postgres=16、errors=[]。
--
-- 本迁移做三件事：
--   1) 主键补 tenant_id，外键同步改为**复合键**——引用完整性与 RLS 同层，
--      即便写入侧漏了过滤条件，数据库也会直接拒绝跨租户引用；
--   2) 修数据：补回被键设计丢弃的 knowledge_bases 行（只补本该存在的行）；
--   3) `documents` / `decision_trace` / `ingestion_jobs` 补齐 RLS 策略。
--
-- 幂等：可安全重复执行。执行前会先禁用外键触发校验，避免中间态误报。
-- ============================================================

BEGIN;

-- ---------------- 0) 补 tenant_id 列（旧库可能缺） ----------------
ALTER TABLE document_versions   ADD COLUMN IF NOT EXISTS tenant_id TEXT;
ALTER TABLE chunk_vectors       ADD COLUMN IF NOT EXISTS tenant_id TEXT;
ALTER TABLE graph_entity_chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT;

-- ---------------- 1) 同步回填 tenant_id ----------------
-- 权威链：**documents（文档归属）→ document_versions → chunks**。
-- `documents` 是 doc_id 归属的唯一权威：chunks 的租户标签可能是旧缺陷写错的
-- （真机实测：同一 doc_id 在 documents 属 demo，chunks 里却有 8 行标成 acme，
-- 且这 8 行内容与 demo 下的块逐字相同）。因此：
--   ① 版本行的租户由 documents 回填；
--   ② 租户与 documents 不一致的 chunks 属残留，在第 4 步按悬空语义清理；
--   ③ 子表 tenant_id 一律从所属 chunks 回填（它们本就该与块一致）。
--
-- 注意不能反过来用 chunks 投票决定版本行租户：那样会把版本行定成错误的 acme，
-- 而 documents 侧无法容纳同一 doc_id 的两个租户，最终外键仍然对不上。
CREATE TEMP TABLE _kf_doc_tenant ON COMMIT DROP AS
SELECT doc_id, min(tenant_id) AS tenant_id, count(DISTINCT tenant_id) AS n_tenants
  FROM documents GROUP BY doc_id;

UPDATE document_versions v
   SET tenant_id = t.tenant_id
  FROM _kf_doc_tenant t
 WHERE v.doc_id = t.doc_id AND v.tenant_id IS NULL;

DO $$
DECLARE
    ambiguous INTEGER;
    still_null INTEGER;
BEGIN
    SELECT count(*) INTO ambiguous FROM _kf_doc_tenant WHERE n_tenants > 1;
    SELECT count(*) INTO still_null FROM document_versions WHERE tenant_id IS NULL;
    IF ambiguous > 0 THEN
        RAISE NOTICE '有 % 个 doc_id 在 documents 中属于多个租户（归属歧义）: %',
            ambiguous,
            (SELECT string_agg(doc_id, ' | ') FROM _kf_doc_tenant WHERE n_tenants > 1);
    END IF;
    IF still_null > 0 THEN
        RAISE NOTICE '仍有 % 行 document_versions 无租户归属（无法推导）', still_null;
    END IF;
END
$$;

UPDATE chunk_vectors cv SET tenant_id = c.tenant_id
  FROM chunks c WHERE c.chunk_id = cv.chunk_id AND cv.tenant_id IS NULL;

UPDATE graph_entity_chunks ec SET tenant_id = e.tenant_id
  FROM graph_entities e WHERE e.entity_id = ec.entity_id AND ec.tenant_id IS NULL;

-- chunks 的 tenant_id 若与其 doc 的归属租户不一致，说明它是旧缺陷写入的错标行。
-- 这里**不动**这些行：它们随后会被第 4 步的「悬空 chunk 清理」按复合键语义处理，
-- 而不是靠猜测改写租户标签。
DO $$
DECLARE
    mismatched INTEGER;
BEGIN
    SELECT count(*) INTO mismatched
      FROM (SELECT DISTINCT c.tenant_id, c.doc_id FROM chunks c) c
      LEFT JOIN _kf_doc_tenant t ON t.doc_id = c.doc_id
     WHERE t.tenant_id IS DISTINCT FROM c.tenant_id;
    IF mismatched > 0 THEN
        RAISE NOTICE 'chunks 中有 % 组 (租户, doc_id) 与 documents 的归属不一致（旧缺陷残留）', mismatched;
    END IF;
END
$$;

-- ---------------- 2) 解除外键（改主键的前提） ----------------
ALTER TABLE documents             DROP CONSTRAINT IF EXISTS documents_kb_id_fkey;
ALTER TABLE chunks                DROP CONSTRAINT IF EXISTS chunks_kb_id_fkey;
ALTER TABLE graph_entities        DROP CONSTRAINT IF EXISTS graph_entities_kb_id_fkey;
ALTER TABLE graph_edges           DROP CONSTRAINT IF EXISTS graph_edges_kb_id_fkey;
ALTER TABLE graph_communities     DROP CONSTRAINT IF EXISTS graph_communities_kb_id_fkey;
ALTER TABLE structured_records    DROP CONSTRAINT IF EXISTS structured_records_kb_id_fkey;
ALTER TABLE document_versions     DROP CONSTRAINT IF EXISTS document_versions_doc_id_fkey;
ALTER TABLE chunks                DROP CONSTRAINT IF EXISTS chunks_doc_id_version_fkey;
ALTER TABLE chunk_provenance      DROP CONSTRAINT IF EXISTS chunk_provenance_chunk_id_fkey;
ALTER TABLE chunk_acl             DROP CONSTRAINT IF EXISTS chunk_acl_chunk_id_fkey;
ALTER TABLE chunk_vectors         DROP CONSTRAINT IF EXISTS chunk_vectors_chunk_id_fkey;
ALTER TABLE structured_records    DROP CONSTRAINT IF EXISTS structured_records_chunk_id_fkey;
ALTER TABLE graph_edges           DROP CONSTRAINT IF EXISTS graph_edges_src_entity_id_fkey;
ALTER TABLE graph_edges           DROP CONSTRAINT IF EXISTS graph_edges_dst_entity_id_fkey;
ALTER TABLE graph_entity_chunks   DROP CONSTRAINT IF EXISTS graph_entity_chunks_entity_id_fkey;

-- ---------------- 3) 换主键：租户维度进主键 ----------------
ALTER TABLE knowledge_bases DROP CONSTRAINT IF EXISTS knowledge_bases_pkey;
ALTER TABLE knowledge_bases ADD CONSTRAINT knowledge_bases_pkey PRIMARY KEY (tenant_id, kb_id);

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey;
ALTER TABLE documents ADD CONSTRAINT documents_pkey PRIMARY KEY (tenant_id, doc_id);

ALTER TABLE document_versions DROP CONSTRAINT IF EXISTS document_versions_pkey;
ALTER TABLE document_versions ADD CONSTRAINT document_versions_pkey
    PRIMARY KEY (tenant_id, doc_id, version);

ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_pkey;
ALTER TABLE chunks ADD CONSTRAINT chunks_pkey PRIMARY KEY (tenant_id, chunk_id);

ALTER TABLE chunk_provenance DROP CONSTRAINT IF EXISTS chunk_provenance_pkey;
ALTER TABLE chunk_provenance ADD CONSTRAINT chunk_provenance_pkey PRIMARY KEY (tenant_id, chunk_id);

ALTER TABLE chunk_vectors DROP CONSTRAINT IF EXISTS chunk_vectors_pkey;
ALTER TABLE chunk_vectors ADD CONSTRAINT chunk_vectors_pkey PRIMARY KEY (tenant_id, chunk_id);

ALTER TABLE chunk_acl DROP CONSTRAINT IF EXISTS chunk_acl_pkey;
ALTER TABLE chunk_acl ADD CONSTRAINT chunk_acl_pkey
    PRIMARY KEY (tenant_id, chunk_id, principal_type, principal_id);

ALTER TABLE structured_records DROP CONSTRAINT IF EXISTS structured_records_pkey;
ALTER TABLE structured_records ADD CONSTRAINT structured_records_pkey
    PRIMARY KEY (tenant_id, record_id);

ALTER TABLE graph_entities DROP CONSTRAINT IF EXISTS graph_entities_pkey;
ALTER TABLE graph_entities ADD CONSTRAINT graph_entities_pkey PRIMARY KEY (tenant_id, entity_id);

ALTER TABLE graph_edges DROP CONSTRAINT IF EXISTS graph_edges_pkey;
ALTER TABLE graph_edges ADD CONSTRAINT graph_edges_pkey PRIMARY KEY (tenant_id, edge_id);

ALTER TABLE graph_entity_chunks DROP CONSTRAINT IF EXISTS graph_entity_chunks_pkey;
ALTER TABLE graph_entity_chunks ADD CONSTRAINT graph_entity_chunks_pkey
    PRIMARY KEY (tenant_id, entity_id, chunk_id);

-- ---------------- 4) 修数据：先清掉悬空的跨租户残留 ----------------
-- 真机残留（有据可查，非猜测）：chunks 里有 8 行 (acme, demo, 数据库设计方案_合并完整版, v1)，
-- 而 documents / document_versions 里该 doc 只属于 (demo, company)。这 8 行的
-- **内容与 demo/company 下的块逐字相同**，只是 chunk_id 序号不同——正是
-- `ON CONFLICT (doc_id) DO UPDATE` 只改版本号、不改归属留下的产物：
-- 第二个租户的块被写成了第一个租户的租户标签，同时 document_versions 侧
-- 没有对应版本行（外键悬空）。
--
-- 处理原则：**仅删除引用已确认悬空的行**，即满足以下任一条件的 chunk——
--   (a) 不存在对应的 document_versions 行；或
--   (b) 其 tenant_id 与 documents 中该 doc_id 的归属租户不一致（旧缺陷写错的标签）。
-- 不删除、不合并任何归属正确的行。子表行显式删除（外键此刻已解除）。
CREATE TEMP TABLE _kf_orphan_chunks ON COMMIT DROP AS
SELECT c.tenant_id, c.chunk_id
  FROM chunks c
  LEFT JOIN document_versions v
         ON v.doc_id = c.doc_id AND v.version = c.version
  LEFT JOIN _kf_doc_tenant t ON t.doc_id = c.doc_id
 WHERE v.doc_id IS NULL OR t.tenant_id IS DISTINCT FROM c.tenant_id;

DO $$
DECLARE
    n INTEGER;
BEGIN
    SELECT count(*) INTO n FROM _kf_orphan_chunks;
    IF n > 0 THEN
        RAISE NOTICE '清理悬空 chunk 行: % 条（无对应 document_versions，属跨租户残留）', n;
    END IF;
END
$$;

-- 子表一律按 chunk_id 连接删除：子表的 tenant_id 是 002 迁移**事后**回填的，
-- 实测存在与所属 chunk 不一致的历史值（chunk_provenance 有 acme/company 的错标行），
-- 用 tenant_id 做连接条件会漏删，留下悬空引用。
DELETE FROM chunk_provenance p USING _kf_orphan_chunks o WHERE p.chunk_id = o.chunk_id;
DELETE FROM chunk_acl        a USING _kf_orphan_chunks o WHERE a.chunk_id = o.chunk_id;
DELETE FROM chunk_vectors   cv USING _kf_orphan_chunks o WHERE cv.chunk_id = o.chunk_id;
DELETE FROM structured_records s USING _kf_orphan_chunks o WHERE s.chunk_id = o.chunk_id;
DELETE FROM chunks           c USING _kf_orphan_chunks o WHERE c.chunk_id = o.chunk_id;

-- 子表租户改齐：chunk_provenance / chunk_acl / chunk_vectors 的 tenant_id 是
-- 002 迁移**事后**回填的（真机实测存在 acme/company 这类与所属 chunk 不符的历史值），
-- 而复合外键要求子表与块同租户。这些行是**有内容的元数据**（ACL 规则、溯源偏移），
-- 不能像悬空块那样删掉——按块的实际归属把它们移动过去（只改租户标签，内容不动）。
-- 放在悬空块清理**之后**执行，避免把已删除块的元数据又指向不存在的块。
-- 注：这几张表都没有别的复合唯一键，改 tenant_id 不会产生主键冲突。
UPDATE chunk_provenance p SET tenant_id = c.tenant_id
  FROM chunks c WHERE c.chunk_id = p.chunk_id AND p.tenant_id IS DISTINCT FROM c.tenant_id;
UPDATE chunk_acl a SET tenant_id = c.tenant_id
  FROM chunks c WHERE c.chunk_id = a.chunk_id AND a.tenant_id IS DISTINCT FROM c.tenant_id;
UPDATE chunk_vectors cv SET tenant_id = c.tenant_id
  FROM chunks c WHERE c.chunk_id = cv.chunk_id AND cv.tenant_id IS DISTINCT FROM c.tenant_id;

-- 兜底：仍无租户的子表行（块已不存在）按块删除，避免外键悬空
DELETE FROM chunk_provenance p WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = p.chunk_id);
DELETE FROM chunk_acl        a WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = a.chunk_id);
DELETE FROM chunk_vectors   cv WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = cv.chunk_id);

-- ---------------- 5) 修数据：补回被键设计丢弃的 knowledge_bases 行 ----------------
-- 此时 kb_id 的全局唯一约束已随主键变更消失，多租户可以合法复用同名 kb。
-- 只是把「本应存在却被 DO NOTHING 丢掉」的行补回来；不合并、不改动既有行。
INSERT INTO knowledge_bases (kb_id, tenant_id, name, chunking_strategy)
SELECT DISTINCT d.kb_id, d.tenant_id, d.kb_id, 'doc_structured'
  FROM documents d
  LEFT JOIN knowledge_bases kb ON kb.tenant_id = d.tenant_id AND kb.kb_id = d.kb_id
 WHERE kb.kb_id IS NULL
ON CONFLICT (tenant_id, kb_id) DO NOTHING;

-- chunks 侧同样可能引用了不存在的库（同一根因的另一面）
INSERT INTO knowledge_bases (kb_id, tenant_id, name, chunking_strategy)
SELECT DISTINCT c.kb_id, c.tenant_id, c.kb_id, 'doc_structured'
  FROM chunks c
  LEFT JOIN knowledge_bases kb ON kb.tenant_id = c.tenant_id AND kb.kb_id = c.kb_id
 WHERE kb.kb_id IS NULL
ON CONFLICT (tenant_id, kb_id) DO NOTHING;

-- 数据修复结果如实汇报（不静默假装干净）
DO $$
DECLARE
    missing_docs INTEGER;
    missing_chunks INTEGER;
    null_tenants INTEGER;
BEGIN
    SELECT count(*) INTO missing_docs FROM documents d
      LEFT JOIN knowledge_bases kb ON kb.tenant_id = d.tenant_id AND kb.kb_id = d.kb_id
     WHERE kb.kb_id IS NULL;
    SELECT count(*) INTO missing_chunks FROM chunks c
      LEFT JOIN knowledge_bases kb ON kb.tenant_id = c.tenant_id AND kb.kb_id = c.kb_id
     WHERE kb.kb_id IS NULL;
    SELECT count(*) INTO null_tenants FROM document_versions WHERE tenant_id IS NULL;
    IF missing_docs > 0 OR missing_chunks > 0 OR null_tenants > 0 THEN
        RAISE NOTICE '修复后仍有残留: documents 缺库=%, chunks 缺库=%, document_versions 缺租户=%',
                     missing_docs, missing_chunks, null_tenants;
    END IF;
END
$$;

-- ---------------- 5) 重建外键（复合键，与 RLS 同层） ----------------
ALTER TABLE documents ADD CONSTRAINT documents_kb_id_fkey
    FOREIGN KEY (tenant_id, kb_id) REFERENCES knowledge_bases(tenant_id, kb_id) ON DELETE CASCADE;
ALTER TABLE chunks ADD CONSTRAINT chunks_kb_id_fkey
    FOREIGN KEY (tenant_id, kb_id) REFERENCES knowledge_bases(tenant_id, kb_id) ON DELETE CASCADE;
ALTER TABLE graph_entities ADD CONSTRAINT graph_entities_kb_id_fkey
    FOREIGN KEY (tenant_id, kb_id) REFERENCES knowledge_bases(tenant_id, kb_id) ON DELETE CASCADE;
ALTER TABLE graph_edges ADD CONSTRAINT graph_edges_kb_id_fkey
    FOREIGN KEY (tenant_id, kb_id) REFERENCES knowledge_bases(tenant_id, kb_id) ON DELETE CASCADE;
ALTER TABLE graph_communities ADD CONSTRAINT graph_communities_kb_id_fkey
    FOREIGN KEY (tenant_id, kb_id) REFERENCES knowledge_bases(tenant_id, kb_id) ON DELETE CASCADE;
ALTER TABLE structured_records ADD CONSTRAINT structured_records_kb_id_fkey
    FOREIGN KEY (tenant_id, kb_id) REFERENCES knowledge_bases(tenant_id, kb_id) ON DELETE CASCADE;

ALTER TABLE document_versions ADD CONSTRAINT document_versions_doc_id_fkey
    FOREIGN KEY (tenant_id, doc_id) REFERENCES documents(tenant_id, doc_id) ON DELETE CASCADE;

ALTER TABLE chunks ADD CONSTRAINT chunks_doc_id_version_fkey
    FOREIGN KEY (tenant_id, doc_id, version)
    REFERENCES document_versions(tenant_id, doc_id, version) ON DELETE CASCADE;

ALTER TABLE chunk_provenance ADD CONSTRAINT chunk_provenance_chunk_id_fkey
    FOREIGN KEY (tenant_id, chunk_id) REFERENCES chunks(tenant_id, chunk_id) ON DELETE CASCADE;
ALTER TABLE chunk_acl ADD CONSTRAINT chunk_acl_chunk_id_fkey
    FOREIGN KEY (tenant_id, chunk_id) REFERENCES chunks(tenant_id, chunk_id) ON DELETE CASCADE;
ALTER TABLE chunk_vectors ADD CONSTRAINT chunk_vectors_chunk_id_fkey
    FOREIGN KEY (tenant_id, chunk_id) REFERENCES chunks(tenant_id, chunk_id) ON DELETE CASCADE;
-- structured_records 从 `ON DELETE SET NULL` 改为 `CASCADE`：
-- 复合外键的 SET NULL 会把**所有**引用列置空，包括 NOT NULL 的 tenant_id，
-- 删除块时直接报 NotNullViolation（真机删除旧块时复现）。
-- 语义上也该 CASCADE：这些记录是从块里派生的，块没了记录就没有依据。
ALTER TABLE structured_records ADD CONSTRAINT structured_records_chunk_id_fkey
    FOREIGN KEY (tenant_id, chunk_id) REFERENCES chunks(tenant_id, chunk_id) ON DELETE CASCADE;

ALTER TABLE graph_edges ADD CONSTRAINT graph_edges_src_entity_id_fkey
    FOREIGN KEY (tenant_id, src_entity_id) REFERENCES graph_entities(tenant_id, entity_id) ON DELETE CASCADE;
ALTER TABLE graph_edges ADD CONSTRAINT graph_edges_dst_entity_id_fkey
    FOREIGN KEY (tenant_id, dst_entity_id) REFERENCES graph_entities(tenant_id, entity_id) ON DELETE CASCADE;
ALTER TABLE graph_entity_chunks ADD CONSTRAINT graph_entity_chunks_entity_id_fkey
    FOREIGN KEY (tenant_id, entity_id) REFERENCES graph_entities(tenant_id, entity_id) ON DELETE CASCADE;

-- ---------------- 6) 补齐缺失的 RLS（documents 此前只有列没有策略） ----------------
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_documents_tenant ON documents;
CREATE POLICY p_documents_tenant ON documents
    USING (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE decision_trace ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_trace_tenant ON decision_trace;
CREATE POLICY p_trace_tenant ON decision_trace
    USING (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_job_tenant ON ingestion_jobs;
CREATE POLICY p_job_tenant ON ingestion_jobs
    USING (tenant_id = current_setting('app.tenant_id', true));

-- 版本表与向量映射表：检索侧会读（版本过滤 / 点位映射），同样纳入 RLS
ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_versions_tenant ON document_versions;
CREATE POLICY p_versions_tenant ON document_versions
    USING (tenant_id = current_setting('app.tenant_id', true));

ALTER TABLE chunk_vectors ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS p_vectors_tenant ON chunk_vectors;
CREATE POLICY p_vectors_tenant ON chunk_vectors
    USING (tenant_id = current_setting('app.tenant_id', true));

-- 说明：`tenants` 与 `workspaces` **刻意不加 RLS**——它们是全局注册表
-- （租户清单 / 工作区清单），任何按租户过滤的策略都会让新租户查不到自己。
-- 这里如实标注，避免被误读为遗漏。它们不含企业知识内容。

-- ---------------- 7) 辅助索引与唯一键 ----------------
-- 结构化记录幂等键补租户维度（与写入侧 ON CONFLICT 目标一致）
DROP INDEX IF EXISTS uq_struct_chunk_schema_sku;
CREATE UNIQUE INDEX IF NOT EXISTS uq_struct_tenant_chunk_schema_sku
    ON structured_records (tenant_id, chunk_id, schema_id, COALESCE(sku, ''));

CREATE INDEX IF NOT EXISTS idx_documents_tenant_doc ON documents(tenant_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_status ON chunks(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_versions_tenant_doc ON document_versions(tenant_id, doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_tenant_kbid ON knowledge_bases(tenant_id, kb_id);

COMMIT;
