-- ============================================================
-- 003_fix_graph_tenant_scope.sql
-- 修正 002 中图存储唯一键的租户维度缺失。
--
-- 背景（真机校验发现）：002 的 `uq_entity_kb_norm(kb_id, norm_name)` 与
-- `uq_edge_pair(kb_id, src, dst, relation)` 只按 kb_id 判重。当**两个租户使用
-- 同名 kb**时，后摄入租户的 `ON CONFLICT` 会命中前一个租户的行，导致
-- 「B 租户的实体被并入 A 租户」——即图存储层面的跨租户串数据。
--
-- 本迁移把唯一键扩展为带 tenant_id；对**已存在**的库做就地修正。
-- 幂等：全部 `IF EXISTS` / `IF NOT EXISTS`。
--
-- 数据修复说明：已被并入的行无法在 DDL 层安全还原（无法判定归属），
-- 受影响租户需重新摄入（`--graph`）以重建自身图；本文件只负责约束修正，
-- 不猜测性地删除既有数据。
-- ============================================================

BEGIN;

-- 1) 实体唯一键：kb 级 → 租户+kb 级
DROP INDEX IF EXISTS uq_entity_kb_norm;
CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_tenant_kb_norm
    ON graph_entities(tenant_id, kb_id, norm_name);

-- 2) 边唯一键：kb 级 → 租户+kb 级
DROP INDEX IF EXISTS uq_edge_pair;
CREATE UNIQUE INDEX IF NOT EXISTS uq_edge_tenant_pair
    ON graph_edges(tenant_id, kb_id, src_entity_id, dst_entity_id, relation);

-- 3) 社区索引同步带上 tenant_id
DROP INDEX IF EXISTS idx_comm_kb;
CREATE INDEX IF NOT EXISTS idx_comm_tenant_kb ON graph_communities(tenant_id, kb_id, level);

COMMIT;
