"""清理重摄入留下的"旧块并存"：删除**同一文档里没有 dense 向量的旧块**。

背景：重摄入沿用了原 `current_version`（多为 1），而 stale 逻辑是
`WHERE version <> 当前版本`，版本没变 → 旧块不会被标记 stale，
于是新旧块并存在同一版本下。实测 `CFA_Design` 变成 371 块（旧 182 + 新 189）。

判据（安全边界）：
  * 只处理**同时存在有 dense 与无 dense 块的文档**——说明该文档确实重摄入成功过；
  * 只删除其中**无 dense 向量**的块（即修复前写入的旧块）；
  * 整篇文档都没有 dense 的（如源文件缺失、从未重摄入）**一律不动**。

执行前后都打印计数，可回滚（有备份 db/backup_before_reingest.sql）。
"""
from __future__ import annotations

import os
import sys

import psycopg

DSN = os.environ.get("DATABASE_URL",
                     "postgresql://kf:kf_dev_password@localhost:15432/knowledge_fabric")

# 找"混合"文档：同 (tenant, kb, doc, version) 下既有 dense 又有 non-dense
MIXED = """
SELECT c.tenant_id, c.kb_id, c.doc_id, c.version,
       count(*) FILTER (WHERE cv.dense_point_id IS NOT NULL) AS with_dense,
       count(*) FILTER (WHERE cv.dense_point_id IS NULL) AS without_dense
  FROM chunks c
  LEFT JOIN chunk_vectors cv ON cv.chunk_id = c.chunk_id AND cv.tenant_id = c.tenant_id
 WHERE c.status = 'active'
 GROUP BY 1,2,3,4
HAVING count(*) FILTER (WHERE cv.dense_point_id IS NOT NULL) > 0
   AND count(*) FILTER (WHERE cv.dense_point_id IS NULL) > 0
"""

DELETE_OLD = """
DELETE FROM chunks
 WHERE status = 'active'
   AND tenant_id = %s AND kb_id = %s AND doc_id = %s AND version = %s
   AND chunk_id NOT IN (
       SELECT chunk_id FROM chunk_vectors WHERE tenant_id = %s AND dense_point_id IS NOT NULL
   )
"""


def main() -> int:
    dry = "--apply" not in sys.argv
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(MIXED)
        rows = cur.fetchall()
        if not rows:
            print("没有发现'新旧并存'的文档，无需清理。")
            return 0

        print(f"{'库':<18}{'文档':<44}{'版本':>4}{'有dense':>8}{'无dense':>8}  处理")
        print("-" * 96)
        total_del = 0
        for tenant, kb, doc_id, version, with_dense, without_dense in rows:
            print(f"{tenant + '/' + kb:<18}{doc_id[:42]:<44}{version:>4}"
                  f"{with_dense:>8}{without_dense:>8}", end="")
            if dry:
                print("  （预演，未删除）")
                total_del += without_dense
                continue
            cur.execute(DELETE_OLD, (tenant, kb, doc_id, version, tenant))
            deleted = cur.rowcount
            total_del += deleted
            print(f"  已删 {deleted}")

    if dry:
        print("-" * 96)
        print(f"[预演] 将删除 {total_del} 个无 dense 向量的旧块。")
        print("确认后加 --apply 执行：python scripts/purge_legacy_chunks.py --apply")
    else:
        print("-" * 96)
        print(f"共删除 {total_del} 个旧块。")
        print("建议随后跑：python scripts/plan_reingest.py  确认各库状态")
    return 0


if __name__ == "__main__":
    sys.exit(main())
