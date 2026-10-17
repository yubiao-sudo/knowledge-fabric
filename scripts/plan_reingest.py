"""按文档粒度清点：哪些文档有 dense 向量、哪些溯源退化、源文件是否还在。

比按库统计更有用——acme/demo 是混合状态：修复后重摄入过的文档是好的，
修复前的旧文档没有 dense 向量。控制台默认连的就是这个库，所以必须先看清它。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import psycopg

DSN = os.environ.get("DATABASE_URL",
                     "postgresql://kf:kf_dev_password@localhost:15432/knowledge_fabric")

SQL = """
SELECT c.tenant_id, c.kb_id, c.doc_id, c.version,
       count(*) AS chunks,
       count(*) FILTER (WHERE cv.dense_point_id IS NOT NULL) AS with_dense,
       count(*) FILTER (WHERE p.cleaning_ops::text LIKE '%provenance_approx%') AS prov_approx,
       max(d.source_uri) AS source_uri
  FROM chunks c
  LEFT JOIN chunk_vectors cv ON cv.chunk_id = c.chunk_id AND cv.tenant_id = c.tenant_id
  LEFT JOIN chunk_provenance p ON p.chunk_id = c.chunk_id AND p.tenant_id = c.tenant_id
  LEFT JOIN documents d ON d.doc_id = c.doc_id AND d.tenant_id = c.tenant_id
 WHERE c.status = 'active'
 GROUP BY 1, 2, 3, 4
 ORDER BY 1, 2, 3
"""

with psycopg.connect(DSN) as conn, conn.cursor() as cur:
    cur.execute(SQL)
    rows = cur.fetchall()

print(f"{'库':<16}{'文档':<40}{'块':>5}{'有dense':>8}{'溯源退化':>9}  状态")
print("-" * 100)
todo: List[Dict[str, Any]] = []
missing: List[Dict[str, Any]] = []
for tenant, kb, doc_id, version, chunks, dense, approx, uri in rows:
    chunks, dense, approx = int(chunks), int(dense), int(approx)
    healthy = dense > 0 and approx == 0
    exists = bool(uri) and os.path.isfile(str(uri))
    status = "OK" if healthy else ("需重摄入" if exists else "需重摄入·源缺失")
    print(f"{tenant + '/' + kb:<16}{doc_id[:38]:<40}{chunks:>5}{dense:>8}{approx:>9}  {status}")
    item = {"tenant": tenant, "kb": kb, "doc_id": doc_id, "version": int(version or 1),
            "uri": uri}
    if not healthy:
        (todo if exists else missing).append(item)

print("-" * 100)
print(f"需重摄入且源文件在: {len(todo)}")
print(f"需重摄入但源文件丢失: {len(missing)}")
for m in missing:
    print(f"    · {m['tenant']}/{m['kb']} :: {m['doc_id'][:40]}  (uri={m['uri']})")

import json
with open(r"F:\knowledge-fabric\db\reingest_plan.json", "w", encoding="utf-8") as fh:
    json.dump({"todo": todo, "missing": missing}, fh, ensure_ascii=False, indent=2)
print("\n清单已写入 db/reingest_plan.json")
