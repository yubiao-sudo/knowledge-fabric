# kf-retrieve · 检索链路服务

Knowledge Fabric 检索底座（Python + FastAPI）。链路：
**查询理解 → 多通道召回 → 融合 → 重排 → 组装**。
任一路径（Qdrant / TEI / vLLM）不可用都只降级并记 note，绝不 5xx。

## 启动

```bash
# 依赖
pip install -r requirements.txt

# 启动（端口来自 RETRIEVAL_PORT，默认 8080）
cd services/retrieve
KF_RETRIEVE_SEED_JSON=./seed.json python -m uvicorn kf_retrieve.app:app --port 8080
# 或：python -m uvicorn app:app --port 8080
```

- `QDRANT_URL` 已配置且可连 → 用 Qdrant；否则回退 **MemoryBackend**（进程内 BM25）。
- `TEI_EMBED_URL` 未配置 → 记忆后端不做 dense，只做 sparse。
- `TEI_RERANK_URL` 未配置 → 复用融合分，`trace.reranked=false`。
- `VLLM_URL` / `OPENAI_BASE_URL` 未配置 → HyDE / 扩展 / 分解跳过并记 note；Agentic 只跑 1 轮。
- `OPENSEARCH_URL` 未配置 → 稀疏通道走 Qdrant 稀疏 / MemoryBM25；配置但连不上 → 回退 + note。
- `DATABASE_APP_URL`（`kf_app` 角色）未配置 → graph 通道跳过 + note。
- `AGENTIC_MAX_ROUNDS` 覆盖 Agentic 全局硬限（默认 3）。
- `KF_RETRIEVE_SEED_JSON` → 启动时把 json 里的 chunk 灌入 MemoryBackend（便于无 Docker 验证；
  块内可带 `skus`/`price`/`currency`/`graph` 字段以覆盖结构化与图通道）。

## HTTP 契约（下游唯一调用方：`gateway/src/clients/retrieval.rs`）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | `{status, service, index{configured,collection}, reranker{configured}}` |
| GET | `/v1/profiles` | 档位名 + `global_budget` + `fusion_weights`（读自 `configs/retrieval_profiles.yaml`） |
| POST | `/v1/search` | 检索主入口 |

请求：`{"profile","query","tenant","kb","top_k","max_context_tokens","principal","graph_enabled"}`（`kb` / `top_k` 可为 null；`graph_enabled` 缺省 false）。
响应：`{"evidence":[...], "citations":[...], "rerank_ms":float, "trace":{...}}`。

- `evidence[].score` 为**重排后最终分**；`source` 取 provenance `source_uri`；
  `citations[].char_range` 取 provenance `char_start/char_end`。
- `trace` 为本服务自用可观测字段（gateway 以 `serde_json::Value` 宽松解析）。

## 档位（`configs/retrieval_profiles.yaml`）

| profile | channels | fusion | rerank | 备注 |
|---|---|---|---|---|
| naive | dense | none | ✗ | 仅稠密 |
| hybrid_rerank | dense,sparse | rrf | ✓ | 安全默认 |
| exact_lookup | sparse,structured | weighted(0.3/0.7) | ✓ | sparse_boost 2.0，结构化精确过滤 |
| hyde | dense,sparse | rrf | ✓ | 需生成端点 |
| multi_query | dense,sparse | rrf | ✓ | 需生成端点 |
| decompose_agentic | dense,sparse | rrf | ✓ | Agentic ≤3 轮 + 饱和/预算停止 |
| graph_global | dense,sparse,graph | rrf | ✓ | GraphRAG + 社区摘要 evidence |
| strict_refuse | dense,sparse | rrf | ✓ | 最高分 < 0.55 → 拒答 |

`global_budget`：`max_latency_ms=3000`、`max_context_tokens=8000`、`max_agentic_rounds=3`。

## 召回通道（真实语义，契约第 3 节）

| 通道 | 实现 | 关键约束 |
|---|---|---|
| `dense` | Qdrant 稠密向量 / MemoryBackend + TEI | **不做** lang 过滤（跨语言召回是设计本意） |
| `sparse` | OpenSearch 分语言 BM25（`OPENSEARCH_URL` 已配置且连得通）否则 Qdrant 稀疏 / MemoryBM25 | **保留 payload lang 过滤**；`trace.sparse_backend` 如实标注 `opensearch`/`qdrant`/`memory`；连不上 → 回退 + note |
| `structured` | Qdrant payload 精确过滤（`skus`=MatchAny、`price`=Range、`currency`=match）+ 稀疏排序；MemoryBackend 按 payload 字段精确匹配后 BM25 打分 | 条件只来自查询理解的 `exact` 词；**识别不到条件 → 空 + note「structured 通道：查询无结构化条件，已跳过」**，绝不退化成再跑 BM25 |
| `graph` | GraphRAG：PG 图读取（`DATABASE_APP_URL` / `kf_app` 角色，会话内 `SET LOCAL app.tenant_id` 满足 RLS）；MemoryBackend 从 seed 的 `graph` 字段同语义 | 需 **档位要求 graph** 且 **`graph_enabled=true`**（默认关闭）；存储缺失/库未开启 → 跳过 + note；`graph_global` 额外产出 `chunk_type=community_summary` 的 `community:<id>` evidence |

Agentic：`run_search` 外层 ≤`min(3, 档位 max_rounds)` 轮（`AGENTIC_MAX_ROUNDS` 覆盖全局 3）；
停止条件 `saturation` / `budget` / `max_rounds`；**无生成端点时不伪造多轮**，只跑 1 轮并标
`trace.agentic_stop="no_generator"`。

`trace` 关键字段：`channels` / `sparse_backend` / `structured{filters,matched 或 skipped}` /
`graph{enabled,hops,entities_matched,communities_used}` / `agentic_rounds` /
`agentic_stop` / `agentic_rounds_detail` / `notes`。

## 一致性红线

- **稀疏编码**：`kf_retrieve/sparse.py` 与 `services/ingest/kf_ingest/embed.py` 逐行等价，
  由 `tests/test_sparse_golden.py` 兜底（查询侧与索引侧必须同编码）。
- **Qdrant payload 过滤字段**：`tenant_id / kb_id / lang / version / status / chunk_type`
  与 `kf_ingest/index.py:_payload` 对齐。

## 测试

```bash
cd services/retrieve
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests -q
```
