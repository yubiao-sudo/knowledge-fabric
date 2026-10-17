"""Knowledge Fabric · 检索链路（Python + FastAPI）。

    from kf_retrieve import run_search, SearchRequest

对外契约（网关 `gateway/src/clients/retrieval.rs` 是唯一调用方）：

    POST /v1/search
      请求：{ profile, query, tenant, kb, top_k, max_context_tokens }
      响应：{ evidence, citations, rerank_ms, trace }

链路编排：查询理解 → 多通道召回 → 融合 → 重排 → 组装。
任一路径（Qdrant / TEI / vLLM）不可用都只降级并记 note，绝不 500。
"""

from .app import app  # noqa: F401
from .backends import MemoryBackend, build_backend  # noqa: F401
from .service import SearchRequest, run_search  # noqa: F401

__all__ = [
    "app",
    "MemoryBackend",
    "build_backend",
    "SearchRequest",
    "run_search",
]
