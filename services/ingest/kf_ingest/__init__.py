"""Knowledge Fabric · 摄入流水线（Python 侧）。

    from kf_ingest import run_ingest, IngestContext

    python -m kf_ingest.cli --path ./手册.pdf --tenant default --kb demo

六个阶段的纯函数式入口都在这里导出，便于单独测试与替换：

    解析 parse_pdf  →  清洗 clean_document  →  切片 chunk_document
    →  增强 enrich_chunks  →  嵌入 encode_chunks  →  索引 index_chunks
"""

from .models import (  # noqa: F401
    Block,
    Chunk,
    CleaningReport,
    IngestContext,
    ParsedDoc,
    ProvenanceSpan,
    SectionNode,
)
from .parse import parse_pdf  # noqa: F401
from .clean import (  # noqa: F401
    clean_document,
    compute_provenance,
    detect_running_boilerplate,
    normalize_text,
)
from .chunk import chunk_document, split_sentences  # noqa: F401
from .enrich import enrich_chunks  # noqa: F401
from .embed import encode_chunks, lexical_sparse  # noqa: F401
from .index import index_chunks, point_id  # noqa: F401
from .pipeline import document_features, run_ingest  # noqa: F401
from .tokenize import TokenCounter, detect_lang  # noqa: F401

__all__ = [
    "Block",
    "Chunk",
    "CleaningReport",
    "IngestContext",
    "ParsedDoc",
    "ProvenanceSpan",
    "SectionNode",
    "parse_pdf",
    "clean_document",
    "compute_provenance",
    "detect_running_boilerplate",
    "normalize_text",
    "chunk_document",
    "split_sentences",
    "enrich_chunks",
    "encode_chunks",
    "lexical_sparse",
    "index_chunks",
    "point_id",
    "document_features",
    "run_ingest",
    "TokenCounter",
    "detect_lang",
]
