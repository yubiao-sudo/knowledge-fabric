"""Knowledge Fabric · 摄入领域模型

所有阶段共享的数据契约。两条硬性要求贯穿全局：

1. **provenance**：每个 chunk 都必须能映射回原文位置（page + 字符区间），
   它是引用溯源、点击回原文、审计的唯一依据。清洗必须可逆。
2. **lang**：多语言库的稀疏索引按语言分片，因此语言标签必须在切片阶段就写入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---- 块类型：版面的最小单元 ----
BLOCK_TITLE = "title"
BLOCK_PARAGRAPH = "paragraph"
BLOCK_LIST = "list"
BLOCK_TABLE = "table"
BLOCK_FIGURE = "figure"
BLOCK_FORMULA = "formula"
BLOCK_HEADER = "header"
BLOCK_FOOTER = "footer"
BLOCK_PAGE_NUMBER = "page_number"

# 原子块：不允许被切开，否则信息完整性受损
ATOMIC_BLOCK_TYPES = frozenset({BLOCK_TABLE, BLOCK_FIGURE, BLOCK_FORMULA})

# 页眉 / 页脚 / 页码：清洗的首要目标
BOILERPLATE_BLOCK_TYPES = frozenset({BLOCK_HEADER, BLOCK_FOOTER, BLOCK_PAGE_NUMBER})


@dataclass
class Block:
    """解析产出的结构化块（版面感知的最小单元）。"""

    block_type: str
    text: str
    page: int
    order: int
    bbox: Optional[List[float]] = None      # [x0, y0, x1, y1]
    level: Optional[int] = None             # 标题层级（1 起）
    max_font_size: float = 0.0
    uid: str = ""                           # 文档内唯一，用于清洗标记与溯源
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_atomic(self) -> bool:
        return self.block_type in ATOMIC_BLOCK_TYPES

    @property
    def is_boilerplate(self) -> bool:
        return self.block_type in BOILERPLATE_BLOCK_TYPES


@dataclass
class SectionNode:
    """章节树节点，供结构感知切片使用。"""

    level: int
    title: str
    block_uid: Optional[str] = None
    children: List["SectionNode"] = field(default_factory=list)

    def path(self) -> List[str]:
        return [self.title]


@dataclass
class ParsedDoc:
    """解析阶段产出：结构化文档树 + 原文页文本。"""

    doc_id: str
    version: int
    lang: str
    doc_type: str
    blocks: List[Block] = field(default_factory=list)
    section_tree: List[SectionNode] = field(default_factory=list)
    raw_pages: List[str] = field(default_factory=list)   # 每页原始文本，用于计算字符偏移
    page_sizes: List[List[float]] = field(default_factory=list)  # [[w, h], ...]
    scanner: str = ""                                    # 使用的解析器标识
    warnings: List[str] = field(default_factory=list)

    @property
    def needs_ocr(self) -> bool:
        return "needs_ocr" in self.warnings

    def page_height(self, page: int) -> float:
        if 0 <= page < len(self.page_sizes):
            return float(self.page_sizes[page][1])
        return 0.0

    def page_count(self) -> int:
        return len(self.raw_pages)


@dataclass
class ProvenanceSpan:
    """清洗可逆的核心：切片片段 → 原文位置。"""

    source_uri: str
    page: int
    char_start: int
    char_end: int
    bbox: Optional[List[float]] = None
    cleaning_ops: List[str] = field(default_factory=list)
    original_text: str = ""


@dataclass
class Chunk:
    """切片产出。parent 与 child 共同构成 small-to-big 检索结构。"""

    chunk_id: str
    chunk_type: str                          # text / table / code / formula / figure / clause / question
    content: str
    lang: str
    page: int
    order: int
    parent_chunk_id: Optional[str] = None
    context_prefix: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    token_count: int = 0
    provenance: Optional[ProvenanceSpan] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    structured: Optional[Dict[str, Any]] = None
    acl: List[Dict[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_parent(self) -> bool:
        return self.parent_chunk_id is None


@dataclass
class CleaningReport:
    """清洗审计：可解释地记录删掉了什么、为什么。"""

    total_blocks: int = 0
    removed_blocks: int = 0
    removed_by_type: Dict[str, int] = field(default_factory=dict)
    removed_samples: List[str] = field(default_factory=list)
    deduped_blocks: int = 0
    boilerplate_keys: List[str] = field(default_factory=list)
    page_count: int = 0

    def note_removed(self, block_type: str, text: str) -> None:
        self.removed_blocks += 1
        self.removed_by_type[block_type] = self.removed_by_type.get(block_type, 0) + 1
        if len(self.removed_samples) < 20:
            self.removed_samples.append(text[:80])


@dataclass
class IngestContext:
    """一次摄入作业的上下文。"""

    tenant_id: str
    kb_id: str
    doc_id: str
    source_uri: str
    source_path: str = ""                # 本地文件路径（source_uri 不是本地文件时使用）
    version: int = 1
    lang: str = ""
    doc_type: str = "manual"
    strategy: str = "doc_structured"
    graph: bool = False                    # 是否建图（--graph 开启，同时置 graph_enabled）
    parsed: Optional[ParsedDoc] = None
    parents: List[Chunk] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    cleaning: CleaningReport = field(default_factory=CleaningReport)
    embedded: int = 0
    embedded_model: str = ""
    indexed: int = 0
    collection: str = ""
    stage_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def all_chunks(self) -> List[Chunk]:
        return self.parents + self.chunks

    def summary(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "version": self.version,
            "lang": self.lang,
            "strategy": self.strategy,
            "pages": self.parsed.page_count() if self.parsed else 0,
            "blocks": len(self.parsed.blocks) if self.parsed else 0,
            "parents": len(self.parents),
            "children": len(self.chunks),
            "removed_blocks": self.cleaning.removed_blocks,
            "deduped_blocks": self.cleaning.deduped_blocks,
            "embedded": self.embedded,
            "indexed": self.indexed,
            "errors": self.errors,
        }
