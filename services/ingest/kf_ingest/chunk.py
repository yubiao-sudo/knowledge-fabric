"""切片：策略引擎，而不是一刀切的固定参数。

默认组合（四段式）：

    版面感知  ──►  结构递归  ──►  父子块  ──►  上下文增强

本模块实现前三段（上下文增强由 enrich 阶段写入 `context_prefix`）：

  - **版面感知**：块边界对齐版块；表格 / 公式 / 图片为原子块，不跨块切分。
  - **结构递归**：在 章节 → 段落 → 句子 三级边界累积切分，遵守 token 上限；
                 段落之间可以合并以凑满目标粒度，但**绝不在句子中间切断**。
  - **父子块**：小块入库检索、父块回填生成，以 parent_chunk_id 关联；
                 重叠只作用于同级子块，父块内部不重叠。

两条容易踩坑的地方已在代码中落实：
  1. **token 计量以实际嵌入模型 tokenizer 为准**（见 tokenize.TokenCounter），
     不用字符数近似——中文/英文/日文在同一字符长度下的 token 数差异显著。
  2. **句子边界语言感知**：中文按 。！？；切，不能按 `.` 切。

**策略差异化**（单元发现见 `strategies.py`，此处只做装配）：

| 策略 | 子块单元 | 父块 |
|---|---|---|
| `doc_structured` / `table_centric` | 结构递归（句子累积） | 有 |
| `semantic` | 语义断点（相邻句相似度低谷） | 有 |
| `clause_atomic` | 单条款 | 无 |
| `code_aware` | 单函数 / 类 | 无 |
| `visual_page` | 整页 | 无 |

是否有父块由配置文件里该策略的 `pipeline` 是否含 `parent_child` 决定（配置即契约），
因此 `pipeline=None` 时默认产出父块，保持既有调用方的行为不变。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .models import (
    ATOMIC_BLOCK_TYPES,
    BLOCK_FIGURE,
    BLOCK_FORMULA,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TITLE,
    Block,
    Chunk,
    ProvenanceSpan,
)
from .strategies import (
    Encoder,
    clause_spans,
    code_spans,
    markdown_table_structured,
    semantic_boundaries,
)
from .tokenize import TokenCounter, detect_lang

_ZH_SENT = re.compile(r"(?<=[。！？；♪])")
_LATIN_SENT = re.compile(r"(?<=[.!?])\s+")
_SENT_END = re.compile(r"[。！？；.!?]")
_MD_HEADER_SEP = re.compile(r"^\|[\s:|-]+\|$")

# 原子 chunk 类型：整块不可切分的信息载体
ATOMIC_CHUNK_TYPES = frozenset({"table", "figure", "formula"})


@dataclass
class ChunkingResult:
    parents: List[Chunk] = field(default_factory=list)
    children: List[Chunk] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Settings:
    """切片参数的集中解包（只读一次，避免各处散落 params.get）。"""

    child_tokens: int
    parent_tokens: int
    parent_break_level: int
    overlap_tokens: int
    group_rows: int
    repeat_header: bool
    emit_structured: bool
    min_child_tokens: int
    max_child_tokens: int
    breakpoint_percentile: float
    keep_numbering: bool
    unit: str
    keep_indent: bool


# 父块默认上限。旧默认 1600 与子块默认 320 恰好构成 1:5，导致父块总是"攒满 5 个子块"
# 才断、必然横跨章节；这里收到 600（约 2 个子块）并阻断该巧合。
DEFAULT_PARENT_TOKENS = 600
# 父块分组在"层级 <= 该值"的标题处断开。旧实现是 ==1，而 PDF 的章节标题常被判为
# level 2（字号比例 1.35~1.6），于是章节边界全部失效、父块退化成纯 token 预算切分。
DEFAULT_PARENT_BREAK_LEVEL = 2


def _settings(params: Dict[str, Any]) -> _Settings:
    child = int(params.get("child_tokens", 320))
    parent = int(params.get("parent_tokens", DEFAULT_PARENT_TOKENS))
    # 父块不得大到吞掉多个子块：上限压到 3× child，且至少要有 1 个子的余量
    parent = min(parent, max(child + 1, child * 3))
    return _Settings(
        child_tokens=child,
        parent_tokens=parent,
        parent_break_level=int(params.get("parent_break_level", DEFAULT_PARENT_BREAK_LEVEL)),
        overlap_tokens=max(0, int(child * float(params.get("overlap_ratio", 0.12)))),
        group_rows=int(params.get("group_rows", 20)),
        repeat_header=bool(params.get("repeat_header_per_group", True)),
        emit_structured=bool(params.get("emit_structured", False)),
        min_child_tokens=int(params.get("min_child_tokens", max(1, child // 2))),
        max_child_tokens=int(params.get("max_child_tokens", child)),
        breakpoint_percentile=float(params.get("breakpoint_percentile", 0.85)),
        keep_numbering=bool(params.get("keep_numbering", True)),
        unit=str(params.get("unit", "function")),
        keep_indent=bool(params.get("keep_indent", True)),
    )


# ------------------------------------------------------------------ 句子切分
def split_sentences(text: str, lang: str) -> List[str]:
    """语言感知的句子切分。中文/日文用句末标点，拉丁语系用标点+空白。"""
    if not text:
        return []
    parts: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        segs = _ZH_SENT.split(para) if lang in ("zh", "ja") else _LATIN_SENT.split(para)
        parts.extend(s.strip() for s in segs if s and s.strip())
    return parts


def hard_split(text: str, max_tokens: int, counter: TokenCounter) -> List[str]:
    """单句超限时的兜底硬切：按 token 预算窗口切分。"""
    if max_tokens <= 0 or not text:
        return [text] if text else []
    if counter.count(text) <= max_tokens:
        return [text]

    ratio = max_tokens / counter.count(text)
    window = max(1, int(len(text) * ratio))
    out: List[str] = []
    cursor = 0
    while cursor < len(text):
        piece = text[cursor:cursor + window]
        while len(piece) > 1 and counter.count(piece) > max_tokens:
            piece = piece[: max(1, len(piece) - max(1, len(piece) // 10))]
        if not piece:
            break
        out.append(piece)
        cursor += len(piece)
    return out


def _clean_carry(text: str) -> str:
    """把 overlap 尾部对齐到句子边界，避免下一个切片以半句开头。"""
    if not text:
        return ""
    match = _SENT_END.search(text)
    if match and match.end() < len(text):
        return text[match.end():].strip()
    return "" if match else text.strip()


# ------------------------------------------------------------------ 章节路径
def build_block_sections(blocks: Sequence[Block]) -> Dict[str, List[str]]:
    """为每个块计算所属章节路径（标题块归属其自身开启的新章节）。"""
    stack: List[Tuple[int, str]] = []
    out: Dict[str, List[str]] = {}
    for block in blocks:
        if block.block_type == BLOCK_TITLE:
            level = block.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text.strip()))
        out[block.uid] = [t for _, t in stack] or ["(根)"]
    return out


# ------------------------------------------------------------------ 表格处理
def split_markdown_table(
    markdown: str,
    counter: TokenCounter,
    group_rows: int,
    max_tokens: int,
) -> List[str]:
    """大表按行组切分，**每组都重复表头**，保证每个切片自身信息完整。"""
    lines = [ln for ln in markdown.split("\n") if ln.strip()]
    if len(lines) < 3:
        return [markdown]

    header = lines[0]
    sep = lines[1] if _MD_HEADER_SEP.match(lines[1]) else "|" + "---|" * max(1, header.count("|") - 1)
    body = lines[2:]
    head_tokens = counter.count(header) + counter.count(sep)

    groups: List[str] = []
    current: List[str] = []
    current_tokens = head_tokens

    for row in body:
        row_tokens = counter.count(row)
        if current and (len(current) >= group_rows or current_tokens + row_tokens > max_tokens):
            groups.append("\n".join([header, sep] + current))
            current = []
            current_tokens = head_tokens
        current.append(row)
        current_tokens += row_tokens

    if current:
        groups.append("\n".join([header, sep] + current))
    return groups or [markdown]


def _join_with_map(blocks: Sequence[Block]) -> Tuple[str, List[Tuple[int, Block]]]:
    """把一组块拼成整段文本，并记录每块的起始偏移（用于把单元映射回块）。"""
    pieces: List[str] = []
    marks: List[Tuple[int, Block]] = []
    cursor = 0
    for block in blocks:
        text = block.text or ""
        if not text:
            continue
        if pieces:
            pieces.append("\n")
            cursor += 1
        marks.append((cursor, block))
        pieces.append(text)
        cursor += len(text)
    return "".join(pieces), marks


def _block_at(marks: Sequence[Tuple[int, Block]], offset: int) -> Optional[Block]:
    found: Optional[Block] = marks[0][1] if marks else None
    for start, block in marks:
        if start <= offset:
            found = block
        else:
            break
    return found


# ------------------------------------------------------------------ 装配器
class _Emitter:
    """父/子块的统一发射器：各策略路径共用，避免 id 与溯源逻辑分叉。"""

    def __init__(
        self,
        *,
        doc_id: str,
        version: int,
        lang: str,
        acl: Sequence[Dict[str, str]],
        sections: Dict[str, List[str]],
        counter: TokenCounter,
        provenance_map: Optional[Dict[str, ProvenanceSpan]],
        settings: _Settings,
        with_parents: bool,
        sentence_encoder: Optional[Encoder] = None,
    ) -> None:
        self.doc_id = doc_id
        self.version = version
        self.lang = lang
        self.acl = list(acl)
        self.sections = sections
        self.counter = counter
        self.provenance_map = provenance_map or {}
        self.settings = settings
        self.with_parents = with_parents
        self.sentence_encoder = sentence_encoder
        self.result = ChunkingResult()
        self.notes: List[str] = []
        self.skipped_atomic = 0
        self._parent_seq = 0
        self._child_seq = 0

    # ---- 分组 ----
    def groups(self, blocks: Sequence[Block]) -> List[List[Block]]:
        """按结构边界（标题层级 <= parent_break_level / token 预算）聚成父块候选组。

        标题边界优先于 token 预算：先看标题，再看预算，保证父块不横跨章节。

        **超大块不再贴着前一组走**：真机上有一份文档含 7729 字符的 SQL 代码块
        （单块 ≈2180 token），旧实现会把它塞进上一个父组，得到 2180 token 的父块
        ——远超 `parent_tokens` 目标，且嵌入耗时是非线性放大的（27s vs 319token 的 4s）。
        这里让"单块就超预算"的块自行成组，避免污染相邻内容。
        """
        groups: List[List[Block]] = []
        current: List[Block] = []
        tokens = 0
        break_level = max(1, int(self.settings.parent_break_level))
        limit = self.settings.parent_tokens
        for block in blocks:
            if (block.block_type == BLOCK_TITLE
                    and (block.level if block.level is not None else 9) <= break_level
                    and current):
                groups.append(current)
                current, tokens = [], 0
            block_tokens = self.counter.count(block.text) if block.text else 0
            # 原子块给父块留出余量，避免父块被一张大表挤爆
            reserve = limit // 2 if block.block_type in ATOMIC_BLOCK_TYPES else 0
            if current and tokens + block_tokens + reserve > limit:
                groups.append(current)
                current, tokens = [], 0
            current.append(block)
            tokens += block_tokens
            # 单块自己就超过上限：立刻封组，别让它拖累后面的块
            if block_tokens > limit:
                groups.append(current)
                current, tokens = [], 0
        if current:
            groups.append(current)
        return groups

    def page_groups(self, blocks: Sequence[Block]) -> List[List[Block]]:
        """按页码分组（页级单元用）。"""
        groups: List[List[Block]] = []
        current: List[Block] = []
        page: Optional[int] = None
        for block in blocks:
            if page is not None and block.page != page:
                groups.append(current)
                current = []
            current.append(block)
            page = block.page
        if current:
            groups.append(current)
        return groups

    # ---- 发射 ----
    def parent(self, group: Sequence[Block]) -> str:
        """发射父块，返回**最后一个**父块 id（供未细分的调用方使用）。

        一般应改用 `parents(group)` 拿到全部 id，以便把子块平均挂到各父块上。
        """
        ids = self.parents(group)
        return ids[-1] if ids else ""

    def parents(self, group: Sequence[Block]) -> List[str]:
        """发射该组的父块，返回全部 id。

        若组内容超过 `parent_tokens`（真机上有一份文档含 7729 字符 / ≈2180 token 的
        SQL 代码块，单块就超标），按 token 预算**切成多个父块**而不是硬塞一个超大块：
        超大父块会非线性推高嵌入耗时（2180 token 需 27s，319 token 只要 4s），
        且作为"上游上下文"对下游毫无用处。切成多个既守住预算，也不丢内容。
        """
        text = "\n\n".join(b.text for b in group if b.text)
        limit = self.settings.parent_tokens
        if self.counter.count(text) <= limit:
            return [self._emit_parent(group, text)]

        pieces = self._split_by_tokens(text, limit)
        ids = [self._emit_parent(group, piece) for piece in pieces]
        self.notes.append(
            f"父块组超过 parent_tokens={limit}，已拆成 {len(pieces)} 个父块"
        )
        return ids

    def _split_by_tokens(self, text: str, limit: int) -> List[str]:
        """按 token 预算把文本切成若干段，尽量在换行/句界处断开。"""
        if limit <= 0 or not text:
            return [text]
        out: List[str] = []
        buffer = ""
        for line in text.splitlines(keepends=True):
            candidate = buffer + line
            if buffer and self.counter.count(candidate) > limit:
                out.append(buffer)
                buffer = line
            else:
                buffer = candidate
            # 单行本身超限：按字符硬切
            while self.counter.count(buffer) > limit and len(buffer) > 1:
                ratio = limit / max(1, self.counter.count(buffer))
                cut = max(1, int(len(buffer) * ratio))
                out.append(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            out.append(buffer)
        return [p for p in out if p.strip()] or [text]

    def _emit_parent(self, group: Sequence[Block], text: str) -> str:
        self._parent_seq += 1
        parent_id = f"{self.doc_id}:v{self.version}:P{self._parent_seq}"
        # 父块实际覆盖的章节集合。父块按 token 预算切分时可能跨章节，此时子块继承的
        # section_path（取首块）会系统性标错——这里如实记录覆盖范围，供下游与审计使用。
        covered: List[List[str]] = []
        for block in group:
            path = self.sections.get(block.uid, ["(根)"])
            if path not in covered:
                covered.append(path)
        sections_covered = [" / ".join(p) for p in covered]
        self.result.parents.append(
            Chunk(
                chunk_id=parent_id,
                chunk_type="text",
                content=text,
                lang=self.lang,
                page=group[0].page,
                order=self._parent_seq,
                parent_chunk_id=None,
                section_path=self.sections.get(group[0].uid, ["(根)"]),
                token_count=self.counter.count(text),
                acl=self.acl,
                meta={
                    "source_blocks": [b.uid for b in group],
                    "role": "parent",
                    "sections_covered": sections_covered,
                    "spans_multiple_sections": len(covered) > 1,
                },
            )
        )
        return parent_id

    def emit(
        self,
        block: Block,
        kind: str,
        text: str,
        parent_id: Optional[str],
        structured: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        content = text.strip()
        if not content:
            if block.block_type in ATOMIC_BLOCK_TYPES:
                self.skipped_atomic += 1
            return
        self._child_seq += 1
        section_path = self.sections.get(block.uid, ["(根)"])
        meta: Dict[str, Any] = {
            "source_blocks": [block.uid],
            "role": "child",
            "block_type": block.block_type,
            "sibling_section": section_path,
        }
        if extra_meta:
            meta.update(extra_meta)
        self.result.children.append(
            Chunk(
                chunk_id=_make_child_id(self.doc_id, self.version, self._child_seq, content),
                chunk_type=kind,
                content=content,
                lang=self.lang,
                page=block.page,
                order=self._child_seq,
                parent_chunk_id=parent_id,
                section_path=section_path,
                token_count=self.counter.count(content),
                provenance=self._prov(block),
                acl=self.acl,
                structured=structured,
                meta=meta,
            )
        )

    def emit_atomic(self, block: Block, parent_id: Optional[str]) -> None:
        """原子块：表格可按行组切分（每组重复表头），其余整块发出。"""
        if not block.text:
            # 无文本的原子块（图片 / 未识别公式）暂不入库：占位文本只会污染向量索引
            self.skipped_atomic += 1
            return
        kind = _chunk_type_of(block)
        if block.block_type == BLOCK_TABLE and self.settings.repeat_header:
            pieces = split_markdown_table(
                block.text, self.counter, self.settings.group_rows, self.settings.child_tokens
            )
        else:
            pieces = [block.text]
        for piece in pieces:
            structured = (
                markdown_table_structured(piece)
                if self.settings.emit_structured and block.block_type == BLOCK_TABLE
                else None
            )
            self.emit(block, kind, piece, parent_id, structured=structured)

    # ---- 各策略路径 ----
    def _pick_parent(self, parent_ids: Sequence[str], offset_tokens: int) -> Optional[str]:
        """按**已产出 token 量**把子块挂到对应的父块上。

        父块可能因超出 parent_tokens 被拆成多个（见 `parents()`），此时若所有子块都
        挂到第一个父块，其余父块就成了没有子块的空壳。这里按每个父块的目标大小
        （parent_tokens）估算它覆盖的区间，再据此分配。
        """
        if not parent_ids:
            return None
        if len(parent_ids) == 1:
            return parent_ids[0]
        size = max(1, self.settings.parent_tokens)
        index = min(len(parent_ids) - 1, max(0, offset_tokens // size))
        return parent_ids[index]

    def add_structural(self, blocks: Sequence[Block]) -> None:
        """结构递归（doc_structured / table_centric）：句子累积到 child_tokens。"""
        for group in self.groups(blocks):
            parent_ids = self.parents(group) if self.with_parents else []
            self._emit_structural_group(group, parent_ids)

    def _emit_structural_group(self, group: Sequence[Block],
                               parent_ids: Sequence[str] = ()) -> None:
        """对一组块做结构递归切分（被 doc_structured 与各策略的兜底路径共用）。"""
        settings = self.settings
        buffer = ""
        buffer_tokens = 0
        buffer_block: Optional[Block] = None
        produced = 0          # 本组已产出 token 量，用于选父块

        def current_parent() -> Optional[str]:
            return self._pick_parent(parent_ids, produced)

        def flush() -> None:
            nonlocal buffer, buffer_tokens, buffer_block
            if buffer.strip() and buffer_block is not None:
                self.emit(buffer_block, "text", buffer, current_parent())
            buffer, buffer_tokens, buffer_block = "", 0, None

        for block in group:
            if block.block_type in ATOMIC_BLOCK_TYPES:
                # 原子块自带完整性（每个组都重复表头），不把尾部作为 overlap
                # 带入后续块：否则表格行片段会混进无关正文，既重复索引又丢失表头关联。
                flush()
                self.emit_atomic(block, current_parent())
                produced += self.counter.count(block.text) if block.text else 0
                continue
            if not block.text:
                continue
            sentences = split_sentences(block.text, self.lang) or [block.text]
            for sentence in sentences:
                subs = (
                    hard_split(sentence, settings.child_tokens, self.counter)
                    if self.counter.count(sentence) > settings.child_tokens
                    else [sentence]
                )
                for sub in subs:
                    sub_tokens = self.counter.count(sub)
                    if (buffer.strip() and buffer_block is not None
                            and buffer_tokens + sub_tokens > settings.child_tokens):
                        previous = buffer
                        self.emit(buffer_block, "text", previous, current_parent())
                        produced += buffer_tokens
                        carry = (
                            _clean_carry(self.counter.tail(previous, settings.overlap_tokens))
                            if settings.overlap_tokens
                            else ""
                        )
                        buffer = carry
                        buffer_tokens = self.counter.count(carry) if carry else 0
                        buffer_block = block if carry else None
                    buffer = (buffer + "\n" + sub).strip() if buffer.strip() else sub
                    buffer_tokens = self.counter.count(buffer)
                    if buffer_block is None:
                        buffer_block = block
        flush()

    def _split_oversized(self, text: str) -> List[str]:
        """超限文本按**句子边界**再切（不退化为整块，也不在句中断开）。"""
        limit = self.settings.max_child_tokens
        out: List[str] = []
        buffer = ""
        for sentence in split_sentences(text, self.lang) or [text]:
            pieces = (
                hard_split(sentence, limit, self.counter)
                if self.counter.count(sentence) > limit
                else [sentence]
            )
            for piece in pieces:
                if buffer and self.counter.count(buffer) + self.counter.count(piece) > limit:
                    out.append(buffer)
                    buffer = piece
                else:
                    buffer = (buffer + "\n" + piece).strip() if buffer else piece
        if buffer:
            out.append(buffer)
        return out or [text]

    def add_semantic(self, blocks: Sequence[Block]) -> None:
        """语义断点（semantic）：相邻句相似度低谷处断开，粒度受 min/max 收敛。"""
        joiner = "" if self.lang in ("zh", "ja") else " "
        for group in self.groups(blocks):
            parent_ids = self.parents(group) if self.with_parents else []
            pending: List[Tuple[str, Block]] = []
            produced = 0

            def current_parent() -> Optional[str]:
                return self._pick_parent(parent_ids, produced)

            def flush() -> None:
                nonlocal pending, produced
                produced += self._emit_semantic(pending, current_parent(), joiner)
                pending = []

            for block in group:
                if block.block_type in ATOMIC_BLOCK_TYPES:
                    flush()
                    self.emit_atomic(block, current_parent())
                    produced += self.counter.count(block.text) if block.text else 0
                    continue
                if not block.text:
                    continue
                for sentence in split_sentences(block.text, self.lang) or [block.text]:
                    pending.append((sentence, block))
            flush()

    def _emit_semantic(
        self, pending: Sequence[Tuple[str, Block]], parent_id: Optional[str], joiner: str
    ) -> int:
        """发射语义切片，返回产出的 token 总量（供父块分配使用）。"""
        if not pending:
            return 0
        settings = self.settings
        sentences = [sentence for sentence, _ in pending]
        produced = 0
        boundaries = semantic_boundaries(
            sentences,
            counter=self.counter,
            min_tokens=settings.min_child_tokens,
            max_tokens=settings.max_child_tokens,
            percentile=settings.breakpoint_percentile,
            encoder=self.sentence_encoder,
        )
        for index in range(len(boundaries) - 1):
            start, end = boundaries[index], boundaries[index + 1]
            if start >= end:
                continue
            block = pending[start][1]
            text = joiner.join(sentences[start:end]).strip()
            pieces = (
                hard_split(text, settings.max_child_tokens, self.counter)
                if self.counter.count(text) > settings.max_child_tokens
                else [text]
            )
            for piece in pieces:
                self.emit(block, "text", piece, parent_id)
                produced += self.counter.count(piece)
        return produced

    def add_units(
        self,
        blocks: Sequence[Block],
        *,
        kind: str,
        spans_fn: Callable[[str], List[Tuple[int, ...]]],
        keep_numbering: bool = True,
    ) -> None:
        """条款 / 代码：整组文本上按序号或语法锚点切单元，再映射回所在块。"""
        settings = self.settings
        for group in self.groups(blocks):
            parent_ids = self.parents(group) if self.with_parents else []
            produced = 0

            def current_parent() -> Optional[str]:
                return self._pick_parent(parent_ids, produced)

            text_blocks: List[Block] = []
            for block in group:
                if block.block_type in ATOMIC_BLOCK_TYPES:
                    self.emit_atomic(block, current_parent())
                    produced += self.counter.count(block.text) if block.text else 0
                elif block.text:
                    text_blocks.append(block)
            if not text_blocks:
                continue

            joined, marks = _join_with_map(text_blocks)
            spans = spans_fn(joined)
            if not spans:
                # 识别不到锚点就**不猜**：回退结构递归，避免把整组压成一个大块
                self.notes.append(
                    f"{kind}：未识别到单元锚点，{len(text_blocks)} 个块回退结构递归切分"
                )
                self._emit_structural_group(text_blocks, parent_ids)
                continue

            oversized = 0
            for span in spans:
                start, body_start, end = (
                    (span[0], span[1], span[2]) if len(span) == 3 else (span[0], span[0], span[1])
                )
                block = _block_at(marks, start)
                if block is None:
                    continue
                piece = joined[start:end]
                if kind == "clause" and not keep_numbering:
                    piece = joined[body_start:end]
                if kind == "code" and not settings.keep_indent:
                    piece = "\n".join(line.strip() for line in piece.split("\n"))
                piece = piece.strip()
                if not piece:
                    continue
                if self.counter.count(piece) > settings.max_child_tokens:
                    # 最后一个锚点会一直延伸到组尾（后续可能是代码 / 表格 / 正文），
                    # 或单条本身超长：按句子边界拆开，总好过产出一个吞掉全部内容的大块。
                    oversized += 1
                    for sub in self._split_oversized(piece):
                        self.emit(block, kind, sub, current_parent())
                        produced += self.counter.count(sub)
                    continue
                self.emit(block, kind, piece, current_parent())
                produced += self.counter.count(piece)
            if oversized:
                self.notes.append(
                    f"{kind}：{oversized} 个单元超过 max_child_tokens"
                    f"({settings.max_child_tokens})，已按句子边界拆分（牺牲单条原子性）"
                )

    def add_pages(self, blocks: Sequence[Block]) -> None:
        """页级单元（visual_page）。"""
        self.notes.append(
            "visual_page：按页级单元切分；ColPali 页级视觉向量需多模态服务，当前未接入，"
            "页内文本仍走常规稀疏/稠密编码；页级块无精确字符偏移（provenance 缺失）"
        )
        for group in self.page_groups(blocks):
            text = "\n\n".join(b.text for b in group if b.text).strip()
            if not text:
                continue
            self.emit(group[0], "text", text, None, extra_meta={"unit": "page"})

    # ---- 收尾 ----
    def finalize(self, strategy: str) -> ChunkingResult:
        children = self.result.children
        settings = self.settings
        self.result.stats = {
            "strategy": strategy,
            "with_parents": self.with_parents,
            "parents": len(self.result.parents),
            "children": len(children),
            "child_tokens_target": settings.child_tokens,
            "parent_tokens_target": settings.parent_tokens,
            "overlap_tokens": settings.overlap_tokens,
            "avg_child_tokens": round(
                sum(c.token_count for c in children) / max(1, len(children)), 1
            ),
            "max_child_tokens": max((c.token_count for c in children), default=0),
            "atomic_children": sum(1 for c in children if c.chunk_type in ATOMIC_CHUNK_TYPES),
            "skipped_atomic": self.skipped_atomic,
            "unit_types": dict(Counter(c.chunk_type for c in children)),
            "notes": self.notes,
        }
        return self.result

    def _prov(self, block: Block) -> Optional[ProvenanceSpan]:
        base = self.provenance_map.get(block.uid)
        if base is None:
            return None
        return ProvenanceSpan(
            source_uri=base.source_uri,
            page=base.page,
            char_start=base.char_start,
            char_end=base.char_end,
            bbox=base.bbox,
            cleaning_ops=list(base.cleaning_ops),
            original_text=base.original_text,
        )


# ------------------------------------------------------------------ 主流程
def chunk_document(
    blocks: Sequence[Block],
    *,
    doc_id: str,
    version: int,
    lang: str,
    params: Dict[str, Any],
    counter: TokenCounter,
    strategy: str = "doc_structured",
    pipeline: Optional[Sequence[str]] = None,
    acl: Optional[List[Dict[str, str]]] = None,
    provenance_map: Optional[Dict[str, ProvenanceSpan]] = None,
    sentence_encoder: Optional[Encoder] = None,
) -> ChunkingResult:
    """按策略产出父块与子块。

    params 来自 configs/chunking_strategies.yaml 的 defaults + 策略 params：
        child_tokens / parent_tokens / overlap_ratio / group_rows /
        repeat_header_per_group / emit_structured /
        min_child_tokens / max_child_tokens / breakpoint_percentile /
        atomic_unit / keep_numbering / unit / keep_indent

    `pipeline` 为该策略在 YAML 里声明的阶段列表；含 `parent_child` 才产出父块。
    传 `None` 时默认产出父块（兼容既有调用方）。
    """
    with_parents = True if pipeline is None else ("parent_child" in {str(s) for s in pipeline})
    emitter = _Emitter(
        doc_id=doc_id,
        version=version,
        lang=lang,
        acl=acl or [],
        sections=build_block_sections(blocks),
        counter=counter,
        provenance_map=provenance_map,
        settings=_settings(params),
        with_parents=with_parents,
        sentence_encoder=sentence_encoder,
    )

    if strategy == "visual_page":
        emitter.add_pages(blocks)
    elif strategy == "clause_atomic":
        emitter.add_units(
            blocks, kind="clause", spans_fn=clause_spans,
            keep_numbering=emitter.settings.keep_numbering,
        )
    elif strategy == "code_aware":
        emitter.add_units(
            blocks, kind="code",
            spans_fn=lambda text: code_spans(text, unit=emitter.settings.unit),
        )
    elif strategy == "semantic":
        emitter.add_semantic(blocks)
    else:
        emitter.add_structural(blocks)

    return emitter.finalize(strategy)


def _chunk_type_of(block: Block) -> str:
    if block.block_type == BLOCK_TABLE:
        return "table"
    if block.block_type == BLOCK_FORMULA:
        return "formula"
    if block.block_type == BLOCK_FIGURE:
        return "figure"
    if block.block_type == BLOCK_PARAGRAPH and block.meta.get("is_code"):
        return "code"
    return "text"


def _make_child_id(doc_id: str, version: int, seq: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}:v{version}:C{seq}-{digest}"


def lang_of_blocks(blocks: Sequence[Block]) -> str:
    """以块文本合计判定文档语言。"""
    return detect_lang("\n".join(b.text for b in blocks[:50] if b.text))
