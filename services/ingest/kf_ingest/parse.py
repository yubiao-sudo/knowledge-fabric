"""文档解析：把 PDF 变成**结构化文档树**，而不是一坨纯文本。

分流的判断放在上游（数字原生 / 复杂排版 / 扫描件 / 疑难页），本模块实现
"数字原生 + 复杂排版" 这一最主干的路径：

  PyMuPDF 抽文本与坐标  ──►  版面块分类（标题/段落/列表/表格/图片）
                         ──►  阅读顺序重排
                         ──►  章节树构建

表格走 PyMuPDF 的表格识别并转 Markdown（保留表头），后续切片阶段不允许切开。
扫描件与疑难页由 model_registry.yaml 中的 OCR / VLM 处理，本模块只负责**标记**
（warnings += "needs_ocr"），交由上游切换到对应解析器。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    BLOCK_FIGURE,
    BLOCK_FOOTER,
    BLOCK_HEADER,
    BLOCK_LIST,
    BLOCK_PAGE_NUMBER,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    BLOCK_TITLE,
    Block,
    ParsedDoc,
    SectionNode,
)
from .tokenize import detect_lang

SCANNER = "pymupdf"

_LIST_PATTERN = re.compile(
    r"^\s*(?:[-•*·▪◦]|(?:\d{1,3}|[一二三四五六七八九十]{1,3})\s*[.)、．]|\(\d{1,3}\)|（\d{1,3}）)\s+"
)
_PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:[-—–·]\s*)?(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,7})\s*(?:[-—–·]\s*)?$")
_SENTENCE_END = re.compile(r"[。！？.!?:：；;]\s*$")


def parse_pdf(
    path: str,
    doc_id: str,
    version: int = 1,
    doc_type: str = "",
    lang: str = "",
    max_pages: Optional[int] = None,
) -> ParsedDoc:
    """解析 PDF，产出结构化文档树。

    参数:
        path: 本地 PDF 路径
        doc_id / version: 幂等主键组成部分
        doc_type: 文档类型提示（manual / contract / regulation / code ...），可空
        lang: 语言提示；为空则自动判定
        max_pages: 调试用，限制解析页数
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError("缺少 PyMuPDF，请先 `pip install pymupdf`") from exc

    doc = fitz.open(path)
    if max_pages is not None:
        pages = range(min(max_pages, doc.page_count))
    else:
        pages = range(doc.page_count)

    blocks: List[Block] = []
    raw_pages: List[str] = []
    page_sizes: List[List[float]] = []
    empty_pages = 0
    scanned_pages = 0

    for page_index in pages:
        page = doc.load_page(page_index)
        rect = page.rect
        page_sizes.append([float(rect.width), float(rect.height)])

        raw = page.get_text("text") or ""
        raw_pages.append(raw)

        image_count = sum(1 for b in page.get_text("dict").get("blocks", []) if b.get("type") == 1)
        if len(raw.strip()) < 20:
            empty_pages += 1
            if image_count > 0:
                scanned_pages += 1

        # 表格必须先识别：其区域内的单元格会被 get_text 拆成碎片文本块，
        # 需要在抽取文本块时按表格区域剔除，否则表格内容会被索引两次
        # （一次是干净的 Markdown，一次是被打散的单元格）。
        table_blocks = _extract_page_tables(page, page_index)
        table_rects = [b.bbox for b in table_blocks if b.bbox]

        blocks.extend(_extract_page_blocks(page, page_index, rect, table_rects))
        blocks.extend(table_blocks)

    doc.close()

    # 阅读顺序重排：先按页，再按纵向位置，最后按横向
    blocks.sort(key=lambda b: (b.page, round(b.bbox[1], 1) if b.bbox else 0.0,
                               round(b.bbox[0], 1) if b.bbox else 0.0))
    for order, block in enumerate(blocks):
        block.order = order
        block.uid = f"p{block.page}-b{order}"

    body_size = _estimate_body_font_size(blocks)
    for block in blocks:
        _classify_block(block, body_size)

    parsed = ParsedDoc(
        doc_id=doc_id,
        version=version,
        lang=lang or detect_lang("\n".join(raw_pages[:5])),
        doc_type=doc_type or "manual",
        blocks=blocks,
        raw_pages=raw_pages,
        page_sizes=page_sizes,
        scanner=SCANNER,
    )
    parsed.section_tree = _build_section_tree(blocks)

    total = max(1, len(raw_pages))
    if scanned_pages / total >= 0.5:
        parsed.warnings.append("needs_ocr")          # 交由 OCR/VLM 分支处理
    elif empty_pages / total >= 0.3:
        parsed.warnings.append("sparse_text_layer")  # 文本层稀疏，建议复核解析质量
    if not blocks:
        parsed.warnings.append("no_blocks_extracted")
    if not parsed.doc_type:
        parsed.doc_type = "manual"

    return parsed


# ------------------------------------------------------------------ 格式分流
MARKDOWN_SCANNER = "markdown"

# 支持的格式：后缀 → 解析器标识。**只有这一份判定**，CLI / HTTP / 目录遍历共用，
# 避免出现「CLI 支持、HTTP 不支持」或「界面只认 PDF」这类口径分裂。
# 未列入的后缀（.docx / .txt / .jpg ...）一律**不支持**，由调用方如实报告为 skipped，
# 不允许静默放过（否则用户会以为已经建好库）。
SUPPORTED_SUFFIXES: Dict[str, str] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
}


def suffix_parser(path: str) -> Optional[str]:
    """返回该路径对应的解析器标识；不支持则返回 None。"""
    return SUPPORTED_SUFFIXES.get(os.path.splitext(path)[1].lower())


def parse_document(
    path: str,
    doc_id: str,
    version: int = 1,
    doc_type: str = "",
    lang: str = "",
    max_pages: Optional[int] = None,
) -> ParsedDoc:
    """按后缀把文档分派到对应解析器。不支持的格式**明确报错**，不退化成纯文本硬塞。"""
    kind = suffix_parser(path)
    if kind is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"不支持的文档格式 {os.path.splitext(path)[1]!r}（支持：{supported}）")
    if kind == "pdf":
        return parse_pdf(path, doc_id, version=version, doc_type=doc_type,
                          lang=lang, max_pages=max_pages)
    return parse_markdown(path, doc_id, version=version, doc_type=doc_type, lang=lang)


def discover_documents(
    directory: str,
    recursive: bool = False,
    match: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """扫描目录，返回 `(受支持的文档列表, 被跳过的文件列表)`。

    命名沿用 `scripts/batch_ingest.py` 的口径（同一套遍历规则）：
      - 隐藏文件 / 隐藏目录不进（避免把 `.git` `.venv` 这类目录卷进来）；
      - `~$` 前缀的 Office 临时文件不进；
      - `match` 为文件名包含的子串，用于先小范围试跑；
      - 非受支持后缀进 skipped，由调用方**如实列出**，绝不静默丢掉。
    """
    todo: List[str] = []
    skipped: List[str] = []
    root_dir = os.path.abspath(directory)
    if recursive:
        walker: Any = os.walk(root_dir)
    else:
        try:
            entries = os.listdir(root_dir)
        except OSError:
            entries = []
        walker = [(root_dir, [], entries)]

    for root, dirs, files in walker:
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith(".") or name.startswith("~$"):
                continue
            if match and match not in name:
                continue
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            if suffix_parser(full) is None:
                skipped.append(full)
            else:
                todo.append(full)
    return todo, skipped


# ------------------------------------------------------------------ Markdown
_FENCE = re.compile(r"^\s*(```|~~~)\s*([A-Za-z0-9_+.#-]*)\s*$")
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_H1 = re.compile(r"^\s*=+\s*$")
_HORIZONTAL_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}[.)]\s+)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_HTML_COMMENT_LINE = re.compile(r"^\s*<!--.*?-->\s*$")
_BLOCKQUOTE = re.compile(r"^\s*>\s?(.*)$")


def _read_text(path: str) -> Tuple[str, Optional[str]]:
    """读取文本。优先 utf-8-sig（顺带吃掉 BOM，否则 BOM 会挡住 `#` 标题识别），
    再退 gb18030；都失败才用替换字符兜底并**如实记警告**。"""
    with open(path, "rb") as fh:
        data = fh.read()
    for encoding, warning in (("utf-8-sig", None), ("gb18030", "decoded_gb18030")):
        try:
            return data.decode(encoding), warning
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace"), "decoded_with_replacement"


def _front_matter_end(lines: List[str]) -> int:
    """YAML front matter（文件开头的 `---` 到下一个 `---`）不产出块。

    只在**首行**是 `---` 时判定，避免把正文里的分隔线误当元数据。
    返回正文起始行号；没有 front matter 时返回 0。
    """
    first = 0
    while first < len(lines) and not lines[first].strip():
        first += 1
    if first >= len(lines) or lines[first].strip() != "---":
        return 0
    for index in range(first + 1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            return index + 1
    return 0


def _is_structural(line: str, lines: List[str], index: int) -> bool:
    """该行是否开启一个新的结构块（用于判断段落是否结束）。"""
    if not line.strip():
        return True
    if _HTML_COMMENT_LINE.match(line) or _HORIZONTAL_RULE.match(line):
        return True
    if _FENCE.match(line) or _ATX_HEADING.match(line):
        return True
    if _TABLE_ROW.match(line) or _LIST_ITEM.match(line) or _BLOCKQUOTE.match(line):
        return True
    # setext h1：当前行有内容、下一行是 `===`
    if index + 1 < len(lines) and _SETEXT_H1.match(lines[index + 1]):
        return True
    return False


def parse_markdown(
    path: str,
    doc_id: str,
    version: int = 1,
    doc_type: str = "",
    lang: str = "",
) -> ParsedDoc:
    """解析 Markdown 为**同一套**结构化文档树（与 PDF 共享受下游全部阶段）。

    与 PDF 的差异（如实标注，不假装一致）：
      - **无分页**：Markdown 没有页概念，所有块的 `page` 恒为 0，并记警告
        `no_page_layout`。定位靠 `provenance` 的字符区间与 `source_uri`；
      - **无版面**：字号/页眉页脚那套规则不适用，块类型直接由 Markdown 语法决定；
      - **无 bbox**：`bbox` 留空。

    块类型映射（沿用既有下游语义，不另造一套）：
      - `#`..`######` / `===`      → `title`（带 level）
      - ``` 围栏代码               → `paragraph` + `meta.is_code`（下游 `_kind` → `code`，
        由 `code_aware` 策略按语法边界切分）
      - `| a | b |` 表格           → `table`（**原子块**，保留表头重复能力）
      - `-` / `*` / `1.` 列表      → `list`
      - `>` 引用 / 普通段落        → `paragraph`
    """
    text, decode_warning = _read_text(path)
    lines = text.splitlines()
    blocks: List[Block] = []
    warnings: List[str] = []

    start = _front_matter_end(lines)
    if start > 0:
        warnings.append("front_matter_stripped")

    def emit(block_type: str, block_text: str, **kwargs: Any) -> None:
        blocks.append(Block(block_type=block_type, text=block_text, page=0, order=0, **kwargs))

    index = start
    total = len(lines)
    while index < total:
        line = lines[index]

        if not line.strip() or _HTML_COMMENT_LINE.match(line) or _HORIZONTAL_RULE.match(line):
            index += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            info = fence.group(2)
            index += 1
            body: List[str] = []
            while index < total and not lines[index].strip().startswith(marker):
                body.append(lines[index])
                index += 1
            index += 1  # 跳过闭合围栏
            code = "\n".join(body).strip("\n")
            if code:
                emit(BLOCK_PARAGRAPH, code, meta={"is_code": True, "code_lang": info})
            continue

        if _TABLE_ROW.match(line):
            rows: List[str] = []
            while index < total and _TABLE_ROW.match(lines[index]):
                rows.append(lines[index].strip())
                index += 1
            # 分隔行（|---|:--:|）不是数据行，但不从文本里删——保留才是合法 Markdown
            data_rows = [r for r in rows if not re.fullmatch(r"\|[\s:|-]+\|", r)]
            cell_count = max(
                (len([c for c in r.strip("|").split("|")]) for r in data_rows), default=0
            )
            emit(BLOCK_TABLE, "\n".join(rows),
                 meta={"rows": len(data_rows), "cols": cell_count})
            continue

        heading = _ATX_HEADING.match(line)
        if heading:
            emit(BLOCK_TITLE, heading.group(2).strip(), level=len(heading.group(1)))
            index += 1
            continue

        if index + 1 < total and _SETEXT_H1.match(lines[index + 1]) and line.strip():
            emit(BLOCK_TITLE, line.strip(), level=1)
            index += 2
            continue

        if _LIST_ITEM.match(line):
            items: List[str] = []
            while index < total and _LIST_ITEM.match(lines[index]) and lines[index].strip():
                items.append(lines[index].strip())
                index += 1
            emit(BLOCK_LIST, "\n".join(items), meta={"items": len(items)})
            continue

        quote = _BLOCKQUOTE.match(line)
        if quote:
            quoted: List[str] = []
            while index < total:
                match = _BLOCKQUOTE.match(lines[index])
                if not match or not lines[index].strip():
                    break
                quoted.append(match.group(1).strip())
                index += 1
            emit(BLOCK_PARAGRAPH, "\n".join(q for q in quoted if q), meta={"quote": True})
            continue

        paragraph: List[str] = [line.strip()]
        index += 1
        while index < total and not _is_structural(lines[index], lines, index):
            paragraph.append(lines[index].strip())
            index += 1
        emit(BLOCK_PARAGRAPH, "\n".join(paragraph))

    for order, block in enumerate(blocks):
        block.order = order
        block.uid = f"p0-b{order}"

    parsed = ParsedDoc(
        doc_id=doc_id,
        version=version,
        # 用**全文**判定语言：只取开头若干字符时，若文档以代码块 / 英文表格开头，
        # 采样窗口会被拉丁字母占满而误判成 en（真机复现见 tokenize.detect_lang 注释）。
        lang=lang or detect_lang(text),
        doc_type=doc_type or "manual",
        blocks=blocks,
        raw_pages=[text],
        page_sizes=[],
        scanner=MARKDOWN_SCANNER,
        warnings=list(warnings),
    )
    parsed.section_tree = _build_section_tree(blocks)

    parsed.warnings.append("no_page_layout")   # 无分页：page 恒为 0，定位靠字符区间
    if decode_warning:
        parsed.warnings.append(decode_warning)
    if not blocks:
        parsed.warnings.append("no_blocks_extracted")

    return parsed


# ------------------------------------------------------------------ 页面抽取
def _mostly_inside(bbox: List[float], rects: List[List[float]], threshold: float = 0.5) -> bool:
    """判断文本块是否主要落在给定区域内（用于剔除表格区域里被重复抽取的碎片）。

    用「占自身面积的比例」而非「完全包含」：跨在表格边缘的正文段落会被保留，
    只有真正属于表格的单元格碎片（大部分面积在表内）才被丢弃。
    """
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    if area <= 0:
        return False
    for rect in rects:
        overlap_x = max(0.0, min(bbox[2], rect[2]) - max(bbox[0], rect[0]))
        overlap_y = max(0.0, min(bbox[3], rect[3]) - max(bbox[1], rect[1]))
        if overlap_x * overlap_y / area >= threshold:
            return True
    return False


def _extract_page_blocks(
    page: Any, page_index: int, rect: Any, exclude_rects: Optional[List[List[float]]] = None
) -> List[Block]:
    out: List[Block] = []
    data = page.get_text("dict") or {}
    page_height = float(rect.height)

    for raw_block in data.get("blocks", []):
        bbox = [float(v) for v in raw_block.get("bbox", [0.0, 0.0, 0.0, 0.0])]

        if raw_block.get("type") == 1:
            out.append(
                Block(block_type=BLOCK_FIGURE, text="", page=page_index, order=0,
                      bbox=bbox, meta={"image": True, "page_height": page_height})
            )
            continue

        # 落在表格区域内的碎片文本块由 Markdown 表块代表，此处跳过以避免重复索引
        if exclude_rects and _mostly_inside(bbox, exclude_rects):
            continue

        lines: List[str] = []
        sizes: List[Tuple[float, int]] = []
        for line in raw_block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans)
            if line_text.strip():
                lines.append(line_text)
            for span in spans:
                sizes.append((float(span.get("size", 0.0)), len(span.get("text", ""))))

        text = "\n".join(lines).strip()
        if not text:
            continue

        max_size = max((s for s, _ in sizes), default=0.0)
        out.append(
            Block(block_type=BLOCK_PARAGRAPH, text=text, page=page_index, order=0,
                  bbox=bbox, max_font_size=max_size,
                  meta={"line_count": len(lines), "page_height": page_height})
        )

    return out


def _extract_page_tables(page: Any, page_index: int) -> List[Block]:
    """表格识别 → Markdown（保留表头），随后由切片阶段按行组切分并重复表头。"""
    out: List[Block] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return out
    try:
        tables = finder()
    except Exception:
        return out

    page_height = float(page.rect.height)

    for table in getattr(tables, "tables", []) or []:
        try:
            rows = table.extract()
        except Exception:
            continue
        markdown = rows_to_markdown(rows)
        if not markdown:
            continue
        bbox = [float(v) for v in getattr(table, "bbox", (0, 0, 0, 0))]
        out.append(
            Block(block_type=BLOCK_TABLE, text=markdown, page=page_index, order=0,
                  bbox=bbox,
                  meta={"rows": len(rows), "cols": len(rows[0]) if rows else 0,
                        "page_height": page_height})
        )
    return out


def rows_to_markdown(rows: List[List[Any]], max_rows: int = 500) -> str:
    """表格行 → Markdown。第一行作为表头；空单元格补空串。"""
    if not rows:
        return ""

    cleaned: List[List[str]] = []
    for row in rows[:max_rows]:
        cleaned.append([_cell_text(cell) for cell in (row or [])])

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    header = cleaned[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip().replace("|", "/")


# ------------------------------------------------------------------ 分类与结构
def _estimate_body_font_size(blocks: List[Block]) -> float:
    """以字符数加权的字号中位数作为正文字号基准。"""
    weighted: List[Tuple[float, int]] = [
        (b.max_font_size, len(b.text)) for b in blocks
        if b.max_font_size > 0 and b.block_type == BLOCK_PARAGRAPH
    ]
    if not weighted:
        return 0.0
    total_chars = sum(w for _, w in weighted)
    acc = 0
    for size, weight in sorted(weighted):
        acc += weight
        if acc >= total_chars / 2:
            return size
    return sorted(weighted)[-1][0]


def _classify_block(block: Block, body_size: float) -> None:
    """在版面位置与字号基础上判定块类型。页眉/页脚的最终去留交给清洗阶段按跨页频率决定。"""
    if not block.bbox:
        return
    # 原子块（表格/图片）的类型已由抽取阶段确定，不能被字号或位置规则改写：
    # 否则表格会退化成普通段落而失去「不切开」的语义，页面顶部的图会变成页眉。
    if block.is_atomic:
        return
    page_height = block.meta.get("page_height")

    if body_size > 0 and block.max_font_size >= body_size * 1.15 and len(block.text) <= 80 \
            and not _SENTENCE_END.search(block.text):
        block.block_type = BLOCK_TITLE
        ratio = block.max_font_size / body_size
        if ratio >= 1.6:
            block.level = 1
        elif ratio >= 1.35:
            block.level = 2
        else:
            block.level = 3
        return

    # 字号启发式之外的第二判据：**编号式标题**。
    # 真机缺陷：章节标题 16pt / 正文 11pt → ratio=1.4545 落在 [1.35,1.6) → level 2，
    # 而 level 2 在旧的分组逻辑里不被当作章节边界，6 章被压成 2 个父块。
    # 编号（"第 3 章" "3.1" "（三）"）是**语言/排版无关的强特征**，
    # 不该只靠字号比例去赌。层级由编号深度推断。
    if len(block.text) <= 60 and "\n" not in block.text \
            and not _SENTENCE_END.search(block.text) and _NUMBERED_HEADING.match(block.text):
        block.block_type = BLOCK_TITLE
        block.level = _level_from_numbering(block.text, body_size, block.max_font_size)
        block.meta["title_source"] = "numbering"
        return

    if _LIST_PATTERN.match(block.text):
        block.block_type = BLOCK_LIST
        return

    if isinstance(page_height, (int, float)) and page_height > 0:
        top, bottom = block.bbox[1], block.bbox[3]
        # 大字号（显著大于正文）的块**不作页眉判定**：它更可能是章节标题 / 提示框，
        # 而 6% 的相对位置阈值在页边距偏大的排版里会把正文顶部内容卷进来。
        # 被误判成 header 后会被 clean 阶段按跨页频率删除——正文内容就此静默消失。
        looks_like_heading = body_size > 0 and block.max_font_size >= body_size * 1.15
        if top < page_height * 0.06 and not looks_like_heading:
            block.block_type = BLOCK_HEADER
            return
        if bottom > page_height * 0.94:
            block.block_type = (
                BLOCK_PAGE_NUMBER if _PAGE_NUMBER_PATTERN.match(block.text) else BLOCK_FOOTER
            )
            return

    block.block_type = BLOCK_PARAGRAPH


def _build_section_tree(blocks: List[Block]) -> List[SectionNode]:
    """按标题层级构建章节树。"""
    roots: List[SectionNode] = []
    stack: List[SectionNode] = []

    for block in blocks:
        if block.block_type != BLOCK_TITLE:
            continue
        level = block.level or 1
        node = SectionNode(level=level, title=block.text.strip(), block_uid=block.uid)

        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def flatten_section_tree(tree: List[SectionNode]) -> List[List[str]]:
    """章节树 → 所有路径（供调试与评测使用）。"""
    out: List[List[str]] = []

    def walk(nodes: List[SectionNode], prefix: List[str]) -> None:
        for node in nodes:
            path = prefix + [node.title]
            out.append(path)
            walk(node.children, path)

    walk(tree, [])
    return out


# ------------------------------------------------------------------ 章节树质量诊断
# 编号式标题的强特征：中文"第 N 章/节/条/部分"、阿拉伯"N." / "N.N" / "N.N.N"、
# Markdown 风格已由解析器直接给出 level，不在此列。
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9一二三四五六七八九十百]+\s*[章节条篇部]"      # 第 3 章 / 第十二节
    r"|[0-9]{1,2}(?:\.[0-9]{1,2}){0,2}[.、)]?\s+\S"        # 3. / 3.1 / 3.1.2 后接文字
    r"|[（(][0-9一二三四五六七八九十]{1,3}[)）]\s*\S"        # （三）标题
    r")"
)


def _level_from_numbering(text: str, body_size: float, font_size: float) -> int:
    """由编号形态推断标题层级，字号只作辅助。

    约定：`第 N 章` / 单级 `N.` → 1 级；`N.N` → 2 级；`N.N.N` / `（三）` → 3 级。
    字号更大时**下调层级数字**（更靠上），但不会把一个明确的 `3.1.2` 提升成章级——
    编号深度比字号更可信。
    """
    stripped = text.strip()
    depth = 1
    if re.match(r"^\s*第\s*[0-9一二三四五六七八九十百]+\s*[章篇部]", stripped):
        depth = 1
    elif re.match(r"^\s*[（(][0-9一二三四五六七八九十]{1,3}[)）]", stripped):
        depth = 3
    else:
        dotted = re.match(r"^\s*([0-9]{1,2}(?:\.[0-9]{1,2}){0,2})", stripped)
        if dotted:
            depth = min(3, dotted.group(1).count(".") + 1)
    # 字号明显更大时上提一级（例如 "3.1 章节" 用了章级字号）
    if body_size > 0 and font_size >= body_size * 1.5:
        depth = max(1, depth - 1)
    return depth


def structured_heading_cues(blocks: Sequence[Block]) -> int:
    """统计**看起来像编号式标题**的块数量（不含已识别为 title 的）。

    这是"解析质量自检"的输入：一份文档若大量出现编号式标题行、
    却几乎没被识别为 title，说明字号启发式失效了。
    """
    count = 0
    for block in blocks:
        if block.block_type == BLOCK_TITLE:
            continue
        text = (block.text or "").strip()
        # 标题往往很短；长段落里出现"第 3 章"是正文引用，不算线索
        if not text or len(text) > 60 or "\n" in text:
            continue
        if _SENTENCE_END.search(text):
            continue
        if _NUMBERED_HEADING.match(text):
            count += 1
    return count


def section_tree_quality(parsed: ParsedDoc) -> Dict[str, Any]:
    """诊断章节树质量，供流水线如实记录与告警。

    **为什么需要它**：父块分组只在标题处按章节断开（`chunk.parent_break_level`）。
    当标题识别失效时，父块会退化成"按 token 预算硬切"，必然横跨章节——
    真机实测过 6 个章节被压成 2 个父块、每个父块含 3~4 章的内容，
    检索于是返回整节而非目标模块。这类退化**此前完全静默**。

    判定：编号式标题线索很多（>= 3）但识别出的 title 极少（< 线索数的 1/3）
    → 标 `heading_under_detected`，因为父块已不可能按章节切分。
    """
    titles = [b for b in parsed.blocks if b.block_type == BLOCK_TITLE]
    cues = structured_heading_cues(parsed.blocks)
    levels = sorted({(b.level or 1) for b in titles})
    roots = len(parsed.section_tree)

    issues: List[str] = []
    if cues >= 3 and len(titles) < max(1, cues // 3):
        issues.append("heading_under_detected")
    if not titles and cues == 0 and len(parsed.blocks) > 20:
        # 没有任何标题线索：可能是纯散文（正常），也可能是解析没抽出结构（异常）
        issues.append("no_heading_detected")

    return {
        "titles": len(titles),
        "levels": levels,
        "roots": roots,
        "numbered_cues": cues,
        "issues": issues,
        # 父块能否按章节切分的直接判据
        "parents_will_split_by_section": bool(titles) and not issues,
    }
