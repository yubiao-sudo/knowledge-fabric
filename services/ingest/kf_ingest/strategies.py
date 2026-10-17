"""切片策略的「单元发现」（unit discovery）。

各切片策略的差异不在参数字面量，而在**子块单元的粒度**：

    策略             子块单元
    doc_structured   结构递归：句子累积到 child_tokens
    table_centric    表格按行组、正文按结构递归
    semantic         语义断点：相邻句相似度低谷处断开
    clause_atomic    条款原子：单条（第X条 / Article N / Section N / §N / 1.1）
    code_aware       语法单元：单函数 / 类 / 顶层导入块
    visual_page      整页（供页级视觉嵌入）

本模块只做「文本 → 单元」的纯计算，不涉及父块分组、chunk id 生成、provenance
与落库（那些在 `chunk.py`）。因此它是可单测、可复用的最小内核。

**语义断点的相似度有两个来源**，二者共用同一套断点与合并算法，只是相似度不同：

  - 调用方传入 `encoder`（TEI 嵌入）→ 真实向量余弦，跨语言、语义可靠；
  - 未传入 → 词法代理（与索引侧同源的稀疏分词做的 tf 余弦），零依赖可离线跑。

词法代理的断点质量弱于真实嵌入，但**方向正确且可复现**，且在无嵌入端点时不静默
退化为「整块不切」——由 `chunk.py` 把实际使用的来源写进切片统计。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# encoder: 一批文本 → 一批向量（与 TEIClient.embed 同形）
Encoder = Callable[[Sequence[str]], List[List[float]]]


# ------------------------------------------------------------------ 相似度
def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # 稀疏字典按较短一侧遍历，避免长文本 O(n*m) 退化
    left, right = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
    return max(0.0, min(1.0, dot))


def _cosine_dense(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _lexical_vectors(texts: Sequence[str]) -> List[Dict[str, float]]:
    """词法代理向量：与索引侧同源的稀疏分词 + tf 权重 + L2 归一。"""
    from .embed import tokenize_for_sparse

    out: List[Dict[str, float]] = []
    for text in texts:
        counts = Counter(tokenize_for_sparse(text))
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        out.append({term: value / norm for term, value in counts.items()})
    return out


def adjacent_similarities(
    sentences: Sequence[str], encoder: Optional[Encoder] = None
) -> List[float]:
    """相邻句相似度（长度 n-1）。encoder 不可用或返回异常时回退词法代理。"""
    if len(sentences) < 2:
        return []
    if encoder is not None:
        try:
            vectors = encoder(list(sentences))
        except Exception:
            vectors = None
        if vectors and len(vectors) == len(sentences):
            return [
                _cosine_dense(vectors[i], vectors[i + 1]) for i in range(len(sentences) - 1)
            ]
    vectors_sparse = _lexical_vectors(sentences)
    return [
        _cosine_sparse(vectors_sparse[i], vectors_sparse[i + 1])
        for i in range(len(sentences) - 1)
    ]


def quantile(values: Sequence[float], q: float) -> float:
    """线性插值分位数（numpy 口径），用于把「多大距离算断点」交给配置决定。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


# ------------------------------------------------------------------ 语义断点
def semantic_boundaries(
    sentences: Sequence[str],
    *,
    counter: Any,
    min_tokens: int = 200,
    max_tokens: int = 400,
    percentile: float = 0.85,
    encoder: Optional[Encoder] = None,
) -> List[int]:
    """返回单元起始下标（含 0，并以 n 收尾）。

    断点规则：相邻句距离（1 - 相似度）p 分位以上的间隙断开，再按
    `min_tokens`（不足则继续合并）与 `max_tokens`（超出则强制断开）收敛粒度。
    """
    total = len(sentences)
    if total == 0:
        return []
    if total == 1:
        return [0, 1]

    distances = [1.0 - value for value in adjacent_similarities(sentences, encoder)]
    threshold = quantile(distances, percentile)
    cuts = {index + 1 for index, distance in enumerate(distances) if distance > threshold}

    boundaries = [0]
    cursor = 0
    tokens = 0
    for index, sentence in enumerate(sentences):
        size = counter.count(sentence)
        if cursor < index:
            if index in cuts and tokens >= min_tokens:
                boundaries.append(index)
                cursor, tokens = index, 0
            elif tokens + size > max_tokens:
                boundaries.append(index)
                cursor, tokens = index, 0
        tokens += size
    boundaries.append(total)
    return boundaries


def semantic_units(
    sentences: Sequence[str],
    *,
    counter: Any,
    min_tokens: int = 200,
    max_tokens: int = 400,
    percentile: float = 0.85,
    encoder: Optional[Encoder] = None,
    joiner: str = "",
) -> List[str]:
    """语义断点切分的文本单元（`semantic_boundaries` 的文本化包装）。"""
    cleaned = [s.strip() for s in sentences if s and s.strip()]
    boundaries = semantic_boundaries(
        cleaned, counter=counter, min_tokens=min_tokens, max_tokens=max_tokens,
        percentile=percentile, encoder=encoder,
    )
    return [
        joiner.join(cleaned[boundaries[i]:boundaries[i + 1]]).strip()
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]


# ------------------------------------------------------------------ 条款原子
# 条款序号只认「行首」形式，避免正文里的 1.1 被误判为条款起点。
_CLAUSE_MARK = re.compile(
    r"(?m)^[ \t\u3000]*("
    r"第[一二三四五六七八九十百千零〇0-9]{1,6}[条章节款项]"
    r"|(?:ARTICLE|Article|article)\s+\d{1,3}[A-Za-z]?"
    r"|(?:SECTION|Section|section)\s+\d{1,3}(?:\.\d{1,3})*"
    r"|§\s*\d{1,3}(?:\.\d{1,3})*"
    r"|\d{1,2}(?:\.\d{1,2}){1,3}"
    r")"
)


def clause_spans(text: str) -> List[Tuple[int, int, int]]:
    """条款切分区间：(起点, 序号之后的位置, 终点)。起点含序号，便于 `keep_numbering`。

    识别不到 2 处以上条款序号时返回空列表——**不猜**，交由调用方回退整块。
    条款前的引言（如标题、颁布信息）作为独立单元保留，不并入第一条。
    """
    if not text or not text.strip():
        return []
    matches = list(_CLAUSE_MARK.finditer(text))
    if len(matches) < 2:
        return []

    spans: List[Tuple[int, int, int]] = []
    if text[: matches[0].start()].strip():
        # 条款前的引言（标题 / 颁布信息等）单独成单元，不并入第一条
        spans.append((0, 0, matches[0].start()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append((match.start(), match.end(), end))
    return spans


def clause_units(text: str, *, keep_numbering: bool = True) -> List[str]:
    """条款文本单元。`keep_numbering=False` 时剥掉序号（逐条引用仍可按 order 定位）。"""
    out: List[str] = []
    for start, body_start, end in clause_spans(text):
        piece = text[start:end] if keep_numbering else text[body_start:end]
        piece = piece.strip()
        if piece:
            out.append(piece)
    return out or ([text.strip()] if text and text.strip() else [])


# ------------------------------------------------------------------ 代码语法单元
_CODE_FUNC = (
    r"(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+)?"
    r"(?:def|fn|func|function)\s+[A-Za-z_]"
)
_CODE_CLASS = r"(?:pub\s+)?(?:class|struct|impl|enum|trait|interface)\s+[A-Za-z_]"
_CODE_MODULE = r"#(?:include|import|define|pragma)\b|(?:package|namespace|module)\s+\S"


def _unit_alternatives(unit: str) -> List[str]:
    key = (unit or "").strip().lower()
    if key in ("function", "def", "fn", "func", "method"):
        return [_CODE_FUNC]
    if key in ("class", "type", "struct", "impl", "interface"):
        return [_CODE_CLASS]
    if key in ("module", "file", "import", "package", "header"):
        return [_CODE_MODULE]
    return [_CODE_FUNC, _CODE_CLASS, _CODE_MODULE]


def code_spans(text: str, *, unit: str = "function") -> List[Tuple[int, int]]:
    """语法单元区间（起点, 终点）。行首锚定，避免调用点被误判为定义。"""
    if not text or not text.strip():
        return []
    pattern = re.compile(r"(?m)^[ \t]*(?:" + "|".join(_unit_alternatives(unit)) + r")")
    marks = [match.start() for match in pattern.finditer(text)]
    if not marks:
        return []

    spans: List[Tuple[int, int]] = []
    head = text[: marks[0]]
    if head.strip():
        spans.append((0, marks[0]))
    for index, start in enumerate(marks):
        end = marks[index + 1] if index + 1 < len(marks) else len(text)
        spans.append((start, end))
    return spans


def code_units(text: str, *, unit: str = "function", keep_indent: bool = True) -> List[str]:
    """代码文本单元。`keep_indent=False` 时对每个单元做行内去缩进（便于嵌入模型）。"""
    out: List[str] = []
    for start, end in code_spans(text, unit=unit):
        piece = text[start:end]
        if not keep_indent:
            piece = "\n".join(line.strip() for line in piece.split("\n"))
        piece = piece.rstrip()
        if piece.strip():
            out.append(piece)
    return out or ([text.rstrip()] if text and text.strip() else [])


# ------------------------------------------------------------------ 表格结构化
_SEP_CELL = re.compile(r":?-{2,}:?")


def markdown_table_structured(markdown: str) -> Dict[str, Any]:
    """把 markdown 表格解析成结构化字段（供结构化通道与精确过滤使用）。

    只做**无损**转换：列名与单元格原样保留，不做类型推断——类型推断应由下游
    结构化通道按字段语义决定（价格 / 数量 / 日期各有各的规则）。
    """
    lines = [line.strip() for line in (markdown or "").split("\n") if line.strip()]
    if len(lines) < 2:
        return {}

    def cells(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    columns = cells(lines[0])
    rows = [cells(line) for line in lines[1:]]
    # 丢弃分隔行（|---|:--:|）
    rows = [row for row in rows if row and not all(_SEP_CELL.fullmatch(c or "-") for c in row)]
    return {"columns": columns, "rows": rows, "row_count": len(rows)}
