"""清洗：洗掉无关信息（页眉 / 页脚 / 页码 / 水印 / 模板噪声），并且**必须可逆**。

两条设计红线：

1. **跨页重复检测优先于关键词匹配**。
   做法：把每页的候选行按"页面内相对纵向位置"归入上下边距带，对文本做规范化
   （数字统一替换为 #、去空白、小写），统计同一规范化键出现在多少个**不同页**上。
   超过阈值（默认 50%）即判定为 running header / footer / 页码，整类移除。
   这比"匹配关键词『第 X 页』"稳健得多，也不会误删正文里偶发的相同短语。

2. **可逆性**。
   每个被保留的块都附带溯源信息（来源页 + 字符区间 + 已执行的清洗操作），
   并保留原始文本片段，支持一键还原原文用于审计。

规则覆盖不到的水印与声明，由 model_registry.yaml 中的 bge-small 语义聚类补位
（见 `detect_semantic_boilerplate` 的接口说明）。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import (
    BLOCK_FOOTER,
    BLOCK_HEADER,
    BLOCK_PAGE_NUMBER,
    BLOCK_TABLE,
    BLOCK_TITLE,
    BOILERPLATE_BLOCK_TYPES,
    ATOMIC_BLOCK_TYPES,
    Block,
    Chunk,
    CleaningReport,
    ParsedDoc,
    ProvenanceSpan,
)

DEFAULT_TOP_RATIO = 0.10
DEFAULT_BOTTOM_RATIO = 0.10
DEFAULT_MIN_PAGE_RATIO = 0.50
SIMHASH_MAX_DISTANCE = 3

_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
_INLINE_WS = re.compile(r"[ \t\u00a0\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")
_DIGIT_RUN = re.compile(r"\d+")
_PAGE_NUMBER_KEY = re.compile(r"^[#\-–—·•*]+$")


# ------------------------------------------------------------------ 文本规范化
def normalize_text(text: str) -> str:
    """NFKC 归一 + 断字连字合并 + 空白规整 + 全半角统一。"""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _HYPHEN_BREAK.sub(r"\1\2", out)          # PDF 换行断字：exam-\\nple -> example
    lines = [_INLINE_WS.sub(" ", line).strip() for line in out.split("\n")]
    lines = [line for line in lines if line]
    return _MULTI_NL.sub("\n\n", "\n".join(lines)).strip()


def boilerplate_key(text: str) -> str:
    """跨页比较用的规范化键：小写、数字统一替换为 #、去除所有空白。"""
    norm = normalize_text(text).lower()
    norm = _DIGIT_RUN.sub("#", norm)
    return re.sub(r"\s+", "", norm)


# ------------------------------------------------------------------ SimHash
def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text.lower())


def simhash(text: str, shingle: int = 3, bits: int = 64) -> int:
    """字符/词混元 shingle 的 64 位 SimHash，用于近似重复检测。"""
    toks = _tokens(text)
    if not toks:
        return 0
    grams = [" ".join(toks[i:i + shingle]) for i in range(max(1, len(toks) - shingle + 1))]
    vector = [0] * bits
    for gram in grams:
        h = _hash64(gram)
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------ 跨页样板检测
def detect_running_boilerplate(
    blocks: Sequence[Block],
    page_heights: Sequence[float],
    top_ratio: float = DEFAULT_TOP_RATIO,
    bottom_ratio: float = DEFAULT_BOTTOM_RATIO,
    min_page_ratio: float = DEFAULT_MIN_PAGE_RATIO,
) -> Tuple[Set[str], Dict[str, int]]:
    """返回 (应移除的 block uid 集合, 命中的规范化键 -> 出现页数)。"""
    total_pages = len(page_heights)
    if total_pages == 0:
        return set(), {}

    candidates: List[Tuple[str, str, Block]] = []
    for block in blocks:
        in_margin = False
        if block.bbox and total_pages > 0:
            height = page_heights[block.page] if block.page < total_pages else 0.0
            if height > 0:
                top, bottom = block.bbox[1], block.bbox[3]
                if top < height * top_ratio or bottom > height * (1.0 - bottom_ratio):
                    in_margin = True
        if not (in_margin or block.block_type in BOILERPLATE_BLOCK_TYPES):
            continue
        key = boilerplate_key(block.text)
        if key:
            candidates.append((block.uid, key, block))

    key_pages: Dict[str, Set[int]] = {}
    for _, key, block in candidates:
        key_pages.setdefault(key, set()).add(block.page)

    remove: Set[str] = set()
    hit: Dict[str, int] = {}

    # 规则一：跨页重复（≥ min_page_ratio 的页面出现同一规范化键）
    if total_pages >= 2:
        for key, pages in key_pages.items():
            if len(pages) / total_pages >= min_page_ratio:
                hit[key] = len(pages)
                for uid, k, _ in candidates:
                    if k == key:
                        remove.add(uid)

    # 规则二：边距带内的纯页码/纯分隔符块，一律移除
    for uid, key, block in candidates:
        if _PAGE_NUMBER_KEY.match(key) and "#" in key:
            remove.add(uid)
            hit.setdefault(key, len(key_pages.get(key, ())))

    return remove, hit


def detect_semantic_boilerplate(
    blocks: Sequence[Block],
    embed_fn=None,
    similarity_threshold: float = 0.92,
    min_occurrences: int = 3,
) -> Set[str]:
    """语义层面的跨页样板检测（水印、免责声明、抬头块）。

    规则之外的残余样板靠这里补位：对边距带候选块做嵌入，余弦相似度聚类，
    同簇出现在 ≥ min_occurrences 个不同页面的簇整体移除。

    embed_fn: Callable[[List[str]], List[List[float]]]，由 model_registry 的
              `boilerplate-bge-small` 提供；为 None 时本函数不生效（跳过即可）。
    """
    if embed_fn is None:
        return set()

    candidates = [b for b in blocks
                  if b.block_type in BOILERPLATE_BLOCK_TYPES or _in_margin(b)]
    if len(candidates) < min_occurrences:
        return set()

    try:
        vectors = embed_fn([b.text for b in candidates])
    except Exception:
        return set()

    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return num / (na * nb) if na and nb else 0.0

    clusters: List[Tuple[List[int], Set[int]]] = []
    for idx, vec in enumerate(vectors):
        placed = False
        for member_idx, pages in clusters:
            if cosine(vec, vectors[member_idx[0]]) >= similarity_threshold:
                member_idx.append(idx)
                pages.add(candidates[idx].page)
                placed = True
                break
        if not placed:
            clusters.append(([idx], {candidates[idx].page}))

    remove: Set[str] = set()
    for member_idx, pages in clusters:
        if len(pages) >= min_occurrences and len(member_idx) >= min_occurrences:
            for i in member_idx:
                remove.add(candidates[i].uid)
    return remove


def _in_margin(block: Block) -> bool:
    return "in_margin_band" in block.meta


# ------------------------------------------------------------------ 去重
def dedupe_blocks(
    blocks: Sequence[Block],
    max_distance: int = SIMHASH_MAX_DISTANCE,
    min_len_exact: int = 40,
    min_len_near: int = 120,
) -> Tuple[List[Block], int]:
    """精确去重（sha256）+ 近似去重（SimHash 汉明距离）。

    标题、表格、代码等原子/结构性块不参与去重——它们重复出现通常是有意为之。
    """
    seen_exact: Set[str] = set()
    seen_sim: List[int] = []
    kept: List[Block] = []
    removed = 0

    for block in blocks:
        norm = normalize_text(block.text)
        skip = block.block_type == BLOCK_TITLE or block.block_type in ATOMIC_BLOCK_TYPES
        if skip or len(norm) < min_len_exact:
            kept.append(block)
            continue

        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if digest in seen_exact:
            removed += 1
            continue
        seen_exact.add(digest)

        if len(norm) >= min_len_near:
            sh = simhash(norm)
            if any(hamming(sh, prev) <= max_distance for prev in seen_sim):
                removed += 1
                continue
            seen_sim.append(sh)

        kept.append(block)

    return kept, removed


# ------------------------------------------------------------------ 主流程
def clean_document(
    parsed: ParsedDoc,
    *,
    remove_boilerplate: bool = True,
    dedupe: bool = True,
    top_ratio: float = DEFAULT_TOP_RATIO,
    bottom_ratio: float = DEFAULT_BOTTOM_RATIO,
    min_page_ratio: float = DEFAULT_MIN_PAGE_RATIO,
    embed_fn=None,
) -> Tuple[List[Block], CleaningReport]:
    """执行清洗，返回 (保留的块, 审计报告)。

    每个被保留块的 meta["cleaning_ops"] 记录实际执行过的操作，供溯源与一键还原。
    """
    report = CleaningReport(total_blocks=len(parsed.blocks), page_count=parsed.page_count())
    page_heights = [size[1] for size in parsed.page_sizes]

    remove_uids: Set[str] = set()
    if remove_boilerplate:
        by_key, hit = detect_running_boilerplate(
            parsed.blocks, page_heights, top_ratio, bottom_ratio, min_page_ratio
        )
        remove_uids |= by_key
        report.boilerplate_keys = sorted(hit.keys())
        if embed_fn is not None:
            remove_uids |= detect_semantic_boilerplate(parsed.blocks, embed_fn=embed_fn)

    kept: List[Block] = []
    for block in parsed.blocks:
        if block.uid in remove_uids:
            report.note_removed(block.block_type, block.text)
            continue

        norm = normalize_text(block.text)
        if not norm:
            report.note_removed(block.block_type, block.text)
            continue

        ops = list(block.meta.get("cleaning_ops") or [])
        if norm != block.text:
            ops.append("normalize")
        # 保存原始文本，支持一键还原
        block.meta["raw_text"] = block.text
        block.meta["cleaning_ops"] = ops
        block.text = norm
        kept.append(block)

    if dedupe:
        kept, removed = dedupe_blocks(kept)
        report.deduped_blocks = removed

    return kept, report


# ------------------------------------------------------------------ 溯源
def _norm_with_index(text: str) -> Tuple[str, List[int]]:
    """去掉空白并小写，同时维护"规范化位置 -> 原文位置"的映射。"""
    chars: List[str] = []
    index: List[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(ch.lower())
        index.append(i)
    return "".join(chars), index


def compute_provenance(
    block: Block,
    raw_pages: Sequence[str],
    source_uri: str,
) -> ProvenanceSpan:
    """计算块在原文中的字符区间。

    通过"去空白规范化 + 位置映射"定位，避免因换行/空白差异导致偏移错位。
    定位失败时退化为整页区间，并在 cleaning_ops 中标记 provenance_approx，
    绝不静默给出错误偏移。
    """
    page_text = raw_pages[block.page] if 0 <= block.page < len(raw_pages) else ""
    ops = list(block.meta.get("cleaning_ops") or [])

    norm_page, index = _norm_with_index(page_text)
    norm_block, _ = _norm_with_index(block.text)

    if norm_block and norm_page:
        pos = norm_page.find(norm_block)
        if pos >= 0:
            start = index[pos]
            end = index[pos + len(norm_block) - 1] + 1
            return ProvenanceSpan(
                source_uri=source_uri,
                page=block.page,
                char_start=start,
                char_end=end,
                bbox=block.bbox,
                cleaning_ops=ops,
                original_text=page_text[start:end],
            )

    ops.append("provenance_approx")
    return ProvenanceSpan(
        source_uri=source_uri,
        page=block.page,
        char_start=0,
        char_end=len(page_text),
        bbox=block.bbox,
        cleaning_ops=ops,
        original_text=page_text,
    )


# ------------------------------------------------------------------ 块级溯源（切片后）
# 为什么不能只靠 compute_provenance：
#   `compute_provenance` 按**单个块**去页文本里定位，而切片产出的是**跨块的拼接文本**，
#   拼接串在页文本里根本找不到 → 全部退化 `provenance_approx` + 整页区间。
#   实测该退化率高达 86%（102 条溯源里 88 条是整页区间），"点击回原文"因此失准。
# 正确做法：拿到切片内容后，沿着**跨越的页**在原文里定位，并维护跨页偏移映射。
# 可接受的最短锚点（规范化后字符数）。短于此的"命中"没有区分度——
# 实测 5 个字符的锚点会在无关页面上假阳性命中（如"页面内容"），
# 宁可如实标记 provenance_approx，也不给一个看似精确却错误的偏移。
# 与 common.py 的 simhash 最小长度（120）相比这里可以小得多，因为要求的是
# **连续**命中而非单个 n-gram，但仍须高于常见词的字符规模。
MIN_ANCHOR_CHARS = 20
# 最长可定位前缀搜索的窗口。早期实现按**字符比例**取前缀（90%/75%/…/10%），
# 实测有两个方向的失效：长块取到很长的前缀而匹配不上 → 退化成整页区间；
# 短块取到几字符的前缀 → 命中无关位置。改为按**窗口大小**递降搜索，
# 并以"至少覆盖 min(MIN_ANCHOR_CHARS, 长度)"约束可信度。
_PREFIX_WINDOWS: Tuple[int, ...] = (2000, 1000, 600, 400, 240, 160, 100, MIN_ANCHOR_CHARS)


def _concat_pages(raw_pages: Sequence[str], start: int, end: int) -> Tuple[str, List[int]]:
    """拼接 raw_pages[start..end]（含端点），返回 (拼接文本, 每页在拼接串中的起始偏移)。"""
    pieces: List[str] = []
    offsets: List[int] = []
    cursor = 0
    for page in range(start, end + 1):
        offsets.append(cursor)
        text = raw_pages[page] if 0 <= page < len(raw_pages) else ""
        pieces.append(text)
        cursor += len(text)
    return "".join(pieces), offsets


def _page_of_offset(offsets: Sequence[int], offset: int) -> int:
    """给定拼接串内的偏移，返回它落在第几页（offsets 的下标）。"""
    found = 0
    for index, start in enumerate(offsets):
        if start <= offset:
            found = index
        else:
            break
    return found


_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}")
_WORD_RUN = re.compile(r"[a-z0-9]+")


def _anchors(text: str) -> List[str]:
    """定位锚点：CJK bigram + 拉丁词。

    为什么不用整段/前缀子串：清洗阶段的 `normalize_text` 会做 NFKC 归一
    （全角→半角、标点统一），因此**块文本与原始页文本在字符层面已经不同**，
    实测整段规范化匹配对页 3~5 的块全部失配。bigram 对这类差异天然稳健
    ——实测同一批块 bigram 命中率 100%。
    """
    out: List[str] = []
    for token in _WORD_RUN.findall(text.lower()):
        out.append(token)
    for run in _CJK_RUN.findall(text):
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def _locate(chunk_text: str, page_text: str,
            norm_page: Optional[Tuple[str, List[int]]] = None) -> Optional[Tuple[int, int, int]]:
    """把切片内容定位到页文本，返回 `(char_start, char_end, anchored_chars)`。

    三级策略（逐级放宽）：
      1. **精确子串命中** —— 最理想，区间覆盖整块；
      2. **规范化整段命中**（去空白 + 小写）—— 应对 PDF 抽文本的空格/换行差异；
      3. **最长可定位前缀**（按窗口递降）—— 缓解轻度差异；
      4. **bigram 锚点** —— 应对 NFKC 等字符级差异：取块内按序第一个能在
         页内找到的 bigram/词作为起点，终点按块长比例估算并标注为估计值。

    第 3、4 级都只保证**起点可靠**，调用方据此标注 `provenance_partial`；
    一级都失败则返回 None，由调用方标记 `provenance_approx` 并退回页内区间
    ——缺偏移好过给一个指向无关位置的"精确"偏移。

    `norm_page` 可传入已算好的 `(规范化文本, 位置索引)`，避免对每个切片重复
    规范化整页（那是二次复杂度，见 `compute_chunk_provenance` 的说明）。
    """
    if not chunk_text or not page_text:
        return None
    pos = page_text.find(chunk_text)
    if pos >= 0:
        return pos, pos + len(chunk_text), len(chunk_text)

    if norm_page is None:
        norm_page = _norm_with_index(page_text)
    norm_text, index = norm_page
    if not norm_text:
        return None

    norm_chunk, _ = _norm_with_index(chunk_text)
    if norm_chunk:
        found = norm_text.find(norm_chunk)
        if found >= 0:
            start = index[found]
            end = index[min(found + len(norm_chunk) - 1, len(index) - 1)] + 1
            return start, end, len(norm_chunk)

        min_anchor = max(1, min(MIN_ANCHOR_CHARS, len(norm_chunk) // 10))
        for window in _PREFIX_WINDOWS:
            take = min(window, len(norm_chunk))
            if take < min_anchor:
                break
            found = norm_text.find(norm_chunk[:take])
            if found >= 0:
                start = index[found]
                end = index[min(found + take - 1, len(index) - 1)] + 1
                return start, end, take

    # 第 4 级：bigram 锚点（跨 NFKC 差异）。
    # 关键约束：必须是**连续窗口**命中，不能靠零散 bigram——"内容""系统"这类
    # 常见词会在无关页面上偶然命中，实测会给出看似精确的错误偏移（假阳性）。
    grams = _anchors(chunk_text)
    if grams:
        hits: List[int] = []
        for gram in grams[:400]:
            hits.append(norm_text.find(gram))
        run_start = -1
        run_len = 0
        best_start = best_len = -1
        for i, found in enumerate(hits):
            if found >= 0:
                if run_start < 0:
                    run_start = i
                run_len += 1
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
            else:
                run_start, run_len = -1, 0
        min_run = max(1, min(MIN_ANCHOR_CHARS, len(grams) // 10))
        if best_len >= min_run:
            start_norm = hits[best_start]
            if start_norm >= 0:
                start = index[start_norm]
                span = max(1, min(len(norm_chunk) or len(grams), len(norm_text) - start_norm))
                end_norm = min(start_norm + span - 1, len(index) - 1)
                end = index[end_norm] + 1
                return start, end, span
    return None


def compute_chunk_provenance(
    chunk: Chunk,
    raw_pages: Sequence[str],
    source_uri: str,
    *,
    page_hint: int = 0,
    norm_cache: Optional[Dict[int, Tuple[str, List[int]]]] = None,
) -> ProvenanceSpan:
    """计算**切片**在原文中的字符区间（跨页安全）。

    返回的 `char_start` / `char_end` 是**锚定页**（chunk.page）内的偏移，
    与 `models.ProvenanceSpan` 的既有语义一致；跨页切片另在 cleaning_ops 里
    记录 `provenance_end_page=N`，下游可据此判断区间是否越出单页。
    定位失败才退化为整页区间并标 `provenance_approx`——“缺偏移好过给错偏移”。

    `norm_cache` 用于**按页缓存**"去空白规范化 + 位置映射"的结果。这一步是 O(页字符)，
    而它原本对每个切片重算一次 → 总共 O(切片数 × 页字符)，即**二次复杂度**。
    真机实测（145k 字符的 Markdown、164 个切片）每次 142 ms、合计 23 s，
    且内存吃紧时进一步放大。调用方传入同一个 dict 即可把这块开销摊成 O(页字符)。
    """
    start_page = chunk.page if chunk.page is not None else page_hint
    if start_page < 0 or start_page >= len(raw_pages):
        start_page = max(0, min(page_hint, max(0, len(raw_pages) - 1)))
    end_page = min(len(raw_pages) - 1, start_page + 1) if raw_pages else start_page

    def _norm_of(page: int) -> Tuple[str, List[int]]:
        if norm_cache is not None and page in norm_cache:
            return norm_cache[page]
        text = raw_pages[page] if 0 <= page < len(raw_pages) else ""
        value = _norm_with_index(text)
        if norm_cache is not None:
            norm_cache[page] = value
        return value

    ops = ["provenance_chunk"]
    anchor_text = raw_pages[start_page] if 0 <= start_page < len(raw_pages) else ""
    located = _locate(chunk.content or "", anchor_text, norm_page=_norm_of(start_page))
    if located is None and end_page > start_page:
        merged, offsets = _concat_pages(raw_pages, start_page, end_page)
        merged_hit = _locate(chunk.content or "", merged)
        if merged_hit is not None:
            merged_start, merged_end, taken = merged_hit
            base = offsets[0]
            located = (merged_start - base, merged_end - base, taken)
            found_end_page = start_page + _page_of_offset(offsets, merged_end)
            if found_end_page != start_page:
                ops.append(f"provenance_end_page={found_end_page}")

    if located is not None:
        char_start, char_end, taken = located
        # 锚点没覆盖整块内容时如实标注：说明这是"起点可靠的局部区间"，
        # 而不是把整块都算进引用范围。
        norm_len = len(_norm_with_index(chunk.content or "")[0])
        if norm_len and taken < norm_len:
            ops.append(f"provenance_partial={taken}/{norm_len}")
        return ProvenanceSpan(
            source_uri=source_uri,
            page=start_page,
            char_start=int(char_start),
            char_end=int(char_end),
            cleaning_ops=ops,
            original_text=anchor_text[char_start:char_end],
        )

    ops.append("provenance_approx")
    return ProvenanceSpan(
        source_uri=source_uri,
        page=start_page,
        char_start=0,
        char_end=len(anchor_text),
        cleaning_ops=ops,
        original_text=anchor_text,
    )


def span_union(
    spans: Sequence[Optional[ProvenanceSpan]],
    source_uri: str,
) -> Optional[ProvenanceSpan]:
    """把多个子块的溯源合并成父块溯源。

    **偏移是页内偏移**，因此只有当子块都在同一页时首尾并集才有意义。
    真机实测过一个反例：父块横跨两页时，取首子块 `char_start` 与末子块
    `char_end` 会得到 `[653, 97]` 这种**倒置区间**（653 是第 4 页偏移、
    97 是第 5 页偏移），对外是非法数据。

    处理：同页取首尾并集；跨页时只保留**首页区间**（起点可靠），
    并把结束页写进 `provenance_end_page`，`char_end` 保证不小于 `char_start`。
    """
    valid = [s for s in spans if s is not None]
    if not valid:
        return None
    valid.sort(key=lambda s: (s.page, s.char_start))
    first, last = valid[0], valid[-1]
    approx = any("provenance_approx" in s.cleaning_ops for s in valid)

    if last.page == first.page:
        char_end = max(first.char_end, last.char_end)
        ops = ["provenance_union"]
    else:
        # 跨页：只保留首页区间，避免把不同页的页内偏移拼成非法区间
        same_page = [s for s in valid if s.page == first.page]
        char_end = max(s.char_end for s in same_page)
        ops = ["provenance_union", f"provenance_end_page={last.page}",
               "provenance_spans_pages"]
    if approx:
        ops.append("provenance_approx")
    return ProvenanceSpan(
        source_uri=source_uri,
        page=first.page,
        char_start=int(first.char_start),
        char_end=int(max(char_end, first.char_start)),
        cleaning_ops=ops,
        original_text=first.original_text,
    )
