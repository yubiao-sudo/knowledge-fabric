"""Token 计量。

切片阈值必须以**实际嵌入模型的 tokenizer** 计数，不能用字符数近似——
同一"字符长度"在中文、英文、日文下的 token 数差异显著。

优先使用 transformers 加载 BAAI/bge-m3 的 tokenizer；不可用时回退到
按语言加权的启发式估算（中文≈1 token/字，拉丁≈1.3 token/词）。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z0-9\u00c0-\u024f]+")
_PUNCT = re.compile(r"[^\sA-Za-z0-9\u00c0-\u024f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

# 显式禁用真实 tokenizer（离线环境 / 单测），统一走启发式估算
_HEURISTIC_ALIASES = frozenset({"", "heuristic", "none", "off"})


class TokenCounter:
    """token 计数器。优先真实 tokenizer，否则启发式。"""

    def __init__(self, model_id: str = "BAAI/bge-m3") -> None:
        self.model_id = model_id
        self.tokenizer = self._try_load(model_id)

    @staticmethod
    def _try_load(model_id: str):
        if not model_id or model_id.lower() in _HEURISTIC_ALIASES:
            return None
        if os.environ.get("KF_DISABLE_TOKENIZER"):
            return None
        try:
            from transformers import AutoTokenizer  # 延迟导入，避免强依赖

            return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except Exception:
            return None

    @property
    def backend(self) -> str:
        return f"tokenizer:{self.model_id}" if self.tokenizer is not None else "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return heuristic_count(text)

    def truncate(self, text: str, max_tokens: int) -> str:
        """按 token 预算截断（用于原子块超限时的兜底）。"""
        if max_tokens <= 0:
            return ""
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        # 启发式：按比例粗略截断
        total = heuristic_count(text)
        if total <= max_tokens:
            return text
        ratio = max_tokens / total
        return text[: max(1, int(len(text) * ratio))]

    def tail(self, text: str, tokens: int) -> str:
        """取尾部 N token（用于切片 overlap）。"""
        if tokens <= 0 or not text:
            return ""
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)[-tokens:]
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        ratio = min(1.0, tokens / max(1, heuristic_count(text)))
        return text[-max(1, int(len(text) * ratio)):]


def heuristic_count(text: str) -> int:
    """按语言加权的估算：CJK 每字 ≈ 1 token；拉丁每词 ≈ 1.3 token；标点 ≈ 0.5。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    punct = len(_PUNCT.findall(text))
    return int(cjk + words * 1.3 + punct * 0.5) + 1


@lru_cache(maxsize=4)
def default_counter(model_id: str = "BAAI/bge-m3") -> TokenCounter:
    return TokenCounter(model_id)


# 汉字的「信息密度权重」：一个汉字承载的信息约等于 3~5 个拉丁字母。
# 这样中英混排的技术文档（正文中文 + 大量英文标识符/代码/表格）才不会被判成 en。
CJK_LATIN_WEIGHT = 5
# 绝对量下限：只要出现这么多汉字，就认定该文本「中文主导」，**不再看比例**。
# 必要性：真机有一份中文设计文档（文件名中文、正文中文）因为夹杂大量代码，
# 全文汉字 6733 / 拉丁字母 73940，比例仅 0.09——只靠加权比例仍会被判成 en，
# 于是中文查询召不回它（稀疏通道按语言标签过滤）。绝对量是这里唯一可靠的信号。
HAN_ABSOLUTE_FLOOR = 300


def detect_lang(text: str) -> str:
    """轻量语言判定：返回 BCP-47 主语言标签。

    生产环境应替换为 fastText lid.176；这里只做区分中/日/韩/拉丁的快速判定，
    足以支撑"分语言稀疏索引路由"这一最小需求。

    ⚠️ 2026-10-17 修正（真机复现的缺陷）：原规则 `han >= latin` 按**原始字符数**
    比较，把中英混排的技术文档判成 `en`；稀疏通道按文档语言标签过滤，于是
    **中文查询召回为 0**（同一文档英文查询能召回）。两处修正：
      1. 汉字按 `CJK_LATIN_WEIGHT` 倍权重参与比较（抵消英文标识符/代码的字符密度优势）；
      2. 汉字绝对量 ≥ `HAN_ABSOLUTE_FLOOR` 即判 `zh`（应对代码占比极高、比例失衡的文档）。
    """
    if not text:
        return "unknown"
    hira_kata = len(re.findall(r"[\u3040-\u30ff]", text))
    hangul = len(re.findall(r"[\uac00-\ud7af]", text))
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    # 韩文按同样口径参与比较，避免「韩文 + 少量汉字」被判成 zh
    if hangul > max(han * CJK_LATIN_WEIGHT, latin, hira_kata):
        return "ko"
    if hira_kata > 0:
        return "ja"
    if han > 0 and (han * CJK_LATIN_WEIGHT >= latin or han >= HAN_ABSOLUTE_FLOOR):
        return "zh"
    if latin > 0:
        return "en"
    return "unknown"


def is_valid_embedding_lang(lang: Optional[str]) -> bool:
    return bool(lang) and lang != "unknown"
