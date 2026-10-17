"""查询理解：语言判定 + profile 驱动的查询增强 + 结构化精确识别。

设计红线：**增强步骤（HyDE / 多查询扩展 / 查询分解）只有在配置了生成端点
（`VLLM_URL` 或 `OPENAI_BASE_URL`）时才执行**；没有 LLM 时跳过并记 note，
绝不因为没有 LLM 就报错——检索链路必须始终可用。

语言判定复刻 `kf_ingest/tokenize.py:detect_lang` 的规则逻辑（检索服务不 import
kf_ingest），足以支撑「分语言稀疏索引路由」这一最小需求。

生成器以 `gen: Callable[[str], str]` 形式注入，便于单测 monkeypatch；
`default_generator()` 按环境变量构造，未配置返回 None。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ---- 与 kf_ingest/tokenize.py 等价的轻量语言判定 ----
def detect_lang(text: str) -> str:
    """返回 BCP-47 主语言标签：zh / ja / ko / en / unknown。"""
    if not text:
        return "unknown"
    hira_kata = len(re.findall(r"[\u3040-\u30ff]", text))
    hangul = len(re.findall(r"[\uac00-\ud7af]", text))
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    if hangul > max(han, latin, hira_kata):
        return "ko"
    if hira_kata > 0:
        return "ja"
    if han > 0 and han >= latin:
        return "zh"
    if latin > 0:
        return "en"
    return "unknown"


# ---- 结构化精确识别（正则即可）：型号 / SKU / 价格 / 数字 / 代码片段 ----
# SKU / 型号形如 MX-200、AB_123、XPS-13
_SKU = re.compile(r"\b[A-Z]{2,}[-_]?\d{2,}[A-Z0-9]*\b")
# 价格形如 1999 元 / ¥1999 / ￥1,999.00 / $1999 / 1999元
_PRICE = re.compile(r"(?:[¥￥$]\s?\d[\d,]*\.?\d*|\b\d[\d,]*\.?\d*\s?(?:元|块|美元|美金|dollars?|yuan)\b)", re.IGNORECASE)
_NUMBER = re.compile(r"\d")
_CODE_HINT = re.compile(r"[{}();=<>\+\-\*/]{2,}|\b(function|def|class|import|return|SELECT|FROM)\b")
# 常见单位/规格
_SPEC = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mm|cm|m|kg|g|gb|tb|mb|hz|ghz|w|v|a|mah|inch|寸)\b", re.IGNORECASE)


def extract_exact_terms(query: str) -> Dict[str, Any]:
    """识别查询里的型号 / 价格 / 数字 / 代码片段，供结构化过滤与稀疏加权。"""
    skus = _dedupe([m.group(0) for m in _SKU.finditer(query or "")])
    prices = _dedupe([m.group(0).strip() for m in _PRICE.finditer(query or "")])
    specs = _dedupe([m.group(0).strip() for m in _SPEC.finditer(query or "")])
    has_number = bool(_NUMBER.search(query or ""))
    has_code = bool(_CODE_HINT.search(query or ""))
    return {
        "skus": skus,
        "prices": prices,
        "specs": specs,
        "has_number": has_number,
        "has_code": has_code,
        "has_code_or_identifier": has_code or bool(skus),
        "has_number_or_price": has_number or bool(prices),
    }


@dataclass
class UnderstandResult:
    """查询理解的产物：原始查询 + 派生的检索用查询集合 + 结构化特征。"""

    query: str
    lang: str
    exact: Dict[str, Any] = field(default_factory=dict)
    hyde_doc: Optional[str] = None
    expansions: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def all_queries(self) -> List[str]:
        """送入召回的查询集合（原始 + HyDE + 扩展 + 分解，去重保序）。"""
        out = [self.query]
        if self.hyde_doc:
            out.append(self.hyde_doc)
        out.extend(self.expansions)
        out.extend(self.sub_queries)
        return _dedupe(out)


def understand(
    query: str,
    profile: Dict[str, Any],
    *,
    gen: Optional[Callable[[str], str]] = None,
    notes: Optional[List[str]] = None,
) -> UnderstandResult:
    """按档位做查询理解与增强。gen 为 None 时只做语言判定 + 结构化识别。

    参数 `gen` 是「prompt -> 文本」的生成函数；未提供时，凡需要生成的增强
    步骤（HyDE / 扩展 / 分解）一律跳过并记录 note，不报错。
    """
    result = UnderstandResult(query=query, lang=detect_lang(query))
    result.exact = extract_exact_terms(query)
    if notes is not None:
        result.notes = notes

    if profile.get("hyde"):
        _apply_hyde(result, gen)
    expansion = profile.get("query_expansion")
    if expansion:
        _apply_expansion(result, expansion, gen)
    if profile.get("decompose"):
        _apply_decompose(result, gen)

    if profile.get("agentic", {}).get("enabled") or profile.get("graph", {}).get("enabled"):
        # Agentic / Graph 的真实启用条件（档位 + 请求/kb 开关）由 service 层判定，
        # 此处不再臆测「未接入降级」，避免与实现语义相矛盾。
        pass
    return result


# ------------------------------------------------------------------ 增强步骤
def _apply_hyde(result: UnderstandResult, gen: Optional[Callable[[str], str]]) -> None:
    if gen is None:
        result.notes.append("HyDE 跳过：未配置生成端点（VLLM_URL / OPENAI_BASE_URL）")
        return
    prompt = (
        "假设你已知答案。请写一段 2-3 句的虚构文档片段，风格与目标知识库一致，"
        "用于提升稠密检索召回。只输出片段本身，不要解释。\n\n问题：" + result.query
    )
    doc = _safe_gen(gen, prompt)
    if doc:
        result.hyde_doc = doc
    else:
        result.notes.append("HyDE 生成失败，已回退原查询")


def _apply_expansion(result: UnderstandResult, expansion: Dict[str, Any],
                      gen: Optional[Callable[[str], str]]) -> None:
    n = int(expansion.get("n", 3) or 3)
    mode = str(expansion.get("mode", "paraphrase"))
    if gen is None:
        result.notes.append(f"多查询扩展({mode})跳过：未配置生成端点")
        return
    prompt = (
        f"用同义改写（{mode}）给出 {n} 个与下述问题等价但措辞不同的检索查询，"
        "每行一个，不要编号、不要解释。\n\n问题：" + result.query
    )
    text = _safe_gen(gen, prompt)
    if not text:
        result.notes.append("多查询扩展生成失败，已回退原查询")
        return
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    result.expansions = _dedupe([ln for ln in lines if ln and ln != result.query])[:n]


def _apply_decompose(result: UnderstandResult, gen: Optional[Callable[[str], str]]) -> None:
    if gen is None:
        result.notes.append("查询分解跳过：未配置生成端点")
        return
    prompt = (
        "把下述多跳问题拆成 1-3 个可独立检索的子问题，每行一个，不要编号、不要解释。\n\n"
        "问题：" + result.query
    )
    text = _safe_gen(gen, prompt)
    if not text:
        result.notes.append("查询分解生成失败，已回退原查询")
        return
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    result.sub_queries = _dedupe([ln for ln in lines if ln and ln != result.query])[:3]


def _safe_gen(gen: Callable[[str], str], prompt: str) -> str:
    try:
        return (gen(prompt) or "").strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ 默认生成器
def llm_base_url() -> str:
    """优先 VLLM_URL，其次 OPENAI_BASE_URL。未配置返回空串。"""
    return (os.environ.get("VLLM_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()


def llm_model_id() -> str:
    return (os.environ.get("VLLM_MODEL") or "Qwen/Qwen2.5-1.5B-Instruct").strip()


def default_generator(timeout: float = 30.0) -> Optional[Callable[[str], str]]:
    """按环境变量构造生成函数；未配置生成端点则返回 None。"""
    base = llm_base_url()
    if not base:
        return None

    def gen(prompt: str) -> str:
        import httpx

        payload = {
            "model": llm_model_id(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.2,
        }
        headers = {}
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = httpx.post(
            f"{base.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "")

    return gen


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
