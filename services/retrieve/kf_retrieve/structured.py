"""结构化精确过滤通道：把查询里的结构化条件转成 payload 过滤。

契约第 3 节（`structured` 通道）：

  - 条件**只能**来自查询理解阶段已有的 `exact` 词（`query.extract_exact_terms`），
    **不得凭空生成**；识别不到任何条件时，通道返回空并记 note，禁止退化成再跑一遍 BM25。
  - 过滤维度（与 ingest 写入的 payload 键名一致，契约第 2 节）：
      * `skus`     string[]  → Qdrant MatchAny（任一命中）/ Memory 交集匹配
      * `price`    float     → Qdrant Range（精确值区间）/ Memory 数值区间
      * `currency` string    → Qdrant match / Memory 精确相等
  - 查询侧**不依赖 PG**（`structured_records` 仅用于摄入侧审计与 SQL 精确查询）。

本模块只做「条件识别 + 条件判定」两件纯计算；Qdrant / Memory 的具体执行分别落在
`backends.py` 里，保证两条后端路径语义等价。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 价格符号 → 币种；与摄入侧 `structured.py` 的映射保持一致
_CUR_SYM = {"¥": "CNY", "￥": "CNY", "$": "USD", "€": "EUR", "£": "GBP"}
# 价格单位词 → 币种
_CUR_WORD = {
    "元": "CNY", "块": "CNY", "人民币": "CNY", "rmb": "CNY", "cny": "CNY",
    "美元": "USD", "美金": "USD", "usd": "USD", "dollars": "USD", "dollar": "USD",
    "欧元": "EUR", "eur": "EUR",
    "英镑": "GBP", "gbp": "GBP",
}

_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)")


def parse_price(text: str) -> "tuple[Optional[float], Optional[str]]":
    """从单条价格字面量（如 `1999 元` / `¥1999` / `$1,999.00`）解析 (数值, 币种)。"""
    raw = text or ""
    amount = None
    match = _AMOUNT.search(raw)
    if match:
        try:
            amount = float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            amount = None
    currency: Optional[str] = None
    for sym, cur in _CUR_SYM.items():
        if sym in raw:
            currency = cur
            break
    if currency is None:
        low = raw.lower()
        for word, cur in _CUR_WORD.items():
            if word in low:
                currency = cur
                break
    return amount, currency


def structured_conditions(exact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 `exact` 词转成结构化过滤条件（**缺失即为空，绝不臆造**）。

    返回：`{"skus": [...], "price": {"gte": v, "lte": v} | None, "currency": "CNY" | None}`。
    `price` 用精确值区间包裹，语义等价于「精确价格匹配」（契约要求 price 用 Range）。
    """
    exact = exact or {}
    skus = _dedupe([str(s) for s in (exact.get("skus") or []) if str(s).strip()])
    price: Optional[Dict[str, float]] = None
    currency: Optional[str] = None
    for literal in exact.get("prices") or []:
        amount, cur = parse_price(str(literal))
        if amount is None:
            continue
        price = {"gte": amount, "lte": amount}
        currency = cur
        break
    return {"skus": skus, "price": price, "currency": currency}


def has_conditions(cond: Optional[Dict[str, Any]]) -> bool:
    """是否至少识别出一个结构化条件（skus 或 price）。"""
    if not cond:
        return False
    return bool(cond.get("skus")) or cond.get("price") is not None


def matches_conditions(chunk: Dict[str, Any], cond: Optional[Dict[str, Any]]) -> bool:
    """Memory 侧等价判定：按 payload 字段（skus / price / currency）精确匹配。

    与 Qdrant 的 MatchAny / Range / match 语义保持一致：
      - skus：块内任一 SKU 命中查询 SKU 即通过；
      - price：块价格落在查询价格区间内即通过；
      - currency：与查询币种精确相等即通过。
    """
    if not cond:
        return False
    skus = cond.get("skus") or []
    if skus:
        doc_skus = {str(s) for s in (chunk.get("skus") or [])}
        if not doc_skus.intersection({str(s) for s in skus}):
            return False
    price = cond.get("price")
    if price is not None:
        value = chunk.get("price")
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        low = _as_float(price.get("gte"))
        high = _as_float(price.get("lte"))
        if low is not None and numeric < low:
            return False
        if high is not None and numeric > high:
            return False
    currency = cond.get("currency")
    if currency:
        if str(chunk.get("currency") or "") != str(currency):
            return False
    return True


def qdrant_conditions(models: Any, cond: Optional[Dict[str, Any]]) -> List[Any]:
    """把结构化条件转成 Qdrant `FieldCondition` 列表（skus=MatchAny、price=Range、currency=match）。"""
    clauses: List[Any] = []
    if not cond:
        return clauses
    skus = cond.get("skus") or []
    if skus:
        clauses.append(models.FieldCondition(
            key="skus", match=models.MatchAny(any=[str(s) for s in skus])
        ))
    price = cond.get("price")
    if price is not None:
        clauses.append(models.FieldCondition(
            key="price", range=models.Range(gte=_as_float(price.get("gte")), lte=_as_float(price.get("lte")))
        ))
    currency = cond.get("currency")
    if currency:
        clauses.append(models.FieldCondition(key="currency", match=models.MatchValue(value=str(currency))))
    return clauses


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
