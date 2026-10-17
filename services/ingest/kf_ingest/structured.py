"""结构化双轨：从表格块与文本抽取 SKU / 名称 / 价格 / 币种 / 规格属性。

两条落点：

1. **Postgres `structured_records`**：补全 `tenant_id` / `kb_id` / `chunk_id` /
   `schema_id` / `sku` / `name` / `price` / `currency` / `attrs` / `payload` /
   `doc_id` / `version`，靠唯一键 `uq_struct_chunk_schema_sku`
   `(chunk_id, schema_id, COALESCE(sku,''))` 做 ON CONFLICT —— 重跑同一文档不产生重复行。
2. **Qdrant payload**：固定键名 `skus` / `price` / `currency` / `attrs`（契约第 2 节，键名不许改）。

抽取规则轻量、可解释、**不依赖 LLM**：表格块用列头 → 单元格映射；文本块用 SKU/价格形态正则。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import Chunk, IngestContext
from .strategies import markdown_table_structured

# ------------------------------------------------------------------ 形态识别
# SKU / 型号：2–6 个大写字母 + 可选分隔符 + ≥2 位数字（如 MX-100 / KX2000）。
# 用否定环视而非 \b：SKU 常与中文直接相连（"MX-100型设备"），\b 在 CJK 边界不成立。
_SKU_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,6}[-_]?\d{2,}[A-Za-z0-9]*(?![A-Za-z0-9])")

_CUR_SYM = {"¥": "CNY", "￥": "CNY", "$": "USD", "€": "EUR", "£": "GBP"}
_CUR_WORD = {
    "元": "CNY", "人民币": "CNY", "rmb": "CNY", "cny": "CNY",
    "美元": "USD", "美金": "USD", "usd": "USD",
    "欧元": "EUR", "eur": "EUR",
    "英镑": "GBP", "gbp": "GBP",
    "日元": "JPY", "円": "JPY", "jpy": "JPY",
    "港币": "HKD", "港元": "HKD", "hkd": "HKD",
}
_UNIT_MULT = {"万元": 10000.0, "万": 10000.0}

_PRICE_SYM_RE = re.compile(r"(?P<sym>[¥￥$€£])\s*(?P<amt>\d[\d,]*(?:\.\d+)?)")
_PRICE_UNIT_RE = re.compile(
    r"(?P<amt>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>"
    r"万元|元|人民币|美元|美金|欧元|英镑|日元|円|港币|港元|"
    r"RMB|CNY|USD|EUR|GBP|JPY|HKD)",
    re.IGNORECASE,
)

# 列头 → 角色
_SKU_HEADERS = {
    "型号", "型号规格", "产品型号", "sku", "料号", "货号", "编号", "产品编号",
    "零件号", "part", "partno", "partnumber", "model", "modelno", "item", "itemno",
}
_NAME_HEADERS = {
    "名称", "品名", "产品", "产品名称", "商品", "商品名称", "描述", "说明",
    "规格名称", "name", "product", "title",
}
_PRICE_HEADERS = {"价格", "单价", "售价", "报价", "金额", "定价", "price", "unitprice"}
_CUR_HEADERS = {"币种", "货币", "币别", "currency"}


def _to_amount(raw: str) -> Optional[float]:
    try:
        return float((raw or "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def _detect_currency(text: str) -> Optional[str]:
    if not text:
        return None
    for sym, cur in _CUR_SYM.items():
        if sym in text:
            return cur
    low = text.lower()
    for word, cur in _CUR_WORD.items():
        if word in low:
            return cur
    return None


def _find_price(text: str) -> Tuple[Optional[float], Optional[str]]:
    """返回 (数值价格, 币种)。优先「符号+数值」，其次「数值+单位」。"""
    for match in _PRICE_SYM_RE.finditer(text or ""):
        amount = _to_amount(match.group("amt"))
        if amount is not None:
            return amount, _CUR_SYM.get(match.group("sym"))
    for match in _PRICE_UNIT_RE.finditer(text or ""):
        amount = _to_amount(match.group("amt"))
        if amount is None:
            continue
        unit = match.group("unit")
        currency = _CUR_WORD.get(unit.lower()) or _CUR_WORD.get(unit)
        return amount * _UNIT_MULT.get(unit, 1.0), currency
    return None, _detect_currency(text or "")


def _dedupe(items: Sequence[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ------------------------------------------------------------------ 表格抽取
def _norm_header(header: str) -> str:
    return re.sub(r"[\s/()（）\[\]【】]", "", (header or "")).lower()


def _role_of(header: str) -> str:
    key = _norm_header(header)
    if key in _SKU_HEADERS:
        return "sku"
    if key in _NAME_HEADERS:
        return "name"
    if key in _PRICE_HEADERS:
        return "price"
    if key in _CUR_HEADERS:
        return "currency"
    return "attr"


def _column_roles(columns: Sequence[str]) -> Dict[int, str]:
    roles: Dict[int, str] = {}
    used: set = set()
    for index, column in enumerate(columns):
        role = _role_of(column)
        if role != "attr" and role in used:
            role = "attr"  # 同类列只认第一个，其余当规格属性
        roles[index] = role
        if role != "attr":
            used.add(role)
    return roles


def extract_from_table(markdown: str) -> Dict[str, Any]:
    parsed = markdown_table_structured(markdown or "")
    columns = parsed.get("columns") or []
    rows = parsed.get("rows") or []
    if not columns or not rows:
        return extract_from_text(markdown or "")

    roles = _column_roles(columns)
    schema_id = "table:" + "|".join(str(c) for c in columns)
    skus: List[str] = []
    records: List[Dict[str, Any]] = []
    primary: Optional[Dict[str, Any]] = None

    for row in rows:
        joined = " ".join(str(c) for c in row)
        sku: Optional[str] = None
        name: Optional[str] = None
        price: Optional[float] = None
        currency: Optional[str] = None
        attrs: Dict[str, Any] = {}
        for index, column in enumerate(columns):
            cell = str(row[index]).strip() if index < len(row) else ""
            role = roles.get(index, "attr")
            if role == "sku" and not sku and cell:
                sku = cell
            elif role == "name" and not name and cell:
                name = cell
            elif role == "price" and price is None:
                price, currency = _find_price(cell)
            elif role == "currency" and not currency and cell:
                currency = _detect_currency(cell) or cell.upper()
            elif role == "attr" and cell:
                attrs[column] = cell
        if sku is None:
            found = _SKU_RE.search(joined)
            sku = found.group(0) if found else None
        if price is None:
            price, detected = _find_price(joined)
            currency = currency or detected
        if currency is None:
            currency = _detect_currency(joined)
        if sku:
            skus.append(sku)
        record = {
            "schema_id": schema_id, "sku": sku, "name": name, "price": price,
            "currency": currency, "attrs": attrs,
            "payload": {"columns": [str(c) for c in columns], "cells": [str(c) for c in row]},
        }
        records.append(record)
        if primary is None:
            primary = record

    return {
        "skus": _dedupe(skus),
        "price": primary.get("price") if primary else None,
        "currency": primary.get("currency") if primary else None,
        "attrs": dict(primary.get("attrs") or {}) if primary else {},
        "records": records,
    }


# ------------------------------------------------------------------ 文本抽取
def extract_from_text(text: str) -> Dict[str, Any]:
    text = text or ""
    skus = _dedupe(_SKU_RE.findall(text))
    price, currency = _find_price(text)
    records: List[Dict[str, Any]] = []
    if skus:
        for sku in skus:
            records.append({
                "schema_id": "inline", "sku": sku, "name": None, "price": price,
                "currency": currency, "attrs": {}, "payload": {"text": text[:200]},
            })
    elif price is not None:
        records.append({
            "schema_id": "inline", "sku": None, "name": None, "price": price,
            "currency": currency, "attrs": {}, "payload": {"text": text[:200]},
        })
    return {"skus": skus, "price": price, "currency": currency, "attrs": {}, "records": records}


def extract_chunk(chunk: Chunk) -> Dict[str, Any]:
    """抽取单个块的 payload 与 records（表格块走列头映射，其余走文本形态）。"""
    if chunk.chunk_type == "table":
        return extract_from_table(chunk.content or "")
    return extract_from_text(chunk.content or "")


# ------------------------------------------------------------------ 落库
def collect_records(ctx: IngestContext, chunks: Sequence[Chunk]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        info = extract_chunk(chunk)
        for record in info["records"]:
            out.append({
                **record,
                "tenant_id": ctx.tenant_id,
                "kb_id": ctx.kb_id,
                "chunk_id": chunk.chunk_id,
                "doc_id": ctx.doc_id,
                "version": ctx.version,
            })
    return out


def record_key(record: Dict[str, Any]) -> Tuple[Any, Any, str]:
    """与唯一键 `uq_struct_chunk_schema_sku` 对齐的去重键。"""
    return (record.get("chunk_id"), record.get("schema_id"), record.get("sku") or "")


_INSERT_RECORD = (
    "INSERT INTO structured_records"
    "(tenant_id, kb_id, chunk_id, schema_id, sku, name, price, currency, attrs, payload, doc_id, version) "
    "VALUES (%(tenant_id)s, %(kb_id)s, %(chunk_id)s, %(schema_id)s, %(sku)s, %(name)s, %(price)s, "
    "%(currency)s, %(attrs)s, %(payload)s, %(doc_id)s, %(version)s) "
    # 冲突目标带 tenant_id：chunk_id 只在本租户内唯一，不带租户会让第二个租户的
    # 同名块覆盖第一个租户的结构化记录（与主键缺租户是同一类缺陷）。
    "ON CONFLICT (tenant_id, chunk_id, schema_id, COALESCE(sku, '')) DO UPDATE SET "
    "name = EXCLUDED.name, price = EXCLUDED.price, currency = EXCLUDED.currency, "
    "attrs = EXCLUDED.attrs, payload = EXCLUDED.payload, "
    "kb_id = EXCLUDED.kb_id, doc_id = EXCLUDED.doc_id, version = EXCLUDED.version, "
    "updated_at = now()"
)


def write_structured_records(conn: Any, ctx: IngestContext, chunks: Sequence[Chunk]) -> int:
    """幂等写入 structured_records。同键（chunk_id, schema_id, sku）只保留一条。"""
    from psycopg.types.json import Jsonb

    records = collect_records(ctx, chunks)
    unique: Dict[Tuple[Any, Any, str], Dict[str, Any]] = {}
    for record in records:
        unique[record_key(record)] = record
    with conn.cursor() as cur:
        for record in unique.values():
            cur.execute(_INSERT_RECORD, {
                "tenant_id": record["tenant_id"],
                "kb_id": record["kb_id"],
                "chunk_id": record["chunk_id"],
                "schema_id": record["schema_id"],
                "sku": record["sku"],
                "name": record["name"],
                "price": record["price"],
                "currency": record["currency"],
                "attrs": Jsonb(record["attrs"] or {}),
                "payload": Jsonb(record["payload"] or {}),
                "doc_id": record["doc_id"],
                "version": record["version"],
            })
    return len(unique)
