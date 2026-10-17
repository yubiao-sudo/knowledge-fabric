"""块级 ACL 解析：`user:u1,group:g1,!user:u9` → 规则列表。

`!` 前缀 = deny。解析结果写入 `chunk_acl`（含 `tenant_id`）与 chunk 的 `acl`
字段（Qdrant payload 的 `acl` 沿用既有格式 `[{principal_type, principal_id, effect}]`）。

解析规则：

  - 逗号分隔；空白项跳过。
  - `!` 前缀标记 deny，其余为 allow。
  - `type:id`；没有 `:` 时默认 `principal_type=user`，整段作为 principal_id。
  - 同一 `(type, id)` 出现多次时 **deny 优先**（拒绝比允许更强）。
"""

from __future__ import annotations

from typing import Dict, List

DEFAULT_PRINCIPAL_TYPE = "user"


def parse_acl(spec: str) -> List[Dict[str, str]]:
    """把 ACL 字符串解析成规则列表（保持首次出现顺序，deny 覆盖 allow）。"""
    if not spec or not spec.strip():
        return []

    merged: Dict[tuple, Dict[str, str]] = {}
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        effect = "allow"
        if token.startswith("!"):
            effect = "deny"
            token = token[1:].strip()
            if not token:
                continue

        if ":" in token:
            ptype, pid = token.split(":", 1)
            ptype = ptype.strip().lower() or DEFAULT_PRINCIPAL_TYPE
            pid = pid.strip()
        else:
            ptype, pid = DEFAULT_PRINCIPAL_TYPE, token
        if not pid:
            continue

        key = (ptype, pid)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "principal_type": ptype,
                "principal_id": pid,
                "effect": effect,
            }
        elif effect == "deny":
            existing["effect"] = "deny"
    return list(merged.values())
