"""档位解析：读 `configs/retrieval_profiles.yaml`，合并默认值，safe_default 回退。

- 8 个 profile 名、`global_budget` 硬限、`fusion_weights` 全部来自配置文件，**不硬编码**。
- 未知名一律回退到 `safe_default`（YAML 里的 hybrid_rerank），并置 `known=False`。
- 每个档位把缺省字段补齐成一份自包含的解析结果，编排层只管读，不再做防御性取值。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import load_retrieval_config

# 档位字段默认值（与 retrieval_profiles.yaml 的语义对齐）
PROFILE_DEFAULTS: Dict[str, Any] = {
    "channels": ["dense", "sparse"],
    "top_k": 20,
    "rerank": False,
    "rerank_top_n": 12,
    "fusion": "none",
    "fusion_k": 60,
    "sparse_boost": 1.0,
    "hyde": False,
    "query_expansion": None,
    "decompose": False,
    "agentic": {"enabled": False, "max_rounds": 3},
    "graph": {"enabled": False, "mode": "entity", "max_hops": 2},
    # Agentic 循环参数：新增块数阈值（饱和判定）与累计 token 预算
    "min_new_evidence": 1,
    "budget_tokens": 8000,
    "answerability_threshold": None,
    "on_low_score": None,
    "timeout_ms": 1500,
    "filters": [],
}

# global_budget 缺省（YAML 缺失时兜底，正常路径一律读配置）
BUDGET_DEFAULTS: Dict[str, Any] = {
    "max_latency_ms": 3000,
    "max_context_tokens": 8000,
    "max_agentic_rounds": 3,
}

# 融合权重缺省（YAML 缺失时兜底）
FUSION_WEIGHT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "default": {"dense": 0.6, "sparse": 0.4},
}


def safe_default_name(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or load_retrieval_config()
    return cfg.get("safe_default") or "hybrid_rerank"


def profile_names(cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or load_retrieval_config()
    return list((cfg.get("profiles") or {}).keys())


def budget_of(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """global_budget：单查询端到端硬上限。"""
    cfg = cfg or load_retrieval_config()
    merged = dict(BUDGET_DEFAULTS)
    merged.update(cfg.get("global_budget") or {})
    return merged


def fusion_weights_for(profile_name: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """按档位取融合权重：命中同名则用，否则用 default。"""
    cfg = cfg or load_retrieval_config()
    table = dict(cfg.get("fusion_weights") or FUSION_WEIGHT_DEFAULTS)
    weights = table.get(profile_name) or table.get("default") or FUSION_WEIGHT_DEFAULTS["default"]
    return {k: float(v) for k, v in weights.items()}


def resolve_profile(name: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """解析档位：合并默认值 + safe_default 回退。

    返回的字典额外带 `name` / `known` / `safe_default` / `budget` / `fusion_weights`，
    供编排层直接使用。未知档位名会落到 safe_default 并标记 `known=False`。
    """
    cfg = cfg or load_retrieval_config()
    profiles = cfg.get("profiles") or {}
    safe_name = safe_default_name(cfg)

    known = name in profiles
    raw = dict(profiles.get(name) or profiles.get(safe_name) or {})

    out = dict(PROFILE_DEFAULTS)
    for key, value in raw.items():
        if key in ("agentic", "graph") and isinstance(value, dict):
            base = dict(PROFILE_DEFAULTS.get(key) or {})
            # 配置里给出了该能力的具体参数但未写 enabled 时，视为已启用
            if value and "enabled" not in value:
                base["enabled"] = True
            base.update(value)
            out[key] = base
        else:
            out[key] = value

    channels = out.get("channels") or []
    out["channels"] = list(channels)
    out["name"] = name if known else safe_name
    out["requested"] = name
    out["known"] = known
    out["safe_default"] = safe_name
    out["budget"] = budget_of(cfg)
    out["fusion_weights"] = fusion_weights_for(out["name"], cfg)
    return out


def profiles_summary(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`GET /v1/profiles` 的载荷：档位名 + global_budget（读自配置文件）。"""
    cfg = cfg or load_retrieval_config()
    profiles = sorted((cfg.get("profiles") or {}).keys())
    return {
        "profiles": profiles,
        "safe_default": safe_default_name(cfg),
        "global_budget": budget_of(cfg),
        "fusion_weights": cfg.get("fusion_weights") or FUSION_WEIGHT_DEFAULTS,
    }
