"""配置装载与策略路由。

配置即契约：切片策略、检索档位、模型注册表全部来自 configs/ 下的 YAML，
代码里不硬编码业务参数。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

# 项目根目录：kf_ingest/ -> services/ingest/ -> services/ -> 项目根
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def project_root() -> str:
    return os.environ.get("KF_PROJECT_ROOT", _DEFAULT_ROOT)


def config_dir() -> str:
    return os.path.join(project_root(), "configs")


@lru_cache(maxsize=8)
def load_yaml(name: str) -> Dict[str, Any]:
    """从 configs/ 装载 YAML（带缓存）。name 形如 'chunking_strategies.yaml'。"""
    import yaml

    path = os.path.join(config_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_chunking_config() -> Dict[str, Any]:
    return load_yaml("chunking_strategies.yaml")


def load_retrieval_config() -> Dict[str, Any]:
    return load_yaml("retrieval_profiles.yaml")


def load_model_registry() -> Dict[str, Any]:
    return load_yaml("model_registry.yaml")


def _match(cond: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """单条路由规则的匹配。全部条件必须同时成立。"""
    for key, expected in cond.items():
        actual = features.get(key)

        if isinstance(expected, bool):
            if bool(actual) != expected:
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif key.endswith("_in"):
            if actual not in (expected if isinstance(expected, list) else [expected]):
                return False
        else:
            if actual != expected:
                return False
    return True


def route_strategy(cfg: Dict[str, Any], features: Dict[str, Any]) -> str:
    """按 routing 规则选择切片策略（首个命中生效）。

    规则里可写 `default: <strategy>` 作为兜底（YAML 中最后一条），
    未命中任何规则时回退到 `default_strategy`。
    """
    for rule in cfg.get("routing", []):
        cond = rule.get("if") or {}
        if not cond:
            fallback = rule.get("default")
            if fallback:
                return fallback
            continue
        if _match(cond, features):
            return rule.get("use", cfg.get("default_strategy", "doc_structured"))
    return cfg.get("default_strategy", "doc_structured")


def strategy_params(cfg: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """合并 defaults 与具体策略的 params。"""
    defaults = dict(cfg.get("defaults") or {})
    params = dict((cfg.get("strategies", {}).get(strategy, {}) or {}).get("params") or {})
    merged = {**defaults, **params}
    return merged


def strategy_pipeline(cfg: Dict[str, Any], strategy: str) -> list:
    return list((cfg.get("strategies", {}).get(strategy, {}) or {}).get("pipeline") or [])


def profile(name: str, retrieval_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """取检索档位定义；未知档位回退到 safe_default。"""
    cfg = retrieval_cfg or load_retrieval_config()
    profiles = cfg.get("profiles") or {}
    if name in profiles:
        return dict(profiles[name])
    return dict(profiles.get(cfg.get("safe_default", "hybrid_rerank"), {}))
