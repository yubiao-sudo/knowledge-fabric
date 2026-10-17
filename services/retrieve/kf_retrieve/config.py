"""检索服务配置装载。

配置即契约：检索档位、预算、融合权重全部来自 `configs/retrieval_profiles.yaml`，
代码里不硬编码业务参数。装载方式与 `services/ingest/kf_ingest/config.py` 保持一致
（`KF_PROJECT_ROOT` 覆盖 + `lru_cache`），让两侧共用同一份 YAML。

检索服务刻意不复用 ingest 的配置模块，避免 import 到 PyMuPDF 等重依赖。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

# 项目根目录：kf_retrieve/ -> services/retrieve/ -> services/ -> 项目根
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def project_root() -> str:
    return os.environ.get("KF_PROJECT_ROOT", _DEFAULT_ROOT)


def config_dir() -> str:
    return os.path.join(project_root(), "configs")


@lru_cache(maxsize=8)
def load_yaml(name: str) -> Dict[str, Any]:
    """从 configs/ 装载 YAML（带缓存）。name 形如 'retrieval_profiles.yaml'。"""
    import yaml

    path = os.path.join(config_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_retrieval_config() -> Dict[str, Any]:
    """装载检索档位配置；返回值只读，调用方需自建副本再改。"""
    return load_yaml("retrieval_profiles.yaml")


@lru_cache(maxsize=4)
def load_answerability_calibration() -> Dict[str, Any]:
    """按库标定的答案可达阈值（由 `scripts/calibrate_answerability.py` 生成）。

    为什么不写死一个全局阈值：实测同一套查询在不同规模的库上分数分布差别很大——
    644 块的 `acme/demo` 噪声底（域外探针的 p90）是 0.5493，而 16 块的库只有 0.3496。
    固定阈值要么在大库上过度拒答，要么在小库上放行越界问题。

    未标定的库返回空字典，调用方回退到 `assembly.answerability_floor`。
    """
    import json

    path = os.path.join(config_dir(), "answerability_calibration.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def answerability_threshold_for(tenant: str, kb: Optional[str],
                                basis: str = "cosine") -> Optional[float]:
    """取该 (tenant, kb) 在指定判据下的标定阈值；没有标定则返回 None（由调用方回退）。

    `basis`：
      * `"cosine"` —— embedding 余弦（默认判据）；
      * `"rerank"` —— cross-encoder 概率（分离度更好，但要求重排真的跑过）。

    两种判据的量纲与分布完全不同，必须各自标定、各自存阈值，不能混用。
    """
    table = (load_answerability_calibration().get("kb") or {})
    field = "rerank_threshold" if basis == "rerank" else "threshold"
    for key in ((f"{tenant}/{kb}" if kb else ""), f"{tenant}/default"):
        entry = table.get(key)
        if isinstance(entry, dict) and entry.get(field) is not None:
            try:
                return float(entry[field])
            except (TypeError, ValueError):
                return None
    return None
