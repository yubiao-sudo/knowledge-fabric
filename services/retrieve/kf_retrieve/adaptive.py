"""按可用显存自适应裁剪重排窗口。

## 为什么需要

cross-encoder 的开销随「候选数 × 序列长度²」增长，在 4GB 卡上实测：

| 配置 | 耗时 |
|---|---|
| 12 候选 × 512 token | 9.39s |
| 12 候选 × 256 token | 4.62s |
| 4 候选 × 256 token | **1.5s** |

小卡只能排 4 个候选，大卡却没必要这么抠——把窗口写死成 4 等于浪费算力。
这里按**总显存**分档（总显存比"空闲显存"稳定：空闲值受同机其他进程影响，
且 WDDM 驱动报告的数字明显偏保守）。

## 重要约束：换档位必须重标定

`rerank_top_n` 改变候选窗口 → cross-encoder 的 top-1 分数分布随之改变 →
拒答阈值必须跟着换。所以标定结果按**档位**分别存放（见
`configs/answerability_calibration.json` 的 `tiers`），降级顺序是：

1. 当前档位专属阈值（最准）
2. 全局底线 `assembly.answerability_floor_rerank`（保守，并在 trace 里标注未标定）

**绝不跨档位复用阈值**：那会让拒答行为在高档机器上系统性偏移。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Tier:
    """一个显存档位。"""

    name: str
    min_total_mb: int          # 进入该档位的最低总显存
    rerank_top_n: int          # 重排窗口（打分与返回共用）
    max_length: int            # cross-encoder 序列上限
    note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rerank_top_n": self.rerank_top_n,
            "max_length": self.max_length,
            "min_total_mb": self.min_total_mb,
            "note": self.note,
        }


# 按总显存升序匹配，取第一个满足 min_total_mb 的档位。
# 阈值来自真机实测（GTX 1650 4GB）：4 候选 ×256 token ≈ 1.5s，仍在 3s 预算内。
TIERS: tuple[Tier, ...] = (
    Tier("compact", 0, rerank_top_n=4, max_length=256,
         note="4GB 级显卡：重排窗口必须收紧，否则注意力爆显存"),
    Tier("standard", 6144, rerank_top_n=10, max_length=384,
         note="6~10GB：可放宽到 10 个候选、384 token"),
    Tier("wide", 10240, rerank_top_n=20, max_length=512,
         note="≥10GB：可用 20 个候选、512 token，排序质量最好"),
)

CPU_TIER = Tier("cpu", -1, rerank_top_n=2, max_length=256,
                note="无 GPU：cross-encoder 在 CPU 上约 3.4s/对，只能排 2 个")


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def detect(*, total_mb: Optional[int] = None,
           cuda_available: Optional[bool] = None) -> tuple[Tier, str]:
    """探测当前设备应使用的档位，返回 `(档位, 判定依据)`。

    参数仅用于测试注入；正常运行全部自动探测。
    """
    # 显式覆盖优先：让运维能在不改代码的情况下压测或规避故障。
    forced = _env("KF_RERANK_TOP_N", "")
    if forced:
        try:
            n = int(forced)
            if n > 0:
                tier = _tier_for_top_n(n)
                return tier, f"由 KF_RERANK_TOP_N={n} 强制指定（档位 {tier.name}）"
        except ValueError:
            pass

    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    if not cuda_available:
        return CPU_TIER, "未检测到可用 CUDA"

    if total_mb is None:
        try:
            import torch

            total_mb = int(torch.cuda.get_device_properties(0).total_memory // 1024 ** 2)
        except Exception as exc:
            return CPU_TIER, f"读取显存失败: {exc}"

    chosen = TIERS[0]      # min_total_mb=0，循环至少命中一次；先给默认值避免静态检查告警
    for tier in TIERS:
        if total_mb >= tier.min_total_mb:
            chosen = tier
    return chosen, f"总显存 {total_mb} MB → 档位 {chosen.name}"


def _tier_for_top_n(n: int) -> Tier:
    """把显式 top_n 映射到最接近的已知档位（用于阈值查找）。"""
    best = TIERS[0]
    for tier in TIERS:
        if tier.rerank_top_n == n:
            return tier
        if tier.rerank_top_n <= n:
            best = tier
    return Tier(f"custom_{n}", -1, rerank_top_n=n, max_length=best.max_length,
                note=f"自定义窗口 {n}")


_CACHE: Dict[str, Any] = {}


def current() -> tuple[Tier, str]:
    """进程内缓存一次（显存不会在运行中变大）。"""
    if "tier" not in _CACHE:
        _CACHE["tier"] = detect()
    return _CACHE["tier"]


def reset_cache() -> None:
    _CACHE.clear()


def effective_top_n(profile_value: Any) -> tuple[int, str, str]:
    """算出生效的重排窗口，返回 `(top_n, 档位名, 说明)`。

    **档位值优先于配置**：`configs/retrieval_profiles.yaml` 里的 `rerank_top_n`
    是 4GB 卡时代的经验值（4），若让它压住档位值，自适应就永远不生效——
    大卡上仍然只排 4 个候选，等于白装了一块好显卡。

    想固定窗口就设 `KF_RERANK_TOP_N=<n>`（`detect()` 会优先读它），
    那条路径才是"我就要这个数"的明确表达。
    """
    tier, reason = current()
    try:
        configured = int(profile_value) if profile_value is not None else 0
    except (TypeError, ValueError):
        configured = 0
    if configured and configured != tier.rerank_top_n:
        reason = (f"{reason}；配置值 {configured} 已被档位值 {tier.rerank_top_n} 覆盖"
                  f"（如需固定请设 KF_RERANK_TOP_N）")
    return tier.rerank_top_n, tier.name, reason


def status() -> Dict[str, Any]:
    tier, reason = current()
    out = tier.as_dict()
    out["reason"] = reason
    out["forced_by_env"] = bool(_env("KF_RERANK_TOP_N", ""))
    return out
