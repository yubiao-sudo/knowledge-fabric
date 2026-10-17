"""重排：cross-encoder（bge-reranker-v2-m3）客户端。

两条路径，按可用性自动选择：

1. `TEI_RERANK_URL` 已配置 → `POST /rerank`（TEI 服务化部署）；
2. 未配置 → **进程内本地重排**（`LocalReranker`），GPU 优先、fp16，缺 GPU 时回落 CPU。

两条都不可用 → **复用融合分**，`trace.reranked=false` 并记 note。绝不因重排不可用而失败。

## 为什么值得把重排真正跑起来（实测依据）

embedding 余弦（双塔）对"语义邻接的难负例"几乎没有分辨力：本语料上
可答 min=0.5865 vs 不可答 max=0.5529，**间隙仅 +0.0336**；
同一批查询换成 cross-encoder（单塔，query 与 doc 拼接后一起过模型）后，
原始分**可答 min=−1.695 vs 不可答 max=−2.200，间隙 +0.505**。
最难的那条负例「如何用 Rust 写一个 WebSocket 服务器」余弦 0.5529（几乎撞线），
cross-encoder 给出 −3.316，与最弱可答项 −1.695 明显分开。

延迟（真机 GTX 1650 4GB，fp16）：**12 个候选 0.91s（约 76ms/对）**，显存 1104 MB，
可与 bge-m3 嵌入模型（2257 MB）共存。CPU 上则是 33.8s，不可用——
所以本地重排**必须**在有 CUDA 时才有意义。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


def rerank_url() -> str:
    return (os.environ.get("TEI_RERANK_URL") or "").strip()


def rerank_model_id() -> str:
    return (os.environ.get("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3").strip()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _tier_max_length() -> int:
    """序列上限跟随显存档位（`adaptive.py`），可用 `KF_RERANK_MAX_LEN` 强制覆盖。

    bge-m3 那侧的实测（GTX 1650 4GB, fp16, 12 候选）：
    maxlen=512 → 9.39s，256 → 4.62s，128 → 2.29s；注意力中间量随 batch×seq² 增长。
    而 top-1 分数从 0.9004 只降到 0.8722——判"这段文字能否回答该问题"用 256 足够。
    """
    forced = _env("KF_RERANK_MAX_LEN", "")
    if forced:
        try:
            return max(64, int(forced))
        except ValueError:
            pass
    try:
        from .adaptive import current as current_tier

        return current_tier()[0].max_length
    except Exception:
        return 256


class RerankClient:
    """TEI `/rerank` 客户端。未配置地址时不要构造。"""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        if not base_url:
            raise ValueError("未配置 TEI_RERANK_URL")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rerank(self, query: str, documents: Sequence[str],
               top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回 `[{"index": i, "score": s}, ...]`（按相关性降序）。"""
        import httpx

        payload: Dict[str, Any] = {"query": query, "texts": list(documents)}
        if top_n:
            payload["raw_scores"] = False
        resp = httpx.post(f"{self.base_url}/rerank", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        results = resp.json()
        out: List[Dict[str, Any]] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            score = item.get("score", item.get("relevance_score", 0.0))
            out.append({"index": int(item.get("index", 0)), "score": float(score)})
        return out


class LocalReranker:
    """进程内 cross-encoder 重排（GPU 自动选择 + fp16）。

    与 `LocalEmbedClient` 同一套设备仲裁思路：能被显式指定（`KF_RERANK_DEVICE`），
    否则有 CUDA 就用 CUDA。**不参与**多服务抢卡仲裁——它只有 ~1.1GB，
    与 bge-m3 的 2.2GB 可以共存于 4GB 卡（实测合计 3361 MB）。

    分数语义：`rerank()` 返回 **sigmoid 归一后的概率 [0,1]**（bge-reranker 系列惯例），
    与 TEI `/rerank` 的量纲一致，调用方不必区分来源。
    """

    def __init__(self, model_id: str = "", device: str = "",
                 max_length: int = 0, batch_size: int = 0) -> None:
        self.model_id = model_id or rerank_model_id()
        self.requested_device = device or _env("KF_RERANK_DEVICE", "")
        self.max_length = max_length or _tier_max_length()
        self.batch_size = max(1, int(_env("KF_RERANK_BATCH", "12") or 12))
        self._tok = None
        self._model = None
        self._load_error = ""
        self.device = ""
        self.device_reason = ""
        self.dtype_name = ""
        self._load()

    # ---- 生命周期 ----
    def _pick_device(self) -> tuple[str, str]:
        want = self.requested_device.lower()
        if want == "cpu":
            return "cpu", "由 KF_RERANK_DEVICE 显式指定为 cpu"
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu", "未检测到可用 CUDA"
            if want.startswith("cuda"):
                return "cuda:0", "由 KF_RERANK_DEVICE 显式指定"
            # **不依赖 mem_get_info 做判断**：WDDM 驱动下它报告的空闲显存明显偏保守
            # （实测 bge-m3 占 2257MB 后就报不足，而 reranker 的 1.1GB 其实放得下）。
            # 直接尝试 CUDA，真 OOM 时由 `_load` 捕获并回落 CPU。
            return "cuda:0", "有可用 CUDA，优先使用（OOM 时自动回落 CPU）"
        except Exception as exc:
            return "cpu", f"探测 CUDA 失败: {exc}"

    def _load(self) -> None:
        self.device, self.device_reason = self._pick_device()
        try:
            self._load_on(self.device)
        except Exception as exc:
            # CUDA 显存不足 / 与嵌入模型抢显存：如实记录原因并回落 CPU，
            # 而不是让整个重排功能失效。
            if self.device.startswith("cuda"):
                self.device_reason = (
                    f"{self.device} 加载失败（{type(exc).__name__}: {str(exc)[:80]}），已回落 CPU"
                )
                self.device = "cpu"
                try:
                    self._free_cuda()
                    self._load_on("cpu")
                    return
                except Exception as exc2:
                    self._load_error = f"CPU 回落也失败: {type(exc2).__name__}: {exc2}"
                    self._model = self._tok = None
                    return
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._model = self._tok = None

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_on(self, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, local_files_only=True, dtype=dtype)
        model.eval()
        model.to(device)
        self._tok = tok
        self._model = model
        self.dtype_name = "float16" if dtype is torch.float16 else "float32"

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tok is not None

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "local",
            "model": self.model_id,
            "ready": self.ready,
            "device": self.device,
            "device_reason": self.device_reason,
            "dtype": self.dtype_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "load_error": self._load_error or None,
        }

    # ---- 打分 ----
    def rerank(self, query: str, documents: Sequence[str],
               top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回 `[{"index": i, "score": p}, ...]`，score 为 sigmoid 后的概率。"""
        if not self.ready:
            raise RuntimeError(f"本地重排不可用: {self._load_error}")
        docs = [str(d or "") for d in documents]
        if not docs:
            return []

        import torch

        raw: List[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start:start + self.batch_size]
            enc = self._tok(
                [query] * len(batch), batch,
                padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits.view(-1).float()
            raw.extend(logits.cpu().tolist())

        out = [{"index": i, "score": _sigmoid(v)} for i, v in enumerate(raw)]
        out.sort(key=lambda kv: -kv["score"])
        if top_n:
            out = out[:max(1, int(top_n))]
        return out

    def score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        """只返回与 `documents` 同序的 sigmoid 概率（供答案可达性判定复用）。"""
        if not self.ready:
            return []
        docs = [str(d or "") for d in documents]
        if not docs:
            return []
        import torch

        raw: List[float] = []
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start:start + self.batch_size]
            enc = self._tok(
                [query] * len(batch), batch,
                padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits.view(-1).float()
            raw.extend(logits.cpu().tolist())
        return [_sigmoid(v) for v in raw]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = 2.718281828459045 ** (-x)
        return 1.0 / (1.0 + z)
    z = 2.718281828459045 ** x
    return z / (1.0 + z)


_LOCAL_RERANKER: Optional[LocalReranker] = None
_LOCAL_RERANK_TRIED = False


def try_local_reranker() -> Optional[LocalReranker]:
    """惰性单例：只尝试加载一次，失败后不再重试（避免每次查询都卡几十秒）。"""
    global _LOCAL_RERANKER, _LOCAL_RERANK_TRIED
    if _LOCAL_RERANK_TRIED:
        return _LOCAL_RERANKER
    _LOCAL_RERANK_TRIED = True
    if _env("KF_DISABLE_LOCAL_RERANK", "0") in ("1", "true", "yes"):
        _LOCAL_RERANKER = None
        return None
    try:
        candidate = LocalReranker()
        _LOCAL_RERANKER = candidate if candidate.ready else None
    except Exception:
        _LOCAL_RERANKER = None
    return _LOCAL_RERANKER


def try_rerank_client() -> Optional[Any]:
    """优先 TEI 服务化重排；未配置则回落进程内本地重排；都没有返回 None。"""
    url = rerank_url()
    if url:
        return RerankClient(url)
    return try_local_reranker()


def local_rerank_status() -> Dict[str, Any]:
    """给 `/health` 用：如实报告本地重排是否可用（不改动惰性加载状态）。"""
    if _LOCAL_RERANKER is not None:
        return _LOCAL_RERANKER.describe()
    if _LOCAL_RERANK_TRIED:
        return {"backend": "none", "ready": False, "note": "已尝试加载但不可用"}
    return {"backend": "none", "ready": False, "note": "尚未加载（首次重排时惰性加载）"}


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    top_n: int,
    client: Optional[RerankClient] = None,
    candidate_limit: int = 0,
) -> Dict[str, Any]:
    """对候选块重排。

    返回 `{"candidates": [...], "reranked": bool, "ms": float, "note": str|None}`。
    重排不可用时原样返回候选（保留融合分），`reranked=False`。

    `candidate_limit` 限制**送去打分的候选数**（缺省 = `top_n`）。
    这一点此前是错的：旧实现把**全部候选**（融合后可能有 50 个）都送进 cross-encoder，
    `top_n` 只截断输出。真机实测因此 rerank 阶段耗时 **3.9s**（模型本身只需 1.5s）。
    cross-encoder 的开销随「候选数 × 序列长度²」增长，4GB 卡上 12 个候选 ×512 token
    要 9.4s，而 4 个 ×256 token 只要 1.1s——所以窗口必须收紧。
    """
    import time

    started = time.perf_counter()
    if client is None or not candidates:
        return {
            "candidates": candidates,
            "reranked": False,
            "ms": _ms(started),
            "note": None if candidates else "无候选，跳过重排",
        }

    limit = max(1, int(top_n))
    window = candidates[:max(limit, int(candidate_limit or limit))]
    docs = [str(c.get("content") or "") for c in window]
    try:
        results = client.rerank(query, docs, top_n=limit)
    except Exception as exc:
        return {
            "candidates": candidates,
            "reranked": False,
            "ms": _ms(started),
            "note": f"重排不可用，复用融合分: {exc}",
        }

    if not results:
        return {
            "candidates": candidates,
            "reranked": False,
            "ms": _ms(started),
            "note": "重排返回空，复用融合分",
        }

    reranked: List[Dict[str, Any]] = []
    used_indexes = set()
    for item in sorted(results, key=lambda kv: -float(kv.get("score", 0.0))):
        idx = item.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(window) or idx in used_indexes:
            continue
        used_indexes.add(idx)
        entry = dict(window[idx])
        entry["score"] = float(item.get("score", 0.0))
        reranked.append(entry)

    if not reranked:
        return {
            "candidates": candidates,
            "reranked": False,
            "ms": _ms(started),
            "note": "重排结果索引非法，复用融合分",
        }

    # 只返回被重排的 top_n：混合了不同量纲的尾部会污染后续预算组装
    return {
        "candidates": reranked,
        "reranked": True,
        "ms": _ms(started),
        "note": None,
    }


def _ms(started: float) -> float:
    import time

    return round((time.perf_counter() - started) * 1000.0, 3)
