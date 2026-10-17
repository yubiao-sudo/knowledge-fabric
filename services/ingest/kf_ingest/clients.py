"""外部服务客户端：TEI（嵌入 / 重排）与 OpenAI 兼容生成端点。

全部配置来自环境变量，且**失败即优雅降级**：未配置或不可达时返回 None /
抛 `ServiceUnavailable`，由上层决定跳过该阶段或走替代路径。
绝不因为一个可选模型不可用而中断整个摄入链路。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx


class ServiceUnavailable(RuntimeError):
    """外部推理服务不可用（未配置 / 连接失败 / 非 2xx）。"""


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def tei_embed_url() -> str:
    return env("TEI_EMBED_URL")


def tei_rerank_url() -> str:
    return env("TEI_RERANK_URL")


def embed_model_id() -> str:
    return env("EMBED_MODEL", "BAAI/bge-m3")


def embed_dim() -> int:
    try:
        return int(env("EMBED_DIM", "1024") or 1024)
    except ValueError:
        return 1024


def llm_base_url() -> str:
    """OpenAI 兼容端点（vLLM）。enrich 阶段生成上下文 / 假设问题用。"""
    return env("VLLM_URL") or env("OPENAI_BASE_URL")


def llm_model_id() -> str:
    return env("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def llm_api_key() -> str:
    return env("OPENAI_API_KEY", "")


def qdrant_url() -> str:
    return env("QDRANT_URL")


def database_url() -> str:
    return env("DATABASE_URL")


def default_collection(tenant: str, kb: str) -> str:
    """集合命名：**一个租户 + 一个库 = 一个集合**。

    刻意不按语言分集合。多语言的正确处理是「一个集合内用 lang 字段做过滤」：
    稠密向量本身跨语言（可跨语言召回），稀疏/BM25 本质单语言，
    因此由检索侧**只对稀疏通道**加 lang 过滤即可。
    若按语言拆集合，稠密通道就被迫只能看到单一语言，反而放弃跨语言能力。
    """
    return f"kf_{tenant}_{kb}".replace("-", "_")


def connect_qdrant(url: Optional[str] = None, timeout: float = 15.0) -> Optional[Any]:
    """连 Qdrant；未配置或连不上返回 None（由调用方记 note 后跳过落库）。"""
    target = url if url is not None else qdrant_url()
    if not target:
        return None
    try:
        from qdrant_client import QdrantClient

        return QdrantClient(url=target, timeout=timeout)
    except Exception:
        return None


def connect_pg(dsn: Optional[str] = None) -> Optional[Any]:
    """连 Postgres；未配置或连不上返回 None（同上，绝不因此中断摄入）。"""
    target = dsn if dsn is not None else database_url()
    if not target:
        return None
    try:
        import psycopg

        return psycopg.connect(target)
    except Exception:
        return None


class _HTTPClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        if not base_url:
            raise ServiceUnavailable("未配置服务地址")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = httpx.post(url, json=payload, timeout=timeout or self.timeout)
        except Exception as exc:  # 连接失败 / 超时
            raise ServiceUnavailable(f"请求 {url} 失败: {exc}") from exc
        if resp.status_code >= 400:
            raise ServiceUnavailable(f"{url} 返回 {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def get_json(self, path: str, timeout: Optional[float] = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = httpx.get(url, timeout=timeout or self.timeout)
        except Exception as exc:
            raise ServiceUnavailable(f"请求 {url} 失败: {exc}") from exc
        if resp.status_code >= 400:
            raise ServiceUnavailable(f"{url} 返回 {resp.status_code}")
        return resp.json()


class TEIClient(_HTTPClient):
    """Text Embeddings Inference 客户端。

    - `embed(texts)`   → 稠密向量（`POST /embed`，batch 自动切分）
    - `embed_sparse(texts)` → 学习式稀疏权重（`POST /embed_sparse`）；该端点
      仅在部署了 sparse 能力的模型时存在，缺失时抛 ServiceUnavailable，
      由调用方回退到词法稀疏编码。
    - `rerank(query, docs, top_n)` → `POST /rerank`
    """

    def embed(self, texts: List[str], batch_size: int = 32, truncate: bool = True) -> List[List[float]]:
        out: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if not batch:
                continue
            vectors = self.post_json("/embed", {"inputs": batch, "truncate": truncate})
            if batch and isinstance(vectors, list) and vectors and isinstance(vectors[0], (int, float)):
                vectors = [vectors]  # 单条输入时 TEI 返回一维数组
            out.extend([[float(x) for x in vec] for vec in vectors])
        return out

    def embed_sparse(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, List]]:
        out: List[Dict[str, List]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if not batch:
                continue
            sparse = self.post_json("/embed_sparse", {"inputs": batch})
            if isinstance(sparse, dict):
                sparse = [sparse]
            out.extend(sparse)
        return out

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"query": query, "texts": documents}
        if top_n:
            payload["raw_scores"] = False
        results = self.post_json("/rerank", payload)
        return results if isinstance(results, list) else []


# ------------------------------------------------------------------ 本地嵌入（TEI 缺席时的回退）
# 为什么需要它：整条链路原本假设「TEI 一定在」，一旦 TEI 未部署，dense 向量全为
# None，`naive` / `hybrid_rerank` 等依赖稠密通道的档位**恒定返回 0 证据**，
# 而 health 仍显示配置正常——失效是静默的。真机实测：TEI 未起时 8 个档位里 6 个
# 返回 0 证据，Qdrant 里的点只有 sparse 没有 dense。
# 本类用 sentence-transformers / transformers 在本进程内算嵌入，接口与 TEIClient
# 的 embed() 完全一致，因此检索与摄入两侧都可零改动接入既有降级链。
_LOCAL_EMBED_CACHE: Dict[str, Any] = {}
_LOCAL_EMBED_FAILED: Dict[str, str] = {}

# 模型在 GPU 上的粗略占用（MB）。bge-m3 实测 2166MB，取 2400 留分配余量。
_GPU_MODEL_FOOTPRINT_MB = 2400
# 需要留出的余量（其他服务/框架开销）。
_GPU_HEADROOM_MB = 600

# 跨进程标记文件目录：同一台机器上多个服务会各自加载一份嵌入模型，
# 而 4GB 卡**装不下两份**。仅靠 `torch.cuda.mem_get_info()` 判断不可靠——
# 真机复现过：摄入服务已占 2.2GB 时，检索服务仍认为显存充足并加载了第二份
# （GPU 用量 3831/4096MiB，两个模型互相 thrash，一次摄入从秒级劣化到 10 分钟）。
# 因此用一个进程间标记文件显式仲裁：谁先占下 GPU，谁写标记，后来者回落 CPU。
_GPU_MARKER_DIR = os.environ.get("KF_GPU_MARKER_DIR") or os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp", "kf_gpu_claim"
)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil  # 可选依赖

        return psutil.pid_exists(pid)
    except Exception:
        pass
    try:  # Windows：tasklist 之外的最轻办法
        if os.name == "nt":
            import subprocess

            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True, timeout=5)
            return str(pid) in (out.stdout or "")
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _gpu_claim_holder() -> Optional[int]:
    """返回当前占用 GPU 标记的活进程 pid；无标记或属主已退出则返回 None。"""
    try:
        os.makedirs(_GPU_MARKER_DIR, exist_ok=True)
        marker = os.path.join(_GPU_MARKER_DIR, "cuda.claim")
        if not os.path.exists(marker):
            return None
        with open(marker, "r", encoding="ascii") as fh:
            pid = int((fh.read() or "0").strip() or 0)
    except Exception:
        return None
    if pid and pid != os.getpid() and _pid_alive(pid):
        return pid
    # 陈旧标记（属主已退出或就是自己）→ 清理后视为无占用
    try:
        os.remove(marker)
    except Exception:
        pass
    return None


def _claim_gpu() -> None:
    try:
        os.makedirs(_GPU_MARKER_DIR, exist_ok=True)
        with open(os.path.join(_GPU_MARKER_DIR, "cuda.claim"), "w", encoding="ascii") as fh:
            fh.write(str(os.getpid()))
        # 进程正常退出时释放标记，避免后来的服务被陈旧标记挡住
        # （进程被强杀时标记会残留，但 `_gpu_claim_holder` 会检测属主存活并自动清理）
        import atexit

        atexit.register(_release_gpu_claim)
    except Exception:
        pass


def _release_gpu_claim() -> None:
    try:
        marker = os.path.join(_GPU_MARKER_DIR, "cuda.claim")
        if os.path.exists(marker):
            with open(marker, "r", encoding="ascii") as fh:
                if int((fh.read() or "0").strip() or 0) == os.getpid():
                    os.remove(marker)
    except Exception:
        pass


def _pick_device(requested: str) -> Tuple[str, str]:
    """决定本地嵌入用哪个设备，返回 `(device, 说明)`。

    判定顺序（任一条不满足即回落 cpu，原因写进 describe 供 /health 暴露）：
      1. `KF_EMBED_DEVICE` 显式指定 → 无条件尊重（部署方可强制）；
      2. CUDA 可用；
      3. **没有其他存活进程已占用 GPU 标记**（跨进程仲裁，见 `_GPU_MARKER_DIR`）；
      4. 剩余显存够放下模型 + 余量。
    """
    if requested:
        if requested.startswith("cuda"):
            _claim_gpu()
        return requested, f"由 KF_EMBED_DEVICE 显式指定为 {requested}"
    try:
        import torch
    except Exception as exc:
        return "cpu", f"torch 不可用（{exc}），回落 cpu"
    if not torch.cuda.is_available():
        return "cpu", "CUDA 不可用，回落 cpu"

    holder = _gpu_claim_holder()
    if holder is not None:
        return "cpu", (
            f"GPU 已被 pid={holder} 的服务占用（同一台机器上每个服务各加载一份嵌入模型，"
            "本机 4GB 显存装不下两份），回落 cpu 以避免互相 thrash"
        )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
    except Exception as exc:
        return "cpu", f"无法查询显存（{exc}），保守回落 cpu"
    need = _GPU_MODEL_FOOTPRINT_MB + _GPU_HEADROOM_MB
    if free_mb < need:
        return "cpu", (
            f"显存不足：空闲 {free_mb}MB / 共 {total_mb}MB，需 {need}MB，回落 cpu"
        )
    _claim_gpu()
    return "cuda", f"显存充足且无其他服务占用（空闲 {free_mb}MB / 共 {total_mb}MB）"


class LocalEmbedClient:
    """进程内本地嵌入客户端（与 `TEIClient.embed` 同签名）。

    加载顺序：sentence-transformers → transformers + 手动池化。
    只在模型**本地已有**时加载（`local_files_only`），绝不因缺模型而联网，
    也绝不返回随机向量——拿不到真权重就返回 None，让上层如实降级。
    """

    backend = "local"

    def __init__(self, model_id: str, device: str = "", max_length: int = 0) -> None:
        self.model_id = model_id
        self.requested_device = device or env("KF_EMBED_DEVICE", "")
        self.device, self.device_reason = _pick_device(self.requested_device)
        # 序列上限可配：bge-m3 官方支持 8192，但位置编码与注意力缓冲会按该上限预留。
        # 4GB 卡上 8192 容易把显存撑爆（实测 3901/4096 MiB 后开始换页、吞吐塌到 0.6 块/秒）。
        # 切片本身只有 200~400 token，8192 纯属浪费，默认收到 1024。
        self.max_length = max_length or int(env("KF_EMBED_MAX_LEN", "1024") or 1024)
        self.batch_size = max(1, int(env("KF_EMBED_BATCH", "16") or 16))
        self._st = None
        self._tok = None
        self._model = None
        self._load_error = ""
        self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            self._load_error = f"torch 不可用: {exc}"
            return
        # 方式一：sentence-transformers（自带池化配置，最省事）
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(
                self.model_id, device=self.device, local_files_only=True
            )
            return
        except Exception as exc:
            self._load_error = f"sentence-transformers 加载失败: {exc}"
        # 方式二：transformers + CLS 池化（BGE 系列的标准池化方式）
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"
                self.device_reason += "；加载时 CUDA 不可用，改用 cpu"
            self._tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_id, local_files_only=True)
            self._model.eval()
            self._model.to(self.device)
            self._load_error = ""
        except Exception as exc:
            self._load_error = f"{self._load_error}; transformers 加载失败: {exc}"

    @property
    def ready(self) -> bool:
        return self._st is not None or self._model is not None

    @property
    def tokenizer(self) -> Any:
        """已加载的 tokenizer（供 `TokenCounter` 复用，避免重复加载）。

        为什么重要：摄入链路里**同一份 bge-m3 tokenizer 会被加载两次**——
        一次是嵌入模型内部，一次是 `tokenize.default_counter()` 自己
        `AutoTokenizer.from_pretrained`。实测重复加载在内存吃紧时要 **43 秒**
        （正常 3.8 秒，此处系统已因多份模型驻留而 thrash），并额外占内存。
        复用同一实例可同时省下这两笔开销。
        """
        if self._st is not None:
            return getattr(self._st, "tokenizer", None)
        return self._tok

    def describe(self) -> Dict[str, Any]:
        # 报告**实际**设备：优先问模型自己（sentence-transformers 可能因显存不足
        # 自行回落），而不是回显我们请求的设备。
        actual = self.device
        if self._st is not None:
            actual = str(getattr(self._st, "device", self.device))
        return {
            "backend": "local",
            "model": self.model_id,
            "ready": self.ready,
            "device": actual,
            "device_reason": self.device_reason,
            "requested_device": self.requested_device or "(auto)",
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "load_error": self._load_error or None,
        }

    def embed(self, texts: List[str], batch_size: int = 0,
              truncate: bool = True) -> List[List[float]]:
        if not self.ready:
            raise ServiceUnavailable(f"本地嵌入不可用: {self._load_error}")
        batch = max(1, int(batch_size or self.batch_size))
        if self._st is not None:
            vectors = self._st.encode(
                list(texts), batch_size=batch, normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in vectors]

        import torch

        out: List[List[float]] = []
        for start in range(0, len(texts), batch):
            chunk = list(texts[start:start + batch])
            enc = self._tok(
                chunk, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            if self.device:
                enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state[:, 0]      # CLS 池化
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)  # L2 归一
            out.extend([[float(x) for x in row] for row in hidden.cpu().tolist()])
        return out


def try_local_embed(model_id: Optional[str] = None) -> Optional[LocalEmbedClient]:
    """取（并缓存）本地嵌入客户端；不可用返回 None。

    `KF_DISABLE_LOCAL_EMBED=1` 可显式关闭（单测与纯离线环境用）。
    加载失败的原因缓存在 `_LOCAL_EMBED_FAILED` 里，供 `/health` 如实暴露，
    而不是静默退化成"没有向量"。
    """
    if env("KF_DISABLE_LOCAL_EMBED"):
        return None
    target = model_id or embed_model_id()
    if target in _LOCAL_EMBED_CACHE:
        return _LOCAL_EMBED_CACHE[target]
    if target in _LOCAL_EMBED_FAILED:
        return None
    client = LocalEmbedClient(target)
    if not client.ready:
        _LOCAL_EMBED_FAILED[target] = client._load_error or "未知原因"
        return None
    _LOCAL_EMBED_CACHE[target] = client
    return client


def local_embed_status(model_id: Optional[str] = None) -> Dict[str, Any]:
    """本地嵌入可用性（供 health 如实报告，不假装配置正常）。"""
    target = model_id or embed_model_id()
    cached = _LOCAL_EMBED_CACHE.get(target)
    if cached is not None:
        return cached.describe()
    if target in _LOCAL_EMBED_FAILED:
        return {"backend": "local", "model": target, "ready": False,
                "load_error": _LOCAL_EMBED_FAILED[target]}
    return {"backend": "local", "model": target, "ready": False, "load_error": None,
            "note": "尚未尝试加载"}


class LLMClient(_HTTPClient):
    """OpenAI 兼容 Chat Completions 客户端（vLLM 承载）。"""

    def __init__(self, base_url: str, model_id: str, api_key: str = "", timeout: float = 60.0) -> None:
        super().__init__(base_url, timeout)
        self.model_id = model_id
        self.api_key = api_key

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256,
             temperature: float = 0.2) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"
        try:
            import httpx as _httpx

            resp = _httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        except Exception as exc:
            raise ServiceUnavailable(f"请求 {url} 失败: {exc}") from exc
        if resp.status_code >= 400:
            raise ServiceUnavailable(f"{url} 返回 {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ServiceUnavailable("LLM 返回空 choices")
        return str((choices[0].get("message") or {}).get("content") or "")


def try_tei_embed(timeout: float = 30.0) -> Optional[Any]:
    url = tei_embed_url()
    return TEIClient(url, timeout) if url else None


def try_embed_client(timeout: float = 30.0) -> Optional[Any]:
    """按优先级取可用的嵌入客户端：**TEI 优先，本地回退**。

    返回 None 表示两条路都不可用——调用方据此如实记 note，而不是静默产出空向量。
    """
    tei = try_tei_embed(timeout)
    if tei is not None:
        return tei
    return try_local_embed()


def try_tei_rerank(timeout: float = 30.0) -> Optional[TEIClient]:
    url = tei_rerank_url()
    return TEIClient(url, timeout) if url else None


def try_llm(timeout: float = 60.0) -> Optional[LLMClient]:
    base = llm_base_url()
    return LLMClient(base, llm_model_id(), llm_api_key(), timeout) if base else None
