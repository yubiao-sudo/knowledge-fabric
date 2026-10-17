"""召回后端：进程内 MemoryBackend + QdrantBackend，以及数据加载与后端选择。

- **MemoryBackend**：无 Docker 环境跑通与单测用。按与索引侧**同一套稀疏编码**
  做词法打分（BM25 风格：tf + 长度归一 + idf），支持
  `tenant / kb / lang / version / status=active` 过滤与 `chunk_type` 过滤。
  dense 侧仅在配置了 `TEI_EMBED_URL` 时启用（否则只做 sparse）。
- **QdrantBackend**：`qdrant-client`，命名向量 `dense`（COSINE）+ `sparse`，
  用 `query_points` 分别做两种检索；payload 过滤字段与
  `services/ingest/kf_ingest/index.py:_payload` 对齐
  （`tenant_id / kb_id / lang / version / status / chunk_type`）。
- **后端选择**：`QDRANT_URL` 已配置且可连 → Qdrant；否则 → MemoryBackend 并记 note。
  绝不因为连不上 Qdrant 就 500。

数据加载：`load_from_json(path)` 读一份 chunk 列表；`add_chunks()` 供测试灌数据；
环境变量 `KF_RETRIEVE_SEED_JSON` 指向的 json 会在服务启动时自动灌入 MemoryBackend。
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sparse import tokenize_for_sparse
from .structured import matches_conditions, qdrant_conditions

# chunk_id -> Qdrant 点 ID 的确定性 UUID（与 index.py:point_id 完全一致）
_POINT_NS = uuid.UUID("6f1a3c9e-2b47-5d8e-9a10-1c2d3e4f5a6b")


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def normalize_entity(text: str) -> str:
    """归一化实体名：全角转半角（NFKC）+ 小写 + 压缩空白 + 去首尾空白。

    与 `kf_ingest/graph.py:normalize_entity` 同规则——图检索侧必须用同一套归一化，
    否则查询词与库内 `graph_entities.norm_name` 对不上。
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm.lower()


def graph_query_terms(query: str) -> List[str]:
    """查询切词 → 规范化词 + 相邻二元组（去重保序）。

    与摄入侧建图时的实体归一化对齐：CJK 单字 + 相邻双字（bigram）、拉丁整词；
    额外把「相邻 token 拼接」也作为候选，覆盖 `MX-200` 这类被标点切开的型号。
    """
    toks = tokenize_for_sparse(query or "")
    cands: List[str] = []
    for tok in toks:
        cands.append(normalize_entity(tok))
    for i in range(len(toks) - 1):
        cands.append(normalize_entity(toks[i] + toks[i + 1]))
    return _dedupe([c for c in cands if c])


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out



def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NS, chunk_id))


def default_collection(tenant: str, kb: str) -> str:
    """集合命名：**一个租户 + 一个库 = 一个集合**（必须与 ingest 侧同名，否则永远召不回）。

    不按语言拆集合：多语言靠「集合内 lang 字段过滤」，且**只对稀疏通道过滤**
    （稀疏/BM25 单语言，稠密跨语言），见 service.py 的 filters 构造。
    """
    return f"kf_{tenant}_{kb}".replace("-", "_")


# ------------------------------------------------------------------ 归一化
def normalize_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把 Qdrant payload / JSON 记录归一成检索侧统一的 chunk dict。"""
    payload = payload or {}
    section_path = payload.get("section_path") or []
    section = payload.get("section")
    if not section and isinstance(section_path, list) and section_path:
        section = " > ".join(str(p) for p in section_path if p)
    vectors = payload.get("vectors") or {}
    dense = payload.get("dense") or vectors.get("dense")
    return {
        "chunk_id": str(payload.get("chunk_id") or ""),
        "parent_chunk_id": payload.get("parent_chunk_id"),
        "content": str(payload.get("content") or ""),
        "chunk_type": payload.get("chunk_type"),
        "doc_id": payload.get("doc_id"),
        "tenant_id": payload.get("tenant_id"),
        "kb_id": payload.get("kb_id"),
        "version": payload.get("version"),
        "lang": payload.get("lang"),
        "page": int(payload.get("page") or 0),
        "status": payload.get("status") or "active",
        "section": section,
        "source": payload.get("source_uri") or payload.get("source"),
        "char_start": payload.get("char_start"),
        "char_end": payload.get("char_end"),
        "acl": payload.get("acl") or [],
        "dense": dense,
        # 契约第 2 节：结构化双轨 + GraphRAG 的 payload 键名（ingest 写入 → retrieve 读取）
        "skus": [str(s) for s in (payload.get("skus") or [])],
        "price": payload.get("price"),
        "currency": payload.get("currency"),
        "attrs": payload.get("attrs") or {},
        "entities": [str(e) for e in (payload.get("entities") or [])],
        # Memory 侧 GraphRAG：seed 数据可携带 graph 字段（实体 / 社区）
        "graph": payload.get("graph") or {},
    }


def matches_filters(chunk: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """按 payload 字段做过滤，与 ingest/index.py 的 `_payload` 字段名对齐。"""
    if not filters:
        return True
    for field in ("tenant_id", "kb_id", "lang", "version", "chunk_type"):
        want = filters.get(field)
        if want in (None, "", []):
            continue
        if str(chunk.get(field)) != str(want):
            return False
    status = filters.get("status", "active")
    if status and str(chunk.get("status")) != str(status):
        return False
    return True


# ------------------------------------------------------------------ BM25 打分（可复用）
def bm25_rank(query: str, docs: Sequence[Dict[str, Any]], top_k: int,
              channel: str) -> List[Dict[str, Any]]:
    """对给定 docs 做 BM25 词法打分，返回 `[{chunk, score, channel}]`。

    MemoryBackend 的稀疏 / 结构化 / 图通道共用同一实现，保证语义一致；
    图通道「回 backend 取 payload 打分」即复用此函数。
    """
    query_terms = tokenize_for_sparse(query)
    if not query_terms:
        return []
    term_counts = [Counter(tokenize_for_sparse(d.get("content") or "")) for d in docs]
    lengths = [sum(c.values()) for c in term_counts]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
    df = Counter()
    for counter in term_counts:
        for term in counter.keys():
            df[term] += 1
    n_docs = len(docs)
    k1, b = 1.5, 0.75

    scored: List[Dict[str, Any]] = []
    for doc, counter, dl in zip(docs, term_counts, lengths):
        score = 0.0
        for term in set(query_terms):
            tf = counter.get(term, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl if avgdl else 1.0))
            score += idf * (tf * (k1 + 1.0)) / denom
        if score > 0.0:
            scored.append({"chunk": doc, "score": score, "channel": channel})
    scored.sort(key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
    return scored[:top_k]


# ------------------------------------------------------------------ TEI 嵌入
class TEIEmbedClient:
    """TEI 稠密嵌入客户端（`POST /embed`）。未配置地址时不要构造。"""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        if not base_url:
            raise ValueError("未配置 TEI_EMBED_URL")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/embed", json={"inputs": list(texts), "truncate": True}, timeout=self.timeout
        )
        resp.raise_for_status()
        vectors = resp.json()
        if vectors and isinstance(vectors[0], (int, float)):
            vectors = [vectors]  # 单条输入时 TEI 返回一维数组
        return [[float(x) for x in vec] for vec in vectors]


def try_embed_client() -> Optional[Any]:
    """取可用的稠密嵌入客户端：**TEI 优先，本地回退**。

    只有 TEI 一条路时，未部署 TEI 的机器上 dense 通道恒为空 → `naive` /
    `hybrid_rerank` 等档位静默返回 0 证据（真机实测 6/8 个档位如此）。
    本地回退让稠密检索在不依赖 Docker 的情况下也能真正工作。
    """
    url = env("TEI_EMBED_URL")
    if url:
        return TEIEmbedClient(url)
    return try_local_embed()


# ------------------------------------------------------------------ 本地嵌入（回退）
_LOCAL_MODEL_CACHE: Dict[str, Any] = {}
_LOCAL_MODEL_FAILED: Dict[str, str] = {}

# 与摄入侧 `kf_ingest.clients` 保持同一口径：模型实测占用与需留的余量。
_GPU_MODEL_FOOTPRINT_MB = 2400
_GPU_HEADROOM_MB = 600
# 跨进程标记目录：本机 4GB 显存装不下两份 bge-m3，必须显式仲裁谁用 GPU。
# 两个服务共用同一标记文件（与摄入侧常量一致）。
_GPU_MARKER_DIR = os.environ.get("KF_GPU_MARKER_DIR") or os.path.join(
    os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp", "kf_gpu_claim"
)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        pass
    try:
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
        marker = os.path.join(_GPU_MARKER_DIR, "cuda.claim")
        if not os.path.exists(marker):
            return None
        with open(marker, "r", encoding="ascii") as fh:
            pid = int((fh.read() or "0").strip() or 0)
    except Exception:
        return None
    if pid and pid != os.getpid() and _pid_alive(pid):
        return pid
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


def pick_embed_device(requested: str) -> Tuple[str, str]:
    """决定本地嵌入设备，返回 `(device, 原因)`。

    **背景（真机性能故障）**：本机 GTX 1650 4GB，摄入与检索**各自**加载一份
    bge-m3（实测每份 2166MB，两份 4.3GB > 4GB），显存被占满到 3831/4096MB、
    两个模型持续 thrash，一次 13 块的摄入劣化到 10 分钟以上。
    仅靠 `torch.cuda.mem_get_info()` **不可靠**（真机复现：摄入已占 2.2GB 时
    检索仍判定"充足"并加载了第二份），因此用**进程间标记文件**显式仲裁。

    规则：`KF_EMBED_DEVICE` 显式指定 > CUDA 可用 > 无其他服务占用 GPU > 显存充足。
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
            f"GPU 已被 pid={holder} 的服务占用（每个服务各加载一份嵌入模型，"
            "本机 4GB 显存装不下两份），回落 cpu 以避免互相 thrash"
        )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
    except Exception as exc:
        return "cpu", f"无法查询显存（{exc}），保守回落 cpu"
    need = _GPU_MODEL_FOOTPRINT_MB + _GPU_HEADROOM_MB
    if free_mb < need:
        return "cpu", f"显存不足：空闲 {free_mb}MB，需 {need}MB，回落 cpu"
    _claim_gpu()
    return "cuda", f"显存充足且无其他服务占用（空闲 {free_mb}MB）"


class LocalEmbedClient:
    """进程内本地稠密嵌入（与 `TEIEmbedClient.embed` 同签名）。

    与摄入侧 `kf_ingest.clients.LocalEmbedClient` 保持**同一模型、同一池化方式**
    （CLS + L2 归一）与**同一设备选择口径**，否则查询向量与索引向量不在同一空间，
    或两个服务抢同一块显存导致双双劣化。
    两服务不跨包 import 是既有约定，故各自实现、行为对齐。

    只在模型**本地已有**时加载，缺模型不会联网下载；拿不到真权重就返回 None。
    """

    backend = "local"

    def __init__(self, model_id: str, device: str = "") -> None:
        self.model_id = model_id
        self.requested_device = device or env("KF_EMBED_DEVICE")
        self.device, self.device_reason = pick_embed_device(self.requested_device)
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
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(
                self.model_id, device=self.device, local_files_only=True
            )
            return
        except Exception as exc:
            self._load_error = f"sentence-transformers 加载失败: {exc}"
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

    def describe(self) -> Dict[str, Any]:
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
            "load_error": self._load_error or None,
        }

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.ready:
            raise RuntimeError(f"本地嵌入不可用: {self._load_error}")
        if self._st is not None:
            vectors = self._st.encode(list(texts), normalize_embeddings=True,
                                      show_progress_bar=False)
            return [[float(x) for x in row] for row in vectors]

        import torch

        enc = self._tok(list(texts), padding=True, truncation=True,
                        max_length=8192, return_tensors="pt")
        if self.device:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self._model(**enc).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        return [[float(x) for x in row] for row in hidden.cpu().tolist()]


def try_local_embed(model_id: Optional[str] = None) -> Optional[LocalEmbedClient]:
    """取（并缓存）本地嵌入客户端；不可用返回 None。`KF_DISABLE_LOCAL_EMBED=1` 关闭。"""
    if env("KF_DISABLE_LOCAL_EMBED"):
        return None
    target = model_id or env("EMBED_MODEL") or "BAAI/bge-m3"
    if target in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[target]
    if target in _LOCAL_MODEL_FAILED:
        return None
    client = LocalEmbedClient(target)
    if not client.ready:
        _LOCAL_MODEL_FAILED[target] = client._load_error or "未知原因"
        return None
    _LOCAL_MODEL_CACHE[target] = client
    return client


def local_embed_status(model_id: Optional[str] = None) -> Dict[str, Any]:
    """本地嵌入可用性（供 health 如实报告）。"""
    target = model_id or env("EMBED_MODEL") or "BAAI/bge-m3"
    cached = _LOCAL_MODEL_CACHE.get(target)
    if cached is not None:
        return cached.describe()
    if target in _LOCAL_MODEL_FAILED:
        return {"backend": "local", "model": target, "ready": False,
                "load_error": _LOCAL_MODEL_FAILED[target]}
    return {"backend": "local", "model": target, "ready": False, "load_error": None,
            "note": "尚未尝试加载"}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


# ------------------------------------------------------------------ MemoryBackend
class MemoryBackend:
    """进程内检索后端：词法稀疏（BM25）+ 可选稠密（TEI）。"""

    def __init__(self, *, collection: str = "memory",
                 embed_client: Optional[TEIEmbedClient] = None) -> None:
        self.collection = collection
        self.embed_client = embed_client
        self._all: Dict[str, Dict[str, Any]] = {}
        self._parents: Dict[str, Dict[str, Any]] = {}
        self._searchable: List[Dict[str, Any]] = []
        # Memory 侧 GraphRAG（从 seed 的 `graph` 字段构建，供无依赖测试）
        self._graph_entity_chunks: Dict[str, set] = {}
        self._graph_edges: set = set()
        self._graph_communities: List[Dict[str, Any]] = []

    # ---- 数据装载 ----
    def add_chunks(self, chunks: Sequence[Dict[str, Any]]) -> int:
        for raw in chunks:
            chunk = normalize_chunk(raw)
            cid = chunk["chunk_id"]
            if not cid:
                continue
            self._all[cid] = chunk
        self._rebuild()
        return len(self._all)

    def load_from_json(self, path: str) -> int:
        """读一份 chunk 列表（数组，或 `{"chunks": [...]}`）。"""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        chunks = data.get("chunks") if isinstance(data, dict) else data
        if not isinstance(chunks, list):
            raise ValueError(f"种子文件格式非法（应为数组或 {{'chunks': [...]}}）: {path}")
        return self.add_chunks(chunks)

    def _rebuild(self) -> None:
        parent_ids = {c["parent_chunk_id"] for c in self._all.values() if c.get("parent_chunk_id")}
        self._parents = {pid: self._all[pid] for pid in parent_ids if pid in self._all}
        self._searchable = [c for c in self._all.values() if c["chunk_id"] not in parent_ids]
        self._build_graph()

    def _build_graph(self) -> None:
        """从 seed 的 `graph` 字段构建内存图（实体↔块、块内共现边、社区）。

        seed 的 chunk payload 可为：
            `"graph": {"entities": [...], "edges": [[a, b], ...], "communities": [...]}`
        `entities` / `edges` 用归一化实体名；`communities` 为 `{id, summary, entities, chunk_ids}`。
        """
        entity_chunks: Dict[str, set] = {}
        edges: set = set()
        communities: List[Dict[str, Any]] = []
        seen_comm: set = set()
        for cid, chunk in self._all.items():
            graph = chunk.get("graph") or {}
            norms = _dedupe([normalize_entity(e) for e in (graph.get("entities") or [])])
            norms = [n for n in norms if n]
            for norm in norms:
                entity_chunks.setdefault(norm, set()).add(cid)
            for i in range(len(norms)):
                for j in range(i + 1, len(norms)):
                    a, b = sorted((norms[i], norms[j]))
                    edges.add((a, b))
            for pair in graph.get("edges") or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a, b = normalize_entity(str(pair[0])), normalize_entity(str(pair[1]))
                    if a and b:
                        edges.add(tuple(sorted((a, b))))
            for comm in graph.get("communities") or []:
                if isinstance(comm, dict) and comm.get("id") and str(comm["id"]) not in seen_comm:
                    seen_comm.add(str(comm["id"]))
                    communities.append(comm)
        self._graph_entity_chunks = entity_chunks
        self._graph_edges = edges
        self._graph_communities = communities

    def count(self) -> int:
        return len(self._searchable)

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "memory",
            "collection": self.collection,
            "chunks": len(self._all),
            "searchable": len(self._searchable),
            "dense": bool(self.embed_client),
        }

    def get_chunk(self, chunk_id: str, collection: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self._all.get(str(chunk_id))

    def get_chunks(self, chunk_ids: Sequence[str],
                   collection: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """批量取块，与 QdrantBackend.get_chunks 同语义。"""
        out: Dict[str, Dict[str, Any]] = {}
        for cid in chunk_ids:
            found = self._all.get(str(cid))
            if found is not None:
                out[str(cid)] = found
        return out

    def embed_query(self, query: str, collection: Optional[str] = None) -> Optional[List[float]]:
        if self.embed_client is None:
            return None
        try:
            return [float(x) for x in self.embed_client.embed([query])[0]]
        except Exception:
            return None

    def dense_scores(self, query: str, chunk_ids: Sequence[str],
                     collection: Optional[str] = None,
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """内存后端的归一化稠密分（余弦）。无嵌入端点时返回 {}。"""
        vec = self.embed_query(query, collection=collection)
        if not vec:
            return {}
        wanted = {str(c) for c in chunk_ids}
        out: Dict[str, float] = {}
        for chunk in self._all.values():
            cid = str(chunk.get("chunk_id") or "")
            if cid not in wanted:
                continue
            chunk_vec = chunk.get("dense_vector") or chunk.get("query_vector")
            if chunk_vec:
                out[cid] = _cosine(vec, chunk_vec)
        return out

    def dense_score_by_vector(self, query: str,
                              collection: Optional[str] = None) -> Optional[List[float]]:
        """取查询的稠密向量（内存后端同样支持，供答案可达性判分）。"""
        return self.embed_query(query, collection=collection)

    def get_vectors(self, chunk_ids: Sequence[str],
                    collection: Optional[str] = None) -> Dict[str, List[float]]:
        """批量取块的稠密向量（内存后端从已载入的块里取）。"""
        wanted = {str(c) for c in chunk_ids}
        out: Dict[str, List[float]] = {}
        for chunk in self._all.values():
            cid = str(chunk.get("chunk_id") or "")
            if cid in wanted:
                vec = chunk.get("dense_vector") or chunk.get("query_vector")
                if vec:
                    out[cid] = [float(x) for x in vec]
        return out

    def get_parent(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pid = chunk.get("parent_chunk_id")
        return self._parents.get(str(pid)) if pid else None

    # ---- 检索 ----
    def search(self, channel: str, query: str, *, filters: Optional[Dict[str, Any]] = None,
               top_k: int = 20, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        docs = [c for c in self._searchable if matches_filters(c, filters)]
        if not docs:
            return []
        if channel == "dense":
            return self._search_dense(query, docs, top_k)
        if channel == "sparse":
            return self._search_bm25(query, docs, top_k, channel)
        if channel == "structured":
            cond = (filters or {}).get("structured")
            if not cond:
                return []
            filtered = [d for d in docs if matches_conditions(d, cond)]
            return self._search_bm25(query, filtered, top_k, channel)
        return []

    def graph_search(self, query: str, *, filters: Optional[Dict[str, Any]] = None,
                     top_k: int = 20, collection: Optional[str] = None,
                     max_hops: int = 2, mode: Optional[str] = None) -> Dict[str, Any]:
        """内存 GraphRAG：查询词匹配实体 → 沿边扩 `max_hops` 跳 → 取关联块并 BM25 打分。

        返回 `{"hits": [...], "entities_matched": [...], "communities_used": n}`，
        与 PG 图路径（`kf_retrieve/graph.py`）语义一致。
        """
        empty = {"hits": [], "entities_matched": [], "communities_used": 0}
        if not self._graph_entity_chunks:
            return empty
        docs = [c for c in self._searchable if matches_filters(c, filters)]
        qnorm = normalize_entity(query)
        cands = set(graph_query_terms(query))
        matched = sorted(
            norm for norm in self._graph_entity_chunks
            if norm in cands or (norm and norm in qnorm)
        )
        if not matched:
            return empty

        reached = self._expand_graph(matched, max_hops)
        ordered_norms = _dedupe(list(matched) + sorted(reached))
        chunk_ids: List[str] = []
        seen: set = set()
        for norm in ordered_norms:
            for cid in sorted(self._graph_entity_chunks.get(norm, ())):
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)

        doc_by_id = {c["chunk_id"]: c for c in docs}
        selected = [doc_by_id[cid] for cid in chunk_ids if cid in doc_by_id]
        hits = bm25_rank(query, selected, top_k, "graph")

        communities_used = 0
        if mode == "community_summary":
            members = set(ordered_norms)
            for comm in self._graph_communities:
                comm_norms = {normalize_entity(e) for e in (comm.get("entities") or [])}
                if not (comm_norms & members):
                    continue
                hits.append({
                    "chunk": {
                        "chunk_id": "community:" + str(comm.get("id")),
                        "doc_id": "community:" + str(comm.get("id")),
                        "chunk_type": "community_summary",
                        "content": str(comm.get("summary") or ""),
                        "community_chunk_ids": list(comm.get("chunk_ids") or []),
                        "score": 1.0,
                        "retrieval_channel": "graph",
                    },
                    "score": 1.0,
                    "channel": "graph",
                })
                communities_used += 1

        hits.sort(key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
        return {
            "hits": hits[: top_k + communities_used],
            "entities_matched": matched,
            "communities_used": communities_used,
        }

    def _expand_graph(self, start: Sequence[str], max_hops: int) -> set:
        """沿无向边做 BFS 扩展，返回除起点外新到达的实体集合。"""
        reached = set(start)
        frontier = set(start)
        for _ in range(max(0, int(max_hops))):
            nxt = set()
            for a, b in self._graph_edges:
                if a in frontier and b not in reached:
                    nxt.add(b)
                if b in frontier and a not in reached:
                    nxt.add(a)
            if not nxt:
                break
            reached |= nxt
            frontier = nxt
        return reached - set(start)

    def _search_dense(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if self.embed_client is None:
            return []
        try:
            qvec = self.embed_client.embed([query])[0]
        except Exception:
            return []
        scored = []
        for doc in docs:
            vec = doc.get("dense")
            if not vec:
                continue
            scored.append({"chunk": doc, "score": _cosine(qvec, vec), "channel": "dense"})
        scored.sort(key=lambda h: (-h["score"], h["chunk"]["chunk_id"]))
        return scored[:top_k]

    def _search_bm25(self, query: str, docs: List[Dict[str, Any]], top_k: int,
                     channel: str) -> List[Dict[str, Any]]:
        return bm25_rank(query, docs, top_k, channel)


# ------------------------------------------------------------------ QdrantBackend
class QdrantBackend:
    """Qdrant 后端：命名向量 dense（COSINE）+ sparse，`query_points` 双通道。"""

    def __init__(self, client: Any, *, collection: str,
                 embed_client: Optional[TEIEmbedClient] = None, dim: int = 1024) -> None:
        self.client = client
        self.collection = collection
        self.embed_client = embed_client
        self.dim = dim
        # 最近一次检索失败的原因（供观测）。
        # 原实现把异常吞成"空结果"，**故障与"确实没有命中"无法区分**——
        # 真机复现：某些库的 dense 通道恒为 0 命中，日志上完全看不出为什么，
        # 直到逐库对比才发现是 dense 检索在报错。
        self.last_error: str = ""
        self.last_channel: str = ""

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": "qdrant",
            "collection": self.collection,
            "dense": bool(self.embed_client),
        }

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return int(getattr(info, "points_count", 0) or 0)
        except Exception:
            return 0

    def get_chunk(self, chunk_id: str, collection: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.get_chunks([chunk_id], collection=collection).get(str(chunk_id))

    def get_chunks(self, chunk_ids: Sequence[str],
                   collection: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """批量取块（一次 Qdrant retrieve 往返）。空实现返回 {}，调用方按缺失处理。"""
        wanted = [str(c) for c in chunk_ids if c]
        if not wanted:
            return {}
        name = collection or self.collection
        try:
            records = self.client.retrieve(
                collection_name=name,
                ids=[point_id(cid) for cid in wanted],
                with_payload=True,
            )
        except Exception:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for record in records or []:
            payload = getattr(record, "payload", None) or {}
            chunk = normalize_chunk(payload)
            cid = str(chunk.get("chunk_id") or "")
            if cid:
                out[cid] = chunk
        return out

    def embed_query(self, query: str, collection: Optional[str] = None) -> Optional[List[float]]:
        """查询的稠密向量（用于归一化答案可达分）。"""
        if self.embed_client is None:
            return None
        try:
            return [float(x) for x in self.embed_client.embed([query])[0]]
        except Exception:
            return None

    def dense_scores(self, query: str, chunk_ids: Sequence[str],
                     collection: Optional[str] = None,
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """指定块与查询的**归一化稠密分数**（Qdrant Cosine 分即余弦）。

        用于答案可达性判定：融合分（RRF）对"能否回答"没有区分度（见
        `assemble.answerability_scores` 的实测说明），必须另取余弦分。
        把候选 id 作为 Must 条件一次性查询，避免逐条取向量。
        """
        if self.embed_client is None or not chunk_ids:
            return {}
        try:
            from qdrant_client import models as _m
        except Exception:
            return {}
        vec = self.embed_query(query, collection=collection)
        if not vec:
            return {}
        name = collection or self.collection
        must = [_m.FieldCondition(key="chunk_id", match=_m.MatchAny(any=list(chunk_ids)))]
        base = _qdrant_filter(_m, {**(filters or {}), "chunk_id": None})
        if base is not None and getattr(base, "must", None):
            must.extend(base.must)
        try:
            result = self.client.query_points(
                collection_name=name, query=vec, using="dense",
                limit=len(chunk_ids), query_filter=_m.Filter(must=must),
                with_payload=True,
            )
        except Exception:
            return {}
        out: Dict[str, float] = {}
        for point in getattr(result, "points", result) or []:
            cid = str((getattr(point, "payload", None) or {}).get("chunk_id") or "")
            if cid:
                out[cid] = float(getattr(point, "score", 0.0) or 0.0)
        return out

    def dense_score_by_vector(self, query: str,
                              collection: Optional[str] = None) -> Optional[List[float]]:
        """取查询的稠密向量，供**不依赖档位通道**的答案可达性判分使用。

        与 `embed_query` 同义，方法名表达用途：调用方拿到它之后再配合
        `get_vectors()` 现算余弦，这样只跑 sparse 的档位（如 exact_lookup）
        也能得到规范化分数。
        """
        return self.embed_query(query, collection=collection)

    def get_vectors(self, chunk_ids: Sequence[str],
                    collection: Optional[str] = None) -> Dict[str, List[float]]:
        """批量取块的稠密向量（一次 Qdrant retrieve 往返）。"""
        wanted = [str(c) for c in chunk_ids if c]
        if not wanted:
            return {}
        name = collection or self.collection
        try:
            records = self.client.retrieve(
                collection_name=name,
                ids=[point_id(cid) for cid in wanted],
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            return {}
        out: Dict[str, List[float]] = {}
        for record in records or []:
            vector = getattr(record, "vector", None)
            if isinstance(vector, dict):
                vector = vector.get("dense")
            cid = str(((getattr(record, "payload", None) or {}).get("chunk_id")) or "")
            if cid and vector:
                try:
                    out[cid] = [float(x) for x in vector]
                except Exception:
                    continue
        return out

    def search(self, channel: str, query: str, *, filters: Optional[Dict[str, Any]] = None,
               top_k: int = 20, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        name = collection or self.collection
        self.last_error = ""
        self.last_channel = channel
        try:
            from qdrant_client import models
        except Exception as exc:
            self.last_error = f"qdrant_client 不可用: {exc}"
            return []
        query_filter = _qdrant_filter(models, filters or {})
        try:
            if channel == "dense":
                if self.embed_client is None:
                    self.last_error = "无稠密嵌入端点（dense 通道不可用）"
                    return []
                vector = self.embed_client.embed([query])[0]
                result = self.client.query_points(
                    collection_name=name, query=vector, using="dense",
                    limit=top_k, query_filter=query_filter, with_payload=True,
                )
            elif channel in ("sparse", "structured"):
                from .sparse import sparse_vector

                sv = sparse_vector(query)
                if not sv["indices"]:
                    self.last_error = "查询分词后为空，稀疏检索跳过"
                    return []
                result = self.client.query_points(
                    collection_name=name,
                    query=models.SparseVector(indices=sv["indices"], values=sv["values"]),
                    using="sparse", limit=top_k, query_filter=query_filter, with_payload=True,
                )
            else:
                self.last_error = f"未知通道 {channel}"
                return []
        except Exception as exc:
            # **不再静默**：把原因记下来，由 service 层作为 note 暴露给调用方。
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        points = getattr(result, "points", result) or []
        hits: List[Dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            hits.append({
                "chunk": normalize_chunk(payload),
                "score": float(getattr(point, "score", 0.0) or 0.0),
                "channel": channel,
            })
        hits = [h for h in hits if h["chunk"]["chunk_id"]]
        return hits[:top_k]


def _qdrant_filter(models: Any, filters: Dict[str, Any]) -> Any:
    """把过滤字段转成 Qdrant Filter；字段名与 ingest/index.py 的 payload 对齐。

    `filters["structured"]` 存在时追加结构化精确过滤条件（skus=MatchAny / price=Range /
    currency=match），实现 `structured` 通道的「真实精确过滤」语义。
    """
    must = []
    for field in ("tenant_id", "kb_id", "lang", "version", "chunk_type"):
        value = filters.get(field)
        if value in (None, "", []):
            continue
        must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    status = filters.get("status", "active")
    if status:
        must.append(models.FieldCondition(key="status", match=models.MatchValue(value=status)))
    must.extend(qdrant_conditions(models, filters.get("structured")))
    return models.Filter(must=must) if must else None


# ------------------------------------------------------------------ 后端选择
def _connect_qdrant(url: str, timeout: float = 5.0) -> Optional[Any]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, timeout=timeout)
        client.get_collections()
        return client
    except Exception:
        return None


def build_backend(
    *,
    collection: Optional[str] = None,
    embed_client: Optional[TEIEmbedClient] = None,
    qdrant_url: Optional[str] = None,
    memory: Optional[MemoryBackend] = None,
) -> Tuple[Any, List[str]]:
    """选择召回后端：Qdrant 可用则用之，否则回退 MemoryBackend（不抛异常）。"""
    notes: List[str] = []
    url = qdrant_url if qdrant_url is not None else env("QDRANT_URL")
    name = collection or env("RETRIEVAL_COLLECTION", "kf_default_default")

    if url:
        try:
            client = _connect_qdrant(url)
            if client is not None:
                return QdrantBackend(client, collection=name, embed_client=embed_client), notes
            notes.append("Qdrant 连接失败，回退 MemoryBackend")
        except Exception as exc:
            notes.append(f"Qdrant 不可用，回退 MemoryBackend: {exc}")
    else:
        notes.append("未配置 QDRANT_URL，使用 MemoryBackend")

    if memory is None:
        memory = MemoryBackend(collection=name, embed_client=embed_client)
    seed = env("KF_RETRIEVE_SEED_JSON")
    if seed and not memory.describe().get("chunks"):
        try:
            memory.load_from_json(seed)
            notes.append(f"已从 {seed} 装载 {memory.describe().get('chunks')} 条 chunk")
        except Exception as exc:
            notes.append(f"种子文件装载失败: {exc}")
    return memory, notes
