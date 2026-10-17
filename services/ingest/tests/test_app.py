"""摄入 HTTP 接口测试：上传即摄入、文档列表、健康检查与降级。

要点：**无任何外部依赖**（不配 QDRANT_URL / DATABASE_URL / TEI_EMBED_URL）时，
上传仍必须成功返回，只是 embedded/indexed 计数为 0 —— 与 CLI 同一套降级语义。
"""

import pytest
from fastapi.testclient import TestClient

from kf_ingest.app import app

BODY = (
    "第 {n} 章说明 MX-{n}00 型设备的安装与配置流程。本产品面向企业知识库场景，"
    "支持多语言语料的解析、清洗、切片与混合检索。安装前请备份现有数据。"
)


def _make_pdf(path):
    import pymupdf as fitz

    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(60, 20, 535, 45), "企业产品手册 v2.1",
                            fontsize=9, fontname="china-s")
        page.insert_textbox(fitz.Rect(60, 795, 535, 820), f"第 {i + 1} 页",
                            fontsize=9, fontname="china-s", align=1)
        page.insert_textbox(fitz.Rect(60, 90, 535, 125), f"1.{i + 1} 产品概述",
                            fontsize=18, fontname="china-s")
        page.insert_textbox(fitz.Rect(60, 140, 535, 600), BODY.format(n=i + 1),
                            fontsize=11, fontname="china-s")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture()
def client(monkeypatch):
    # 显式清空外部依赖地址：验证「零依赖也能上传成功」
    for name in ("QDRANT_URL", "DATABASE_URL", "TEI_EMBED_URL", "VLLM_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(app)


@pytest.fixture()
def pdf(tmp_path):
    return _make_pdf(str(tmp_path / "manual.pdf"))


def _upload(client, pdf, **overrides):
    data = {"tenant": "acme", "kb": "demo", "lang": "zh", "dry_run": "true"}
    data.update({k: str(v) for k, v in overrides.items()})
    with open(pdf, "rb") as fh:
        return client.post(
            "/v1/ingest",
            files={"file": ("manual.pdf", fh.read(), "application/pdf")},
            data=data,
        )


def test_health_reports_dependency_config(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "kf-ingest"
    assert body["deps"] == {"qdrant": False, "postgres": False, "tei_embed": False}


def test_dry_run_upload_returns_summary_without_deps(client, pdf):
    resp = _upload(client, pdf)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["errors"] == []
    assert body["dry_run"] is True
    assert body["tenant"] == "acme" and body["kb"] == "demo"
    assert body["collection"] == "kf_acme_demo"
    # 只跑到切片：有子块、无嵌入/索引
    assert body["counts"]["children"] > 0
    assert body["counts"]["embedded"] == 0 and body["counts"]["indexed"] == 0
    assert body["doc_id"] == "manual"          # 留空则取文件名
    assert body["lang"] == "zh"
    assert body["strategy"]                      # 自动路由出了策略名
    # 清洗确实生效（跨页页眉 / 页脚被洗掉）
    assert body["cleaning"]["removed_blocks"] > 0


def test_reals_run_degrades_gracefully_without_deps(client, pdf):
    resp = _upload(client, pdf, dry_run="false", doc_id="manual_http")
    assert resp.status_code == 200
    body = resp.json()
    # 不配 Qdrant / Postgres 也不许报错，只记 note
    assert body["ok"] is True
    assert body["errors"] == []
    assert body["counts"]["indexed"] == 0
    # Qdrant / Postgres 客户端未连接 → 直接跳过落库，连 note 都不需要（无外部依赖）
    assert ((body["stages"].get("index") or {}).get("notes") or []) == []
    assert body["doc_id"] == "manual_http"
    assert body["notes"] == []


def test_documents_listing_without_database_records_note(client):
    body = client.get("/v1/documents", params={"tenant": "acme"}).json()
    assert body["documents"] == []
    assert body["notes"] and "Postgres" in body["notes"][0]


def test_documents_listing_filters_by_kb_param(client, monkeypatch):
    """kb 为空串时不应报错（SQL 里用 `%s = ''` 做可选过滤）。"""
    body = client.get("/v1/documents", params={"tenant": "acme", "kb": ""}).json()
    assert body["notes"] and body["documents"] == []


def test_upload_without_file_is_rejected(client):
    resp = client.post("/v1/ingest", data={"tenant": "acme", "kb": "demo"})
    assert resp.status_code == 422


def test_strategy_param_is_honoured(client, pdf):
    body = _upload(client, pdf, strategy="clause_atomic").json()
    assert body["strategy"] == "clause_atomic"
    assert body["counts"]["children"] > 0


def test_cors_headers_present_for_browser_console(client):
    """控制台以 file:// 打开时 Origin 为 null，服务端需放行跨域。"""
    resp = client.get("/health", headers={"Origin": "null"})
    assert resp.headers.get("access-control-allow-origin") == "*"
