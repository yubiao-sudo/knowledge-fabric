@echo off
REM Pure ASCII only (cmd.exe parses .cmd with the system ANSI codepage).
setlocal
cd /d F:\knowledge-fabric\services\retrieve
set PYTHONPATH=F:\knowledge-fabric\services\retrieve
set QDRANT_URL=http://localhost:6333
set DATABASE_APP_URL=postgresql://kf_app:kf_app_password@localhost:15432/knowledge_fabric
set AGENTIC_MAX_ROUNDS=3
set TEI_EMBED_URL=
set TEI_RERANK_URL=
set EMBED_MODEL=BAAI/bge-m3
set RERANK_MODEL=BAAI/bge-reranker-v2-m3
set HF_HUB_OFFLINE=1
REM Unbuffered: otherwise stdout redirected to a log file stays block-buffered
REM and the log looks empty until the process exits.
set PYTHONUNBUFFERED=1
set KF_WARMUP=1
set KF_EMBED_MAX_LEN=512
set KF_EMBED_BATCH=8
REM Rerank window / sequence length follow the VRAM tier automatically.
REM Only set these to force values: KF_RERANK_TOP_N, KF_RERANK_MAX_LEN
python -m uvicorn kf_retrieve.app:app --port 8082 --host 127.0.0.1