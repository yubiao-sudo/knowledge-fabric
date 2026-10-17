@echo off
REM Pure ASCII only (cmd.exe parses .cmd with the system ANSI codepage).
setlocal
cd /d F:\knowledge-fabric\services\ingest
set PYTHONPATH=F:\knowledge-fabric\services\ingest
set QDRANT_URL=http://localhost:6333
set DATABASE_URL=postgresql://kf:kf_dev_password@localhost:15432/knowledge_fabric
set TEI_EMBED_URL=
set EMBED_MODEL=BAAI/bge-m3
set HF_HUB_OFFLINE=1
REM Unbuffered: otherwise stdout redirected to a log file stays block-buffered
REM and the log looks empty until the process exits.
set PYTHONUNBUFFERED=1
set KF_WARMUP=1
set KF_EMBED_MAX_LEN=512
set KF_EMBED_BATCH=8
REM Ingest runs on CPU: retrieve starts first and owns the GPU (4GB card).
set KF_EMBED_DEVICE=cpu
python -m uvicorn kf_ingest.app:app --port 8083 --host 127.0.0.1