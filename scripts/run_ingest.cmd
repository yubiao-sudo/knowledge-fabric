@echo off
setlocal
cd /d F:\knowledge-fabric\services\ingest
set PYTHONPATH=F:\knowledge-fabric\services\ingest
set QDRANT_URL=http://localhost:6333
set DATABASE_URL=postgresql://kf:kf_dev_password@localhost:15432/knowledge_fabric
set TEI_EMBED_URL=
set EMBED_MODEL=BAAI/bge-m3
set HF_HUB_OFFLINE=1
set KF_WARMUP=1
set KF_EMBED_MAX_LEN=512
set KF_EMBED_BATCH=8
REM 摄入走 CPU：检索服务先启动已占用 GPU，这里不跟它抢
set KF_EMBED_DEVICE=cpu
python -m uvicorn kf_ingest.app:app --port 8083 --host 127.0.0.1
