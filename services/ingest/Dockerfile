FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates libgl1 libglib2.0-0 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# 两个入口：CLI（离线批处理）与 HTTP 接口（上传即摄入，供控制台 / agent 接入）
# CLI：      python -m kf_ingest.cli --help
# HTTP：     python -m uvicorn kf_ingest.app:app --host 0.0.0.0 --port 8083
CMD ["python", "-m", "kf_ingest.cli", "--help"]
