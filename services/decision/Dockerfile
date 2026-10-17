FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LAYA_CACHE_DIR=/models \
    DECISION_PORT=8090

WORKDIR /app

# torch 需要的基础运行库
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://localhost:8090/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
