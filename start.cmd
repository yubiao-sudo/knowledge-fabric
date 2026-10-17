@echo off
setlocal
chcp 65001 >nul
title Knowledge Fabric - 启动

echo ============================================================
echo  Knowledge Fabric 启动
echo ============================================================
echo.

REM ---------- 1) 依赖容器 ----------
echo [1/3] 启动 Postgres / Qdrant ...
docker info >nul 2>&1
if errorlevel 1 (
  echo   Docker 未运行，尝试启动 Docker Desktop ...
  if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
  ) else (
    echo   [错误] 找不到 Docker Desktop.exe，请手动启动 Docker
    pause
    exit /b 1
  )
  echo   等待 Docker 就绪（最多 180 秒）...
  for /l %%i in (1,1,18) do (
    timeout /t 10 /nobreak >nul
    docker info >nul 2>&1
    if not errorlevel 1 goto :docker_ready
  )
  echo   [错误] Docker 启动超时
  pause
  exit /b 1
)
:docker_ready
docker start kf-postgres kf-qdrant >nul 2>&1
echo   等待容器健康检查 ...
for /l %%i in (1,1,12) do (
  timeout /t 5 /nobreak >nul
  docker exec kf-postgres pg_isready -U kf >nul 2>&1
  if not errorlevel 1 goto :db_ready
)
echo   [警告] Postgres 健康检查未通过，继续尝试启动服务
:db_ready

REM ---------- 2) 检索服务（先启动，抢 GPU）----------
echo.
echo [2/3] 启动检索服务（8082，占用 GPU）...
if not exist "%~dp0scripts\run_retrieve.cmd" (
  echo   [错误] 缺少 scripts\run_retrieve.cmd
  pause
  exit /b 1
)
start "kf-retrieve :8082" cmd /c "%~dp0scripts\run_retrieve.cmd"

REM ---------- 3) 摄入服务（后启动，回落 CPU）----------
echo [3/3] 启动摄入服务（8083，CPU）...
start "kf-ingest :8083" cmd /c "%~dp0scripts\run_ingest.cmd"

echo.
echo ============================================================
echo  服务在独立窗口启动。首次启动要加载模型，约 3 分钟：
echo    - 嵌入 bge-m3     约 45 秒
echo    - 重排 reranker    约 79 秒
echo.
echo  确认就绪（两个都显示 ready=true 才算好）：
echo    curl http://127.0.0.1:8082/health
echo    curl http://127.0.0.1:8083/health
echo.
echo  或者直接跑自检：
echo    python scripts\check_adaptive_rerank.py
echo.
echo  控制台： 双击 console\index.html
echo ============================================================
pause
