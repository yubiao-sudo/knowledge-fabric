@echo off
setlocal
chcp 65001 >nul
title Knowledge Fabric - 停止

echo 停止检索/摄入服务（容器保留，数据不丢）...

for %%P in (8082 8083) do (
  for /f "tokens=5" %%A in ('netstat -ano ^| findstr ":%%P" ^| findstr "LISTENING"') do (
    taskkill /PID %%A /F >nul 2>&1
    echo   已停端口 %%P ^(PID %%A^)
  )
)

echo.
echo 如需连容器一起停：  docker stop kf-postgres kf-qdrant
echo 数据在 Docker volume 里，不会因停止而丢失。
timeout /t 3 /nobreak >nul
