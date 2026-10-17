"""一键启动：Docker 容器 + 检索服务（GPU）+ 摄入服务（CPU）。

## 为什么用 Python 而不是纯 .cmd

`.cmd` 文件由 cmd.exe 按**系统 ANSI 码页**（中文 Windows 是 GBK/936）解析。
用 UTF-8 写含中文的批处理，中文会变成乱码，而且乱码字节会把命令行本身撑断
——实测报出 `'句笉鍒?Docker' 不是内部或外部命令` 这类错，连 `goto`/`if` 都被破坏。

所以：`.cmd` 只做一件事——用纯 ASCII 调起本脚本；中文输出、流程控制、错误处理
全部留在这里（Python 内部统一 UTF-8，不受码页影响）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CONTAINERS = ["kf-postgres", "kf-qdrant"]
PORTS = {8082: "检索", 8083: "摄入"}


def say(msg: str = "") -> None:
    print(msg, flush=True)


def run(cmd: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except FileNotFoundError:
        return 127, f"找不到命令: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "超时"
    except Exception as exc:
        return 1, str(exc)


def port_listening(port: int) -> bool:
    code, out = run(["netstat", "-ano"], timeout=20)
    if code != 0:
        return False
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line.upper():
            return True
    return False


def ensure_docker() -> bool:
    say("[1/3] 检查 Docker ...")
    code, _ = run(["docker", "info"], timeout=25)
    if code == 0:
        say("      Docker 已就绪")
        return True

    exe = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Docker" / "Docker" / "Docker Desktop.exe"
    if not exe.exists():
        say(f"      [错误] Docker 未运行，且找不到 {exe}")
        say("      请手动启动 Docker Desktop 后重试")
        return False
    say(f"      Docker 未运行，正在启动 Docker Desktop ...")
    try:
        subprocess.Popen([str(exe)], close_fds=True)
    except Exception as exc:
        say(f"      [错误] 启动失败: {exc}")
        return False

    for i in range(1, 19):
        time.sleep(10)
        code, _ = run(["docker", "info"], timeout=20)
        if code == 0:
            say(f"      Docker 就绪（约 {i*10} 秒）")
            return True
    say("      [错误] 等待 180 秒仍未就绪")
    return False


def ensure_containers() -> bool:
    say("\n[2/3] 启动数据容器 ...")
    code, out = run(["docker", "start", *CONTAINERS], timeout=120)
    if code != 0:
        say(f"      [错误] docker start 失败: {out.strip()[:200]}")
        return False
    # 等 Postgres 真的能接受连接
    for i in range(1, 13):
        code, _ = run(["docker", "exec", "kf-postgres", "pg_isready", "-U", "kf"],
                      timeout=20)
        if code == 0:
            say(f"      Postgres 已就绪（约 {i*5} 秒）")
            break
        time.sleep(5)
    else:
        say("      [警告] Postgres 健康检查未通过，仍继续")

    for name in CONTAINERS:
        code, out = run(["docker", "inspect", "-f", "{{.State.Status}}", name],
                        timeout=20)
        state = out.strip() if code == 0 else "?"
        say(f"      {name:<14} {state}")
    return True


def launch(name: str, script: str, log_name: str) -> None:
    """后台启动服务，输出写到 logs/<service>.log。

    这里踩过两个坑，都记录一下：

    1. 用 `start "" cmd /c ...` 开新窗口：子进程**继承父进程的 stdout 句柄**，
       `start.cmd` 会一直等管道 EOF —— 服务明明起来了，批处理却"卡住不返回"
       （真机实测几十分钟不退出）。
    2. 改成 `CREATE_NEW_CONSOLE` + DEVNULL：父进程能退出了，但服务窗口是**空白**的，
       出问题看不到任何日志。

    最终方案：不开新窗口，stdout/stderr 重定向到日志文件。父进程干净退出，
    日志可随时查看（见下方提示）。
    """
    path = SCRIPTS / script
    if not path.exists():
        say(f"      [错误] 缺少 {path}")
        return
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / log_name
    try:
        handle = open(log_path, "a", encoding="utf-8", errors="replace")
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====\n")
        handle.flush()
        subprocess.Popen(
            ["cmd.exe", "/c", str(path)],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except Exception as exc:
        say(f"      [错误] 启动失败: {exc}")
        return
    say(f"      已启动 {name}  日志: logs\\{log_name}")


def pids_by_pattern() -> dict[int, str]:
    """按命令行找已在运行（或正在启动）的服务进程。

    只看端口不够：uvicorn 从进程创建到 listen 有几十秒窗口（要加载模型），
    这期间重复执行 `start.cmd` 会**再起一份**，于是同一服务堆出多份进程，
    既抢显存又让日志串台（真机实测：残留了 2 份 kf_retrieve）。
    """
    import json

    code, out = run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
    ], timeout=40)
    found: dict[int, str] = {}
    if code != 0 or not out.strip():
        return found
    try:
        data = json.loads(out)
    except Exception:
        return found
    for item in (data if isinstance(data, list) else [data]):
        if not isinstance(item, dict):
            continue
        cmdline = str(item.get("CommandLine") or "")
        if "kf_retrieve.app" in cmdline or "kf_ingest.app" in cmdline:
            found[int(item["ProcessId"])] = cmdline
    return found


def ensure_services() -> None:
    say("\n[3/3] 启动检索 / 摄入服务 ...")
    existing = pids_by_pattern()
    # 检索先启动：它要占 GPU（4GB 卡装不下两份嵌入模型），摄入随后回落 CPU。
    plans = {8082: ("kf-retrieve :8082", "run_retrieve.cmd", "retrieve.log",
                    "kf_retrieve.app"),
             8083: ("kf-ingest :8083", "run_ingest.cmd", "ingest.log",
                    "kf_ingest.app")}
    for port, (name, script, log_name, marker) in plans.items():
        same = [pid for pid, cmd in existing.items() if marker in cmd]
        if port_listening(port) or same:
            detail = f"端口 {port} 已监听" if port_listening(port) else f"进程 {same} 已存在"
            say(f"      {PORTS[port]}：{detail}，跳过")
            continue
        launch(name, script, log_name)


def main() -> int:
    say("=" * 62)
    say("  Knowledge Fabric 启动")
    say("=" * 62)
    if not ensure_docker():
        return 1
    if not ensure_containers():
        return 1
    ensure_services()

    say()
    say("=" * 62)
    say("  服务在后台运行，日志在 logs\\ 目录：")
    say("    logs\\retrieve.log   检索（8082）")
    say("    logs\\ingest.log     摄入（8083）")
    say()
    say("  实时看日志（另开一个窗口）：")
    say("    powershell -c \"Get-Content logs\\retrieve.log -Wait -Tail 20\"")
    say()
    say("  首次启动要加载模型，约 3 分钟：")
    say("    嵌入 bge-m3    约 45 秒")
    say("    重排 reranker   约 79 秒")
    say()
    say("  确认就绪（两个都 ready 才算好）：")
    say("    curl http://127.0.0.1:8082/health")
    say("    curl http://127.0.0.1:8083/health")
    say()
    say("  一步自检（推荐）：")
    say("    python scripts\\check_adaptive_rerank.py")
    say()
    say("  停止：  stop.cmd")
    say("  控制台：双击 console\\index.html")
    say("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
