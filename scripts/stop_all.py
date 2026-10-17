"""一键停止：结束检索/摄入服务进程（容器保留，数据不丢）。

同 `start_all.py`：中文输出放在 Python 里，`.cmd` 只做纯 ASCII 调起，
避免 cmd.exe 按 GBK 解析 UTF-8 批处理导致乱码与语法崩坏。

**不能只按端口杀**：实测出现过"绑定端口失败但仍存活的孤儿进程"——
它不占端口，`netstat` 查不到，却一直握着旧的日志句柄、占着显存。
所以这里同时按**命令行特征**匹配，两类都杀。
"""
from __future__ import annotations

import subprocess
import sys

PORTS = {8082: "检索服务", 8083: "摄入服务"}
# 服务进程的命令行特征（uvicorn 的 app 路径）
PATTERNS = ("kf_retrieve.app", "kf_ingest.app")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        return 1, str(exc)


def pids_on_port(port: int) -> list[int]:
    code, out = run(["netstat", "-ano"])
    if code != 0:
        return []
    pids = set()
    for line in out.splitlines():
        upper = line.upper()
        if "LISTENING" not in upper or f":{port}" not in line:
            continue
        parts = line.split()
        if parts and parts[-1].isdigit():
            pids.add(int(parts[-1]))
    return sorted(pids)


def pids_by_cmdline() -> dict[int, str]:
    """按命令行匹配服务进程（含绑定失败、不占端口的孤儿）。"""
    code, out = run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
    ], timeout=40)
    found: dict[int, str] = {}
    if code != 0 or not out.strip():
        return found
    import json

    try:
        data = json.loads(out)
    except Exception:
        return found
    for item in (data if isinstance(data, list) else [data]):
        if not isinstance(item, dict):
            continue
        cmdline = str(item.get("CommandLine") or "")
        if any(p in cmdline for p in PATTERNS):
            found[int(item["ProcessId"])] = cmdline
    return found


def main() -> int:
    say("停止 Knowledge Fabric 服务（容器保留，数据不丢）")
    say()

    # 1) 端口占用者
    victims: dict[int, str] = {}
    for port, label in PORTS.items():
        for pid in pids_on_port(port):
            victims[pid] = f"端口 {port}（{label}）"

    # 2) 命令行匹配（补上不占端口的孤儿）
    for pid, cmdline in pids_by_cmdline().items():
        if pid in victims:
            continue
        which = "检索" if "kf_retrieve" in cmdline else "摄入"
        victims[pid] = f"{which}服务（未占端口，按命令行匹配）"

    if not victims:
        say("  没有发现运行中的服务进程")
    for pid, why in sorted(victims.items()):
        code, _ = run(["taskkill", "/PID", str(pid), "/F"])
        if code == 0:
            say(f"  已停止 PID {pid} — {why}")
        else:
            say(f"  [警告] 无法停止 PID {pid}（可能需要管理员权限）")

    say()
    if victims:
        say(f"共停止 {len(victims)} 个进程。")
        say("提示：如显存没释放，等几秒再看 nvidia-smi。")
    say("如需连数据库一起停：  docker stop kf-postgres kf-qdrant")
    say("数据在 Docker volume 里，停止不会丢。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
