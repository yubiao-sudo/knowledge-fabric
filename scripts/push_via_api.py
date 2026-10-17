"""通过 GitHub REST API 把本地仓库推上去（绕过被封的 github.com:443）。

## 为什么需要这个脚本

本机实测（2026-10）：`github.com:443` 连不上，但 **`api.github.com`、`ssh.github.com:443`、
`github.com:22` 都可达**。`git push` 走的是 `github.com:443`，所以常规方式必失败。

GitHub 的 Git Data API（`/git/blobs`、`/git/trees`、`/git/commits`、`/git/refs`）
在 `api.github.com` 上提供等价能力：先上传 blob，再建 tree/commit，最后更新 ref
——效果等同于一次 push，且不需要 `github.com:443`。

## 认证

需要一个 **Personal Access Token**（classic 勾 `repo`，或 fine-grained 给
`Contents: Read and write`）。**账号密码不行**：GitHub 已废弃密码认证（实测 401）。

## 用法

    set GITHUB_TOKEN=ghp_xxx
    python scripts/push_via_api.py --dry-run
    python scripts/push_via_api.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import httpx

API = "https://api.github.com"


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {out.stderr[:300]}")
    return out.stdout.strip()


def repo_root() -> str:
    return git("rev-parse", "--show-toplevel")


def local_files(ref: str) -> List[str]:
    """该 commit 里的全部文件路径（相对仓库根，正斜杠）。

    必须关掉 `core.quotepath`：默认情况下 git 会把非 ASCII 路径转义成
    `"docs/\\344\\275\\277..."` 这种形式，直接拿去 open() 会报 Invalid argument。
    """
    listing = git("-c", "core.quotepath=false", "ls-tree", "-r", "--name-only", ref)
    return [p for p in listing.splitlines() if p.strip()]


def file_bytes(root: str, rel: str) -> bytes:
    with open(os.path.join(root, rel.replace("/", os.sep)), "rb") as fh:
        return fh.read()


class GH:
    def __init__(self, token: str) -> None:
        self.token = token
        self.client = httpx.Client(
            base_url=API,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "knowledge-fabric-push",
            },
            timeout=120.0,
        )

    def _call(self, method: str, url: str, **kw: Any) -> Any:
        for attempt in range(4):
            try:
                resp = self.client.request(method, url, **kw)
            except Exception as exc:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code < 300:
                return resp.json() if resp.content else {}
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {url} -> {resp.status_code} {resp.text[:300]}")
        raise RuntimeError("unreachable")

    def me(self) -> Dict[str, Any]:
        return self._call("GET", "/user")

    def repo(self, owner: str, name: str) -> Dict[str, Any]:
        return self._call("GET", f"/repos/{owner}/{name}")

    def create_blob(self, owner: str, name: str, data: bytes) -> str:
        payload = {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        }
        return self._call("POST", f"/repos/{owner}/{name}/git/blobs", json=payload)["sha"]

    def create_tree(self, owner: str, name: str,
                    entries: List[Dict[str, Any]]) -> str:
        return self._call("POST", f"/repos/{owner}/{name}/git/trees",
                          json={"tree": entries})["sha"]

    def create_commit(self, owner: str, name: str, message: str,
                      tree_sha: str, parents: List[str]) -> str:
        return self._call("POST", f"/repos/{owner}/{name}/git/commits",
                          json={"message": message, "tree": tree_sha,
                                "parents": parents})["sha"]

    def ref(self, owner: str, name: str, branch: str) -> Dict[str, Any] | None:
        try:
            return self._call("GET", f"/repos/{owner}/{name}/git/ref/heads/{branch}")
        except RuntimeError as exc:
            if "-> 404" in str(exc) or "-> 409" in str(exc):
                return None
            raise

    def create_ref(self, owner: str, name: str, branch: str, sha: str) -> Dict[str, Any]:
        return self._call("POST", f"/repos/{owner}/{name}/git/refs",
                          json={"ref": f"refs/heads/{branch}", "sha": sha})

    def update_ref(self, owner: str, name: str, branch: str, sha: str) -> Dict[str, Any]:
        return self._call("PATCH", f"/repos/{owner}/{name}/git/refs/heads/{branch}",
                          json={"sha": sha, "force": False})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="yubiao-sudo")
    ap.add_argument("--repo", default="knowledge-fabric")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--ref", default="HEAD", help="要推送的本地 commit")
    ap.add_argument("--message", default="", help="缺省用本地 commit 的说明")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-blob", action="store_true",
                    help="复用远端已有 blob（同内容不会重复上传，此处仅作提示）")
    args = ap.parse_args()

    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if not token:
        print("缺少 GITHUB_TOKEN 环境变量（需要 Personal Access Token）")
        return 2

    root = repo_root()
    files = local_files(args.ref)
    message = args.message or git("log", "-1", "--pretty=%B", args.ref)
    parents_local = git("rev-list", "--max-parents=0", args.ref).splitlines()
    # 空仓库首次推送：父提交为空；后续推送应使用远端 HEAD 作为父
    print(f"仓库根: {root}")
    print(f"文件数: {len(files)}")
    print(f"提交说明: {message.splitlines()[0] if message else '(空)'}")

    if args.dry_run:
        total = 0
        for rel in files:
            total += len(file_bytes(root, rel))
        print(f"[dry-run] 待上传 {total/1024/1024:.2f} MB，未发送任何请求")
        return 0

    gh = GH(token)
    me = gh.me()
    print(f"认证用户: {me['login']}")

    info = gh.repo(args.owner, args.repo)
    print(f"目标仓库: {info['full_name']}  private={info['private']}  "
          f"大小={info['size']} KB  default_branch={info['default_branch']}")

    existing = gh.ref(args.owner, args.repo, args.branch)
    parents: List[str] = []
    if existing:
        parents = [existing["object"]["sha"]]
        print(f"远端 {args.branch} 已存在，将作为父提交: {parents[0][:12]}")
    else:
        print(f"远端 {args.branch} 不存在，将创建首个提交")

    # 1) 逐文件上传 blob（内容按 SHA 去重，GitHub 侧自动处理）
    entries: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for idx, rel in enumerate(files, 1):
        data = file_bytes(root, rel)
        sha = gh.create_blob(args.owner, args.repo, data)
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": sha})
        if idx % 20 == 0 or idx == len(files):
            print(f"  blob {idx}/{len(files)}  ({time.perf_counter()-start:.1f}s)")

    # 2) tree -> 3) commit -> 4) ref
    tree_sha = gh.create_tree(args.owner, args.repo, entries)
    print(f"tree: {tree_sha[:12]}")
    commit_sha = gh.create_commit(args.owner, args.repo, message, tree_sha, parents)
    print(f"commit: {commit_sha[:12]}")

    if existing:
        gh.update_ref(args.owner, args.repo, args.branch, commit_sha)
        print(f"已更新 refs/heads/{args.branch}")
    else:
        gh.create_ref(args.owner, args.repo, args.branch, commit_sha)
        print(f"已创建 refs/heads/{args.branch}")

    print(f"\n完成：https://github.com/{args.owner}/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
