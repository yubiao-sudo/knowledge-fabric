"""控制台验收：在真实浏览器里点一遍 + 量布局，失败退出非零。

## 为什么需要它

控制台是 `file://` 打开的单页，没有构建、没有类型检查、也没有框架兜底。
改完只能靠肉眼点一遍——而"点了没反应"恰恰是最难靠肉眼发现的一类问题
（比如按钮其实被禁用了、默认查询在该库里本来就该拒答、脚本早退导致
后面的监听全没挂上）。

这个脚本用 headless Edge/Chrome 真加载页面，然后：
  1. **交互**：切遍 5 个页签，点生成接入码、点 Agent 视角预览（真实检索）、
     校验接入码格式、校验未选文件时按钮禁用、校验状态灯已更新；
  2. **布局**：量关键元素的盒子，查横向溢出、零尺寸、按钮过窄、文字裁切、
     窄屏适配；
  3. **降级**：把服务地址指向死端口，确认页面不炸、提示可读、按钮能恢复。

## 用法

    python scripts/check_console.py

需要本机有 Edge 或 Chrome。找不到浏览器时**跳过并返回 0**（不算失败），
避免在无 GUI 的环境里假报警。要强制跳过用 `--skip`。
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSOLE = os.path.join(ROOT, "console", "index.html")

BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

# ---------------------------------------------------------------- 注入脚本

INTERACT = r"""
<div id="__test"></div>
<script>
(function () {
  const log = [], errs = [];
  window.addEventListener("error", (e) => errs.push("error: " + (e.message || e)));
  window.addEventListener("unhandledrejection", (e) =>
    errs.push("reject: " + ((e.reason && e.reason.message) || e.reason)));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const tab = (n) => document.querySelector('.tab[data-tab="' + n + '"]');
  const txt = (id) => (document.getElementById(id) || {}).textContent || "";
  const html = (id) => (document.getElementById(id) || {}).innerHTML || "";
  async function step(name, fn) {
    try { await fn(); log.push("OK   " + name); }
    catch (e) { log.push("FAIL " + name + " :: " + ((e && e.message) || e)); }
  }
  const assert = (c, m) => { if (!c) throw new Error(m); };
  (async () => {
    await sleep(1200);
    await step("5 个导航项与面板一一对应", async () => {
      const names = [...document.querySelectorAll(".tab")].map((t) => t.dataset.tab);
      assert(names.length === 5, "导航项数 " + names.length);
      for (const n of names) assert(document.getElementById("panel-" + n), "缺面板 " + n);
    });
    await step("切到文档库并渲染", async () => {
      tab("docs").click(); await sleep(2200);
      assert(!document.getElementById("panel-docs").hidden, "面板没显示");
      assert(html("docs-result").length > 40, "docs-result 为空");
      assert(txt("page-title") === "文档库", "标题没换");
    });
    await step("切到检索页并加载档位", async () => {
      tab("search").click(); await sleep(2200);
      assert(document.querySelectorAll("#profile option").length >= 1, "无档位");
    });
    await step("切到接入页并渲染令牌/配置", async () => {
      tab("access").click(); await sleep(1600);
      assert(html("keys-list").length > 20, "keys-list 空");
      assert(txt("agent-config").indexOf("kb_base_url") >= 0, "配置没渲染");
    });
    await step("生成接入码（格式 KF-XXXX-XXXX）", async () => {
      document.getElementById("btn-pairing").click(); await sleep(2600);
      const code = txt("pairing-code").trim();
      assert(/^KF-[A-Z0-9]{4}-[A-Z0-9]{4}$/.test(code), "格式不对: " + code);
      assert(!document.getElementById("pairing-box").hidden, "框没显示");
    });
    await step("Agent 视角预览（真实检索）", async () => {
      document.getElementById("btn-preview").click(); await sleep(9000);
      const b = html("preview-result");
      assert(b.length > 60, "预览为空");
      assert(/refused/.test(b), "没有 refused 字段");
    });
    await step("切到服务设置并渲染配置表", async () => {
      tab("settings").click(); await sleep(2500);
      assert(html("settings-detail").indexOf("重排模型") >= 0, "配置表没渲染");
    });
    await step("回到摄入页：未选文件时按钮禁用", async () => {
      tab("ingest").click(); await sleep(400);
      assert(document.getElementById("btn-ingest").disabled === true, "没禁用");
      assert(html("flow").indexOf("chip") >= 0, "阶段芯片没渲染");
    });
    await step("服务状态灯已更新", async () => {
      const a = txt("lbl-retrieve"), b = txt("lbl-ingest");
      assert(a && a !== "—", "检索状态未更新");
      assert(b && b !== "—", "摄入状态未更新");
    });
    document.getElementById("__test").textContent =
      "===RESULT===\n" + log.join("\n") + "\n===ERRORS===\n"
      + (errs.length ? errs.join("\n") : "(none)") + "\n===END===";
  })();
})();
</script>
"""

AUDIT = r"""
<div id="__audit"></div>
<script>
(function () {
  const out = [];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function box(sel) {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { w: Math.round(r.width), h: Math.round(r.height) };
  }
  function ng(name, sel, minW, minH) {
    const b = box(sel);
    if (!b) { out.push("FAIL " + name + " : 找不到 " + sel); return; }
    if (b.w < minW || b.h < minH) {
      out.push("FAIL " + name + " : " + b.w + "x" + b.h + " < " + minW + "x" + minH);
    } else { out.push("OK   " + name + " : " + b.w + "x" + b.h); }
  }
  (async () => {
    await sleep(1500);
    const de = document.documentElement;
    const ov = de.scrollWidth - de.clientWidth;
    out.push((ov <= 1 ? "OK   " : "FAIL ") + "无横向溢出 : " + ov);
    ng("侧边栏", ".sidebar", 200, 400);
    const tabs = [...document.querySelectorAll(".tab")];
    const bad = tabs.filter((t) => {
      const r = t.getBoundingClientRect();
      return !t.offsetParent || r.width < 80 || r.height < 24;
    }).map((t) => t.dataset.tab);
    out.push((bad.length === 0 ? "OK   " : "FAIL ") + "5 个导航项可点击 : "
      + (bad.length ? bad.join(",") : "全部正常"));
    ng("顶栏", ".topbar", 600, 40);
    ng("租户输入", "#tenant", 60, 24);
    const vis = ["ingest","docs","search","access","settings"].filter((n) => {
      const p = document.getElementById("panel-" + n); return p && !p.hidden;
    });
    out.push((vis.length === 1 && vis[0] === "ingest" ? "OK   " : "FAIL ")
      + "初始只显示摄入面板 : " + vis.join(","));
    ng("拖放区", "#drop", 200, 60);
    ng("开始摄入按钮", "#btn-ingest", 96, 28);
    ng("阶段流水线", "#flow", 200, 16);
    const b = document.getElementById("btn-ingest");
    out.push((b.disabled && parseFloat(getComputedStyle(b).opacity) < 0.9 ? "OK   " : "FAIL ")
      + "禁用态有视觉区分");
    document.querySelector('.tab[data-tab="search"]').click();
    await sleep(1800);
    ng("查询输入", "#query", 300, 50);
    ng("检索按钮", "#btn-search", 96, 28);
    out.push((document.querySelectorAll("#quick-queries button").length >= 3 ? "OK   " : "FAIL ")
      + "快捷问题按钮 >= 3");
    document.querySelector('.tab[data-tab="access"]').click();
    await sleep(1800);
    ng("令牌列表区", "#keys-list", 200, 30);
    const kl = document.getElementById("keys-list");
    out.push((kl.scrollHeight <= kl.clientHeight + 4 || getComputedStyle(kl).overflowY === "auto"
      ? "OK   " : "WARN ") + "长列表限高可滚动");
    ng("配置输出", "#agent-config", 200, 60);
    document.querySelector('.tab[data-tab="settings"]').click();
    await sleep(1800);
    ng("服务地址输入", "#base-ingest", 150, 24);
    ng("配置详情", "#settings-detail", 200, 60);
    const clipped = [];
    document.querySelectorAll(".hint,.stat b,label").forEach((el) => {
      if (el.scrollHeight - el.clientHeight > 4 && getComputedStyle(el).overflow !== "visible") {
        clipped.push(el.id || el.className || el.tagName);
      }
    });
    out.push((clipped.length === 0 ? "OK   " : "WARN ") + "无文字裁切 : " + clipped.length);
    de.style.width = "760px";
    await sleep(400);
    const ov2 = de.scrollWidth - de.clientWidth;
    out.push((ov2 <= 2 ? "OK   " : "WARN ") + "窄屏 760px 无横向溢出 : " + ov2);
    document.getElementById("__audit").textContent =
      "===RESULT===\n" + out.join("\n") + "\n===END===";
  })();
})();
</script>
"""

DEGRADED_PRE = r"""
<script>
  try {
    localStorage.setItem("kf.console.base-retrieve", "http://127.0.0.1:59999");
    localStorage.setItem("kf.console.base-ingest", "http://127.0.0.1:59998");
  } catch (e) {}
</script>
"""

DEGRADED = r"""
<div id="__deg"></div>
<script>
(function () {
  const out = [], errs = [];
  window.addEventListener("error", (e) => errs.push("error: " + (e.message || e)));
  window.addEventListener("unhandledrejection", (e) =>
    errs.push("reject: " + ((e.reason && e.reason.message) || e.reason)));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const txt = (id) => (document.getElementById(id) || {}).textContent || "";
  const html = (id) => (document.getElementById(id) || {}).innerHTML || "";
  (async () => {
    await sleep(3500);
    out.push((txt("lbl-retrieve").indexOf("未连接") >= 0 ? "OK   " : "FAIL ")
      + "检索显示未连接");
    out.push((txt("lbl-ingest").indexOf("未连接") >= 0 ? "OK   " : "FAIL ")
      + "摄入显示未连接");
    out.push((document.getElementById("base-retrieve").value.indexOf("59999") >= 0 ? "OK   " : "FAIL ")
      + "记住自定义地址");
    document.querySelector('.tab[data-tab="search"]').click();
    await sleep(400);
    document.getElementById("btn-search").click();
    await sleep(2500);
    out.push((/检索失败|Failed|服务/.test(html("search-result")) ? "OK   " : "FAIL ")
      + "检索失败有可读提示");
    out.push((document.getElementById("btn-search").disabled === false ? "OK   " : "FAIL ")
      + "失败后按钮恢复");
    document.querySelector('.tab[data-tab="access"]').click();
    await sleep(400);
    document.getElementById("btn-pairing").click();
    await sleep(2500);
    out.push((/失败/.test(txt("pairing-code")) ? "OK   " : "FAIL ")
      + "接入码失败有明确文案");
    out.push((document.getElementById("btn-pairing").disabled === false ? "OK   " : "FAIL ")
      + "接入码按钮恢复");
    out.push((/失败|错误|连不上|HTTP/.test(html("keys-list")) ? "OK   " : "FAIL ")
      + "令牌列表失败有提示");
    document.querySelector('.tab[data-tab="settings"]').click();
    await sleep(2500);
    out.push((/未连接/.test(html("settings-detail")) ? "OK   " : "FAIL ")
      + "设置页标出未连接");
    out.push("---- 未捕获异常 ----");
    out.push(errs.length ? errs.join("\n") : "(none)");
    document.getElementById("__deg").textContent =
      "===RESULT===\n" + out.join("\n") + "\n===END===";
  })();
})();
</script>
"""


def find_browser() -> str | None:
    for path in BROWSER_CANDIDATES:
        if os.path.isfile(path):
            return path
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_page(kind: str) -> str:
    with io.open(CONSOLE, "r", encoding="utf-8") as fh:
        html = fh.read()
    if kind == "interact":
        html = html.replace("</body>", INTERACT + "\n</body>", 1)
    elif kind == "audit":
        html = html.replace("</body>", AUDIT + "\n</body>", 1)
    else:
        html = html.replace("<body>", "<body>\n" + DEGRADED_PRE, 1)
        html = html.replace("</body>", DEGRADED + "\n</body>", 1)
    path = os.path.join(tempfile.gettempdir(), f"kf_console_{kind}.html")
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def run(browser: str, page: str, budget_ms: int) -> str:
    profile = tempfile.mkdtemp(prefix="kf_cchk_")
    try:
        proc = subprocess.run(
            [
                browser, "--headless=new", "--disable-gpu", "--no-sandbox",
                f"--user-data-dir={profile}",
                f"--virtual-time-budget={budget_ms}",
                "--window-size=1500,1080", "--dump-dom",
                "file:///" + page.replace("\\", "/"),
            ],
            capture_output=True, timeout=180,
        )
        return proc.stdout.decode("utf-8", "replace")
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def parse(dom: str) -> tuple[list[str], list[str]]:
    m = re.search(r"===RESULT===(.*?)===END===", dom, re.S)
    if not m:
        return [], ["(没有拿到结果：脚本可能在更早处抛错)"]
    block = m.group(1)
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    body = [ln for ln in lines if ln.startswith(("OK", "FAIL", "WARN", "----")) or ln == "(none)"]
    return body, []


def main() -> int:
    ap = argparse.ArgumentParser(description="控制台验收（headless 浏览器）")
    ap.add_argument("--skip", action="store_true", help="跳过（无浏览器环境）")
    args = ap.parse_args()

    if args.skip:
        print("已按 --skip 跳过控制台验收")
        return 0

    browser = find_browser()
    if not browser:
        print("未找到 Edge/Chrome，跳过控制台验收（不算失败）")
        return 0

    print("控制台验收 ·", CONSOLE)
    print("浏览器:", browser)
    total_fail = 0
    for kind, label, budget in (
        ("interact", "交互", 32000),
        ("audit", "布局", 26000),
        ("degraded", "降级", 26000),
    ):
        page = build_page(kind)
        dom = run(browser, page, budget)
        body, errs = parse(dom)
        fails = [ln for ln in body if ln.startswith("FAIL")] + errs
        total_fail += len(fails)
        print(f"\n--- {label}（{kind}）: "
              f"{sum(1 for x in body if x.startswith('OK'))} 通过 / {len(fails)} 失败")
        for ln in body:
            print("   ", ln)
        for e in errs:
            print("    FAIL", e)

    print("\n" + "=" * 56)
    if total_fail:
        print(f"结果：{total_fail} 项失败")
        return 1
    print("结果：全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
