"""CLI 目录遍历测试：`--path` 给目录时的批量行为与退出码。

全部用 `--dry-run` 跑，**零外部依赖**（不连 Qdrant / Postgres），只验证遍历、
doc_id 派生、跳过项如实列出、以及错误参数被拦下。
"""

import json
import os

import pytest

from kf_ingest.cli import main


def _write(path, text, encoding="utf-8"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(text.encode(encoding))
    return path


def _run(capsys, argv):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_directory_mode_batches_supported_files(tmp_path, capsys):
    root = str(tmp_path)
    _write(os.path.join(root, "灯塔需求.md"), "# 灯塔\n\n互助平台需求说明。\n")
    _write(os.path.join(root, "其他文档.md"), "# 其他\n\n另一份文档正文。\n")
    _write(os.path.join(root, "旧稿.docx"), "not supported")
    _write(os.path.join(root, "note.txt"), "not supported")

    code, out, err = _run(capsys, [
        "--path", root, "--tenant", "demo", "--kb", "light", "--dry-run", "--json",
    ])

    assert code == 0
    report = json.loads(out)
    assert report["processed"] == 2
    assert report["succeeded"] == 2
    assert report["failed"] == 0
    # 不支持的格式必须被**列出**，不允许静默跳过
    assert sorted(os.path.basename(p) for p in report["skipped_unsupported"]) == ["note.txt", "旧稿.docx"]
    # doc_id 由文件名派生
    ids = sorted(d["doc_id"] for d in report["documents"])
    assert ids == ["其他文档", "灯塔需求"]
    assert report["dry_run"] is True


def test_directory_mode_doc_id_comes_from_filename_not_flag(tmp_path, capsys):
    root = str(tmp_path)
    _write(os.path.join(root, "手册A.md"), "# A\n\n正文甲。\n")

    code, out, _ = _run(capsys, ["--path", root, "--dry-run", "--json"])
    assert code == 0
    assert json.loads(out)["documents"][0]["doc_id"] == "手册A"


def test_directory_mode_rejects_explicit_doc_id(tmp_path, capsys):
    """目录模式下硬指定 --doc-id 会让多个文档互相覆盖，必须直接退出码 2 拦下。"""
    root = str(tmp_path)
    _write(os.path.join(root, "a.md"), "# a\n\n正文。\n")

    code, _, err = _run(capsys, ["--path", root, "--doc-id", "same", "--dry-run"])
    assert code == 2
    assert "不能指定 --doc-id" in err


def test_directory_mode_without_supported_files_exits_2(tmp_path, capsys):
    root = str(tmp_path)
    _write(os.path.join(root, "a.docx"), "x")
    _write(os.path.join(root, "b.txt"), "y")

    code, _, err = _run(capsys, ["--path", root, "--dry-run"])
    assert code == 2
    assert "没有受支持的文档" in err


def test_directory_mode_match_narrows_scope(tmp_path, capsys):
    root = str(tmp_path)
    _write(os.path.join(root, "灯塔需求.md"), "# 灯塔\n\n正文。\n")
    _write(os.path.join(root, "简历.md"), "# 简历\n\n正文。\n")

    code, out, _ = _run(capsys, ["--path", root, "--dry-run", "--match", "灯塔", "--json"])
    assert code == 0
    report = json.loads(out)
    assert report["processed"] == 1
    assert report["documents"][0]["doc_id"] == "灯塔需求"


def test_directory_mode_recursive(tmp_path, capsys):
    root = str(tmp_path)
    _write(os.path.join(root, "top.md"), "# top\n\n顶层正文。\n")
    _write(os.path.join(root, "nested", "deep.md"), "# deep\n\n子目录正文。\n")

    code, out, _ = _run(capsys, ["--path", root, "--dry-run", "--json"])
    assert code == 0
    assert json.loads(out)["processed"] == 1

    code, out, _ = _run(capsys, ["--path", root, "--dry-run", "--recursive", "--json"])
    assert code == 0
    assert json.loads(out)["processed"] == 2


def test_single_file_mode_still_honours_doc_id(tmp_path, capsys):
    path = _write(os.path.join(str(tmp_path), "手册.md"), "# 手册\n\n正文。\n")
    code, out, _ = _run(capsys, ["--path", path, "--doc-id", "custom-id", "--dry-run", "--json"])
    assert code == 0
    assert json.loads(out)["doc_id"] == "custom-id"


def test_single_file_mode_rejects_unsupported_format_gracefully(tmp_path, capsys):
    """单文件给一个不支持的格式：必须明确报错(退出码 1)并说明支持哪些格式，
    不能假装成功。"""
    path = _write(os.path.join(str(tmp_path), "a.docx"), "x")
    code, out, _ = _run(capsys, ["--path", path, "--dry-run", "--json"])
    assert code == 1
    summary = json.loads(out)
    assert summary["errors"]
    assert "不支持的文档格式" in summary["errors"][0]


def test_missing_path_exits_2(tmp_path, capsys):
    code, _, err = _run(capsys, ["--path", os.path.join(str(tmp_path), "nope")])
    assert code == 2
    assert "路径不存在" in err


def test_markdown_parse_warning_surfaces_in_summary(tmp_path, capsys):
    """Markdown 无分页这一事实必须出现在摘要里（`no_page_layout`），
    不能让下游误以为页码引用可用。"""
    path = _write(os.path.join(str(tmp_path), "m.md"), "# 标题\n\n正文段落。\n")
    code, out, _ = _run(capsys, ["--path", path, "--dry-run", "--json"])
    assert code == 0
    summary = json.loads(out)
    assert summary["stages"]["parse"]["scanner"] == "markdown"
    assert "no_page_layout" in summary["stages"]["parse"]["warnings"]
