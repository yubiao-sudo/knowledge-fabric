"""摄入 CLI 入口。

    python -m kf_ingest.cli --path ./手册.pdf --tenant default --kb demo --doc-id manual_v1
    python -m kf_ingest.cli --path ./手册.pdf --dry-run          # 只跑到切片，不调外部服务
    python -m kf_ingest.cli --path ./手册.pdf --json             # 输出机读摘要

`--path` **同时接受文件与目录**：给目录时遍历其中所有受支持的文档（PDF / Markdown）
逐个摄入，每个文档按文件名派生 doc_id。这是「一个公司有很多文档」的批量入口。

默认行为：读环境变量里的 QDRANT_URL / DATABASE_URL，能连就连；连不上只跳过，
并把跳过原因写进摘要的 notes。因此「无依赖也能跑通切片」是刻意的默认。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .clients import (
    connect_pg,
    connect_qdrant,
    database_url,
    default_collection,
    embed_dim,
    qdrant_url,
)
from .models import IngestContext
from .parse import SUPPORTED_SUFFIXES, discover_documents
from .pipeline import ingestion_summary, run_ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kf_ingest",
        description="Knowledge Fabric 摄入：解析 → 清洗 → 切片 → 增强 → 嵌入 → 索引",
    )
    parser.add_argument("--path", required=True,
                        help="本地文档路径，或一个目录（目录=批量遍历其中 "
                             f"{'/'.join(sorted(SUPPORTED_SUFFIXES))}）")
    parser.add_argument("--tenant", default=os.environ.get("DEFAULT_TENANT", "default"))
    parser.add_argument("--kb", default="default")
    parser.add_argument("--doc-id", dest="doc_id", default="",
                        help="仅单文件有效；目录模式下由文件名派生（指定会报错）")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--doc-type", dest="doc_type", default="",
                        help="manual / contract / regulation / code ...")
    parser.add_argument("--lang", default="", help="语言提示；留空自动判定")
    parser.add_argument("--strategy", default="", help="强制切片策略；留空按文档特征路由")
    parser.add_argument("--source-uri", dest="source_uri", default="",
                        help="对象存储路径；留空则用本地路径")
    parser.add_argument("--collection", default="", help="Qdrant 集合名；留空自动派生")
    parser.add_argument("--max-pages", dest="max_pages", type=int, default=0)
    parser.add_argument("--recursive", action="store_true",
                        help="目录模式下递归子目录（默认只扫顶层）")
    parser.add_argument("--match", default="",
                        help="目录模式下只处理文件名包含该子串的文档（便于先小范围试跑）")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="只跑到切片与关键词，不调外部服务、不落库")
    parser.add_argument("--no-enrich", dest="no_enrich", action="store_true",
                        help="跳过生成类增强（上下文前缀 / 假设问题）")
    parser.add_argument("--graph", action="store_true",
                        help="开启 GraphRAG 建图（同时把 knowledge_bases.graph_enabled 置 TRUE）")
    parser.add_argument("--acl", default="",
                        help="块级 ACL，逗号分隔，如 user:u1,group:g1,!user:u9（! 前缀 = deny）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not os.path.exists(args.path):
        print(f"路径不存在: {args.path}", file=sys.stderr)
        return 2

    if os.path.isdir(args.path):
        return _ingest_directory(args)
    return _ingest_file(args, args.path)


# ------------------------------------------------------------------ 单文件
def _ingest_file(args: argparse.Namespace, path: str) -> int:
    qdrant, pg = _connect(args)
    collection = args.collection or default_collection(args.tenant, args.kb)
    try:
        doc_id = args.doc_id or os.path.splitext(os.path.basename(path))[0]
        ctx = _run_one(args, path, doc_id, qdrant, pg, collection)
    finally:
        _close_all(pg, qdrant)

    summary = ingestion_summary(ctx, collection=collection, dry_run=bool(args.dry_run))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_human(summary)
    return 0 if not ctx.errors else 1


# ------------------------------------------------------------------ 目录批量
def _ingest_directory(args: argparse.Namespace) -> int:
    """遍历目录，对每个受支持的文档**逐个**走同一条流水线。

    两个如实口径：
      - 不支持的格式**列出来**（`skipped_unsupported`），不静默跳过；
      - 单个文档失败不中断其余文档，失败数计入 `failed` 并决定退出码。
    """
    if args.doc_id:
        print("目录模式下不能指定 --doc-id：doc_id 由文件名派生，硬指定会让多个文档互相覆盖",
              file=sys.stderr)
        return 2

    todo, skipped = discover_documents(
        args.path, recursive=args.recursive, match=args.match or None
    )
    if not todo:
        print(
            f"目录内没有受支持的文档（支持 {', '.join(sorted(SUPPORTED_SUFFIXES))}）："
            f"{os.path.abspath(args.path)}",
            file=sys.stderr,
        )
        if skipped:
            print(f"另跳过 {len(skipped)} 个不支持的格式", file=sys.stderr)
        return 2

    qdrant, pg = _connect(args)
    collection = args.collection or default_collection(args.tenant, args.kb)
    summaries: List[Dict[str, Any]] = []
    failures = 0
    try:
        for index, path in enumerate(todo, 1):
            name = os.path.basename(path)
            if not args.json:
                print(f"[{index}/{len(todo)}] {name}", flush=True)
            ctx = _run_one(args, path, os.path.splitext(name)[0], qdrant, pg, collection)
            summary = ingestion_summary(ctx, collection=collection, dry_run=bool(args.dry_run))
            summary["source_path"] = path
            summary["ok"] = not ctx.errors
            summaries.append(summary)
            if ctx.errors:
                failures += 1
            if not args.json:
                _print_human(summary)
    finally:
        _close_all(pg, qdrant)

    report = {
        "dir": os.path.abspath(args.path),
        "tenant": args.tenant,
        "kb": args.kb,
        "collection": collection,
        "dry_run": bool(args.dry_run),
        "recursive": bool(args.recursive),
        "processed": len(todo),
        "succeeded": len(todo) - failures,
        "failed": failures,
        "skipped_unsupported": skipped,
        "documents": summaries,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if skipped:
            print(f"\n跳过 {len(skipped)} 个不支持的格式（{', '.join(sorted(SUPPORTED_SUFFIXES))} 之外）：")
            for path in skipped:
                print(f"  - {os.path.basename(path)}")
        print(f"\n完成：成功 {report['succeeded']} / 失败 {report['failed']} / "
              f"跳过(不支持格式) {len(skipped)}")
    return 1 if failures else 0


# ------------------------------------------------------------------ 共用
def _connect(args: argparse.Namespace) -> Tuple[Any, Any]:
    """`dry_run` 一律不连外部依赖（这条是零依赖跑通的保证）。"""
    if args.dry_run:
        return None, None
    return connect_qdrant(qdrant_url()), connect_pg(database_url())


def _run_one(
    args: argparse.Namespace,
    path: str,
    doc_id: str,
    qdrant: Any,
    pg: Any,
    collection: str,
) -> IngestContext:
    ctx = IngestContext(
        tenant_id=args.tenant,
        kb_id=args.kb,
        doc_id=doc_id,
        source_uri=args.source_uri or path,
        source_path=path,
        version=args.version,
        lang=args.lang,
        doc_type=args.doc_type,
        graph=bool(args.graph),
    )
    run_ingest(
        ctx,
        strategy=args.strategy or None,
        dry_run=args.dry_run,
        qdrant_client=qdrant,
        pg_conn=pg,
        collection=collection,
        dim=embed_dim(),
        max_pages=args.max_pages or None,
        with_enrich=not args.no_enrich,
        graph=bool(args.graph),
        acl_spec=args.acl,
    )
    return ctx


def _close_all(*objects: Any) -> None:
    for obj in objects:
        if obj is None:
            continue
        try:
            obj.close()
        except Exception:
            pass


def _print_human(summary: Dict[str, Any]) -> None:
    counts = summary["counts"]
    print(f"文档 {summary['doc_id']} v{summary['version']}  lang={summary['lang']}  "
          f"strategy={summary['strategy']}")
    print(f"  父块 {counts['parents']}  子块 {counts['children']}  "
          f"已嵌入 {counts['embedded']}  已索引 {counts['indexed']}")
    clean = summary["cleaning"]
    print(f"  清洗：移除 {clean['removed_blocks']} 块（{clean['removed_by_type']}），"
          f"去重 {clean['deduped_blocks']} 块")
    chunk_stats = (summary["stages"] or {}).get("chunk") or {}
    if chunk_stats:
        print(f"  切片：平均子块 {chunk_stats.get('avg_child_tokens')} token，"
              f"最大 {chunk_stats.get('max_child_tokens')}，"
              f"原子子块 {chunk_stats.get('atomic_children')}")
    for name in ("parse", "enrich", "embed", "index", "opensearch"):
        stage = (summary["stages"] or {}).get(name) or {}
        notes = stage.get("notes") if isinstance(stage, dict) else None
        if notes:
            print(f"  {name} 提示：{'; '.join(str(n) for n in notes)}")
    parse_stats = (summary["stages"] or {}).get("parse") or {}
    parse_warnings = parse_stats.get("warnings") if isinstance(parse_stats, dict) else None
    if parse_warnings:
        print(f"  解析警告：{'; '.join(str(w) for w in parse_warnings)}")
    derive = (summary["stages"] or {}).get("derive") or {}
    if derive:
        print(f"  派生：ACL 规则 {derive.get('acl_rules', 0)}，"
              f"结构化记录 {derive.get('structured_records', 0)}，"
              f"实体 {derive.get('entity_norms', 0)}")
    graph_stats = (summary["stages"] or {}).get("index", {}) or {}
    graph_stats = graph_stats.get("graph") if isinstance(graph_stats, dict) else None
    if graph_stats:
        print(f"  建图：实体 {graph_stats.get('entities', 0)}，边 {graph_stats.get('edges', 0)}，"
              f"社区 {graph_stats.get('communities', 0)}")
    if summary["errors"]:
        print("  错误：" + "; ".join(summary["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
