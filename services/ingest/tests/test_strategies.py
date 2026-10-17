"""切片策略差异化测试：单元发现（语义断点 / 条款 / 代码 / 表格结构化 / 页级）。

背景：此前 6 个切片策略里 5 个等价于 `doc_structured`（各策略的专属参数没有被消费），
`chunk_eval.py` 会因此显式告警「本次 A/B 无效」。本测试保护这些差异**真实生效**：

  - 单元发现函数（strategies.py）的纯逻辑；
  - `chunk_document` 按 strategy / pipeline 分派到不同单元与父子结构；
  - 差异必须在「子块类型 / 数量 / 粒度」上可观测。
"""

import pytest

from kf_ingest.chunk import chunk_document
from kf_ingest.config import load_chunking_config, strategy_params, strategy_pipeline
from kf_ingest.models import BLOCK_PARAGRAPH, BLOCK_TABLE, BLOCK_TITLE, Block
from kf_ingest.strategies import (
    adjacent_similarities,
    clause_units,
    code_units,
    markdown_table_structured,
    quantile,
    semantic_units,
)
from kf_ingest.tokenize import TokenCounter


@pytest.fixture()
def counter():
    """强制走启发式计数，测试不依赖网络与模型权重。"""
    return TokenCounter("heuristic")


_UID_SEQ = [0]


def _block(text, block_type=BLOCK_PARAGRAPH, page=0, level=None):
    _UID_SEQ[0] += 1
    return Block(block_type=block_type, text=text, page=page, order=0,
                 bbox=[0, 0, 100, 100], level=level, uid=f"s{_UID_SEQ[0]}")


def _cfg():
    return load_chunking_config()


# ---------------------------------------------------------------- 语义断点
def test_quantile_matches_linear_interpolation():
    assert quantile([], 0.5) == 0.0
    assert quantile([3.0], 0.9) == 3.0
    assert quantile([0.0, 1.0], 0.5) == 0.5
    assert quantile([0.0, 1.0, 2.0, 3.0], 1.0) == 3.0


def test_adjacent_similarities_use_lexical_proxy_without_encoder():
    sentences = ["保修期为二十四个月。", "保修期是二十四个月。", "磁盘空间不足会中断安装。"]
    sims = adjacent_similarities(sentences)
    assert len(sims) == 2
    # 语义相近的前两句相似度应高于跨话题的第三句
    assert sims[0] > sims[1]


def test_adjacent_similarities_prefer_encoder_vectors():
    sentences = ["甲", "乙", "丙"]
    calls = []

    def encoder(texts):
        calls.append(list(texts))
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    sims = adjacent_similarities(sentences, encoder)
    assert calls and calls[0] == sentences
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(0.0)


def test_adjacent_similarities_fall_back_when_encoder_fails():
    def broken(texts):
        raise RuntimeError("TEI 不可用")

    sims = adjacent_similarities(["保修期二十四个月。", "保修期二十四个月。"], broken)
    assert len(sims) == 1 and sims[0] > 0.9


def test_semantic_units_split_at_topic_shift(counter):
    """跨话题处应断开：前 5 句谈保修、后 4 句谈安装条件。"""
    warranty = [
        "产品享有保修服务。", "保修期限为二十四个月。", "保修自购买之日起计算。",
        "保修期内更换零部件不收费。", "保修范围覆盖主机与配件。",
    ]
    install = [
        "安装前需要备份数据。", "安装要求磁盘剩余空间充足。",
        "安装过程会校验用户权限。", "安装完成后需要重启服务。",
    ]
    units = semantic_units(
        warranty + install, counter=counter, min_tokens=10, max_tokens=200, percentile=0.5,
    )
    assert len(units) >= 2, "跨话题处应产生断点"
    assert len(units) < len(warranty) + len(install), "不应退化为一句一块"


def test_semantic_units_does_not_break_when_percentile_is_one(counter):
    sentences = ["配置说明如下。", "配置说明如下。", "配置说明如下。"]
    units = semantic_units(
        sentences, counter=counter, min_tokens=1, max_tokens=999, percentile=1.0,
    )
    assert len(units) == 1  # 百分位取 1.0 → 无间隙超过阈值 → 不断点


# ---------------------------------------------------------------- 条款原子
def test_clause_units_split_chinese_clauses():
    text = (
        "第一条 本规定适用于所有内部系统。\n"
        "第二条 用户须妥善保管账号密码。\n"
        "第三条 违规操作将暂停账号权限。"
    )
    units = clause_units(text)
    assert len(units) == 3
    assert units[0].startswith("第一条")
    assert "第二条" not in units[0]


def test_clause_units_split_english_and_section_forms():
    text = "Article 1 Scope of application.\nArticle 2 User obligations.\nSection 2.1 Reporting."
    units = clause_units(text)
    assert len(units) == 3
    assert units[2].startswith("Section 2.1")


def test_clause_units_strip_numbering_when_requested():
    text = "第一条 本规定适用于所有内部系统。\n第二条 用户须妥善保管账号密码。"
    units = clause_units(text, keep_numbering=False)
    assert len(units) == 2
    assert units[0].startswith("本规定适用")


def test_clause_units_keep_preamble_and_never_guess():
    text = "颁布信息：2026 年发布。\n第一条 适用范围。\n第二条 责任划分。"
    units = clause_units(text)
    assert len(units) == 3
    assert units[0].startswith("颁布信息")
    # 识别不到序号时返回原文，不臆测切分
    plain = "这是一段没有任何条款序号的普通正文。"
    assert clause_units(plain) == [plain]


# ---------------------------------------------------------------- 代码语法单元
def test_code_units_split_functions_and_classes():
    code = (
        "import os\nimport sys\n\n"
        "def parse(path):\n    return open(path).read()\n\n"
        "class Loader:\n    def load(self):\n        return 1\n"
    )
    funcs = code_units(code, unit="function")
    # 首个锚点之前的内容作为「前言单元」保留，不丢弃导入语句
    assert funcs[0].lstrip().startswith("import os")
    assert any(u.lstrip().startswith("def parse") for u in funcs)

    classes = code_units(code, unit="class")
    assert classes[0].lstrip().startswith("import os")       # 前言单元
    assert any(u.lstrip().startswith("class Loader") for u in classes)
    # 除前言外，按类切分的单元不应再包含其它类别的定义锚点
    assert all("def parse" not in u for u in classes[1:])


def test_code_units_keep_indent_toggle():
    code = "def a():\n    x = 1\n    return x\n"
    indented = code_units(code, unit="function", keep_indent=True)
    stripped = code_units(code, unit="function", keep_indent=False)
    assert any(line.startswith("    ") for line in indented[0].split("\n"))
    assert not any(line.startswith(" ") for line in stripped[0].split("\n"))


# ---------------------------------------------------------------- 表格结构化
def test_markdown_table_structured_parses_columns_and_rows():
    markdown = ("| 型号 | 价格 | 库存 |\n| --- | --- | --- |\n"
                "| MX-100 | 1200 | 35 |\n| MX-200 | 1999 | 12 |")
    parsed = markdown_table_structured(markdown)
    assert parsed["columns"] == ["型号", "价格", "库存"]
    assert parsed["rows"][0] == ["MX-100", "1200", "35"]
    assert parsed["row_count"] == 2


def test_markdown_table_structured_keeps_negative_numbers():
    markdown = "| k | v |\n| --- | --- |\n| a | -5 |"
    parsed = markdown_table_structured(markdown)
    assert parsed["rows"] == [["a", "-5"]]


# ---------------------------------------------------------------- 策略分派
def test_pipeline_without_parent_child_produces_flat_chunks(counter):
    """配置即契约：pipeline 含 parent_child 才产出父块。"""
    cfg = _cfg()
    blocks = [
        _block("第一条 适用范围。"),
        _block("第二条 责任划分。"),
        _block("第三条 附则。"),
    ]
    flat = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params=strategy_params(cfg, "clause_atomic"), counter=counter,
        strategy="clause_atomic", pipeline=strategy_pipeline(cfg, "clause_atomic"),
    )
    assert flat.parents == [], "clause_atomic 的 pipeline 无 parent_child → 不产出父块"
    assert [c.chunk_type for c in flat.children] == ["clause", "clause", "clause"]
    assert all(c.parent_chunk_id is None for c in flat.children)

    structured = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params=strategy_params(cfg, "doc_structured"), counter=counter,
        strategy="doc_structured", pipeline=strategy_pipeline(cfg, "doc_structured"),
    )
    assert structured.parents, "doc_structured 的 pipeline 含 parent_child → 应产出父块"


def test_different_strategies_produce_different_output(counter):
    """差异必须可观测，否则 chunk_eval 的 A/B 依旧无效。"""
    cfg = _cfg()
    code_text = "\n".join(f"def handler_{i}(req):\n    return req + {i}" for i in range(6))
    prose_text = "本产品是一款面向企业的知识库底座，支持多语言检索。" * 12
    blocks = [
        _block("配置说明", BLOCK_TITLE, level=1),
        _block(prose_text),
        _block(code_text),
    ]
    outputs = {}
    for strategy in ("doc_structured", "code_aware", "semantic"):
        result = chunk_document(
            blocks, doc_id="d1", version=1, lang="zh",
            params=strategy_params(cfg, strategy), counter=counter,
            strategy=strategy, pipeline=strategy_pipeline(cfg, strategy),
        )
        outputs[strategy] = [(c.chunk_type, c.content) for c in result.children]

    assert outputs["doc_structured"] != outputs["code_aware"]
    assert outputs["doc_structured"] != outputs["semantic"]
    assert any(kind == "code" for kind, _ in outputs["code_aware"])


def test_visual_page_emits_one_unit_per_page(counter):
    cfg = _cfg()
    blocks = [
        _block("第一页正文。", page=0),
        _block("第一页续文。", page=0),
        _block("第二页正文。", page=1),
        _block("第三页正文。", page=2),
    ]
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params=strategy_params(cfg, "visual_page"), counter=counter,
        strategy="visual_page", pipeline=strategy_pipeline(cfg, "visual_page"),
    )
    assert len(result.children) == 3
    assert [c.page for c in result.children] == [0, 1, 2]
    assert all(c.meta["unit"] == "page" for c in result.children)
    # 页级块不臆造字符偏移：provenance 缺失总好过给错偏移
    assert all(c.provenance is None for c in result.children)
    assert any("ColPali" in note for note in result.stats["notes"])


def test_table_centric_emits_structured_fields(counter):
    cfg = _cfg()
    rows = ["| 型号 | 价格 | 库存 |", "| --- | --- | --- |"]
    rows += [f"| MX-{i} | {i}000 | {i} |" for i in range(12)]
    blocks = [_block("规格表", BLOCK_TITLE, level=1), _block("\n".join(rows), BLOCK_TABLE)]
    params = strategy_params(cfg, "table_centric")
    params["group_rows"] = 5
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh", params=params, counter=counter,
        strategy="table_centric", pipeline=strategy_pipeline(cfg, "table_centric"),
    )
    tables = [c for c in result.children if c.chunk_type == "table"]
    assert len(tables) >= 2
    assert all(t.content.startswith("| 型号 | 价格 | 库存 |") for t in tables)
    assert tables[0].structured["columns"] == ["型号", "价格", "库存"]
    assert tables[0].structured["row_count"] == 5


def test_code_aware_notes_when_no_anchor_found(counter):
    cfg = _cfg()
    blocks = [_block("这是一段没有任何代码定义锚点的普通文本，仅作为兜底验证。")]
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params=strategy_params(cfg, "code_aware"), counter=counter,
        strategy="code_aware", pipeline=strategy_pipeline(cfg, "code_aware"),
    )
    assert len(result.children) == 1
    assert any("未识别到单元锚点" in note for note in result.stats["notes"])


def test_oversized_unit_is_split_at_sentence_boundary(counter):
    """最后一个锚点会一直延伸到组尾：超长单元必须按句子边界拆开，而不是吞掉全部内容。"""
    cfg = _cfg()
    long_clause = "第一条 " + "".join(
        f"本款第{i}项规定了具体的操作要求与责任划分标准。 " for i in range(30)
    )
    blocks = [_block(long_clause), _block("第二条 附则，自发布之日起施行。")]
    params = strategy_params(cfg, "clause_atomic")
    params["max_child_tokens"] = 60
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh", params=params, counter=counter,
        strategy="clause_atomic", pipeline=strategy_pipeline(cfg, "clause_atomic"),
    )
    assert len(result.children) > 2, "超长单元应被拆成多个子块"
    assert result.stats["max_child_tokens"] <= 60
    assert any("按句子边界拆分" in note for note in result.stats["notes"])
    # 末条仍然独立成块，不应被前一条吞掉
    assert any("第二条" in c.content for c in result.children)


def test_no_anchor_group_falls_back_to_structural_recursion(counter):
    """识别不到锚点时应回退结构递归，而不是把整组压成一个巨大块。"""
    cfg = _cfg()
    prose = "".join(f"第{i}段说明文中不包含任何条款序号的普通叙述内容。 " for i in range(40))
    blocks = [_block(prose)]
    params = strategy_params(cfg, "clause_atomic")
    params["child_tokens"] = 60
    params["max_child_tokens"] = 60
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh", params=params, counter=counter,
        strategy="clause_atomic", pipeline=strategy_pipeline(cfg, "clause_atomic"),
    )
    assert len(result.children) > 1, "无锚点时必须回退结构递归，而不是整组一块"
    assert any("回退结构递归切分" in note for note in result.stats["notes"])
    assert result.stats["max_child_tokens"] <= 180  # 远小于整段总量


def test_default_call_still_produces_parent_child(counter):
    """向后兼容：不传 strategy / pipeline 时行为与旧实现一致。"""
    blocks = [
        _block("产品概述", BLOCK_TITLE, level=1),
        _block("本产品是一款面向企业的知识库底座。" * 20),
    ]
    result = chunk_document(
        blocks, doc_id="d1", version=1, lang="zh",
        params={"child_tokens": 100, "parent_tokens": 400, "overlap_ratio": 0.12},
        counter=counter,
    )
    assert result.parents and result.children
    assert all(c.parent_chunk_id for c in result.children)
    assert result.stats["strategy"] == "doc_structured"
    assert result.stats["with_parents"] is True
