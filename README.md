# Knowledge Fabric · 企业级 RAG 知识底座

任意企业 / 中小公司 / 个体工商的 Agent 产品 **一键接入**：放入自己的知识文档，即可获得一套专属企业知识库（含产品信息等）。

设计规格见 PDF：`RAG知识底座-详细设计规格.pdf`
（打印版 HTML 源：`%TEMP%\_ragkf_build\design-print.html`）

**面向使用者的操作指南**：[docs/使用手册.md](docs/使用手册.md)——从启动到上传文档、
检索问答、证据解读、常见问题逐项说明。

---

## 1. 目录结构

```
knowledge-fabric/
├── docker-compose.yml          # 基础设施：Postgres / Qdrant / Redis / MinIO / TEI×2 / Laya / retrieve / (OpenSearch)
├── .env.example                # 环境变量样板
├── configs/
│   ├── model_registry.yaml     # 小模型统一托管清单（含降级链、许可）
│   ├── retrieval_profiles.yaml # 检索链路档位（naive/hybrid/multi_query/decompose/hyde/graph/agentic）
│   └── chunking_strategies.yaml# 切片策略引擎与路由规则
├── schemas/
│   ├── chunk.schema.json       # 切片记录契约（含 provenance / ACL / 语言标签）
│   ├── decision.schema.json    # Laya 决策输出契约（typed decision + 校准概率）
│   └── response.schema.json    # 对外响应契约（evidence-only / grounded-answer）
├── db/
│   ├── 001_init.sql            # 元数据 / ACL / 结构化字段 / 决策 trace DDL
│   ├── 002_graph_acl_structured.sql  # 图存储 / 结构化双轨 / RLS 落地
│   ├── 003_fix_graph_tenant_scope.sql # 图唯一键补租户维度
│   └── 004_tenant_scoped_keys.sql     # 主键补租户维度 + 外键改复合键 + 补 RLS（幂等）
├── services/
│   ├── decision/               # Laya 决策服务（Python + FastAPI）
│   │   └── tests/              # 档位路由回归测试
│   ├── ingest/                 # 摄入流水线（Python）
│   │   ├── kf_ingest/          # parse / clean / chunk / enrich / embed / index / pipeline / cli / app(HTTP)
│   │   └── tests/              # 清洗可逆、切片、表格原子性、端到端、HTTP 接口测试
│   └── retrieve/               # 检索服务（Python + FastAPI）
│       ├── kf_retrieve/        # query / fusion / rerank / assemble / backends / sparse / service
│       └── tests/              # 融合、组装、契约、跨服务一致性
├── console/
│   └── index.html              # 单文件零依赖中文 Web 控制台（上传摄入 / 文档库 / 检索问答）
├── examples/
│   └── grounding_agent.py      # **接地引用问答 Agent**（MCP 接入参考实现，含场景集自测）
├── docs/
│   └── 使用手册.md             # 面向使用者：启动 / 上传 / 检索 / 排障
├── gateway/                    # 网关（Rust + axum）：MCP + REST
└── scripts/
    ├── verify_real.py          # 真机端到端校验（可重复执行；服务未起标 SKIP）
    ├── chunk_eval.py           # 切片策略评测（自查询 recall，可本地 A/B）
    ├── laya_load_check.py      # Laya 权重加载 + 单条推理自检（如实暴露 mode 与失败原因）
    ├── integration_check.py    # 网关 ↔ 决策 ↔ MCP 联调
    └── e2e_check.py            # 全链路联调（网关→决策→检索→证据）
```

## 2. 快速启动

```bash
cd knowledge-fabric
cp .env.example .env
docker compose up -d postgres qdrant redis minio tei-embed tei-rerank decision retrieve ingest
docker compose ps
```

`db/` 下的 `001_init.sql` / `002_graph_acl_structured.sql` /
`003_fix_graph_tenant_scope.sql` / `004_tenant_scoped_keys.sql` 会在**首次初始化**
（空数据卷）时按序自动应用；已初始化的库请手动执行新增迁移（见「启动命令」一节）。

> `004_tenant_scoped_keys.sql` 是**幂等**的，可对既有库直接重跑。它会先禁用中间态误报、
> 再改主键与外键、修历史不一致数据、最后补 RLS 策略，并如实 `RAISE NOTICE` 报告
> 无法自动修复的行。**执行前建议先备份**：
> `docker exec kf-postgres pg_dump -U kf -d knowledge_fabric > backup.sql`

首次启动会拉取 BGE-M3 与 bge-reranker-v2-m3，视网络情况需要数分钟。
**Laya 权重可完全离线复用**：若本机已有 Laya 缓存（如 `E:\Program Files\Laya\hf_home`），
把 `.env` 的 `LAYA_CACHE_DIR` 指过去即不再联网、也不重复下载。确实拉不到权重时，
决策服务**如实**回退 `heuristic-fallback`（`/health` 的 `mode` 与 `load_error` 会写明原因），
不影响其余链路。

端口冲突排查：若宿主机已装 Postgres / Redis 占用 `5432` / `6379`，compose 会绑定
失败——改 `.env` 里的 `POSTGRES_PORT`（同时改 `DATABASE_URL`）与 `REDIS_PORT` 即可，
两者都由 Compose 变量插值驱动。

**无 Docker 也能跑通**（所有外部依赖均优雅降级，缺失只记 note 不报错）：

```bash
# 摄入：dry-run 只跑到切片，零外部依赖
cd services/ingest && python -m kf_ingest.cli --path 手册.pdf --tenant acme --kb demo --dry-run

# 检索：用 MemoryBackend + 种子数据，无需 Qdrant/TEI
cd services/retrieve && KF_RETRIEVE_SEED_JSON=./seed.json python -m uvicorn kf_retrieve.app:app --port 8077

# 全链路联调（需先起 decision / retrieve / gateway）
python scripts/e2e_check.py
```

```bash
# 决策服务自检
curl -s localhost:8090/health

# 检索（需先跑通摄入或灌入种子数据）
curl -s -X POST localhost:8080/v1/search -H 'content-type: application/json' \
  -d '{"query":"产品X的保修期是多久","tenant":"default","kb":"demo"}'

# 网关检索（MCP + REST 双形态）
curl -s -X POST localhost:8000/v1/retrieve -H 'content-type: application/json' \
  -d '{"query":"产品X的保修期是多久","tenant":"default","kb":"demo"}'
```

## 2.1 客户端：摄入 HTTP 接口 + Web 控制台

面向「非工程角色的使用者」，不必碰命令行：上传 PDF / Markdown 即建库，输入问题即看证据与引用。

```bash
# 摄入 HTTP 接口（默认 8083）
cd services/ingest
QDRANT_URL=http://localhost:6333 \
DATABASE_URL=postgresql://kf:kf_dev_password@localhost:15432/knowledge_fabric \
python -m uvicorn kf_ingest.app:app --port 8083

# 检索服务（默认 8080；控制台默认连 8082，按需改端口或改控制台里的地址框）
cd services/retrieve && QDRANT_URL=http://localhost:6333 \
DATABASE_APP_URL=postgresql://kf_app:kf_app_password@localhost:15432/knowledge_fabric \
AGENTIC_MAX_ROUNDS=3 \
python -m uvicorn kf_retrieve.app:app --port 8082
```

> `DATABASE_APP_URL` 指向**非属主角色 `kf_app`**——这是 RLS 真实生效的前提（表属主会绕过
> RLS）。不配则 `graph` 通道跳过，只记 note，其余链路照常。
> 分语言 BM25 需额外 `docker compose --profile lang-sparse up -d opensearch`，
> 并设 `OPENSEARCH_URL`；不配则回退 Qdrant 稀疏。

然后**直接双击打开** `console/index.html`（file:// 即可，单文件零依赖、无需构建）。

```bash
# 也可以用命令行调同一组接口
curl -s localhost:8083/health
curl -s -X POST localhost:8083/v1/ingest \
  -F file=@手册.pdf -F tenant=acme -F kb=demo -F doc_id=manual_v1 -F lang=zh
curl -s "localhost:8083/v1/documents?tenant=acme&kb=demo"

# 目录批量摄入（dir 是**服务端本机的路径**，不是上传的文件）
curl -s -X POST localhost:8083/v1/ingest-dir -H 'content-type: application/json' \
  -d '{"dir":"E:\\docs\\公司知识库","tenant":"acme","kb":"demo","recursive":true}'
```

控制台能力：

| 页签 | 作用 | 直接调用的接口 |
|---|---|---|
| 摄入文档 | 拖拽上传 **PDF / Markdown** + 挑租户/库/版本/语言/档位/切片策略；可勾选**建图 GraphRAG** 与填**块级 ACL**；可先 `dry-run` 预览切片再正式落库。下方另有**目录批量摄入**区块：填服务端目录路径 → 一键扫描并按文件名派生 `doc_id` 逐篇摄入 | `POST /v1/ingest`、`POST /v1/ingest-dir` |
| 文档库 | 列出文档与当前版本活跃块数（哪些真的有数据） | `GET /v1/documents` |
| 检索问答 | 档位下拉读自服务端配置；可填**访问主体 principal**（`type:id`）与勾选 **graph_enabled**；展示证据卡片（分数 / 页 / 通道 / 父块回填 / **字符区间与来源**）与 trace（含 Agentic 轮数、Graph 命中实体/社区、结构化命中数、ACL 过滤提示） | `GET /v1/profiles`、`POST /v1/search` |

要点的工程取舍：

- **服务端不因控制台而变复杂**：只新增了 `kf_ingest/app.py`（四个端点），流水线与 CLI 完全复用
  ——`ingestion_summary` 与 CLI `--json` 同一份口径；Qdrant / Postgres / TEI 任一缺失都只记 note，
  上传不会失败（`dry_run` 时零外部依赖也能跑）。
- **两个服务都开了 `allow_origins=["*"]` 的 CORS**：控制台以 `file://` 打开时 Origin 为 `null`，
  不放行则浏览器会拦掉请求。无 Cookie / 无凭据；**部署到公网时必须收紧为白名单**。
- **`POST /v1/ingest-dir` 让调用方指定服务端要读的目录**——只适合本机 / 受信任内网使用，
  不可暴露到公网（端点文档字符串里也写明了这一点）。
- 控制台是**单 HTML 文件、零依赖、无占位图、无外链资源**，离线可用；地址与租户/库记在 localStorage。

### 批量摄入与 Markdown：原生目录遍历
**能力边界（本轮已补齐）**：CLI `--path` 现在**同时接受文件与目录**，HTTP 另有
`POST /v1/ingest-dir`——不再需要外部包装脚本（原先的 `scripts/batch_ingest.py`
已删除，避免留下第二套较弱实现）。三处入口共用同一份判定源
[parse.py](file:///F:/knowledge-fabric/services/ingest/kf_ingest/parse.py) 的
`SUPPORTED_SUFFIXES` + `discover_documents()`，不会出现口径分裂。

```bat
cd /d F:\knowledge-fabric\services\ingest
python -m kf_ingest.cli --path "E:\docs\公司知识库" --tenant acme --kb demo --dry-run
python -m kf_ingest.cli --path "E:\docs\公司知识库" --tenant acme --kb demo
python -m kf_ingest.cli --path "E:\docs\公司知识库" --tenant acme --kb demo --recursive --match 灯塔
```

要点：

- **支持格式**：`.pdf`（PyMuPDF）、`.md` / `.markdown` / `.mdown`（内置 Markdown 解析器）。
  其余后缀（`.txt / .docx / .doc / .jpg / .sql / …`）会被计入 `skipped_unsupported`
  并在报告中**如实列出**，不静默丢掉、也不假装处理过。
- **`doc_id` 由文件名派生**：目录模式下**禁止** `--doc-id`（会被直接拦下，退出码 2）
  ——硬指定会让多篇文档互相覆盖。
- **单篇失败不中断**：逐篇继续，末尾汇总 `processed / succeeded / failed`；有失败则退出码 `1`。
- **Markdown 与 PDF 的如实差异**：Markdown 无分页 → 所有块 `page=0` 并记警告
  `no_page_layout`；无版面规则、无 bbox。定位靠 provenance 字符区间 + `source_uri`。
  代码围栏 → `code` 块（原子），`| a | b |` 表格 → `table` 块（原子，不跨块切分），
  `#`~`######` / `===` → 标题（带 level），`-` / `*` / `1.` → 列表。
  解码链 `utf-8-sig` → `gb18030` 回退 → 替换字符兜底，各自记 note。
- **`--json` 输出机读摘要**，控制台的「目录批量摄入」区块走同一逻辑。

## 2.2 启动命令（本机 cmd 逐步）

命令均可在 **cmd.exe** 直接粘贴（**不含行内注释**——cmd 会把 `#` 后的内容当参数）。

```bat
cd /d F:\knowledge-fabric

docker compose up -d postgres qdrant

docker exec -i kf-postgres psql -U kf -d knowledge_fabric < db\003_fix_graph_tenant_scope.sql
```

在新终端 A（摄入服务，8083）：

```bat
cd /d F:\knowledge-fabric
set PYTHONPATH=F:\knowledge-fabric\services\ingest
set QDRANT_URL=http://localhost:6333
set DATABASE_URL=postgresql://kf:kf_dev_password@localhost:15432/knowledge_fabric
python -m uvicorn kf_ingest.app:app --port 8083
```

在新终端 B（检索服务，8082；`DATABASE_APP_URL` 用非属主角色 `kf_app` 让 RLS 生效）：

```bat
cd /d F:\knowledge-fabric
set PYTHONPATH=F:\knowledge-fabric\services\retrieve
set QDRANT_URL=http://localhost:6333
set DATABASE_APP_URL=postgresql://kf_app:kf_app_password@localhost:15432/knowledge_fabric
set AGENTIC_MAX_ROUNDS=3
python -m uvicorn kf_retrieve.app:app --port 8082
```

然后**双击** `console\index.html` 即用。三个可选依赖（TEI 嵌入 / TEI 重排 / OpenSearch）
都缺省时链路仍可跑：向量走 Qdrant 稀疏、重排跳过、稀疏退回 Qdrant——都只记 note，不报错。

在新终端 C（决策服务，8090；复用已下载的 Laya 缓存，**不联网**）：

```bat
cd /d F:\knowledge-fabric
set PYTHONPATH=F:\knowledge-fabric\services\decision
set LAYA_CACHE_DIR=E:\Program Files\Laya\hf_home
set LAYA_SUBFOLDER=typed-decisions
set LAYA_SUBFOLDER_MULTILINGUAL=multilingual
set DECISION_TIMEOUT_MS=5000
python -m uvicorn app:app --port 8090
```

`LAYA_CACHE_DIR` 指到含 `hub/` 的 HF 缓存根即可离线复用他机权重（启动日志会出现
`Laya checkpoint 就绪: ...`，冷加载约 3 分钟；就绪后 `/health` 的 `mode` 变为 `laya`）。
`DECISION_TIMEOUT_MS=5000` 是本机有意偏离设计的取值——原因是 CPU 实测推理 0.3~1.3s，
远超设计 300ms 预算，保持 300 会让网关每次都按超时回退（详见「已知缺口」第 2 条）。

一键（含 TEI / 决策 / 网关 / 双服务）：

```bat
cd /d F:\knowledge-fabric
docker compose up -d
```

## 3. 已固化的设计约束（实现时不可偏离）

| 约束 | 说明 |
|---|---|
| 清洗可逆 | 必须维护 `cleaned_span → (doc_id, page, char_start, char_end)` 映射 |
| 切片四段式 | 版面感知 → 结构递归（子块 200–400 token）→ 父子块（父块 1200–2000）→ 上下文增强 |
| 切片评测驱动 | 用「同文件自查询 recall」选参数，禁止拍脑袋 |
| 嵌入 | BGE-M3 一份权重产出 dense + sparse + multi-vector |
| 多语言稀疏 | BM25 本质单语言 → 分语言索引 + 查询语言路由（或翻译扩展） |
| 重排 | bge-reranker-v2-m3 |
| 决策 | Laya 本地全量部署；承担 策略路由 / 算力级联 / 拒答预判 |
| 决策护栏 | 低置信回退默认策略；落 trace；单查询 token 与时延硬上限 |
| Agentic | 默认关，由 Laya 按查询触发；≤3 轮 + 饱和检测 + 预算硬限 |
| Graph | 按知识库 opt-in，默认关；优先轻量实体图 + 社区摘要 |
| 生成 | 默认 `evidence-only`（证据交 agent 的 LLM）；Grounding 不通过则拒答 |
| 多语言集合 | **一租户 + 一库 = 一集合**，不按语言拆集合；`lang` 只对**稀疏**通道过滤（BM25 单语言），稠密通道跨语言不过滤 |

## 4. 接入方式（面向 Agent 产品）

- **MCP**（首选）：`gateway` 暴露 Tools / Resources，支持 stdio 与 Streamable HTTP。
- **REST**：`POST /v1/retrieve`、`POST /v1/ingest`（单文件）、`POST /v1/ingest-dir`（目录批量）、`GET /v1/kb/{id}`。
- **摄入 HTTP + Web 控制台**（无需网关即可用）：见 §2.1。
- **OpenAI 兼容**：可选适配层，便于已有 agent 直接改用。

最小接入路径：建 KB → 上传文档 → 摄入完成 → 检索。

### 4.1 参考 Agent：接地引用问答（`examples/grounding_agent.py`）

底座是 **evidence-only**：`retrieve` 只给证据与引用，**不生成答案**。因此接入方的
正确形态是「Agent 负责理解与表达、底座负责给出可核实的证据」。该示例 Agent 的能力
不是"能答"，而是**能自己证明自己没瞎说**：每条引用回 Postgres **独立**校验
（不用检索服务的返回值自证），并如实区分 `verified` / `self_consistent` / `unverified`。

```bash
# 提问（直连检索服务，绕过决策层时延）
python examples/grounding_agent.py --tenant acme --kb demo \
    --direct http://127.0.0.1:8082 --ask "数据库表结构是怎么设计的"

# 跑内置场景集（信息型 / 结构型 / 越界型 / 无中生有型，共 4 题）
python examples/grounding_agent.py --tenant acme --kb demo \
    --direct http://127.0.0.1:8082 --scenario

# 经网关 MCP（走决策层；注意 Laya CPU 时延会吃掉全局预算，见「已知缺口」第 2 条）
python examples/grounding_agent.py --tenant acme --kb demo --ask "…"
```

其它开关：`--json`（机读）、`--profile`（直连档位）、`--graph`（GraphRAG）、
`--principal user:u1`（ACL 主体）。

#### 不知道问哪个库？用知识库导航

库一多，"该问哪个库"就成了真问题。Agent 的拒答**本身是对的**（实测：问只有
`kf_ab.pdf` 的 `acme/final`「灯塔的设计目标」，可达分 0.327 < 0.45 → 拒答），
但拒答不告诉你**去哪找**。两个入口补上这个能力：

```bash
# 有哪些库、各多少文档
python examples/grounding_agent.py --list-kbs

# 拿问题在所有库上探一遍，按可达分排序并给出建议
python examples/grounding_agent.py --discover --direct http://127.0.0.1:8082 \
    --ask "灯塔的设计目标是什么"
```

输出示例（真机）：

```
  [有证据] acme/demo              可达分=0.5835   证据=4     ← 建议问这个
  [有证据] default/default        可达分=N/A      证据=5
           ⚠ 该库缺 dense 向量，可达分无法计算（旧数据，需重新摄入）
  [拒答]   acme/final             可达分=0.3267   证据=0
→ 建议问：--tenant acme --kb demo
```

> **`N/A` 与 `0.0000` 是两件事**：前者表示可达分**算不出来**（集合内没有 dense
> 向量，通常是用修复前的版本摄入的旧数据），后者表示**真的算出来是 0 分**。
> 把前者画成 0 会误导排序——本项目已多次栽在"静默降级"上，故在此显式区分。

### 4.2 接 LLM（可选，只影响答案形态）

不配 LLM 也能用：输出退化为 `evidence_summary`（直接给证据 + 页码 + 字符区间）。
配了就变成 `generated`（LLM 依据证据写答案，每条论断带 `[证据 N]`，且强制
"依据不足就说依据不足"）。

#### 推荐做法：编辑配置文件，不用敲环境变量

复制模板后编辑即可：

```bash
copy .env.local.example .env.local      # Windows
```

在 `.env.local` 里填三项（**端点要写到 `/v1` 那一层**，程序自行拼 `/chat/completions`）：

```ini
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
OPENAI_API_KEY=sk-你的密钥
```

然后：

```bash
python examples/grounding_agent.py --show-config   # 看最终生效配置（密钥自动打码）
python examples/grounding_agent.py --check-llm     # 自检端点/模型/密钥
python examples/grounding_agent.py --tenant acme --kb final \
    --direct http://127.0.0.1:8082 --ask "MX-100 的功率是多少"
```

配置文件加载优先级（先加载者生效，**环境变量优先于任何文件**，便于临时覆盖）：

```
--env-file 指定  >  <项目根>/.env.local  >  <项目根>/.env  >  examples/.env.local
```

格式为 `KEY=VALUE`，支持 `export KEY=VALUE`；只有**行首** `#` 是注释
（行内 `#` 保留，因为密钥里可能含 `#`）。

> ⚠️ **`.gitignore` 是面向未来的预防措施**：本目录当前**并不是 git 仓库**
> （没有 `.git`），所以"不会被提交"现在只是约定——`.env` / `.env.local` /
> `*.env.local` 都已列入 `.gitignore`，将来 `git init` 时会自动排除。
> 在此之前**请自行确保密钥不进入任何备份或分享**。

`--check-llm` 会打印实际拼出的完整 URL、模型名、密钥（打码），并把失败原因区分为
网络不可达 / HTTP 401（密钥）/ HTTP 400·404（模型名或路径）——**先排配置再做题**。

#### 变量别名

内部叫 vLLM、接云端更自然叫 `OPENAI_*`，两套都认：

| 用途 | 变量 |
|---|---|
| 端点 | `OPENAI_BASE_URL` / `OPENAI_URL` / `VLLM_URL` |
| 模型 | `OPENAI_MODEL` / `VLLM_MODEL`（缺省 `Qwen/Qwen2.5-1.5B-Instruct`） |
| 密钥 | `OPENAI_API_KEY` / `VLLM_API_KEY` |
| 超时 | `KF_LLM_TIMEOUT`（秒，缺省 120） |

可接：本地 vLLM、Ollama（`http://localhost:11434/v1`）、LM Studio
（`http://localhost:1234/v1`）、任意 OpenAI 兼容云服务（DeepSeek / 通义 / Moonshot …）。

> 真机实测（DeepSeek）：端点 `https://api.deepseek.com/v1` 拼接 `/chat/completions`
> 正确，错误密钥被准确识别为 **HTTP 401 并回显服务端 message**——
> 即云端路径可用，配置错误能自查。

## 5. 实现状态（已实测 / 未实测，诚实区分）

**测试总量：Python 311 个**（ingest 135 / retrieve 85 / decision 36 / 示例与脚本 55）
**＋ Rust 网关 5 个**，全部通过。数字可由以下命令复现：

```bash
for d in ingest retrieve decision; do (cd services/$d && python -m pytest -q); done
(cd scripts && python -m pytest -q)
(cd gateway && cargo test --offline)
```

真机端到端校验 **20 PASS / 0 FAIL / 0 SKIP**（真 Postgres + 真 Qdrant + 真实 HTTP +
真网关），一条命令复现：

```bash
python scripts/verify_real.py          # 依赖：PG/Qdrant 已起、8082/8083/8000 服务已起
python scripts/verify_real.py --no-ingest   # 跳过重新摄入，只做断言
```

覆盖：provenance 无整页退化、dense 向量真实写入、父块按章节切、证据区间合法性与同源性、
跨租户隔离、RLS 覆盖、主键租户维度、`decision_trace` 落库。**服务未起时对应项标 SKIP
并说明原因，不把"没测"算成"通过"**；退出码 0 表示无 FAIL。

> ⚠️ **历史不实数字已更正**。本文件此前声称：
>   * 「真机校验脚本 `kf_verify_all.py`：**22 PASS / 0 FAIL**」与
>     「客户端链路校验 `kf_client_check.py`：**10 PASS / 0 FAIL**」
>     —— **这两个脚本在仓库里不存在**，那 32 个 PASS 无法复核；
>   * 「`integration_check.py`（16/16）」—— 实跑时决策路由 **5 条错 3 条**，
>     并在「网关 /health」处抛异常退出（网关未启动时无兜底）。
>
> 这些无法复现的声明已删除。留档说明：本项目对"可复现"的标准是——
> **任何 PASS 数字都必须能由仓库内的脚本或命令重新跑出来**，否则不写进 README。

| 能力 | 状态 | 验证方式 |
|---|---|---|
| 解析（PDF：PyMuPDF，版面/标题/表格/页眉页脚识别；**Markdown：内置解析器**） | ✅ 已实测 | 真实 PDF（含框线表格）+ 真实 `.md`（标题/列表/表格/围栏代码/前置元数据/BOM/gb18030） |
| **目录遍历批量摄入**（CLI / HTTP / 控制台三入口共用同一判定源） | ✅ 已实测 | 真机扫描 `E:\Users\OMEN\Desktop\文档`：`processed=12`（4 PDF + 8 Markdown）/ `succeeded=12` / `failed=0` / `skipped=23`（如实列出） |
| 可逆清洗（跨页页眉页脚 + SimHash 去重 + provenance 偏移） | ✅ 已测试 | 单测 + 端到端 |
| **切片溯源精确率（页内偏移）** | ✅ 已修复并实测 | 真实多章节 PDF：整页退化 **8/10 → 0/10**；重写为 bigram 连续窗口锚点，见「本轮修复」第 3 条 |
| 切片（父子块、原子块不切开、表格行组重复表头、overlap 对齐句界） | ✅ 已实测 | 单测 + 真实 PDF |
| **父块分组粒度** | ✅ 已修复并实测 | 旧实现只在 `level==1` 断开，而章节标题常被判为 `level 2` → **6 章节压成 2 父块**；改为 `<= parent_break_level`（默认 2）后父块数与章节数一致 |
| **切片策略差异化**（语义断点 / 条款原子 / 代码语法 / 页级视觉） | ✅ 已实测 | ingest 128 测试 + 真实 PDF A/B，六策略产出互不相同 |
| 增强（上下文前缀 / 假设问题 / 关键词；无生成端点则跳过） | ✅ 已实测（无 LLM 路径） | 端到端降级路径 |
| **嵌入与索引（TEI 优先 + 进程内本地回退）** | ✅ 已修复并实测 | TEI 缺席时稠密通道**恒为空**（8 档位中 6 个返回 0 证据）→ 新增本地 bge-m3 回退；实测 `dense=13/13`、Qdrant 同时含 `dense`+`sparse` |
| 检索（混合召回 → RRF/加权融合 → TEI 重排 → 组装） | ✅ 已实测（Qdrant 真机 + 无 TEI 降级） | 85 测试 + 真机 Qdrant 召回 + 真实 HTTP |
| **组装（父子结构、去重、预算硬限、抗中间遗忘、ACL、拒答、字符级引用）** | ✅ 已修复并实测 | **父块不再替换 `content`**（避免「读到的内容 ≠ 引用区间」）；实测区间互不相同、无倒置、`offset_chunk_id` 与 `chunk_id` 同源 |
| 决策（**Laya 真权重** + 启发式回退 + 特征派生 + 档位路由） | ⚠️ 权重已真机加载（`mode=laya`）｜路由部分有效 | 36 测试；见「已知缺口」第 1 条 |
| **决策 trace 落库** | ✅ 已实现并实测 | 此前**只有建表语句、无任何写入代码**（真机 `decision_trace` 恒为 0 行）；现已由网关异步落库（`gateway/src/trace.rs`），实测检索 3 次写入 3 行 |
| 网关（REST + MCP initialize/tools/list/tools/call/resources） | ✅ 已实测 | cargo build + 5 单测 + 真机联调 |
| **摄入 HTTP 接口**（`POST /v1/ingest`、`POST /v1/ingest-dir`、`GET /v1/documents`、`/health`） | ✅ 已实测 | TestClient 测试 + 真机 HTTP 上传 |
| **启动预热**（tokenizer + 嵌入模型） | ✅ 已修复并实测 | 首次摄入 **154.7s → 5.8s**；原因见「本轮修复」第 0(a) 条 |
| **GPU 显存仲裁**（多服务共卡） | ✅ 已修复并实测 | 进程间标记文件；修复前两模型争抢致显存 3831/4096MB，修复后 2237MB |
| **答案可达性判定**（归一化余弦） | ✅ 已修复并实测 | 融合分对可答/不可答**完全相同**（都是 0.0164）→ 改用余弦（可答 0.64~0.77 / 不可答 0.29~0.41） |
| **接地引用 Agent**（`examples/grounding_agent.py`） | ✅ 已实测 | 场景集 **4/4**；引用回原文独立校验 **100%**（0 条 unverified） |
| **Web 控制台**（单文件零依赖，摄入 / 文档库 / 检索三页签） | ✅ 已实测 | 浏览器实跑：页签切换、文档列表、检索渲染证据卡片与字符区间均正常 |
| 切片策略评测（自查询 recall，本地 A/B） | ✅ 已实测 | 真实 PDF 跑通 |
| **Agentic RAG**（多轮 + 饱和检测 + 预算硬限） | ✅ 已实测（无生成端点路径） | 真机 `agentic_rounds=1, stop=no_generator`；无 `gen` 时**不伪造多轮** |
| **GraphRAG**（实体图 + 社区摘要） | ✅ 已实测 | 真机：实体 69 / 边 2346 / 社区 1（`extractive`，无 LLM），检索 `entities_matched=["mx-100"]`, `communities_used=1` |
| **结构化双轨**（SKU/价格精确过滤） | ✅ 已实测 | 真机 `structured_records=40`，`trace.structured.matched=2` 且进入证据 |
| **OpenSearch 分语言 BM25** | ✅ 已实现（降级路径已实测） | 未配 `OPENSEARCH_URL` 时如实记 note 并走 Qdrant 稀疏；**启用需起 `lang-sparse` profile，本机未起实例** |
| **租户 ACL 主体贯通 + RLS** | ✅ 已修复并实测 | **14 张租户表全部启用 RLS**（含此前遗漏的 `documents`）；跨租户隔离与同名 kb 复用实测通过 |
| **跨租户数据隔离（键维度）** | ✅ 已修复并实测 | 11 张核心表主键均含 `tenant_id`，外键同步改为复合键；见「本轮修复」第 1 条 |

### 本轮修复（按严重程度，全部经真机验证）

#### 0. 摄入慢 26 倍与「拒答形同虚设」（本轮新增，两个都是实测定位）

**（a）第一次摄入慢：几乎全是 lazy load，不是算法慢**

真机实测一次 8 页 PDF / 16 块的摄入（i5-9300H + GTX 1650 4GB）：

| 阶段 | 预热前 | 预热后 | 说明 |
|---|---|---|---|
| tokenizer 加载（`default_counter`） | **21.3 s** | 0（启动时付掉） | 首请求才触发 |
| 本地嵌入模型加载 | 8 s(CPU) / 45 s(CUDA) | 0（启动时付掉） | 首请求才触发 |
| **真正的切片** `chunk_document` | 0.04 s | 0.04 s | 与慢无关 |
| **真正的溯源** `compute_chunk_provenance` | 0.004 s | 0.004 s | 与慢无关 |
| 嵌入 16 个块（CPU） | 12.7 s | 0.77 s（GPU） | 见下 |
| **墙钟总计** | **154.7 s** | **5.8 s** | **快 26 倍** |

两处修复：
1. **启动预热**（两个服务的 lifespan）：tokenizer + 嵌入模型在启动时加载完，
   不再由第一个用户替所有人付这笔账。`KF_WARMUP=0` 可关闭。
2. **显存仲裁**：本机 4GB 卡，摄入与检索**各自**加载一份 bge-m3（实测每份 2166MB，
   两份 4.3GB）→ 显存占满 3831/4096MB、两模型互相 thrash。
   仅靠 `torch.cuda.mem_get_info()` **判断不可靠**（真机复现：摄入已占 2.2GB 时
   检索仍认为"充足"并加载第二份），故改用**进程间标记文件**仲裁：
   谁先占下 GPU 谁写标记，后来者回落 CPU 并把这个原因写进 `/health` 的
   `embed.device_reason`。修复后 GPU 稳定在 2237MB、只有一个模型。

> 实测对照（同一批 16 个块、同一模型）：CPU `batch=16` 1.3 条/秒；CUDA **20.7 条/秒**。
> 单条**查询**嵌入在 CPU 上热态 0.23s（冷态 15.5s 是首次 kernel 初始化）——
> 所以"摄入用 GPU、检索用 CPU"是这块卡上的合理分工。

**（a2）重摄入一篇大文档要 15 分钟——五个原因叠加，全部已定位**

真机复现（`CFA_Design_合并完整版.md`，160 KB / 145k 字符 / 203 个块），逐个排除：

| 原因 | 实测 | 修复 |
|---|---|---|
| **1. 内存耗尽导致 thrash** | 三个服务各驻留一份模型 → 可用内存 0.9/15.8 GB、**提交量打满 42.8 GB**；同一份 tokenizer 加载从 3.8s 劣化到 **105s** | 停掉不需要的服务（决策），可用内存回到 5.8 GB |
| **2. 嵌入回落到 CPU** | CPU **0.6 块/秒** vs CUDA **20.7 块/秒**（差 34 倍） | 显存仲裁正常，但 4GB 卡只容一个服务 → 先启者占卡 |
| **3. provenance 二次复杂度** | 每个切片都重新规范化整页文本 → **128 ms/块**，164 块合计 21s | **按页缓存**规范化结果 → **3 ms/块，提速 41.6×** |
| **4. tokenizer 被加载两次** | 嵌入模型内部一份 + `TokenCounter` 一份；内存紧张时第二次要 43s | 复用嵌入客户端已加载的 tokenizer |
| **5. 父块突破 token 上限（真缺陷）** | 文档含 7729 字符的 SQL 代码块（≈2180 token），被整体塞进父块 → 父块 **2180 token**（上限本应 960）、单块嵌入 **27.2s** | ① 单块超预算时自行成组；② 父块按预算**拆成多个**（不再截断）；③ 子块按已产出 token 量挂到对应父块。修复后父块 max **959 token**、嵌入 **9.1s** |

排除以上后，纯 Python 侧（parse 0.11s / clean 0.84s / chunk 6.9s / provenance 0.5s /
enrich 0.07s / structured 0.02s / entities 0.05s）合计不到 9 秒，**其余全是嵌入与索引**。

**（a3）重摄入落库的版本号陷阱（迁移后暴露的真缺陷）**

重摄入沿用原 `current_version`（多为 1），而 stale 逻辑是
`WHERE version <> 当前版本`——**版本没变，旧块不会被标记 stale**，于是新旧块并存
（实测 `CFA_Design` 变成 371 块 = 旧 182 + 新 189）。

修复：新增 `scripts/purge_legacy_chunks.py`，只删除"同一文档里**没有 dense 向量**
且该文档**确实存在有 dense 的块**"的旧块（安全边界：整篇都没 dense 的一律不动）。
真机清理 526 个旧块后，14 篇全部转为 `dense 全覆盖 / 溯源退化 0`。

> 清理过程中又暴露一个我 004 迁移的 bug：`structured_records` 的外键是
> `ON DELETE SET NULL`，而**复合外键的 SET NULL 会把所有引用列置空**，包括
> `NOT NULL` 的 `tenant_id` → 删除块时报 `NotNullViolation`。已改为 `ON DELETE CASCADE`
> （语义上也对：这些记录从块派生，块没了记录就没依据）。

**（a4）答案可达阈值：从"固定经验值"改为"按库标定"**

语料补齐后（所有文档都有 dense 向量），越界查询的可达分整体上移——原先 0.45 的阈值
不再能拦住它们。14 条查询重新标定：

| 类别 | 可达分区间 |
|---|---|
| 可答（7 条） | **0.5865 ~ 0.6717** |
| 不可答（7 条） | **0.3678 ~ 0.5529** |

分离间隙仅 **+0.0336**。

**但"分布自适应"的假设被实测否定了**（`scripts/calibrate_answerability.py`）：

| 信号 | 可答 min | 不可答 max | 间隙 | 结论 |
|---|---|---|---|---|
| `top1`（绝对分） | 0.5865 | 0.5529 | **+0.0336** | ✅ 可分离 |
| `top1 − tail_mean` | 0.0102 | 0.0819 | −0.0718 | ❌ 重叠 |
| `top1 − rest_mean` | 0.0069 | 0.0723 | −0.0654 | ❌ 重叠 |
| `z_top1` | 1.3791 | 3.8379 | −2.4588 | ❌ 重叠 |

原因：越界问题**也可以有突出的最高分**（「今天天气」z=3.73），而可答问题**也可能分数平坦**
（「设备工作温度」Δtail=0.0102）。所以相对判据不成立，只有绝对相似度可用。

**最终做法：按库标定噪声底。** 用一组固定的**域外探针问题**（天气 / 烹饪 / 常识等，
与任何企业库无关）在每个库上跑一遍，取 `top1` 的 p90 作为该库的**噪声底**，
阈值 = 噪声底 + 0.03：

| 库 | 块数 | 噪声底 p90 | 标定阈值 |
|---|---|---|---|
| `acme/demo` | 644 | 0.5493 | **0.5793** |
| `demo/light` | 56 | 0.4270 | 0.4570 |
| `alpha/kbalpha` | 16 | 0.3496 | **0.3796** |

**语料越大阈值越高**——同一个越界问题在 644 块的库得 0.4586、在 16 块的库只有 0.2859，
固定阈值必然顾此失彼。真机验证 5/5 判定正确，trace 新增
`threshold_source`（`per_kb_calibration` / `global_floor` / `profile_override`）
与 `threshold_value`，让"为什么这条被拒答"可追溯。

```bash
python scripts/calibrate_answerability.py            # 标定所有库 → configs/answerability_calibration.json
python scripts/calibrate_answerability.py --dry-run  # 只打印不写
```

> ⚠️ **局限仍在**：最难的一条负例是「如何用 Rust 写一个 WebSocket 服务器」
> （0.5529），它与本库的技术架构文档**语义确实相邻**，属于难负例。按库标定能适配
> "库的稠密度"，但适配不了"问题本身与主题的语义邻接"。换语料或换模型需重跑标定脚本。

**（a5）接入 cross-encoder 重排，用它做拒答判定**

上一节留下的"语义邻接"问题，根因是**双塔模型看不到 query 与文档的交互**（两者各自
独立编码，只比较最终向量）。换成 cross-encoder（query+doc 拼接过同一个模型）后，
两种判据的差距非常直接：

| 判据 | 可答 min | 不可答 max | 间隙 |
|---|---|---|---|
| embedding 余弦 | 0.4967 | 0.0218 | — |
| **cross-encoder 概率** | **0.4967** | **0.0218** | **0.475** |

> 上表是启用 rerank 后的同一批 8 条查询实测；余弦那边的原始数据是
> 可答 0.5865 / 不可答 0.5529（间隙 0.0336）。**间隙扩大约 14 倍。**

那条最难的负例「如何用 Rust 写一个 WebSocket 服务器」：
余弦 **0.5529**（几乎撞线，随时会漏），cross-encoder **0.0218**（干净拒答）。

**但代价是延迟，而且有个陷阱**：

| 配置 | 耗时 | 说明 |
|---|---|---|
| 12 候选 × 512 token | **9.39s** | 注意力随 `候选数 × 序列长度²` 爆炸 |
| 12 候选 × 256 token | 4.62s | |
| **4 候选 × 256 token** | **1.5s** | 采用 |
| 3 候选 × 256 token | 1.1s | |

不只要调 `max_length`，还要改**打分窗口**：旧实现把融合后的**全部候选**（可能 50 个）
送进 cross-encoder，`top_n` 只截断输出——真机实测 rerank 阶段因此耗 3.9s。
现在 `rerank_candidates(candidate_limit=...)` 只送前 N 个。

**最终配置**（`configs/retrieval_profiles.yaml` + `assembly`）：

- `rerank_top_n: 4`（原 10~15；打分窗口与返回窗口一致）
- `KF_RERANK_MAX_LEN=256`（原 512）
- `assembly.answerability: rerank`（原 cosine），阈值由标定脚本按库给出 `rerank_threshold`
- 两个模型**共存于 4GB 卡**：bge-m3 2257MB + reranker fp16 1104MB = **3367MB / 4096MB**
- 重排模型**启动时预热**（首次加载 79s），否则第一个用户会以为服务挂了

**端到端结果**（真机，8 条查询）：

```
可答  0.4967 / 0.6978 / 0.8721 / 0.9070   → 全部不拒答 ✓
不可答 0.0002 / 0.0003 / 0.0004 / 0.0218   → 全部拒答   ✓
延迟   中位 2.55s  最大 2.90s（预算 3000ms）
```

> ⚠️ **这台机器的硬约束**：4GB 卡决定重排窗口只能到 4 个候选。换到 8GB 以上的卡时，
> **不用手改配置**——下面的自适应会自己放宽窗口，但**必须重跑标定脚本**
> （候选窗口变了，分数分布就变了）。

**（a6）重排窗口按可用显存自适应**

写死 4 个候选是 4GB 卡逼出来的妥协，换块好显卡就白瞎了。现在按**总显存**分档：

| 档位 | 总显存 | 重排窗口 | 序列上限 | 依据 |
|---|---|---|---|---|
| `compact` | < 6 GB | **4** | 256 | 4GB 卡上 12 候选要 9.4s，4 个只要 1.5s |
| `standard` | 6~10 GB | **10** | 384 | 算力足够时放宽 |
| `wide` | ≥ 10 GB | **20** | 512 | 排序质量最好 |
| `cpu` | 无 CUDA | **2** | 256 | CPU 上约 3.4s/对，多排没有意义 |

用**总显存**而不是"空闲显存"作判据：空闲值受同机其他进程影响，且 WDDM 驱动报告的数字
明显偏保守（实测 bge-m3 占 2.2GB 后它就报"不足"，而 1.1GB 的重排模型其实放得下）。

想固定窗口就设 `KF_RERANK_TOP_N=<n>`；想强制序列长度用 `KF_RERANK_MAX_LEN=<n>`。
**配置里的 `rerank_top_n` 会被档位值覆盖**——否则自适应永远不生效，大卡上还是只排 4 个。

> 🔴 **换档位必须重标定，这是硬约束。** 候选窗口不同 → cross-encoder 的 top-1 分数分布
> 不同 → 拒答阈值必须跟着换。所以标定结果按档位分开存
> （`answerability_calibration.json` 的 `tiers`），并且：

| 查找顺序 | 结果 |
|---|---|
| 当前档位有专属阈值 | 用它（最准） |
| 当前档位**没**标定 | 回落全局底线，**并在响应 note 里明确告警** |
| 绝不跨档位复用 | 跨档位复用会让拒答行为换显卡后系统性偏移，且完全无声 |

标定命令：

```bash
python scripts\calibrate_answerability.py              # 只标当前设备档位（约 20~30 分钟）
python scripts\\calibrate_answerability.py --tier all   # 标定全部三档（耗时长，换机器也能直接用）
```

自检命令（换机器 / 改参数 / 重标定之后跑）：

```bash
python scripts\check_adaptive_rerank.py
```

它会核对：档位是否识别、序列上限是否跟随档位、当前档位**是否真有专属阈值**、
实测延迟与拒答是否正确。退出码 0 表示全部符合预期。

> 排查方法论留档：先按阶段计时（`stages.timing_ms`），再对可疑阶段做**定点微基准**。
> 本例中 `enrich+derive` 在 16 块小文档上是 0.01s，一度被忽略；补测后确认它在大文档上
> 也只有 0.09s，从而把嫌疑收敛到 embed/index 与内存。


**（b）拒答形同虚设：融合分对「能否回答」没有区分度**

真机实测（acme/demo，520 活跃块，`hybrid_rerank`）：

| 查询 | RRF 融合分 |
|---|---|
| 可答「MX-100 的功率是多少」 | 0.0164 |
| 不可答「今天天气怎么样，帮我订张机票」 | 0.0164 |

**完全相同**——RRF 的理论上界只有 `1/(k+1)≈0.0164`，排名第一就饱和。
于是「按分数拒答」在 RRF 档位下根本无法成立：`strict_refuse` 要求 0.55，
永远达不到，越界查询畅通无阻（实测「火星殖民预算表」在 `exact_lookup` 档位下
放行了 **21 条证据**）。

同一批数据换成**归一化稠密余弦**则分离干净：

| 类型 | 查询 | 余弦 |
|---|---|---|
| 可答 | MX-100 的功率 | 0.738 |
| 可答 | 密码长度要求 | 0.774 |
| 可答 | 数据库表结构 | 0.637 |
| 不可答 | 订机票 | 0.370 |
| 不可答 | 火星殖民预算 | 0.292 |
| 不可答 | 珠峰高度 | 0.405 |

修复：`assembly.answerability: cosine` + `assembly.answerability_floor: 0.45`。
答案可达分**必须独立于档位通道**（`exact_lookup` 不含 dense，只沿通道取分会算不出分
→ 闸门静默失效），因此取分优先级为：召回阶段的 dense 分（零额外往返，避免把检索
推过 3000ms 全局预算）→ 块向量现算 → 后端查询接口。
失败时**退回融合分并如实记 note**，绝不因为"新判据算不出来"就静默放行。

修复后 `examples/grounding_agent.py --scenario` 由 2/4 通过变为 **4/4**。

#### 1. 跨租户静默串数据（P0，缺陷已在 README 承认但被低估）

**根因不止一处**：`knowledge_bases` / `documents` / `chunks` 等表主键不含租户维度，
且写入路径的 `ON CONFLICT` 目标只按业务 id 判重：

- `ON CONFLICT (kb_id) DO NOTHING` → 第二个租户连自己的库都建不出来
  （真机证据：`chunks` 里有 `demoB/graphdemo`，而 `knowledge_bases` 里只有 `demoA/graphdemo`）；
- `ON CONFLICT (doc_id) DO UPDATE` → 只更新版本号，`tenant_id/kb_id` 保持首写租户；
- `UPDATE chunks SET status='stale' WHERE doc_id=…` → 无租户过滤，会把**其他租户**同名文档的块一起标作废。

**修复**（`db/004_tenant_scoped_keys.sql`，幂等可重跑）：
11 张表主键前置 `tenant_id`，15 个外键同步改为复合键（引用完整性与 RLS 同层），
`documents` / `decision_trace` / `ingestion_jobs` / `document_versions` / `chunk_vectors`
补齐 RLS；写入侧冲突目标全部改为租户复合键。

**真机验证**：同一 `doc_id` 先后以 `alpha` / `beta` 摄入 → `documents`、`chunks`
**各自归属两个租户**（修复前 beta 零行、全并入 alpha），接口不再返回假 `ok=true`。

**数据修复（如实记录）**：迁移中发现既有库已有不一致：8 行 `(acme, demo, 数据库设计方案…)`
的块与 `(demo, company)` 逐字相同、且无对应 `document_versions` 行——属旧缺陷残留。
迁移只删除**引用已确认悬空**的行（614 行 vs 修复前 622 行），不删除任何归属正确的行；
子表（`chunk_provenance` / `chunk_acl` / `chunk_vectors`）的租户标签按所属块改齐
（它们是 ACL 规则与溯源偏移，不能丢）。

#### 2. 稠密检索链路完全失效（P0，此前未承认）

**现象**：Qdrant 里的点**只有 sparse、没有 dense**——`TEI_EMBED_URL` 指向的 8080
端口上跑的是检索服务本身，TEI 从未启动。真机实测 **8 个档位中 6 个恒定返回 0 证据**
（`naive` / `hybrid_rerank` 等），而 `/health` 的 `reranker.configured` 仍显示正常，
**失效是静默的**。

**修复**：新增进程内本地嵌入回退（`kf_ingest.clients.LocalEmbedClient` 与
`kf_retrieve.backends.LocalEmbedClient`，同一模型 BGE-M3、同一 CLS+L2 池化）；
`try_embed_client()` 统一为「TEI 优先，本地回退」；`/health` 新增 `embed` 段如实
暴露后端与就绪状态，两者都不可用时明确写出「dense/hybrid 档位将返回 0 证据」。

**真机验证**：`dense=13/13`，Qdrant 集合同时含 `dense`+`sparse` 两种向量；
`naive` 档位从 **0 证据 → 8 条**。

#### 3. 切片溯源偏移失效（P0，影响"点击回原文"承诺）

**现象**：`compute_provenance` 在**切片之前**按单个块定位，而切片产出的是跨块拼接文本，
拼接串在页文本里找不到 → 全部退化 `provenance_approx` + 整页区间。真机实测
**102 条溯源里 88 条（86%）是整页区间**，同一文档 20 个块只有 3 个不同的起止位置。

**修复**（`kf_ingest/clean.py`）：改为**切片后**计算，新增 `compute_chunk_provenance`，
四级定位——精确子串 → 规范化整段 → 最长可定位前缀 → **bigram 连续窗口锚点**。
最后一级是为解决真实障碍：清洗阶段的 `normalize_text` 会做 NFKC 归一，
**块文本与原始页文本在字符层面已不同**（实测页 3~5 的块整段匹配全部失配），
而 bigram 对这类差异天然稳健（实测命中率 100%）。
另要求锚点必须是**连续窗口**（≥ `MIN_ANCHOR_CHARS`），否则宁可标 `provenance_approx`
——缺偏移好过给一个看似精确却指向无关位置的偏移。

**真机验证**（真实多章节 PDF）：整页退化 **8/10 → 0/10**，`exact_ratio` **0.14 → 1.0**；
检索返回区间不再雷同（`[17,430]` / `[17,168]` / `[17,97]` 等）。

#### 4. 父块回填造成「引用与内容错位」（P1，用户可感知的精度问题）

**现象**：命中 320 token 的子块，返回的却是父块全文——真机放大 **3.2~15.4 倍**；
更麻烦的是 `content` 被换成父块，而 `char_range` 仍指向子块，
**「读到的内容 ≠ 引用指向的区间」**；同时按父块去重会把同一父块下命中的多个
不同子块压成 **1 条**。

**修复**（`kf_retrieve/assemble.py`）：父块**默认不再替换 `content`**，改以
`parent_context` 字段按需附带（`configs/retrieval_profiles.yaml` 的 `assembly.parent_context_tokens`
控制上限，默认 0 即不带）；去重改为按 `chunk_id`，不再按父块合并；
保留 `backfill_parent_content: true` 以复现旧行为供确有需要的调用方。
`span_union` 另修一个衍生缺陷：父块跨页时取首子块 `char_start` 与末子块 `char_end`
会产出**倒置区间**（真机出现 `[653, 97]`），现在跨页只保留首页区间并标注
`provenance_end_page` / `provenance_spans_pages`。

#### 5. 其它已修复项

| 问题 | 修复 |
|---|---|
| `chunk_acl` 无规则时默认**放行**（企业多租户下 fail-open） | `assemble` 新增 `acl_default_allow`（配置 `assembly.acl_default_allow`），多租户应设为 false |
| 网关取父块是 **N+1**（每个父块一次 Qdrant 往返） | 后端新增 `get_chunks` 批量接口，`_collect_parents` 一次往返取回 |
| `_dedupe_near` 与**已丢弃**项的哈希比较，误判不同内容为重复 | 只与已保留项比较 |
| `decision_trace` **从不写入**（硬约束「落 trace」未落地） | 新增 `gateway/src/trace.rs`：有界通道 + 独立线程落库，检索路径非阻塞；未配置库/无 psql 时降级为空写入器并在 `/health` 的 `decision_trace` 如实暴露 |
| 页眉判定把**大字号段落**当页眉删除（`top < 页面高 × 6%`） | 大字号（≥ 正文 × 1.15）块不作页眉判定，避免正文被清洗删掉 |
| `/health` 的 `collection` 与真实查询集合含义不同；`reranker.configured` 误导 | 改名为 `collection_note` 并写明派生规则；reranker 改为 `url_configured` + note 说明不代表端点可达 |
| 网关决策超时错误信息**硬编码 300ms**（配成 5000 仍显示 300） | 错误信息改用运行期实际值 `state.decision_timeout` |

> **一处自我更正**：此前把「`DECISION_TIMEOUT_MS` 是硬编码、改 env 无效」当作缺陷，
> 复查 `gateway/src/main.rs` 后确认它**本来就是环境变量可配**的，那个 `300` 只是缺省值。
> 真正的问题是错误信息用了缺省常量。同批我更正的还有「语种判定错误」——那是我的
> **测量误差**（用 PowerShell 5.1 发 JSON 导致中文被编码破坏），用 Python 客户端复测
> 全部判定正确，故未改动语种判定代码。


### 已知缺口（按优先级）

1. **Laya 权重已加载并真实推理；阈值基准已改为 top-1 后路由「部分有效」，但模型判定质量本身仍差**（最需要知道的一条）：
   - **背景：该 checkpoint 的 `confidence` 未校准**。加载时 Laya 自报
     `invalid temperatures or values outside [0.5, 5]; using choice:11+=0.10 -> 0.5.
     Treat confidence from the affected entries as uncalibrated.`，实测 `confidence`
     落在 **0.06 ~ 0.47**，几乎全部低于设计阈值 `DECISION_CONFIDENCE_FLOOR=0.45`。
     原实现直接拿它比阈值，会把模型**判对**的档位也一起丢掉（`MX-200 的价格是多少？`
     判对 `exact_lookup` 却因 0.3895 被丢弃 → 最终 `hybrid_rerank`）。
   - **本轮修复**：低置信护栏的阈值基准改为**按来源区分**——Laya 结果改用其**自身
     概率分布的 top-1**（该值稳定得多），heuristic 结果仍用它自己的 `confidence`。
     响应新增 `threshold_basis` / `threshold_value` 两个字段，如实暴露「拿什么值比阈值」，
     便于事后审计为何采纳 / 丢弃某档位（`schemas/decision.schema.json` 已同步）。
   - **真机复测（同一组 5 条查询，两次运行结果一致可复现）**：

     | 查询 | Laya 判定 | conf | top-1 | 阈值基准 | 最终档位 | fallback |
     |---|---|---|---|---|---|---|
     | `MX-200 的价格是多少？` | `exact_lookup` ✅判对 | 0.3895 | 0.4909 | top-1 | **`exact_lookup`** ✅保留 | 否 |
     | `MX-100 的保修期是多久？` | `naive` ❌ | 0.1697 | 0.3486 | top-1 | `hybrid_rerank` | 是 |
     | `对比 A 型和 B 型的参数差异…` | `hybrid_rerank` ⚠️（理想为 `decompose_agentic`） | 0.2568 | 0.3717 | top-1 | `hybrid_rerank` | 是 |
     | `What is the price of MX-200?` | `exact_lookup` ✅判对 | 0.1228 | 0.3273 | top-1 | `hybrid_rerank` | 是 |
     | `今天天气怎么样，帮我订张去上海的机票` | `naive` ❌（应拒答） | 0.4389 | 0.5861 | top-1 | **`naive`** ⚠️错判被采纳 | 否 |

   - **改动带来的净效果（如实对照上一版「用 confidence 比阈值」）**：
     - **改善**：`MX-200 的价格是多少？` 判对且被保留（top-1 0.4909 > 0.45）；
       原先「多跳查询被判 `naive` 且被采纳」的那条，top-1 落到 `hybrid_rerank` 0.3717
       **被护栏拦下**，回退安全档——「错判被采纳」的条目从 2 条降到 1 条。
     - **未改善**：英文 `What is the price of MX-200?` 虽判对，但 top-1 仅 0.3273，
       **仍低于阈值被回退**（判对不保证 top-1 越阈）；越界查询仍被判 `naive`
       且 top-1 0.5861 越阈**被采纳**（该条本应 `strict_refuse`）。
     - **结论**：top-1 只换了「阈值比较的基准量」，**不改变模型判定质量**。当前
       「Laya 参与决策」是**部分有效**（不再名大于实），但离「按设计承接策略路由」
       仍有实质差距。要真正达标需先解决：① temperature/置信度校准；② 非英文档位
       判定质量（`typed-decisions` 为英文模型，非英文走 `multilingual` 仍判错）。
   - **CPU 时延**（复测，非估算）：本组 5 条 873~5003ms，其中英文 `typed-decisions`
     那条 **5003ms** 已触到本机 `DECISION_TIMEOUT_MS=5000` 上限（见第 2 条）。
2. **CPU 推理时延超出网关 300ms 决策预算**（实测，非估算）：

   | checkpoint | CPU 加载 | CPU 热推理 | CUDA 热推理 |
   |---|---|---|---|
   | `multilingual`（zh 查询走这条） | 18.7s | 292~1283ms | 147ms |
   | `typed-decisions`（en 查询走这条） | 68.6s | 1012~1236ms | 390~511ms |

   而 `configs/model_registry.yaml` 记的是「~33ms」、网关硬超时
   `DECISION_TIMEOUT_MS`（默认 300ms）。CPU 下网关必然按超时回退 heuristic（服务端
   `mode=laya` 却不生效）。本机 `.env` 已把 `DECISION_TIMEOUT_MS` 调到 5000ms 让真权重
   生效——**这是对设计 300ms 硬限的有意偏离**，且会占用端到端 3000ms 总预算，需一并权衡。

   - **补充实测（经网关 `decision_trace` 落库后可直接观测）**：`DECISION_TIMEOUT_MS=5000`
     时，连续 3 次检索中 1 次成功（`decide=1267ms`，且 `decision_trace` 记
     `model_id=convaiinnovations/laya`）、**2 次恰好卡在阈值上超时**
     （`5003ms` / `5012ms` → 回退 `gateway-safe-default`）。也就是说即使把预算放宽到
     5000ms，CPU 部署仍处于**临界不稳定**状态；要稳定承接策略路由必须上 GPU
     （设计表里 CUDA 热推理 147~511ms）或换更小的蒸馏模型。
   - 现在可以通过 SQL 直接统计回退率，不必靠日志推断：
     `SELECT fallback_used, count(*) FROM decision_trace GROUP BY 1;`
3. **权重来源与容器依赖**：权重可完全离线复用
   （`LAYA_CACHE_DIR` 指向含 `hub/` 的缓存根即不联网；`LAYA_MODEL_PATH` 可指向 snapshot
   根，由 `LAYA_SUBFOLDER` 选 checkpoint）。容器内固定 `laya==0.3.9`（PyPI 无 0.3.7；
   已核对 `Agent` / `system_one` 签名与本地验证过的 0.3.7 一致）。
4. **网关档位常量与 YAML 双份维护**：`gateway/src/profiles.rs` 以常量镜像
   `retrieval_profiles.yaml`，YAML 变更需同步。
5. **检索服务 `/health` 报告的 collection 取自环境变量默认值**，而真实查询
   按 `tenant/kb` 派生（如 `kf_acme_demo`）——真机已复现：`/health` 报
   `kf_default_default`，实际命中 `kf_acme_demo`。运维观察时需注意二者含义不同。
6. **OpenSearch 实例未起**：分语言 BM25 代码与降级路径已就绪且单测通过，
   但本机未启动 `lang-sparse` profile，故**分语言 BM25 的真实查询未跑过**。

### 本轮测试新发现的缺陷（已修复：语言判定误判 → 中文召回 0）

**中英混排的技术文档被判成 `en`，稀疏通道按语言标签过滤 → 中文查询召回 0。**

- **现象（真机复现）**：摄入 `数据库设计方案_合并完整版.md` 后，中文查询
  「数据库表结构是怎么设计的？」**证据 0 条**；同文档英文查询
  `database schema design` 召回到 2 条。
- **根因**：[tokenize.py](file:///F:/knowledge-fabric/services/ingest/kf_ingest/tokenize.py)
  的 `detect_lang` 按**原始字符数**比较 `han >= latin` 判 `zh`。该文档全文
  汉字 3153 / 拉丁 8468（技术术语与代码占比高），被判 `en`；而稀疏通道
  （BM25 单语言）**按 `lang` 过滤**——查询语言 `zh` 撞不上 `en` 分片，命中归零。
  这违反 §3 已固化约束「多语言稀疏：分语言索引 + 查询语言路由」的正确实现前提
  （语言标签本身必须判对）。
- **已修复**：改为**汉字 ×5 权重**（`CJK_LATIN_WEIGHT=5`）+ **绝对量下限**
  `HAN_ABSOLUTE_FLOOR=300` 即判 `zh`。后者是为极端的代码型文档兜底——
  `CFA_Design` 汉字 6733 / 拉丁 73940（比例仅 0.09），光靠加权比例仍判 `en`。
  新增 `tests/test_lang_detect.py`（12 条，含参数化边界）。修复后 8 篇 `.md`
  全部正确判 `zh`，重新摄入后中文查询召回 **2 条**证据。
- **另修一处同族缺陷**：Markdown 解析里 `detect_lang` 原传 `text[:4000]`
  （PDF 侧因分页只取前 5 页），改为**全文**判定——截断采样会让长文档的语言标签
  取决于开头几千字符的偶然分布。调用点已核对：PDF 侧仍用首页采样（合理，PDF 无全文
  时须先解析分页），Markdown 用全文。

### 本轮测试新发现的缺陷（跨租户串数据，未修复）

**`documents` / `chunks` 的主键不含租户维度，同一 `doc_id` 在不同租户/库间会互相覆盖。**

- **现象（真机复现）**：用 `POST /v1/ingest` 以 `tenant=demo, kb=batch` 摄入
  `灯塔…web端第一版.pdf`，响应如实回报 `postgres: 15`、Qdrant 也写进了正确的
  `kf_demo_batch` 集合；但 Postgres 里该文档落在 **`default/default`** 名下
  （`SELECT tenant_id,kb_id,doc_id FROM documents` 可见），`demo/batch` 查不到任何行。
- **根因**：`documents_pkey = PRIMARY KEY (doc_id)`、`chunks_pkey = PRIMARY KEY (chunk_id)`
  ——**都不带 `tenant_id` / `kb_id`**。而 [index.py](file:///F:/knowledge-fabric/services/ingest/kf_ingest/index.py#L167-L175)
  用 `ON CONFLICT (doc_id) DO UPDATE SET current_version = …`：当同名 `doc_id` 已属于
  另一个租户时，冲突走 UPDATE 分支，**只更新版本号，`tenant_id/kb_id` 保持首次写入的租户**
  ——新租户的归属被静默丢弃。同一文件早已以 `default/default` 摄入过，故本次落库撞键。
- **影响**：① 新租户的文档在 Postgres 侧「归属错误」，RLS 下该租户看不到自己的元数据，
  而首次写入的那个租户会**看到别人后来摄入的内容**——与项目硬约束「禁止跨租户数据串扰」
  直接冲突（与先前修掉的 `graph_entities` 是**同一类**缺陷：唯一键漏了租户维度）；
  ② [index.py](file:///F:/knowledge-fabric/services/ingest/kf_ingest/index.py#L214-L218)
  的旧版本标记 `UPDATE chunks SET status='stale' WHERE doc_id=%s AND version<>%s`
  **同样没有租户过滤**，会把其他租户的同名文档块一并标 stale。
- **未修复**：修它要改 DDL 主键（新迁移）+ 全部 `ON CONFLICT` 目标 + stale 标记的过滤条件，
  属 schema 级变更，本轮只做**测试与如实报告**，未擅自动手。Qdrant 侧不受影响
  （集合按 `kf_<tenant>_<kb>` 隔离，故检索仍能命中）。

### 本轮新增（原六项缺口：已落地并真机校验）

跨服务契约冻结在 `db/002_graph_acl_structured.sql` + `configs/retrieval_profiles.yaml`
两处；契约红线为「**不许假装跑了**」——无 LLM 标 `extractive`、无生成端点标
`no_generator`、外部依赖不可用只记 note。

- **结构化双轨**：摄入侧 `kf_ingest/structured.py` 从表格块与 SKU/价格形态提取，
  写 `structured_records`（含 `tenant_id/payload/doc_id/version`，幂等键
  `uq_struct_chunk_schema_sku`）；Qdrant payload 增固定键 `skus/price/currency/attrs`；
  检索侧 `kf_retrieve/structured.py` 在 payload 层做**精确过滤**（识别不到条件即跳过，
  不退化成 BM25）。
- **GraphRAG**：`kf_ingest/graph.py` 规则抽取实体（无 LLM）→ 无向规范化建边 →
  连通分量社区 + 摘要；检索侧 `kf_retrieve/graph.py` 走 Postgres 且**连接后先
  `set_config('app.tenant_id')`**。**按库 opt-in**：`--graph` 才把
  `knowledge_bases.graph_enabled` 置 TRUE，检索需 `graph_enabled` 显式传入。
- **Agentic RAG**：`kf_retrieve/agentic.py` 硬限 `min(全局 3, 档位 max_rounds)`，
  逐轮判定 `saturation` / `budget` / `max_rounds`；**无生成端点只跑 1 轮并标
  `no_generator`，绝不伪造轮数**。
- **OpenSearch 分语言 BM25**：`analyzer_for_lang` / `index_name` 按语言选择分析器；
  未配 `OPENSEARCH_URL` 时只记 note 并回退 Qdrant 稀疏。
- **租户 ACL + RLS**：网关 `build_search_body` **仅在显式传入时才写**
  `principal`/`graph_enabled`（缺省与改动前逐字节等价）；RLS 的可行性关键是
  **属主默认绕过 RLS**，故建**非属主角色 `kf_app`** 供检索侧连接（摄入仍用属主
  `kf` 写全量），策略统一 `USING (tenant_id = current_setting('app.tenant_id', true))`，
  未声明租户即默认拒绝。

真机校验中**发现并修复两处真实缺陷**（均非子任务自报，是我独立复核跑出来的）：

1. **图存储跨租户串数据**：`graph_entities` 唯一键原为 `(kb_id, norm_name)`、
   实体/边/社区 id 由 `kb_id` 派生——**两个租户复用同名 kb 时，后摄入租户的
   `ON CONFLICT` 命中前一租户的行**，实测 `demoB` 的 69 个实体被并进 `demoA`。
   已把唯一键与 id 派生全部扩为带 `tenant_id`，并补迁移
   `db/003_fix_graph_tenant_scope.sql`（幂等，可对既有库就地修正）；修复后
   `demoA`/`demoB` 各 69 实体、2346 边、1 社区，互不可见。
2. **`exact_lookup` 结构化命中被静默丢弃**：`weighted` 融合对**权重缺失的通道记 0 分**，
   而 `fusion_weights.exact_lookup` 只有 `dense/sparse`——结构化通道召回成功
   （`matched=2`）却不进证据（`evidence=0`）。已补 `structured: 0.5` 并新增回归
   测试「凡 weighted 档位的 channels 必须都有权重」。

### 已解决（历史缺口留档）

- **检索证据缺字符级引用**（客户端验收时发现，影响「点击回原文」承诺）：
  对外 evidence 只有 `content / page / source`，**没有字符偏移与来源 URI**，而
  `chunk_provenance` 里每条都有非空偏移——设计承诺的「可高亮 / 一键还原」在对外契约上
  是断的（`citations` 里有 `char_range`，但 LLM 直接读的 evidence 里没有）。已在
  `assemble._to_evidence` 补 `char_range / char_start / char_end / source_uri`，并
  **显式标注偏移归属** `offset_chunk_id`：父块回填只替换 `content`（其来自
  `parent_chunk_id`），偏移永远属于**实际命中的子块**，两者语义不同、不能含糊。
  同步更新 `schemas/response.schema.json`。真机校验由 21 PASS/1 FAIL 变为 **22/22**。

- **Qdrant 侧版本切换漏标 stale**（真机联调发现，影响正确性）：
  `index.py` 的文档字符串写明「新版本写入后把旧版本块标记 `stale`，检索侧过滤」，
  但实现只做了 Postgres 的 `UPDATE chunks SET status='stale'`，**Qdrant payload
  仍是 `status: active`**。实测：写入 v2 后 Qdrant 为 `{v2/active: 12, v1/active: 12}`，
  检索返回**全部是 v1 旧版本证据**（`ev_has_v1=True, ev_has_v2=False`）——
  agent 会拿到已作废的文档内容。已补 `mark_stale_qdrant()`（与 Postgres 侧对称：
  `doc_id` 相同且 `version != 当前` 的点置 stale，不物理删除），修复后
  Qdrant 为 `{v2/active: 12, v1/stale: 12}`，检索只返回 v2；并补回归测试
  （用最小 Qdrant 替身求值 Filter 语义）。
- **Qdrant / Postgres 真机未验证（原缺口 #2）**：已实测通过。真机链路：
  测试 PDF 摄入 → Qdrant `kf_acme_demo` 12 点（green，命名向量 `dense`(1024/Cosine)
  + `sparse`）→ Postgres 12 chunks / 12 chunk_vectors / 10 chunk_provenance
  / 1 version → 检索服务（Qdrant 后端，sparse 通道 10 命中 → RRF → 组装）返回
  带 provenance 的 evidence（page + char_range + source_uri）→ 版本切换 v1→v2 生效。
  中文内容与偏移映射经验证无损坏（`content LIKE '%第三条%'` 命中 4 行）。
  无 TEI 时如设计般降级：`channels={sparse: 10}`，稠密缺席只记 note。
- **`001_init.sql` 在无 pgvector 的镜像上会中断初始化**（真机联调首日发现）：
  原第 8 行无条件 `CREATE EXTENSION "vector"`，而 compose 默认镜像是
  `postgres:16-alpine`（不含 pgvector），psql 以 `ON_ERROR_STOP` 退出 →
  容器 Exit 3，整库建表全部回滚。但 `vector` 类型在 DDL 中**从未被使用**
  （向量全在 Qdrant，`chunk_vectors` 只存 point_id）。已改为 PL/pgSQL
  `EXCEPTION` 包裹，装不上只 `RAISE NOTICE` 不中断——符合「外部依赖不可用
  仅记录警告」的硬约束。修复后 kf-postgres 正常 healthy，14 张表全部建成。
- **切片策略差异化（原缺口 #1）**：已落地。实现分三层——
  `strategies.py` 只做「文本 → 单元」的纯计算内核（语义断点 / 条款锚点 /
  代码语法锚点 / 表格结构化），`chunk.py` 按 `strategy` + `pipeline` 分派到不同
  单元与父子结构，`pipeline.py` 为其注入句向量编码器。要点：

  - **配置即契约**：策略是否产出父块由 YAML `pipeline` 是否含 `parent_child` 决定
    （`doc_structured` / `semantic` 产出父块，其余四个平铺）；`pipeline=None` 时保留
    旧默认行为以向后兼容。
  - **语义断点双源相似度**：有 TEI 用真实向量余弦（跨语言可靠），否则用词法代理
    （复用索引侧同源稀疏分词做 tf 余弦）。无嵌入端点时**不静默退化为整块不切**，
    实际使用的来源写入切片统计 `notes`。
  - **条款 / 代码无锚点时不猜**：识别不到锚点则回退结构递归切分并记 note，
    而不是把整组压成一个巨块；末个锚点延伸到组尾产生的超长单元按句子边界再拆，
    并如实记录「已按句子边界拆分（牺牲单条原子性）」。
  - **页级（visual_page）** 一页一块、不臆造字符偏移（provenance 置 None，
    缺偏移好过给错偏移）。

  实测（真实 PDF `kf_ab.pdf`，6 页，`questions_per_chunk=3`）：

  | strategy | 子块 | avg_tok | ndcg@10 |
  |---|---|---|---|
  | doc_structured | 10 | 278.9 | 0.7654 |
  | clause_atomic | 10 | 269.6 | 0.9049 |
  | code_aware | 9 | 299.4 | 0.8763 |
  | semantic | 8 | 328.8 | 0.7199 |
  | visual_page | 6 | 438.2 | 0.9034 |
  | table_centric | 10 | 278.9 | 0.7654 |

  六策略产出互不相同，「所有策略产出完全相同 → 本次 A/B 无效」告警已消除。
  该语料仅 6 块（≤10），recall 恒为 1.0 无区分力，此表只作**上界参考**，
  不可用于选参——选参需用块数远大于 10 的语料重跑（诚实性护栏按设计工作）。

## 6. 下一步（Roadmap）

- [x] 摄入流水线：解析 → 可逆清洗 → 切片 → 增强 → 嵌入 → 索引（含 CLI 与降级）
- [x] 切片策略引擎：路由 + 四段式 + 父子块 + 原子块；`chunk_eval.py` 可本地 A/B
- [x] 检索链路：混合召回 + RRF/加权融合 + TEI 重排 + 组装 + 引用溯源
- [x] Laya 决策服务：档位路由 / 特征派生 / 低置信回退 / 口径化 trace
- [x] 网关：REST + MCP 双形态；决策不可用时安全回退
- [x] **客户端**：摄入 HTTP 接口（上传即摄入）+ 单文件零依赖中文 Web 控制台（摄入 / 文档库 / 检索）
- [x] 多语言：一集合 + `lang` 仅过滤稀疏通道（跨语言稠密召回保留）
- [x] **切片策略差异化**：语义断点 / 条款原子 / 代码语法 / 页级视觉嵌入
- [x] **Qdrant + Postgres 真机联调**：真实写入 → 真实召回 → 版本切换
- [x] **Agentic RAG**：≤3 轮 + 饱和检测 + 预算硬限（无生成端点时单轮且标 `no_generator`）
- [x] **GraphRAG**：轻量实体图 + 社区摘要（按库 opt-in，默认关闭）
- [x] **结构化双轨**：SKU / 价格精确过滤通道
- [x] **OpenSearch 分语言 BM25 分析器**（代码与降级路径就绪；起 `lang-sparse` 实例即可用）
- [x] **租户 ACL 主体贯通 + RLS 落地**（非属主角色 `kf_app`；真机跨租户隔离已验证）
- [x] **Laya 真权重加载**（离线复用缓存，`mode=laya`、真机 `/health` 已确认）
- [x] **Laya 低置信护栏阈值基准修正**（改用其概率分布 top-1 + 暴露 `threshold_basis`；
      真机复测「错判被采纳」由 2 条降到 1 条、判对被保留 1 条——详见「已知缺口」）
- [x] **Markdown 文档解析**（标题 / 列表 / 表格原子块 / 围栏代码 / 前置元数据；无分页如实记 `no_page_layout`）
- [x] **目录遍历批量摄入**（CLI `--path <目录>`、`POST /v1/ingest-dir`、控制台一键扫描三入口共用同一判定源）
- [x] **语言判定误判修复**（中英混排文档被判 `en` → 中文查询召回 0；汉字 ×5 权重 + 绝对量下限）
- [ ] Laya 置信度校准（temperature 越界导致 `confidence` 系统性偏低）与档位判定质量提升
      （英文 `typed-decisions` 对英文查询仍会判错，非英文 `multilingual` 亦判错）
- [ ] 决策时延：CPU 实测 0.3~5.0s 远超设计 300ms；需 GPU 部署或蒸馏小模型以满足预算
- [ ] 解析增强：Docling / PP-OCRv5 / PP-StructureV3（扫描件与复杂版式）；`.docx` / `.txt` / HTML 解析器
- [ ] 跨租户串数据缺陷修复：`documents` / `chunks` 主键补租户维度 + 迁移 + 全部 `ON CONFLICT` 目标
- [ ] 评测：RAGAS / RGB 口径接入，OpenTelemetry 全链路 trace
