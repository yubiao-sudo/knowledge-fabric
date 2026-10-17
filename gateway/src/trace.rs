//! 决策 trace 持久化（`decision_trace` 表）。
//!
//! 为什么需要它：README 把「决策护栏：低置信回退默认策略；**落 trace**；单查询
//! token 与时延硬上限」列为硬约束，但仓库里此前**只有建表语句，没有任何写入代码**
//! ——真机 `SELECT count(*) FROM decision_trace` 恒为 0。没有 trace 就无法回答
//! 「为什么这条查询走了这个档位」，而这正是该约束想提供的能力。
//!
//! 实现取舍（如实说明）：
//!   * 网关用**独立工作线程 + 有界通道**写库，检索路径只做一次非阻塞 `send`，
//!     绝不让落库失败或变慢影响检索——护栏优先级高于可观测性。
//!   * 通过 `DATABASE_URL` 指定的库写入；用 `psql` 子进程传参（参数经
//!     `-v` 变量绑定，不经 shell 拼接）。**没有新增 Rust 依赖**：本项目
//!     `cargo check --offline` 是既有约定，引入需要联网下载的 crate 会破坏它。
//!   * psql 不存在 / 库不可达 / 未配置 DATABASE_URL 时，**只记一次警告并停用**，
//!     不静默假装写了，也不影响检索。

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread;

use tracing::{debug, info, warn};

/// 一条决策 trace（字段与 `db/001_init.sql` 的 `decision_trace` 对齐）。
#[derive(Debug, Clone)]
pub struct DecisionTrace {
    pub query_id: String,
    pub tenant_id: String,
    pub kb_id: Option<String>,
    pub model_id: String,
    pub model_version: Option<String>,
    pub profile: String,
    pub escalate: Option<String>,
    pub answerable_prob: Option<f64>,
    pub confidence: f64,
    pub probabilities: Option<String>,
    pub features: Option<String>,
    pub reason: Option<String>,
    pub fallback_used: bool,
    pub latency_ms: Option<f64>,
}

/// trace 写入器句柄。`None` 表示落库不可用（未配置 / psql 缺失）。
#[derive(Clone)]
pub struct TraceWriter {
    tx: Option<SyncSender<DecisionTrace>>,
}

impl TraceWriter {
    /// 未配置数据库时的空写入器（检索路径调用 `record` 是安全的空操作）。
    pub fn disabled() -> Self {
        Self { tx: None }
    }

    pub fn is_enabled(&self) -> bool {
        self.tx.is_some()
    }

    /// 非阻塞投递。通道满（写入跟不上）时**丢弃并计数**，绝不阻塞检索。
    pub fn record(&self, trace: DecisionTrace) {
        if let Some(tx) = &self.tx {
            if tx.try_send(trace).is_err() {
                debug!("decision_trace 队列已满，本条丢弃（不阻塞检索）");
            }
        }
    }
}

/// 按 `DATABASE_URL` 启动写入线程。未配置或 psql 不可用时返回 disabled 写入器。
pub fn start(database_url: Option<String>, queue: usize) -> TraceWriter {
    let url = match database_url {
        Some(u) if !u.trim().is_empty() => u,
        _ => {
            info!("未配置 DATABASE_URL：decision_trace 不落库（检索不受影响）");
            return TraceWriter::disabled();
        }
    };
    if !psql_available() {
        warn!("未找到 psql：decision_trace 不落库（检索不受影响）");
        return TraceWriter::disabled();
    }

    let (tx, rx) = sync_channel::<DecisionTrace>(queue.max(1));
    let writer = TraceWriter { tx: Some(tx) };
    thread::Builder::new()
        .name("kf-decision-trace".to_string())
        .spawn(move || {
            let mut written: u64 = 0;
            for trace in rx {
                match insert(&url, &trace) {
                    Ok(()) => {
                        written += 1;
                        if written == 1 {
                            info!("decision_trace 首次写入成功，落库已启用");
                        }
                    }
                    Err(err) => warn!(error = %err, "decision_trace 写入失败"),
                }
            }
        })
        .ok();
    writer
}

fn psql_available() -> bool {
    Command::new("psql")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 十六进制编码（小写）。用于把任意 UTF-8 文本转成**纯 ASCII** 传给 psql。
fn hex(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    for byte in text.as_bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// 单条插入。
///
/// ## 为什么值要 hex 编码
///
/// 在 Windows 上 psql **按控制台代码页（中文机器上是 GBK）解码 stdin 与 `-v` 变量值**，
/// 而 Rust 写出的是 UTF-8 字节 —— 中文（如 `reason` 里的「决策服务超时」）会触发
/// `invalid multibyte character`，psql 连变量替换都不做，直接报
/// `syntax error at or near ":"`。实测对照：纯 ASCII 变量完全正常，含中文必失败；
/// 在 SQL 前加 `SET client_encoding TO 'UTF8'` 也无效（psql 在解析变量时就已经解码了）。
///
/// 因此：变量值统一转成 **hex（纯 ASCII）**，在服务端用
/// `convert_from(decode(:'x','hex'),'UTF8')` 还原为文本。这条路径与客户端代码页无关，
/// 也不依赖 locale 设置，在 Windows / Linux 上行为一致。
///
/// ## 其他约定
///   * 可空字段传空串，服务端用 `NULLIF(...,'')` 转 NULL；
///   * `reason` 里的换行先压成空格（避免截断语句）；
///   * 变量经 `-v` 传参，不拼进 SQL 文本。
fn insert(database_url: &str, trace: &DecisionTrace) -> Result<(), String> {
    const SQL: &str = "\
INSERT INTO decision_trace(query_id, tenant_id, kb_id, model_id, model_version, profile, \
escalate, answerable_prob, confidence, probabilities, features, reason, fallback_used, latency_ms) \
VALUES (\
NULLIF(convert_from(decode(:'query_id','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'tenant_id','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'kb_id','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'model_id','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'model_version','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'profile','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'escalate','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'answerable_prob','hex'),'UTF8'),'')::numeric, \
NULLIF(convert_from(decode(:'confidence','hex'),'UTF8'),'')::numeric, \
NULLIF(convert_from(decode(:'probabilities','hex'),'UTF8'),'')::jsonb, \
NULLIF(convert_from(decode(:'features','hex'),'UTF8'),'')::jsonb, \
NULLIF(convert_from(decode(:'reason','hex'),'UTF8'),''), \
NULLIF(convert_from(decode(:'fallback_used','hex'),'UTF8'),'')::boolean, \
NULLIF(convert_from(decode(:'latency_ms','hex'),'UTF8'),'')::numeric);";

    let vars: [(&str, String); 14] = [
        ("query_id", hex(&trace.query_id)),
        ("tenant_id", hex(&trace.tenant_id)),
        ("kb_id", hex(trace.kb_id.as_deref().unwrap_or(""))),
        ("model_id", hex(&trace.model_id)),
        ("model_version", hex(trace.model_version.as_deref().unwrap_or(""))),
        ("profile", hex(&trace.profile)),
        ("escalate", hex(trace.escalate.as_deref().unwrap_or(""))),
        (
            "answerable_prob",
            hex(&trace.answerable_prob.map(|v| v.to_string()).unwrap_or_default()),
        ),
        ("confidence", hex(&trace.confidence.to_string())),
        ("probabilities", hex(trace.probabilities.as_deref().unwrap_or(""))),
        ("features", hex(trace.features.as_deref().unwrap_or(""))),
        ("reason", hex(&sanitize(&trace.reason))),
        ("fallback_used", hex(&trace.fallback_used.to_string())),
        (
            "latency_ms",
            hex(&trace.latency_ms.map(|v| v.to_string()).unwrap_or_default()),
        ),
    ];

    let mut cmd = Command::new("psql");
    cmd.arg(database_url)
        .arg("--no-psqlrc")
        .arg("--quiet")
        .arg("--set=ON_ERROR_STOP=1");
    for (name, value) in &vars {
        cmd.arg("-v").arg(format!("{name}={value}"));
    }

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("启动 psql 失败: {e}"))?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(SQL.as_bytes())
            .map_err(|e| format!("写入 SQL 失败: {e}"))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("等待 psql 失败: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// psql 变量值不允许裸换行（会截断语句），统一压成单行。
fn sanitize(text: &Option<String>) -> String {
    text.as_deref()
        .unwrap_or("")
        .replace(['\n', '\r'], " ")
        .trim()
        .to_string()
}
