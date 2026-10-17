//! 网关共享状态。

use std::time::Duration;

/// 全局预算硬限（对齐 configs/retrieval_profiles.yaml → global_budget）。
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// 单查询端到端时延硬上限（ms）：超限中断并降级返回已获得证据。
    pub max_latency_ms: u64,
    /// 送入 agent 的证据 token 预算。
    pub max_context_tokens: u64,
    /// Agentic 迭代硬上限。
    pub max_agentic_rounds: u32,
}

#[derive(Clone)]
pub struct AppState {
    /// 复用连接池的出站 HTTP 客户端。
    pub http: reqwest::Client,
    /// 决策服务基址（Laya）。
    pub decision_url: String,
    /// 检索链路服务基址；链路尚未实现，未配置时优雅降级（见 clients/retrieval.rs）。
    pub retrieval_url: Option<String>,
    /// 缺省租户。
    pub default_tenant: String,
    /// 安全默认档（见 profiles.rs）。
    pub safe_default_profile: String,
    /// 决策调用硬超时。
    pub decision_timeout: Duration,
    /// 决策 trace 写入器（未配置数据库时为空写入器，检索路径不受影响）。
    pub trace: crate::trace::TraceWriter,
    /// 全局预算。
    pub budget: Budget,
}
