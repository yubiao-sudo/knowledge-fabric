//! 检索档位与全局预算常量。
//!
//! 权威来源：`configs/retrieval_profiles.yaml`（8 个档位名 + global_budget 硬限）。
//! 为保证网关不引入额外 YAML 依赖，这里以常量镜像该文件；两处必须保持一致。

/// 合法检索档位（与 retrieval_profiles.yaml 的 profiles 键、decision.schema.json 的枚举一致）。
pub const PROFILES: [&str; 8] = [
    "naive",
    "hybrid_rerank",
    "exact_lookup",
    "hyde",
    "multi_query",
    "decompose_agentic",
    "graph_global",
    "strict_refuse",
];

/// 安全默认档：决策层故障时一律回退到此档（retrieval_profiles.yaml → safe_default）。
pub const SAFE_DEFAULT_PROFILE: &str = "hybrid_rerank";

/// 决策护栏允许的算力级联档（decision.schema.json → escalate 枚举）。
pub const ESCALATIONS: [&str; 4] = ["none", "rerank", "llm_rerank", "agentic"];

/// 全局预算：单查询端到端硬上限（ms）。
pub const MAX_LATENCY_MS: u64 = 3000;
/// 全局预算：送入 agent 的证据 token 预算。
pub const MAX_CONTEXT_TOKENS: u64 = 8000;
/// 全局预算：Agentic 迭代硬上限。
pub const MAX_AGENTIC_ROUNDS: u32 = 3;
/// 拒答阈值（与决策服务 ANSWERABLE_FLOOR 对齐）。
pub const ANSWERABLE_FLOOR: f64 = 0.55;

/// profile 是否合法。
pub fn is_valid_profile(profile: &str) -> bool {
    PROFILES.contains(&profile)
}

/// 各档位的单次检索超时（ms，见 retrieval_profiles.yaml → profiles.*.timeout_ms）。
/// 实际执行时还会被全局 `max_latency_ms` 二次收敛。
pub fn profile_timeout_ms(profile: &str) -> u64 {
    match profile {
        "naive" => 800,
        "hybrid_rerank" => 1500,
        "exact_lookup" => 1200,
        "hyde" => 2000,
        "multi_query" => 2500,
        "decompose_agentic" => 6000,
        "graph_global" => 8000,
        "strict_refuse" => 1500,
        _ => 1500,
    }
}
