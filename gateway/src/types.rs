//! 请求 / 响应 DTO。
//!
//! 字段名与枚举严格对齐权威契约：
//!   - `schemas/response.schema.json`（对外响应）
//!   - `schemas/decision.schema.json`（决策结果）
//! 可选字段用 `Option` + `skip_serializing_if`，缺省不输出，符合 schema 的 optional 语义。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------- 对外请求

/// `POST /v1/retrieve` 请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct RetrieveRequest {
    pub query: String,
    #[serde(default = "default_tenant")]
    pub tenant: String,
    /// 不传则由决策模型路由到租户下的全部可见知识库。
    #[serde(default)]
    pub kb: Option<String>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// 调用主体（任意 JSON，如 `{"user":"u1","groups":["g1"],"deny":["u9"]}`）。
    /// 网关**不做** ACL 判定，只把该字段原样转发给检索服务 `/v1/search`，由检索侧
    /// 执行 chunk 级 ACL 过滤（deny 优先）；缺省不传即不过滤。
    #[serde(default)]
    pub principal: Option<Value>,
    /// 是否启用 GraphRAG 通道：原样转发给检索服务，并作为决策特征
    /// `features.graph_enabled` 供 `build_graph` 判定；缺省视为未启用。
    #[serde(default)]
    pub graph_enabled: Option<bool>,
}

fn default_tenant() -> String {
    "default".to_string()
}

/// `POST /v1/ingest` 请求体。
#[derive(Debug, Clone, Deserialize)]
pub struct IngestRequest {
    pub tenant: String,
    pub kb: String,
    /// 对象存储路径或上传凭据。
    pub source_uri: String,
    #[serde(default)]
    pub doc_type: Option<String>,
    #[serde(default)]
    pub lang: Option<String>,
}

/// `POST /v1/ingest` 响应体：受理凭证（真正的入库由 services/ingest 消费者完成）。
#[derive(Debug, Clone, Serialize)]
pub struct IngestResponse {
    pub job_id: String,
    /// queued / running / done / failed
    pub status: String,
    pub stage: String,
    pub tenant: String,
    pub kb: String,
    pub source_uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    pub note: String,
}

/// `GET /v1/kb/{id}` 响应体：知识库元信息（结构对齐 knowledge_bases 表）。
#[derive(Debug, Clone, Serialize)]
pub struct KbInfo {
    pub kb_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    pub name: String,
    pub status: String,
    pub chunking_strategy: String,
    pub retrieval_profile: String,
    pub graph_enabled: bool,
    pub languages: Vec<String>,
    pub stats: KbStats,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct KbStats {
    pub documents: u64,
    pub chunks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_ingested_at: Option<String>,
}

// ---------------------------------------------------------------- 决策契约

/// 网关 → 决策服务的请求体（对齐 services/decision/app.py → DecideRequest）。
#[derive(Debug, Clone, Serialize)]
pub struct DecideRequest {
    pub query_id: String,
    pub query: String,
    pub tenant_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kb_id: Option<String>,
    pub features: QueryFeatures,
}

/// 决策输入特征（用于线上增量训练）。
#[derive(Debug, Clone, Serialize)]
pub struct QueryFeatures {
    pub lang: Option<String>,
    pub length: usize,
    pub has_code_or_identifier: bool,
    pub has_number_or_price: bool,
    pub kb_scale: u64,
    pub session_turn: u32,
    pub historical_no_answer_rate: f64,
    pub kb_languages: Vec<String>,
    pub graph_enabled: bool,
}

/// 决策结果（对齐 decision.schema.json）。
///
/// 必填字段未做 `default`，任何缺失都会导致反序列化失败 → 网关侧安全回退。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub query_id: String,
    pub trace_id: String,
    pub model: ModelInfo,
    /// 选中的检索档位（校验后再使用，非法值触发回退）。
    pub profile: String,
    /// 算力级联档：none / rerank / llm_rerank / agentic。
    pub escalate: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agentic: Option<Agentic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<GraphDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answerable_prob: Option<f64>,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_used: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agentic {
    #[serde(default)]
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rounds: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDecision {
    #[serde(default)]
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_hops: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ece: Option<f64>,
}

// ---------------------------------------------------------------- 对外响应

/// `POST /v1/retrieve` 响应体（对齐 response.schema.json）。
#[derive(Debug, Clone, Serialize)]
pub struct RetrieveResponse {
    pub query_id: String,
    /// evidence-only / grounded-answer / agentic-answer / graph-global。
    pub mode: String,
    pub decision: Decision,
    pub evidence: Vec<Evidence>,
    pub citations: Vec<Citation>,
    pub grounding: Grounding,
    pub trace: Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub chunk_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_chunk_id: Option<String>,
    pub content: String,
    /// 重排后最终分。
    pub score: f64,
    pub source: String,
    pub page: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_type: Option<String>,
    /// dense / sparse / colbert / structured / graph。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_channel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub doc_id: String,
    pub page: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// provenance 区间，支持点击回原文。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub char_range: Option<Vec<i64>>,
}

/// Grounding 校验结果；不通过则拒答。
#[derive(Debug, Clone, Serialize, Default)]
pub struct Grounding {
    pub passed: bool,
    pub score: f64,
    pub refused: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refuse_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Trace {
    pub stages_ms: StagesMs,
    /// 实际执行的 Agentic 轮数（链路未接入时为 0）。
    pub agentic_rounds: u32,
    pub tokens_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_estimate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StagesMs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub understand: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decide: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieve: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assemble: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<f64>,
}

// ---------------------------------------------------------------- 健康检查

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub gateway: GatewayInfo,
    pub decision_service: UpstreamHealth,
    pub retrieval_service: RetrievalHealth,
    pub safe_default_profile: String,
    /// 决策 trace 落库是否启用（未配置 DATABASE_URL / 无 psql 时为 false）。
    /// 显式暴露，避免"以为 trace 在落库其实一条没写"。
    pub decision_trace: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GatewayInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpstreamHealth {
    pub reachable: bool,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalHealth {
    pub configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

// ---------------------------------------------------------------- 测试

#[cfg(test)]
mod tests {
    use super::RetrieveRequest;

    #[test]
    fn parses_principal_and_graph_enabled_verbatim() {
        let req: RetrieveRequest = serde_json::from_value(serde_json::json!({
            "query": "MX-200 的价格",
            "principal": {"user": "u1", "groups": ["g1"], "deny": ["u9"]},
            "graph_enabled": true,
        }))
        .expect("应能解析 principal / graph_enabled");
        let principal = req.principal.expect("principal 应被保留");
        assert_eq!(principal["user"], serde_json::json!("u1"));
        assert_eq!(principal["deny"], serde_json::json!(["u9"]));
        assert_eq!(req.graph_enabled, Some(true));
    }

    #[test]
    fn optional_acl_fields_default_to_none() {
        let req: RetrieveRequest =
            serde_json::from_value(serde_json::json!({ "query": "q" })).expect("最小请求应可解析");
        assert!(req.principal.is_none());
        assert!(req.graph_enabled.is_none());
    }
}
