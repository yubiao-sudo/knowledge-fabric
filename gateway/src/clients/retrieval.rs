//! 检索链路调用点。
//!
//! 检索链路（dense+sparse 混合召回 → RRF → TEI 重排 → 组装）尚未实现：
//! 本模块是与下游检索服务的对接点，并保证「检索不可用绝不阻断网关」的优雅降级。
//! 期望下游契约：`POST {RETRIEVAL_URL}/v1/search`
//!   请求：{ profile, query, tenant, kb, top_k, max_context_tokens }
//!   响应：{ evidence: [...], citations: [...], rerank_ms }
//!   其中 evidence / citations 条目分别对齐 response.schema.json。

use std::time::{Duration, Instant};

use serde_json::Value;
use tracing::info;

use crate::state::AppState;
use crate::types::{Citation, Evidence};

/// 检索结果：证据 + 引用 + 分阶段耗时。
#[derive(Debug, Clone, Default)]
pub struct RetrievalOutcome {
    /// 下游检索链路是否可用（false 表示已降级为「无证据」）。
    pub available: bool,
    pub evidence: Vec<Evidence>,
    pub citations: Vec<Citation>,
    pub retrieve_ms: f64,
    pub rerank_ms: f64,
    /// 降级原因（用于 grounding.refuse_reason 溯源）。
    pub note: Option<String>,
}

impl RetrievalOutcome {
    /// 构造降级结果：不报错、不阻断，交由上层 grounding 拒答。
    pub fn unavailable(note: impl Into<String>, retrieve_ms: f64) -> Self {
        Self {
            available: false,
            note: Some(note.into()),
            retrieve_ms,
            ..Default::default()
        }
    }
}

/// 按档位执行检索；超时 / 不可用时返回降级结果。
///
/// `principal` / `graph_enabled` 由网关**原样转发**给下游 `/v1/search`，
/// 网关自身不做任何 ACL 判定（判定在检索侧 assemble 阶段完成）。
pub async fn run(
    state: &AppState,
    profile: &str,
    query: &str,
    tenant: &str,
    kb: Option<&str>,
    top_k: Option<u32>,
    principal: Option<&Value>,
    graph_enabled: Option<bool>,
    timeout: Duration,
) -> RetrievalOutcome {
    let started = Instant::now();

    let Some(base) = state.retrieval_url.as_deref() else {
        return RetrievalOutcome::unavailable(
            "检索链路未接入（未配置 RETRIEVAL_URL）",
            elapsed_ms(started),
        );
    };

    let body = build_search_body(
        profile,
        query,
        tenant,
        kb,
        top_k,
        state.budget.max_context_tokens,
        principal,
        graph_enabled,
    );
    let url = format!("{}/v1/search", base.trim_end_matches('/'));

    match state
        .http
        .post(&url)
        .timeout(timeout)
        .json(&body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
            Ok(value) => from_service(&value, started),
            Err(err) => RetrievalOutcome::unavailable(
                format!("检索响应解析失败: {err}"),
                elapsed_ms(started),
            ),
        },
        Ok(resp) => RetrievalOutcome::unavailable(
            format!("检索服务返回 HTTP {}", resp.status()),
            elapsed_ms(started),
        ),
        Err(err) if err.is_timeout() => RetrievalOutcome::unavailable(
            "检索服务超时（档位超时 / 预算受限）",
            elapsed_ms(started),
        ),
        Err(err) => {
            RetrievalOutcome::unavailable(format!("检索服务不可用: {err}"), elapsed_ms(started))
        }
    }
}

/// 构造下游 `/v1/search` 请求体。
///
/// `principal` 与 `graph_enabled` **原样转发**：显式传入才写入 body，缺省则不出现在
/// body 中（保留下游默认语义）。网关不做任何 ACL 判定。
#[allow(clippy::too_many_arguments)]
fn build_search_body(
    profile: &str,
    query: &str,
    tenant: &str,
    kb: Option<&str>,
    top_k: Option<u32>,
    max_context_tokens: u64,
    principal: Option<&Value>,
    graph_enabled: Option<bool>,
) -> Value {
    let mut body = serde_json::json!({
        "profile": profile,
        "query": query,
        "tenant": tenant,
        "kb": kb,
        "top_k": top_k,
        "max_context_tokens": max_context_tokens,
    });
    if let Some(principal) = principal {
        body["principal"] = principal.clone();
    }
    if let Some(enabled) = graph_enabled {
        body["graph_enabled"] = Value::Bool(enabled);
    }
    body
}

fn from_service(value: &Value, started: Instant) -> RetrievalOutcome {
    let evidence: Vec<Evidence> = parse_items(value.get("evidence"));
    let citations: Vec<Citation> = parse_items(value.get("citations"));
    let rerank_ms = value
        .get("rerank_ms")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let retrieve_ms = (elapsed_ms(started) - rerank_ms).max(0.0);
    info!(
        evidence = evidence.len(),
        citations = citations.len(),
        "检索链路返回"
    );
    RetrievalOutcome {
        available: true,
        evidence,
        citations,
        retrieve_ms,
        rerank_ms,
        note: None,
    }
}

/// 逐条宽松解析：单条非法不影响整体（抵御下游字段漂移）。
fn parse_items<T: serde::de::DeserializeOwned>(value: Option<&Value>) -> Vec<T> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| serde_json::from_value(item.clone()).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

// ---------------------------------------------------------------- 测试

#[cfg(test)]
mod tests {
    use super::build_search_body;
    use serde_json::json;

    #[test]
    fn forwards_principal_and_graph_enabled_verbatim() {
        let principal = json!({"user": "u1", "groups": ["g1"], "deny": ["u9"]});
        let body = build_search_body(
            "hybrid_rerank",
            "MX-200 的价格",
            "t1",
            None,
            Some(5),
            8000,
            Some(&principal),
            Some(true),
        );
        // principal 必须原样出现（网关不得改写 / 裁剪）
        assert_eq!(body["principal"], principal);
        assert_eq!(body["graph_enabled"], json!(true));
        assert_eq!(body["profile"], json!("hybrid_rerank"));
        assert_eq!(body["max_context_tokens"], json!(8000));
    }

    #[test]
    fn omits_acl_fields_when_absent() {
        let body =
            build_search_body("naive", "q", "t1", None, None, 8000, None, None);
        assert!(body.get("principal").is_none());
        assert!(body.get("graph_enabled").is_none());
    }

    #[test]
    fn graph_enabled_false_is_forwarded_not_dropped() {
        let body =
            build_search_body("graph_global", "汇总", "t1", None, None, 8000, None, Some(false));
        assert_eq!(body["graph_enabled"], json!(false));
    }
}
