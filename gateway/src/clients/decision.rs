//! 决策服务（Laya）调用 + 安全回退护栏。
//!
//! 护栏：决策层故障（不可用 / 超时 / 返回非法 profile）绝不能中断检索，
//! 一律回退安全默认档 `hybrid_rerank`，并在 decision 中标记 `fallback_used = true`。

use std::time::Instant;

use tracing::warn;

use crate::profiles::{self, SAFE_DEFAULT_PROFILE};
use crate::state::AppState;
use crate::types::{DecideRequest, Decision, ModelInfo, QueryFeatures, RetrieveRequest};

/// 决策调用硬超时的**缺省值**（ms）。可用环境变量 `DECISION_TIMEOUT_MS` 覆盖；
/// 此处只是未配置时的兜底，运行期实际值在 `AppState.decision_timeout`。
pub const DECISION_TIMEOUT_MS: u64 = 300;

/// 调用决策服务获取检索档位；任何异常都返回安全回退决策。
pub async fn decide(state: &AppState, query_id: &str, req: &RetrieveRequest) -> Decision {
    let features = QueryFeatures {
        // 真实实现由查询理解阶段填充：语言检测 / 长度 / 型号标识 / 数字价格等
        lang: None,
        length: req.query.chars().count(),
        has_code_or_identifier: looks_like_identifier(&req.query),
        has_number_or_price: req.query.chars().any(|c| c.is_ascii_digit()),
        kb_scale: 0,
        session_turn: 0,
        historical_no_answer_rate: 0.0,
        kb_languages: Vec::new(),
        // 由调用方显式声明的图通道开关透传给决策侧（build_graph 依据此判定）；
        // 未声明时保持既有行为（false）。
        graph_enabled: req.graph_enabled.unwrap_or(false),
    };
    let body = DecideRequest {
        query_id: query_id.to_string(),
        query: req.query.clone(),
        tenant_id: req.tenant.clone(),
        kb_id: req.kb.clone(),
        features,
    };

    let url = format!("{}/decide", state.decision_url.trim_end_matches('/'));
    let started = Instant::now();
    // 运行期实际超时（毫秒）——用于错误信息。**不能用常量**：那会在
    // DECISION_TIMEOUT_MS 配成 5000 时仍然报"超时 > 300ms"，是假信息。
    let budget_ms = state.decision_timeout.as_millis();

    match state
        .http
        .post(&url)
        .timeout(state.decision_timeout)
        .json(&body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => match resp.json::<Decision>().await {
            Ok(mut decision) => {
                if let Some(reason) = validation_error(&decision) {
                    return fallback(query_id, &reason, started);
                }
                // escalate 非法时就地规整（profile 才是路由关键，不因此整体回退）
                if !profiles::ESCALATIONS.contains(&decision.escalate.as_str()) {
                    decision.escalate = "rerank".to_string();
                }
                // Agentic 轮数按全局硬限收敛
                if let Some(agentic) = decision.agentic.as_mut() {
                    if agentic.max_rounds.unwrap_or(0) > profiles::MAX_AGENTIC_ROUNDS {
                        agentic.max_rounds = Some(profiles::MAX_AGENTIC_ROUNDS);
                    }
                }
                if decision.fallback_used.is_none() {
                    decision.fallback_used = Some(false);
                }
                decision
            }
            Err(err) => fallback(
                query_id,
                &format!("决策响应无法解析为 Decision 契约: {err}"),
                started,
            ),
        },
        Ok(resp) => fallback(
            query_id,
            &format!("决策服务返回 HTTP {}", resp.status()),
            started,
        ),
        Err(err) if err.is_timeout() => fallback(
            query_id,
            &format!("决策服务超时（> {budget_ms}ms）"),
            started,
        ),
        Err(err) => fallback(query_id, &format!("决策服务不可用: {err}"), started),
    }
}

/// 契约校验：校验失败即视为非法输出，触发回退。
fn validation_error(decision: &Decision) -> Option<String> {
    if !profiles::is_valid_profile(&decision.profile) {
        return Some(format!("决策返回非法 profile: {}", decision.profile));
    }
    if decision.model.id.trim().is_empty() {
        return Some("决策 model.id 为空".to_string());
    }
    if decision.trace_id.trim().is_empty() {
        return Some("决策缺少 trace_id".to_string());
    }
    if !(0.0..=1.0).contains(&decision.confidence) {
        return Some(format!("决策 confidence 越界: {}", decision.confidence));
    }
    None
}

/// 安全回退决策：profile = safe_default，fallback_used = true。
fn fallback(query_id: &str, reason: &str, started: Instant) -> Decision {
    warn!(query_id, reason, "决策回退安全默认档");
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    Decision {
        query_id: query_id.to_string(),
        trace_id: uuid::Uuid::new_v4().simple().to_string(),
        model: ModelInfo {
            id: "gateway-safe-default".to_string(),
            version: "0".to_string(),
            params: None,
            latency_ms: Some(latency_ms),
        },
        profile: SAFE_DEFAULT_PROFILE.to_string(),
        escalate: "rerank".to_string(),
        agentic: None,
        graph: None,
        answerable_prob: Some(1.0),
        confidence: 0.0,
        probabilities: None,
        calibration: None,
        features: None,
        reason: Some(format!("{reason}；已回退安全默认档 {SAFE_DEFAULT_PROFILE}")),
        fallback_used: Some(true),
    }
}

/// 粗判查询是否含型号 / 标识符（真实实现应在查询理解阶段完成）。
fn looks_like_identifier(query: &str) -> bool {
    let has_upper = query.chars().any(|c| c.is_ascii_uppercase());
    let has_digit = query.chars().any(|c| c.is_ascii_digit());
    let has_separator = query.contains('-') || query.contains('_') || query.contains('/');
    (has_upper && has_digit) || has_separator || query.to_ascii_lowercase().contains("sku")
}
