//! `GET /health`：网关自身状态 + 决策服务可达性。
//!
//! 始终返回 200：决策服务不可达是预期情况（网关自身健康，靠安全回退兜底）。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::Json;

use crate::state::AppState;
use crate::types::{GatewayInfo, HealthResponse, RetrievalHealth, UpstreamHealth};

pub async fn handle(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let url = format!("{}/health", state.decision_url.trim_end_matches('/'));
    let started = Instant::now();

    let (reachable, detail) = match state
        .http
        .get(&url)
        .timeout(state.decision_timeout)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => (true, None),
        Ok(resp) => (false, Some(format!("HTTP {}", resp.status()))),
        Err(err) if err.is_timeout() => (false, Some("探测超时".to_string())),
        Err(err) => (false, Some(format!("不可达: {err}"))),
    };
    let latency_ms = Some(started.elapsed().as_secs_f64() * 1000.0);

    Json(HealthResponse {
        status: "ok".to_string(),
        gateway: GatewayInfo {
            name: "kf-gateway".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        decision_service: UpstreamHealth {
            reachable,
            url,
            latency_ms,
            detail,
        },
        retrieval_service: RetrievalHealth {
            configured: state.retrieval_url.is_some(),
            url: state.retrieval_url.clone(),
        },
        safe_default_profile: state.safe_default_profile.clone(),
        decision_trace: state.trace.is_enabled(),
    })
}
