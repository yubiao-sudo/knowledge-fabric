//! Knowledge Fabric 网关
//!
//! 对外暴露两种接入形态，目标是「任意 Agent 产品一键接入」：
//!   - MCP（首选）：JSON-RPC 2.0，Tools / Resources，支持 Streamable HTTP
//!   - REST：/v1/retrieve、/v1/ingest、/v1/kb/{id}
//!
//! 默认输出模式为 evidence-only：本服务是检索底座，把证据交给 agent 的 LLM 组织生成。
//!
//! 环境变量：DECISION_URL / DECISION_TIMEOUT_MS / RETRIEVAL_URL / GATEWAY_PORT / DEFAULT_TENANT。

mod clients;
mod error;
mod handlers;
mod profiles;
mod state;
mod trace;
mod types;

use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::state::{AppState, Budget};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "kf_gateway=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let decision_url = env_or("DECISION_URL", "http://localhost:8090");
    let retrieval_url = std::env::var("RETRIEVAL_URL")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let default_tenant = env_or("DEFAULT_TENANT", "default");
    let port = env_or("GATEWAY_PORT", "8000");
    let decision_timeout_ms = env_parse(
        "DECISION_TIMEOUT_MS",
        clients::decision::DECISION_TIMEOUT_MS,
    );

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    let state = Arc::new(AppState {
        http,
        decision_url: decision_url.clone(),
        retrieval_url: retrieval_url.clone(),
        default_tenant: default_tenant.clone(),
        safe_default_profile: profiles::SAFE_DEFAULT_PROFILE.to_string(),
        decision_timeout: Duration::from_millis(decision_timeout_ms),
        // 决策 trace 落库：未配置 DATABASE_URL / 无 psql 时自动降级为空写入器，
        // 绝不因为可观测性影响检索可用性。
        trace: trace::start(std::env::var("DATABASE_URL").ok(), 512),
        budget: Budget {
            max_latency_ms: profiles::MAX_LATENCY_MS,
            max_context_tokens: profiles::MAX_CONTEXT_TOKENS,
            max_agentic_rounds: profiles::MAX_AGENTIC_ROUNDS,
        },
    });

    let app = Router::new()
        .route("/health", get(handlers::health::handle))
        .route("/v1/retrieve", post(handlers::retrieve::handle))
        .route("/v1/ingest", post(handlers::ingest::handle))
        .route("/v1/kb/:kb_id", get(handlers::kb::handle))
        .route("/mcp", post(handlers::mcp::handle))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(
        %addr,
        decision_url = %decision_url,
        retrieval_configured = retrieval_url.is_some(),
        default_tenant = %default_tenant,
        safe_default = %profiles::SAFE_DEFAULT_PROFILE,
        "kf-gateway 启动"
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("收到退出信号，开始优雅关闭");
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_parse(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}
