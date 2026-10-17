//! `POST /v1/ingest`：校验入参 → 生成 job_id → 返回 queued。
//!
//! 真正的入库由 `services/ingest` 的消费者完成（解析 → 清洗 → 切片 → 增强 → 嵌入 → 索引）。
//! 投递点：写入 Postgres `ingestion_jobs`（见 db/001_init.sql）并投递 Redis 队列；
//! 消费端就绪后在此处补投递即可，对外契约不变。

use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::Json;
use tracing::info;

use crate::error::GatewayError;
use crate::state::AppState;
use crate::types::{IngestRequest, IngestResponse};

/// REST 入口。
pub async fn handle(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<IngestRequest>, JsonRejection>,
) -> Result<Json<IngestResponse>, GatewayError> {
    let Json(req) = payload?;
    Ok(Json(run(&state, req)?))
}

/// 摄入受理逻辑（REST 与 MCP `tools/call` 共用）。
/// `state` 预留：投递队列 / DB 需要其中的连接与配置信息。
pub fn run(_state: &AppState, req: IngestRequest) -> Result<IngestResponse, GatewayError> {
    if req.tenant.trim().is_empty() {
        return Err(GatewayError::invalid_request("tenant 不能为空"));
    }
    if req.kb.trim().is_empty() {
        return Err(GatewayError::invalid_request("kb 不能为空"));
    }
    if req.source_uri.trim().is_empty() {
        return Err(GatewayError::invalid_request("source_uri 不能为空"));
    }

    let job_id = uuid::Uuid::new_v4().simple().to_string();
    info!(
        job_id = %job_id,
        tenant = %req.tenant,
        kb = %req.kb,
        source_uri = %req.source_uri,
        "已受理摄入作业（待投递队列）"
    );

    Ok(IngestResponse {
        job_id,
        status: "queued".to_string(),
        stage: "queued".to_string(),
        tenant: req.tenant,
        kb: req.kb,
        source_uri: req.source_uri,
        doc_type: req.doc_type,
        lang: req.lang,
        note: "已排队；由 services/ingest 消费者执行 解析→清洗→切片→增强→嵌入→索引".to_string(),
    })
}
