//! `GET /v1/kb/{id}`：返回知识库元信息。
//!
//! 当前为占位实现，结构对齐 db/001_init.sql 的 `knowledge_bases` 表；
//! 接入元数据库后替换字段来源即可，对外结构保持不变。

use axum::extract::Path;
use axum::Json;

use crate::error::GatewayError;
use crate::profiles;
use crate::types::{KbInfo, KbStats};

pub async fn handle(Path(kb_id): Path<String>) -> Result<Json<KbInfo>, GatewayError> {
    let kb_id = kb_id.trim().to_string();
    if kb_id.is_empty() {
        return Err(GatewayError::invalid_request("kb_id 不能为空"));
    }

    // 占位：真实实现读取 knowledge_bases（name / languages / stats / retrieval_profile / graph_enabled）。
    Ok(Json(KbInfo {
        name: format!("kb-{kb_id}"),
        kb_id,
        tenant_id: None,
        status: "not_provisioned".to_string(),
        chunking_strategy: "doc_structured".to_string(),
        retrieval_profile: profiles::SAFE_DEFAULT_PROFILE.to_string(),
        graph_enabled: false,
        languages: Vec::new(),
        stats: KbStats {
            documents: 0,
            chunks: 0,
            last_ingested_at: None,
        },
        note: "占位结构：接入元数据库后返回真实元信息".to_string(),
    }))
}
