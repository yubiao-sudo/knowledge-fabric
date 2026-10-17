//! `POST /mcp`：MCP JSON-RPC 2.0（Streamable HTTP）。
//!
//! 支持 `initialize` / `tools/list` / `tools/call` / `resources/list`；其中 `tools/call`
//! 真正路由到与 REST 同一套 handler 逻辑（retrieve / ingest），而非返回空壳。
//! 未知 method 返回 JSON-RPC 错误 -32601。

use std::sync::Arc;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::error::GatewayError;
use crate::handlers::{ingest, retrieve};
use crate::state::AppState;
use crate::types::{IngestRequest, RetrieveRequest};

/// MCP 协议版本（Streamable HTTP）。
const PROTOCOL_VERSION: &str = "2024-11-05";

// JSON-RPC 标准错误码。
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

/// MCP 入口（Streamable HTTP：单次 JSON-RPC 请求 → 单次 JSON 响应）。
pub async fn handle(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Json<Value> {
    match payload {
        Ok(Json(body)) => Json(dispatch(&state, body).await),
        Err(err) => Json(rpc_error(
            Value::Null,
            PARSE_ERROR,
            &format!("请求体解析失败: {}", err.body_text()),
        )),
    }
}

async fn dispatch(state: &AppState, body: Value) -> Value {
    let id = body.get("id").cloned().unwrap_or(Value::Null);
    let Some(method) = body.get("method").and_then(Value::as_str) else {
        return rpc_error(id, INVALID_REQUEST, "无效请求：缺少 method");
    };

    match method {
        "initialize" => rpc_result(id, initialize_result()),
        "tools/list" => rpc_result(id, json!({ "tools": tools() })),
        "tools/call" => match call_tool(state, &body).await {
            Ok(result) => rpc_result(id, result),
            Err((code, message)) => rpc_error(id, code, &message),
        },
        // 资源目录：接入元数据库后列出各 KB（resource uri 形如 kf://kb/{id}）。
        "resources/list" => rpc_result(id, json!({ "resources": [] })),
        "ping" => rpc_result(id, json!({})),
        _ => rpc_error(id, METHOD_NOT_FOUND, &format!("method not found: {method}")),
    }
}

/// `tools/call`：按 name 路由到真实 handler 逻辑。
async fn call_tool(state: &AppState, body: &Value) -> Result<Value, (i64, String)> {
    let params = body.get("params").cloned().unwrap_or(Value::Null);
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| (INVALID_PARAMS, "tools/call 缺少 params.name".to_string()))?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    match name {
        "retrieve" => {
            let req: RetrieveRequest = serde_json::from_value(arguments)
                .map_err(|e| (INVALID_PARAMS, format!("retrieve 参数非法: {e}")))?;
            let resp = retrieve::run(state, req).await.map_err(to_rpc)?;
            let value = serde_json::to_value(&resp)
                .map_err(|e| to_rpc(GatewayError::internal(e.to_string())))?;
            Ok(tool_result(value))
        }
        "ingest" => {
            let req: IngestRequest = serde_json::from_value(arguments)
                .map_err(|e| (INVALID_PARAMS, format!("ingest 参数非法: {e}")))?;
            let resp = ingest::run(state, req).map_err(to_rpc)?;
            let value = serde_json::to_value(&resp)
                .map_err(|e| to_rpc(GatewayError::internal(e.to_string())))?;
            Ok(tool_result(value))
        }
        other => Err((INVALID_PARAMS, format!("未知工具: {other}"))),
    }
}

/// 包装为 MCP tools/call 结果（content + structuredContent）。
fn tool_result(value: Value) -> Value {
    let text = serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string());
    json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": value,
        "isError": false
    })
}

fn to_rpc(err: GatewayError) -> (i64, String) {
    match err {
        GatewayError::InvalidRequest(message) => (INVALID_PARAMS, message),
        other => (INTERNAL_ERROR, other.to_string()),
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {
            "name": "knowledge-fabric-gateway",
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "subscribe": false, "listChanged": false }
        }
    })
}

fn tools() -> Value {
    json!([
        {
            "name": "retrieve",
            "description": "在企业知识库中检索证据（evidence-only，不生成答案）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "检索问题" },
                    "tenant": { "type": "string", "description": "租户，缺省 default" },
                    "kb": { "type": "string", "description": "知识库 id，缺省由决策路由" },
                    "top_k": { "type": "integer", "description": "可选召回上限" },
                    "principal": {
                        "type": ["object", "array", "string"],
                        "description": "调用主体（任意 JSON，如 {\"user\":\"u1\"}）；网关原样转发给检索侧做 ACL 判定，缺省不过滤"
                    },
                    "graph_enabled": { "type": "boolean", "description": "是否启用 GraphRAG 通道，缺省 false" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "ingest",
            "description": "投递文档以构建 / 更新企业知识库",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tenant": { "type": "string" },
                    "kb": { "type": "string" },
                    "source_uri": { "type": "string", "description": "对象存储路径或上传凭据" },
                    "doc_type": { "type": "string" },
                    "lang": { "type": "string", "description": "BCP-47 语言标签" }
                },
                "required": ["tenant", "kb", "source_uri"]
            }
        }
    ])
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}
