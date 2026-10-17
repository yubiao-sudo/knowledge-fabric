//! 统一错误类型 → 结构化错误响应（含 code / message）+ HTTP 状态码。
//!
//! 设计原则：下游（决策 / 检索）故障一律就地降级、不抛错；网关自身只在
//! 「入参非法」与「内部异常」两类情况下返回错误。

use std::fmt;

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug)]
pub enum GatewayError {
    /// 入参非法（HTTP 400）。
    InvalidRequest(String),
    /// 网关内部异常（HTTP 500）。
    Internal(String),
}

impl GatewayError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        GatewayError::InvalidRequest(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        GatewayError::Internal(message.into())
    }

    pub fn code(&self) -> &'static str {
        match self {
            GatewayError::InvalidRequest(_) => "invalid_request",
            GatewayError::Internal(_) => "internal_error",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            GatewayError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            GatewayError::InvalidRequest(m) | GatewayError::Internal(m) => m,
        }
    }
}

impl fmt::Display for GatewayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

impl std::error::Error for GatewayError {}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    code: &'static str,
    message: String,
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let status = self.status();
        let code = self.code();
        let message = self.message().to_string();
        if status.is_server_error() {
            tracing::error!(code, message, "网关内部错误");
        } else {
            tracing::warn!(code, message, "请求参数错误");
        }
        let body = ErrorEnvelope {
            error: ErrorDetail { code, message },
        };
        (status, Json(body)).into_response()
    }
}

impl From<JsonRejection> for GatewayError {
    fn from(rejection: JsonRejection) -> Self {
        GatewayError::invalid_request(format!("请求体不是合法 JSON：{}", rejection.body_text()))
    }
}
