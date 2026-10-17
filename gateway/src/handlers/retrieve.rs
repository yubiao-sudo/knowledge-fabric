//! `POST /v1/retrieve`：决策 → 检索 → 组装 evidence-only 响应。
//!
//! 关键护栏：
//!   1. 决策层故障回退安全默认档（由 clients/decision.rs 保证）；
//!   2. 全局时延预算 `max_latency_ms` 硬限，超限中断并返回已获得证据；
//!   3. 证据 token 预算 `max_context_tokens` 硬限，超限截断；
//!   4. Grounding 不通过则拒答（仍返回证据供 agent 判断）。

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::Json;
use tracing::{debug, info};

use crate::clients::decision;
use crate::clients::retrieval::{self, RetrievalOutcome};
use crate::error::GatewayError;
use crate::profiles::{self, ANSWERABLE_FLOOR};
use crate::state::AppState;
use crate::types::{
    Citation, Decision, Evidence, Grounding, RetrieveRequest, RetrieveResponse, StagesMs, Trace,
};

/// REST 入口。
pub async fn handle(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<RetrieveRequest>, JsonRejection>,
) -> Result<Json<RetrieveResponse>, GatewayError> {
    let Json(req) = payload?;
    Ok(Json(run(&state, req).await?))
}

/// 检索核心逻辑（REST 与 MCP `tools/call` 共用）。
pub async fn run(state: &AppState, req: RetrieveRequest) -> Result<RetrieveResponse, GatewayError> {
    if req.query.trim().is_empty() {
        return Err(GatewayError::invalid_request("query 不能为空"));
    }
    let tenant = if req.tenant.trim().is_empty() {
        state.default_tenant.clone()
    } else {
        req.tenant.clone()
    };

    let started = Instant::now();
    let query_id = uuid::Uuid::new_v4().simple().to_string();

    // 阶段一：决策（300ms 超时 + 非法 profile 的安全回退）
    let t_decide = Instant::now();
    let decision = decision::decide(state, &query_id, &req).await;
    let decide_ms = elapsed_ms(t_decide);
    let profile = decision.profile.clone();

    // Agentic 轮数按全局硬限收敛（是否真正执行取决于检索链路接入）
    let requested_rounds = decision
        .agentic
        .as_ref()
        .and_then(|a| a.max_rounds)
        .unwrap_or(0);
    let effective_rounds = requested_rounds.min(state.budget.max_agentic_rounds);
    debug!(profile = %profile, requested_rounds, effective_rounds, "决策完成");

    // 阶段二：检索（全局时延预算 + 档位超时双重硬限，超时优雅降级）
    let remaining_ms = state
        .budget
        .max_latency_ms
        .saturating_sub(elapsed_ms(started) as u64);
    let profile_timeout =
        Duration::from_millis(profiles::profile_timeout_ms(&profile).min(remaining_ms.max(1)));

    let (mut outcome, budget_exceeded) = if remaining_ms == 0 {
        (
            RetrievalOutcome::unavailable("全局时延预算已耗尽（max_latency_ms）", 0.0),
            true,
        )
    } else {
        let future = retrieval::run(
            state,
            &profile,
            &req.query,
            &tenant,
            req.kb.as_deref(),
            req.top_k,
            req.principal.as_ref(),
            req.graph_enabled,
            profile_timeout,
        );
        match tokio::time::timeout(Duration::from_millis(remaining_ms), future).await {
            Ok(result) => (result, false),
            Err(_) => (
                RetrievalOutcome::unavailable(
                    "超出全局时延预算 max_latency_ms，已中断检索并降级返回",
                    0.0,
                ),
                true,
            ),
        }
    };

    // 阶段三：组装——证据预算 / 引用 / grounding / trace
    let t_assemble = Instant::now();
    let tokens_used = apply_context_budget(&mut outcome.evidence, state.budget.max_context_tokens);
    let citations = if outcome.citations.is_empty() {
        derive_citations(&outcome.evidence)
    } else {
        std::mem::take(&mut outcome.citations)
    };
    let grounding = build_grounding(&decision, &outcome, budget_exceeded);
    let assemble_ms = elapsed_ms(t_assemble);
    let (retrieve_ms, rerank_ms) = (outcome.retrieve_ms, outcome.rerank_ms);

    // 决策 trace 落库（非阻塞；未配置库时是空操作）。
    // 这是 README 硬约束「落 trace」的落地点——此前只有建表、没有写入。
    state.trace.record(crate::trace::DecisionTrace {
        query_id: query_id.clone(),
        tenant_id: tenant.clone(),
        kb_id: req.kb.clone(),
        model_id: decision.model.id.clone(),
        model_version: Some(decision.model.version.clone()),
        profile: profile.clone(),
        escalate: Some(decision.escalate.clone()),
        answerable_prob: decision.answerable_prob,
        confidence: decision.confidence,
        probabilities: decision
            .probabilities
            .as_ref()
            .and_then(|p| serde_json::to_string(p).ok()),
        features: decision
            .features
            .as_ref()
            .and_then(|f| serde_json::to_string(f).ok()),
        reason: decision.reason.clone(),
        fallback_used: decision.fallback_used.unwrap_or(false),
        latency_ms: Some(decide_ms),
    });

    info!(
        query_id = %query_id,
        profile = %profile,
        fallback_used = decision.fallback_used.unwrap_or(false),
        evidence = outcome.evidence.len(),
        total_ms = elapsed_ms(started),
        "检索完成"
    );

    Ok(RetrieveResponse {
        query_id,
        mode: "evidence-only".to_string(),
        decision,
        evidence: outcome.evidence,
        citations,
        grounding,
        // agentic_rounds 为实际执行轮数：链路接入前为 0
        trace: Trace {
            stages_ms: StagesMs {
                understand: None,
                decide: Some(decide_ms),
                retrieve: Some(retrieve_ms),
                rerank: Some(rerank_ms),
                assemble: Some(assemble_ms),
                generate: None,
            },
            agentic_rounds: 0,
            tokens_used,
            cost_estimate: None,
        },
    })
}

/// 证据 token 预算硬限：近似 token（字符数 / 4）超限即截断，优先保留高分证据。
fn apply_context_budget(evidence: &mut Vec<Evidence>, max_tokens: u64) -> u64 {
    let mut used: u64 = 0;
    let mut kept: Vec<Evidence> = Vec::with_capacity(evidence.len());
    for item in evidence.drain(..) {
        let cost = approx_tokens(&item.content);
        if used + cost > max_tokens && !kept.is_empty() {
            break;
        }
        used += cost;
        kept.push(item);
    }
    *evidence = kept;
    used
}

/// 近似 token 估算（真实实现应以 BGE-M3 tokenizer 为准）。
fn approx_tokens(text: &str) -> u64 {
    (text.chars().count() as u64 + 3) / 4
}

/// 下游未返回引用时，用证据派生最小引用（doc_id 暂以 source 兜底，provenance 接入后替换）。
fn derive_citations(evidence: &[Evidence]) -> Vec<Citation> {
    evidence
        .iter()
        .map(|item| Citation {
            doc_id: item.source.clone(),
            page: item.page,
            section: None,
            url: None,
            char_range: None,
        })
        .collect()
}

/// Grounding 校验：不通过则拒答；仍返回已获得证据，由 agent 决定是否采用。
fn build_grounding(
    decision: &Decision,
    outcome: &RetrievalOutcome,
    budget_exceeded: bool,
) -> Grounding {
    let top_score = outcome.evidence.first().map(|e| e.score).unwrap_or(0.0);

    if decision.profile == "strict_refuse" {
        return refuse(top_score, "决策档位 strict_refuse：无答案倾向，优先拒答");
    }
    if let Some(prob) = decision.answerable_prob {
        if prob < ANSWERABLE_FLOOR {
            return refuse(
                top_score,
                format!("answerable_prob={prob} 低于阈值 {ANSWERABLE_FLOOR}"),
            );
        }
    }
    if budget_exceeded {
        return refuse(top_score, "超出全局时延预算，已降级返回");
    }
    if !outcome.available {
        return refuse(
            top_score,
            outcome
                .note
                .clone()
                .unwrap_or_else(|| "检索链路不可用".to_string()),
        );
    }
    if outcome.evidence.is_empty() {
        return refuse(0.0, "未召回任何证据");
    }
    Grounding {
        passed: true,
        score: top_score,
        refused: false,
        refuse_reason: None,
    }
}

fn refuse(score: f64, reason: impl Into<String>) -> Grounding {
    Grounding {
        passed: false,
        score,
        refused: true,
        refuse_reason: Some(reason.into()),
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}
