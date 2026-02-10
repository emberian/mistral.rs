use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use mistralrs_core::introspection::{IntrospectionModel, ModelInfo};
use serde::Serialize;

/// Shared state between MCP tools and the dashboard.
pub struct SharedState {
    pub model: Mutex<IntrospectionModel>,
    pub model_info: ModelInfo,
    pub experiments: RwLock<HashMap<String, Experiment>>,
    pub port: u16,
}

// ── Experiment Data Types ───────────────────────────────────────────

#[derive(Clone, Serialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub status: String,
    pub created_at: String,
    pub config: ExperimentConfig,
    pub results: Option<ExperimentResults>,
}

#[derive(Clone, Serialize)]
pub struct ExperimentConfig {
    pub experiment_type: String,
    pub prompt: String,
    pub steering_layers: Option<Vec<usize>>,
    pub steering_scale: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct ExperimentResults {
    pub logit_lens: Option<LogitLensData>,
}

#[derive(Clone, Serialize)]
pub struct LogitLensData {
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub layers: Vec<LayerData>,
}

#[derive(Clone, Serialize)]
pub struct LayerData {
    pub layer_idx: usize,
    pub layer_type: String,
    pub top1_prob: f32,
    pub top_tokens: Vec<TokenProb>,
}

#[derive(Clone, Serialize)]
pub struct TokenProb {
    pub token: String,
    pub token_id: u32,
    pub probability: f32,
}
