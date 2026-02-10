use std::sync::Arc;

use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;

use crate::state::{Experiment, ExperimentConfig, ExperimentResults, SharedState, LogitLensData, LayerData, TokenProb};

fn err(msg: impl std::fmt::Display) -> rmcp::model::ErrorData {
    rmcp::model::ErrorData::internal_error(msg.to_string(), None)
}

// ── MCP Tool Request Types ──────────────────────────────────────────

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ModelInfoRequest {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TokenizeRequest {
    #[schemars(description = "Text to tokenize")]
    pub text: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForwardIntrospectRequest {
    #[schemars(description = "Input text to run through the model")]
    pub text: String,

    #[schemars(description = "Number of top tokens to return per layer (default: 10)")]
    pub top_k: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SetSteeringVectorRequest {
    #[schemars(description = "Layer indices to apply the steering vector to")]
    pub layers: Vec<usize>,

    #[schemars(description = "Scale factor for the steering vector")]
    pub scale: f64,

    #[schemars(description = "Steering vector as flat f32 values (hidden_size dimension)")]
    pub vector: Vec<f32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ClearSteeringRequest {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListExperimentsRequest {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetExperimentRequest {
    #[schemars(description = "Experiment ID")]
    pub id: String,
}

// ── MCP Server ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct IntrospectServer {
    pub state: Arc<SharedState>,
    tool_router: ToolRouter<Self>,
}

impl IntrospectServer {
    pub fn new(state: Arc<SharedState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl IntrospectServer {
    #[tool(description = "Get model architecture information: layer count, layer types (GDN vs full attention), hidden size, MoE config, etc.")]
    fn model_info(
        &self,
        Parameters(_req): Parameters<ModelInfoRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let info = &self.state.model_info;
        let text = serde_json::to_string_pretty(info).map_err(|e| err(e))?;
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(description = "Tokenize text into token IDs and decoded token strings.")]
    fn tokenize(
        &self,
        Parameters(req): Parameters<TokenizeRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let model = self.state.model.lock().map_err(|e| err(e))?;
        let (ids, tokens) = model.tokenize(&req.text).map_err(|e| err(e))?;

        let mut out = format!("{} tokens:\n", ids.len());
        for (i, (id, tok)) in ids.iter().zip(tokens.iter()).enumerate() {
            out.push_str(&format!("  {:>4}: {:>6} = {:?}\n", i, id, tok));
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(description = "Run a forward pass with introspection enabled. Captures hidden states at every layer, runs logit lens analysis, and returns top-k token predictions per layer. Results are saved and viewable on the dashboard.")]
    fn forward_introspect(
        &self,
        Parameters(req): Parameters<ForwardIntrospectRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let top_k = req.top_k.unwrap_or(10);
        let model = self.state.model.lock().map_err(|e| err(e))?;

        // Tokenize
        let (token_ids, tokens) = model.tokenize(&req.text).map_err(|e| err(e))?;

        // Forward pass with introspection
        let result = model.forward_introspect(&req.text).map_err(|e| err(e))?;

        // Logit lens
        let lens = model.logit_lens(&result.hidden_states).map_err(|e| err(e))?;

        // Build per-layer data
        let num_layers = lens.layer_probs.len();
        let mut layers_data = Vec::with_capacity(num_layers);
        let mut out = format!(
            "Forward introspect: {} tokens, {} layers\n\n",
            tokens.len(),
            num_layers
        );

        for (layer_idx, probs_tensor) in lens.layer_probs.iter().enumerate() {
            let layer_type = if layer_idx == 0 {
                "embedding"
            } else if layer_idx % self.state.model_info.full_attention_interval == 0 {
                "full_attention"
            } else {
                "gdn"
            };

            // probs_tensor shape: (batch=1, vocab_size)
            let probs_vec: Vec<f32> = probs_tensor
                .squeeze(0)
                .map_err(|e| err(e))?
                .to_vec1()
                .map_err(|e| err(e))?;

            // Get top-k tokens
            let mut indexed: Vec<(usize, f32)> =
                probs_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_tokens: Vec<TokenProb> = indexed
                .iter()
                .take(top_k)
                .map(|&(id, prob)| TokenProb {
                    token: model.decode_token(id as u32),
                    token_id: id as u32,
                    probability: prob,
                })
                .collect();

            out.push_str(&format!("Layer {:>2} ({:>14}):", layer_idx, layer_type));
            for tp in top_tokens.iter().take(5) {
                out.push_str(&format!("  {:.4} {:?}", tp.probability, tp.token));
            }
            out.push('\n');

            layers_data.push(LayerData {
                layer_idx,
                layer_type: layer_type.to_string(),
                top1_prob: indexed.first().map(|x| x.1).unwrap_or(0.0),
                top_tokens,
            });
        }

        // Save experiment
        let exp_id = uuid::Uuid::new_v4().to_string();
        let experiment = Experiment {
            id: exp_id.clone(),
            name: format!(
                "logit_lens: {:?}",
                if req.text.len() > 50 {
                    &req.text[..50]
                } else {
                    &req.text
                }
            ),
            status: "completed".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .to_string(),
            config: ExperimentConfig {
                experiment_type: "logit_lens".to_string(),
                prompt: req.text,
                steering_layers: None,
                steering_scale: None,
            },
            results: Some(ExperimentResults {
                logit_lens: Some(LogitLensData {
                    tokens: tokens.clone(),
                    token_ids: token_ids.clone(),
                    layers: layers_data,
                }),
            }),
        };

        {
            let mut experiments = self.state.experiments.write().map_err(|e| err(e))?;
            experiments.insert(exp_id.clone(), experiment);
        }

        let port = self.state.port;
        out.push_str(&format!(
            "\nExperiment saved: {}\nDashboard: http://localhost:{}/experiment/{}\n",
            exp_id, port, exp_id
        ));

        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(description = "Set a steering vector on specified layers. The vector is broadcast-added to the residual stream after each layer's forward pass.")]
    fn set_steering_vector(
        &self,
        Parameters(req): Parameters<SetSteeringVectorRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let model = self.state.model.lock().map_err(|e| err(e))?;
        let device = model.device();

        let hidden_size = self.state.model_info.hidden_size;
        if req.vector.len() != hidden_size {
            return Err(err(format!(
                "Vector length {} doesn't match hidden_size {}",
                req.vector.len(),
                hidden_size
            )));
        }

        let vector =
            candle_core::Tensor::new(req.vector.as_slice(), device).map_err(|e| err(e))?;
        let scaled = (&vector * req.scale).map_err(|e| err(e))?;

        for &layer in &req.layers {
            if layer >= self.state.model_info.num_layers {
                return Err(err(format!(
                    "Layer {} out of range (model has {} layers)",
                    layer, self.state.model_info.num_layers
                )));
            }
            model.set_steering_vector(layer, scaled.clone());
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Steering vector set on {} layers: {:?} (scale: {})",
            req.layers.len(),
            req.layers,
            req.scale
        ))]))
    }

    #[tool(description = "Clear all steering vectors from the model.")]
    fn clear_steering_vectors(
        &self,
        Parameters(_req): Parameters<ClearSteeringRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let model = self.state.model.lock().map_err(|e| err(e))?;
        model.clear_steering_vectors();
        Ok(CallToolResult::success(vec![Content::text(
            "All steering vectors cleared.",
        )]))
    }

    #[tool(description = "List all saved experiments with their IDs, names, and status.")]
    fn list_experiments(
        &self,
        Parameters(_req): Parameters<ListExperimentsRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let experiments = self.state.experiments.read().map_err(|e| err(e))?;
        if experiments.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No experiments yet. Use forward_introspect to create one.",
            )]));
        }
        let mut out = format!("{} experiments:\n", experiments.len());
        for exp in experiments.values() {
            out.push_str(&format!(
                "  {} [{}] {}\n",
                exp.id, exp.status, exp.name
            ));
        }
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    #[tool(description = "Get detailed results for a specific experiment by ID.")]
    fn get_experiment(
        &self,
        Parameters(req): Parameters<GetExperimentRequest>,
    ) -> Result<CallToolResult, rmcp::model::ErrorData> {
        let experiments = self.state.experiments.read().map_err(|e| err(e))?;
        match experiments.get(&req.id) {
            Some(exp) => {
                let json = serde_json::to_string_pretty(exp).map_err(|e| err(e))?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            None => Err(err(format!("Experiment not found: {}", req.id))),
        }
    }
}

#[tool_handler]
impl ServerHandler for IntrospectServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: Implementation {
                name: "introspect".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            },
            instructions: Some(
                "Activation-level introspection server for Qwen3-Coder-Next. \
                 Tools: model_info, tokenize, forward_introspect (logit lens), \
                 set_steering_vector, clear_steering_vectors, list_experiments, \
                 get_experiment. Results are viewable on the HTML dashboard."
                    .to_string(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
