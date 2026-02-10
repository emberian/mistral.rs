//! Public API for activation-level introspection experiments.
//!
//! Provides [`IntrospectionModel`] - a high-level wrapper around model internals
//! that handles weight loading, tokenization, and introspection operations
//! (logit lens, steering vector injection, hidden state capture).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use indicatif::MultiProgress;
use mistralrs_quant::ShardedSafeTensors;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::device_map::DummyDeviceMapper;
use crate::models::qwen3_next::{self, Config, Model};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::{
    text_models_inputs_processor::FlashParams, NormalLoadingMetadata,
};

/// High-level model wrapper for introspection experiments.
pub struct IntrospectionModel {
    model: Model,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
}

/// Result of a forward pass with introspection.
pub struct IntrospectionResult {
    /// Model output logits, shape (batch, seq_len, vocab_size).
    pub logits: Tensor,
    /// Hidden states at each layer.
    /// Index 0 = after embedding, index i+1 = after decoder layer i.
    pub hidden_states: Vec<Tensor>,
}

/// Per-layer logit lens probability vectors (softmax over vocab).
pub struct LogitLensResult {
    /// One tensor per layer, shape (batch, vocab_size) — softmax probs at last token position.
    pub layer_probs: Vec<Tensor>,
}

/// Serializable model architecture info.
#[derive(Clone, Serialize)]
pub struct ModelInfo {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub full_attention_interval: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
}

impl IntrospectionModel {
    /// Load a model for introspection from HuggingFace Hub or a local directory.
    ///
    /// `model_path` can be:
    /// - A HuggingFace model ID (e.g., `"Qwen/Qwen3-Coder-Next"`)
    /// - An absolute/relative path to a local model directory
    ///
    /// The directory must contain `config.json`, `tokenizer.json`, and
    /// safetensors weight files (sharded or single).
    pub fn load(model_path: &str, device: Device, dtype: DType) -> anyhow::Result<Self> {
        let (config_path, weight_files, tokenizer_path) = if Path::new(model_path).is_dir() {
            Self::resolve_local_paths(model_path)?
        } else {
            Self::resolve_hf_paths(model_path)?
        };

        // Parse model config
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        tracing::info!(
            "Loading Qwen3-Next: {} layers, hidden_size={}, {} experts ({} active)",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_tok,
        );

        // Memory-map safetensors files into a sharded VarBuilder
        let vb = unsafe {
            ShardedSafeTensors::sharded(
                &weight_files,
                dtype,
                &device,
                None,
                Arc::new(|_| true),
            )?
        };

        let normal_loading_metadata = NormalLoadingMetadata {
            mapper: Box::new(DummyDeviceMapper {
                nm_device: device.clone(),
            }),
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        };

        let model = Model::new(
            &config,
            vb,
            true, // is_gptx (unused for qwen3_next)
            normal_loading_metadata,
            AttentionImplementation::Eager,
        )?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        tracing::info!("Model loaded successfully on {:?}", device);

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
        })
    }

    fn resolve_local_paths(dir: &str) -> anyhow::Result<(PathBuf, Vec<PathBuf>, PathBuf)> {
        let dir = Path::new(dir);
        let config_path = dir.join("config.json");
        anyhow::ensure!(
            config_path.exists(),
            "config.json not found in {}",
            dir.display()
        );

        let tokenizer_path = dir.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found in {}",
            dir.display()
        );

        // Find safetensors weight files
        let index_path = dir.join("model.safetensors.index.json");
        let weight_files = if index_path.exists() {
            let index_str = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
            let weight_map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index: missing weight_map"))?;
            let filenames: HashSet<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            let mut files: Vec<PathBuf> = filenames.into_iter().map(|f| dir.join(f)).collect();
            files.sort();
            files
        } else {
            let single = dir.join("model.safetensors");
            anyhow::ensure!(
                single.exists(),
                "No safetensors files found in {}",
                dir.display()
            );
            vec![single]
        };

        Ok((config_path, weight_files, tokenizer_path))
    }

    fn resolve_hf_paths(model_id: &str) -> anyhow::Result<(PathBuf, Vec<PathBuf>, PathBuf)> {
        use hf_hub::api::sync::Api;

        tracing::info!("Downloading model from HuggingFace: {}", model_id);

        let api = Api::new()?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        // Check for sharded weights
        let weight_files = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)?;
                let weight_map = index["weight_map"]
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index"))?;
                let filenames: HashSet<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                let mut files = Vec::new();
                for filename in &filenames {
                    tracing::info!("Downloading weight shard: {}", filename);
                    files.push(repo.get(filename)?);
                }
                files.sort();
                files
            }
            Err(_) => vec![repo.get("model.safetensors")?],
        };

        Ok((config_path, weight_files, tokenizer_path))
    }

    /// Run a forward pass capturing hidden states at every layer.
    ///
    /// Not safe to call concurrently — the caller must serialize access.
    pub fn forward_introspect(&self, text: &str) -> anyhow::Result<IntrospectionResult> {
        self.forward_introspect_layers(text, None)
    }

    /// Run a forward pass capturing hidden states at specific layers only.
    ///
    /// `layers` specifies which layers to capture (0 = embedding, 1..N = decoder layers).
    /// Pass `None` to capture all layers. Capturing fewer layers reduces memory copies
    /// during the forward pass — significant on memory-bandwidth-bound systems.
    pub fn forward_introspect_layers(
        &self,
        text: &str,
        layers: Option<std::collections::HashSet<usize>>,
    ) -> anyhow::Result<IntrospectionResult> {
        // Set selective capture before the forward pass
        {
            let intro = self.model.introspection.lock().unwrap();
            // Need to drop and re-acquire as mutable for the field write.
            // (Interior mutability via Mutex means &self is fine.)
            drop(intro);
            let mut intro = self.model.introspection.lock().unwrap();
            intro.capture_layers = layers;
        }

        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();

        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: true,
        };

        let seq_len = input_ids.len();
        let (logits, hidden_states) = self.model.forward_introspect(
            &input_tensor,
            &[0],
            vec![(seq_len, seq_len)],
            None,
            &flash_params,
        )?;

        // Reset capture_layers
        {
            let mut intro = self.model.introspection.lock().unwrap();
            intro.capture_layers = None;
        }

        Ok(IntrospectionResult {
            logits,
            hidden_states,
        })
    }

    /// Run logit lens on hidden states from [`forward_introspect`].
    ///
    /// Returns per-layer softmax probability vectors at the last token position.
    pub fn logit_lens(&self, hidden_states: &[Tensor]) -> anyhow::Result<LogitLensResult> {
        let layer_probs = self.model.logit_lens_all(hidden_states)?;
        Ok(LogitLensResult { layer_probs })
    }

    /// Tokenize text and return (token_ids, token_strings).
    pub fn tokenize(&self, text: &str) -> anyhow::Result<(Vec<u32>, Vec<String>)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let tokens: Vec<String> = ids
            .iter()
            .map(|&id| {
                self.tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| format!("<{}>", id))
            })
            .collect();
        Ok((ids, tokens))
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> String {
        self.tokenizer
            .decode(&[id], false)
            .unwrap_or_else(|_| format!("<{}>", id))
    }

    /// Set a steering vector for a specific layer.
    pub fn set_steering_vector(&self, layer_idx: usize, vector: Tensor) {
        self.model.set_steering_vector(layer_idx, vector);
    }

    /// Set the same steering vector (scaled) across a range of layers.
    pub fn set_steering_vectors_range(
        &self,
        layers: std::ops::Range<usize>,
        vector: &Tensor,
        scale: f64,
    ) -> anyhow::Result<()> {
        self.model
            .set_steering_vectors_range(layers, vector, scale)?;
        Ok(())
    }

    /// Clear all steering vectors.
    pub fn clear_steering_vectors(&self) {
        self.model.clear_steering_vectors();
    }

    /// Get model architecture information.
    pub fn model_info(&self) -> ModelInfo {
        let layer_types: Vec<String> = (0..self.config.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.config.full_attention_interval == 0 {
                    "full_attention".to_string()
                } else {
                    "gdn".to_string()
                }
            })
            .collect();

        ModelInfo {
            hidden_size: self.config.hidden_size,
            num_layers: self.config.num_hidden_layers,
            num_attention_heads: self.config.num_attention_heads,
            num_kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim,
            full_attention_interval: self.config.full_attention_interval,
            num_experts: self.config.num_experts,
            num_experts_per_tok: self.config.num_experts_per_tok,
            vocab_size: self.config.vocab_size,
            layer_types,
        }
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the device the model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the raw model config.
    pub fn config(&self) -> &qwen3_next::Config {
        &self.config
    }
}
