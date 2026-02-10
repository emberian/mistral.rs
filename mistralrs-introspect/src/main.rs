mod dashboard;
mod mcp;
mod state;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::Result;
use axum::{routing::get, Router};
use candle_core::{DType, Device};
use clap::Parser;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use tracing_subscriber::EnvFilter;

use mcp::IntrospectServer;
use state::SharedState;

#[derive(Parser)]
#[command(name = "introspect", about = "Introspection MCP server + dashboard")]
struct Args {
    /// HuggingFace model ID or local path to model directory
    #[arg(short, long)]
    model: String,

    /// Port to serve on
    #[arg(short, long, default_value = "3131")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Data type for model weights (f16, bf16, f32)
    #[arg(long, default_value = "f16")]
    dtype: String,
}

fn parse_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f32" | "float32" => Ok(DType::F32),
        other => anyhow::bail!("Unknown dtype: {}", other),
    }
}

fn select_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        tracing::info!("Using Metal device");
        return Ok(Device::new_metal(0)?);
    }

    #[cfg(feature = "cuda")]
    {
        tracing::info!("Using CUDA device 0");
        return Ok(Device::new_cuda(0)?);
    }

    #[allow(unreachable_code)]
    {
        tracing::info!("Using CPU device");
        Ok(Device::Cpu)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let dtype = parse_dtype(&args.dtype)?;
    let device = select_device()?;

    // Load model
    tracing::info!("Loading model: {}", args.model);
    let model =
        mistralrs_core::introspection::IntrospectionModel::load(&args.model, device, dtype)?;

    let model_info = model.model_info();

    let shared_state = Arc::new(SharedState {
        model: Mutex::new(model),
        model_info,
        experiments: RwLock::new(HashMap::new()),
        port: args.port,
    });

    // MCP service (Streamable HTTP transport)
    let mcp_state = shared_state.clone();
    let mcp_service: StreamableHttpService<IntrospectServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(IntrospectServer::new(mcp_state.clone())),
            Default::default(),
            StreamableHttpServerConfig::default(),
        );

    // Axum router: MCP + dashboard + API
    let app = Router::new()
        .nest_service("/mcp", mcp_service)
        .route("/", get(dashboard::index))
        .route("/experiment/{id}", get(dashboard::experiment_detail))
        .route("/api/experiments", get(dashboard::api_experiments))
        .route("/api/experiments/{id}", get(dashboard::api_experiment))
        .route("/api/model_info", get(dashboard::api_model_info))
        .with_state(shared_state);

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting server at http://{}", addr);
    tracing::info!("  Dashboard: http://{}/", addr);
    tracing::info!("  MCP endpoint: http://{}/mcp", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
