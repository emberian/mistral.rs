use std::sync::Arc;

use axum::{
    extract::{Path, State},
    response::{Html, IntoResponse, Json},
};

use crate::state::SharedState;

// ── HTML Dashboard ──────────────────────────────────────────────────

pub async fn index(State(state): State<Arc<SharedState>>) -> Html<String> {
    let experiments = state.experiments.read().unwrap();
    let info = &state.model_info;

    let mut exp_rows = String::new();
    let mut sorted_exps: Vec<_> = experiments.values().collect();
    sorted_exps.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    for exp in &sorted_exps {
        exp_rows.push_str(&format!(
            r#"<tr>
                <td><a href="/experiment/{id}">{id_short}...</a></td>
                <td>{name}</td>
                <td><span class="badge badge-{status}">{status}</span></td>
                <td>{exp_type}</td>
                <td>{prompt}</td>
            </tr>"#,
            id = exp.id,
            id_short = &exp.id[..8],
            name = html_escape(&exp.name),
            status = exp.status,
            exp_type = exp.config.experiment_type,
            prompt = html_escape(&exp.config.prompt.chars().take(80).collect::<String>()),
        ));
    }

    if exp_rows.is_empty() {
        exp_rows = r#"<tr><td colspan="5" class="empty">No experiments yet. Use MCP tools to create one.</td></tr>"#.to_string();
    }

    let layer_types_json = serde_json::to_string(&info.layer_types).unwrap_or_default();

    Html(format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Introspect Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>{CSS}</style>
    <meta http-equiv="refresh" content="10">
</head>
<body>
<header>
    <h1>Introspect Dashboard</h1>
    <span class="subtitle">Activation-level introspection for Qwen3-Coder-Next</span>
</header>
<main>
    <section class="model-info">
        <h2>Model Architecture</h2>
        <div class="info-grid">
            <div class="info-card">
                <span class="label">Layers</span>
                <span class="value">{num_layers}</span>
            </div>
            <div class="info-card">
                <span class="label">Hidden Size</span>
                <span class="value">{hidden_size}</span>
            </div>
            <div class="info-card">
                <span class="label">Attention Heads</span>
                <span class="value">{num_heads}</span>
            </div>
            <div class="info-card">
                <span class="label">KV Heads</span>
                <span class="value">{num_kv_heads}</span>
            </div>
            <div class="info-card">
                <span class="label">Experts</span>
                <span class="value">{num_experts} ({active} active)</span>
            </div>
            <div class="info-card">
                <span class="label">Full Attn Interval</span>
                <span class="value">every {attn_interval} layers</span>
            </div>
            <div class="info-card">
                <span class="label">Vocab Size</span>
                <span class="value">{vocab_size}</span>
            </div>
            <div class="info-card">
                <span class="label">MCP Endpoint</span>
                <span class="value">http://localhost:{port}/mcp</span>
            </div>
        </div>
        <div id="layer-type-chart" style="height:80px;margin-top:12px;"></div>
    </section>

    <section class="experiments">
        <h2>Experiments ({exp_count})</h2>
        <table>
            <thead>
                <tr><th>ID</th><th>Name</th><th>Status</th><th>Type</th><th>Prompt</th></tr>
            </thead>
            <tbody>{exp_rows}</tbody>
        </table>
    </section>
</main>
<script>
    // Layer type visualization
    const layerTypes = {layer_types_json};
    const colors = layerTypes.map(t => t === 'full_attention' ? '#4ecdc4' : '#556270');
    Plotly.newPlot('layer-type-chart', [{{
        x: layerTypes.map((_, i) => i),
        y: layerTypes.map(() => 1),
        type: 'bar',
        marker: {{ color: colors }},
        hovertext: layerTypes.map((t, i) => `Layer ${{i}}: ${{t}}`),
        hoverinfo: 'text',
    }}], {{
        margin: {{ t: 0, b: 20, l: 30, r: 10 }},
        xaxis: {{ title: 'Layer Index', dtick: 4 }},
        yaxis: {{ visible: false }},
        bargap: 0.1,
        height: 80,
    }}, {{ displayModeBar: false }});
</script>
</body>
</html>"##,
        CSS = CSS,
        num_layers = info.num_layers,
        hidden_size = info.hidden_size,
        num_heads = info.num_attention_heads,
        num_kv_heads = info.num_kv_heads,
        num_experts = info.num_experts,
        active = info.num_experts_per_tok,
        attn_interval = info.full_attention_interval,
        vocab_size = info.vocab_size,
        port = state.port,
        exp_count = sorted_exps.len(),
        exp_rows = exp_rows,
        layer_types_json = layer_types_json,
    ))
}

pub async fn experiment_detail(
    State(state): State<Arc<SharedState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let experiments = state.experiments.read().unwrap();
    let exp = match experiments.get(&id) {
        Some(e) => e.clone(),
        None => {
            return Html(format!(
                r#"<!DOCTYPE html><html><body><h1>Experiment not found: {}</h1><a href="/">Back</a></body></html>"#,
                html_escape(&id)
            ));
        }
    };

    let exp_json = serde_json::to_string(&exp).unwrap_or_default();

    let logit_lens_script = if let Some(ref results) = exp.results {
        if let Some(ref lens) = results.logit_lens {
            let heatmap_data: Vec<Vec<f32>> = lens
                .layers
                .iter()
                .map(|l| vec![l.top1_prob])
                .collect();
            let layer_labels: Vec<String> = lens
                .layers
                .iter()
                .map(|l| format!("L{} ({})", l.layer_idx, l.layer_type))
                .collect();

            let mut top_tokens_html = String::new();
            for layer in &lens.layers {
                top_tokens_html.push_str(&format!(
                    r#"<div class="layer-tokens">
                        <h4>Layer {} ({})</h4>
                        <div class="token-bars">"#,
                    layer.layer_idx, layer.layer_type
                ));
                for tp in layer.top_tokens.iter().take(10) {
                    let width = (tp.probability * 100.0 * 4.0).min(100.0);
                    top_tokens_html.push_str(&format!(
                        r#"<div class="token-bar">
                            <div class="bar" style="width:{width}%"></div>
                            <span class="token-label">{:.4} {:?}</span>
                        </div>"#,
                        tp.probability,
                        tp.token,
                        width = width,
                    ));
                }
                top_tokens_html.push_str("</div></div>");
            }

            format!(
                r#"
                <section>
                    <h2>Logit Lens: Top-1 Confidence by Layer</h2>
                    <div id="logit-lens-heatmap"></div>
                </section>
                <section>
                    <h2>Per-Layer Top Token Predictions</h2>
                    <div class="layer-tokens-container">{top_tokens_html}</div>
                </section>
                <script>
                    const heatmapZ = {heatmap_json};
                    const layerLabels = {labels_json};
                    Plotly.newPlot('logit-lens-heatmap', [{{
                        z: heatmapZ,
                        y: layerLabels,
                        x: ['last token'],
                        type: 'heatmap',
                        colorscale: 'YlGnBu',
                        reversescale: true,
                        hovertemplate: '%{{y}}<br>P(top-1): %{{z:.4f}}<extra></extra>',
                    }}], {{
                        margin: {{ t: 20, b: 40, l: 150, r: 40 }},
                        height: Math.max(400, layerLabels.length * 16),
                        yaxis: {{ autorange: 'reversed' }},
                    }});
                </script>"#,
                top_tokens_html = top_tokens_html,
                heatmap_json = serde_json::to_string(&heatmap_data).unwrap_or_default(),
                labels_json = serde_json::to_string(&layer_labels).unwrap_or_default(),
            )
        } else {
            String::new()
        }
    } else {
        "<p>No results available.</p>".to_string()
    };

    Html(format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Experiment: {name}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>{CSS}</style>
</head>
<body>
<header>
    <a href="/" class="back">&larr; Dashboard</a>
    <h1>{name}</h1>
    <span class="subtitle">{id}</span>
</header>
<main>
    <section class="experiment-config">
        <h2>Configuration</h2>
        <div class="info-grid">
            <div class="info-card">
                <span class="label">Type</span>
                <span class="value">{exp_type}</span>
            </div>
            <div class="info-card">
                <span class="label">Status</span>
                <span class="value badge badge-{status}">{status}</span>
            </div>
            <div class="info-card wide">
                <span class="label">Prompt</span>
                <span class="value prompt">{prompt}</span>
            </div>
        </div>
    </section>
    {logit_lens_script}
    <section>
        <h2>Raw Data</h2>
        <details>
            <summary>JSON</summary>
            <pre class="json">{exp_json}</pre>
        </details>
    </section>
</main>
</body>
</html>"##,
        CSS = CSS,
        name = html_escape(&exp.name),
        id = exp.id,
        exp_type = exp.config.experiment_type,
        status = exp.status,
        prompt = html_escape(&exp.config.prompt),
        logit_lens_script = logit_lens_script,
        exp_json = html_escape(&exp_json),
    ))
}

// ── JSON API ────────────────────────────────────────────────────────

pub async fn api_experiments(State(state): State<Arc<SharedState>>) -> impl IntoResponse {
    let experiments = state.experiments.read().unwrap();
    let list: Vec<_> = experiments.values().collect();
    Json(serde_json::to_value(&list).unwrap_or_default())
}

pub async fn api_experiment(
    State(state): State<Arc<SharedState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let experiments = state.experiments.read().unwrap();
    match experiments.get(&id) {
        Some(exp) => Json(serde_json::to_value(exp).unwrap_or_default()).into_response(),
        None => (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not found"})),
        )
            .into_response(),
    }
}

pub async fn api_model_info(State(state): State<Arc<SharedState>>) -> impl IntoResponse {
    Json(serde_json::to_value(&state.model_info).unwrap_or_default())
}

// ── Utilities ───────────────────────────────────────────────────────

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const CSS: &str = r#"
    :root {
        --bg: #0d1117;
        --surface: #161b22;
        --border: #30363d;
        --text: #c9d1d9;
        --text-dim: #8b949e;
        --accent: #58a6ff;
        --green: #3fb950;
        --yellow: #d29922;
        --red: #f85149;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
        background: var(--bg);
        color: var(--text);
        line-height: 1.5;
    }
    header {
        padding: 24px 32px;
        border-bottom: 1px solid var(--border);
    }
    header h1 { font-size: 20px; font-weight: 600; }
    header .subtitle { color: var(--text-dim); font-size: 13px; }
    header .back {
        color: var(--accent);
        text-decoration: none;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 8px;
    }
    main { padding: 24px 32px; max-width: 1400px; }
    section { margin-bottom: 32px; }
    h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: var(--text); }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 12px;
    }
    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px 16px;
    }
    .info-card.wide { grid-column: 1 / -1; }
    .info-card .label {
        display: block;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-dim);
        margin-bottom: 4px;
    }
    .info-card .value { font-size: 14px; font-weight: 500; }
    .info-card .prompt { font-family: monospace; font-size: 12px; word-break: break-all; }
    table {
        width: 100%;
        border-collapse: collapse;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
    }
    th, td { padding: 8px 12px; text-align: left; font-size: 13px; }
    th { background: var(--bg); color: var(--text-dim); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
    td { border-top: 1px solid var(--border); }
    td a { color: var(--accent); text-decoration: none; font-family: monospace; font-size: 12px; }
    td.empty { text-align: center; color: var(--text-dim); padding: 24px; }
    .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }
    .badge-completed { background: #1a3a2a; color: var(--green); }
    .badge-running { background: #2a2a1a; color: var(--yellow); }
    .badge-failed { background: #2a1a1a; color: var(--red); }
    .layer-tokens-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 16px;
    }
    .layer-tokens {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 12px;
    }
    .layer-tokens h4 { font-size: 12px; color: var(--text-dim); margin-bottom: 8px; }
    .token-bar { display: flex; align-items: center; margin-bottom: 3px; }
    .token-bar .bar {
        height: 16px;
        background: var(--accent);
        opacity: 0.6;
        border-radius: 2px;
        min-width: 2px;
    }
    .token-bar .token-label {
        margin-left: 8px;
        font-size: 11px;
        font-family: monospace;
        color: var(--text-dim);
        white-space: nowrap;
    }
    pre.json {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        font-size: 11px;
        overflow-x: auto;
        max-height: 400px;
    }
    details summary {
        cursor: pointer;
        color: var(--accent);
        font-size: 13px;
        margin-bottom: 8px;
    }
    .js-plotly-plot .plotly .modebar { display: none !important; }
"#;
