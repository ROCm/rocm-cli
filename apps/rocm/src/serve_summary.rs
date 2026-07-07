// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine-agnostic deployment summary for `rocm serve`.
//!
//! By default `rocm serve <model>` no longer streams raw engine logs. Instead it
//! shows an animated status indicator while the server starts, runs a small
//! inference smoke test, and then prints a compact summary table: deployment
//! status, the full inference endpoint (with port), the API-qualified model name,
//! and the measured time-to-first-token / throughput.
//!
//! Everything here operates at the CLI/HTTP layer *above* the per-engine
//! adapters, so the summary shape is identical for every serving engine
//! (lemonade, vLLM). Only two things vary by engine, and both are already
//! normalized elsewhere: the health path used for readiness, and whether the
//! server reports token usage (which only affects whether throughput is
//! exact, approximated, or `n/a`).

use std::fmt::Write as _;
use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

use crossterm::QueueableCommand;
use crossterm::cursor::MoveToColumn;
use crossterm::terminal::{Clear, ClearType};
use rocm_core::AppPaths;

use crate::providers::{self, ChatMessage, ChatRequest, ProviderStreamEvent};

/// Tiny prompt used for the startup smoke test. Kept short so the probe adds only
/// a second or two to an otherwise-ready deployment.
const SMOKE_PROMPT: &str = "Reply with a short one-sentence greeting.";
/// Cap the smoke-test generation so a slow or verbose model cannot stall startup.
const SMOKE_MAX_TOKENS: u32 = 32;
/// Braille spinner frames (matching the dashboard's visual language).
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Best-effort metrics measured against a freshly-started server. Every field is
/// optional: any probe failure leaves it `None` and the summary renders `n/a`.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct SmokeMetrics {
    /// Time from request send to the first streamed content token.
    pub ttft: Option<Duration>,
    /// Generation throughput in tokens/sec (inter-token rate after the first token).
    pub gen_tps: Option<f64>,
}

/// Everything the summary table renders. Built by `serve()` from the
/// engine-neutral launch result, so the shape is identical for every engine.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeploymentSummary {
    pub engine: String,
    /// Canonical / API-qualified model id clients pass as `"model"`.
    pub api_model: String,
    /// Full chat-completions endpoint, e.g. `http://127.0.0.1:1337/v1/chat/completions`.
    pub chat_endpoint: String,
    pub service_id: String,
    /// `"ready"`, `"starting"`, or an existing service's status.
    pub status: String,
    /// True when an equivalent server was already running and nothing was spawned.
    pub already_running: bool,
    pub metrics: SmokeMetrics,
    /// GPU/device warnings folded in from the serve plan.
    pub notes: Vec<String>,
}

/// Render `Some(ttft)` as a human duration, or `n/a` when the probe did not
/// produce a first token.
pub(crate) fn format_ttft(ttft: Option<Duration>) -> String {
    match ttft {
        Some(duration) => {
            let millis = duration.as_secs_f64() * 1000.0;
            if millis < 1000.0 {
                format!("{} ms", millis.round() as u64)
            } else {
                format!("{:.2} s", duration.as_secs_f64())
            }
        }
        None => "n/a".to_owned(),
    }
}

/// Render `Some(tps)` as `NN.N tok/s`, or `n/a` when throughput was not measured.
pub(crate) fn format_tps(tps: Option<f64>) -> String {
    match tps {
        Some(value) if value.is_finite() && value > 0.0 => format!("{value:.1} tok/s"),
        _ => "n/a".to_owned(),
    }
}

/// Render the deployment summary table as plain text (returned rather than
/// printed so it is unit-testable and identical across engines).
pub(crate) fn render_summary(summary: &DeploymentSummary) -> String {
    // The server answered its health check within the startup window. A launch
    // that timed out lands here with `status == "starting"`; the heading and a
    // note must make that visibly different from a healthy deployment so the
    // summary is never mistaken for success.
    let ready = summary.status == "ready";
    let heading = if summary.already_running {
        "Deployment summary (already running)"
    } else if ready {
        "Deployment summary"
    } else {
        "Deployment summary (not ready yet)"
    };

    // (label, value) rows, in the order the ticket calls out. Throughput is
    // labelled "approx" because it is derived from streamed SSE chunk counts
    // (~1 token per chunk), not the engine's own token accounting.
    let rows: Vec<(&str, String)> = vec![
        ("status", summary.status.clone()),
        ("engine", summary.engine.clone()),
        ("model", summary.api_model.clone()),
        ("endpoint", summary.chat_endpoint.clone()),
        ("time to first token", format_ttft(summary.metrics.ttft)),
        ("throughput (approx)", format_tps(summary.metrics.gen_tps)),
        ("service", summary.service_id.clone()),
        ("stop", format!("rocm services stop {}", summary.service_id)),
        (
            "logs",
            format!("rocm logs --service {}", summary.service_id),
        ),
    ];

    let label_width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);

    let mut out = String::new();
    out.push_str(heading);
    out.push('\n');
    for (label, value) in rows {
        let _ = writeln!(out, "  {label:<label_width$}  {value}");
    }
    if !ready && !summary.already_running {
        let _ = writeln!(
            out,
            "  note: the server did not report ready before the startup timeout; \
             it may still be loading — check `rocm logs --service {}` or `rocm services list`",
            summary.service_id
        );
    }
    for note in &summary.notes {
        let _ = writeln!(out, "  note: {note}");
    }
    out
}

/// Run a single small chat completion against the just-started local server and
/// measure time-to-first-token and generation throughput. Best-effort: any error
/// (server not OpenAI-compatible, refused, timed out) yields empty metrics rather
/// than failing the launch.
pub(crate) fn run_smoke_test(paths: &AppPaths, api_model: &str) -> SmokeMetrics {
    let request = ChatRequest {
        model: Some(api_model.to_owned()),
        messages: vec![ChatMessage {
            role: "user".to_owned(),
            content: SMOKE_PROMPT.to_owned(),
        }],
        max_tokens: Some(SMOKE_MAX_TOKENS),
        rocm_tools: false,
    };

    let start = Instant::now();
    let mut first_token: Option<Instant> = None;
    let mut last_token: Option<Instant> = None;
    let mut token_chunks: u64 = 0;

    let result = providers::provider_stream_chat_with_callback(
        paths,
        "local",
        &request,
        &mut |event: ProviderStreamEvent| {
            if !event.content.is_empty() {
                let now = Instant::now();
                first_token.get_or_insert(now);
                last_token = Some(now);
                token_chunks += 1;
            }
            Ok(())
        },
    );

    if result.is_err() {
        return SmokeMetrics::default();
    }

    compute_metrics(start, first_token, last_token, token_chunks)
}

/// Pure metric math, split out so it can be unit-tested without a live server.
/// Throughput is the inter-token rate: tokens generated after the first, divided
/// by the elapsed time between the first and last token.
fn compute_metrics(
    start: Instant,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    token_chunks: u64,
) -> SmokeMetrics {
    let ttft = first_token.map(|first| first.saturating_duration_since(start));
    let gen_tps = match (first_token, last_token) {
        (Some(first), Some(last)) if token_chunks > 1 => {
            let window = last.saturating_duration_since(first).as_secs_f64();
            if window > 0.0 {
                Some((token_chunks as f64 - 1.0) / window)
            } else {
                None
            }
        }
        _ => None,
    };
    SmokeMetrics { ttft, gen_tps }
}

/// A carriage-return status indicator written to stderr. Disabled (a no-op) when
/// stderr is not a TTY, so piped/redirected output never receives control
/// characters. Keeps stdout clean for the summary table.
pub(crate) struct Spinner {
    enabled: bool,
    idx: usize,
    label: String,
    active: bool,
}

impl Spinner {
    pub(crate) fn new(label: impl Into<String>) -> Self {
        Self {
            enabled: std::io::stderr().is_terminal(),
            idx: 0,
            label: label.into(),
            active: false,
        }
    }

    /// Change the message shown next to the spinner (e.g. "Running smoke test…").
    pub(crate) fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
        self.render_current();
    }

    /// Advance to the next animation frame and repaint.
    pub(crate) fn tick(&mut self) {
        self.idx = self.idx.wrapping_add(1);
        self.render_current();
    }

    fn render_current(&mut self) {
        if !self.enabled {
            return;
        }
        let frame = SPINNER_FRAMES[self.idx % SPINNER_FRAMES.len()];
        let mut err = std::io::stderr();
        let _ = err.queue(MoveToColumn(0));
        let _ = err.queue(Clear(ClearType::CurrentLine));
        let _ = write!(err, "{frame} {}", self.label);
        let _ = err.flush();
        self.active = true;
    }

    /// Erase the spinner line so the summary table starts on a clean line.
    pub(crate) fn clear(&mut self) {
        if self.enabled && self.active {
            let mut err = std::io::stderr();
            let _ = err.queue(MoveToColumn(0));
            let _ = err.queue(Clear(ClearType::CurrentLine));
            let _ = err.flush();
            self.active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_summary() -> DeploymentSummary {
        DeploymentSummary {
            engine: "vllm".to_owned(),
            api_model: "Qwen/Qwen2.5-7B-Instruct".to_owned(),
            chat_endpoint: "http://127.0.0.1:1337/v1/chat/completions".to_owned(),
            service_id: "vllm-qwen-1720000000".to_owned(),
            status: "ready".to_owned(),
            already_running: false,
            metrics: SmokeMetrics {
                ttft: Some(Duration::from_millis(180)),
                gen_tps: Some(42.15),
            },
            notes: Vec::new(),
        }
    }

    #[test]
    fn format_ttft_uses_ms_under_a_second_and_seconds_above() {
        assert_eq!(format_ttft(Some(Duration::from_millis(180))), "180 ms");
        assert_eq!(format_ttft(Some(Duration::from_millis(1500))), "1.50 s");
        assert_eq!(format_ttft(None), "n/a");
    }

    #[test]
    fn format_tps_renders_rate_or_na() {
        assert_eq!(format_tps(Some(42.15)), "42.1 tok/s");
        assert_eq!(format_tps(None), "n/a");
        assert_eq!(format_tps(Some(0.0)), "n/a");
        assert_eq!(format_tps(Some(f64::INFINITY)), "n/a");
    }

    #[test]
    fn summary_shows_endpoint_model_and_metrics() {
        let rendered = render_summary(&base_summary());
        assert!(rendered.contains("Deployment summary"));
        assert!(rendered.contains("http://127.0.0.1:1337/v1/chat/completions"));
        assert!(rendered.contains("Qwen/Qwen2.5-7B-Instruct"));
        assert!(rendered.contains("180 ms"));
        assert!(rendered.contains("42.1 tok/s"));
        assert!(rendered.contains("rocm services stop vllm-qwen-1720000000"));
    }

    #[test]
    fn summary_structure_is_identical_across_engines() {
        // The same fields render in the same order regardless of engine; only the
        // engine/model/endpoint values differ. Compare the row *labels*.
        fn labels(text: &str) -> Vec<String> {
            text.lines()
                .filter(|line| line.starts_with("  ") && !line.starts_with("  note:"))
                .map(|line| {
                    line.trim_start()
                        .split("  ")
                        .next()
                        .unwrap_or("")
                        .to_owned()
                })
                .collect()
        }

        let mut vllm = base_summary();
        vllm.engine = "vllm".to_owned();
        let mut lemonade = base_summary();
        lemonade.engine = "lemonade".to_owned();
        lemonade.api_model = "Qwen3-0.6B-GGUF".to_owned();
        lemonade.chat_endpoint = "http://127.0.0.1:8000/v1/chat/completions".to_owned();
        lemonade.metrics = SmokeMetrics::default(); // engine reported no usage

        assert_eq!(
            labels(&render_summary(&vllm)),
            labels(&render_summary(&lemonade))
        );
        // Even when metrics are missing, the rows still exist and read n/a.
        let rendered = render_summary(&lemonade);
        assert!(rendered.contains("time to first token"));
        assert!(rendered.contains("n/a"));
    }

    #[test]
    fn already_running_is_flagged_in_the_heading() {
        let mut summary = base_summary();
        summary.already_running = true;
        assert!(render_summary(&summary).contains("already running"));
    }

    #[test]
    fn readiness_timeout_does_not_read_as_success() {
        // A launch that never became ready lands here with status "starting". The
        // heading must flag it and a note must point the user at the logs, so the
        // summary is not mistaken for a healthy deployment.
        let mut summary = base_summary();
        summary.status = "starting".to_owned();
        let rendered = render_summary(&summary);
        assert!(rendered.contains("not ready yet"), "heading: {rendered}");
        assert!(rendered.contains("did not report ready"));
        assert!(rendered.contains("rocm logs --service"));
    }

    #[test]
    fn throughput_row_is_labelled_approximate() {
        assert!(render_summary(&base_summary()).contains("throughput (approx)"));
    }

    #[test]
    fn notes_are_rendered_when_present() {
        let mut summary = base_summary();
        summary.notes = vec!["selected GPU 0 has low free VRAM".to_owned()];
        assert!(render_summary(&summary).contains("note: selected GPU 0 has low free VRAM"));
    }

    #[test]
    fn compute_metrics_derives_ttft_and_throughput() {
        let start = Instant::now();
        let first = start + Duration::from_millis(200);
        let last = first + Duration::from_millis(500);
        let metrics = compute_metrics(start, Some(first), Some(last), 11);
        assert_eq!(metrics.ttft, Some(Duration::from_millis(200)));
        // 10 tokens after the first over 0.5s => 20 tok/s.
        let tps = metrics.gen_tps.expect("throughput present");
        assert!((tps - 20.0).abs() < 0.001, "unexpected tps {tps}");
    }

    #[test]
    fn compute_metrics_without_tokens_is_empty() {
        let start = Instant::now();
        assert_eq!(
            compute_metrics(start, None, None, 0),
            SmokeMetrics::default()
        );
    }

    #[test]
    fn compute_metrics_single_token_has_ttft_but_no_throughput() {
        let start = Instant::now();
        let first = start + Duration::from_millis(120);
        let metrics = compute_metrics(start, Some(first), Some(first), 1);
        assert_eq!(metrics.ttft, Some(Duration::from_millis(120)));
        assert_eq!(metrics.gen_tps, None);
    }
}
