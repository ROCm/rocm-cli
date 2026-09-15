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
use std::time::{Duration, Instant};

use rocm_core::AppPaths;

use crate::providers::{self, ChatMessage, ChatRequest, ProviderStreamEvent};

/// Tiny prompt used for the startup smoke test. Kept short so the probe adds only
/// a second or two to an otherwise-ready deployment.
const SMOKE_PROMPT: &str = "Reply with a short one-sentence greeting.";
/// Cap the smoke-test generation so a slow or verbose model cannot stall startup.
const SMOKE_MAX_TOKENS: u32 = 32;

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
    /// The model reference the user requested on the command line.
    pub requested_model: String,
    /// Canonical / API-qualified model id clients pass as `"model"`. May differ
    /// from `requested_model` when a recipe alias resolves to a canonical id.
    pub api_model: String,
    /// Full chat-completions endpoint, e.g. `http://127.0.0.1:1337/v1/chat/completions`.
    pub chat_endpoint: String,
    pub service_id: String,
    /// `"ready"`, `"starting"`, or an existing service's status.
    pub status: String,
    /// True when an equivalent server was already running and nothing was spawned.
    pub already_running: bool,
    pub metrics: SmokeMetrics,
    /// The endpoint API key for a freshly launched *public* server, shown once as
    /// secure client configuration. `None` for loopback binds (no auth) and for an
    /// already-running service (whose key was delivered at its own launch).
    pub api_key: Option<String>,
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
    // When the requested reference resolved to a different canonical id, show
    // both so a silent substitution can never masquerade as the requested model;
    // when they match, a single `model` row keeps the common case uncluttered.
    let model_rows: Vec<(&str, String)> = if summary.requested_model == summary.api_model {
        vec![("model", summary.api_model.clone())]
    } else {
        vec![
            ("requested model", summary.requested_model.clone()),
            ("resolved model", summary.api_model.clone()),
        ]
    };

    let rows: Vec<(&str, String)> = [
        vec![
            ("status", summary.status.clone()),
            ("engine", summary.engine.clone()),
        ],
        model_rows,
        vec![
            ("endpoint", summary.chat_endpoint.clone()),
            ("time to first token", format_ttft(summary.metrics.ttft)),
            ("throughput (approx)", format_tps(summary.metrics.gen_tps)),
            ("service", summary.service_id.clone()),
            ("stop", format!("rocm services stop {}", summary.service_id)),
            (
                "logs",
                format!("rocm logs --service {}", summary.service_id),
            ),
        ],
    ]
    .concat();

    let label_width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);

    let mut out = String::new();
    out.push_str(heading);
    out.push('\n');
    for (label, value) in rows {
        let _ = writeln!(out, "  {label:<label_width$}  {value}");
    }
    if let Some(api_key) = summary.api_key.as_deref() {
        // Secure client configuration: the intended one-time channel for handing
        // the key to the user. It is printed here to the terminal only — never to
        // logs, `rocm services`, or `rocm logs`.
        let _ = writeln!(out, "  api key: {api_key}");
        let _ = writeln!(
            out,
            "  note: this key is shown only now — clients must send `Authorization: Bearer <key>`"
        );
    }
    if !ready && !summary.already_running {
        // Two different ways to miss "ready", and conflating them misleads: a
        // `running` service answered and advertised the model but could not serve
        // a request yet, which is also why the metrics rows are empty — the smoke
        // test only runs against a service that can actually serve.
        let explanation = if summary.status == "running" {
            "the server is up and lists the model, but it could not serve a request yet, \
             so no smoke test was run; it is most likely still loading"
        } else {
            "the server did not report ready before the startup timeout; it may still be loading"
        };
        let _ = writeln!(
            out,
            "  note: {explanation} — check `rocm logs --service {}` or `rocm services list`",
            summary.service_id
        );
    }
    for note in &summary.notes {
        let _ = writeln!(out, "  note: {note}");
    }
    out
}

/// Whether a managed service's recorded status means it *failed to become
/// ready* — the only state the post-failure OOM note should fire on.
///
/// The status vocabulary from `status_for_readiness` is `ready` (serving),
/// `running` (endpoint up, model still loading — healthy, deliberately kept
/// distinct from `starting` so `rocmd` does not restart a slow-loading model),
/// and `starting` (endpoint never came up). Only `starting` is a failure; a
/// bare `!= "ready"` check would wrongly treat a healthy still-loading `running`
/// service as a failed launch.
pub(crate) fn serve_failed_to_become_ready(status: &str) -> bool {
    status == "starting"
}

/// Builds the actionable memory-knob note for a serve that failed to become
/// ready with an out-of-memory signature in its engine log. Returns `None` when
/// the serve became ready or the log carries no OOM signature, so healthy
/// deployments and unrelated failures are never cluttered with memory advice.
///
/// The note names `--gpu-memory-utilization` and `--gpu` and is worded for the
/// shared-node case rather than as unconditional advice — vLLM reserves a
/// fraction of *total* VRAM, so a value good for a shared card would degrade a
/// dedicated one. When the model simply does not fit, it says so and points at a
/// smaller/quantized model instead of the knob (which would only trade an
/// earlier OOM for a later one), matching the `rocm diagnose` vLLM-OOM entry it
/// then routes the user to. It shares
/// [`rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT`] verbatim with the pre-launch
/// low-VRAM note so both surfaces point at the same fix.
///
/// `hint_already_present` says that shared hint is already in the summary's
/// notes (the pre-launch low-VRAM warning added it), and de-duplicates *that
/// fragment only*: the rest of the note is new information the pre-launch guess
/// does not carry — that this attempt really did run out of GPU memory rather
/// than might, the "if the model doesn't fit, lowering the reservation won't
/// help" branch, and the `rocm diagnose --symptom` command with the user's own
/// failing line. Low VRAM leading to an OOM is the causal chain this note exists
/// for, so suppressing the whole note there would silence it on its most likely
/// trigger.
pub(crate) fn oom_memory_note(
    status: &str,
    log_tail: &str,
    hint_already_present: bool,
) -> Option<String> {
    if !serve_failed_to_become_ready(status) {
        return None;
    }
    // The symptom is vLLM's own subprocess output and it lands inside a
    // single-quoted shell word in a sentence that invites the user to paste the
    // command, so it is only routed through verbatim when it can be rendered as
    // one intact quoted argument; otherwise the canonical symptom stands in. See
    // [`rocm_core::quotable_in_single_quotes`] for why this rejects rather than
    // escapes. The engine's startup-failure hint guards the same text the same
    // way.
    //
    // `None` is also the "this tail shows no OOM" answer, so it doubles as the
    // OOM gate: asking `vllm_log_shows_oom` first would evaluate the same
    // predicate over the same string twice.
    let symptom = rocm_core::vllm_oom_diagnose_symptom(log_tail)?;
    let symptom = if rocm_core::quotable_in_single_quotes(&symptom) {
        symptom
    } else {
        rocm_core::VLLM_OOM_CANONICAL_SYMPTOM.to_owned()
    };
    // Printed as its own sentence, or omitted when the pre-launch warning
    // already printed the identical text.
    let hint = if hint_already_present {
        String::new()
    } else {
        format!("{} ", rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT)
    };
    Some(format!(
        "the serve attempt ran out of GPU memory. {hint}If the model simply does not fit in this \
         GPU's VRAM, lowering the reservation will not help — serve a smaller or quantized model \
         instead (rocm-cli serves one model on a single GPU). To have the tool pick the \
         case-appropriate fix, run `rocm diagnose --symptom '{symptom}'`."
    ))
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
        temperature: None,
        top_p: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_summary() -> DeploymentSummary {
        DeploymentSummary {
            engine: "vllm".to_owned(),
            requested_model: "Qwen/Qwen2.5-7B-Instruct".to_owned(),
            api_model: "Qwen/Qwen2.5-7B-Instruct".to_owned(),
            chat_endpoint: "http://127.0.0.1:1337/v1/chat/completions".to_owned(),
            service_id: "vllm-qwen-1720000000".to_owned(),
            status: "ready".to_owned(),
            already_running: false,
            metrics: SmokeMetrics {
                ttft: Some(Duration::from_millis(180)),
                gen_tps: Some(42.15),
            },
            api_key: None,
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
        lemonade.requested_model = "Qwen3-0.6B-GGUF".to_owned();
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
    fn matching_request_and_resolved_show_a_single_model_row() {
        let rendered = render_summary(&base_summary());
        assert!(rendered.contains("  model  "));
        assert!(!rendered.contains("requested model"));
        assert!(!rendered.contains("resolved model"));
    }

    #[test]
    fn divergent_request_and_resolved_are_both_reported() {
        // A recipe alias that resolves to a different canonical id must surface
        // both identities so a substitution is never silent (EAI-7370).
        let mut summary = base_summary();
        summary.requested_model = "qwen".to_owned();
        summary.api_model = "Qwen3-4B-Instruct-2507-GGUF".to_owned();
        let rendered = render_summary(&summary);
        let row = |label: &str, value: &str| {
            rendered.lines().any(|line| {
                line.trim_start().starts_with(label) && line.trim_end().ends_with(value)
            })
        };
        assert!(row("requested model", "qwen"), "{rendered}");
        assert!(
            row("resolved model", "Qwen3-4B-Instruct-2507-GGUF"),
            "{rendered}"
        );
        // The bare `model` row is not used when the two differ.
        assert!(
            !rendered
                .lines()
                .any(|line| line.trim_start().starts_with("model ")),
            "{rendered}"
        );
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
    fn a_still_loading_launch_explains_the_missing_smoke_test() {
        // A launch whose model lists but cannot serve yet lands at "running". The
        // smoke test is skipped by design (there is nothing to measure), so the
        // empty metrics rows must be explained rather than left to read as a
        // failed measurement against a healthy server.
        let mut summary = base_summary();
        summary.status = "running".to_owned();
        summary.metrics = SmokeMetrics::default();
        let rendered = render_summary(&summary);
        assert!(rendered.contains("not ready yet"), "heading: {rendered}");
        assert!(
            rendered.contains("no smoke test was run"),
            "the skipped smoke test must be explained: {rendered}"
        );
        assert!(
            rendered.contains("still loading"),
            "the user needs to know this resolves on its own: {rendered}"
        );
        assert!(rendered.contains("rocm services list"));
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
    fn oom_signatures_are_detected_case_insensitively() {
        // The signatures the ticket calls out, plus casing variants the engine
        // log can emit.
        assert!(rocm_core::vllm_log_shows_oom(
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
        ));
        assert!(rocm_core::vllm_log_shows_oom("HIP OUT OF MEMORY"));
        assert!(rocm_core::vllm_log_shows_oom(
            "RuntimeError: CUDA out of memory"
        ));
    }

    #[test]
    fn unrelated_failures_are_not_flagged_as_oom() {
        assert!(!rocm_core::vllm_log_shows_oom(
            "OSError: model weights not found; check the model id"
        ));
        assert!(!rocm_core::vllm_log_shows_oom(""));
        // vLLM's generic EngineCore wrapper is the terminal line for *any*
        // startup crash, not just OOM; treating it as an OOM signature would
        // misreport unrelated failures as memory exhaustion.
        assert!(!rocm_core::vllm_log_shows_oom(
            "ERROR Engine core initialization failed"
        ));
    }

    #[test]
    fn oom_note_names_both_memory_knobs_on_a_failed_serve() {
        let note = oom_memory_note(
            "starting",
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            false,
        )
        .expect("an OOM failure must produce a note");
        assert!(note.contains("--gpu-memory-utilization"), "{note}");
        assert!(note.contains("--gpu <index>"), "{note}");
        assert!(note.contains("ran out of GPU memory"), "{note}");
        // The knob is not unconditional: when the model simply does not fit, the
        // note must say lowering the reservation will not help and point at a
        // smaller/quantized model, matching the `rocm diagnose` vLLM-OOM entry.
        assert!(
            note.contains("smaller or quantized model"),
            "the note must carry the model-too-large caveat: {note}"
        );
        assert!(
            note.contains("rocm diagnose --symptom"),
            "the note must route the user to the conditional diagnose entry: {note}"
        );
    }

    #[test]
    fn oom_note_is_withheld_for_a_ready_serve_or_a_clean_log() {
        // A serve that became ready is healthy even if the log mentions memory.
        assert_eq!(
            oom_memory_note("ready", "torch.OutOfMemoryError: HIP out of memory", false),
            None
        );
        // A failure with no OOM signature must not be given memory advice.
        assert_eq!(
            oom_memory_note("starting", "OSError: model weights not found", false),
            None
        );
        // ...and neither case becomes advisable just because the pre-launch
        // low-VRAM warning already fired.
        assert_eq!(
            oom_memory_note("ready", "torch.OutOfMemoryError: HIP out of memory", true),
            None
        );
        assert_eq!(
            oom_memory_note("starting", "OSError: model weights not found", true),
            None
        );
    }

    #[test]
    fn the_shared_hint_is_de_duplicated_without_losing_the_rest_of_the_oom_note() {
        // The pre-launch low-VRAM warning already printed the shared hint
        // verbatim, and low VRAM leading to an OOM is the causal chain this note
        // exists for -- so what must be dropped is that one fragment, not the
        // note. Everything else it carries is information the pre-launch guess
        // does not have: that this attempt actually ran out of GPU memory, the
        // "if the model doesn't fit, lowering the reservation won't help"
        // branch, and the diagnose command with the user's real failing line.
        let log_tail = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let note = oom_memory_note("starting", log_tail, true)
            .expect("an OOM failure must still carry a note when the hint was already printed");

        // The hint the pre-launch warning printed appears zero further times...
        assert!(
            !note.contains(rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT),
            "the shared hint must not be printed a second time: {note}"
        );
        // ...and the content only this note has must all survive.
        assert!(
            note.contains("ran out of GPU memory"),
            "the note must confirm this attempt really did OOM: {note}"
        );
        assert!(
            note.contains("smaller or quantized model"),
            "the model-too-large branch is absent from the pre-launch hint: {note}"
        );
        assert_eq!(
            quoted_symptom_argument(&note),
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            "the diagnose command must carry the user's own failing line: {note}"
        );

        // Counted across the whole summary, the hint is printed exactly once:
        // the pre-launch note keeps it, the OOM note does not repeat it.
        let pre_launch = rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT.to_owned();
        let printed = [pre_launch, note]
            .iter()
            .filter(|line| line.contains(rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT))
            .count();
        assert_eq!(printed, 1, "the shared hint must appear exactly once");
    }

    /// The `--symptom` value the note actually hands the user, read back out of
    /// the rendered text exactly the way a shell would: the note's prose carries
    /// its own apostrophes (`GPU's VRAM`), so the command is extracted from
    /// between its backticks first, then the first `'...'` word inside it.
    fn quoted_symptom_argument(note: &str) -> &str {
        note.split("run `")
            .nth(1)
            .and_then(|rest| rest.split('`').next())
            .expect("the note must print a runnable command")
            .split("--symptom '")
            .nth(1)
            .and_then(|rest| rest.split('\'').next())
            .expect("the command must carry a --symptom value")
    }

    #[test]
    fn an_apostrophe_in_the_failing_line_cannot_break_out_of_the_printed_command() {
        // A realistic vLLM traceback tail: apostrophes are routine in Python
        // error text, and this line scores well above MIN_SCORE_FOR_MATCH
        // (torch.OutOfMemoryError + HIP out of memory), so the "route the user's
        // real line" branch selects it.
        let log_tail = concat!(
            "  File \"/opt/vllm/worker.py\", line 212, in load_model\n",
            "ERROR 09-14 12:00:01 engine.py:389] torch.OutOfMemoryError: HIP out of memory. ",
            "Tried to allocate 7.21 GiB. GPU 0 can't allocate the model's weights.\n"
        );
        let note =
            oom_memory_note("starting", log_tail, false).expect("an OOM failure must carry a note");

        // The rendered command must be one intact single-quoted argument: no
        // byte the vLLM subprocess printed may close the quote and land outside
        // it in a command the note invites the user to paste.
        let command = note
            .split("run `")
            .nth(1)
            .and_then(|rest| rest.split('`').next())
            .expect("the note must print a runnable command");
        assert_eq!(
            command.matches('\'').count(),
            2,
            "the --symptom argument must stay a single balanced quoted word: {command}"
        );
        // Pin which branch ran, not just that no apostrophe survived: "contains
        // no `'`" also holds if the apostrophes were silently deleted from the
        // user's line, which is the escaping-style behaviour the fallback exists
        // to avoid. Rejection means the canonical symptom, exactly.
        let symptom = quoted_symptom_argument(&note);
        assert_eq!(
            symptom,
            rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            "a quote-bearing line must be rejected in favour of the canonical symptom, \
             not silently rewritten: {symptom:?}"
        );
        // ...and the command it does print must still report a cause.
        assert!(
            rocm_core::vllm_oom_symptom_is_diagnosable(symptom),
            "the fallback symptom must still be diagnosable: {symptom:?}"
        );
    }

    #[test]
    fn control_bytes_from_the_log_never_reach_the_printed_command() {
        // vLLM's logger colourises; an ANSI-coloured OOM line must not repaint
        // the user's terminal from inside rocm-cli's own serve summary.
        let log_tail = "\u{1b}[31mRuntimeError: HIP out of memory\u{1b}[0m\u{7}";
        let note =
            oom_memory_note("starting", log_tail, false).expect("an OOM failure must carry a note");
        assert!(
            !note.chars().any(|c| c.is_control() && c != '\n'),
            "no control byte may survive into the printed note: {note:?}"
        );
        // Pin the exact value rather than the absence of a few fragments: an
        // absence check cannot fail for the defect it names, since a stripper
        // that drops only the escape byte and the `[` leaves `31m`/`0m` behind,
        // which contains neither `[31m` nor `[0m` and carries no control byte.
        let symptom = quoted_symptom_argument(&note);
        assert_eq!(
            symptom,
            rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            "a control-byte-bearing line must be rejected in favour of the canonical \
             symptom, not stripped into a lookalike: {symptom:?}"
        );
    }

    #[test]
    fn oom_note_quotes_a_clean_failing_line_verbatim() {
        // The guard must not cost the common case its own error text: a line
        // with no quote and no control byte is still routed into the command.
        let note = oom_memory_note(
            "starting",
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            false,
        )
        .expect("an OOM failure must carry a note");
        assert_eq!(
            quoted_symptom_argument(&note),
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
        );
    }

    #[test]
    fn oom_note_renders_in_the_summary_notes() {
        let mut summary = base_summary();
        summary.status = "starting".to_owned();
        if let Some(note) = oom_memory_note(
            &summary.status,
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            false,
        ) {
            summary.notes.push(note);
        }
        let rendered = render_summary(&summary);
        assert!(rendered.contains("note: the serve attempt ran out of GPU memory"));
        assert!(rendered.contains("--gpu-memory-utilization"));
    }

    #[test]
    fn api_key_client_config_shown_only_when_present() {
        // Loopback / no key: the summary must not mention an api key at all.
        let plain = render_summary(&base_summary());
        assert!(!plain.contains("api key"), "{plain}");

        // Public bind: the key is shown once as secure client config.
        let mut summary = base_summary();
        summary.api_key = Some("endpoint-secret".to_owned());
        let rendered = render_summary(&summary);
        assert!(rendered.contains("api key: endpoint-secret"), "{rendered}");
        assert!(rendered.contains("shown only now"), "{rendered}");
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
