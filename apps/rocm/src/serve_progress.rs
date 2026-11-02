// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Interactive `rocm serve` startup progress.
//!
//! Summary mode used to tick one static "Starting …" label and, after 45
//! seconds, print a finished "not ready yet" summary while the server kept
//! loading. This module decides the status line from the endpoint and the
//! engine log, and when that wait is allowed to stop.
//!
//! State matrix for the wait. Every pending row has an exit when nothing
//! new arrives; a download that stops advancing does not spin until the
//! overall deadline.
//!
//! | State | What we observe | Status line | Exit if data never arrives |
//! | --- | --- | --- | --- |
//! | unreachable | port down, no meaningful log, process alive | `Starting {model} on {engine}…` | silent limit, or the deadline |
//! | downloading | latest log line is a download, the endpoint is not listing the model, and progress is still moving | that log line | stall limit once the line (or its percent) freezes |
//! | loading | latest log line is anything else meaningful, port still down | that log line | silent-load limit once the line freezes |
//! | listing | HTTP lists the model, inference has not answered | the log line, or `Waiting for {model} to serve…` | the overall deadline (the server is answering) |
//! | serving | inference answered | wait returns ready | immediate |
//! | failed | engine state `failed`, or every recorded pid is dead | last log line, or a fixed failure line | immediate |
//!
//! A log line that still says "download" does not pull a listed model back
//! into the download stall. The server has answered; the exit stays the
//! deadline.
//!
//! A piped serve keeps the historical status budget and returns the best
//! readiness instead of this failure. That path has no status line; holding
//! it for a full model load would block CI and the assistant on a server
//! that never answers.

use std::fmt::Write as _;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use rocm_core::EndpointReadiness;

use crate::engine_runtime_select::probe_service_http_readiness;
use crate::service_hints::{service_logs_hint, service_stop_hint};

/// How long a download percent (or line) may stay unchanged before it is hung.
pub(crate) const DOWNLOAD_STALL: Duration = Duration::from_mins(3);

/// Piped / captured `rocm serve` returns the best readiness after this long.
/// Interactive summary mode follows the stage instead.
pub(crate) const PIPED_SERVE_STATUS_BUDGET: Duration = Duration::from_secs(45);

/// Port down and no meaningful log line.
pub(crate) const SILENT_UNREACHABLE: Duration = Duration::from_mins(2);

/// A load line that stops changing while the port is still down.
pub(crate) const SILENT_LOAD: Duration = Duration::from_mins(5);

const POLL: Duration = Duration::from_millis(250);
const LOG_TAIL_LINES: usize = 40;
const LOG_TAIL_BYTES: u64 = 64 * 1024;
const MAX_LABEL_CHARS: usize = 160;

/// How the interactive wait treats time, and whether giving up is an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServeStartupPolicy {
    pub bounds: ServeStartupBounds,
    pub poll: Duration,
    /// Stop with an error at the deadline and the stall limits.
    ///
    /// `false` for piped output: the deadline returns the best readiness, the
    /// way the 45-second wait did.
    pub stop_on_give_up: bool,
    /// Read engine-failed and process-liveness.
    ///
    /// Off for the piped path so a serve that exits in CI still returns a
    /// readiness status.
    pub watch_engine: bool,
}

/// The clocks [`decide_serve_startup`] compares against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServeStartupBounds {
    pub deadline: Duration,
    pub download_stall: Duration,
    pub silent_unreachable: Duration,
    pub silent_load: Duration,
}

impl ServeStartupPolicy {
    /// Terminal summary: follow the stage until ready or a bounded failure.
    pub(crate) const fn interactive() -> Self {
        Self {
            bounds: ServeStartupBounds {
                deadline: rocm_core::SERVE_STARTUP_DEADLINE,
                download_stall: DOWNLOAD_STALL,
                silent_unreachable: SILENT_UNREACHABLE,
                silent_load: SILENT_LOAD,
            },
            poll: POLL,
            stop_on_give_up: true,
            watch_engine: true,
        }
    }

    /// Piped / captured output. `timeout` is the historical readiness budget.
    pub(crate) const fn status_budget(timeout: Duration) -> Self {
        Self {
            bounds: ServeStartupBounds {
                deadline: timeout,
                download_stall: Duration::MAX,
                silent_unreachable: Duration::MAX,
                silent_load: Duration::MAX,
            },
            poll: POLL,
            stop_on_give_up: false,
            watch_engine: false,
        }
    }
}

/// One sample of engine and process state, separate from the HTTP probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServeStartupSignals {
    pub engine_failed: bool,
    pub process_alive: bool,
    /// Latest meaningful log line, already one sanitized line.
    pub log_line: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServeStartupTiming {
    pub elapsed: Duration,
    pub since_progress: Duration,
}

/// Lifecycle row the status line is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServeStartupPhase {
    Unreachable,
    Downloading,
    Loading,
    Listing,
    Serving,
    Failed,
}

/// What the spinner should show for this sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServeStartupView {
    pub phase: ServeStartupPhase,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServeStartupStopReason {
    EngineFailed,
    ProcessExited,
    DownloadStalled,
    Silent,
    SilentLoad,
    Deadline,
}

/// The wait stopped before the server could serve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServeStartupStop {
    pub phase: ServeStartupPhase,
    pub reason: ServeStartupStopReason,
    /// Spinner label for this sample.
    ///
    /// May contain the spinner ellipsis. The failure text quotes
    /// [`Self::log_line`] instead, which is ASCII.
    pub label: String,
    pub log_line: Option<String>,
    pub process_alive: bool,
    pub readiness: EndpointReadiness,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServeStartupDecision {
    Continue(ServeStartupView),
    Ready,
    /// Piped path: the status budget elapsed. Hand back the best readiness.
    BudgetElapsed(EndpointReadiness),
    Stop(ServeStartupStop),
}

pub(crate) fn starting_label(model: &str, engine: &str) -> String {
    format!("Starting {model} on {engine}…")
}

pub(crate) fn waiting_label(model: &str) -> String {
    format!("Waiting for {model} to serve…")
}

/// Latest meaningful engine-log line, sanitized to one ASCII line.
///
/// Carriage-return progress bars keep only their last segment. ANSI and
/// other control bytes are stripped. Blank lines and lines with no words
/// are skipped so a rule of dashes cannot cover a real stage.
pub(crate) fn latest_meaningful_log_line(path: &Path) -> Option<String> {
    let lines = rocm_core::tail_lines(path, LOG_TAIL_LINES, LOG_TAIL_BYTES).ok()?;
    lines.into_iter().rev().find_map(|line| {
        let sanitized = sanitize_log_line(&line);
        is_meaningful(&sanitized).then_some(sanitized)
    })
}

pub(crate) fn sanitize_log_line(raw: &str) -> String {
    let segment = raw
        .rsplit(['\r', '\n'])
        .find(|part| !part.trim().is_empty())
        .unwrap_or(raw);
    let stripped = rocm_core::terminal::strip_terminal_control_sequences(segment);
    let ascii: String = stripped.chars().filter(char::is_ascii).collect();
    let collapsed = ascii.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_label(&collapsed)
}

fn truncate_label(line: &str) -> String {
    if line.chars().count() <= MAX_LABEL_CHARS {
        return line.to_owned();
    }
    let mut out: String = line
        .chars()
        .take(MAX_LABEL_CHARS.saturating_sub(3))
        .collect();
    out.push_str("...");
    out
}

fn is_meaningful(line: &str) -> bool {
    line.chars().filter(char::is_ascii_alphabetic).count() >= 4 && line.len() >= 6
}

fn line_is_download(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("download") || lower.contains("fetching ") || lower.contains("pulling ")
}

/// Last `NN%` in the line, so a bar whose ETA keeps moving but whose percent
/// does not is still stalled.
fn download_fingerprint(line: &str) -> String {
    last_percent(line).map_or_else(|| line.to_owned(), |pct| format!("pct:{pct}"))
}

fn last_percent(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    let mut search_end = bytes.len();
    while let Some(rel) = line[..search_end].rfind('%') {
        let mut start = rel;
        while start > 0 && bytes[start - 1].is_ascii_digit() {
            start -= 1;
        }
        if start < rel {
            return Some(&line[start..rel]);
        }
        search_end = rel;
    }
    None
}

fn progress_token(readiness: EndpointReadiness, signals: &ServeStartupSignals) -> String {
    if signals.engine_failed {
        return "engine-failed".to_owned();
    }
    if !signals.process_alive {
        return "process-exited".to_owned();
    }
    if readiness == EndpointReadiness::Listing {
        return format!("listing:{}", signals.log_line.as_deref().unwrap_or(""));
    }
    if let Some(line) = signals
        .log_line
        .as_deref()
        .filter(|line| line_is_download(line))
    {
        return format!("download:{}", download_fingerprint(line));
    }
    if let Some(line) = &signals.log_line {
        return format!("log:{line}");
    }
    "unreachable".to_owned()
}

fn classify(
    model: &str,
    engine: &str,
    readiness: EndpointReadiness,
    signals: &ServeStartupSignals,
) -> ServeStartupView {
    if signals.engine_failed || !signals.process_alive {
        let label = signals.log_line.clone().unwrap_or_else(|| {
            if signals.engine_failed {
                "the engine reported failed".to_owned()
            } else {
                "the server process exited".to_owned()
            }
        });
        return ServeStartupView {
            phase: ServeStartupPhase::Failed,
            label,
        };
    }
    if readiness == EndpointReadiness::Serving {
        return ServeStartupView {
            phase: ServeStartupPhase::Serving,
            label: starting_label(model, engine),
        };
    }
    // Listing wins over a stale download line. vLLM advertises the model
    // before inference answers, and the last log line often still says
    // "download". That must not arm the download stall.
    if readiness == EndpointReadiness::Listing {
        return ServeStartupView {
            phase: ServeStartupPhase::Listing,
            label: signals
                .log_line
                .clone()
                .unwrap_or_else(|| waiting_label(model)),
        };
    }
    if let Some(line) = signals
        .log_line
        .as_ref()
        .filter(|line| line_is_download(line))
    {
        return ServeStartupView {
            phase: ServeStartupPhase::Downloading,
            label: line.clone(),
        };
    }
    if let Some(line) = &signals.log_line {
        return ServeStartupView {
            phase: ServeStartupPhase::Loading,
            label: line.clone(),
        };
    }
    ServeStartupView {
        phase: ServeStartupPhase::Unreachable,
        label: starting_label(model, engine),
    }
}

fn quoted_log(model: &str, engine: &str, view: &ServeStartupView) -> Option<String> {
    if view.label == starting_label(model, engine) || view.label == waiting_label(model) {
        return None;
    }
    match view.label.as_str() {
        "the engine reported failed" | "the server process exited" => None,
        _ => Some(view.label.clone()),
    }
}

fn stop_decision(
    model: &str,
    engine: &str,
    view: &ServeStartupView,
    reason: ServeStartupStopReason,
    process_alive: bool,
    readiness: EndpointReadiness,
) -> ServeStartupDecision {
    ServeStartupDecision::Stop(ServeStartupStop {
        phase: view.phase,
        reason,
        log_line: quoted_log(model, engine, view),
        label: view.label.clone(),
        process_alive,
        readiness,
    })
}

/// Decide the next status line and whether the wait stops.
///
/// `readiness` is the best probe so far, never a downgrade: one refused
/// connection after the model has listed does not put the line back on
/// "Starting" or arm the silent-unreachable exit.
pub(crate) fn decide_serve_startup(
    model: &str,
    engine: &str,
    readiness: EndpointReadiness,
    signals: &ServeStartupSignals,
    timing: ServeStartupTiming,
    policy: &ServeStartupPolicy,
) -> ServeStartupDecision {
    let mut signals = signals.clone();
    if !policy.watch_engine {
        signals.engine_failed = false;
        signals.process_alive = true;
    }
    let view = classify(model, engine, readiness, &signals);
    let process_alive = signals.process_alive;

    if signals.engine_failed {
        return stop_decision(
            model,
            engine,
            &view,
            ServeStartupStopReason::EngineFailed,
            process_alive,
            readiness,
        );
    }
    if !process_alive {
        return stop_decision(
            model,
            engine,
            &view,
            ServeStartupStopReason::ProcessExited,
            process_alive,
            readiness,
        );
    }
    if readiness == EndpointReadiness::Serving || view.phase == ServeStartupPhase::Serving {
        return ServeStartupDecision::Ready;
    }
    if timing.elapsed >= policy.bounds.deadline {
        return if policy.stop_on_give_up {
            stop_decision(
                model,
                engine,
                &view,
                ServeStartupStopReason::Deadline,
                process_alive,
                readiness,
            )
        } else {
            ServeStartupDecision::BudgetElapsed(readiness)
        };
    }
    if view.phase == ServeStartupPhase::Downloading
        && timing.since_progress >= policy.bounds.download_stall
    {
        return stop_decision(
            model,
            engine,
            &view,
            ServeStartupStopReason::DownloadStalled,
            process_alive,
            readiness,
        );
    }
    if view.phase == ServeStartupPhase::Unreachable
        && timing.elapsed >= policy.bounds.silent_unreachable
    {
        return stop_decision(
            model,
            engine,
            &view,
            ServeStartupStopReason::Silent,
            process_alive,
            readiness,
        );
    }
    if view.phase == ServeStartupPhase::Loading
        && timing.since_progress >= policy.bounds.silent_load
    {
        return stop_decision(
            model,
            engine,
            &view,
            ServeStartupStopReason::SilentLoad,
            process_alive,
            readiness,
        );
    }
    ServeStartupDecision::Continue(view)
}

/// The status written to the service record when the wait stops.
///
/// A live process stays `running`. `starting` is what `rocmd` restarts once
/// the stale-start window elapses
/// (`find_recoverable_service_detects_stale_starting_manifest`), which would
/// kill the server [`ServeStartupStop::user_message`] just said is still
/// running. `running` with a live pid is what
/// `recovery_leaves_a_live_startup_alone_and_still_restarts_a_dead_one`
/// leaves alone until [`rocm_core::SERVE_STARTUP_DEADLINE`]. Engine failure
/// and a dead process are `failed`, which recovery is supposed to restart.
pub(crate) fn recorded_status_after_stop(stop: &ServeStartupStop) -> &'static str {
    match stop.reason {
        ServeStartupStopReason::EngineFailed | ServeStartupStopReason::ProcessExited => "failed",
        _ if stop.process_alive => "running",
        _ => "starting",
    }
}

impl ServeStartupStop {
    /// Text the CLI prints when the wait stops.
    ///
    /// A log line is quoted only when [`Self::label`] is that line. The
    /// spinner's own "Starting" and "Waiting" labels are not quoted.
    pub(crate) fn user_message(&self, model: &str, engine: &str, service_id: &str) -> String {
        let minutes = rocm_core::SERVE_STARTUP_DEADLINE.as_secs() / 60;
        let headline = match self.reason {
            ServeStartupStopReason::EngineFailed => {
                format!("{model} on {engine} failed: the engine reported failed")
            }
            ServeStartupStopReason::ProcessExited => {
                format!("{model} on {engine} failed: the server process exited")
            }
            ServeStartupStopReason::DownloadStalled => {
                format!("{model} on {engine} stopped making download progress")
            }
            ServeStartupStopReason::Silent => {
                format!("{model} on {engine} produced no log and never answered")
            }
            ServeStartupStopReason::SilentLoad => {
                format!("{model} on {engine} stopped making progress while loading")
            }
            ServeStartupStopReason::Deadline => {
                format!("{model} on {engine} did not become ready within {minutes} minutes")
            }
        };
        let mut message = headline;
        if let Some(line) = &self.log_line {
            message.push_str(": ");
            message.push_str(line);
        }
        message.push('\n');
        let logs = service_logs_hint(service_id);
        if self.process_alive {
            let stop = service_stop_hint(service_id);
            let _ = write!(
                message,
                "The server is still running (service {service_id}). Follow it with `{logs}`, or stop it with `{stop}`."
            );
        } else {
            let _ = write!(
                message,
                "The server process is gone (service {service_id}). The log is `{logs}`."
            );
        }
        message
    }
}

/// Poll until the endpoint can serve, reporting each stage through `on_view`.
///
/// `observe` returns the model name the probe should ask for (a container
/// engine may not know it until it starts) and the latest engine/process
/// sample. The same HTTP probe as the historical readiness wait is used, so
/// "serving" still means an inference request came back.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wait_for_serve_startup(
    engine_name: &str,
    model: &str,
    host: &str,
    port: u16,
    endpoint_api_key: Option<&str>,
    policy: &ServeStartupPolicy,
    observe: &mut dyn FnMut() -> (String, ServeStartupSignals),
    on_view: &mut dyn FnMut(&ServeStartupView),
) -> Result<EndpointReadiness, ServeStartupStop> {
    let started = Instant::now();
    let mut progress_at = started;
    let mut token = String::new();
    let mut best = EndpointReadiness::Unreachable;

    loop {
        let (probe_model, signals) = observe();
        let signals = adjusted_signals(signals, policy);
        let next_token = progress_token(best, &signals);
        if next_token != token {
            token = next_token;
            progress_at = Instant::now();
        }
        match step(
            model,
            engine_name,
            best,
            &signals,
            timing(started, progress_at),
            policy,
            on_view,
        ) {
            LoopStep::Continue => {}
            LoopStep::Ready => return Ok(EndpointReadiness::Serving),
            LoopStep::Budget(readiness) => return Ok(readiness),
            LoopStep::Stop(stop) => return Err(stop),
        }

        match probe_service_http_readiness(engine_name, host, port, &probe_model, endpoint_api_key)
        {
            EndpointReadiness::Serving => return Ok(EndpointReadiness::Serving),
            EndpointReadiness::Listing => best = EndpointReadiness::Listing,
            EndpointReadiness::Unreachable => {}
        }

        let (_, signals) = observe();
        let signals = adjusted_signals(signals, policy);
        let next_token = progress_token(best, &signals);
        if next_token != token {
            token = next_token;
            progress_at = Instant::now();
        }
        match step(
            model,
            engine_name,
            best,
            &signals,
            timing(started, progress_at),
            policy,
            on_view,
        ) {
            LoopStep::Continue => {}
            LoopStep::Ready => return Ok(EndpointReadiness::Serving),
            LoopStep::Budget(readiness) => return Ok(readiness),
            LoopStep::Stop(stop) => return Err(stop),
        }

        let remaining = policy.bounds.deadline.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            continue;
        }
        thread::sleep(policy.poll.min(remaining));
    }
}

const fn adjusted_signals(
    mut signals: ServeStartupSignals,
    policy: &ServeStartupPolicy,
) -> ServeStartupSignals {
    if !policy.watch_engine {
        signals.engine_failed = false;
        signals.process_alive = true;
    }
    signals
}

fn timing(started: Instant, progress_at: Instant) -> ServeStartupTiming {
    ServeStartupTiming {
        elapsed: started.elapsed(),
        since_progress: progress_at.elapsed(),
    }
}

enum LoopStep {
    Continue,
    Ready,
    Budget(EndpointReadiness),
    Stop(ServeStartupStop),
}

fn step(
    model: &str,
    engine: &str,
    readiness: EndpointReadiness,
    signals: &ServeStartupSignals,
    timing: ServeStartupTiming,
    policy: &ServeStartupPolicy,
    on_view: &mut dyn FnMut(&ServeStartupView),
) -> LoopStep {
    // `decide_serve_startup` applies `watch_engine` again. The signals here
    // are already adjusted, and the policy still carries the flag, so a
    // second pass is a no-op. Passing the policy through keeps the stall
    // clocks on one struct.
    match decide_serve_startup(model, engine, readiness, signals, timing, policy) {
        ServeStartupDecision::Continue(view) => {
            on_view(&view);
            LoopStep::Continue
        }
        ServeStartupDecision::Ready => LoopStep::Ready,
        ServeStartupDecision::BudgetElapsed(readiness) => LoopStep::Budget(readiness),
        ServeStartupDecision::Stop(stop) => LoopStep::Stop(stop),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::service_hints::{service_logs_hint, service_stop_hint};
    use crate::test_support::unique_suffix;

    fn alive() -> ServeStartupSignals {
        ServeStartupSignals {
            engine_failed: false,
            process_alive: true,
            log_line: None,
        }
    }

    fn at(elapsed_ms: u64, since_ms: u64) -> ServeStartupTiming {
        ServeStartupTiming {
            elapsed: Duration::from_millis(elapsed_ms),
            since_progress: Duration::from_millis(since_ms),
        }
    }

    fn interactive() -> ServeStartupPolicy {
        ServeStartupPolicy::interactive()
    }

    fn assert_continue(decision: ServeStartupDecision) -> ServeStartupView {
        match decision {
            ServeStartupDecision::Continue(view) => view,
            other => panic!("expected the wait to keep going, got {other:?}"),
        }
    }

    fn assert_stop(decision: ServeStartupDecision) -> ServeStartupStop {
        match decision {
            ServeStartupDecision::Stop(stop) => stop,
            other => panic!("expected the wait to stop, got {other:?}"),
        }
    }

    #[test]
    fn unreachable_keeps_the_starting_label_until_the_silent_limit() {
        let view = assert_continue(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &alive(),
            at(1_000, 1_000),
            &interactive(),
        ));
        assert_eq!(view.phase, ServeStartupPhase::Unreachable);
        assert_eq!(view.label, starting_label("qwen", "vllm"));
    }

    #[test]
    fn unreachable_with_no_log_stops_when_nothing_arrives() {
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &alive(),
            at(
                SILENT_UNREACHABLE.as_millis() as u64,
                SILENT_UNREACHABLE.as_millis() as u64,
            ),
            &interactive(),
        ));
        assert_eq!(stop.reason, ServeStartupStopReason::Silent);
        assert_eq!(stop.phase, ServeStartupPhase::Unreachable);
        assert_eq!(stop.label, starting_label("qwen", "vllm"));
        let message = stop.user_message("qwen", "vllm", "svc-1");
        assert!(
            message.contains("produced no log and never answered"),
            "{message}"
        );
        assert!(message.contains("The server is still running"), "{message}");
        assert!(message.contains(&service_logs_hint("svc-1")), "{message}");
        assert!(message.contains(&service_stop_hint("svc-1")), "{message}");
        assert!(
            !message.contains("Deployment summary"),
            "a silent start must not become a finished summary: {message}"
        );
        // The stop command is the one the approval gate runs. That it stops
        // a live service is `@id:service-stop-yes-stops-a-running-service`
        // in tests/e2e-cucumber/features/service_stop.feature. The logs
        // command is `services_text::render_service_logs_text`, proven to
        // tail the service log by `render_service_logs_text_tails_manifest_log`.
    }

    #[test]
    fn downloading_past_the_old_budget_keeps_reporting_the_log_line() {
        let line = "Downloading model.safetensors: 40%";
        let mut signals = alive();
        signals.log_line = Some(line.to_owned());
        // 50s is past the old 45s budget. Progress moved 1s ago, so this is
        // not a hung download.
        let view = assert_continue(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(50_000, 1_000),
            &interactive(),
        ));
        assert_eq!(view.phase, ServeStartupPhase::Downloading);
        assert_eq!(view.label, line);
    }

    #[test]
    fn hung_download_stops_and_the_message_quotes_the_stalled_line() {
        let line = "Downloading model.safetensors: 12%";
        let mut signals = alive();
        signals.log_line = Some(line.to_owned());
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(
                DOWNLOAD_STALL.as_millis() as u64 + 1_000,
                DOWNLOAD_STALL.as_millis() as u64,
            ),
            &interactive(),
        ));
        assert_eq!(stop.reason, ServeStartupStopReason::DownloadStalled);
        assert_eq!(stop.label, line);
        let message = stop.user_message("qwen", "vllm", "svc-1");
        assert!(message.contains(line), "{message}");
        assert!(
            message.contains("stopped making download progress"),
            "{message}"
        );
        assert!(message.contains("The server is still running"), "{message}");
        assert!(
            !message.contains("not ready yet"),
            "a hung download is a failure, not a finished summary: {message}"
        );
        assert!(
            Duration::from_millis(DOWNLOAD_STALL.as_millis() as u64)
                < rocm_core::SERVE_STARTUP_DEADLINE,
            "the stall must fire before the overall deadline"
        );
    }

    #[test]
    fn a_percent_that_does_not_change_is_the_download_fingerprint() {
        assert_eq!(
            download_fingerprint("Downloading model.safetensors: 12%| ETA 0:40"),
            "pct:12"
        );
        assert_eq!(
            download_fingerprint("Downloading model.safetensors: 12%| ETA 0:01"),
            "pct:12",
            "a moving ETA at the same percent is not progress"
        );
        assert_ne!(
            download_fingerprint("Downloading model.safetensors: 13%"),
            download_fingerprint("Downloading model.safetensors: 12%")
        );
    }

    #[test]
    fn loading_past_the_old_budget_keeps_reporting_the_log_line() {
        let line = "Loading safetensors checkpoint shards: 40%";
        let mut signals = alive();
        signals.log_line = Some(line.to_owned());
        let view = assert_continue(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(50_000, 1_000),
            &interactive(),
        ));
        assert_eq!(view.phase, ServeStartupPhase::Loading);
        assert_eq!(view.label, line);
    }

    #[test]
    fn loading_with_no_new_log_stops() {
        let line = "Loading safetensors checkpoint shards: 40%";
        let mut signals = alive();
        signals.log_line = Some(line.to_owned());
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(
                SILENT_LOAD.as_millis() as u64 + 1_000,
                SILENT_LOAD.as_millis() as u64,
            ),
            &interactive(),
        ));
        assert_eq!(stop.reason, ServeStartupStopReason::SilentLoad);
        assert_eq!(stop.label, line);
        let message = stop.user_message("qwen", "vllm", "svc-1");
        assert!(message.contains(line), "{message}");
        assert!(
            message.contains("stopped making progress while loading"),
            "{message}"
        );
        assert!(message.contains("The server is still running"), "{message}");
    }

    #[test]
    fn listing_past_the_old_budget_keeps_waiting() {
        let mut signals = alive();
        signals.log_line = Some("Loading safetensors checkpoint shards: 80%".to_owned());
        // The log has been quiet for longer than the load stall. Listing
        // still waits: the server is answering, and a GPU load can sit on
        // one line. The exit is the overall deadline, not the stall.
        let view = assert_continue(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Listing,
            &signals,
            at(50_000, SILENT_LOAD.as_millis() as u64),
            &interactive(),
        ));
        assert_eq!(view.phase, ServeStartupPhase::Listing);
        assert_eq!(view.label, "Loading safetensors checkpoint shards: 80%");
    }

    #[test]
    fn listing_keeps_a_stale_download_line_off_the_download_stall() {
        // vLLM often still has "download" as the last log line after it has
        // started advertising the model. That line must not arm the stall:
        // the server has answered, and the exit is the overall deadline.
        let line = "Downloading model.safetensors: 100%";
        let mut signals = alive();
        signals.log_line = Some(line.to_owned());
        let view = assert_continue(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Listing,
            &signals,
            at(50_000, DOWNLOAD_STALL.as_millis() as u64),
            &interactive(),
        ));
        assert_eq!(view.phase, ServeStartupPhase::Listing);
        assert_eq!(view.label, line);
    }

    #[test]
    fn listing_without_a_log_says_it_is_waiting_to_serve() {
        let view = assert_continue(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Listing,
            &alive(),
            at(1_000, 1_000),
            &interactive(),
        ));
        assert_eq!(view.phase, ServeStartupPhase::Listing);
        assert_eq!(view.label, waiting_label("qwen"));
    }

    #[test]
    fn listing_at_the_deadline_stops_without_a_not_ready_summary() {
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Listing,
            &alive(),
            at(rocm_core::SERVE_STARTUP_DEADLINE.as_millis() as u64, 0),
            &interactive(),
        ));
        assert_eq!(stop.reason, ServeStartupStopReason::Deadline);
        assert_eq!(stop.readiness, EndpointReadiness::Listing);
        assert_eq!(stop.label, waiting_label("qwen"));
        let message = stop.user_message("qwen", "vllm", "svc-1");
        let minutes = rocm_core::SERVE_STARTUP_DEADLINE.as_secs() / 60;
        assert!(message.contains(&format!("{minutes} minutes")), "{message}");
        assert!(message.contains("The server is still running"), "{message}");
        assert!(message.contains(&service_logs_hint("svc-1")), "{message}");
        assert!(
            !message.contains("not ready yet") && !message.contains("Deployment summary"),
            "{message}"
        );
    }

    #[test]
    fn serving_is_ready() {
        let decision = decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Serving,
            &alive(),
            at(1_000, 0),
            &interactive(),
        );
        assert!(matches!(decision, ServeStartupDecision::Ready));
    }

    #[test]
    fn engine_failed_stops_immediately_and_quotes_the_log() {
        let mut signals = alive();
        signals.engine_failed = true;
        signals.log_line = Some("CUDA out of memory".to_owned());
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(0, 0),
            &interactive(),
        ));
        assert_eq!(stop.reason, ServeStartupStopReason::EngineFailed);
        assert_eq!(stop.label, "CUDA out of memory");
        let message = stop.user_message("qwen", "vllm", "svc-1");
        assert!(message.contains("the engine reported failed"), "{message}");
        assert!(message.contains("CUDA out of memory"), "{message}");
    }

    #[test]
    fn process_exited_stops_without_claiming_the_server_is_running() {
        let mut signals = alive();
        signals.process_alive = false;
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(0, 0),
            &interactive(),
        ));
        assert_eq!(stop.reason, ServeStartupStopReason::ProcessExited);
        assert!(!stop.process_alive);
        let message = stop.user_message("qwen", "vllm", "svc-1");
        assert!(message.contains("the server process exited"), "{message}");
        assert!(message.contains("The server process is gone"), "{message}");
        assert!(
            !message.contains("still running"),
            "a dead process must not be described as still running: {message}"
        );
        assert!(message.contains(&service_logs_hint("svc-1")), "{message}");
    }

    #[test]
    fn piped_serve_returns_at_the_status_budget_instead_of_following_a_download() {
        let mut signals = alive();
        signals.log_line = Some("Downloading model.safetensors: 40%".to_owned());
        let policy = ServeStartupPolicy::status_budget(Duration::from_secs(45));
        let decision = decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &signals,
            at(45_000, 0),
            &policy,
        );
        assert_eq!(
            decision,
            ServeStartupDecision::BudgetElapsed(EndpointReadiness::Unreachable)
        );
    }

    #[test]
    fn sanitize_log_line_keeps_the_latest_progress_segment_and_drops_ansi() {
        let raw = "\u{1b}[32mold line\rDownloading model.safetensors: 12%\u{1b}[0m";
        assert_eq!(sanitize_log_line(raw), "Downloading model.safetensors: 12%");
    }

    #[test]
    fn latest_meaningful_log_line_skips_blank_and_rule_lines() {
        let path = std::env::temp_dir().join(format!("serve-progress-{}.log", unique_suffix()));
        std::fs::write(
            &path,
            "Loading safetensors checkpoint shards: 40%\n\n----------\n",
        )
        .unwrap();
        let line = latest_meaningful_log_line(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            line.as_deref(),
            Some("Loading safetensors checkpoint shards: 40%")
        );
    }

    #[test]
    fn failure_message_names_the_same_logs_command_as_the_summary() {
        let hint = service_logs_hint("svc-1");
        let summary = crate::serve_summary::DeploymentSummary {
            engine: "vllm".to_owned(),
            requested_model: "qwen".to_owned(),
            api_model: "qwen".to_owned(),
            chat_endpoint: "http://127.0.0.1:1/v1/chat/completions".to_owned(),
            service_id: "svc-1".to_owned(),
            status: "starting".to_owned(),
            already_running: false,
            metrics: crate::serve_summary::SmokeMetrics::default(),
            api_key: None,
            notes: Vec::new(),
        };
        let rendered = crate::serve_summary::render_summary(&summary);
        assert!(rendered.contains(&hint), "{rendered}");
        let stop = assert_stop(decide_serve_startup(
            "qwen",
            "vllm",
            EndpointReadiness::Unreachable,
            &alive(),
            at(SILENT_UNREACHABLE.as_millis() as u64, 0),
            &interactive(),
        ));
        assert!(stop.user_message("qwen", "vllm", "svc-1").contains(&hint));
    }

    fn short_policy(deadline: Duration, download_stall: Duration) -> ServeStartupPolicy {
        ServeStartupPolicy {
            bounds: ServeStartupBounds {
                deadline,
                download_stall,
                silent_unreachable: Duration::from_secs(30),
                silent_load: Duration::from_secs(30),
            },
            poll: Duration::from_millis(20),
            stop_on_give_up: true,
            watch_engine: true,
        }
    }

    #[test]
    fn wait_follows_a_download_until_the_server_serves() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let started = Instant::now();
        let serve_after = Duration::from_millis(400);
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 2048];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]);
                let inference_ready = started.elapsed() >= serve_after;
                let (status_line, body) = if request.starts_with("POST /v1/chat/completions ") {
                    if inference_ready {
                        (
                            "HTTP/1.1 200 OK",
                            r#"{"choices":[{"message":{"content":"ok"}}]}"#,
                        )
                    } else {
                        ("HTTP/1.1 503 Service Unavailable", r#"{"error":"loading"}"#)
                    }
                } else {
                    ("HTTP/1.1 200 OK", r#"{"data":[{"id":"qwen"}]}"#)
                };
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });

        let mut labels = Vec::new();
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &short_policy(Duration::from_secs(5), Duration::from_secs(30)),
            &mut || {
                (
                    "qwen".to_owned(),
                    ServeStartupSignals {
                        engine_failed: false,
                        process_alive: true,
                        log_line: Some("Downloading model.safetensors: 40%".to_owned()),
                    },
                )
            },
            &mut |view| labels.push(view.label.clone()),
        );
        drop(server);

        assert_eq!(result.unwrap(), EndpointReadiness::Serving);
        assert!(
            started.elapsed() >= serve_after,
            "the wait returned before the server could serve"
        );
        assert!(
            labels
                .iter()
                .any(|label| label == "Downloading model.safetensors: 40%"),
            "the status line must show the download while the wait continues: {labels:?}"
        );
    }

    #[test]
    fn wait_stops_a_hung_download_before_the_deadline() {
        let line = "Downloading model.safetensors: 12%";
        let stall = Duration::from_millis(200);
        let deadline = Duration::from_secs(5);
        let started = Instant::now();
        let mut labels = Vec::new();
        // Nothing is listening: the probe fails fast on connection refused.
        let port = {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            port
        };
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &short_policy(deadline, stall),
            &mut || {
                (
                    "qwen".to_owned(),
                    ServeStartupSignals {
                        engine_failed: false,
                        process_alive: true,
                        log_line: Some(line.to_owned()),
                    },
                )
            },
            &mut |view| labels.push(view.label.clone()),
        );
        let elapsed = started.elapsed();
        let stop = result.expect_err("a frozen download must fail the wait");
        assert_eq!(stop.reason, ServeStartupStopReason::DownloadStalled);
        assert_eq!(stop.label, line);
        let message = stop.user_message("qwen", "vllm", "svc-1");
        assert!(message.contains(line), "{message}");
        assert!(message.contains("still running"), "{message}");
        assert!(
            labels.iter().any(|label| label == line),
            "the spinner must have shown the stalled line: {labels:?}"
        );
        assert!(
            elapsed < Duration::from_secs(3),
            "stall fired at {elapsed:?}, not before the {deadline:?} deadline"
        );
    }

    #[test]
    fn wait_reports_serving_once_inference_answers_from_the_first_probe() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 2048];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]);
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"qwen"}]}"#
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        let started = Instant::now();
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &short_policy(Duration::from_secs(5), Duration::from_secs(30)),
            &mut || ("qwen".to_owned(), alive()),
            &mut |_| {},
        );
        drop(server);
        assert_eq!(result.unwrap(), EndpointReadiness::Serving);
        assert!(
            started.elapsed() < SILENT_UNREACHABLE,
            "a server that is already serving must not wait out the silent limit"
        );
    }

    fn refused_port() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    }

    #[test]
    fn stop_records_failed_only_for_a_dead_server() {
        let stop = |reason, process_alive, readiness| ServeStartupStop {
            phase: ServeStartupPhase::Loading,
            reason,
            label: String::new(),
            log_line: None,
            process_alive,
            readiness,
        };
        assert_eq!(
            recorded_status_after_stop(&stop(
                ServeStartupStopReason::EngineFailed,
                true,
                EndpointReadiness::Unreachable
            )),
            "failed"
        );
        assert_eq!(
            recorded_status_after_stop(&stop(
                ServeStartupStopReason::ProcessExited,
                false,
                EndpointReadiness::Listing
            )),
            "failed"
        );
        // A live server that stalled is still up: the record must not say
        // failed while the message tells the user it is still running, and
        // must not say `starting`, which rocmd restarts once it is stale.
        assert_eq!(
            recorded_status_after_stop(&stop(
                ServeStartupStopReason::DownloadStalled,
                true,
                EndpointReadiness::Unreachable
            )),
            "running"
        );
        assert_eq!(
            recorded_status_after_stop(&stop(
                ServeStartupStopReason::Silent,
                true,
                EndpointReadiness::Unreachable
            )),
            "running"
        );
        assert_eq!(
            recorded_status_after_stop(&stop(
                ServeStartupStopReason::Deadline,
                true,
                EndpointReadiness::Listing
            )),
            "running"
        );
    }

    #[test]
    fn wait_stops_at_once_when_the_process_has_exited() {
        let started = Instant::now();
        let mut views = 0;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            refused_port(),
            None,
            &short_policy(Duration::from_secs(5), Duration::from_secs(30)),
            &mut || {
                (
                    "qwen".to_owned(),
                    ServeStartupSignals {
                        engine_failed: false,
                        process_alive: false,
                        log_line: Some("Traceback: boom".to_owned()),
                    },
                )
            },
            &mut |_| views += 1,
        );
        let stop = result.expect_err("a dead process cannot become ready");
        assert_eq!(stop.reason, ServeStartupStopReason::ProcessExited);
        assert_eq!(views, 0, "no spinner frame for a server that is gone");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn wait_hands_back_listing_at_the_deadline_with_a_stop() {
        // Row: listing, inference never answers. The interactive wait ends in
        // a deadline stop (not a finished summary) carrying `Listing`; the
        // piped budget returns `Listing` as `Ok`.
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 2048];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]);
                let (status_line, body) = if request.starts_with("POST /v1/chat/completions ") {
                    ("HTTP/1.1 503 Service Unavailable", r#"{"error":"loading"}"#)
                } else {
                    ("HTTP/1.1 200 OK", r#"{"data":[{"id":"qwen"}]}"#)
                };
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });

        let mut labels = Vec::new();
        let interactive_result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &short_policy(Duration::from_millis(1_500), Duration::from_secs(30)),
            &mut || ("qwen".to_owned(), alive()),
            &mut |view| labels.push(view.label.clone()),
        );
        let mut budget = ServeStartupPolicy::status_budget(Duration::from_millis(1_500));
        budget.poll = Duration::from_millis(20);
        let piped_result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &budget,
            &mut || ("qwen".to_owned(), alive()),
            &mut |_| {},
        );
        drop(server);

        let stop = interactive_result.expect_err("the interactive wait must end in a stop");
        assert_eq!(stop.reason, ServeStartupStopReason::Deadline);
        assert_eq!(stop.readiness, EndpointReadiness::Listing);
        assert!(
            labels.iter().any(|label| *label == waiting_label("qwen")),
            "once the model is listed the line says it is waiting to serve: {labels:?}"
        );
        assert_eq!(piped_result.unwrap(), EndpointReadiness::Listing);
    }

    #[test]
    fn wait_keeps_a_changing_download_alive_past_the_stall_limit() {
        // A percent that keeps advancing re-arms the stall clock, so a long
        // healthy download is not mistaken for a hung one.
        let stall = Duration::from_millis(300);
        let mut step = 0_u32;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            refused_port(),
            None,
            &short_policy(Duration::from_millis(1_200), stall),
            &mut || {
                step += 1;
                (
                    "qwen".to_owned(),
                    ServeStartupSignals {
                        engine_failed: false,
                        process_alive: true,
                        // A new percent on every sample re-arms the stall clock.
                        // Wall-clock buckets can sit still long enough to look hung.
                        log_line: Some(format!("Downloading model.safetensors: {step}%")),
                    },
                )
            },
            &mut |_| {},
        );
        let stop = result.expect_err("the deadline still ends the wait");
        assert_eq!(
            stop.reason,
            ServeStartupStopReason::Deadline,
            "a moving download must hit the deadline, not the stall limit"
        );
        assert!(step > 3);
    }
}
