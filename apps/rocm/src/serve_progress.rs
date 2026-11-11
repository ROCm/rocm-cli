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
//! | downloading | latest log line is a download, the endpoint is not listing the model, and progress is still moving | that log line | stall limit once the line (or its percent) freezes, or the deadline |
//! | loading | latest log line is anything else meaningful, port still down | that log line | silent-load limit once the line freezes, or the deadline |
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

use crate::engine_runtime_select::{service_http_inference_answers, service_http_lists_model};
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
    /// How far apart inference probes are once the endpoint lists the model.
    pub inference_backoff: InferenceProbeBackoff,
}

/// Spacing between inference probes while the endpoint lists the model but
/// cannot answer yet.
///
/// Listing is a short GET; the inference probe is a real generation request
/// that can hold the engine for up to [`rocm_core::INFERENCE_PROBE_TIMEOUT`].
/// The first probe runs as soon as the model is listed, then the gap starts at
/// `first` and doubles up to `max`. A zero `first` probes every poll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InferenceProbeBackoff {
    pub first: Duration,
    pub max: Duration,
}

impl InferenceProbeBackoff {
    /// Probe on every poll: the piped status budget, bounded at 45 seconds.
    pub(crate) const EVERY_POLL: Self = Self {
        first: Duration::ZERO,
        max: Duration::ZERO,
    };
}

/// First gap between inference probes on the interactive wait.
pub(crate) const INFERENCE_PROBE_FIRST_GAP: Duration = Duration::from_secs(1);

/// When the next inference probe may run. See [`InferenceProbeBackoff`].
#[derive(Debug)]
pub(crate) struct InferenceProbeThrottle {
    backoff: InferenceProbeBackoff,
    gap: Duration,
    next_at: Option<Instant>,
}

impl InferenceProbeThrottle {
    pub(crate) const fn new(backoff: InferenceProbeBackoff) -> Self {
        Self {
            backoff,
            gap: backoff.first,
            next_at: None,
        }
    }

    /// Readiness for one poll. `probe` runs only when the model is listed and
    /// the gap since the last failed probe has passed; the next gap is counted
    /// from when that probe returned, read from `clock`.
    pub(crate) fn readiness(
        &mut self,
        listed: bool,
        clock: &mut dyn FnMut() -> Instant,
        probe: impl FnOnce() -> bool,
    ) -> EndpointReadiness {
        if !listed {
            return EndpointReadiness::Unreachable;
        }
        if self.next_at.is_some_and(|at| clock() < at) {
            return EndpointReadiness::Listing;
        }
        if probe() {
            return EndpointReadiness::Serving;
        }
        let returned = clock();
        self.next_at = Some(returned.checked_add(self.gap).unwrap_or(returned));
        self.gap = self.gap.saturating_mul(2).min(self.backoff.max);
        EndpointReadiness::Listing
    }
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
            inference_backoff: InferenceProbeBackoff {
                first: INFERENCE_PROBE_FIRST_GAP,
                max: rocm_core::INFERENCE_PROBE_RETRY_INTERVAL,
            },
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
            inference_backoff: InferenceProbeBackoff::EVERY_POLL,
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

/// The error `start_managed_service` returns when the wait stopped.
///
/// `Display` is [`ServeStartupStop::user_message`], plus memory guidance when
/// the server did not start. Only a stop that left the server running
/// (`server_running`) has a process for the caller to supervise; a spawn
/// failure or a server that ended has none.
#[derive(Debug)]
pub(crate) struct ServeStartupFailed {
    pub message: String,
    pub server_running: bool,
}

impl std::fmt::Display for ServeStartupFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ServeStartupFailed {}

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
/// are skipped so a rule of dashes cannot cover a real stage. Credentials
/// are masked ([`redact_log_line`]) before the line is truncated, so a key
/// cut at the length limit cannot show its prefix.
pub(crate) fn latest_meaningful_log_line(
    path: &Path,
    endpoint_api_key: Option<&str>,
) -> Option<String> {
    let lines = rocm_core::tail_lines(path, LOG_TAIL_LINES, LOG_TAIL_BYTES).ok()?;
    lines.into_iter().rev().find_map(|line| {
        let sanitized = sanitize_log_line(&line, endpoint_api_key);
        is_meaningful(&sanitized).then_some(sanitized)
    })
}

pub(crate) fn sanitize_log_line(raw: &str, endpoint_api_key: Option<&str>) -> String {
    let segment = raw
        .rsplit(['\r', '\n'])
        .find(|part| !part.trim().is_empty())
        .unwrap_or(raw);
    let stripped = rocm_core::terminal::strip_terminal_control_sequences(segment);
    let ascii: String = stripped.chars().filter(char::is_ascii).collect();
    let collapsed = ascii.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_label(&redact_log_line(&collapsed, endpoint_api_key))
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
    let signals = effective_signals(signals.clone(), policy);
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
        && timing.since_progress >= policy.bounds.silent_unreachable
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

impl ServeStartupStop {
    /// The wait gave up on a server that is still up: a stall or the deadline.
    ///
    /// Engine failure and a dead process end the launch instead. Those are
    /// settled like a `--wait` that saw the server exit: a dead process is
    /// recorded `stopped`; a record the engine already marked `failed` keeps
    /// that status, and `rocmd`'s recovery treats it like any `failed` service
    /// (its message names the stop command while its process is still up); a
    /// stop another terminal recorded is left alone. The CLI does not start the
    /// background helper for either. A server left running is recorded
    /// `running`, which `rocmd` leaves alone until
    /// [`rocm_core::SERVE_STARTUP_DEADLINE`] (see
    /// `recovery_leaves_a_live_startup_alone_and_still_restarts_a_dead_one`),
    /// so it does not kill the server this message says is still running.
    /// [`decide_serve_startup`] returns `ProcessExited` before it checks any
    /// stall or the deadline, so those stops always have `process_alive`.
    pub(crate) const fn leaves_server_running(&self) -> bool {
        self.process_alive
            && !matches!(
                self.reason,
                ServeStartupStopReason::EngineFailed | ServeStartupStopReason::ProcessExited
            )
    }

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
        if self.leaves_server_running() {
            let stop = service_stop_hint(service_id);
            let _ = write!(
                message,
                "The server is still running (service {service_id}). Follow it with `{logs}`, or stop it with `{stop}`."
            );
        } else if self.process_alive {
            // The engine said failed but its process has not exited yet.
            let stop = service_stop_hint(service_id);
            let _ = write!(
                message,
                "Its process is still up (service {service_id}). The log is `{logs}`; `{stop}` stops it."
            );
        } else {
            let _ = write!(
                message,
                "The server did not start (service {service_id}). The log is `{logs}`."
            );
        }
        message
    }
}

/// Poll until the endpoint can serve, reporting each stage through `on_view`.
///
/// `observe` returns the model name the probe should ask for (a container
/// engine may not know it until it starts) and the latest engine/process
/// sample. The same HTTP checks as the historical readiness wait are used, so
/// "serving" still means an inference request came back. Once the model is
/// listed, the inference request is spaced by the policy's
/// [`InferenceProbeBackoff`] rather than sent every poll.
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
    let mut throttle = InferenceProbeThrottle::new(policy.inference_backoff);

    // One observe, one probe, one decision and at most one status-line update
    // per poll, so the spinner advances once per tick.
    loop {
        let (probe_model, signals) = observe();
        let mut signals = effective_signals(signals, policy);
        signals.log_line = signals
            .log_line
            .map(|line| redact_log_line(&line, endpoint_api_key));
        // A failed engine or a dead process is decided without waiting on a
        // probe that cannot succeed.
        if !signals.engine_failed && signals.process_alive {
            let listed =
                service_http_lists_model(engine_name, host, port, &probe_model, endpoint_api_key);
            match throttle.readiness(listed, &mut Instant::now, || {
                service_http_inference_answers(host, port, &probe_model, endpoint_api_key)
            }) {
                EndpointReadiness::Serving => best = EndpointReadiness::Serving,
                EndpointReadiness::Listing => best = EndpointReadiness::Listing,
                EndpointReadiness::Unreachable => {}
            }
        }
        let next_token = progress_token(best, &signals);
        if next_token != token {
            token = next_token;
            progress_at = Instant::now();
        }
        let timing = ServeStartupTiming {
            elapsed: started.elapsed(),
            since_progress: progress_at.elapsed(),
        };
        match decide_serve_startup(model, engine_name, best, &signals, timing, policy) {
            ServeStartupDecision::Continue(view) => on_view(&view),
            ServeStartupDecision::Ready => return Ok(EndpointReadiness::Serving),
            ServeStartupDecision::BudgetElapsed(readiness) => return Ok(readiness),
            ServeStartupDecision::Stop(stop) => return Err(stop),
        }

        let remaining = policy.bounds.deadline.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            continue;
        }
        thread::sleep(policy.poll.min(remaining));
    }
}

/// The signals the policy looks at.
///
/// The piped path ignores engine failure and process liveness so a serve that
/// exits in CI still returns a readiness status.
const fn effective_signals(
    mut signals: ServeStartupSignals,
    policy: &ServeStartupPolicy,
) -> ServeStartupSignals {
    if !policy.watch_engine {
        signals.engine_failed = false;
        signals.process_alive = true;
    }
    signals
}

const REDACTED: &str = "[redacted]";

/// Mask credentials in an engine log line before it reaches the status line
/// or the failure text.
///
/// Engines echo their launch arguments (vLLM prints its non-default args),
/// and those can carry the endpoint key or a hub token. The line is already
/// one ASCII line with single spaces. Masked: the endpoint key wherever it
/// appears, the value after a secret-named flag or key (`--api-key X`,
/// `api_key=X`, `'api_key': 'X'`, `"api_key":"X"`, `Authorization: Bearer X`),
/// and Hugging Face `hf_` tokens. A list value (`'api_key': ['a', 'b']`) has
/// only its first item masked.
pub(crate) fn redact_log_line(line: &str, endpoint_api_key: Option<&str>) -> String {
    let line = match endpoint_api_key.filter(|key| !key.is_empty()) {
        Some(key) => line.replace(key, REDACTED),
        None => line.to_owned(),
    };
    let mut out: Vec<String> = Vec::new();
    let mut mask_next = false;
    for word in line.split(' ') {
        if mask_next {
            mask_next = false;
            if word.eq_ignore_ascii_case("bearer") || word.eq_ignore_ascii_case("basic") {
                out.push(word.to_owned());
                mask_next = true;
                continue;
            }
            out.push(mask_value(word));
            continue;
        }
        if let Some((name, value)) = word.split_once('=')
            && is_secret_name(name)
        {
            out.push(if value.is_empty() {
                format!("{name}=")
            } else {
                format!("{name}={}", mask_value(value))
            });
            continue;
        }
        // Compact JSON or YAML: `"api_key":"X"`.
        if let Some((name, value)) = word.split_once(':')
            && !value.is_empty()
            && is_secret_name(name)
        {
            out.push(format!("{name}:{}", mask_compact_value(value)));
            continue;
        }
        if names_a_following_secret(word) {
            mask_next = true;
            out.push(word.to_owned());
            continue;
        }
        if looks_like_hf_token(word) {
            out.push(REDACTED.to_owned());
            continue;
        }
        out.push(word.to_owned());
    }
    out.join(" ")
}

/// Whether `word` names a secret (`--api-key`, `HF_TOKEN`, `'api_key':`,
/// `Authorization:`, `Bearer`).
fn is_secret_name(word: &str) -> bool {
    let name = word
        .trim_start_matches('-')
        .trim_matches(|c: char| matches!(c, '\'' | '"' | ':' | ',' | '{' | '('))
        .to_ascii_lowercase()
        .replace('-', "_");
    if name.is_empty() {
        return false;
    }
    name == "authorization"
        || name == "bearer"
        || ["api_key", "apikey", "token", "password", "secret", "private_key"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        // `secret_key`, `token_value`: a secret word anywhere in a `_` name.
        || name
            .split('_')
            .any(|part| matches!(part, "secret" | "token" | "password" | "passwd"))
}

/// A flag (`--api-key`), a key ending in `:` (`'api_key':`,
/// `Authorization:`), or `Bearer`, whose next word is the secret. A bare word
/// such as "token" in prose does not mask what follows it.
fn names_a_following_secret(word: &str) -> bool {
    if word.eq_ignore_ascii_case("bearer") {
        return true;
    }
    let unquoted = word.trim_end_matches(['\'', '"', ',']);
    (word.starts_with('-') || unquoted.ends_with(':')) && is_secret_name(word)
}

/// Keep the quoting and trailing punctuation around a masked value, so a
/// Python-dict log line stays readable.
fn mask_value(word: &str) -> String {
    if word.contains(REDACTED) {
        return word.to_owned();
    }
    let body_start = word
        .find(|c: char| !matches!(c, '\'' | '"' | '(' | '['))
        .unwrap_or(word.len());
    // The end of the last body character, which may be wider than one byte.
    let body_end = word
        .char_indices()
        .rev()
        .find(|(_, c)| !matches!(c, '\'' | '"' | ',' | ')' | ']' | '}'))
        .map_or(body_start, |(index, c)| index + c.len_utf8());
    if body_end <= body_start {
        return word.to_owned();
    }
    format!("{}{REDACTED}{}", &word[..body_start], &word[body_end..])
}

/// Mask one compact value (`"X","port":8000}` or `X,port:8000}`) and keep
/// what follows it, so the rest of a one-word JSON object stays readable.
fn mask_compact_value(value: &str) -> String {
    if let Some(quote) = value.chars().next().filter(|c| matches!(c, '"' | '\'')) {
        let rest = &value[1..];
        return match rest.find(quote) {
            Some(end) if end > 0 => format!("{quote}{REDACTED}{}", &rest[end..]),
            _ => value.to_owned(),
        };
    }
    let end = value.find([',', '}', ']']).unwrap_or(value.len());
    if end == 0 {
        return value.to_owned();
    }
    format!("{REDACTED}{}", &value[end..])
}

fn looks_like_hf_token(word: &str) -> bool {
    let body = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
    body.strip_prefix("hf_")
        .is_some_and(|rest| rest.len() >= 20 && rest.chars().all(|c| c.is_ascii_alphanumeric()))
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
        assert!(message.contains("The server did not start"), "{message}");
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
        assert_eq!(
            sanitize_log_line(raw, None),
            "Downloading model.safetensors: 12%"
        );
    }

    #[test]
    fn latest_meaningful_log_line_skips_blank_and_rule_lines() {
        let path = std::env::temp_dir().join(format!("serve-progress-{}.log", unique_suffix()));
        std::fs::write(
            &path,
            "Loading safetensors checkpoint shards: 40%\n\n----------\n",
        )
        .unwrap();
        let line = latest_meaningful_log_line(&path, None);
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
            anthropic_endpoint: None,
            service_id: "svc-1".to_owned(),
            status: "starting".to_owned(),
            already_running: false,
            metrics: crate::serve_summary::SmokeMetrics::default(),
            api_key: None,
            notes: Vec::new(),
        };
        let rendered = crate::serve_summary::render_summary(&summary);
        // The message names only the hint: the rendered summary can carry an
        // endpoint key.
        assert!(
            rendered.contains(&hint),
            "the deployment summary must name `{hint}`"
        );
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
            inference_backoff: InferenceProbeBackoff::EVERY_POLL,
        }
    }

    /// The interactive wait's inference probe, under a clock that moves only
    /// when the test says: a model listed but never answering is probed at 0,
    /// 1, 3, 7 and 15 seconds, then every 15, never once per 250 ms poll. Each
    /// probe takes the full 8-second timeout, and the next gap counts from
    /// when it returned.
    #[test]
    fn inference_probe_backs_off_to_the_retry_interval_after_listing() {
        use std::cell::Cell;
        let start = Instant::now();
        let now = Cell::new(start);
        let mut clock = || now.get();
        let mut throttle =
            InferenceProbeThrottle::new(ServeStartupPolicy::interactive().inference_backoff);
        let mut probed_at = Vec::new();
        // Ten minutes of 250 ms polls.
        while now.get() - start < Duration::from_mins(10) {
            let readiness = throttle.readiness(true, &mut clock, || {
                probed_at.push((now.get() - start).as_secs());
                now.set(now.get() + rocm_core::INFERENCE_PROBE_TIMEOUT);
                false
            });
            assert_eq!(readiness, EndpointReadiness::Listing);
            now.set(now.get() + POLL);
        }
        let gaps: Vec<u64> = probed_at
            .windows(2)
            .map(|pair| pair[1] - pair[0] - rocm_core::INFERENCE_PROBE_TIMEOUT.as_secs())
            .collect();
        assert_eq!(&gaps[..5], &[1, 2, 4, 8, 15], "probed at {probed_at:?}");
        assert!(
            gaps[4..].iter().all(|gap| *gap == 15),
            "the gap must stay at the retry interval: {gaps:?}"
        );
        // 2400 polls; one probe each would be 2400 eight-second requests.
        assert!(probed_at.len() < 30, "probed {} times", probed_at.len());
    }

    /// Not listed, the probe never runs; listed, the first probe is not
    /// delayed, and a probe that answers is ready at once.
    #[test]
    fn inference_probe_runs_at_once_when_listed_and_not_before() {
        let now = Instant::now();
        let mut clock = || now;
        let mut throttle =
            InferenceProbeThrottle::new(ServeStartupPolicy::interactive().inference_backoff);
        assert_eq!(
            throttle.readiness(false, &mut clock, || panic!("not listed, no probe")),
            EndpointReadiness::Unreachable
        );
        assert_eq!(
            throttle.readiness(true, &mut clock, || true),
            EndpointReadiness::Serving
        );
    }

    /// The same backoff through the real wait: an endpoint that lists the model
    /// and refuses every chat request gets a handful of inference requests over
    /// a second of 10 ms polls, not one per poll.
    #[test]
    fn wait_spaces_inference_probes_while_the_model_is_listed() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let chat_requests = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&chat_requests);
        let _server = thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 2048];
                let Ok(read) = stream.read(&mut buffer) else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buffer[..read]);
                let (status_line, body) = if request.starts_with("POST /v1/chat/completions ") {
                    counted.fetch_add(1, Ordering::SeqCst);
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
        let mut policy = short_policy(Duration::from_secs(1), Duration::from_secs(30));
        policy.poll = Duration::from_millis(10);
        policy.inference_backoff = InferenceProbeBackoff {
            first: Duration::from_millis(100),
            max: Duration::from_millis(200),
        };
        let mut polls = 0_usize;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &policy,
            &mut || {
                polls += 1;
                ("qwen".to_owned(), alive())
            },
            &mut |_| {},
        );
        let stop = result.expect_err("a model that never answers hits the deadline");
        assert_eq!(stop.reason, ServeStartupStopReason::Deadline);
        assert_eq!(stop.readiness, EndpointReadiness::Listing);
        let probes = chat_requests.load(Ordering::SeqCst);
        // 0, 100, 300, 500, 700, 900 ms: at most a few more under a slow host,
        // where an unthrottled wait sends one per poll.
        assert!(
            (1..=8).contains(&probes),
            "{probes} inference requests over {polls} polls"
        );
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
                } else if inference_ready {
                    ("HTTP/1.1 200 OK", r#"{"data":[{"id":"qwen"}]}"#)
                } else {
                    // Still downloading: the endpoint does not list the model.
                    ("HTTP/1.1 503 Service Unavailable", r#"{"error":"loading"}"#)
                };
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });

        let mut views = Vec::new();
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
            &mut |view| views.push(view.clone()),
        );
        drop(server);

        assert_eq!(result.unwrap(), EndpointReadiness::Serving);
        assert!(
            started.elapsed() >= serve_after,
            "the wait returned before the server could serve"
        );
        assert!(
            views
                .iter()
                .any(|view| view.phase == ServeStartupPhase::Downloading
                    && view.label == "Downloading model.safetensors: 40%"),
            "the status line must show the download while the wait continues: {views:?}"
        );
    }

    #[test]
    fn wait_stops_a_hung_download_before_the_deadline() {
        let line = "Downloading model.safetensors: 12%";
        let stall = Duration::from_millis(200);
        let deadline = Duration::from_secs(5);
        let started = Instant::now();
        let mut labels = Vec::new();
        // The endpoint never lists the model, and answers at once.
        let port = not_listing_port();
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

    /// A local endpoint that answers every request with 503 at once: the
    /// model is never listed. Not a closed port: on Windows a refused loopback
    /// connect takes about two seconds, which outlasts these tests' limits.
    fn not_listing_port() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 2048];
                let _ = stream.read(&mut buffer);
                let body = r#"{"error":"loading"}"#;
                let _ = write!(
                    stream,
                    "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        port
    }

    /// Piped output returns a readiness status at the budget even when the
    /// engine reports failure or the process is gone: that is the historical
    /// 45-second contract scripts and CI read, and it has no stop.
    #[test]
    fn piped_wait_ignores_engine_failure_and_a_dead_process() {
        let port = not_listing_port();
        let mut policy = ServeStartupPolicy::status_budget(Duration::from_millis(200));
        policy.poll = Duration::from_millis(20);
        let mut polls = 0;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            port,
            None,
            &policy,
            &mut || {
                polls += 1;
                (
                    "qwen".to_owned(),
                    ServeStartupSignals {
                        engine_failed: true,
                        process_alive: false,
                        log_line: None,
                    },
                )
            },
            &mut |_| {},
        );
        assert_eq!(result, Ok(EndpointReadiness::Unreachable));
        assert!(
            polls > 1,
            "the piped wait must run to its budget, not stop at once"
        );
    }

    /// Both silent limits count the silence, from the last change of what
    /// the wait observes, not the whole wait: a long startup that goes quiet
    /// is judged on how long it has been quiet. Each row moves one clock past
    /// the limit and holds the other below it, so a check reading `elapsed`
    /// fails a row.
    #[test]
    fn silent_limits_count_from_the_last_change() {
        let decide = |log_line: Option<&str>, elapsed: Duration, since: Duration| {
            let mut signals = alive();
            signals.log_line = log_line.map(str::to_owned);
            decide_serve_startup(
                "qwen",
                "vllm",
                EndpointReadiness::Unreachable,
                &signals,
                ServeStartupTiming {
                    elapsed,
                    since_progress: since,
                },
                &interactive(),
            )
        };
        let short = Duration::from_secs(1);
        // Unreachable, no log: only now silent, after a long startup.
        assert!(matches!(
            decide(None, SILENT_UNREACHABLE * 3, short),
            ServeStartupDecision::Continue(_)
        ));
        let stop = assert_stop(decide(None, SILENT_UNREACHABLE, SILENT_UNREACHABLE));
        assert_eq!(stop.reason, ServeStartupStopReason::Silent);
        // Loading: the time since the line last changed decides, however long
        // the wait has run.
        let load = Some("Loading safetensors checkpoint shards: 40%");
        let stop = assert_stop(decide(load, SILENT_LOAD, SILENT_LOAD));
        assert_eq!(stop.reason, ServeStartupStopReason::SilentLoad);
        assert!(matches!(
            decide(load, SILENT_LOAD * 2, short),
            ServeStartupDecision::Continue(_)
        ));
    }

    #[test]
    fn only_a_stall_or_the_deadline_leaves_the_server_running() {
        let stop = |reason, process_alive| ServeStartupStop {
            phase: ServeStartupPhase::Loading,
            reason,
            label: String::new(),
            log_line: None,
            process_alive,
            readiness: EndpointReadiness::Unreachable,
        };
        // The engine said failed, or the process is gone: the launch is over,
        // so the message does not say the server is still running. (That the
        // CLI does not supervise it is
        // `a_startup_stop_starts_supervision_and_keeps_its_own_error`.)
        for (reason, alive) in [
            (ServeStartupStopReason::EngineFailed, true),
            (ServeStartupStopReason::EngineFailed, false),
            (ServeStartupStopReason::ProcessExited, false),
        ] {
            let stop = stop(reason, alive);
            assert!(!stop.leaves_server_running(), "{reason:?}");
            let message = stop.user_message("qwen", "vllm", "svc-1");
            assert!(!message.contains("still running"), "{reason:?}: {message}");
            assert!(message.contains(&service_logs_hint("svc-1")), "{message}");
            // A failed engine whose process is still up gets the command that
            // stops it; one that is gone does not.
            assert_eq!(
                message.contains(&service_stop_hint("svc-1")),
                alive,
                "{reason:?} alive={alive}: {message}"
            );
            assert_eq!(
                message.contains("Its process is still up"),
                alive,
                "{reason:?} alive={alive}: {message}"
            );
            assert_eq!(
                message.contains("The server did not start"),
                !alive,
                "{reason:?} alive={alive}: {message}"
            );
        }
        for reason in [
            ServeStartupStopReason::DownloadStalled,
            ServeStartupStopReason::Silent,
            ServeStartupStopReason::SilentLoad,
            ServeStartupStopReason::Deadline,
        ] {
            assert!(stop(reason, true).leaves_server_running(), "{reason:?}");
        }
    }

    #[test]
    fn wait_stops_at_once_when_the_process_has_exited() {
        let started = Instant::now();
        let mut views = 0;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            not_listing_port(),
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
            not_listing_port(),
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

    #[test]
    fn wait_observes_and_updates_the_status_line_once_per_poll() {
        // Row: unreachable, refused port. One observe and one status-line
        // update per poll, so the spinner does not tick twice per tick.
        let poll = Duration::from_millis(20);
        let deadline = Duration::from_millis(400);
        let mut policy = short_policy(deadline, Duration::from_secs(30));
        policy.poll = poll;
        let mut observes = 0_u32;
        let mut views = 0_u32;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            not_listing_port(),
            None,
            &policy,
            &mut || {
                observes += 1;
                ("qwen".to_owned(), alive())
            },
            &mut |_| views += 1,
        );
        let stop = result.expect_err("nothing answers, so the deadline ends the wait");
        assert_eq!(stop.reason, ServeStartupStopReason::Deadline);
        // The last observe ends in the stop, which draws no frame.
        assert_eq!(views + 1, observes, "one frame per observed poll");
        let polls = u32::try_from(deadline.as_millis() / poll.as_millis()).unwrap();
        assert!(
            observes <= polls + 2,
            "{observes} observes in {polls} polls: the engine is sampled more than once per poll"
        );
    }

    #[test]
    fn redact_log_line_masks_keys_tokens_and_secret_flags() {
        let key = "sk-endpoint-0123456789";
        for (raw, expected) in [
            (
                "launch --secret-key abc123 --private-key def456 --port 8000",
                "launch --secret-key [redacted] --private-key [redacted] --port 8000",
            ),
            // A value ending in a multi-byte character is cut at a character
            // boundary, not one byte past the start of that character.
            (
                "launch --api-key clé --port 8000",
                "launch --api-key [redacted] --port 8000",
            ),
            (
                "launch --api-key 'clé', --port 8000",
                "launch --api-key '[redacted]', --port 8000",
            ),
            (
                "non-default args: {'model': 'qwen', 'api_key': 'sk-endpoint-0123456789', 'port': 8000}",
                "non-default args: {'model': 'qwen', 'api_key': '[redacted]', 'port': 8000}",
            ),
            (
                "launch --api-key abc123 --port 8000",
                "launch --api-key [redacted] --port 8000",
            ),
            (
                "env HF_TOKEN=hf_abcdefghijklmnopqrstuvwx set",
                "env HF_TOKEN=[redacted] set",
            ),
            (
                "using hf_abcdefghijklmnopqrstuvwx for the hub",
                "using [redacted] for the hub",
            ),
            (
                "header Authorization: Bearer abc.def.ghi sent",
                "header Authorization: Bearer [redacted] sent",
            ),
            (
                r#"config {"api_key":"abc123","port":8000}"#,
                r#"config {"api_key":"[redacted]","port":8000}"#,
            ),
            (
                "header Authorization: Basic dXNlcjpwYXNz sent",
                "header Authorization: Basic [redacted] sent",
            ),
            (
                "serving with key sk-endpoint-0123456789 on 0.0.0.0",
                "serving with key [redacted] on 0.0.0.0",
            ),
        ] {
            assert_eq!(redact_log_line(raw, Some(key)), expected, "{raw}");
        }
        // Ordinary progress lines, including prose about tokens, are untouched.
        for line in [
            "Loading safetensors checkpoint shards: 40%",
            "Downloading tokenizer.json: 100%",
            "the token budget is 8192 tokens",
            "max_num_batched_tokens=8192",
        ] {
            assert_eq!(redact_log_line(line, Some(key)), line);
        }
    }

    #[test]
    fn a_secret_in_the_engine_log_never_reaches_the_status_line_or_the_failure() {
        let key = "sk-endpoint-0123456789";
        let mut labels = Vec::new();
        let mut samples = 0_u32;
        let result = wait_for_serve_startup(
            "vllm",
            "qwen",
            "127.0.0.1",
            not_listing_port(),
            Some(key),
            &short_policy(Duration::from_secs(5), Duration::from_secs(30)),
            &mut || {
                // A live sample first, so the status line draws the line,
                // then the failure that ends the wait.
                samples += 1;
                (
                    "qwen".to_owned(),
                    ServeStartupSignals {
                        engine_failed: samples > 1,
                        process_alive: true,
                        log_line: Some(format!(
                            "error: bad args --api-key {key} --hf-token hf_abcdefghijklmnopqrstuvwx"
                        )),
                    },
                )
            },
            &mut |view| labels.push(view.label.clone()),
        );
        let stop = result.expect_err("a failed engine ends the wait");
        let message = stop.user_message("qwen", "vllm", "svc-1");
        for text in [&stop.label, &message] {
            assert!(
                !text.contains(key),
                "endpoint key leaked: {}",
                text.replace(key, "<KEY>")
            );
            assert!(
                !text.contains("hf_abcdefghijklmnopqrstuvwx"),
                "hub token leaked"
            );
        }
        assert!(
            message.contains("--api-key [redacted]"),
            "the line is still quoted, masked"
        );
        assert!(!labels.is_empty(), "the live sample drew a status line");
        for label in &labels {
            assert!(!label.contains(key), "endpoint key on the status line");
            assert!(label.contains("--api-key [redacted]"), "{label}");
        }
    }

    #[test]
    fn a_key_cut_at_the_length_limit_is_masked_before_truncation() {
        let key = "sk-endpoint-0123456789abcdef";
        // Put the key across the MAX_LABEL_CHARS boundary.
        let pad = "x".repeat(MAX_LABEL_CHARS - 25);
        let raw = format!("args {pad} {key} trailing words");
        let line = sanitize_log_line(&raw, Some(key));
        assert!(
            !line.contains("sk-endpoint"),
            "key prefix survived truncation: {line}"
        );
        assert!(line.chars().count() <= MAX_LABEL_CHARS);
    }
}
