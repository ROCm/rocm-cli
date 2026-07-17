// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bench-run form overlay.
//!
//! A small form that collects endpoint, model (optional), concurrency or
//! auto-ramp toggle, and output path, then emits a
//! [`SideEffect::SpawnJob`] for `rocm bench load`. Mirrors the
//! command-screen pattern: pure reducer functions, no I/O.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::exec::resolve_exe;
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// Which field of the form has focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Field {
    #[default]
    Endpoint,
    Model,
    Concurrency,
    Out,
}

/// Form overlay state.
#[derive(Debug, Clone)]
pub struct BenchRunState {
    pub endpoint: String,
    pub model: String,
    pub concurrency: String,
    pub out: String,
    /// When `true`, `--auto-ramp` is used instead of `--concurrency`.
    pub auto_ramp: bool,
    active_field: Field,
    pub message: Option<String>,
    /// Hint shown in the Out field when it is empty: the daemon-tailed path or
    /// a note that live updates need it configured.
    default_out_hint: String,
    /// When `Some`, this is the default --out path (daemon-tailed CSV).
    tailed_path: Option<String>,
}

impl BenchRunState {
    /// Create a new form.
    ///
    /// `tailed_csv` is the daemon's `bench_results_dir` (from config); when
    /// `Some`, it is used as the default `--out` so rows appear live in the
    /// bench tab.
    pub fn new(tailed_csv: Option<&std::path::Path>) -> Self {
        let (tailed_path, default_out_hint) = if let Some(p) = tailed_csv {
            let s = p.display().to_string();
            (
                Some(s.clone()),
                format!("{s} (daemon-tailed — live updates)"),
            )
        } else {
            (
                None,
                "<data_dir>/bench/results.csv (shared CLI and daemon default)".to_string(),
            )
        };
        Self {
            endpoint: String::new(),
            model: String::new(),
            concurrency: "1,8,32,64".to_string(),
            out: String::new(),
            auto_ramp: false,
            active_field: Field::Endpoint,
            message: None,
            default_out_hint,
            tailed_path,
        }
    }
}

/// Handle a key while the bench-run form is open.
///
/// Returns side effects to pump (a `SpawnJob` on submit, empty otherwise).
pub fn on_key(br: &mut Option<BenchRunState>, jobs: &mut State, key: KeyEvent) -> Vec<SideEffect> {
    let Some(state) = br.as_mut() else {
        return Vec::new();
    };

    match key.code {
        KeyCode::Esc => {
            *br = None;
            return Vec::new();
        }
        KeyCode::Tab => {
            state.active_field = match state.active_field {
                Field::Endpoint => Field::Model,
                Field::Model => Field::Concurrency,
                Field::Concurrency => Field::Out,
                Field::Out => Field::Endpoint,
            };
            state.message = None;
            return Vec::new();
        }
        KeyCode::BackTab => {
            state.active_field = match state.active_field {
                Field::Endpoint => Field::Out,
                Field::Model => Field::Endpoint,
                Field::Concurrency => Field::Model,
                Field::Out => Field::Concurrency,
            };
            state.message = None;
            return Vec::new();
        }
        KeyCode::Char('t')
            if key
                .modifiers
                .contains(crossterm::event::KeyModifiers::CONTROL) =>
        {
            // Ctrl+T toggles auto-ramp (works from any field).
            state.auto_ramp = !state.auto_ramp;
            state.message = None;
            return Vec::new();
        }
        KeyCode::Enter => {
            return try_submit(br, jobs);
        }
        KeyCode::Backspace => {
            match state.active_field {
                Field::Endpoint => state.endpoint.pop(),
                Field::Model => state.model.pop(),
                Field::Concurrency => state.concurrency.pop(),
                Field::Out => state.out.pop(),
            };
            state.message = None;
            return Vec::new();
        }
        KeyCode::Char(c) => {
            match state.active_field {
                Field::Endpoint => state.endpoint.push(c),
                Field::Model => state.model.push(c),
                Field::Concurrency => state.concurrency.push(c),
                Field::Out => state.out.push(c),
            }
            state.message = None;
            return Vec::new();
        }
        _ => {}
    }
    Vec::new()
}

/// Validate and submit the form as a `SpawnJob`.
fn try_submit(br: &mut Option<BenchRunState>, jobs: &mut State) -> Vec<SideEffect> {
    let state = br.as_mut().expect("called only when Some");

    let endpoint = state.endpoint.trim().to_string();
    if endpoint.is_empty() {
        state.message = Some("endpoint is required".to_string());
        return Vec::new();
    }
    // Reject https:// — no TLS backend.
    if endpoint.to_lowercase().starts_with("https://") {
        state.message =
            Some("https:// not supported (no TLS backend compiled in); use http://".to_string());
        return Vec::new();
    }

    let cmd = resolve_exe();
    let mut args = vec!["bench".to_string(), "load".to_string()];
    args.push("--endpoint".to_string());
    args.push(endpoint);

    let model = state.model.trim().to_string();
    if !model.is_empty() {
        args.push("--model".to_string());
        args.push(model);
    }

    if state.auto_ramp {
        args.push("--auto-ramp".to_string());
    } else {
        let conc = state.concurrency.trim().to_string();
        if !conc.is_empty() {
            args.push("--concurrency".to_string());
            args.push(conc);
        }
    }

    // Resolve the output path: use the explicit Out field, else the tailed path,
    // else let the CLI derive its shared `<data_dir>/bench/results.csv` default.
    let out = state.out.trim().to_string();
    if !out.is_empty() {
        args.push("--out".to_string());
        args.push(out);
    } else if let Some(tailed) = state.tailed_path.clone() {
        args.push("--out".to_string());
        args.push(tailed);
    }

    let job_id = "bench-run".to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: job_id,
        cmd,
        args,
    });
    if fx.is_empty() {
        state.message = Some("a bench run is already running".to_string());
        return fx;
    }
    // Close the form only after spawning (the job console is shown by the job-bridge).
    *br = None;
    fx
}

/// Render the bench-run form.
pub fn draw_bench_run(f: &mut Frame, area: Rect, state: &BenchRunState, theme: &Theme) {
    let inner = panel::bento(
        f,
        area,
        Some("Run a bench sweep"),
        BoxRole::Primary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title / hint
            Constraint::Length(1), // endpoint
            Constraint::Length(1), // model
            Constraint::Length(1), // concurrency / auto-ramp toggle
            Constraint::Length(1), // out
            Constraint::Length(1), // message
            Constraint::Min(1),    // spacer
            Constraint::Length(1), // footer
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Tab next field · Ctrl+T toggle auto-ramp · Enter run · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[0],
    );

    render_field(
        f,
        rows[1],
        "endpoint",
        &state.endpoint,
        "(http://127.0.0.1:8000)",
        state.active_field == Field::Endpoint,
        theme,
    );
    render_field(
        f,
        rows[2],
        "model   ",
        &state.model,
        "(optional — auto-detected from /v1/models)",
        state.active_field == Field::Model,
        theme,
    );

    // Concurrency row shows auto-ramp status.
    let (conc_label, conc_value, conc_hint) = if state.auto_ramp {
        (
            "auto-ramp",
            "[ON]",
            "1,2,4,8,16,32,64,128 — stops at saturation",
        )
    } else {
        (
            "concurrency",
            state.concurrency.as_str(),
            "comma-separated, e.g. 1,8,32,64",
        )
    };
    let conc_focused = state.active_field == Field::Concurrency;
    let conc_style = if conc_focused {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.fg)
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!("{conc_label:<12} "),
                Style::default().fg(theme.muted),
            ),
            Span::styled(conc_value, conc_style),
            Span::styled(format!("  {conc_hint}"), Style::default().fg(theme.muted)),
        ])),
        rows[3],
    );

    render_field(
        f,
        rows[4],
        "out     ",
        &state.out,
        &state.default_out_hint,
        state.active_field == Field::Out,
        theme,
    );

    let msg = state.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default().fg(theme.err),
        ))),
        rows[5],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "note: local saturation smoke-test — client-measured throughput, not an official ROCm/AMD benchmark.",
            Style::default().fg(theme.muted),
        ))),
        rows[7],
    );
}

fn render_field(
    f: &mut Frame,
    area: Rect,
    label: &str,
    value: &str,
    hint: &str,
    focused: bool,
    theme: &Theme,
) {
    let shown = if value.is_empty() {
        hint.to_string()
    } else {
        value.to_string()
    };
    let value_style = if focused {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else if value.is_empty() {
        Style::default().fg(theme.muted)
    } else {
        Style::default().fg(theme.fg)
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(format!("{label:<12} "), Style::default().fg(theme.muted)),
            Span::styled(shown, value_style),
        ])),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn type_str(br: &mut Option<BenchRunState>, jobs: &mut State, s: &str) {
        for ch in s.chars() {
            on_key(br, jobs, key(KeyCode::Char(ch)));
        }
    }

    /// Build a form pre-filled with a valid endpoint.
    fn with_endpoint(endpoint: &str) -> Option<BenchRunState> {
        let mut s = BenchRunState::new(None);
        s.endpoint = endpoint.to_string();
        Some(s)
    }

    // ---------- T12: submit → exactly one SpawnJob with correct args ----------

    #[test]
    fn t12_submit_emits_spawn_job_with_correct_args() {
        // Verify https:// is rejected pre-spawn.
        let mut br = with_endpoint("https://example.com");
        let mut jobs = State::default();
        let fx = on_key(&mut br, &mut jobs, key(KeyCode::Enter));
        assert!(
            fx.is_empty(),
            "https:// must be rejected before spawning, got: {fx:?}"
        );
        assert!(
            br.as_ref().unwrap().message.is_some(),
            "error message must be set on https:// rejection"
        );

        // Valid http:// endpoint → spawns one job with the right args.
        let mut br = with_endpoint("http://127.0.0.1:8000");
        let mut jobs = State::default();
        // Tab to model, type a model name.
        on_key(&mut br, &mut jobs, key(KeyCode::Tab));
        type_str(&mut br, &mut jobs, "my-model");
        // Tab to concurrency, leave default "1,8,32,64".
        on_key(&mut br, &mut jobs, key(KeyCode::Tab));
        // Tab to out, leave empty.
        on_key(&mut br, &mut jobs, key(KeyCode::Tab));
        // Back to endpoint field; focus is now Out — press Enter.
        let fx = on_key(&mut br, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "exactly one SideEffect must be emitted");

        match &fx[0] {
            SideEffect::SpawnJob { cmd: _, args, .. } => {
                assert!(
                    args.contains(&"bench".to_string()),
                    "args must include 'bench'"
                );
                assert!(
                    args.contains(&"load".to_string()),
                    "args must include 'load'"
                );
                assert!(
                    args.contains(&"--endpoint".to_string()),
                    "args must include '--endpoint'"
                );
                assert!(
                    args.contains(&"http://127.0.0.1:8000".to_string()),
                    "args must include the endpoint URL"
                );
                assert!(
                    args.contains(&"--model".to_string()),
                    "args must include '--model'"
                );
                assert!(
                    args.contains(&"my-model".to_string()),
                    "args must include the model name"
                );
                // auto-ramp is off by default — --concurrency should be present.
                assert!(
                    args.contains(&"--concurrency".to_string()),
                    "args must include '--concurrency' when auto-ramp is off"
                );
                assert!(
                    !args.contains(&"--auto-ramp".to_string()),
                    "args must NOT include '--auto-ramp' when toggle is off"
                );
            }
            other => panic!("expected SpawnJob, got {other:?}"),
        }
        // Form must be closed after submit.
        assert!(br.is_none(), "form must close after successful submit");
    }

    #[test]
    fn t12_auto_ramp_toggle_replaces_concurrency() {
        let mut br = with_endpoint("http://127.0.0.1:8000");
        let mut jobs = State::default();
        // Toggle auto-ramp with Ctrl+T.
        on_key(
            &mut br,
            &mut jobs,
            KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
        );
        assert!(
            br.as_ref().unwrap().auto_ramp,
            "auto_ramp should be true after Ctrl+T"
        );

        let fx = on_key(&mut br, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "exactly one SideEffect");
        match &fx[0] {
            SideEffect::SpawnJob { args, .. } => {
                assert!(
                    args.contains(&"--auto-ramp".to_string()),
                    "args must include '--auto-ramp' when toggle is on"
                );
                assert!(
                    !args.contains(&"--concurrency".to_string()),
                    "args must NOT include '--concurrency' when auto-ramp is on"
                );
            }
            other => panic!("expected SpawnJob, got {other:?}"),
        }
    }

    #[test]
    fn t12_tailed_path_used_as_default_out() {
        use std::path::Path;
        let tailed = Path::new("/var/rocm/bench/results.csv");
        let mut br = Some(BenchRunState::new(Some(tailed)));
        br.as_mut().unwrap().endpoint = "http://127.0.0.1:8000".to_string();
        let mut jobs = State::default();
        let fx = on_key(&mut br, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1);
        match &fx[0] {
            SideEffect::SpawnJob { args, .. } => {
                let out_idx = args.iter().position(|a| a == "--out");
                assert!(out_idx.is_some(), "--out must be injected for tailed path");
                let out_val = &args[out_idx.unwrap() + 1];
                assert_eq!(
                    out_val, "/var/rocm/bench/results.csv",
                    "tailed path must be used as --out default"
                );
            }
            other => panic!("expected SpawnJob, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_submit_keeps_form_open_with_message() {
        let mut jobs = State::default();
        let mut first = with_endpoint("http://127.0.0.1:8000");
        assert_eq!(on_key(&mut first, &mut jobs, key(KeyCode::Enter)).len(), 1);

        let mut duplicate = with_endpoint("http://127.0.0.1:8000");
        let fx = on_key(&mut duplicate, &mut jobs, key(KeyCode::Enter));

        assert!(fx.is_empty());
        let state = duplicate.expect("deduplicated submission must keep form open");
        assert!(
            state
                .message
                .as_deref()
                .unwrap_or("")
                .contains("already running")
        );
    }

    #[test]
    fn empty_out_hint_describes_shared_default_results_file() {
        let state = BenchRunState::new(None);
        assert!(state.default_out_hint.contains("bench/results.csv"));
        assert!(!state.default_out_hint.contains("<ts>"));
    }

    #[test]
    fn t13_esc_closes_bench_run() {
        let mut br = Some(BenchRunState::new(None));
        let mut jobs = State::default();
        let fx = on_key(&mut br, &mut jobs, key(KeyCode::Esc));
        assert!(fx.is_empty());
        assert!(br.is_none(), "Esc must close the form");
    }

    #[test]
    fn t13_empty_endpoint_rejected() {
        let mut br = Some(BenchRunState::new(None));
        let mut jobs = State::default();
        // endpoint is empty by default.
        let fx = on_key(&mut br, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty(), "empty endpoint must be rejected");
        assert!(br.is_some(), "form stays open on validation error");
        assert!(
            br.as_ref().unwrap().message.is_some(),
            "error message must be set"
        );
    }
}
