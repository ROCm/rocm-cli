// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Command runner overlay (Phase 3 Wave 3).
//!
//! A general escape hatch: type any `rocm …` subcommand and run it through the
//! job-bridge. Because arbitrary subcommands can't be statically classified as
//! read-only, **every** freeform command routes through the approval gate first
//! — the user reviews the exact argv before anything runs. This is the general
//! job-bridge consumer that replaces the frozen `command_screen`. Zero
//! `std::thread::spawn`/`try_recv`.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::approval::{
    ApprovalChoice, ApprovalRequest, ApprovalVerdict, approval_key, draw_approval,
};
use crate::ui::exec::{exe_label, resolve_exe};
use crate::ui::job_console::{ConsoleOutcome, draw_job_console, on_console_key};
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// One stable console for the runner — a second command waits for the first.
const JOB_ID: &str = "command";

/// An approved-but-not-yet-run freeform command.
#[derive(Debug, Clone)]
pub struct PendingCommand {
    pub cmd: String,
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct CommandScreenState {
    pub input: String,
    pub approval: Option<PendingCommand>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Parse the freeform input into argv (whitespace-split). The approval modal
/// shows the parsed argv so the user verifies it before running; quoting is a
/// documented fast-follow.
fn parse_args(input: &str) -> Vec<String> {
    input.split_whitespace().map(str::to_string).collect()
}

/// Whether the argv looks like it carries (or sets) a secret — so the approval
/// modal can warn that a key passed as an argument would be exposed in the
/// process argv and the job log. Defends the env-only key invariant at the one
/// place the TUI accepts freeform text: the escape hatch never *stores* a key,
/// but it should not let one slip through unflagged.
fn looks_secret_bearing(args: &[String]) -> bool {
    args.iter().any(|a| {
        let a = a.to_ascii_lowercase();
        a.contains("key") || a.contains("token") || a.contains("secret") || a.contains("password")
    })
}

/// Handle a key while the overlay is open.
pub fn on_key(
    cs: &mut Option<CommandScreenState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(c) = cs.as_mut() else {
        return Vec::new();
    };

    // 1) Approval modal has focus.
    if let Some(pending) = c.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = c.approval.take() {
                    return spawn_command(c, jobs, pending.cmd, pending.args);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => c.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 2) A job is showing in the console.
    if let Some(job_id) = c.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *cs = None,
            ConsoleOutcome::Dismissed => {
                c.active_job = None;
                c.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3) Input editing. Esc closes; printable chars (incl. `q`) are captured
    // into the command line — `q` is not a close key in a text field.
    match key.code {
        KeyCode::Esc => *cs = None,
        KeyCode::Backspace => {
            c.input.pop();
        }
        KeyCode::Enter => return request_run(c),
        KeyCode::Char(ch) => c.input.push(ch),
        _ => {}
    }
    Vec::new()
}

/// Validate the input and stage an approval (no job until approved).
fn request_run(c: &mut CommandScreenState) -> Vec<SideEffect> {
    let args = parse_args(&c.input);
    if args.is_empty() {
        c.message = Some("type a rocm subcommand to run".to_string());
        return Vec::new();
    }
    let cmd = resolve_exe();
    let mut body = vec![
        format!("{} {}", exe_label(&cmd), args.join(" ")),
        String::new(),
        "This runs the command above through ROCm CLI.".to_string(),
    ];
    if looks_secret_bearing(&args) {
        body.push(String::new());
        body.push(
            "Warning: this looks like it carries a secret. Anything passed as an \
             argument is visible in the process list and the job log — prefer the \
             environment for API keys."
                .to_string(),
        );
    }
    let request = ApprovalRequest::new("run command".to_string(), body);
    c.message = None;
    c.approval = Some(PendingCommand {
        cmd,
        args,
        request,
        choice: ApprovalChoice::default(),
    });
    Vec::new()
}

/// Spawn the approved command job.
fn spawn_command(
    c: &mut CommandScreenState,
    jobs: &mut State,
    cmd: String,
    args: Vec<String>,
) -> Vec<SideEffect> {
    let fx = jobs.apply(StateEvent::StartJob {
        id: JOB_ID.to_string(),
        cmd,
        args,
    });
    if fx.is_empty() {
        c.message = Some("a command is already running".to_string());
        return fx;
    }
    c.active_job = Some(JOB_ID.to_string());
    fx
}

/// Render the overlay (input, or an approval/console on top).
pub fn draw_command_screen(
    f: &mut Frame,
    area: Rect,
    c: &CommandScreenState,
    jobs: &State,
    theme: &Theme,
) {
    if let Some(job_id) = &c.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(72, 40, 100, 12, area);
    let inner = draw_popup_frame(f, popup, "Run a command", theme);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Enter a ROCm CLI subcommand (it runs through the approval gate):",
            Style::default().fg(theme.muted),
        ))),
        rows[0],
    );

    let shown = if c.input.is_empty() {
        "(e.g. examine · services list · version)".to_string()
    } else {
        format!("rocm {}", c.input)
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            shown,
            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
        ))),
        rows[1],
    );

    let msg = c.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[2],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Enter run (needs approval) · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[4],
    );

    if let Some(pending) = &c.approval {
        draw_approval(f, area, &pending.request, pending.choice, theme);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn type_str(cs: &mut Option<CommandScreenState>, jobs: &mut State, s: &str) {
        for ch in s.chars() {
            on_key(cs, jobs, key(KeyCode::Char(ch)));
        }
    }

    #[test]
    fn empty_input_is_rejected() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut cs, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(cs.as_ref().unwrap().approval.is_none());
        assert!(
            cs.as_ref()
                .unwrap()
                .message
                .as_deref()
                .unwrap_or("")
                .contains("type a rocm subcommand")
        );
    }

    #[test]
    fn every_command_is_gated_then_spawns() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        // Even a read-only-looking command goes through approval (safe default).
        type_str(&mut cs, &mut jobs, "examine");
        let fx = on_key(&mut cs, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty(), "must not run before approval");
        let pending = cs.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.args, vec!["examine"]);
        assert!(jobs.jobs.is_empty());
        let fx = on_key(&mut cs, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(cs.as_ref().unwrap().active_job.as_deref(), Some("command"));
    }

    #[test]
    fn args_are_whitespace_split() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        type_str(&mut cs, &mut jobs, "services list");
        on_key(&mut cs, &mut jobs, key(KeyCode::Enter));
        let pending = cs.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.args, vec!["services", "list"]);
    }

    #[test]
    fn q_is_typed_into_input_not_a_close_key() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        type_str(&mut cs, &mut jobs, "q");
        assert!(cs.is_some(), "q is a printable char in the input field");
        assert_eq!(cs.as_ref().unwrap().input, "q");
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        on_key(&mut cs, &mut jobs, key(KeyCode::Esc));
        assert!(cs.is_none());
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        type_str(&mut cs, &mut jobs, "version");
        on_key(&mut cs, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut cs, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(cs.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        type_str(&mut cs, &mut jobs, "examine");
        on_key(&mut cs, &mut jobs, key(KeyCode::Enter)); // stage
        on_key(&mut cs, &mut jobs, key(KeyCode::Char('y'))); // spawn
        on_key(&mut cs, &mut jobs, key(KeyCode::Char('q')));
        assert!(cs.is_none(), "q closes the overlay while a job runs");
    }

    #[test]
    fn secret_bearing_command_gets_a_warning_in_the_approval() {
        let mut cs = Some(CommandScreenState::default());
        let mut jobs = State::default();
        type_str(&mut cs, &mut jobs, "config set-provider-key anthropic");
        on_key(&mut cs, &mut jobs, key(KeyCode::Enter));
        let pending = cs.as_ref().unwrap().approval.as_ref().unwrap();
        assert!(
            pending
                .request
                .body
                .iter()
                .any(|l| l.contains("carries a secret")),
            "secret-shaped argv must surface a warning"
        );
        // A plain command does NOT get the warning.
        let mut cs2 = Some(CommandScreenState::default());
        type_str(&mut cs2, &mut jobs, "examine");
        on_key(&mut cs2, &mut jobs, key(KeyCode::Enter));
        let p2 = cs2.as_ref().unwrap().approval.as_ref().unwrap();
        assert!(
            !p2.request
                .body
                .iter()
                .any(|l| l.contains("carries a secret"))
        );
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        let mut jobs = State::default();
        let mut c1 = Some(CommandScreenState::default());
        type_str(&mut c1, &mut jobs, "examine");
        on_key(&mut c1, &mut jobs, key(KeyCode::Enter));
        on_key(&mut c1, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(c1.as_ref().unwrap().active_job.as_deref(), Some("command"));
        // Fresh runner, another command while the first still runs.
        let mut c2 = Some(CommandScreenState::default());
        type_str(&mut c2, &mut jobs, "version");
        on_key(&mut c2, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut c2, &mut jobs, key(KeyCode::Char('y')));
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = c2.as_ref().unwrap();
        assert!(s.active_job.is_none());
        assert!(
            s.message
                .as_deref()
                .unwrap_or("")
                .contains("already running")
        );
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn snapshot_renders_prompt() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(104, 24);
        let mut term = Terminal::new(backend).unwrap();
        let c = CommandScreenState {
            input: "examine".into(),
            ..Default::default()
        };
        let jobs = State::default();
        term.draw(|f| draw_command_screen(f, f.area(), &c, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Run a command"));
        assert!(out.contains("rocm examine"));
    }
}
