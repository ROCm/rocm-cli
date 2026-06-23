// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Install overlay (Phase 3 Wave 2).
//!
//! A compact form that builds a `rocm install sdk …` invocation (TheRock ROCm
//! wheels). Dry-run is read-only and spawns straight through the job-bridge; a
//! real install is mutating and routes through the approval gate first. The
//! install folder is picked with the Wave-0 [`FolderBrowser`]. Streaming-install
//! archetype, zero thread::spawn/try_recv.

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
use crate::ui::folder_browser::{FolderBrowser, FolderOutcome, draw_folder_browser};
use crate::ui::job_console::{ConsoleOutcome, draw_job_console, on_console_key};
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// Package formats — mirrors `apps/rocm` `InstallFormat` (clap value-enum).
pub const FORMATS: &[&str] = &["wheel", "tarball"];

/// Default install channel (mirrors the `rocm install sdk --channel` default).
const DEFAULT_CHANNEL: &str = "release";

/// Form fields, in vertical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Channel,
    Format,
    Prefix,
    DryRun,
    Launch,
}

pub const FIELDS: &[Field] = &[
    Field::Channel,
    Field::Format,
    Field::Prefix,
    Field::DryRun,
    Field::Launch,
];

/// An approved-but-not-yet-run mutating install.
#[derive(Debug, Clone)]
pub struct PendingInstall {
    pub cmd: String,
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone)]
pub struct InstallManagerState {
    pub field: usize,
    pub channel: String,
    pub format_idx: usize,
    pub prefix: String,
    pub dry_run: bool,
    pub browser: Option<FolderBrowser>,
    pub approval: Option<PendingInstall>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

impl Default for InstallManagerState {
    fn default() -> Self {
        Self {
            field: 0,
            channel: DEFAULT_CHANNEL.to_string(),
            format_idx: 0,
            prefix: String::new(),
            // Default to dry-run: the safe, read-only first step.
            dry_run: true,
            browser: None,
            approval: None,
            active_job: None,
            message: None,
        }
    }
}

impl InstallManagerState {
    fn current_field(&self) -> Field {
        FIELDS[self.field.min(FIELDS.len() - 1)]
    }

    fn move_field(&mut self, delta: isize) {
        let max = FIELDS.len().cast_signed() - 1;
        self.field = (self.field.cast_signed() + delta).clamp(0, max) as usize;
    }

    fn cycle(&mut self) {
        match self.current_field() {
            Field::Format => {
                self.format_idx = (self.format_idx + 1) % FORMATS.len();
            }
            Field::DryRun => self.dry_run = !self.dry_run,
            _ => {}
        }
    }

    fn type_char(&mut self, c: char) {
        match self.current_field() {
            Field::Channel => self.channel.push(c),
            Field::Prefix => self.prefix.push(c),
            _ => {}
        }
    }

    fn backspace(&mut self) {
        match self.current_field() {
            Field::Channel => {
                self.channel.pop();
            }
            Field::Prefix => {
                self.prefix.pop();
            }
            _ => {}
        }
    }

    /// Build the `rocm install sdk …` argv, or an error message.
    fn build_args(&self) -> Result<Vec<String>, String> {
        let channel = self.channel.trim();
        if channel.is_empty() {
            return Err("channel is required (e.g. release)".to_string());
        }
        let mut args = vec![
            "install".to_string(),
            "sdk".to_string(),
            "--channel".to_string(),
            channel.to_string(),
            "--format".to_string(),
            FORMATS[self.format_idx.min(FORMATS.len() - 1)].to_string(),
        ];
        let prefix = self.prefix.trim();
        if !prefix.is_empty() {
            args.push("--prefix".to_string());
            args.push(prefix.to_string());
        }
        if self.dry_run {
            args.push("--dry-run".to_string());
        }
        Ok(args)
    }
}

/// Handle a key while the overlay is open.
pub fn on_key(
    install: &mut Option<InstallManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(i) = install.as_mut() else {
        return Vec::new();
    };

    // 1) Folder browser (install prefix) has focus.
    if let Some(fb) = i.browser.as_mut() {
        match fb.on_key(key.code) {
            FolderOutcome::Chosen(path) => {
                i.prefix = path.to_string_lossy().into_owned();
                i.browser = None;
            }
            FolderOutcome::Cancelled => i.browser = None,
            FolderOutcome::None | FolderOutcome::Navigated => {}
        }
        return Vec::new();
    }

    // 2) Approval modal has focus.
    if let Some(pending) = i.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = i.approval.take() {
                    return spawn_install(i, jobs, pending.cmd, pending.args);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => i.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 3) A job is showing in the console.
    if let Some(job_id) = i.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *install = None,
            ConsoleOutcome::Dismissed => {
                i.active_job = None;
                i.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 4) Form editing. Form archetype (like serve_wizard): Esc closes; text
    // fields (Channel/Prefix) capture printable chars including `q` — `q` is not
    // a close key here (Esc is). Non-text fields ignore typed chars.
    match key.code {
        KeyCode::Esc => *install = None,
        KeyCode::Up => i.move_field(-1),
        KeyCode::Down => i.move_field(1),
        KeyCode::Left | KeyCode::Right => i.cycle(),
        KeyCode::Char(' ') if i.current_field() == Field::DryRun => i.cycle(),
        KeyCode::Tab if i.current_field() == Field::Prefix => {
            let start = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
            i.browser = Some(FolderBrowser::new("Pick an install folder", start));
        }
        KeyCode::Enter => {
            if i.current_field() == Field::Launch {
                return request_launch(i, jobs);
            }
            i.move_field(1);
        }
        KeyCode::Backspace => i.backspace(),
        KeyCode::Char(c) => i.type_char(c),
        _ => {}
    }
    Vec::new()
}

/// Validate + run: dry-run spawns straight away (read-only); a real install
/// stages an approval first.
fn request_launch(i: &mut InstallManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let args = match i.build_args() {
        Ok(a) => a,
        Err(msg) => {
            i.message = Some(msg);
            return Vec::new();
        }
    };
    let cmd = resolve_exe();
    if i.dry_run {
        // Read-only plan resolution — no approval.
        i.message = None;
        return spawn_install(i, jobs, cmd, args);
    }
    let request = ApprovalRequest::new(
        "install ROCm SDK".to_string(),
        vec![
            format!("{} {}", exe_label(&cmd), args.join(" ")),
            String::new(),
            "This downloads and installs TheRock ROCm wheels on this machine.".to_string(),
        ],
    );
    i.message = None;
    i.approval = Some(PendingInstall {
        cmd,
        args,
        request,
        choice: ApprovalChoice::default(),
    });
    Vec::new()
}

/// Launch the install job. The id distinguishes dry-run from a real install so
/// the two consoles never collide.
fn spawn_install(
    i: &mut InstallManagerState,
    jobs: &mut State,
    cmd: String,
    args: Vec<String>,
) -> Vec<SideEffect> {
    let id = if i.dry_run {
        "install-sdk-dryrun"
    } else {
        "install-sdk"
    }
    .to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args,
    });
    if fx.is_empty() {
        i.message = Some("an install job is already running".to_string());
        return fx;
    }
    i.active_job = Some(id);
    fx
}

/// Render the overlay (form, or browser/approval/console on top).
pub fn draw_install_manager(
    f: &mut Frame,
    area: Rect,
    i: &InstallManagerState,
    jobs: &State,
    theme: &Theme,
) {
    if let Some(job_id) = &i.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(72, 70, 96, 20, area);
    let inner = draw_popup_frame(f, popup, "Install — ROCm SDK (TheRock)", theme);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let lines: Vec<Line> = FIELDS
        .iter()
        .enumerate()
        .map(|(idx, field)| field_line(*field, idx == i.field, i, theme))
        .collect();
    f.render_widget(Paragraph::new(lines), rows[0]);

    let msg = i.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[1],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ field · ←→ cycle (Format/Mode) · Tab browse (prefix) · Enter next/launch · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    if let Some(fb) = &i.browser {
        draw_folder_browser(f, area, fb, theme);
    }
    if let Some(pending) = &i.approval {
        draw_approval(f, area, &pending.request, pending.choice, theme);
    }
}

fn field_line<'a>(
    field: Field,
    selected: bool,
    i: &'a InstallManagerState,
    theme: &Theme,
) -> Line<'a> {
    let (label, value): (&str, String) = match field {
        Field::Channel => ("Channel", display(&i.channel, "(e.g. release)")),
        Field::Format => (
            "Format",
            FORMATS[i.format_idx.min(FORMATS.len() - 1)].to_string(),
        ),
        Field::Prefix => (
            "Folder",
            display(&i.prefix, "(default managed folder · Tab to browse)"),
        ),
        Field::DryRun => (
            "Mode",
            if i.dry_run {
                "dry-run (read-only)".to_string()
            } else {
                "install (needs approval)".to_string()
            },
        ),
        Field::Launch => ("", String::new()),
    };

    if field == Field::Launch {
        let style = if selected {
            Style::default()
                .bg(theme.ok)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.ok)
        };
        let label = if i.dry_run {
            "  [ Resolve plan ]  "
        } else {
            "  [ Install ]  "
        };
        return Line::from(Span::styled(label, style));
    }

    let marker = if selected { "▶ " } else { "  " };
    let label_style = if selected {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.muted)
    };
    let value_style = if selected {
        Style::default().fg(theme.fg).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.fg)
    };
    Line::from(vec![
        Span::styled(marker, label_style),
        Span::styled(format!("{label:<8}"), label_style),
        Span::styled(value, value_style),
    ])
}

fn display(v: &str, placeholder: &'static str) -> String {
    if v.is_empty() {
        placeholder.to_string()
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn default_is_dry_run_release_wheel() {
        let i = InstallManagerState::default();
        assert!(i.dry_run);
        assert_eq!(i.channel, "release");
        assert_eq!(FORMATS[i.format_idx], "wheel");
    }

    #[test]
    fn build_args_dry_run_by_default() {
        let i = InstallManagerState::default();
        let args = i.build_args().unwrap();
        assert_eq!(
            args,
            vec![
                "install",
                "sdk",
                "--channel",
                "release",
                "--format",
                "wheel",
                "--dry-run"
            ]
        );
    }

    #[test]
    fn build_args_install_with_prefix_and_tarball() {
        let i = InstallManagerState {
            format_idx: 1, // tarball
            prefix: "/opt/rocm-sdk".into(),
            dry_run: false,
            ..Default::default()
        };
        let args = i.build_args().unwrap();
        assert!(args.windows(2).any(|p| p == ["--format", "tarball"]));
        assert!(args.windows(2).any(|p| p == ["--prefix", "/opt/rocm-sdk"]));
        assert!(!args.contains(&"--dry-run".to_string()));
    }

    #[test]
    fn build_args_requires_channel() {
        let i = InstallManagerState {
            channel: "   ".into(),
            ..Default::default()
        };
        assert!(i.build_args().unwrap_err().contains("channel"));
    }

    #[test]
    fn dry_run_launch_is_read_only_no_approval() {
        let mut ins = Some(InstallManagerState::default()); // dry_run = true
        let mut jobs = State::default();
        ins.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        let fx = on_key(&mut ins, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1, "dry-run spawns without approval");
        assert!(ins.as_ref().unwrap().approval.is_none());
        assert_eq!(
            ins.as_ref().unwrap().active_job.as_deref(),
            Some("install-sdk-dryrun")
        );
    }

    #[test]
    fn real_install_is_gated_then_spawns() {
        let mut ins = Some(InstallManagerState::default());
        let mut jobs = State::default();
        ins.as_mut().unwrap().dry_run = false;
        ins.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        let fx = on_key(&mut ins, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty(), "install must not run before approval");
        assert!(ins.as_ref().unwrap().approval.is_some());
        assert!(jobs.jobs.is_empty());
        let fx = on_key(&mut ins, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(
            ins.as_ref().unwrap().active_job.as_deref(),
            Some("install-sdk")
        );
    }

    #[test]
    fn tab_on_prefix_opens_browser() {
        let mut ins = Some(InstallManagerState::default());
        let mut jobs = State::default();
        ins.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Prefix).unwrap();
        on_key(&mut ins, &mut jobs, key(KeyCode::Tab));
        assert!(ins.as_ref().unwrap().browser.is_some());
        on_key(&mut ins, &mut jobs, key(KeyCode::Esc));
        assert!(ins.as_ref().unwrap().browser.is_none());
        assert!(ins.is_some());
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut ins = Some(InstallManagerState::default());
        let mut jobs = State::default();
        ins.as_mut().unwrap().dry_run = false;
        ins.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        on_key(&mut ins, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut ins, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(ins.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut ins = Some(InstallManagerState::default());
        let mut jobs = State::default();
        on_key(&mut ins, &mut jobs, key(KeyCode::Esc));
        assert!(ins.is_none());
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        // Mirrors the sibling screens: a StartJob for a still-running id no-ops,
        // so spawn_install must surface a message and NOT set active_job.
        let mut jobs = State::default();
        let launch = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        let mut i1 = Some(InstallManagerState::default()); // dry-run
        i1.as_mut().unwrap().field = launch;
        on_key(&mut i1, &mut jobs, key(KeyCode::Enter));
        assert_eq!(
            i1.as_ref().unwrap().active_job.as_deref(),
            Some("install-sdk-dryrun")
        );
        // Fresh overlay, same dry-run while the prior job still runs.
        let mut i2 = Some(InstallManagerState::default());
        i2.as_mut().unwrap().field = launch;
        let fx = on_key(&mut i2, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = i2.as_ref().unwrap();
        assert!(s.active_job.is_none(), "must not point at the stale job");
        assert!(
            s.message
                .as_deref()
                .unwrap_or("")
                .contains("already running")
        );
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn snapshot_renders_form() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(104, 22);
        let mut term = Terminal::new(backend).unwrap();
        let i = InstallManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_install_manager(f, f.area(), &i, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Install"));
        assert!(out.contains("Channel"));
        assert!(out.contains("dry-run"));
        assert!(out.contains("Resolve plan"));
    }
}
