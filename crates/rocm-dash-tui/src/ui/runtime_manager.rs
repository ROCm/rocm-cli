//! Runtime manager overlay (Phase 3 Wave 2).
//!
//! Lists the registered ROCm runtimes (plumbed in from the bin, which owns
//! `rocm-core` — this crate stays core-free per `LRN-20260405-004`) and runs the
//! `rocm runtimes …` lifecycle verbs:
//!
//! - **refresh** (`rocm runtimes list`) — read-only, spawns straight through the
//!   job-bridge.
//! - **activate / rollback / uninstall** — mutating, routed through the approval
//!   gate first.
//! - **adopt** — pick an existing ROCm env folder with the Wave-0
//!   [`FolderBrowser`], then approve `rocm runtimes adopt`.
//! - **import** — type a manifest path, then approve `rocm runtimes import`.
//!
//! Zero `std::thread::spawn`/`try_recv`: every command runs as a job-bridge job.

use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::approval::{
    ApprovalChoice, ApprovalRequest, ApprovalVerdict, approval_key, draw_approval,
};
use crate::ui::exec::{exe_label, resolve_exe};
use crate::ui::folder_browser::{FolderBrowser, FolderOutcome, draw_folder_browser};
use crate::ui::job_console::{ConsoleOutcome, draw_job_console, on_console_key};
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// A flattened registered-runtime entry for the list. Mirrors the fields of the
/// bin's `InstalledRuntimeManifest` (+ active/rollback status from config) that
/// the manager needs, with no `rocm-core` dependency.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RuntimeSummary {
    /// Stable runtime key (what `rocm runtimes activate <key>` takes).
    pub key: String,
    /// Friendly runtime id.
    pub id: String,
    /// Install channel (release/nightly/…).
    pub channel: String,
    /// ROCm version string.
    pub version: String,
    /// Install root path (display only).
    pub root: String,
    /// Whether this runtime is the active default.
    pub active: bool,
    /// Whether this runtime is the rollback target (`previous_runtime_key`).
    pub rollback: bool,
}

/// The mutating lifecycle verbs (everything except the read-only refresh).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAction {
    Activate,
    Rollback,
    Uninstall,
    Adopt,
    Import,
}

impl RuntimeAction {
    /// Stable job id (one console per verb).
    fn job_id(self) -> &'static str {
        match self {
            RuntimeAction::Activate => "runtimes-activate",
            RuntimeAction::Rollback => "runtimes-rollback",
            RuntimeAction::Uninstall => "runtimes-uninstall",
            RuntimeAction::Adopt => "runtimes-adopt",
            RuntimeAction::Import => "runtimes-import",
        }
    }

    fn title(self) -> &'static str {
        match self {
            RuntimeAction::Activate => "activate runtime",
            RuntimeAction::Rollback => "roll back to previous runtime",
            RuntimeAction::Uninstall => "uninstall runtime",
            RuntimeAction::Adopt => "adopt existing ROCm folder",
            RuntimeAction::Import => "import runtime manifest",
        }
    }

    fn explanation(self) -> &'static str {
        match self {
            RuntimeAction::Activate => "This switches the default ROCm runtime for this machine.",
            RuntimeAction::Rollback => "This restores the previously active ROCm runtime.",
            RuntimeAction::Uninstall => "This removes the runtime record from ROCm CLI.",
            RuntimeAction::Adopt => "This registers an existing ROCm folder without modifying it.",
            RuntimeAction::Import => "This registers a runtime from a saved manifest file.",
        }
    }
}

/// An approved-but-not-yet-run mutating runtime op.
#[derive(Debug, Clone)]
pub struct PendingRuntime {
    pub action: RuntimeAction,
    pub cmd: String,
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed. The runtime
/// list itself lives on `AppState` (plumbed from the bin), not here.
#[derive(Debug, Clone, Default)]
pub struct RuntimeManagerState {
    pub selected: usize,
    /// Adopt folder picker (env root). `Some` = picker has focus.
    pub browser: Option<FolderBrowser>,
    /// Import manifest path prompt. `Some` = the text prompt has focus.
    pub import_input: Option<String>,
    pub approval: Option<PendingRuntime>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Derive the conventional Python executable inside an env root, matching
/// rocm-core's per-host layout (`Scripts\python.exe` on Windows, `bin/python`
/// elsewhere) without taking a `rocm-core` dependency. The approval modal shows
/// the exact argv, so the user verifies it before anything runs.
fn derive_python_executable(root: &str) -> String {
    let root = Path::new(root);
    let path = if cfg!(windows) {
        root.join("Scripts").join("python.exe")
    } else {
        root.join("bin").join("python")
    };
    path.to_string_lossy().into_owned()
}

/// Handle a key while the overlay is open. `runtimes` is the plumbed list.
pub fn on_key(
    rm: &mut Option<RuntimeManagerState>,
    runtimes: &[RuntimeSummary],
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(r) = rm.as_mut() else {
        return Vec::new();
    };

    // 1) Adopt folder browser has focus.
    if let Some(fb) = r.browser.as_mut() {
        match fb.on_key(key.code) {
            FolderOutcome::Chosen(path) => {
                r.browser = None;
                let root = path.to_string_lossy().into_owned();
                let args = vec![
                    "runtimes".to_string(),
                    "adopt".to_string(),
                    "--root".to_string(),
                    root.clone(),
                    "--python".to_string(),
                    derive_python_executable(&root),
                ];
                stage_approval(r, RuntimeAction::Adopt, args);
            }
            FolderOutcome::Cancelled => r.browser = None,
            FolderOutcome::None | FolderOutcome::Navigated => {}
        }
        return Vec::new();
    }

    // 2) Import path prompt has focus.
    if let Some(input) = r.import_input.as_mut() {
        match key.code {
            KeyCode::Esc => r.import_input = None,
            KeyCode::Backspace => {
                input.pop();
            }
            KeyCode::Enter => {
                let path = input.trim().to_string();
                if path.is_empty() {
                    r.message = Some("manifest path is required".to_string());
                    r.import_input = None;
                } else {
                    r.import_input = None;
                    let args = vec!["runtimes".to_string(), "import".to_string(), path];
                    stage_approval(r, RuntimeAction::Import, args);
                }
            }
            KeyCode::Char(c) => input.push(c),
            _ => {}
        }
        return Vec::new();
    }

    // 3) Approval modal has focus.
    if let Some(pending) = r.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = r.approval.take() {
                    return spawn_runtime(r, jobs, pending.action, pending.cmd, pending.args);
                }
            }
            Some(ApprovalVerdict::Deny) | Some(ApprovalVerdict::Cancel) => r.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 4) A job is showing in the console.
    if let Some(job_id) = r.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *rm = None,
            ConsoleOutcome::Dismissed => {
                r.active_job = None;
                r.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 5) List navigation + actions.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *rm = None,
        KeyCode::Up | KeyCode::Char('k') => r.selected = r.selected.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') if !runtimes.is_empty() => {
            r.selected = (r.selected + 1).min(runtimes.len() - 1);
        }
        KeyCode::Char('l') => return spawn_refresh(r, jobs),
        KeyCode::Enter | KeyCode::Char('a') => {
            if let Some(rt) = runtimes.get(r.selected) {
                let args = vec![
                    "runtimes".to_string(),
                    "activate".to_string(),
                    rt.key.clone(),
                ];
                stage_approval(r, RuntimeAction::Activate, args);
            } else {
                r.message = Some("no runtime selected".to_string());
            }
        }
        KeyCode::Char('r') => {
            let args = vec!["runtimes".to_string(), "rollback".to_string()];
            stage_approval(r, RuntimeAction::Rollback, args);
        }
        KeyCode::Char('x') => {
            if let Some(rt) = runtimes.get(r.selected) {
                let args = vec![
                    "runtimes".to_string(),
                    "uninstall".to_string(),
                    rt.key.clone(),
                ];
                stage_approval(r, RuntimeAction::Uninstall, args);
            } else {
                r.message = Some("no runtime selected".to_string());
            }
        }
        KeyCode::Char('o') => {
            let start = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
            r.browser = Some(FolderBrowser::new("Pick an existing ROCm folder", start));
        }
        KeyCode::Char('i') => r.import_input = Some(String::new()),
        _ => {}
    }
    Vec::new()
}

/// Stage an approval for a mutating verb (no job yet).
fn stage_approval(r: &mut RuntimeManagerState, action: RuntimeAction, args: Vec<String>) {
    let cmd = resolve_exe();
    let request = ApprovalRequest::new(
        action.title().to_string(),
        vec![
            format!("{} {}", exe_label(&cmd), args.join(" ")),
            String::new(),
            action.explanation().to_string(),
        ],
    );
    r.message = None;
    r.approval = Some(PendingRuntime {
        action,
        cmd,
        args,
        request,
        choice: ApprovalChoice::default(),
    });
}

/// Spawn the read-only `rocm runtimes list` refresh.
fn spawn_refresh(r: &mut RuntimeManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let id = "runtimes-list".to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: resolve_exe(),
        args: vec!["runtimes".to_string(), "list".to_string()],
    });
    if fx.is_empty() {
        r.message = Some("a runtimes refresh is already running".to_string());
        return fx;
    }
    r.active_job = Some(id);
    fx
}

/// Spawn the approved mutating job.
fn spawn_runtime(
    r: &mut RuntimeManagerState,
    jobs: &mut State,
    action: RuntimeAction,
    cmd: String,
    args: Vec<String>,
) -> Vec<SideEffect> {
    let id = action.job_id().to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args,
    });
    if fx.is_empty() {
        r.message = Some(format!("“{}” is already running", action.title()));
        return fx;
    }
    r.active_job = Some(id);
    fx
}

/// Render the overlay (list, or a browser/prompt/approval/console on top).
pub fn draw_runtime_manager(
    f: &mut Frame,
    area: Rect,
    r: &RuntimeManagerState,
    runtimes: &[RuntimeSummary],
    jobs: &State,
    theme: &Theme,
) {
    if let Some(job_id) = &r.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(80, 72, 120, 24, area);
    let inner = draw_popup_frame(f, popup, "Runtimes — ROCm installs", theme);
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

    if runtimes.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No runtimes registered. Press o to adopt an existing folder, \
                 i to import a manifest, or l to refresh.",
                Style::default().fg(theme.muted),
            ))),
            rows[0],
        );
    } else {
        let items: Vec<ListItem> = runtimes
            .iter()
            .map(|rt| {
                let marker = if rt.active {
                    "● "
                } else if rt.rollback {
                    "↺ "
                } else {
                    "  "
                };
                let marker_color = if rt.active {
                    theme.ok
                } else if rt.rollback {
                    theme.warn
                } else {
                    theme.muted
                };
                ListItem::new(Line::from(vec![
                    Span::styled(marker, Style::default().fg(marker_color)),
                    Span::styled(
                        format!("{:<26}", trunc(&rt.key, 26)),
                        Style::default().fg(theme.fg),
                    ),
                    Span::styled(
                        format!("{:<10}", trunc(&rt.channel, 10)),
                        Style::default().fg(theme.muted),
                    ),
                    Span::styled(
                        format!("{:<14}", trunc(&rt.version, 14)),
                        Style::default().fg(theme.accent),
                    ),
                    Span::styled(trunc(&rt.root, 40), Style::default().fg(theme.muted)),
                ]))
            })
            .collect();
        let mut ls = ListState::default();
        ls.select(Some(r.selected.min(runtimes.len().saturating_sub(1))));
        let list = List::new(items).highlight_style(
            Style::default()
                .bg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        );
        f.render_stateful_widget(list, rows[0], &mut ls);
    }

    let msg = r.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[1],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ select · Enter/a activate · r rollback · x uninstall · o adopt · i import · l refresh · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    if let Some(fb) = &r.browser {
        draw_folder_browser(f, area, fb, theme);
    }
    if let Some(input) = &r.import_input {
        draw_import_prompt(f, area, input, theme);
    }
    if let Some(pending) = &r.approval {
        draw_approval(f, area, &pending.request, pending.choice, theme);
    }
}

/// Render the import manifest-path prompt.
fn draw_import_prompt(f: &mut Frame, area: Rect, input: &str, theme: &Theme) {
    let popup = centered_rect(70, 30, 96, 8, area);
    let inner = draw_popup_frame(f, popup, "Import runtime manifest", theme);
    if inner.height == 0 {
        return;
    }
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(inner);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Manifest file path:",
            Style::default().fg(theme.muted),
        ))),
        rows[0],
    );
    let shown = if input.is_empty() {
        "(type a path)".to_string()
    } else {
        input.to_string()
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            shown,
            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
        ))),
        rows[1],
    );
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "Enter confirm · Esc cancel",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );
}

fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let keep: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{keep}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn runtimes() -> Vec<RuntimeSummary> {
        vec![
            RuntimeSummary {
                key: "therock-release-gfx94".into(),
                id: "rocm-6.4".into(),
                channel: "release".into(),
                version: "6.4.0".into(),
                root: "/opt/rocm-6.4".into(),
                active: true,
                rollback: false,
            },
            RuntimeSummary {
                key: "therock-nightly-gfx94".into(),
                id: "rocm-nightly".into(),
                channel: "nightly".into(),
                version: "6.5.0-dev".into(),
                root: "/opt/rocm-nightly".into(),
                active: false,
                rollback: true,
            },
        ]
    }

    #[test]
    fn refresh_is_read_only_and_spawns_without_approval() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut rm, &runtimes(), &mut jobs, key(KeyCode::Char('l')));
        assert_eq!(fx.len(), 1);
        assert!(
            rm.as_ref().unwrap().approval.is_none(),
            "no gate for refresh"
        );
        assert_eq!(
            rm.as_ref().unwrap().active_job.as_deref(),
            Some("runtimes-list")
        );
    }

    #[test]
    fn activate_is_gated_then_spawns_with_selected_key() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        let rts = runtimes();
        // Enter on row 0 stages approval, NO job.
        let fx = on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        let pending = rm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, RuntimeAction::Activate);
        assert_eq!(
            pending.args,
            vec!["runtimes", "activate", "therock-release-gfx94"]
        );
        assert!(jobs.jobs.is_empty());
        // Approve → spawns.
        let fx = on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(
            rm.as_ref().unwrap().active_job.as_deref(),
            Some("runtimes-activate")
        );
    }

    #[test]
    fn uninstall_targets_the_selected_row() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        let rts = runtimes();
        on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Down)); // select row 1
        on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Char('x')));
        let pending = rm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, RuntimeAction::Uninstall);
        assert_eq!(
            pending.args,
            vec!["runtimes", "uninstall", "therock-nightly-gfx94"]
        );
    }

    #[test]
    fn rollback_needs_no_selection() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('r')));
        let pending = rm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, RuntimeAction::Rollback);
        assert_eq!(pending.args, vec!["runtimes", "rollback"]);
    }

    #[test]
    fn activate_with_no_runtimes_surfaces_message_not_approval() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut rm, &[], &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(rm.as_ref().unwrap().approval.is_none());
        assert!(
            rm.as_ref()
                .unwrap()
                .message
                .as_deref()
                .unwrap_or("")
                .contains("no runtime selected")
        );
    }

    #[test]
    fn adopt_opens_browser_then_stages_approval_with_python() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('o')));
        assert!(rm.as_ref().unwrap().browser.is_some());
        // Drive the browser: Enter on row 0 = UseCurrent → chooses cwd.
        let fx = on_key(&mut rm, &[], &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(rm.as_ref().unwrap().browser.is_none());
        let pending = rm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, RuntimeAction::Adopt);
        assert!(pending.args.iter().any(|a| a == "adopt"));
        assert!(pending.args.iter().any(|a| a == "--root"));
        assert!(pending.args.iter().any(|a| a == "--python"));
        // The derived python path uses the host bin/Scripts convention.
        let want = if cfg!(windows) {
            "python.exe"
        } else {
            "bin/python"
        };
        assert!(pending.args.iter().any(|a| a.contains(want)));
    }

    #[test]
    fn import_prompt_then_stages_approval() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('i')));
        assert!(rm.as_ref().unwrap().import_input.is_some());
        for c in "/tmp/m.json".chars() {
            on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char(c)));
        }
        let fx = on_key(&mut rm, &[], &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(rm.as_ref().unwrap().import_input.is_none());
        let pending = rm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, RuntimeAction::Import);
        assert_eq!(pending.args, vec!["runtimes", "import", "/tmp/m.json"]);
    }

    #[test]
    fn import_empty_path_is_rejected() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('i')));
        let fx = on_key(&mut rm, &[], &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(rm.as_ref().unwrap().approval.is_none());
        assert!(rm.as_ref().unwrap().import_input.is_none());
        assert!(
            rm.as_ref()
                .unwrap()
                .message
                .as_deref()
                .unwrap_or("")
                .contains("required")
        );
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        let rts = runtimes();
        on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Enter)); // stage activate
        let fx = on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(rm.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn navigation_clamps() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        let rts = runtimes();
        for _ in 0..10 {
            on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Down));
        }
        assert_eq!(rm.as_ref().unwrap().selected, rts.len() - 1);
        for _ in 0..10 {
            on_key(&mut rm, &rts, &mut jobs, key(KeyCode::Up));
        }
        assert_eq!(rm.as_ref().unwrap().selected, 0);
    }

    #[test]
    fn q_closes_overlay_when_idle() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('q')));
        assert!(rm.is_none());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut rm = Some(RuntimeManagerState::default());
        let mut jobs = State::default();
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('l'))); // refresh spawns
        on_key(&mut rm, &[], &mut jobs, key(KeyCode::Char('q')));
        assert!(rm.is_none());
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        // A StartJob for a still-running id no-ops, so spawn_refresh must surface
        // a message and NOT point active_job at it.
        let mut jobs = State::default();
        let mut rm1 = Some(RuntimeManagerState::default());
        on_key(&mut rm1, &[], &mut jobs, key(KeyCode::Char('l')));
        assert_eq!(
            rm1.as_ref().unwrap().active_job.as_deref(),
            Some("runtimes-list")
        );
        let mut rm2 = Some(RuntimeManagerState::default());
        let fx = on_key(&mut rm2, &[], &mut jobs, key(KeyCode::Char('l')));
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = rm2.as_ref().unwrap();
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
    fn snapshot_lists_runtimes_and_markers() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 22);
        let mut term = Terminal::new(backend).unwrap();
        let r = RuntimeManagerState::default();
        let rts = runtimes();
        let jobs = State::default();
        term.draw(|f| draw_runtime_manager(f, f.area(), &r, &rts, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(out.contains("Runtimes"));
        assert!(out.contains("therock-release-gfx94"));
        assert!(out.contains("activate"));
    }

    #[test]
    fn snapshot_empty_shows_hint() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 22);
        let mut term = Terminal::new(backend).unwrap();
        let r = RuntimeManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_runtime_manager(f, f.area(), &r, &[], &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(out.contains("No runtimes registered"));
    }
}
