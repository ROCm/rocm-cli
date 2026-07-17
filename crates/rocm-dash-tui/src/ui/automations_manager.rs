// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Automations manager overlay (Phase 3 Wave 3).
//!
//! Lists the background checks ("watchers", plumbed in from the bin which owns
//! `rocm-core`) and toggles them via the `rocm automations …` verbs:
//!
//! - **refresh** (`rocm automations list`) — read-only, straight to the job-bridge.
//! - **enable / disable** the selected watcher — mutating, routed through the
//!   approval gate first.
//!
//! The Propose-mode proposal review/approval workflow (queued watcher proposals)
//! is a documented fast-follow; this overlay covers list + enable/disable, which
//! is what gates the watchers that generate those proposals. Zero
//! `std::thread::spawn`/`try_recv`.

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
use crate::ui::job_console::{ConsoleOutcome, on_console_key};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// A flattened background-check entry for the list. Mirrors the bin's
/// `BuiltinWatcherSpec` (+ enabled/mode from config) with no `rocm-core` dep.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AutomationSummary {
    /// Watcher id (what `rocm automations enable|disable <id>` takes).
    pub id: String,
    /// Human summary of what the check does.
    pub summary: String,
    /// Whether the check is currently enabled.
    pub enabled: bool,
    /// Effective mode label (observe/propose/contained).
    pub mode: String,
}

/// The mutating toggle verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomationAction {
    Enable,
    Disable,
}

impl AutomationAction {
    const fn verb(self) -> &'static str {
        match self {
            Self::Enable => "enable",
            Self::Disable => "disable",
        }
    }

    const fn job_id(self) -> &'static str {
        match self {
            Self::Enable => "automations-enable",
            Self::Disable => "automations-disable",
        }
    }
}

/// An approved-but-not-yet-run toggle.
#[derive(Debug, Clone)]
pub struct PendingAutomation {
    pub action: AutomationAction,
    pub cmd: String,
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed. The watcher
/// list lives on `AppState` (plumbed from the bin), not here.
#[derive(Debug, Clone, Default)]
pub struct AutomationsManagerState {
    pub selected: usize,
    pub approval: Option<PendingAutomation>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Handle a key while the overlay is open. `automations` is the plumbed list.
pub fn on_key(
    am: &mut Option<AutomationsManagerState>,
    automations: &[AutomationSummary],
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(a) = am.as_mut() else {
        return Vec::new();
    };

    // 1) Approval modal has focus.
    if let Some(pending) = a.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = a.approval.take() {
                    return spawn_toggle(a, jobs, pending.action, pending.cmd, pending.args);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => a.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 2) A job is showing in the console.
    if let Some(job_id) = a.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *am = None,
            ConsoleOutcome::Dismissed => {
                a.active_job = None;
                a.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3) List navigation + actions.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *am = None,
        KeyCode::Up | KeyCode::Char('k') => a.selected = a.selected.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') if !automations.is_empty() => {
            a.selected = (a.selected + 1).min(automations.len() - 1);
        }
        KeyCode::Char('l') => return spawn_refresh(a, jobs),
        KeyCode::Enter | KeyCode::Char(' ') => {
            if let Some(w) = automations.get(a.selected) {
                let action = if w.enabled {
                    AutomationAction::Disable
                } else {
                    AutomationAction::Enable
                };
                let args = vec![
                    "automations".to_string(),
                    action.verb().to_string(),
                    w.id.clone(),
                ];
                stage_approval(a, action, &w.id, args);
            } else {
                a.message = Some("no automation selected".to_string());
            }
        }
        _ => {}
    }
    Vec::new()
}

/// Stage an approval for a toggle (no job yet).
fn stage_approval(
    a: &mut AutomationsManagerState,
    action: AutomationAction,
    watcher_id: &str,
    args: Vec<String>,
) {
    let cmd = resolve_exe();
    let request = ApprovalRequest::new(
        format!("{} background check “{}”", action.verb(), watcher_id),
        vec![
            format!("{} {}", exe_label(&cmd), args.join(" ")),
            String::new(),
            "This changes which background checks run on this machine.".to_string(),
        ],
    );
    a.message = None;
    a.approval = Some(PendingAutomation {
        action,
        cmd,
        args,
        request,
        choice: ApprovalChoice::default(),
    });
}

/// Spawn the read-only `rocm automations list` refresh.
fn spawn_refresh(a: &mut AutomationsManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let id = "automations-list".to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: resolve_exe(),
        args: vec!["automations".to_string(), "list".to_string()],
    });
    if fx.is_empty() {
        a.message = Some("an automations refresh is already running".to_string());
        return fx;
    }
    a.active_job = Some(id);
    fx
}

/// Spawn the approved toggle job.
fn spawn_toggle(
    a: &mut AutomationsManagerState,
    jobs: &mut State,
    action: AutomationAction,
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
        a.message = Some(format!("an {} job is already running", action.verb()));
        return fx;
    }
    a.active_job = Some(id);
    fx
}

/// Render the overlay (list, or an approval/console on top).
pub fn draw_automations_manager(
    f: &mut Frame,
    area: Rect,
    a: &AutomationsManagerState,
    automations: &[AutomationSummary],
    _jobs: &State,
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Automations — background checks"),
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
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    if automations.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No background checks available. Press l to refresh.",
                Style::default().fg(theme.muted),
            ))),
            rows[0],
        );
    } else {
        let items: Vec<ListItem> = automations
            .iter()
            .map(|w| {
                let (badge, badge_color) = if w.enabled {
                    ("● on ", theme.ok)
                } else {
                    ("○ off", theme.muted)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{badge} "), Style::default().fg(badge_color)),
                    Span::styled(
                        format!("{:<22}", trunc(&w.id, 22)),
                        Style::default().fg(theme.fg),
                    ),
                    Span::styled(
                        format!("{:<10}", trunc(&w.mode, 10)),
                        Style::default().fg(theme.accent),
                    ),
                    Span::styled(trunc(&w.summary, 54), Style::default().fg(theme.muted)),
                ]))
            })
            .collect();
        let mut ls = ListState::default();
        ls.select(Some(a.selected.min(automations.len().saturating_sub(1))));
        let list = List::new(items).highlight_style(
            Style::default()
                .bg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        );
        let list_area = panel::vertical_scrollbar(
            f,
            rows[0],
            automations.len(),
            rows[0].height as usize,
            a.selected,
            theme,
        );
        f.render_stateful_widget(list, list_area, &mut ls);
    }

    let msg = a.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[1],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ select · Enter/Space toggle (needs approval) · l refresh · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    if let Some(pending) = &a.approval {
        draw_approval(f, area, &pending.request, pending.choice, theme);
    }
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

    fn automations() -> Vec<AutomationSummary> {
        vec![
            AutomationSummary {
                id: "therock-update".into(),
                summary: "Emit scheduled TheRock update reminders.".into(),
                enabled: false,
                mode: "observe".into(),
            },
            AutomationSummary {
                id: "server-recover".into(),
                summary: "Restart failed managed services.".into(),
                enabled: true,
                mode: "propose".into(),
            },
        ]
    }

    #[test]
    fn refresh_is_read_only_and_spawns_without_approval() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut am, &automations(), &mut jobs, key(KeyCode::Char('l')));
        assert_eq!(fx.len(), 1);
        assert!(am.as_ref().unwrap().approval.is_none());
        assert_eq!(
            am.as_ref().unwrap().active_job.as_deref(),
            Some("automations-list")
        );
    }

    #[test]
    fn toggle_disabled_watcher_enables_via_gate() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        let ws = automations(); // row 0 is disabled → Enter should ENABLE
        let fx = on_key(&mut am, &ws, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        let pending = am.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, AutomationAction::Enable);
        assert_eq!(
            pending.args,
            vec!["automations", "enable", "therock-update"]
        );
        let fx = on_key(&mut am, &ws, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(
            am.as_ref().unwrap().active_job.as_deref(),
            Some("automations-enable")
        );
    }

    #[test]
    fn toggle_enabled_watcher_disables() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        let ws = automations();
        on_key(&mut am, &ws, &mut jobs, key(KeyCode::Down)); // row 1 is enabled
        on_key(&mut am, &ws, &mut jobs, key(KeyCode::Char(' ')));
        let pending = am.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.action, AutomationAction::Disable);
        assert_eq!(
            pending.args,
            vec!["automations", "disable", "server-recover"]
        );
    }

    #[test]
    fn toggle_with_no_watchers_surfaces_message() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        let fx = on_key(&mut am, &[], &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(am.as_ref().unwrap().approval.is_none());
        assert!(
            am.as_ref()
                .unwrap()
                .message
                .as_deref()
                .unwrap_or("")
                .contains("no automation selected")
        );
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        let ws = automations();
        on_key(&mut am, &ws, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut am, &ws, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(am.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn navigation_clamps() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        let ws = automations();
        for _ in 0..10 {
            on_key(&mut am, &ws, &mut jobs, key(KeyCode::Down));
        }
        assert_eq!(am.as_ref().unwrap().selected, ws.len() - 1);
        for _ in 0..10 {
            on_key(&mut am, &ws, &mut jobs, key(KeyCode::Up));
        }
        assert_eq!(am.as_ref().unwrap().selected, 0);
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut am = Some(AutomationsManagerState::default());
        let mut jobs = State::default();
        on_key(&mut am, &[], &mut jobs, key(KeyCode::Char('l')));
        on_key(&mut am, &[], &mut jobs, key(KeyCode::Char('q')));
        assert!(am.is_none());
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        let mut jobs = State::default();
        let mut a1 = Some(AutomationsManagerState::default());
        on_key(&mut a1, &[], &mut jobs, key(KeyCode::Char('l')));
        assert_eq!(
            a1.as_ref().unwrap().active_job.as_deref(),
            Some("automations-list")
        );
        let mut a2 = Some(AutomationsManagerState::default());
        let fx = on_key(&mut a2, &[], &mut jobs, key(KeyCode::Char('l')));
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = a2.as_ref().unwrap();
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
    fn relaunch_toggle_while_running_surfaces_message_not_stale_console() {
        // The mutating enable/disable job has the same no-op guard as refresh.
        let mut jobs = State::default();
        let ws = automations();
        let mut a1 = Some(AutomationsManagerState::default()); // row 0 disabled → enable
        on_key(&mut a1, &ws, &mut jobs, key(KeyCode::Enter)); // stage
        on_key(&mut a1, &ws, &mut jobs, key(KeyCode::Char('y'))); // spawn enable
        assert_eq!(
            a1.as_ref().unwrap().active_job.as_deref(),
            Some("automations-enable")
        );
        // Fresh overlay, same enable while the prior job still runs.
        let mut a2 = Some(AutomationsManagerState::default());
        on_key(&mut a2, &ws, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut a2, &ws, &mut jobs, key(KeyCode::Char('y')));
        assert!(fx.is_empty(), "no double-spawn for a running toggle id");
        let s = a2.as_ref().unwrap();
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
    fn snapshot_lists_watchers_with_badges() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 20);
        let mut term = Terminal::new(backend).unwrap();
        let a = AutomationsManagerState::default();
        let ws = automations();
        let jobs = State::default();
        term.draw(|f| draw_automations_manager(f, f.area(), &a, &ws, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Automations"));
        assert!(out.contains("therock-update"));
        assert!(out.contains("server-recover"));
        // The enabled/disabled badges actually render (per the function name).
        assert!(out.contains("on"), "enabled watcher shows an on badge");
        assert!(out.contains("off"), "disabled watcher shows an off badge");
    }
}
