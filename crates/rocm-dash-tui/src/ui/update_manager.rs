// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Update overlay (Phase 3 Wave 2).
//!
//! Wraps `rocm update`: a read-only check (and a `--apply --dry-run` preview)
//! that run straight through the job-bridge, plus the mutating `--apply` /
//! `--apply --activate` actions that route through the approval gate first.
//! This is the report-with-gated-apply archetype.

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

/// The update menu actions, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateAction {
    /// `rocm update` — check only (read-only).
    Check,
    /// `rocm update --apply --dry-run` — show the plan (read-only).
    Preview,
    /// `rocm update --apply` — install (mutating → approval).
    Apply,
    /// `rocm update --apply --activate` — install + use as default (mutating).
    ApplyActivate,
}

/// Menu order; `state.selected` indexes this.
pub const ACTIONS: &[UpdateAction] = &[
    UpdateAction::Check,
    UpdateAction::Preview,
    UpdateAction::Apply,
    UpdateAction::ApplyActivate,
];

impl UpdateAction {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Check => "Check for updates",
            Self::Preview => "Preview the update (dry-run)",
            Self::Apply => "Apply the update",
            Self::ApplyActivate => "Apply and use as default",
        }
    }

    /// `rocm` argv (after the binary) for this action.
    fn args(self) -> Vec<String> {
        match self {
            Self::Check => vec!["update".into()],
            Self::Preview => vec!["update".into(), "--apply".into(), "--dry-run".into()],
            Self::Apply => vec!["update".into(), "--apply".into()],
            Self::ApplyActivate => {
                vec!["update".into(), "--apply".into(), "--activate".into()]
            }
        }
    }

    /// Whether this action changes the system (and so needs approval).
    const fn is_mutating(self) -> bool {
        matches!(self, Self::Apply | Self::ApplyActivate)
    }

    /// Stable job id (one console per action kind).
    fn job_id(self) -> String {
        match self {
            Self::Check => "update-check",
            Self::Preview => "update-preview",
            Self::Apply => "update-apply",
            Self::ApplyActivate => "update-apply-activate",
        }
        .to_string()
    }
}

/// An approved-but-not-yet-run mutating update op.
#[derive(Debug, Clone)]
pub struct PendingUpdate {
    pub action: UpdateAction,
    pub cmd: String,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct UpdateManagerState {
    pub selected: usize,
    pub approval: Option<PendingUpdate>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Handle a key while the overlay is open.
pub fn on_key(
    update: &mut Option<UpdateManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(u) = update.as_mut() else {
        return Vec::new();
    };

    // 1) Approval modal has focus.
    if let Some(pending) = u.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = u.approval.take() {
                    return spawn_update(u, jobs, pending.action, pending.cmd);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => u.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 2) A job is showing in the console.
    if let Some(job_id) = u.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *update = None,
            ConsoleOutcome::Dismissed => {
                u.active_job = None;
                u.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3) Menu navigation + action.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *update = None,
        KeyCode::Up | KeyCode::Char('k') => u.selected = u.selected.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            u.selected = (u.selected + 1).min(ACTIONS.len() - 1);
        }
        KeyCode::Enter => return activate_selected(u, jobs),
        _ => {}
    }
    Vec::new()
}

/// Run the selected action: read-only ones spawn immediately; mutating ones
/// stage an approval first.
fn activate_selected(u: &mut UpdateManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let action = ACTIONS[u.selected.min(ACTIONS.len() - 1)];
    let cmd = resolve_exe();
    if action.is_mutating() {
        let request = ApprovalRequest::new(
            action.label().to_string(),
            vec![
                format!("{} {}", exe_label(&cmd), action.args().join(" ")),
                String::new(),
                "This installs a ROCm package update on this machine.".to_string(),
            ],
        );
        u.message = None;
        u.approval = Some(PendingUpdate {
            action,
            cmd,
            request,
            choice: ApprovalChoice::default(),
        });
        Vec::new()
    } else {
        spawn_update(u, jobs, action, cmd)
    }
}

/// Spawn the update job for `action`.
fn spawn_update(
    u: &mut UpdateManagerState,
    jobs: &mut State,
    action: UpdateAction,
    cmd: String,
) -> Vec<SideEffect> {
    let id = action.job_id();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args: action.args(),
    });
    if fx.is_empty() {
        u.message = Some(format!("“{}” is already running", action.label()));
        return fx;
    }
    u.active_job = Some(id);
    fx
}

/// Render the overlay (menu, or the approval modal, or the job console).
pub fn draw_update_manager(
    f: &mut Frame,
    area: Rect,
    u: &UpdateManagerState,
    _jobs: &State,
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Update — ROCm packages"),
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

    let items: Vec<ListItem> = ACTIONS
        .iter()
        .map(|a| {
            let tag = if a.is_mutating() {
                " (needs approval)"
            } else {
                ""
            };
            ListItem::new(Line::from(vec![
                Span::styled(a.label().to_string(), Style::default().fg(theme.fg)),
                Span::styled(tag, Style::default().fg(theme.warn)),
            ]))
        })
        .collect();
    let mut ls = ListState::default();
    ls.select(Some(u.selected.min(ACTIONS.len() - 1)));
    let list = List::new(items).highlight_style(
        Style::default()
            .bg(theme.surface_2)
            .add_modifier(Modifier::BOLD),
    );
    let list_area = panel::vertical_scrollbar(
        f,
        rows[0],
        ACTIONS.len(),
        rows[0].height as usize,
        u.selected,
        theme,
    );
    f.render_stateful_widget(list, list_area, &mut ls);

    let msg = u.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[1],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ select · Enter run · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    if let Some(pending) = &u.approval {
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

    #[test]
    fn check_is_read_only_and_spawns_without_approval() {
        let mut u = Some(UpdateManagerState::default()); // selected 0 = Check
        let mut jobs = State::default();
        let fx = on_key(&mut u, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1);
        assert!(
            u.as_ref().unwrap().approval.is_none(),
            "no gate for read-only"
        );
        assert_eq!(
            u.as_ref().unwrap().active_job.as_deref(),
            Some("update-check")
        );
    }

    #[test]
    fn preview_is_read_only() {
        let mut u = Some(UpdateManagerState::default());
        let mut jobs = State::default();
        u.as_mut().unwrap().selected = ACTIONS
            .iter()
            .position(|a| *a == UpdateAction::Preview)
            .unwrap();
        let fx = on_key(&mut u, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1);
        assert!(u.as_ref().unwrap().approval.is_none());
        assert_eq!(
            u.as_ref().unwrap().active_job.as_deref(),
            Some("update-preview")
        );
    }

    #[test]
    fn apply_is_gated_then_spawns() {
        let mut u = Some(UpdateManagerState::default());
        let mut jobs = State::default();
        u.as_mut().unwrap().selected = ACTIONS
            .iter()
            .position(|a| *a == UpdateAction::Apply)
            .unwrap();
        // Enter stages approval, NO job.
        let fx = on_key(&mut u, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(u.as_ref().unwrap().approval.is_some());
        assert!(jobs.jobs.is_empty());
        // Approve → spawns.
        let fx = on_key(&mut u, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(
            u.as_ref().unwrap().active_job.as_deref(),
            Some("update-apply")
        );
    }

    #[test]
    fn apply_args_carry_apply_and_activate() {
        assert_eq!(UpdateAction::Apply.args(), vec!["update", "--apply"]);
        assert_eq!(
            UpdateAction::ApplyActivate.args(),
            vec!["update", "--apply", "--activate"]
        );
        assert_eq!(
            UpdateAction::Preview.args(),
            vec!["update", "--apply", "--dry-run"]
        );
        assert_eq!(UpdateAction::Check.args(), vec!["update"]);
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut u = Some(UpdateManagerState::default());
        let mut jobs = State::default();
        u.as_mut().unwrap().selected = ACTIONS
            .iter()
            .position(|a| *a == UpdateAction::Apply)
            .unwrap();
        on_key(&mut u, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut u, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(u.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn navigation_clamps() {
        let mut u = Some(UpdateManagerState::default());
        let mut jobs = State::default();
        for _ in 0..10 {
            on_key(&mut u, &mut jobs, key(KeyCode::Down));
        }
        assert_eq!(u.as_ref().unwrap().selected, ACTIONS.len() - 1);
        for _ in 0..10 {
            on_key(&mut u, &mut jobs, key(KeyCode::Up));
        }
        assert_eq!(u.as_ref().unwrap().selected, 0);
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut u = Some(UpdateManagerState::default());
        let mut jobs = State::default();
        on_key(&mut u, &mut jobs, key(KeyCode::Enter)); // Check spawns
        on_key(&mut u, &mut jobs, key(KeyCode::Char('q')));
        assert!(u.is_none());
    }

    #[test]
    fn snapshot_lists_actions_and_gated_tags() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(90, 18);
        let mut term = Terminal::new(backend).unwrap();
        let u = UpdateManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_update_manager(f, f.area(), &u, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Update"));
        assert!(out.contains("Check for updates"));
        assert!(out.contains("needs approval"));
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        // Mirrors engine_manager: a StartJob for a still-running id no-ops, so
        // spawn_update must surface a message and NOT point active_job at it.
        let mut jobs = State::default();
        let mut u1 = Some(UpdateManagerState::default()); // Check
        on_key(&mut u1, &mut jobs, key(KeyCode::Enter));
        assert_eq!(
            u1.as_ref().unwrap().active_job.as_deref(),
            Some("update-check")
        );
        // Fresh overlay, same read-only Check while the prior one still runs.
        let mut u2 = Some(UpdateManagerState::default());
        let fx = on_key(&mut u2, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = u2.as_ref().unwrap();
        assert!(s.active_job.is_none(), "must not point at the stale job");
        assert!(
            s.message
                .as_deref()
                .unwrap_or("")
                .contains("already running")
        );
        assert_eq!(jobs.jobs.len(), 1);
    }
}
