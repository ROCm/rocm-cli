// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine manager overlay (Phase 3 Wave 1).
//!
//! Lists the serving engines ROCm CLI knows about and runs use/install/reinstall
//! **through the approval gate and the job-bridge** — never inline. Installs
//! stream their output into the shared job console (the Wave-0 `running_job`
//! replacement). This is the third operational screen on the proven primitives;
//! it feeds the serve wizard (which engine to launch a model with).
//!
//! The approval *decision* is the user's, captured by the render+event seam; the
//! CLI owns the actual mutation (`rocm config set-default-engine` /
//! `rocm engines install [--reinstall]`) — the read-only chat invariant stands.

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

/// Engine catalog — names + descriptions mirror `apps/rocm` `engine_inventory()`.
/// Kept TUI-local (a stable, small list) so this layer needs no `rocm-core` dep.
pub const ENGINE_CATALOG: &[(&str, &str)] = &[
    (
        "lemonade",
        "default embedded Lemonade server with ROCm llama.cpp backend",
    ),
    (
        "vllm",
        "Linux/WSL ROCm GPU serving engine through external vLLM",
    ),
];

/// A lifecycle operation on an engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineAction {
    /// Set this engine as the default (`config set-default-engine`).
    Use,
    /// Install it into the managed engine folder (`engines install`).
    Install,
    /// Reinstall even if present (`engines install --reinstall`).
    Reinstall,
}

impl EngineAction {
    pub const fn verb(self) -> &'static str {
        match self {
            Self::Use => "use",
            Self::Install => "install",
            Self::Reinstall => "reinstall",
        }
    }

    /// The `rocm` argv (after the binary) for this action on `engine`.
    fn args(self, engine: &str) -> Vec<String> {
        match self {
            Self::Use => vec![
                "config".into(),
                "set-default-engine".into(),
                engine.to_string(),
            ],
            Self::Install => vec!["engines".into(), "install".into(), engine.to_string()],
            Self::Reinstall => vec![
                "engines".into(),
                "install".into(),
                engine.to_string(),
                "--reinstall".into(),
            ],
        }
    }
}

/// An approved-but-not-yet-run engine op.
#[derive(Debug, Clone)]
pub struct PendingEngineOp {
    pub action: EngineAction,
    pub engine: String,
    /// Resolved `rocm` binary (captured at approval time).
    pub cmd: String,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct EngineManagerState {
    pub selected: usize,
    pub approval: Option<PendingEngineOp>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Handle a key while the overlay is open. Mirrors the services-manager seam.
pub fn on_key(
    engines: &mut Option<EngineManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(em) = engines.as_mut() else {
        return Vec::new();
    };

    // 1) Approval modal has focus.
    if let Some(pending) = em.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = em.approval.take() {
                    return spawn_engine_op(em, jobs, pending);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => em.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 2) A job is showing in the console.
    if let Some(job_id) = em.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *engines = None,
            ConsoleOutcome::Dismissed => {
                em.active_job = None;
                // Clear any stale "already running" notice on return to the list.
                em.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3) List navigation + action requests.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *engines = None,
        KeyCode::Up | KeyCode::Char('k') => em.selected = em.selected.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            em.selected = (em.selected + 1).min(ENGINE_CATALOG.len() - 1);
        }
        KeyCode::Char('u') => request_op(em, EngineAction::Use),
        KeyCode::Char('i') => request_op(em, EngineAction::Install),
        KeyCode::Char('r') => request_op(em, EngineAction::Reinstall),
        _ => {}
    }
    Vec::new()
}

/// Stage an approval for the selected engine + action.
fn request_op(em: &mut EngineManagerState, action: EngineAction) {
    let Some((name, _desc)) = ENGINE_CATALOG.get(em.selected.min(ENGINE_CATALOG.len() - 1)) else {
        return;
    };
    let cmd = resolve_exe();
    let args = action.args(name);
    let explain = match action {
        EngineAction::Use => "Sets the default serving engine for ROCm CLI.",
        EngineAction::Install => "Installs the engine into the managed engine folder (streams).",
        EngineAction::Reinstall => "Reinstalls the engine even if it already exists (streams).",
    };
    let request = ApprovalRequest::new(
        format!("{} engine “{}”", action.verb(), name),
        vec![
            format!("{} {}", exe_label(&cmd), args.join(" ")),
            String::new(),
            explain.to_string(),
        ],
    );
    em.message = None;
    em.approval = Some(PendingEngineOp {
        action,
        engine: (*name).to_string(),
        cmd,
        request,
        choice: ApprovalChoice::default(),
    });
}

/// Launch the approved engine op as a background job.
fn spawn_engine_op(
    em: &mut EngineManagerState,
    jobs: &mut State,
    pending: PendingEngineOp,
) -> Vec<SideEffect> {
    // Sanitize the engine name for the job id (any non-alphanumeric char → `-`).
    let key: String = pending
        .engine
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let id = format!("engine-{}-{key}", pending.action.verb());
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: pending.cmd,
        args: pending.action.args(&pending.engine),
    });
    if fx.is_empty() {
        em.message = Some(format!(
            "a {} job for “{}” is already running",
            pending.action.verb(),
            pending.engine
        ));
        return fx;
    }
    em.active_job = Some(id);
    fx
}

/// Render the overlay (list, or the approval modal, or the job console).
pub fn draw_engine_manager(
    f: &mut Frame,
    area: Rect,
    em: &EngineManagerState,
    _jobs: &State,
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Engines — serving backends"),
        BoxRole::Primary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let items: Vec<ListItem> = ENGINE_CATALOG
        .iter()
        .map(|(name, desc)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{name:<12}"), Style::default().fg(theme.accent)),
                Span::styled((*desc).to_string(), Style::default().fg(theme.muted)),
            ]))
        })
        .collect();
    let mut ls = ListState::default();
    ls.select(Some(em.selected.min(ENGINE_CATALOG.len() - 1)));
    let list = List::new(items).highlight_style(
        Style::default()
            .bg(theme.surface_2)
            .add_modifier(Modifier::BOLD),
    );
    let list_area = panel::vertical_scrollbar(
        f,
        body[0],
        ENGINE_CATALOG.len(),
        body[0].height as usize,
        em.selected,
        theme,
    );
    f.render_stateful_widget(list, list_area, &mut ls);

    let msg = em.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        body[1],
    );

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ select · u use · i install · r reinstall · Esc close",
            Style::default().fg(theme.muted),
        ))),
        body[2],
    );

    if let Some(pending) = &em.approval {
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
    fn use_action_builds_config_set_default_engine() {
        assert_eq!(
            EngineAction::Use.args("lemonade"),
            vec!["config", "set-default-engine", "lemonade"]
        );
    }

    #[test]
    fn install_and_reinstall_args() {
        assert_eq!(
            EngineAction::Install.args("vllm"),
            vec!["engines", "install", "vllm"]
        );
        assert_eq!(
            EngineAction::Reinstall.args("vllm"),
            vec!["engines", "install", "vllm", "--reinstall"]
        );
    }

    #[test]
    fn navigation_clamps_to_catalog() {
        let mut em = Some(EngineManagerState::default());
        let mut jobs = State::default();
        for _ in 0..ENGINE_CATALOG.len() + 3 {
            on_key(&mut em, &mut jobs, key(KeyCode::Down));
        }
        assert_eq!(em.as_ref().unwrap().selected, ENGINE_CATALOG.len() - 1);
        for _ in 0..ENGINE_CATALOG.len() + 3 {
            on_key(&mut em, &mut jobs, key(KeyCode::Up));
        }
        assert_eq!(em.as_ref().unwrap().selected, 0);
    }

    #[test]
    fn install_requires_approval_then_spawns_job() {
        let mut em = Some(EngineManagerState::default());
        let mut jobs = State::default();
        // 'i' on the first engine (lemonade) stages approval — NO job yet.
        let fx = on_key(&mut em, &mut jobs, key(KeyCode::Char('i')));
        assert!(fx.is_empty(), "install must not run before approval");
        assert!(em.as_ref().unwrap().approval.is_some());
        assert!(jobs.jobs.is_empty());

        // Approve → one SpawnJob, job registered, console active.
        let fx = on_key(&mut em, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        let s = em.as_ref().unwrap();
        assert!(s.approval.is_none());
        assert_eq!(s.active_job.as_deref(), Some("engine-install-lemonade"));
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn relaunch_while_prior_job_running_surfaces_message() {
        // The reducer no-ops a StartJob for a still-running id. spawn_engine_op
        // must NOT set active_job; it surfaces a message instead.
        let mut jobs = State::default();
        let mut em1 = Some(EngineManagerState::default());
        on_key(&mut em1, &mut jobs, key(KeyCode::Char('i'))); // lemonade install
        on_key(&mut em1, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(
            em1.as_ref().unwrap().active_job.as_deref(),
            Some("engine-install-lemonade")
        );

        // Fresh overlay, same engine+action while the prior job still runs.
        let mut em2 = Some(EngineManagerState::default());
        on_key(&mut em2, &mut jobs, key(KeyCode::Char('i')));
        let fx = on_key(&mut em2, &mut jobs, key(KeyCode::Char('y')));
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = em2.as_ref().unwrap();
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
    fn deny_cancels_without_spawning() {
        let mut em = Some(EngineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut em, &mut jobs, key(KeyCode::Char('u')));
        let fx = on_key(&mut em, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(em.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut em = Some(EngineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut em, &mut jobs, key(KeyCode::Esc));
        assert!(em.is_none());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut em = Some(EngineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut em, &mut jobs, key(KeyCode::Char('i')));
        on_key(&mut em, &mut jobs, key(KeyCode::Char('y')));
        assert!(em.as_ref().unwrap().active_job.is_some());
        on_key(&mut em, &mut jobs, key(KeyCode::Char('q')));
        assert!(em.is_none(), "q must close the overlay even mid-job");
    }

    fn render(em: &EngineManagerState, jobs: &State) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(124, 28);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_engine_manager(f, f.area(), em, jobs, &theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        buf.content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn snapshot_lists_engines() {
        let em = EngineManagerState::default();
        let out = render(&em, &State::default());
        assert!(out.contains("Engines"), "titled overlay");
        assert!(out.contains("lemonade"), "engine listed");
        assert!(out.contains("vllm"), "engine listed");
        assert!(out.contains("i install"), "action hints");
    }

    #[test]
    fn snapshot_shows_approval_modal_on_use() {
        let mut em = Some(EngineManagerState::default());
        let mut jobs = State::default();
        on_key(&mut em, &mut jobs, key(KeyCode::Char('u')));
        let out = render(em.as_ref().unwrap(), &jobs);
        assert!(out.contains("Review:"), "approval modal shown");
        assert!(out.contains("use engine"), "describes the gated action");
        assert!(out.contains("Approve"), "approve button present");
    }
}
