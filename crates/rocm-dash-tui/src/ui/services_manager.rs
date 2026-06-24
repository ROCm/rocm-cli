// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Services manager overlay (Phase 3 Wave 1).
//!
//! The first operational screen rebuilt on the Wave-0 primitives: it lists the
//! managed inference services the daemon surfaces (model · port · status · live
//! `gen_tps`) and runs stop/restart **through the approval gate and the
//! job-bridge** — never inline. This is the pattern the remaining Wave-1 screens
//! (serve_wizard / model_picker / engine_manager) reuse.
//!
//! Mutating actions invoke the CLI (`rocm services stop|restart <id> --yes`) as
//! a background job; the approval *decision* is the user's, captured by the
//! render+event seam, and the CLI owns the actual mutation — the read-only chat
//! invariant is untouched.

use std::collections::HashMap;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

use rocm_dash_core::metrics::Instance;
use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::approval::{
    ApprovalChoice, ApprovalRequest, ApprovalVerdict, approval_key, draw_approval,
};
use crate::ui::exec::{exe_label, resolve_exe};
use crate::ui::format;
use crate::ui::job_console::{ConsoleOutcome, draw_job_console, on_console_key};
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// A lifecycle operation on a managed service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleAction {
    Stop,
    Restart,
}

impl LifecycleAction {
    pub const fn verb(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Restart => "restart",
        }
    }
}

/// An approval awaiting the user's verdict before a lifecycle op runs.
#[derive(Debug, Clone)]
pub struct PendingLifecycle {
    pub action: LifecycleAction,
    pub service_id: String,
    /// The `rocm` binary to invoke, resolved when the approval was staged so a
    /// later `current_exe()` failure can't silently drop an approved action.
    pub cmd: String,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct ServicesManagerState {
    pub selected: usize,
    /// Set while an approval modal is up (mutating op gated).
    pub approval: Option<PendingLifecycle>,
    /// The job id of an in-flight (or just-finished) lifecycle op.
    pub active_job: Option<String>,
}

/// A render-ready row derived from an `Instance`.
#[derive(Debug, Clone, PartialEq)]
pub struct ServiceRow {
    pub id: String,
    pub model: String,
    pub port: Option<u16>,
    pub status: String,
    pub gen_tps: Option<f64>,
}

/// Build the sorted service list from the daemon-surfaced instances.
pub fn service_rows<S: ::std::hash::BuildHasher>(
    instances: &HashMap<String, Instance, S>,
) -> Vec<ServiceRow> {
    let mut rows: Vec<ServiceRow> = instances
        .values()
        .map(|i| ServiceRow {
            id: i.container_id.clone(),
            model: i.model_name.clone(),
            port: i.port,
            status: format!("{:?}", i.status),
            gen_tps: i.gen_tps,
        })
        .collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    rows
}

/// Handle a key while the overlay is open.
///
/// Mutates the overlay + the job model in place and returns the reducer side effects (e.g. `SpawnJob`) for the
/// event loop to run through the job-bridge. Returns `true` (via the closed
/// state) — the caller checks `state.services` for closure.
pub fn on_key<S: ::std::hash::BuildHasher>(
    services: &mut Option<ServicesManagerState>,
    jobs: &mut State,
    instances: &HashMap<String, Instance, S>,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let rows = service_rows(instances);
    let Some(sm) = services.as_mut() else {
        return Vec::new();
    };

    // 1) Approval modal has focus.
    if let Some(pending) = sm.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            // `take()` here always yields `Some` (we are inside the guard), but
            // pattern-match rather than `expect()` so there is no panic path.
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = sm.approval.take() {
                    return spawn_lifecycle(sm, jobs, pending);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => {
                sm.approval = None;
            }
            None => {}
        }
        return Vec::new();
    }

    // 2) A lifecycle job is showing in the console.
    if let Some(job_id) = sm.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *services = None,
            ConsoleOutcome::Dismissed => sm.active_job = None,
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3) List navigation + lifecycle requests.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *services = None,
        KeyCode::Up | KeyCode::Char('k') => {
            sm.selected = sm.selected.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') if !rows.is_empty() => {
            sm.selected = (sm.selected + 1).min(rows.len() - 1);
        }
        KeyCode::Char('s') => request_lifecycle(sm, &rows, LifecycleAction::Stop),
        KeyCode::Char('r') => request_lifecycle(sm, &rows, LifecycleAction::Restart),
        _ => {}
    }
    Vec::new()
}

/// Stage an approval for the selected service.
fn request_lifecycle(sm: &mut ServicesManagerState, rows: &[ServiceRow], action: LifecycleAction) {
    let Some(row) = rows.get(sm.selected) else {
        return;
    };
    let cmd = resolve_exe();
    let request = ApprovalRequest::new(
        format!("{} service “{}”", action.verb(), row.id),
        vec![
            format!(
                "{} services {} {} --yes",
                exe_label(&cmd),
                action.verb(),
                row.id
            ),
            format!("model: {}   port: {}", row.model, port_str(row.port)),
            String::new(),
            format!(
                "This runs the CLI `services {}` for this managed service.",
                action.verb()
            ),
        ],
    );
    sm.approval = Some(PendingLifecycle {
        action,
        service_id: row.id.clone(),
        cmd,
        request,
        choice: ApprovalChoice::default(),
    });
}

/// Launch the approved lifecycle op as a background job. Returns the effects.
/// The command was resolved when the approval was staged, so this never
/// silently drops an approved action.
fn spawn_lifecycle(
    sm: &mut ServicesManagerState,
    jobs: &mut State,
    pending: PendingLifecycle,
) -> Vec<SideEffect> {
    let id = format!("svc-{}-{}", pending.action.verb(), pending.service_id);
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: pending.cmd,
        args: vec![
            "services".into(),
            pending.action.verb().into(),
            pending.service_id,
            "--yes".into(),
        ],
    });
    sm.active_job = Some(id);
    fx
}

fn port_str(port: Option<u16>) -> String {
    port.map_or_else(|| "—".to_string(), |p| p.to_string())
}

/// Render the overlay (list, or the approval modal, or the job console).
pub fn draw_services_manager<S: ::std::hash::BuildHasher>(
    f: &mut Frame,
    area: Rect,
    sm: &ServicesManagerState,
    instances: &HashMap<String, Instance, S>,
    jobs: &State,
    theme: &Theme,
) {
    // The job console takes over while a lifecycle op is in flight / finished.
    if let Some(job_id) = &sm.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(82, 80, 130, 34, area);
    let inner = draw_popup_frame(f, popup, "Services — managed inference servers", theme);
    if inner.height == 0 {
        return;
    }

    let rows = service_rows(instances);
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    if rows.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No managed services. Start one with `rocm serve <model> --managed`.",
                Style::default().fg(theme.muted),
            ))),
            body[0],
        );
    } else {
        let items: Vec<ListItem> = rows
            .iter()
            .map(|r| {
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<22}", trunc(&r.id, 22)),
                        Style::default().fg(theme.fg),
                    ),
                    Span::styled(
                        format!("{:<26}", trunc(&r.model, 26)),
                        Style::default().fg(theme.accent),
                    ),
                    Span::styled(
                        format!(":{:<6}", port_str(r.port)),
                        Style::default().fg(theme.muted),
                    ),
                    Span::styled(format!("{:<10}", r.status), Style::default().fg(theme.ok)),
                    Span::styled(
                        format!("gen {}", format::tps_opt(r.gen_tps)),
                        Style::default().fg(theme.fg),
                    ),
                ]))
            })
            .collect();
        let mut ls = ListState::default();
        ls.select(Some(sm.selected.min(rows.len().saturating_sub(1))));
        let list = List::new(items).highlight_style(
            Style::default()
                .bg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        );
        f.render_stateful_widget(list, body[0], &mut ls);
    }

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ select · s stop · r restart · Esc close",
            Style::default().fg(theme.muted),
        ))),
        body[1],
    );

    // Approval modal sits on top of the list.
    if let Some(pending) = &sm.approval {
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
    use rocm_dash_core::metrics::{Instance, InstanceStatus};

    fn inst(id: &str, model: &str, tps: Option<f64>) -> Instance {
        Instance {
            container_id: id.into(),
            container_name: id.into(),
            model_name: model.into(),
            status: InstanceStatus::Running,
            port: Some(8000),
            gen_tps: tps,
            ..Instance::default()
        }
    }

    fn instances() -> HashMap<String, Instance> {
        let mut m = HashMap::new();
        m.insert("a".into(), inst("svc-a", "llama3", Some(42.0)));
        m.insert("b".into(), inst("svc-b", "qwen", None));
        m
    }

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn rows_are_sorted_by_id() {
        let rows = service_rows(&instances());
        assert_eq!(
            rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["svc-a", "svc-b"]
        );
        assert_eq!(rows[0].gen_tps, Some(42.0));
    }

    #[test]
    fn stop_requires_approval_then_spawns_job() {
        let mut services = Some(ServicesManagerState::default());
        let mut jobs = State::default();
        let insts = instances();

        // 's' on the first row stages an approval — NO job yet (gated).
        let fx = on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('s')));
        assert!(fx.is_empty(), "stop must not run before approval");
        let sm = services.as_ref().unwrap();
        assert!(sm.approval.is_some());
        assert_eq!(sm.approval.as_ref().unwrap().action, LifecycleAction::Stop);
        assert!(jobs.jobs.is_empty(), "no job spawned pre-approval");

        // Approve ('y') → a SpawnJob effect is produced + a job registered.
        let fx = on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1, "approve spawns exactly one job");
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        let sm = services.as_ref().unwrap();
        assert!(sm.approval.is_none());
        assert_eq!(sm.active_job.as_deref(), Some("svc-stop-svc-a"));
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        // Approve a stop so a job is active, then ensure 'q' is never trapped.
        let mut services = Some(ServicesManagerState::default());
        let mut jobs = State::default();
        let insts = instances();
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('s')));
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('y')));
        assert!(services.as_ref().unwrap().active_job.is_some());
        // Job is still Running (terminal events would arrive via the bridge).
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('q')));
        assert!(services.is_none(), "q must close the overlay even mid-job");
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut services = Some(ServicesManagerState::default());
        let mut jobs = State::default();
        let insts = instances();
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('s')));
        let fx = on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(services.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty(), "deny spawns nothing");
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut services = Some(ServicesManagerState::default());
        let mut jobs = State::default();
        let insts = instances();
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Esc));
        assert!(services.is_none());
    }

    #[test]
    fn navigation_clamps_to_rows() {
        let mut services = Some(ServicesManagerState::default());
        let mut jobs = State::default();
        let insts = instances();
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Down));
        assert_eq!(services.as_ref().unwrap().selected, 1);
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Down)); // clamp at last
        assert_eq!(services.as_ref().unwrap().selected, 1);
    }

    fn render(
        sm: &ServicesManagerState,
        jobs: &State,
        insts: &HashMap<String, Instance>,
    ) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 28);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_services_manager(f, f.area(), sm, insts, jobs, &theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        buf.content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn snapshot_lists_services_with_gen_tps() {
        let sm = ServicesManagerState::default();
        let out = render(&sm, &State::default(), &instances());
        assert!(out.contains("Services"), "titled overlay");
        assert!(out.contains("svc-a"), "service id listed");
        assert!(out.contains("llama3"), "model listed");
        assert!(out.contains("s stop"), "lifecycle hints");
    }

    #[test]
    fn snapshot_shows_approval_modal_on_stop() {
        let mut services = Some(ServicesManagerState::default());
        let mut jobs = State::default();
        let insts = instances();
        on_key(&mut services, &mut jobs, &insts, key(KeyCode::Char('s')));
        let out = render(services.as_ref().unwrap(), &jobs, &insts);
        assert!(out.contains("Review:"), "approval modal shown");
        assert!(out.contains("stop service"), "describes the gated action");
        assert!(out.contains("Approve"), "approve button present");
    }
}
