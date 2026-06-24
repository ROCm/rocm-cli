// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Config + provider manager overlay (Phase 3 Wave 3).
//!
//! Folds the frozen `config_manager` and `provider_manager` into one screen:
//!
//! - **Show config** (`rocm config show`) — read-only, straight to the job-bridge.
//! - **Enable / Disable** the selected assistant provider — mutating, routed
//!   through the approval gate (`rocm config enable-provider|disable-provider`).
//!
//! **API keys are NEVER entered or stored here.** Per the chat invariant + D6,
//! the AMD LLM gateway key (and any provider key the chat surface uses) is
//! sourced from the environment ONLY — this overlay deliberately offers no
//! key-entry field and says so. Other config edits (default engine/runtime,
//! telemetry mode) are documented fast-follows. Zero `std::thread::spawn`/`try_recv`.

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
use crate::ui::job_console::{ConsoleOutcome, draw_job_console, on_console_key};
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// Assistant providers (mirrors the bin's `Provider` value-enum). Fixed set, so
/// no plumbing is needed — these are not runtime-discovered data.
pub const PROVIDERS: &[&str] = &["local", "anthropic", "openai"];

/// Menu actions, in display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigAction {
    ShowConfig,
    EnableProvider,
    DisableProvider,
}

pub const ACTIONS: &[ConfigAction] = &[
    ConfigAction::ShowConfig,
    ConfigAction::EnableProvider,
    ConfigAction::DisableProvider,
];

impl ConfigAction {
    const fn label(self) -> &'static str {
        match self {
            Self::ShowConfig => "Show saved config",
            Self::EnableProvider => "Enable provider",
            Self::DisableProvider => "Disable provider",
        }
    }

    const fn is_mutating(self) -> bool {
        !matches!(self, Self::ShowConfig)
    }
}

/// An approved-but-not-yet-run mutating config op.
#[derive(Debug, Clone)]
pub struct PendingConfig {
    pub cmd: String,
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the overlay is closed.
#[derive(Debug, Clone, Default)]
pub struct ConfigManagerState {
    pub action_sel: usize,
    pub provider_sel: usize,
    pub approval: Option<PendingConfig>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Handle a key while the overlay is open.
pub fn on_key(
    cm: &mut Option<ConfigManagerState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(c) = cm.as_mut() else {
        return Vec::new();
    };

    // 1) Approval modal has focus.
    if let Some(pending) = c.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = c.approval.take() {
                    return spawn_config(c, jobs, "config-provider", pending.cmd, pending.args);
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
            ConsoleOutcome::Closed => *cm = None,
            ConsoleOutcome::Dismissed => {
                c.active_job = None;
                c.message = None;
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3) Menu navigation + actions.
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => *cm = None,
        KeyCode::Up | KeyCode::Char('k') => c.action_sel = c.action_sel.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            c.action_sel = (c.action_sel + 1).min(ACTIONS.len() - 1);
        }
        KeyCode::Left | KeyCode::Char('h') => c.provider_sel = c.provider_sel.saturating_sub(1),
        KeyCode::Right | KeyCode::Char('l') => {
            c.provider_sel = (c.provider_sel + 1).min(PROVIDERS.len() - 1);
        }
        KeyCode::Enter => return activate_selected(c, jobs),
        _ => {}
    }
    Vec::new()
}

/// Run the selected action: Show spawns immediately (read-only); provider
/// enable/disable stage an approval first.
fn activate_selected(c: &mut ConfigManagerState, jobs: &mut State) -> Vec<SideEffect> {
    let action = ACTIONS[c.action_sel.min(ACTIONS.len() - 1)];
    let provider = PROVIDERS[c.provider_sel.min(PROVIDERS.len() - 1)];
    let cmd = resolve_exe();
    match action {
        ConfigAction::ShowConfig => {
            let args = vec!["config".to_string(), "show".to_string()];
            spawn_config(c, jobs, "config-show", cmd, args)
        }
        ConfigAction::EnableProvider | ConfigAction::DisableProvider => {
            let verb = if action == ConfigAction::EnableProvider {
                "enable-provider"
            } else {
                "disable-provider"
            };
            let args = vec!["config".to_string(), verb.to_string(), provider.to_string()];
            let request = ApprovalRequest::new(
                format!("{} {}", action.label().to_lowercase(), provider),
                vec![
                    format!("{} {}", exe_label(&cmd), args.join(" ")),
                    String::new(),
                    "This changes which assistant providers are enabled. No API key \
                     is entered or stored — keys are read from the environment only."
                        .to_string(),
                ],
            );
            c.message = None;
            c.approval = Some(PendingConfig {
                cmd,
                args,
                request,
                choice: ApprovalChoice::default(),
            });
            Vec::new()
        }
    }
}

/// Spawn a config job under `id`.
fn spawn_config(
    c: &mut ConfigManagerState,
    jobs: &mut State,
    id: &str,
    cmd: String,
    args: Vec<String>,
) -> Vec<SideEffect> {
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.to_string(),
        cmd,
        args,
    });
    if fx.is_empty() {
        c.message = Some("a config job is already running".to_string());
        return fx;
    }
    c.active_job = Some(id.to_string());
    fx
}

/// Render the overlay (menu, or an approval/console on top).
pub fn draw_config_manager(
    f: &mut Frame,
    area: Rect,
    c: &ConfigManagerState,
    jobs: &State,
    theme: &Theme,
) {
    if let Some(job_id) = &c.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(72, 64, 100, 18, area);
    let inner = draw_popup_frame(f, popup, "Config & providers", theme);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    // Provider selector (drives the enable/disable actions).
    let provider = PROVIDERS[c.provider_sel.min(PROVIDERS.len() - 1)];
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("provider: ", Style::default().fg(theme.muted)),
            Span::styled(
                provider.to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  (←→ change)", Style::default().fg(theme.muted)),
        ])),
        rows[0],
    );

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
    ls.select(Some(c.action_sel.min(ACTIONS.len() - 1)));
    let list = List::new(items).highlight_style(
        Style::default()
            .bg(theme.surface_2)
            .add_modifier(Modifier::BOLD),
    );
    f.render_stateful_widget(list, rows[1], &mut ls);

    let msg = c.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[2],
    );

    // The env-only key notice is always visible — this overlay never takes keys.
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "API keys are read from the environment only — never entered here. \
             ↑↓ action · Enter run · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[3],
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

    #[test]
    fn show_is_read_only_and_spawns_without_approval() {
        let mut cm = Some(ConfigManagerState::default()); // action 0 = ShowConfig
        let mut jobs = State::default();
        let fx = on_key(&mut cm, &mut jobs, key(KeyCode::Enter));
        assert_eq!(fx.len(), 1);
        assert!(cm.as_ref().unwrap().approval.is_none());
        assert_eq!(
            cm.as_ref().unwrap().active_job.as_deref(),
            Some("config-show")
        );
    }

    #[test]
    fn enable_provider_is_gated_then_spawns() {
        let mut cm = Some(ConfigManagerState::default());
        let mut jobs = State::default();
        // action 1 = EnableProvider, provider 0 = local
        cm.as_mut().unwrap().action_sel = 1;
        let fx = on_key(&mut cm, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        let pending = cm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.args, vec!["config", "enable-provider", "local"]);
        assert!(jobs.jobs.is_empty());
        let fx = on_key(&mut cm, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(
            cm.as_ref().unwrap().active_job.as_deref(),
            Some("config-provider")
        );
    }

    #[test]
    fn provider_cursor_selects_the_command_target() {
        let mut cm = Some(ConfigManagerState::default());
        let mut jobs = State::default();
        cm.as_mut().unwrap().action_sel = 2; // DisableProvider
        on_key(&mut cm, &mut jobs, key(KeyCode::Right)); // local → anthropic
        on_key(&mut cm, &mut jobs, key(KeyCode::Enter));
        let pending = cm.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(
            pending.args,
            vec!["config", "disable-provider", "anthropic"]
        );
    }

    #[test]
    fn approval_body_states_keys_are_env_only() {
        let mut cm = Some(ConfigManagerState::default());
        let mut jobs = State::default();
        cm.as_mut().unwrap().action_sel = 1;
        on_key(&mut cm, &mut jobs, key(KeyCode::Enter));
        let pending = cm.as_ref().unwrap().approval.as_ref().unwrap();
        assert!(
            pending
                .request
                .body
                .iter()
                .any(|l| l.contains("environment only")),
            "approval must reaffirm the env-only key invariant"
        );
    }

    #[test]
    fn no_keycapture_path_exists() {
        // Guard the invariant structurally: there is no field on the state that
        // could hold an API key, and no action produces a `set-provider-key`
        // argv. Drive every action and assert none mentions a key subcommand.
        let mut jobs = State::default();
        for action_sel in 0..ACTIONS.len() {
            for provider_sel in 0..PROVIDERS.len() {
                let mut cm = Some(ConfigManagerState {
                    action_sel,
                    provider_sel,
                    ..Default::default()
                });
                on_key(&mut cm, &mut jobs, key(KeyCode::Enter));
                if let Some(p) = cm.as_ref().unwrap().approval.as_ref() {
                    assert!(
                        !p.args.iter().any(|a| a.contains("provider-key")),
                        "the TUI must never invoke set/clear-provider-key"
                    );
                }
                // Reset any spawned read-only job so the next iter isn't blocked.
                jobs = State::default();
            }
        }
    }

    #[test]
    fn navigation_clamps() {
        let mut cm = Some(ConfigManagerState::default());
        let mut jobs = State::default();
        for _ in 0..10 {
            on_key(&mut cm, &mut jobs, key(KeyCode::Down));
        }
        assert_eq!(cm.as_ref().unwrap().action_sel, ACTIONS.len() - 1);
        for _ in 0..10 {
            on_key(&mut cm, &mut jobs, key(KeyCode::Right));
        }
        assert_eq!(cm.as_ref().unwrap().provider_sel, PROVIDERS.len() - 1);
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut cm = Some(ConfigManagerState::default());
        let mut jobs = State::default();
        cm.as_mut().unwrap().action_sel = 1;
        on_key(&mut cm, &mut jobs, key(KeyCode::Enter));
        let fx = on_key(&mut cm, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(cm.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut cm = Some(ConfigManagerState::default());
        let mut jobs = State::default();
        on_key(&mut cm, &mut jobs, key(KeyCode::Enter)); // show spawns
        on_key(&mut cm, &mut jobs, key(KeyCode::Char('q')));
        assert!(cm.is_none());
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        let mut jobs = State::default();
        let mut c1 = Some(ConfigManagerState::default());
        on_key(&mut c1, &mut jobs, key(KeyCode::Enter)); // show
        assert_eq!(
            c1.as_ref().unwrap().active_job.as_deref(),
            Some("config-show")
        );
        let mut c2 = Some(ConfigManagerState::default());
        let fx = on_key(&mut c2, &mut jobs, key(KeyCode::Enter));
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
    fn snapshot_lists_actions_and_env_only_notice() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(100, 18);
        let mut term = Terminal::new(backend).unwrap();
        let c = ConfigManagerState::default();
        let jobs = State::default();
        term.draw(|f| draw_config_manager(f, f.area(), &c, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Config & providers"));
        assert!(out.contains("Show saved config"));
        assert!(out.contains("Enable provider"));
        assert!(out.contains("environment only"));
    }
}
