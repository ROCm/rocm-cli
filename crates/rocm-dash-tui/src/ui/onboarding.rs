// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Onboarding wizard (Phase 3 Wave 2, minimal).
//!
//! The first-run flow, rebuilt on the Wave-0 primitives. This is the **minimal**
//! version the phase-3 plan calls for: get a clean machine to a working ROCm
//! runtime two ways —
//!
//! - **Install ROCm SDK** — a one-shot gated `rocm install sdk` with a Configure
//!   step to pick the channel (Release/Nightly) and an optional version pin
//!   (`--build-date` / `--version`); defaults to `--channel release --format
//!   wheel`.
//! - **Adopt existing folder** — pick an existing ROCm env with the Wave-0
//!   [`FolderBrowser`], then approve `rocm runtimes adopt`.
//!
//! Reinstall / uninstall / show-log sub-modals (the full frozen onboarding),
//! first-run auto-trigger + `onboarding_dismissed` persistence, and the frozen
//! flow's post-install `reconcile_onboarding_engine_preference` are documented
//! fast-follows — this overlay is additive and key-triggered (`n`), so it never
//! touches the frozen tui.rs first-run gate. Both paths run through the approval
//! gate and the job-bridge — zero `std::thread::spawn`/`try_recv`.

use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

use rocm_dash_core::state::{JobStatus, SideEffect, State, StateEvent};

use crate::ui::approval::{
    ApprovalChoice, ApprovalRequest, ApprovalVerdict, approval_key, draw_approval,
};
use crate::ui::exec::{exe_label, resolve_exe};
use crate::ui::folder_browser::{FolderBrowser, FolderOutcome, draw_folder_browser};
use crate::ui::job_console::{ConsoleOutcome, on_console_key};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// Which step of the wizard is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnboardingStep {
    #[default]
    Welcome,
    Choose,
    Done,
}

/// The two minimal setup paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnboardingChoice {
    InstallSdk,
    AdoptExisting,
}

/// Menu order; `choice` indexes this.
pub const CHOICES: &[OnboardingChoice] = &[
    OnboardingChoice::InstallSdk,
    OnboardingChoice::AdoptExisting,
];

impl OnboardingChoice {
    const fn label(self) -> &'static str {
        match self {
            Self::InstallSdk => "Install ROCm SDK (pip)",
            Self::AdoptExisting => "Adopt an existing ROCm folder",
        }
    }

    const fn job_id(self) -> &'static str {
        match self {
            Self::InstallSdk => "onboard-install",
            Self::AdoptExisting => "onboard-adopt",
        }
    }

    const fn title(self) -> &'static str {
        match self {
            Self::InstallSdk => "install ROCm SDK",
            Self::AdoptExisting => "adopt existing ROCm folder",
        }
    }

    const fn explanation(self) -> &'static str {
        match self {
            Self::InstallSdk => "This downloads and installs TheRock ROCm wheels on this machine.",
            Self::AdoptExisting => "This registers an existing ROCm folder without modifying it.",
        }
    }
}

/// Install channel for the SDK path. Mirrors `rocm install sdk --channel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Channel {
    #[default]
    Release,
    Nightly,
}

impl Channel {
    const fn as_arg(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Nightly => "nightly",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Release => "Release",
            Self::Nightly => "Nightly",
        }
    }

    const fn toggled(self) -> Self {
        match self {
            Self::Release => Self::Nightly,
            Self::Nightly => Self::Release,
        }
    }
}

/// Optional version pin for the SDK install. Mirrors the CLI's mutually
/// exclusive `--version` / `--build-date` flags (see `install sdk`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PinMode {
    #[default]
    None,
    BuildDate,
    Version,
}

impl PinMode {
    /// Cycle order for the ↑/↓ selector.
    const ORDER: [Self; 3] = [Self::None, Self::BuildDate, Self::Version];

    const fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::BuildDate => "Build date",
            Self::Version => "Version",
        }
    }

    /// The CLI flag this pin maps to, or `None` for no pin.
    const fn arg(self) -> Option<&'static str> {
        match self {
            Self::None => Option::None,
            Self::BuildDate => Some("--build-date"),
            Self::Version => Some("--version"),
        }
    }

    fn index(self) -> usize {
        Self::ORDER.iter().position(|m| *m == self).unwrap_or(0)
    }

    fn next(self) -> Self {
        Self::ORDER[(self.index() + 1) % Self::ORDER.len()]
    }

    fn prev(self) -> Self {
        Self::ORDER[(self.index() + Self::ORDER.len() - 1) % Self::ORDER.len()]
    }
}

/// SDK install configuration (channel + optional version pin). Held on
/// [`OnboardingState`] while the Configure sub-view has focus; cleared once the
/// resulting `install sdk` command is staged for approval.
#[derive(Debug, Clone, Default)]
pub struct InstallConfig {
    pub channel: Channel,
    pub pin_mode: PinMode,
    pub pin_value: String,
}

/// Build the `install sdk` args from a configuration. A `None` pin (or an empty
/// value) leaves the command at `--channel <ch> --format wheel`, so the default
/// Release path is byte-identical to the pre-toggle behavior.
fn build_install_args(cfg: &InstallConfig) -> Vec<String> {
    let mut args = vec![
        "install".to_string(),
        "sdk".to_string(),
        "--channel".to_string(),
        cfg.channel.as_arg().to_string(),
        "--format".to_string(),
        "wheel".to_string(),
    ];
    let pin = cfg.pin_value.trim();
    if let (Some(flag), false) = (cfg.pin_mode.arg(), pin.is_empty()) {
        args.push(flag.to_string());
        args.push(pin.to_string());
    }
    args
}

/// An approved-but-not-yet-run onboarding op.
#[derive(Debug, Clone)]
pub struct PendingOnboard {
    pub choice: OnboardingChoice,
    pub cmd: String,
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub approval_choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the wizard is closed.
#[derive(Debug, Clone, Default)]
pub struct OnboardingState {
    pub step: OnboardingStep,
    pub choice: usize,
    pub browser: Option<FolderBrowser>,
    /// SDK channel/pin picker; `Some` while the Configure sub-view has focus.
    pub install_config: Option<InstallConfig>,
    pub approval: Option<PendingOnboard>,
    pub active_job: Option<String>,
    pub message: Option<String>,
}

/// Derive the conventional Python executable inside an env root (matches the
/// runtime_manager / rocm-core per-host layout without a `rocm-core` dep).
fn derive_python_executable(root: &str) -> String {
    let root = Path::new(root);
    let path = if cfg!(windows) {
        root.join("Scripts").join("python.exe")
    } else {
        root.join("bin").join("python")
    };
    path.to_string_lossy().into_owned()
}

/// Handle a key while the wizard is open.
pub fn on_key(
    ob: &mut Option<OnboardingState>,
    jobs: &mut State,
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(o) = ob.as_mut() else {
        return Vec::new();
    };

    // 1) Adopt folder browser has focus.
    if let Some(fb) = o.browser.as_mut() {
        match fb.on_key(key.code) {
            FolderOutcome::Chosen(path) => {
                o.browser = None;
                let root = path.to_string_lossy().into_owned();
                let args = vec![
                    "runtimes".to_string(),
                    "adopt".to_string(),
                    "--root".to_string(),
                    root.clone(),
                    "--python".to_string(),
                    derive_python_executable(&root),
                ];
                stage_approval(o, OnboardingChoice::AdoptExisting, args);
            }
            FolderOutcome::Cancelled => o.browser = None,
            FolderOutcome::None | FolderOutcome::Navigated => {}
        }
        return Vec::new();
    }

    // 2) Approval modal has focus.
    if let Some(pending) = o.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.approval_choice);
        pending.approval_choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = o.approval.take() {
                    return spawn_onboard(o, jobs, pending.choice, pending.cmd, pending.args);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => o.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 3) A job is showing in the console.
    if let Some(job_id) = o.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *ob = None,
            ConsoleOutcome::Dismissed => {
                // Only a clean exit (code 0) advances to Done. A failed or
                // cancelled job returns to the Choose step with an honest
                // message so the wizard never claims success it didn't earn.
                let ok = jobs
                    .job(&job_id)
                    .is_some_and(|j| matches!(j.status, JobStatus::Done { code: 0 }));
                o.active_job = None;
                if ok {
                    o.message = None;
                    o.step = OnboardingStep::Done;
                } else {
                    o.message = Some(
                        "That step didn't finish cleanly — review the output, then try again."
                            .to_string(),
                    );
                    o.step = OnboardingStep::Choose;
                }
            }
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 3.5) SDK install configuration (channel + version pin) has focus.
    if o.install_config.is_some() {
        return configure_key(o, key);
    }

    // 4) Step navigation.
    match o.step {
        OnboardingStep::Welcome => match key.code {
            KeyCode::Esc | KeyCode::Char('q') => *ob = None,
            KeyCode::Enter => o.step = OnboardingStep::Choose,
            _ => {}
        },
        OnboardingStep::Choose => match key.code {
            KeyCode::Esc | KeyCode::Char('q') => *ob = None,
            KeyCode::Up | KeyCode::Char('k') => o.choice = o.choice.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                o.choice = (o.choice + 1).min(CHOICES.len() - 1);
            }
            KeyCode::Enter => return activate_choice(o),
            _ => {}
        },
        OnboardingStep::Done => match key.code {
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => *ob = None,
            _ => {}
        },
    }
    Vec::new()
}

/// Run the selected choice: install stages an approval; adopt opens the picker
/// first (its approval is staged once a folder is chosen).
fn activate_choice(o: &mut OnboardingState) -> Vec<SideEffect> {
    match CHOICES[o.choice.min(CHOICES.len() - 1)] {
        OnboardingChoice::InstallSdk => {
            // Open the channel/pin picker; the approval is staged on confirm.
            o.install_config = Some(InstallConfig::default());
        }
        OnboardingChoice::AdoptExisting => {
            let start = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
            o.browser = Some(FolderBrowser::new("Pick an existing ROCm folder", start));
        }
    }
    Vec::new()
}

/// Handle a key while the SDK Configure sub-view has focus: `←/→` toggle the
/// channel, `↑/↓` cycle the pin mode, typing edits the pin value, Enter stages
/// the install for approval, Esc returns to the choose menu.
fn configure_key(o: &mut OnboardingState, key: KeyEvent) -> Vec<SideEffect> {
    match key.code {
        KeyCode::Esc => {
            o.install_config = None;
            o.step = OnboardingStep::Choose;
        }
        KeyCode::Enter => {
            if let Some(cfg) = o.install_config.take() {
                let args = build_install_args(&cfg);
                stage_approval(o, OnboardingChoice::InstallSdk, args);
            }
        }
        other => {
            if let Some(cfg) = o.install_config.as_mut() {
                match other {
                    KeyCode::Left | KeyCode::Right | KeyCode::Tab => {
                        cfg.channel = cfg.channel.toggled();
                    }
                    KeyCode::Up => cfg.pin_mode = cfg.pin_mode.prev(),
                    KeyCode::Down => cfg.pin_mode = cfg.pin_mode.next(),
                    KeyCode::Backspace => {
                        cfg.pin_value.pop();
                    }
                    // Only collect characters once a pin mode is selected, so
                    // the default (None) view ignores stray keystrokes.
                    KeyCode::Char(c) if cfg.pin_mode != PinMode::None && !c.is_control() => {
                        cfg.pin_value.push(c);
                    }
                    _ => {}
                }
            }
        }
    }
    Vec::new()
}

/// Stage an approval for a setup op (no job yet).
fn stage_approval(o: &mut OnboardingState, choice: OnboardingChoice, args: Vec<String>) {
    let cmd = resolve_exe();
    let request = ApprovalRequest::new(
        choice.title().to_string(),
        vec![
            format!("{} {}", exe_label(&cmd), args.join(" ")),
            String::new(),
            choice.explanation().to_string(),
        ],
    );
    o.message = None;
    o.approval = Some(PendingOnboard {
        choice,
        cmd,
        args,
        request,
        approval_choice: ApprovalChoice::default(),
    });
}

/// Spawn the approved setup job.
fn spawn_onboard(
    o: &mut OnboardingState,
    jobs: &mut State,
    choice: OnboardingChoice,
    cmd: String,
    args: Vec<String>,
) -> Vec<SideEffect> {
    let id = choice.job_id().to_string();
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd,
        args,
    });
    if fx.is_empty() {
        o.message = Some(format!("“{}” is already running", choice.title()));
        return fx;
    }
    o.active_job = Some(id);
    fx
}

/// Render the wizard (current step, or a browser/approval/console on top).
pub fn draw_onboarding(
    f: &mut Frame,
    area: Rect,
    o: &OnboardingState,
    _jobs: &State,
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Welcome to ROCm — first-run setup"),
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

    if let Some(cfg) = &o.install_config {
        draw_configure(f, rows[0], cfg, theme);
    } else {
        match o.step {
            OnboardingStep::Welcome => {
                let body = vec![
                    Line::from(Span::styled(
                        "Let's get ROCm set up on this machine.",
                        Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                    )),
                    Line::from(""),
                    Line::from(Span::styled(
                        "You can install the ROCm SDK fresh, or adopt an existing ROCm",
                        Style::default().fg(theme.muted),
                    )),
                    Line::from(Span::styled(
                        "folder you already have.",
                        Style::default().fg(theme.muted),
                    )),
                ];
                f.render_widget(Paragraph::new(body), rows[0]);
            }
            OnboardingStep::Choose => {
                let items: Vec<ListItem> = CHOICES
                    .iter()
                    .map(|c| {
                        ListItem::new(Line::from(vec![
                            Span::styled(c.label().to_string(), Style::default().fg(theme.fg)),
                            Span::styled(" (needs approval)", Style::default().fg(theme.warn)),
                        ]))
                    })
                    .collect();
                let mut ls = ListState::default();
                ls.select(Some(o.choice.min(CHOICES.len() - 1)));
                let list = List::new(items).highlight_style(
                    Style::default()
                        .bg(theme.surface_2)
                        .add_modifier(Modifier::BOLD),
                );
                f.render_stateful_widget(list, rows[0], &mut ls);
            }
            OnboardingStep::Done => {
                f.render_widget(
                    Paragraph::new(Line::from(Span::styled(
                        "Setup step finished. You're ready to go — open the dashboard \
                     or chat to get started.",
                        Style::default().fg(theme.ok).add_modifier(Modifier::BOLD),
                    ))),
                    rows[0],
                );
            }
        }
    }

    let msg = o.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[1],
    );

    let hint = if o.install_config.is_some() {
        "←→ channel · ↑↓ pin · type value · Enter confirm · Esc back"
    } else {
        match o.step {
            OnboardingStep::Welcome => "Enter continue · Esc close",
            OnboardingStep::Choose => "↑↓ select · Enter run · Esc close",
            OnboardingStep::Done => "Enter/Esc close",
        }
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hint,
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    if let Some(fb) = &o.browser {
        draw_folder_browser(f, area, fb, theme);
    }
    if let Some(pending) = &o.approval {
        draw_approval(f, area, &pending.request, pending.approval_choice, theme);
    }
}

/// Render the SDK Configure sub-view: channel toggle, pin-mode selector, and
/// (when a pin is selected) the value field.
fn draw_configure(f: &mut Frame, area: Rect, cfg: &InstallConfig, theme: &Theme) {
    let chan = |c: Channel| -> Span {
        if cfg.channel == c {
            Span::styled(
                format!("[ {} ]", c.label()),
                Style::default().fg(theme.ok).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(
                format!("  {}  ", c.label()),
                Style::default().fg(theme.muted),
            )
        }
    };
    let pin = |m: PinMode| -> Span {
        if cfg.pin_mode == m {
            Span::styled(
                format!("[{}]", m.label()),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(format!(" {} ", m.label()), Style::default().fg(theme.muted))
        }
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Channel:  ", Style::default().fg(theme.fg)),
            chan(Channel::Release),
            Span::raw("  "),
            chan(Channel::Nightly),
        ]),
        Line::from(vec![
            Span::styled("Pin:      ", Style::default().fg(theme.fg)),
            pin(PinMode::None),
            Span::raw(" "),
            pin(PinMode::BuildDate),
            Span::raw(" "),
            pin(PinMode::Version),
        ]),
    ];
    if cfg.pin_mode != PinMode::None {
        let value = if cfg.pin_value.is_empty() {
            let placeholder = match cfg.pin_mode {
                PinMode::BuildDate => "YYYY-MM-DD",
                PinMode::Version => "e.g. 7.14.0a20260605",
                PinMode::None => "",
            };
            Span::styled(placeholder.to_string(), Style::default().fg(theme.muted))
        } else {
            Span::styled(format!("{}_", cfg.pin_value), Style::default().fg(theme.fg))
        };
        lines.push(Line::from(vec![
            Span::styled("Value:    ", Style::default().fg(theme.fg)),
            value,
        ]));
    }
    f.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn welcome_enter_advances_to_choose() {
        let mut ob = Some(OnboardingState::default());
        let mut jobs = State::default();
        assert_eq!(ob.as_ref().unwrap().step, OnboardingStep::Welcome);
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert_eq!(ob.as_ref().unwrap().step, OnboardingStep::Choose);
    }

    #[test]
    fn welcome_esc_closes() {
        let mut ob = Some(OnboardingState::default());
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Esc));
        assert!(ob.is_none());
    }

    #[test]
    fn install_choice_is_gated_then_spawns() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        }); // choice 0 = InstallSdk
        let mut jobs = State::default();
        // Enter opens the channel/pin Configure view (no approval, no job yet).
        let fx = on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(ob.as_ref().unwrap().install_config.is_some());
        assert!(ob.as_ref().unwrap().approval.is_none());
        // Enter again confirms the default (Release · no pin) → stages approval.
        let fx = on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(ob.as_ref().unwrap().install_config.is_none());
        let pending = ob.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.choice, OnboardingChoice::InstallSdk);
        assert_eq!(
            pending.args,
            vec![
                "install",
                "sdk",
                "--channel",
                "release",
                "--format",
                "wheel"
            ],
            "default Release path must stay byte-identical to the pre-toggle args"
        );
        assert!(jobs.jobs.is_empty());
        // Approve → spawns.
        let fx = on_key(&mut ob, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert_eq!(
            ob.as_ref().unwrap().active_job.as_deref(),
            Some("onboard-install")
        );
    }

    #[test]
    fn adopt_choice_opens_browser_then_gates_with_python() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            choice: 1, // AdoptExisting
            ..Default::default()
        });
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(ob.as_ref().unwrap().browser.is_some());
        // Enter on row 0 (UseCurrent) chooses cwd → stages adopt approval.
        let fx = on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(fx.is_empty());
        assert!(ob.as_ref().unwrap().browser.is_none());
        let pending = ob.as_ref().unwrap().approval.as_ref().unwrap();
        assert_eq!(pending.choice, OnboardingChoice::AdoptExisting);
        assert!(pending.args.iter().any(|a| a == "adopt"));
        assert!(pending.args.iter().any(|a| a == "--python"));
        let want = if cfg!(windows) {
            "python.exe"
        } else {
            "bin/python"
        };
        assert!(pending.args.iter().any(|a| a.contains(want)));
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // open Configure
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // confirm → stage install
        let fx = on_key(&mut ob, &mut jobs, key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(ob.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn job_dismiss_advances_to_done() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // open Configure
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // confirm → stage
        on_key(&mut ob, &mut jobs, key(KeyCode::Char('y'))); // spawn
        let id = ob.as_ref().unwrap().active_job.clone().unwrap();
        // Finish the job, then Enter dismisses the console → Done.
        jobs.apply(StateEvent::JobDone { id, code: 0 });
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(ob.as_ref().unwrap().active_job.is_none());
        assert_eq!(ob.as_ref().unwrap().step, OnboardingStep::Done);
        // Enter on Done closes the wizard.
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(ob.is_none());
    }

    #[test]
    fn failed_job_returns_to_choose_with_honest_message_not_done() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // open Configure
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // confirm → stage install
        on_key(&mut ob, &mut jobs, key(KeyCode::Char('y'))); // spawn
        let id = ob.as_ref().unwrap().active_job.clone().unwrap();
        // The install FAILED (non-zero exit). Dismiss must NOT claim success.
        jobs.apply(StateEvent::JobDone { id, code: 1 });
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        let s = ob.as_ref().unwrap();
        assert!(s.active_job.is_none());
        assert_eq!(
            s.step,
            OnboardingStep::Choose,
            "failure must not reach Done"
        );
        assert!(
            s.message
                .as_deref()
                .unwrap_or("")
                .contains("didn't finish cleanly")
        );
    }

    #[test]
    fn choose_navigation_clamps() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        let mut jobs = State::default();
        for _ in 0..5 {
            on_key(&mut ob, &mut jobs, key(KeyCode::Down));
        }
        assert_eq!(ob.as_ref().unwrap().choice, CHOICES.len() - 1);
        for _ in 0..5 {
            on_key(&mut ob, &mut jobs, key(KeyCode::Up));
        }
        assert_eq!(ob.as_ref().unwrap().choice, 0);
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // open Configure
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // confirm → stage
        on_key(&mut ob, &mut jobs, key(KeyCode::Char('y'))); // spawn
        on_key(&mut ob, &mut jobs, key(KeyCode::Char('q')));
        assert!(ob.is_none());
    }

    #[test]
    fn relaunch_while_job_running_surfaces_message_not_stale_console() {
        let mut jobs = State::default();
        let mut o1 = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        on_key(&mut o1, &mut jobs, key(KeyCode::Enter)); // open Configure
        on_key(&mut o1, &mut jobs, key(KeyCode::Enter)); // confirm → stage
        on_key(&mut o1, &mut jobs, key(KeyCode::Char('y')));
        assert_eq!(
            o1.as_ref().unwrap().active_job.as_deref(),
            Some("onboard-install")
        );
        // Fresh wizard, same install while the prior job still runs.
        let mut o2 = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        on_key(&mut o2, &mut jobs, key(KeyCode::Enter)); // open Configure
        on_key(&mut o2, &mut jobs, key(KeyCode::Enter)); // confirm → stage
        let fx = on_key(&mut o2, &mut jobs, key(KeyCode::Char('y'))); // approve
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let s = o2.as_ref().unwrap();
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
    fn snapshot_welcome_and_choose() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        // Welcome.
        let backend = TestBackend::new(100, 20);
        let mut term = Terminal::new(backend).unwrap();
        let o = OnboardingState::default();
        let jobs = State::default();
        term.draw(|f| draw_onboarding(f, f.area(), &o, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("first-run setup"));
        assert!(out.contains("get ROCm set up"));
        // Choose.
        let backend = TestBackend::new(100, 20);
        let mut term = Terminal::new(backend).unwrap();
        let o = OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        };
        term.draw(|f| draw_onboarding(f, f.area(), &o, &jobs, &theme))
            .unwrap();
        let out: String = term
            .backend()
            .buffer()
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Install ROCm SDK"));
        assert!(out.contains("Adopt an existing"));
        assert!(out.contains("needs approval"));
    }

    /// Open the SDK Configure view (choice 0 = InstallSdk) and return the state.
    fn open_configure() -> (Option<OnboardingState>, State) {
        let mut ob = Some(OnboardingState {
            step: OnboardingStep::Choose,
            ..Default::default()
        });
        let mut jobs = State::default();
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(ob.as_ref().unwrap().install_config.is_some());
        (ob, jobs)
    }

    fn type_str(ob: &mut Option<OnboardingState>, jobs: &mut State, s: &str) {
        for c in s.chars() {
            on_key(ob, jobs, key(KeyCode::Char(c)));
        }
    }

    fn staged_args(o: &OnboardingState) -> Vec<String> {
        o.approval.as_ref().unwrap().args.clone()
    }

    #[test]
    fn nightly_toggle_sets_channel_arg() {
        let (mut ob, mut jobs) = open_configure();
        on_key(&mut ob, &mut jobs, key(KeyCode::Right)); // Release → Nightly
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter)); // confirm
        assert_eq!(
            staged_args(ob.as_ref().unwrap()),
            vec![
                "install",
                "sdk",
                "--channel",
                "nightly",
                "--format",
                "wheel"
            ]
        );
    }

    #[test]
    fn channel_toggle_is_reversible() {
        let (mut ob, mut jobs) = open_configure();
        on_key(&mut ob, &mut jobs, key(KeyCode::Right)); // → Nightly
        on_key(&mut ob, &mut jobs, key(KeyCode::Left)); // → Release
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert!(staged_args(ob.as_ref().unwrap()).contains(&"release".to_string()));
    }

    #[test]
    fn nightly_build_date_pin_appends_flag() {
        let (mut ob, mut jobs) = open_configure();
        on_key(&mut ob, &mut jobs, key(KeyCode::Right)); // Nightly
        on_key(&mut ob, &mut jobs, key(KeyCode::Down)); // pin None → Build date
        type_str(&mut ob, &mut jobs, "2026-06-05");
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert_eq!(
            staged_args(ob.as_ref().unwrap()),
            vec![
                "install",
                "sdk",
                "--channel",
                "nightly",
                "--format",
                "wheel",
                "--build-date",
                "2026-06-05"
            ]
        );
    }

    #[test]
    fn version_pin_appends_flag() {
        let (mut ob, mut jobs) = open_configure();
        // pin None → Build date → Version.
        on_key(&mut ob, &mut jobs, key(KeyCode::Down));
        on_key(&mut ob, &mut jobs, key(KeyCode::Down));
        type_str(&mut ob, &mut jobs, "7.14.0a20260605");
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        let args = staged_args(ob.as_ref().unwrap());
        assert!(
            args.windows(2)
                .any(|w| w == ["--version", "7.14.0a20260605"])
        );
        assert!(!args.iter().any(|a| a == "--build-date"));
    }

    #[test]
    fn empty_pin_value_is_omitted() {
        let (mut ob, mut jobs) = open_configure();
        on_key(&mut ob, &mut jobs, key(KeyCode::Down)); // Build date, no value typed
        on_key(&mut ob, &mut jobs, key(KeyCode::Enter));
        assert_eq!(
            staged_args(ob.as_ref().unwrap()),
            vec![
                "install",
                "sdk",
                "--channel",
                "release",
                "--format",
                "wheel"
            ],
            "an empty pin must not add a flag"
        );
    }

    #[test]
    fn chars_ignored_until_a_pin_mode_is_selected() {
        let (mut ob, mut jobs) = open_configure();
        type_str(&mut ob, &mut jobs, "abc"); // pin mode still None
        assert_eq!(
            ob.as_ref()
                .unwrap()
                .install_config
                .as_ref()
                .unwrap()
                .pin_value,
            ""
        );
    }

    #[test]
    fn backspace_edits_pin_value() {
        let (mut ob, mut jobs) = open_configure();
        on_key(&mut ob, &mut jobs, key(KeyCode::Down)); // Build date
        type_str(&mut ob, &mut jobs, "2026-0X");
        on_key(&mut ob, &mut jobs, key(KeyCode::Backspace));
        type_str(&mut ob, &mut jobs, "6");
        assert_eq!(
            ob.as_ref()
                .unwrap()
                .install_config
                .as_ref()
                .unwrap()
                .pin_value,
            "2026-06"
        );
    }

    #[test]
    fn configure_esc_returns_to_choose() {
        let (mut ob, mut jobs) = open_configure();
        on_key(&mut ob, &mut jobs, key(KeyCode::Esc));
        let s = ob.as_ref().unwrap();
        assert!(s.install_config.is_none());
        assert!(s.approval.is_none());
        assert_eq!(s.step, OnboardingStep::Choose);
    }
}
