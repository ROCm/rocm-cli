// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `KeyAction` dispatch: translating a key press (or a resolved mouse hit)
//! into a `KeyAction`, and applying it to reducer state. Split out of
//! `app/mod.rs` to keep the core reducer focused.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind};

use crate::ui;

use super::AppState;
use super::scrollbar::{PaneFocus, ScrollTarget};
use super::summary::summarize_json_value;
use super::types::{ActiveTab, ChatConsent, ChatKeyCtx, Modal};
#[cfg(test)]
use super::types::{ChatProvider, ChatRole};

/// Lines per PageUp/PageDown step in the chat transcript.
const CHAT_SCROLL_STEP: i16 = 5;

/// Run an approved mutating action across the seam and render a concise summary
/// (never a raw JSON dump). Sync + executor-generic so the approve path is
/// unit-testable without tokio; the event loop calls it inside spawn_blocking.
pub(crate) fn run_approved(
    executor: &crate::tool_exec::SharedRocmToolExecutor,
    name: &str,
    args: &serde_json::Value,
) -> String {
    use crate::tool_exec::RocmToolOutcome;
    match executor.execute_approved(name, args) {
        RocmToolOutcome::Result(v) => {
            let body = summarize_json_value(&v);
            if body.is_empty() {
                format!("Approved · {name}: done")
            } else {
                format!("Approved · {name}:\n{body}")
            }
        }
        RocmToolOutcome::Error(e) => format!("Approved · {name} failed: {e}"),
        // A mutating tool's approved replay should not re-request approval; if it
        // somehow does, surface it plainly rather than silently looping.
        RocmToolOutcome::ApprovalRequired(_) => {
            format!("Approved · {name}: unexpected second approval request (not run)")
        }
    }
}

/// Wrap a list cursor by `delta`, cycling within `0..len`. `len == 0` → 0.
const fn wrap_cursor(cur: usize, delta: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len.cast_signed();
    (cur.cast_signed() + delta).rem_euclid(n) as usize
}

/// Resolve a quit request (issue #145): a model still being served gets a
/// confirm prompt first — opening it, not quitting, is this fn's whole job;
/// the event loop's dedicated `quit_confirm_pending` arm owns the actual exit
/// once the prompt resolves. Returns `true` to request immediate exit the
/// same way `apply_action` does, so every quit entry point — the `q` key, the
/// Esc main menu's "Quit" row — shares this one gate rather than each
/// re-deciding (and risking re-deciding wrong) whether to ask first.
fn request_quit(state: &mut AppState) -> bool {
    if state.has_live_instance() {
        state.open_quit_confirm();
        return false;
    }
    true
}

/// Apply a `KeyAction` to mutable state. Returns `true` when the action
/// requests application exit (Quit).
pub(crate) fn apply_action(state: &mut AppState, action: KeyAction) -> bool {
    match action {
        KeyAction::Quit => return request_quit(state),
        KeyAction::SwitchTab(t) => {
            state.active_tab = t;
            state.modal = Modal::None;
            // A fresh tab always starts with focus on its Actions list, never
            // stranded in the Details pane from a previous visit.
            state.pane_focus = PaneFocus::Actions;
        }
        KeyAction::Move(d) => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_move(d);
            } else {
                // Changing the verb selection snaps focus back to the Actions
                // list so Details re-previews the newly selected operation.
                if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                    state.pane_focus = PaneFocus::Actions;
                }
                state.move_selection(d);
            }
        }
        KeyAction::PaneFocusDetail => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.pane_focus = PaneFocus::Detail;
            }
        }
        KeyAction::PaneFocusActions => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                state.pane_focus = PaneFocus::Actions;
            }
        }
        KeyAction::PaneActivate => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                match state.pane_focus {
                    // From the Actions list, Enter steps INTO the Details pane.
                    PaneFocus::Actions => state.pane_focus = PaneFocus::Detail,
                    // From Details, Enter opens the operation's manager.
                    PaneFocus::Detail => {
                        let verb = state.pane_verb_action();
                        return apply_action(state, verb);
                    }
                }
            }
        }
        KeyAction::PaneEscape => {
            // Esc backs out one level: Details → Actions, then Actions → menu.
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving)
                && state.pane_focus == PaneFocus::Detail
            {
                state.pane_focus = PaneFocus::Actions;
            } else {
                return apply_action(state, KeyAction::OpenMenu);
            }
        }
        KeyAction::PaneSelect(i) => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                let last = state.pane_verb_count().saturating_sub(1);
                state.set_selection(state.active_tab, i.min(last));
                state.pane_focus = PaneFocus::Actions;
            }
        }
        KeyAction::SelectFirst => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_first();
            } else {
                state.select_first();
            }
        }
        KeyAction::SelectLast => {
            if state.modal == Modal::ThemePicker {
                state.theme_picker_last();
            } else {
                state.select_last();
            }
        }
        KeyAction::OpenDetail => {
            if matches!(state.active_tab, ActiveTab::Rocm | ActiveTab::Serving) {
                // Verb rows open the matching manager via the existing seam;
                // there is no detail modal on the ROCm/Serving tabs.
                let verb = state.pane_verb_action();
                return apply_action(state, verb);
            }
            if state.selection_len() > 0 {
                state.modal = Modal::Detail;
                state.reset_instance_detail_scroll();
            }
        }
        KeyAction::ToggleHelp => {
            state.modal = if state.modal == Modal::Help {
                Modal::None
            } else {
                state.close_overlays();
                Modal::Help
            };
        }
        KeyAction::CloseModal => state.modal = Modal::None,
        // The operational overlays are mutually exclusive: opening any one first
        // closes the rest (see `close_overlays`), so no open path — key, mouse,
        // or effect — can ever leave two `Some` at once.
        KeyAction::OpenServices => {
            state.close_overlays();
            state.services = Some(crate::ui::services_manager::ServicesManagerState::default());
        }
        KeyAction::OpenServeWizard => {
            state.close_overlays();
            state.serve_wizard = Some(crate::ui::serve_wizard::ServeWizardState::default());
        }
        KeyAction::OpenEngineManager => {
            state.close_overlays();
            state.engine_manager = Some(crate::ui::engine_manager::EngineManagerState::default());
        }
        KeyAction::OpenExamine => {
            state.close_overlays();
            state.examine_manager =
                Some(crate::ui::examine_manager::ExamineManagerState::default());
        }
        KeyAction::OpenUpdate => {
            state.close_overlays();
            state.update_manager = Some(crate::ui::update_manager::UpdateManagerState::default());
        }
        KeyAction::OpenInstall => {
            state.close_overlays();
            state.install_manager =
                Some(crate::ui::install_manager::InstallManagerState::default());
        }
        KeyAction::OpenLogs => {
            state.close_overlays();
            state.logs_view = Some(crate::ui::logs_view::LogsViewState::default());
        }
        KeyAction::OpenRuntimes => {
            state.close_overlays();
            state.runtime_manager =
                Some(crate::ui::runtime_manager::RuntimeManagerState::default());
        }
        KeyAction::OpenOnboarding => {
            state.close_overlays();
            state.onboarding = Some(crate::ui::onboarding::OnboardingState::default());
        }
        KeyAction::OpenAutomations => {
            state.close_overlays();
            state.automations_manager =
                Some(crate::ui::automations_manager::AutomationsManagerState::default());
        }
        KeyAction::OpenCommand => {
            state.close_overlays();
            state.command_screen = Some(crate::ui::command_screen::CommandScreenState::default());
        }
        KeyAction::OpenConfig => {
            state.close_overlays();
            state.config_manager = Some(crate::ui::config_manager::ConfigManagerState::default());
        }
        KeyAction::OpenBenchRun => {
            let bench_csv = state.bench_results_dir.clone();
            state.close_overlays();
            state.bench_run = Some(crate::ui::bench_run::BenchRunState::new(
                bench_csv.as_deref(),
            ));
        }
        KeyAction::OpenThemePicker => state.open_theme_picker(),
        KeyAction::ApplyThemePick => state.apply_theme_pick(),
        KeyAction::OpenMenu => {
            state.modal = Modal::Menu;
            state.menu_sel = 0;
        }
        KeyAction::OpenPalette => {
            state.modal = Modal::Palette;
            state.palette_sel = 0;
        }
        KeyAction::MenuMove(d) => match state.modal {
            Modal::Menu => {
                state.menu_sel = wrap_cursor(state.menu_sel, d, crate::ui::modal::MENU_ITEMS);
            }
            Modal::Palette => {
                state.palette_sel =
                    wrap_cursor(state.palette_sel, d, crate::ui::modal::PALETTE_DESTS.len());
            }
            _ => {}
        },
        KeyAction::OptionsTab(d) => {
            if state.modal == Modal::Options {
                state.options_tab =
                    wrap_cursor(state.options_tab, d, crate::ui::modal::OPTIONS_TABS.len());
            }
        }
        KeyAction::MenuActivate => match state.modal {
            Modal::Menu => match state.menu_sel {
                0 => {
                    state.modal = Modal::Options;
                    state.options_tab = 0;
                }
                1 => {
                    state.modal = Modal::GlobalHelp;
                }
                // Issue #145: this must share `KeyAction::Quit`'s own gate —
                // choosing "Quit" from the Esc menu is a second, independent
                // entry point to the same decision, not a separate one that
                // gets to skip the confirm prompt.
                _ => return request_quit(state),
            },
            Modal::Palette => {
                if let Some((_, tab)) = crate::ui::modal::PALETTE_DESTS.get(state.palette_sel) {
                    state.active_tab = *tab;
                }
                state.modal = Modal::None;
            }
            _ => {}
        },
        KeyAction::ScrollModal(delta) if state.modal == Modal::Detail => {
            state.scroll_instance_detail(delta);
        }
        KeyAction::ScrollModal(_) => {}
        KeyAction::ScrollConsole(dv, dh) => state.scroll_console(dv, dh),
        KeyAction::ScrollDock(dv) => state.scroll_dock(dv),
        KeyAction::ScrollGrab(target, pos, grab_offset) => {
            state.apply_scroll_grab(target, pos, grab_offset);
        }
        KeyAction::ScrollRelease => state.scroll_drag = None,
        KeyAction::ReplayTogglePause => {
            if let Some(r) = state.replay.as_mut() {
                r.paused = !r.paused;
                if r.paused {
                    r.controller.pause();
                } else {
                    r.controller.resume();
                }
            }
        }
        KeyAction::ReplaySpeedUp => {
            if let Some(r) = state.replay.as_mut() {
                r.speed = crate::replay::next_speed(r.speed);
                r.controller.set_speed(r.speed);
            }
        }
        KeyAction::ReplaySpeedDown => {
            if let Some(r) = state.replay.as_mut() {
                r.speed = crate::replay::prev_speed(r.speed);
                r.controller.set_speed(r.speed);
            }
        }
        KeyAction::ReplayJump(delta_s) => {
            if let Some(r) = state.replay.as_ref() {
                r.controller.jump(delta_s);
            }
        }
        KeyAction::ChatInput(c) => state.chat_input.push(c),
        KeyAction::ChatBackspace => {
            state.chat_input.pop();
        }
        KeyAction::ChatSubmit => state.submit_chat(),
        KeyAction::ChatFocus => state.chat_focused = true,
        KeyAction::ChatBlur => state.chat_focused = false,
        KeyAction::ChatConsentAccept => state.accept_chat_consent(),
        KeyAction::ChatConsentDecline => state.decline_chat_consent(),
        KeyAction::ChatDetect => state.request_detect(),
        KeyAction::ChatDetectAccept => state.accept_detect_offer(),
        KeyAction::ChatDetectSave => state.save_detect_offer(),
        KeyAction::ChatDetectDismiss => state.dismiss_detect_offer(),
        KeyAction::ChatScroll(d) => {
            let next = (i32::from(state.chat_scroll) + i32::from(d)).max(0) as usize;
            state.set_chat_scroll(next);
        }
        KeyAction::Nothing => {}
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    Nothing,
    Quit,
    SwitchTab(ActiveTab),
    /// ROCm/Serving tab: move focus into the Details pane (`→`).
    PaneFocusDetail,
    /// ROCm/Serving tab: move focus back to the Actions list (`←`).
    PaneFocusActions,
    /// ROCm/Serving tab: activate the current focus — from the Actions list,
    /// focus the Details pane; from Details, open the operation's manager.
    PaneActivate,
    /// ROCm/Serving tab: Esc — step out of Details back to the Actions list, or,
    /// when already on the list, fall through to the main menu.
    PaneEscape,
    /// ROCm/Serving tab: select the verb at this index and park focus on the
    /// Actions list (from a mouse click on a verb row).
    PaneSelect(usize),
    Move(isize),
    SelectFirst,
    SelectLast,
    OpenDetail,
    ToggleHelp,
    CloseModal,
    OpenThemePicker,
    ApplyThemePick,
    /// Vertical scroll inside the active modal body (positive = down).
    ScrollModal(i16),
    /// Pan the active job console: `(vertical_lines, horizontal_cols)`, negative
    /// = up/left. No-op when no console is showing.
    ScrollConsole(i16, i16),
    /// Scroll the wide-layout right LOGS dock by N lines (negative = toward the
    /// newest line). No-op when the dock isn't showing.
    ScrollDock(i16),
    /// Grab a scrollbar at `position`, retaining the pointer's offset inside the
    /// thumb so subsequent drag events track without a jump.
    ScrollGrab(ScrollTarget, usize, u16),
    /// Release the active scrollbar drag (mouse button up).
    ScrollRelease,
    /// Toggle replay pause / resume. No-op when not replaying.
    ReplayTogglePause,
    /// Step replay speed up or down (clamped). No-op when not replaying.
    ReplaySpeedUp,
    ReplaySpeedDown,
    /// Move the replay playhead by `delta_s` seconds (negative = rewind).
    ReplayJump(i64),
    /// Chat insert-mode: append a character to `chat_input`.
    ChatInput(char),
    /// Chat insert-mode: pop the last character from `chat_input`.
    ChatBackspace,
    /// Chat: submit the current input buffer as a user turn.
    ChatSubmit,
    /// Chat: enter text-entry focus.
    ChatFocus,
    /// Chat: leave text-entry focus.
    ChatBlur,
    /// Chat: accept the detected endpoint (one-time consent).
    ChatConsentAccept,
    /// Chat: decline the detected endpoint.
    ChatConsentDecline,
    /// Chat: probe for a local engine and offer it (in-TUI auto-detect).
    ChatDetect,
    /// Chat: accept the detected local endpoint for this session.
    ChatDetectAccept,
    /// Chat: accept the detected endpoint and persist it to config.
    ChatDetectSave,
    /// Chat: dismiss the detected-endpoint offer, keeping the prior config.
    ChatDetectDismiss,
    /// Chat: scroll the transcript by N lines (positive = down).
    ChatScroll(i16),
    /// Open the btop-style Esc main menu (P4).
    OpenMenu,
    /// Open the "Go to…" command palette (P4).
    OpenPalette,
    /// Move the cursor within the active overlay (Menu / Palette) by N rows.
    MenuMove(isize),
    /// Cycle the Options panel's tab by N (left/right).
    OptionsTab(isize),
    /// Activate the highlighted row in the active overlay (Menu / Palette).
    MenuActivate,
    /// Open the services-manager overlay (Phase 3 Wave 1).
    OpenServices,
    /// Open the serve-wizard overlay (Phase 3 Wave 1).
    OpenServeWizard,
    /// Open the engine-manager overlay (Phase 3 Wave 1).
    OpenEngineManager,
    /// Open the examine overlay (Phase 3 Wave 2).
    OpenExamine,
    /// Open the update overlay (Phase 3 Wave 2).
    OpenUpdate,
    /// Open the install overlay (Phase 3 Wave 2).
    OpenInstall,
    /// Open the runtime manager overlay.
    OpenRuntimes,
    /// Open the onboarding wizard overlay.
    OpenOnboarding,
    /// Open the automations manager overlay.
    OpenAutomations,
    /// Open the command runner overlay.
    OpenCommand,
    /// Open the config & provider manager overlay.
    OpenConfig,
    /// Open the logs overlay (Phase 3 Wave 3).
    OpenLogs,
    /// Open the bench-run form overlay.
    OpenBenchRun,
}

/// Whether a crossterm key event should be acted on. Terminals emit
/// Release/Repeat events in addition to Press (notably Windows Terminal /
/// ConPTY under WSL, and any terminal advertising the kitty keyboard protocol).
///
/// The whole TUI acts on Press only. Both the general [`handle_key`] and the
/// event loop's operational-overlay dispatch share this gate — without it, a
/// single keystroke reaches an overlay's `on_key` more than once, which made
/// Enter in the serve wizard's model picker re-open the picker (seeded with the
/// just-chosen model as a filter) instead of choosing it.
pub(crate) const fn is_actionable_key(kind: KeyEventKind) -> bool {
    matches!(kind, KeyEventKind::Press)
}

pub(crate) fn handle_key(
    k: KeyEvent,
    current: ActiveTab,
    modal: &Modal,
    chat: ChatKeyCtx,
) -> KeyAction {
    if !is_actionable_key(k.kind) {
        return KeyAction::Nothing;
    }
    // Chat tab key handling, placed BEFORE the global hotkey match so focused
    // text entry and the consent prompt absorb keys (the short-circuit that
    // stops `q`, `1`–`5`, etc. from firing while typing / deciding consent).
    if current == ActiveTab::Chat && *modal == Modal::None {
        // A detected-endpoint offer (gate-only) absorbs its decision keys before
        // the normal consent prompt: y use now, n/Esc dismiss. ([s] use & save
        // is wired with persistence.) Other keys fall through to the globals.
        if chat.offer_pending && chat.consent != ChatConsent::Accepted {
            match k.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                    return KeyAction::ChatDetectAccept;
                }
                KeyCode::Char('s' | 'S') => {
                    return KeyAction::ChatDetectSave;
                }
                KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                    return KeyAction::ChatDetectDismiss;
                }
                _ => {}
            }
        }
        match chat.consent {
            ChatConsent::Accepted => {
                // History scroll works whether or not the input is focused.
                match k.code {
                    KeyCode::PageUp => return KeyAction::ChatScroll(-CHAT_SCROLL_STEP),
                    KeyCode::PageDown => return KeyAction::ChatScroll(CHAT_SCROLL_STEP),
                    _ => {}
                }
                if chat.focused {
                    return match k.code {
                        KeyCode::Esc => KeyAction::ChatBlur,
                        KeyCode::Enter => KeyAction::ChatSubmit,
                        KeyCode::Backspace => KeyAction::ChatBackspace,
                        KeyCode::Char(c) => KeyAction::ChatInput(c),
                        _ => KeyAction::Nothing,
                    };
                }
                // Not focused: `i`/`Enter` enter insert mode; other keys fall
                // through to the global hotkeys below.
                if let KeyCode::Char('i') | KeyCode::Enter = k.code {
                    return KeyAction::ChatFocus;
                }
            }
            ChatConsent::Pending | ChatConsent::Declined => {
                // Consent gate: y/Enter accept, n decline, d detect a local
                // engine. Other keys (q, digits, Tab, ?) fall through to the
                // globals so the user isn't trapped.
                match k.code {
                    KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                        return KeyAction::ChatConsentAccept;
                    }
                    KeyCode::Char('n' | 'N') => {
                        return KeyAction::ChatConsentDecline;
                    }
                    KeyCode::Char('d' | 'D') => {
                        return KeyAction::ChatDetect;
                    }
                    _ => {}
                }
            }
            // No endpoint configured: the only gate action is to detect one.
            ChatConsent::Unavailable => {
                if let KeyCode::Char('d' | 'D') = k.code {
                    return KeyAction::ChatDetect;
                }
            }
        }
    }
    // ThemePicker is a navigable modal — j/k/g/G move the cursor, Enter applies.
    if *modal == Modal::ThemePicker {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Char('t') => KeyAction::CloseModal,
            KeyCode::Enter => KeyAction::ApplyThemePick,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::Move(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::Move(-1),
            KeyCode::Char('g') | KeyCode::Home => KeyAction::SelectFirst,
            KeyCode::Char('G') | KeyCode::End => KeyAction::SelectLast,
            _ => KeyAction::Nothing,
        };
    }
    // Detail modal: vertical scroll keys, plus quit/close.
    if *modal == Modal::Detail {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => KeyAction::CloseModal,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::ScrollModal(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::ScrollModal(-1),
            KeyCode::PageDown => KeyAction::ScrollModal(10),
            KeyCode::PageUp => KeyAction::ScrollModal(-10),
            KeyCode::Char('g') | KeyCode::Home => KeyAction::ScrollModal(i16::MIN),
            KeyCode::Char('G') | KeyCode::End => KeyAction::ScrollModal(i16::MAX),
            _ => KeyAction::Nothing,
        };
    }
    // Help absorbs everything except quit / close / ? toggle — the popup is
    // always sized to fit its content, so there's nothing to scroll.
    if *modal == Modal::Help {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => KeyAction::CloseModal,
            _ => KeyAction::Nothing,
        };
    }
    // Global help overlay (opened from the Esc menu): close, same as the
    // contextual Help above.
    if *modal == Modal::GlobalHelp {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => KeyAction::CloseModal,
            _ => KeyAction::Nothing,
        };
    }
    // Esc main menu: ↑↓ cycle Options/Help/Quit, Enter activates, Esc closes.
    if *modal == Modal::Menu {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc => KeyAction::CloseModal,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::MenuMove(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::MenuMove(-1),
            KeyCode::Enter => KeyAction::MenuActivate,
            _ => KeyAction::Nothing,
        };
    }
    // Command palette: ↑↓ choose destination, Enter goes, Esc closes.
    if *modal == Modal::Palette {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc => KeyAction::CloseModal,
            KeyCode::Char('j') | KeyCode::Down => KeyAction::MenuMove(1),
            KeyCode::Char('k') | KeyCode::Up => KeyAction::MenuMove(-1),
            KeyCode::Enter => KeyAction::MenuActivate,
            _ => KeyAction::Nothing,
        };
    }
    // Options panel: ←→ switch settings tab, Esc closes.
    if *modal == Modal::Options {
        return match k.code {
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Esc => KeyAction::CloseModal,
            KeyCode::Char('h') | KeyCode::Left | KeyCode::BackTab => KeyAction::OptionsTab(-1),
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Tab => KeyAction::OptionsTab(1),
            _ => KeyAction::Nothing,
        };
    }
    match k.code {
        KeyCode::Char('q') => KeyAction::Quit,
        // Esc opens the main menu when idle — managers/approval are routed
        // upstream, and Chat-focused Esc is handled by the short-circuit above.
        // On ROCm/Serving, Esc first steps out of the detail pane (resolved
        // against focus in `apply_action`); elsewhere it opens the main menu.
        KeyCode::Esc if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneEscape
        }
        KeyCode::Esc => KeyAction::OpenMenu,
        KeyCode::Char(':') => KeyAction::OpenPalette,
        KeyCode::Char('?') => KeyAction::ToggleHelp,
        KeyCode::Char('t') => KeyAction::OpenThemePicker,
        KeyCode::BackTab => KeyAction::SwitchTab(current.prev()),
        KeyCode::Tab => {
            if k.modifiers.contains(KeyModifiers::SHIFT) {
                KeyAction::SwitchTab(current.prev())
            } else {
                KeyAction::SwitchTab(current.next())
            }
        }
        KeyCode::Char(c @ '1'..='5') => match ActiveTab::from_digit(c) {
            Some(t) => KeyAction::SwitchTab(t),
            None => KeyAction::Nothing,
        },
        KeyCode::PageDown => KeyAction::Move(10),
        KeyCode::PageUp => KeyAction::Move(-10),
        KeyCode::Char(' ') => KeyAction::ReplayTogglePause,
        KeyCode::Char('+' | '=') => KeyAction::ReplaySpeedUp,
        KeyCode::Char('-' | '_') => KeyAction::ReplaySpeedDown,
        KeyCode::Char('[') => KeyAction::ReplayJump(-10),
        KeyCode::Char(']') => KeyAction::ReplayJump(10),
        KeyCode::Char('{') => KeyAction::ReplayJump(-60),
        KeyCode::Char('}') => KeyAction::ReplayJump(60),
        KeyCode::Char('j') | KeyCode::Down => KeyAction::Move(1),
        KeyCode::Char('k') | KeyCode::Up => KeyAction::Move(-1),
        KeyCode::Char('g') | KeyCode::Home => KeyAction::SelectFirst,
        KeyCode::Char('G') | KeyCode::End => KeyAction::SelectLast,
        // The guided-action letter hotkeys live ONLY on Observe (the telemetry
        // surface) — quick jumps into the managers via the existing seam. On
        // ROCm/Serving the Actions list is the single interaction path, so the
        // per-tab letter hotkeys are retired there.
        // Services manager: open where servers live.
        KeyCode::Char('s') if current == ActiveTab::Observe => KeyAction::OpenServices,
        // Serve wizard: launch a model.
        KeyCode::Char('w') if current == ActiveTab::Observe => KeyAction::OpenServeWizard,
        // Engine manager: use/install/reinstall serving engines.
        KeyCode::Char('e') if current == ActiveTab::Observe => KeyAction::OpenEngineManager,
        // Examine: read-only environment check.
        KeyCode::Char('d') if current == ActiveTab::Observe => KeyAction::OpenExamine,
        // Update: check/preview/apply ROCm package updates.
        KeyCode::Char('u') if current == ActiveTab::Observe => KeyAction::OpenUpdate,
        // Install: ROCm SDK (TheRock) install / dry-run.
        KeyCode::Char('i') if current == ActiveTab::Observe => KeyAction::OpenInstall,
        // Logs: browse recent ROCm CLI logs.
        KeyCode::Char('l') if current == ActiveTab::Observe => KeyAction::OpenLogs,
        // Bench-run: launch a bench sweep from the TUI.
        KeyCode::Char('b') if current == ActiveTab::Observe => KeyAction::OpenBenchRun,
        // Runtimes: list/activate/adopt/import ROCm runtimes.
        KeyCode::Char('r') if current == ActiveTab::Observe => KeyAction::OpenRuntimes,
        // Onboarding: first-run setup wizard (install / adopt).
        KeyCode::Char('n') if current == ActiveTab::Observe => KeyAction::OpenOnboarding,
        // Automations: list/enable/disable background checks.
        KeyCode::Char('a') if current == ActiveTab::Observe => KeyAction::OpenAutomations,
        // Command runner: run any ROCm CLI subcommand (gated).
        KeyCode::Char('c') if current == ActiveTab::Observe => KeyAction::OpenCommand,
        // Config & providers.
        KeyCode::Char('p') if current == ActiveTab::Observe => KeyAction::OpenConfig,
        // ROCm/Serving tabs: arrow keys drive the focus-into-detail interaction;
        // Enter is focus-aware (list → focus detail, detail → open the manager).
        KeyCode::Right if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneFocusDetail
        }
        KeyCode::Left if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneFocusActions
        }
        KeyCode::Enter if matches!(current, ActiveTab::Rocm | ActiveTab::Serving) => {
            KeyAction::PaneActivate
        }
        KeyCode::Enter => KeyAction::OpenDetail,
        _ => KeyAction::Nothing,
    }
}

/// Translate a `MouseEvent` into the existing `KeyAction` vocabulary.
///
/// The caller is responsible for the surrounding state context:
/// - `last_tab_bar_area` / `last_body_area` are read off `AppState` by the
///   event loop so this function stays pure on the input event.
/// - Per-tab body clicks are dispatched to the active tab module's
///   `hit_test` from the event loop.
///
/// We only translate the parts of mouse handling that are tab-agnostic:
/// the scroll wheel, and (in the event loop) the tab-bar click. Per-tab
/// click is handled in tab modules.
/// Map a navigation key to a job-console pan delta `(lines, cols)` while a
/// console is showing. `None` for non-scroll keys so they fall through to the
/// console's own action handler (Ctrl+C / q / Esc / Enter). A page is 10 lines.
pub(crate) const fn console_scroll_delta(code: KeyCode) -> Option<(i16, i16)> {
    match code {
        KeyCode::PageDown => Some((10, 0)),
        KeyCode::PageUp => Some((-10, 0)),
        KeyCode::Down => Some((1, 0)),
        KeyCode::Up => Some((-1, 0)),
        KeyCode::Right => Some((0, 4)),
        KeyCode::Left => Some((0, -4)),
        _ => None,
    }
}

pub fn handle_mouse(ev: MouseEvent, modal: &Modal, tab: ActiveTab) -> KeyAction {
    // Domain-tab (ROCm/Serving) and overlay/console scroll is resolved in
    // `resolve_mouse` (it needs `&AppState` for hit-testing and overlay state).
    // This handles the remaining position-independent targets: the scrollable
    // modal body and the Observe instances list. One row per wheel notch.
    let delta: i16 = match ev.kind {
        MouseEventKind::ScrollDown => 1,
        MouseEventKind::ScrollUp => -1,
        _ => return KeyAction::Nothing,
    };
    if *modal == Modal::Detail {
        KeyAction::ScrollModal(delta)
    } else if *modal == Modal::ThemePicker || (*modal == Modal::None && tab == ActiveTab::Observe) {
        KeyAction::Move(delta as isize)
    } else {
        KeyAction::Nothing
    }
}

/// Resolve a left-click at `(x, y)` against `tab_bar_area`. Returns the tab
/// to switch to, or `None` if the click is outside or doesn't land on a chip.
///
/// Uses [`ui::tabs::compute_chip_layout`] so the hit-test geometry exactly
/// mirrors what `draw_tab_bar` rendered. Separator gaps (` · `) between
/// chips are intentional dead zones — clicking the dot does nothing.
pub fn tab_bar_hit(tab_bar_area: ratatui::layout::Rect, x: u16, y: u16) -> Option<ActiveTab> {
    if y != tab_bar_area.y {
        return None;
    }
    let chips = ui::tabs::compute_chip_layout(tab_bar_area.x);
    let bar_right = tab_bar_area.x.saturating_add(tab_bar_area.width);
    for chip in chips {
        if chip.x_end > bar_right {
            // Chip overflows the bar — terminal too narrow to show it; skip.
            continue;
        }
        if x >= chip.x_start && x < chip.x_end {
            return Some(chip.tab);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::layout::Rect;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn hk(c: KeyCode, tab: ActiveTab) -> KeyAction {
        handle_key(press(c), tab, &Modal::None, ChatKeyCtx::default())
    }

    #[test]
    fn q_quits_esc_does_not() {
        assert_eq!(hk(KeyCode::Char('q'), ActiveTab::Home), KeyAction::Quit);
        // P4: Esc opens the main menu (it never quits).
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Observe), KeyAction::OpenMenu);
    }

    #[test]
    fn q_quits_menu_palette_and_options_too() {
        // Menu/Palette/Options used to have no `q` arm at all, silently
        // swallowing the key instead of quitting like every other modal.
        let with_modal = |modal: &Modal| {
            handle_key(
                press(KeyCode::Char('q')),
                ActiveTab::Home,
                modal,
                ChatKeyCtx::default(),
            )
        };
        assert_eq!(with_modal(&Modal::Menu), KeyAction::Quit);
        assert_eq!(with_modal(&Modal::Palette), KeyAction::Quit);
        assert_eq!(with_modal(&Modal::Options), KeyAction::Quit);
    }

    #[test]
    fn quit_exits_immediately_when_nothing_is_serving() {
        // Regression guard (issue #145): the overwhelmingly common case —
        // nothing running — must stay a one-keystroke quit, no prompt.
        let mut s = AppState::new("t".into(), "default-dark".into());
        assert!(apply_action(&mut s, KeyAction::Quit));
        assert!(s.quit_confirm.is_none());
    }

    #[test]
    fn quit_opens_a_confirm_prompt_when_a_model_is_serving() {
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.instances.insert(
            "vllm-1".into(),
            Instance {
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        assert!(!apply_action(&mut s, KeyAction::Quit));
        assert!(s.quit_confirm.is_some());
    }

    #[test]
    fn quitting_from_the_esc_menu_opens_the_same_confirm_prompt() {
        // Regression: choosing "Quit" from the Esc main menu (Modal::Menu's
        // last row) used to `return true` directly, a second quit entry
        // point that completely bypassed the confirm-before-quit gate.
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.instances.insert(
            "vllm-1".into(),
            Instance {
                status: InstanceStatus::Running,
                ..Default::default()
            },
        );
        s.modal = Modal::Menu;
        s.menu_sel = 2; // the "Quit" row
        assert!(!apply_action(&mut s, KeyAction::MenuActivate));
        assert!(s.quit_confirm.is_some());
    }

    #[test]
    fn quit_confirm_key_approves_denies_and_cancels() {
        use crate::ui::approval::ApprovalVerdict;
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.open_quit_confirm();
        assert_eq!(
            s.on_quit_confirm_key(KeyCode::Char('y')),
            Some(ApprovalVerdict::Approve)
        );
        s.open_quit_confirm();
        assert_eq!(
            s.on_quit_confirm_key(KeyCode::Char('n')),
            Some(ApprovalVerdict::Deny)
        );
        s.open_quit_confirm();
        assert_eq!(
            s.on_quit_confirm_key(KeyCode::Esc),
            Some(ApprovalVerdict::Cancel)
        );
    }

    #[test]
    fn chat_esc_then_q_still_quits_via_the_menu() {
        // A terminal that decodes "Alt+q" as a bare Esc followed by a plain
        // `q` (rather than a single Alt-modified KeyEvent) used to quit
        // immediately on Chat, because Esc was a no-op there and `q` fell
        // through to the global `Quit` arm. This PR makes Esc open the main
        // menu on Chat too, so the second event now needs Menu's own `q`
        // arm (added above) to still reach `Quit` instead of being
        // swallowed by the menu.
        let ctx = ChatKeyCtx {
            consent: ChatConsent::Accepted,
            focused: false,
            ..Default::default()
        };
        let after_esc = handle_key(press(KeyCode::Esc), ActiveTab::Chat, &Modal::None, ctx);
        assert_eq!(after_esc, KeyAction::OpenMenu);
        let after_q = handle_key(
            press(KeyCode::Char('q')),
            ActiveTab::Chat,
            &Modal::Menu,
            ctx,
        );
        assert_eq!(after_q, KeyAction::Quit);
    }

    #[test]
    fn tab_cycles_forward_and_wraps() {
        // 5-tab IA: Home → ROCm → Serving → Observe → Chat → Home.
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Rocm)
        );
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Serving),
            KeyAction::SwitchTab(ActiveTab::Observe)
        );
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Observe),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        // Chat wraps back to Home.
        assert_eq!(
            hk(KeyCode::Tab, ActiveTab::Chat),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
    }

    #[test]
    fn action_tab_arrows_and_enter_drive_focus() {
        // → steps into the detail pane, ← steps back, Enter is focus-aware.
        assert_eq!(
            hk(KeyCode::Right, ActiveTab::Rocm),
            KeyAction::PaneFocusDetail
        );
        assert_eq!(
            hk(KeyCode::Left, ActiveTab::Rocm),
            KeyAction::PaneFocusActions
        );
        assert_eq!(hk(KeyCode::Enter, ActiveTab::Rocm), KeyAction::PaneActivate);
        // Arrows are inert on other tabs (no focus model there).
        assert_eq!(hk(KeyCode::Right, ActiveTab::Observe), KeyAction::Nothing);
        // Enter elsewhere keeps its detail-modal meaning.
        assert_eq!(
            hk(KeyCode::Enter, ActiveTab::Observe),
            KeyAction::OpenDetail
        );
    }

    #[test]
    fn action_activate_is_two_step_list_then_open() {
        // Serving verb 0 = "Serve a model" → OpenServeWizard.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Serving;
        s.serving_sel = 0;
        assert_eq!(s.pane_focus, PaneFocus::Actions);
        // First activate steps into the detail pane; no overlay yet.
        apply_action(&mut s, KeyAction::PaneActivate);
        assert_eq!(s.pane_focus, PaneFocus::Detail);
        assert!(s.serve_wizard.is_none(), "must not open before stepping in");
        // Second activate opens the operation's manager.
        apply_action(&mut s, KeyAction::PaneActivate);
        assert!(
            s.serve_wizard.is_some(),
            "detail-focus Enter opens the manager"
        );
        // ROCm verb 2 = "Diagnose (doctor)" → OpenExamine (the other mapping).
        let mut r = AppState::new("t".into(), "default-dark".into());
        r.active_tab = ActiveTab::Rocm;
        r.rocm_sel = 2;
        r.pane_focus = PaneFocus::Detail;
        apply_action(&mut r, KeyAction::PaneActivate);
        assert!(
            r.examine_manager.is_some(),
            "ROCm Diagnose opens the doctor"
        );
    }

    #[test]
    fn action_focus_resets_on_move_and_tab_switch() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::Move(1));
        assert_eq!(s.pane_focus, PaneFocus::Actions, "Move snaps back to list");
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::SwitchTab(ActiveTab::Home));
        assert_eq!(s.pane_focus, PaneFocus::Actions, "tab switch resets focus");
    }

    #[test]
    fn action_esc_backs_out_of_detail_then_opens_menu() {
        // Esc on Action is intercepted (not the global OpenMenu) so it can back
        // out of the detail pane first.
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Rocm), KeyAction::PaneEscape);
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::PaneEscape);
        assert_eq!(s.pane_focus, PaneFocus::Actions, "first Esc → list");
        assert_eq!(s.modal, Modal::None, "first Esc does not open the menu");
        apply_action(&mut s, KeyAction::PaneEscape);
        assert_eq!(s.modal, Modal::Menu, "second Esc opens the menu");
    }

    #[test]
    fn action_select_sets_verb_and_parks_on_list() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.pane_focus = PaneFocus::Detail;
        apply_action(&mut s, KeyAction::PaneSelect(2));
        assert_eq!(s.rocm_sel, 2);
        assert_eq!(s.pane_focus, PaneFocus::Actions);
        // Out-of-range clamps rather than panicking.
        apply_action(&mut s, KeyAction::PaneSelect(999));
        assert!(s.rocm_sel < crate::ui::tabs::rocm::VERB_COUNT);
    }

    #[test]
    fn inline_manager_opens_in_detail_then_backs_out() {
        // Activating a ROCm verb opens its manager inline (focus stays in
        // Details); `←`/Esc backs out — closing the manager and returning focus
        // to the Actions list. This mirrors the event-loop back-out arm.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Rocm;
        s.rocm_sel = 0; // Set up / Install ROCm → OpenInstall
        apply_action(&mut s, KeyAction::PaneActivate); // → Details
        assert_eq!(s.pane_focus, PaneFocus::Detail);
        assert!(!s.has_open_overlay(), "no manager before second activate");
        apply_action(&mut s, KeyAction::PaneActivate); // opens install_manager
        assert!(s.install_manager.is_some(), "verb opens its manager inline");
        assert!(s.has_open_overlay());

        // Esc backs out on a domain tab while a manager is open.
        assert!(s.should_pane_back_out(crossterm::event::KeyCode::Esc));
        // `←` is left to the manager (it may cycle options), not a back-out.
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Left));
        // A normal key does not back out (routes to the manager instead).
        assert!(!s.should_pane_back_out(crossterm::event::KeyCode::Char('j')));

        // The event-loop arm closes the manager + parks focus on Actions.
        s.close_overlays();
        s.pane_focus = PaneFocus::Actions;
        assert!(!s.has_open_overlay(), "back-out closed the inline manager");
        assert_eq!(s.pane_focus, PaneFocus::Actions);
    }

    #[test]
    fn console_scroll_delta_maps_nav_keys_only() {
        use crossterm::event::KeyCode;
        assert_eq!(console_scroll_delta(KeyCode::PageDown), Some((10, 0)));
        assert_eq!(console_scroll_delta(KeyCode::PageUp), Some((-10, 0)));
        assert_eq!(console_scroll_delta(KeyCode::Down), Some((1, 0)));
        assert_eq!(console_scroll_delta(KeyCode::Right), Some((0, 4)));
        // Console action keys are NOT scroll keys (they reach on_console_key).
        assert_eq!(console_scroll_delta(KeyCode::Esc), None);
        assert_eq!(console_scroll_delta(KeyCode::Enter), None);
        assert_eq!(console_scroll_delta(KeyCode::Char('q')), None);
    }

    #[test]
    fn back_tab_and_shift_tab_both_cycle_backward() {
        // prev(Observe) = Serving in the 5-tab IA.
        assert_eq!(
            hk(KeyCode::BackTab, ActiveTab::Observe),
            KeyAction::SwitchTab(ActiveTab::Serving)
        );
        // Home's previous tab is Chat (the last tab).
        assert_eq!(
            hk(KeyCode::BackTab, ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        let shift_tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT);
        assert_eq!(
            handle_key(
                shift_tab,
                ActiveTab::Rocm,
                &Modal::None,
                ChatKeyCtx::default()
            ),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
    }

    #[test]
    fn number_keys_jump_to_tab() {
        // 5-tab: '1'→Home, '2'→ROCm, '3'→Serving, '4'→Observe, '5'→Chat.
        assert_eq!(
            hk(KeyCode::Char('1'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
        assert_eq!(
            hk(KeyCode::Char('2'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Rocm)
        );
        assert_eq!(
            hk(KeyCode::Char('3'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Serving)
        );
        assert_eq!(
            hk(KeyCode::Char('4'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Observe)
        );
        // `5` reaches the Chat tab (digit guard widened to '1'..='5').
        assert_eq!(
            hk(KeyCode::Char('5'), ActiveTab::Home),
            KeyAction::SwitchTab(ActiveTab::Chat)
        );
        assert_eq!(hk(KeyCode::Char('6'), ActiveTab::Home), KeyAction::Nothing);
    }

    #[test]
    fn release_events_are_ignored() {
        let release = KeyEvent::new_with_kind(
            KeyCode::Char('q'),
            KeyModifiers::NONE,
            KeyEventKind::Release,
        );
        assert_eq!(
            handle_key(
                release,
                ActiveTab::Home,
                &Modal::None,
                ChatKeyCtx::default()
            ),
            KeyAction::Nothing
        );
    }

    #[test]
    fn only_press_key_events_are_actionable() {
        // The event loop gates overlay dispatch on this predicate so a single
        // keystroke isn't processed twice by an overlay's `on_key` (Release /
        // Repeat echoes on Windows Terminal / ConPTY / kitty keyboard). The
        // double-fire re-opened the serve wizard's model picker on Enter instead
        // of choosing — this pins Press-only routing.
        assert!(is_actionable_key(KeyEventKind::Press));
        assert!(!is_actionable_key(KeyEventKind::Release));
        assert!(!is_actionable_key(KeyEventKind::Repeat));
    }

    #[test]
    fn jk_arrows_and_g_drive_selection() {
        assert_eq!(
            hk(KeyCode::Char('j'), ActiveTab::Observe),
            KeyAction::Move(1)
        );
        assert_eq!(
            hk(KeyCode::Char('k'), ActiveTab::Observe),
            KeyAction::Move(-1)
        );
        assert_eq!(hk(KeyCode::Down, ActiveTab::Rocm), KeyAction::Move(1));
        assert_eq!(hk(KeyCode::Up, ActiveTab::Rocm), KeyAction::Move(-1));
        assert_eq!(
            hk(KeyCode::Char('g'), ActiveTab::Observe),
            KeyAction::SelectFirst
        );
        assert_eq!(
            hk(KeyCode::Char('G'), ActiveTab::Observe),
            KeyAction::SelectLast
        );
        assert_eq!(
            hk(KeyCode::Enter, ActiveTab::Observe),
            KeyAction::OpenDetail
        );
    }

    #[test]
    fn operational_open_keys_are_tab_scoped() {
        // `s` opens services only on Observe; Nothing elsewhere.
        assert_eq!(
            hk(KeyCode::Char('s'), ActiveTab::Observe),
            KeyAction::OpenServices
        );
        assert_eq!(hk(KeyCode::Char('s'), ActiveTab::Home), KeyAction::Nothing);
        // The letter hotkeys fire ONLY on Observe now — quick jumps into the
        // managers. They open the matching overlay via the seam.
        assert_eq!(
            hk(KeyCode::Char('w'), ActiveTab::Observe),
            KeyAction::OpenServeWizard
        );
        assert_eq!(
            hk(KeyCode::Char('e'), ActiveTab::Observe),
            KeyAction::OpenEngineManager
        );
        assert_eq!(
            hk(KeyCode::Char('d'), ActiveTab::Observe),
            KeyAction::OpenExamine
        );
        assert_eq!(
            hk(KeyCode::Char('i'), ActiveTab::Observe),
            KeyAction::OpenInstall
        );
        // Retired on the domain tabs: the Actions list is the single path there,
        // so the letter hotkeys are inert on ROCm/Serving (and Home/Chat).
        for c in ['w', 'e', 'd', 'u', 'i', 'l', 'r', 'n', 'a', 'c', 'p', 's'] {
            assert_eq!(
                hk(KeyCode::Char(c), ActiveTab::Rocm),
                KeyAction::Nothing,
                "key {c} must be retired on the ROCm tab"
            );
            assert_eq!(
                hk(KeyCode::Char(c), ActiveTab::Serving),
                KeyAction::Nothing,
                "key {c} must be retired on the Serving tab"
            );
        }
        assert_eq!(hk(KeyCode::Char('w'), ActiveTab::Home), KeyAction::Nothing);
        // On the Chat tab none of these open an overlay. `i` means insert mode.
        for c in ['w', 'e', 'd', 'u', 'l'] {
            assert_eq!(
                hk(KeyCode::Char(c), ActiveTab::Chat),
                KeyAction::Nothing,
                "key {c} must not open an overlay from Chat"
            );
        }
        assert_eq!(
            hk(KeyCode::Char('i'), ActiveTab::Chat),
            KeyAction::ChatFocus,
            "i is chat-insert on Chat, never OpenInstall"
        );
    }

    #[test]
    fn opening_an_overlay_closes_the_others() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::OpenServices);
        assert!(s.services.is_some() && s.serve_wizard.is_none() && s.engine_manager.is_none());
        // Opening another overlay (defensive path) clears the prior one.
        apply_action(&mut s, KeyAction::OpenServeWizard);
        assert!(s.serve_wizard.is_some() && s.services.is_none() && s.engine_manager.is_none());
        apply_action(&mut s, KeyAction::OpenEngineManager);
        assert!(s.engine_manager.is_some() && s.services.is_none() && s.serve_wizard.is_none());
        // Wave 2/3 overlays join the mutual-exclusion set.
        apply_action(&mut s, KeyAction::OpenExamine);
        assert!(s.examine_manager.is_some() && s.engine_manager.is_none());
        apply_action(&mut s, KeyAction::OpenUpdate);
        assert!(s.update_manager.is_some() && s.examine_manager.is_none());
        apply_action(&mut s, KeyAction::OpenInstall);
        assert!(s.install_manager.is_some() && s.update_manager.is_none());
        apply_action(&mut s, KeyAction::OpenLogs);
        assert!(s.logs_view.is_some() && s.install_manager.is_none());
        apply_action(&mut s, KeyAction::OpenRuntimes);
        assert!(s.runtime_manager.is_some() && s.logs_view.is_none());
        apply_action(&mut s, KeyAction::OpenOnboarding);
        assert!(s.onboarding.is_some() && s.runtime_manager.is_none());
        apply_action(&mut s, KeyAction::OpenAutomations);
        assert!(s.automations_manager.is_some() && s.onboarding.is_none());
        apply_action(&mut s, KeyAction::OpenCommand);
        assert!(s.command_screen.is_some() && s.automations_manager.is_none());
        apply_action(&mut s, KeyAction::OpenConfig);
        assert!(s.config_manager.is_some() && s.command_screen.is_none());
        // T13: OpenBenchRun joins the mutual-exclusion set.
        apply_action(&mut s, KeyAction::OpenBenchRun);
        assert!(s.bench_run.is_some() && s.config_manager.is_none());
    }

    #[test]
    fn esc_opens_menu_when_idle_on_any_tab() {
        // Idle tabs: Esc opens the btop main menu, Chat included when unfocused
        // (Chat-focused Esc is handled by the short-circuit above this match).
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Home), KeyAction::OpenMenu);
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Observe), KeyAction::OpenMenu);
        assert_eq!(hk(KeyCode::Esc, ActiveTab::Chat), KeyAction::OpenMenu);
        // While an overlay modal owns the screen, Esc closes it (not OpenMenu).
        assert_eq!(
            handle_key(
                press(KeyCode::Esc),
                ActiveTab::Home,
                &Modal::Menu,
                ChatKeyCtx::default()
            ),
            KeyAction::CloseModal
        );
        assert_eq!(
            handle_key(
                press(KeyCode::Esc),
                ActiveTab::Home,
                &Modal::Options,
                ChatKeyCtx::default()
            ),
            KeyAction::CloseModal
        );
    }

    #[test]
    fn colon_opens_command_palette() {
        assert_eq!(
            hk(KeyCode::Char(':'), ActiveTab::Home),
            KeyAction::OpenPalette
        );
    }

    #[test]
    fn menu_navigation_and_activation() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::OpenMenu);
        assert_eq!(s.modal, Modal::Menu);
        // ↓ from Options(0) → Help(1); activate opens the global help.
        apply_action(&mut s, KeyAction::MenuMove(1));
        assert_eq!(s.menu_sel, 1);
        apply_action(&mut s, KeyAction::MenuActivate);
        assert_eq!(s.modal, Modal::GlobalHelp);
        // Menu → Options activation opens the Options panel.
        apply_action(&mut s, KeyAction::OpenMenu);
        apply_action(&mut s, KeyAction::MenuActivate); // sel 0 = Options
        assert_eq!(s.modal, Modal::Options);
        // Options tab cycles and wraps.
        apply_action(&mut s, KeyAction::OptionsTab(-1));
        assert_eq!(s.options_tab, crate::ui::modal::OPTIONS_TABS.len() - 1);
    }

    #[test]
    fn palette_activation_switches_tab() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s, KeyAction::OpenPalette);
        apply_action(&mut s, KeyAction::MenuMove(3)); // Home→ROCm→Serving→Observe
        apply_action(&mut s, KeyAction::MenuActivate);
        assert_eq!(s.active_tab, ActiveTab::Observe);
        assert_eq!(s.modal, Modal::None);
    }

    #[test]
    fn question_mark_toggles_help() {
        assert_eq!(
            hk(KeyCode::Char('?'), ActiveTab::Home),
            KeyAction::ToggleHelp
        );
    }

    #[test]
    fn t_opens_theme_picker() {
        assert_eq!(
            hk(KeyCode::Char('t'), ActiveTab::Home),
            KeyAction::OpenThemePicker
        );
    }

    #[test]
    fn theme_picker_absorbs_navigation_keys() {
        let with_picker = |c| {
            handle_key(
                press(c),
                ActiveTab::Home,
                &Modal::ThemePicker,
                ChatKeyCtx::default(),
            )
        };
        assert_eq!(with_picker(KeyCode::Char('j')), KeyAction::Move(1));
        assert_eq!(with_picker(KeyCode::Char('k')), KeyAction::Move(-1));
        assert_eq!(with_picker(KeyCode::Enter), KeyAction::ApplyThemePick);
        assert_eq!(with_picker(KeyCode::Esc), KeyAction::CloseModal);
        assert_eq!(with_picker(KeyCode::Char('t')), KeyAction::CloseModal);
        assert_eq!(with_picker(KeyCode::Char('q')), KeyAction::Quit);
        assert_eq!(with_picker(KeyCode::Char('1')), KeyAction::Nothing);
    }

    #[test]
    fn tab_bar_hit_matches_per_chip_extents() {
        // 5-tab layout: Home 0..10, ROCm 11..21, Serving 22..35, Observe 36..49,
        // Chat 50..60.
        let bar = Rect::new(0, 0, 80, 1);
        assert_eq!(tab_bar_hit(bar, 5, 0), Some(ActiveTab::Home));
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Rocm));
        assert_eq!(tab_bar_hit(bar, 28, 0), Some(ActiveTab::Serving));
        assert_eq!(tab_bar_hit(bar, 42, 0), Some(ActiveTab::Observe));
        assert_eq!(tab_bar_hit(bar, 55, 0), Some(ActiveTab::Chat));
        // Separator gap between Home (ends 10 excl.) and ROCm (starts 11).
        assert_eq!(tab_bar_hit(bar, 10, 0), None);
        // Wrong row.
        assert_eq!(tab_bar_hit(bar, 5, 2), None);
    }

    #[test]
    fn tab_bar_hit_skips_chips_that_overflow_a_narrow_bar() {
        // Bar can only fit the first two chips (ROCm ends at 21).
        let bar = Rect::new(0, 0, 25, 1);
        assert_eq!(tab_bar_hit(bar, 5, 0), Some(ActiveTab::Home));
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Rocm));
        // Serving chip would be at 22..35 — overflows the 25-wide bar → None.
        assert_eq!(tab_bar_hit(bar, 28, 0), None);
    }

    #[test]
    fn tab_bar_hit_honors_x_offset() {
        // Bar offset 10 columns to the right: Home chip now spans 10..19.
        let bar = Rect::new(10, 0, 80, 1);
        assert_eq!(tab_bar_hit(bar, 15, 0), Some(ActiveTab::Home));
        // Absolute x=5 is left of the offset bar.
        assert_eq!(tab_bar_hit(bar, 5, 0), None);
    }

    #[test]
    fn handle_mouse_routes_scroll_by_modal_and_tab() {
        let scroll_down = MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        // No modal, non-interactive tab → Nothing
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Home),
            KeyAction::Nothing
        );
        // No modal, Observe → Move by ONE (drives the instances selection)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Observe),
            KeyAction::Move(1)
        );
        // Detail modal → ScrollModal by one line
        assert_eq!(
            handle_mouse(scroll_down, &Modal::Detail, ActiveTab::Observe),
            KeyAction::ScrollModal(1)
        );
        // Help / GlobalHelp popups are sized to fit their content — nothing to
        // scroll, so the wheel is a no-op there.
        assert_eq!(
            handle_mouse(scroll_down, &Modal::Help, ActiveTab::Home),
            KeyAction::Nothing
        );
        assert_eq!(
            handle_mouse(scroll_down, &Modal::GlobalHelp, ActiveTab::Home),
            KeyAction::Nothing
        );
        // ThemePicker → Move (drives picker cursor)
        assert_eq!(
            handle_mouse(scroll_down, &Modal::ThemePicker, ActiveTab::Home),
            KeyAction::Move(1)
        );
        // Domain-tab scroll is NOT routed here (resolve_mouse owns it) → Nothing.
        assert_eq!(
            handle_mouse(scroll_down, &Modal::None, ActiveTab::Rocm),
            KeyAction::Nothing
        );
    }

    #[test]
    fn detail_modal_j_k_emit_scroll() {
        let with_detail = |c| {
            handle_key(
                press(c),
                ActiveTab::Observe,
                &Modal::Detail,
                ChatKeyCtx::default(),
            )
        };
        assert_eq!(with_detail(KeyCode::Char('j')), KeyAction::ScrollModal(1));
        assert_eq!(with_detail(KeyCode::Char('k')), KeyAction::ScrollModal(-1));
        assert_eq!(with_detail(KeyCode::PageDown), KeyAction::ScrollModal(10));
        assert_eq!(
            with_detail(KeyCode::Char('g')),
            KeyAction::ScrollModal(i16::MIN)
        );
        assert_eq!(
            with_detail(KeyCode::Char('G')),
            KeyAction::ScrollModal(i16::MAX)
        );
        assert_eq!(with_detail(KeyCode::Esc), KeyAction::CloseModal);
    }

    #[test]
    fn help_modal_absorbs_navigation() {
        let with_help = |c| {
            handle_key(
                press(c),
                ActiveTab::Observe,
                &Modal::Help,
                ChatKeyCtx::default(),
            )
        };
        // The popup is sized to fit its content, so navigation keys are inert.
        assert_eq!(with_help(KeyCode::Char('j')), KeyAction::Nothing);
        assert_eq!(with_help(KeyCode::Char('k')), KeyAction::Nothing);
        assert_eq!(with_help(KeyCode::Tab), KeyAction::Nothing);
        assert_eq!(with_help(KeyCode::Esc), KeyAction::CloseModal);
        assert_eq!(with_help(KeyCode::Enter), KeyAction::CloseModal);
        assert_eq!(with_help(KeyCode::Char('q')), KeyAction::Quit);
    }

    #[test]
    fn global_help_modal_absorbs_navigation() {
        let with_global_help = |c| {
            handle_key(
                press(c),
                ActiveTab::Observe,
                &Modal::GlobalHelp,
                ChatKeyCtx::default(),
            )
        };
        assert_eq!(with_global_help(KeyCode::Char('j')), KeyAction::Nothing);
        assert_eq!(with_global_help(KeyCode::Char('k')), KeyAction::Nothing);
        assert_eq!(with_global_help(KeyCode::PageDown), KeyAction::Nothing);
        assert_eq!(with_global_help(KeyCode::Char('g')), KeyAction::Nothing);
        assert_eq!(with_global_help(KeyCode::Char('G')), KeyAction::Nothing);
        assert_eq!(with_global_help(KeyCode::Esc), KeyAction::CloseModal);
        assert_eq!(with_global_help(KeyCode::Char('q')), KeyAction::Quit);
    }

    #[test]
    fn bracket_keys_emit_replay_jump() {
        assert_eq!(
            hk(KeyCode::Char('['), ActiveTab::Home),
            KeyAction::ReplayJump(-10)
        );
        assert_eq!(
            hk(KeyCode::Char(']'), ActiveTab::Home),
            KeyAction::ReplayJump(10)
        );
        assert_eq!(
            hk(KeyCode::Char('{'), ActiveTab::Home),
            KeyAction::ReplayJump(-60)
        );
        assert_eq!(
            hk(KeyCode::Char('}'), ActiveTab::Home),
            KeyAction::ReplayJump(60)
        );
    }

    #[test]
    fn chat_insert_mode_captures_text_and_shortcircuits_hotkeys() {
        let accepted_focused = ChatKeyCtx {
            focused: true,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        };
        let focused = |c| handle_key(press(c), ActiveTab::Chat, &Modal::None, accepted_focused);
        // Printable chars become input, including ones that are global hotkeys.
        assert_eq!(focused(KeyCode::Char('h')), KeyAction::ChatInput('h'));
        assert_eq!(focused(KeyCode::Char('q')), KeyAction::ChatInput('q'));
        assert_eq!(focused(KeyCode::Char('5')), KeyAction::ChatInput('5'));
        assert_eq!(focused(KeyCode::Backspace), KeyAction::ChatBackspace);
        assert_eq!(focused(KeyCode::Enter), KeyAction::ChatSubmit);
        assert_eq!(focused(KeyCode::Esc), KeyAction::ChatBlur);

        // Accepted but NOT focused: `q` still quits and `i`/Enter enter insert mode.
        let accepted = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        };
        let unfocused = |c| handle_key(press(c), ActiveTab::Chat, &Modal::None, accepted);
        assert_eq!(unfocused(KeyCode::Char('q')), KeyAction::Quit);
        assert_eq!(unfocused(KeyCode::Char('i')), KeyAction::ChatFocus);
        assert_eq!(unfocused(KeyCode::Enter), KeyAction::ChatFocus);
        assert_eq!(
            unfocused(KeyCode::Char('1')),
            KeyAction::SwitchTab(ActiveTab::Home)
        );
    }

    #[test]
    fn chat_consent_gate_maps_keys_and_lets_globals_through() {
        let pending = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Pending,
            offer_pending: false,
        };
        let gate = |c| handle_key(press(c), ActiveTab::Chat, &Modal::None, pending);
        // y / Y / Enter accept; n / N decline.
        assert_eq!(gate(KeyCode::Char('y')), KeyAction::ChatConsentAccept);
        assert_eq!(gate(KeyCode::Enter), KeyAction::ChatConsentAccept);
        assert_eq!(gate(KeyCode::Char('n')), KeyAction::ChatConsentDecline);
        // Globals not trapped by the gate: q quits, digit switches tab.
        assert_eq!(gate(KeyCode::Char('q')), KeyAction::Quit);
        assert_eq!(
            gate(KeyCode::Char('2')),
            KeyAction::SwitchTab(ActiveTab::Rocm)
        );
    }

    #[test]
    fn chat_consent_accept_and_decline_transition_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // No endpoint → Unavailable; accept/decline are no-ops.
        s.set_chat_config(None, false);
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);
        apply_action(&mut s, KeyAction::ChatConsentAccept);
        assert_eq!(s.chat_consent, ChatConsent::Unavailable);

        // Endpoint present, no pre-consent → Pending.
        let llm = crate::llm::LlmConfig {
            base_url: "http://127.0.0.1:8000".into(),
            model: "m".into(),
            api_key: None,
            auth_header: None,
        };
        s.set_chat_config(Some(llm.clone()), false);
        assert_eq!(s.chat_consent, ChatConsent::Pending);
        // Accept → Accepted + focused.
        apply_action(&mut s, KeyAction::ChatConsentAccept);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert!(s.chat_focused);
        // Decline → Declined + unfocused.
        apply_action(&mut s, KeyAction::ChatConsentDecline);
        assert_eq!(s.chat_consent, ChatConsent::Declined);
        assert!(!s.chat_focused);

        // Pre-consent → Accepted immediately.
        s.set_chat_config(Some(llm), true);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
    }

    #[test]
    fn detect_offer_lifecycle_accept_switches_chat() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Chat;
        // Gateway-configured chat, pending consent.
        let gw = crate::llm::LlmConfig {
            base_url: "https://gw/OpenAI".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("k".into()),
            auth_header: Some("Ocp-Apim-Subscription-Key".into()),
        };
        s.set_chat_config(Some(gw), false);
        // Simulate a prior `/provider openai` so the realignment to Local on
        // accept is observable (Local is the default, so starting there would
        // make the assertion below tautological).
        s.active_provider = ChatProvider::Openai;

        // request_detect raises the dispatch edge + detecting flag.
        apply_action(&mut s, KeyAction::ChatDetect);
        assert!(s.chat_detecting && s.chat_detect_dispatch);

        // event_loop reports a detected local engine.
        let local = crate::llm::detected_llm_config("http://localhost:13305/v1", "Llama-3.2-3B");
        s.set_detect_result(Some(local.clone()));
        assert!(!s.chat_detecting);
        assert_eq!(s.chat_detect_offer.as_ref(), Some(&local));

        // Accept the offer → chat switches to the local endpoint + enabled.
        apply_action(&mut s, KeyAction::ChatDetectAccept);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert_eq!(s.chat_llm.as_ref(), Some(&local));
        assert!(s.chat_detect_offer.is_none());
        assert_eq!(
            s.chat_endpoint_rebuild,
            Some(ChatProvider::Openai),
            "accept raises the rebuild edge carrying the previous provider"
        );
        assert_eq!(s.active_provider, ChatProvider::Local);
    }

    #[test]
    fn detect_offer_dismiss_keeps_prior_config() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        let gw = crate::llm::LlmConfig {
            base_url: "https://gw/OpenAI".into(),
            model: "gpt-4o-mini".into(),
            api_key: None,
            auth_header: None,
        };
        s.set_chat_config(Some(gw.clone()), false);
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:8000/v1",
            "x",
        )));
        // Dismiss → offer gone, gateway config + Pending consent intact.
        apply_action(&mut s, KeyAction::ChatDetectDismiss);
        assert!(s.chat_detect_offer.is_none());
        assert_eq!(s.chat_llm.as_ref(), Some(&gw));
        assert_eq!(s.chat_consent, ChatConsent::Pending);
    }

    #[test]
    fn save_detect_offer_accepts_and_raises_persist_edge() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // Start off-Local so the realignment on accept is observable (Local is
        // the default provider; asserting it without this would be tautological).
        s.active_provider = ChatProvider::Openai;
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:13305/v1",
            "Llama-3.2-3B",
        )));
        apply_action(&mut s, KeyAction::ChatDetectSave);
        assert_eq!(s.chat_consent, ChatConsent::Accepted);
        assert!(s.chat_persist_dispatch, "save raises the persist edge");
        assert_eq!(
            s.chat_endpoint_rebuild,
            Some(ChatProvider::Openai),
            "save also raises the rebuild edge carrying the previous provider"
        );
        assert_eq!(s.active_provider, ChatProvider::Local);
        assert_eq!(
            s.chat_llm.as_ref().map(|c| c.base_url.as_str()),
            Some("http://localhost:13305/v1")
        );
        // No offer → save is a no-op (no edge).
        let mut s2 = AppState::new("t".into(), "default-dark".into());
        apply_action(&mut s2, KeyAction::ChatDetectSave);
        assert!(!s2.chat_persist_dispatch);
        assert!(s2.chat_endpoint_rebuild.is_none());
    }

    #[test]
    fn detect_key_available_on_gate_and_offer_keys_take_precedence() {
        // `d` triggers detect from the Unavailable empty-state.
        let unavail = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Unavailable,
            offer_pending: false,
        };
        assert_eq!(
            handle_key(
                press(KeyCode::Char('d')),
                ActiveTab::Chat,
                &Modal::None,
                unavail
            ),
            KeyAction::ChatDetect
        );
        // With an offer pending, y/n map to the offer (not consent).
        let offering = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Pending,
            offer_pending: true,
        };
        assert_eq!(
            handle_key(
                press(KeyCode::Char('y')),
                ActiveTab::Chat,
                &Modal::None,
                offering
            ),
            KeyAction::ChatDetectAccept
        );
        assert_eq!(
            handle_key(
                press(KeyCode::Char('n')),
                ActiveTab::Chat,
                &Modal::None,
                offering
            ),
            KeyAction::ChatDetectDismiss
        );
    }

    #[test]
    fn chat_input_actions_mutate_buffer() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Chat;
        s.chat_focused = true;
        apply_action(&mut s, KeyAction::ChatInput('h'));
        apply_action(&mut s, KeyAction::ChatInput('i'));
        assert_eq!(s.chat_input, "hi");
        apply_action(&mut s, KeyAction::ChatBackspace);
        assert_eq!(s.chat_input, "h");
        apply_action(&mut s, KeyAction::ChatBlur);
        assert!(!s.chat_focused);
        apply_action(&mut s, KeyAction::ChatFocus);
        assert!(s.chat_focused);
    }

    #[test]
    fn chat_submit_pushes_user_turn_and_raises_dispatch() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "what's GPU-2 doing?".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        // Only the user turn is pushed; the agent reply arrives async.
        assert_eq!(s.chat.len(), 1);
        assert_eq!(s.chat[0].role, ChatRole::User);
        assert_eq!(s.chat[0].content, "what's GPU-2 doing?");
        assert!(s.chat_input.is_empty());
        assert!(s.chat_sending, "submit marks the request in flight");
        assert!(s.chat_dispatch, "submit raises the spawn edge");
    }

    #[test]
    fn chat_submit_ignores_empty_input() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "   ".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat.is_empty());
        assert!(!s.chat_sending);
        assert!(!s.chat_dispatch);
    }

    #[test]
    fn chat_submit_ignored_while_request_in_flight() {
        // A second submit before the first reply lands must be a no-op — no
        // second user turn, no second spawn (prevents a racing double request).
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "first".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat_sending);
        assert_eq!(s.chat.len(), 1);
        s.chat_dispatch = false; // simulate event_loop consuming the edge
        s.chat_input = "second".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert_eq!(s.chat.len(), 1, "second submit ignored while in flight");
        assert!(!s.chat_dispatch, "no second dispatch edge raised");
        // After the reply clears the flag, submits work again.
        s.on_chat_reply("done".into());
        assert!(!s.chat_sending);
        s.chat_input = "third".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat_dispatch);
    }

    #[test]
    fn scroll_modal_action_reaches_scroll_instance_detail_for_detail_modal() {
        // Regression: `apply_action`'s ScrollModal dispatch only matched
        // `Modal::Help | Modal::GlobalHelp`, silently dropping the action for
        // `Modal::Detail` even though both `handle_key` and `handle_mouse`
        // emit `ScrollModal` for it (see `detail_modal_j_k_emit_scroll` /
        // `handle_mouse_routes_scroll_by_modal_and_tab`) and the instance
        // Detail modal's body (launch_args/env_vars) can genuinely overflow.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.modal = Modal::Detail;
        s.instance_detail_max_scroll = 10;
        apply_action(&mut s, KeyAction::ScrollModal(3));
        assert_eq!(
            s.instance_detail_scroll, 3,
            "Detail modal scrolls via apply_action"
        );
    }

    #[tokio::test]
    async fn chat_reply_path_appends_agent_turn_and_clears_sending() {
        // The wired ChatSubmit→reply path using the MockAgentClient (no LLM).
        let agent = crate::agent::MockAgentClient::new("GPU-2: 87% util, 71°C");
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "what's GPU-2 doing?".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert!(s.chat_sending);
        // Simulate event_loop: run the agent over the history, deliver the reply.
        let snapshot = s.state_snapshot();
        let reply = crate::agent::AgentClient::complete(&agent, &s.chat, snapshot)
            .await
            .expect("mock reply");
        s.on_chat_reply(reply);
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Agent);
        assert_eq!(s.chat.last().unwrap().content, "GPU-2: 87% util, 71°C");
        assert!(!s.chat_sending);
    }

    #[test]
    fn chat_input_handles_unicode_and_long_text() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_focused = true;
        // Multi-byte / emoji chars push as single chars, no panic.
        for c in "héllo 🚀 café ∑".chars() {
            apply_action(&mut s, KeyAction::ChatInput(c));
        }
        assert_eq!(s.chat_input, "héllo 🚀 café ∑");
        // Backspace removes the trailing multi-byte char correctly.
        apply_action(&mut s, KeyAction::ChatBackspace);
        assert_eq!(s.chat_input, "héllo 🚀 café ");
        // Very long input is accepted.
        for _ in 0..5000 {
            apply_action(&mut s, KeyAction::ChatInput('x'));
        }
        assert!(s.chat_input.len() > 5000);
        // Submitting unicode pushes one user turn, no panic.
        s.chat_input = "什么是 GPU-2?".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        assert_eq!(s.chat[0].content, "什么是 GPU-2?");
    }

    #[test]
    fn chat_scroll_clamps_and_updates_follow_state() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_max_scroll = 20;
        s.chat_scroll = 20;
        apply_action(&mut s, KeyAction::ChatScroll(-100));
        assert_eq!(s.chat_scroll, 0, "scroll clamps at top");
        assert!(!s.chat_follow, "scrolling above the bottom disables follow");
        apply_action(&mut s, KeyAction::ChatScroll(7));
        assert_eq!(s.chat_scroll, 7);
        assert!(!s.chat_follow);
        apply_action(&mut s, KeyAction::ChatScroll(100));
        assert_eq!(s.chat_scroll, 20, "scroll clamps at measured bottom");
        assert!(s.chat_follow, "scrolling to the bottom restores follow");
        // PageUp/PageDown map to ChatScroll on the Chat tab when accepted.
        let accepted = ChatKeyCtx {
            focused: false,
            consent: ChatConsent::Accepted,
            offer_pending: false,
        };
        assert_eq!(
            handle_key(
                press(KeyCode::PageDown),
                ActiveTab::Chat,
                &Modal::None,
                accepted
            ),
            KeyAction::ChatScroll(CHAT_SCROLL_STEP)
        );
        assert_eq!(
            handle_key(
                press(KeyCode::PageUp),
                ActiveTab::Chat,
                &Modal::None,
                accepted
            ),
            KeyAction::ChatScroll(-CHAT_SCROLL_STEP)
        );
    }

    #[tokio::test]
    async fn chat_error_path_appends_error_turn_no_panic() {
        let agent = crate::agent::MockAgentClient::failing();
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat_input = "hi".into();
        apply_action(&mut s, KeyAction::ChatSubmit);
        let snapshot = s.state_snapshot();
        let err = crate::agent::AgentClient::complete(&agent, &s.chat, snapshot)
            .await
            .unwrap_err();
        s.on_chat_error(err.to_string());
        assert_eq!(s.chat.last().unwrap().role, ChatRole::Error);
        assert!(!s.chat_sending);
    }
}
