// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared Actions + Details pane used by the ROCm and Serving tabs.
//!
//! Each domain tab is a left **Actions** bento (its locked verb list) plus a
//! right **`Details: <verb>`** bento that previews the selected operation. The
//! verb rows open EXISTING managers through the EXISTING execution seam
//! (`KeyAction::Open*` → `apply_action` → `RocmToolExecutor` / `ui/approval.rs`)
//! — no second approval path, no reimplementation of the managers.
//!
//! `rocm.rs` and `serving.rs` only declare their verb tables; all rendering,
//! hit-testing, and the list|detail geometry live here so the two tabs cannot
//! drift.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use rocm_dash_core::metrics::ObservationFreshness;

use crate::app::{AppState, KeyAction, PaneFocus};
use crate::ui::format;
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// One row in a domain tab's Actions list, with its inline Details preview.
///
/// Every field is grounded in the real manager flow (no invented data):
/// `summary` = one line on what it does, `steps` = the actual stages the manager
/// walks, `cmd` = the underlying CLI it drives. `action` is the seam verb the row
/// opens — `KeyAction::Nothing` marks a display-only row (Uninstall / Optimize),
/// which renders dimmed with `badge` and is a safe no-op when activated.
pub struct Verb {
    pub icon: &'static str,
    pub label: &'static str,
    pub action: KeyAction,
    pub summary: &'static str,
    pub steps: &'static [&'static str],
    pub cmd: &'static str,
    pub read_only: bool,
    /// Small status pill shown after a display-only row's label (e.g. `soon`).
    pub badge: Option<&'static str>,
}

impl Verb {
    /// A row is actionable when it maps to a real seam verb (not `Nothing`).
    const fn is_actionable(&self) -> bool {
        !matches!(self.action, KeyAction::Nothing)
    }
}

/// Resolve the seam action for the verb at `sel` (clamped). Used by
/// `apply_action` when Enter is pressed on a ROCm/Serving tab.
#[must_use]
pub fn verb_action(verbs: &[Verb], sel: usize) -> KeyAction {
    verbs.get(sel).map_or(KeyAction::Nothing, |v| v.action)
}

/// The 46/54 list|detail split. Single source so [`hit_test`] reconstructs
/// exactly what [`draw`] painted.
fn split_columns(area: Rect) -> std::rc::Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area)
}

/// The Actions list rect (left column) for a domain tab body `area`.
///
/// Used by the mouse-scroll router to gate wheel-driven selection moves to when
/// the pointer is actually over the Actions list (not the Details pane).
#[must_use]
pub fn actions_rect(area: Rect) -> Rect {
    split_columns(area)[0]
}

/// Draw the Actions list (left) + Details preview (right) for one domain tab.
pub fn draw(
    f: &mut Frame,
    area: Rect,
    list_title: &str,
    verbs: &[Verb],
    sel: usize,
    state: &AppState,
    theme: &Theme,
) {
    let focus = state.pane_focus;
    let cols = split_columns(area);
    draw_verb_list(f, cols[0], list_title, verbs, sel, focus, theme);
    draw_detail(f, cols[1], verbs, sel, focus, state, theme);
}

/// Map a left-click in a domain tab's body to an action.
///
/// A verb row selects that verb; anywhere in the detail pane activates (steps
/// in, or opens once focused) — mirroring the keyboard model so the visible
/// Start affordance is honest.
#[must_use]
pub fn hit_test(area: Rect, x: u16, y: u16, verb_count: usize) -> Option<KeyAction> {
    let cols = split_columns(area);
    let (list, detail) = (cols[0], cols[1]);
    let hit = |r: Rect| x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height;

    if hit(detail) {
        return Some(KeyAction::PaneActivate);
    }
    if hit(list) {
        // Reconstruct bento's inner rect (rounded border + adaptive padding).
        let inner = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .padding(panel::padding_for(list))
            .inner(list);
        if x >= inner.x && x < inner.x + inner.width && y >= inner.y {
            // Each verb occupies two rows (label + blank); blanks don't select.
            let row = (y - inner.y) as usize;
            let idx = row / 2;
            if row.is_multiple_of(2) && idx < verb_count {
                return Some(KeyAction::PaneSelect(idx));
            }
        }
    }
    None
}

fn draw_verb_list(
    f: &mut Frame,
    area: Rect,
    title: &str,
    verbs: &[Verb],
    sel: usize,
    focus: PaneFocus,
    theme: &Theme,
) {
    // The list owns focus until the user steps right into the detail pane.
    let list_focused = focus == PaneFocus::Actions;
    let inner = panel::bento(f, area, Some(title), BoxRole::Primary, list_focused, theme);
    if inner.height == 0 {
        return;
    }

    let sel = sel.min(verbs.len().saturating_sub(1));
    // When focus has moved into the detail pane, dim the list cursor so it reads
    // as "parked here" rather than active.
    let cursor_color = if list_focused {
        theme.accent
    } else {
        theme.muted
    };
    let mut lines: Vec<Line> = Vec::new();
    for (i, v) in verbs.iter().enumerate() {
        let selected = i == sel;
        let actionable = v.is_actionable();
        let cur = if selected { "▸ " } else { "  " };
        // Display-only rows read muted; actionable rows use the foreground.
        let label_color = if actionable { theme.fg } else { theme.muted };
        let label_style = if selected {
            Style::default()
                .fg(if actionable {
                    cursor_color
                } else {
                    theme.muted
                })
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(label_color)
        };
        let icon_color = if actionable {
            theme.accent_2
        } else {
            theme.muted
        };
        let mut spans = vec![
            Span::styled(cur, Style::default().fg(cursor_color)),
            Span::styled(format!("{}  ", v.icon), Style::default().fg(icon_color)),
            Span::styled(v.label, label_style),
        ];
        if let Some(badge) = v.badge {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(
                format!(" {badge} "),
                Style::default()
                    .bg(theme.warn)
                    .fg(theme.bg)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        lines.push(Line::from(spans));
        lines.push(Line::from(""));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(
    f: &mut Frame,
    area: Rect,
    verbs: &[Verb],
    sel: usize,
    focus: PaneFocus,
    state: &AppState,
    theme: &Theme,
) {
    let sel = sel.min(verbs.len().saturating_sub(1));
    let Some(v) = verbs.get(sel) else {
        return;
    };
    let actionable = v.is_actionable();
    // The detail pane lights up once the user steps into it with → or Enter.
    let focused = focus == PaneFocus::Detail;

    let title = format!("{} {}", v.icon, v.label);
    let inner = panel::bento(f, area, Some(&title), BoxRole::Secondary, focused, theme);
    if inner.height == 0 {
        return;
    }

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(v.summary, Style::default().fg(theme.fg))),
        Line::from(""),
    ];

    // State-aware block: what's actually installed / known / running for this
    // verb, so the pane reflects the system rather than fixed copy.
    let (live, live_any_held) = live_lines(v.action, state, theme);
    if !live.is_empty() {
        lines.extend(live);
        lines.push(Line::from(""));
    }

    lines.push(Line::from(Span::styled(
        "What you'll do",
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )));
    for (i, step) in v.steps.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {}. ", i + 1),
                Style::default().fg(theme.accent_2),
            ),
            Span::styled(*step, Style::default().fg(theme.fg)),
        ]));
    }
    lines.push(Line::from(""));

    if actionable {
        // Start affordance: filled accent when focused, outlined hint otherwise.
        let (start_style, hint) = if focused {
            (
                Style::default()
                    .fg(theme.bg)
                    .bg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                "   Enter — opens the guided manager",
            )
        } else {
            (
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                "   → or Enter to begin",
            )
        };
        lines.push(Line::from(vec![
            Span::styled("  ▸ Start ", start_style),
            Span::styled(hint, Style::default().fg(theme.muted)),
        ]));
        lines.push(Line::from(Span::styled(
            format!("  Runs: {}", v.cmd),
            Style::default().fg(theme.muted),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if v.read_only {
                "Read-only — nothing is changed."
            } else {
                "Mutating actions ask before they run. Default mode: ask."
            },
            Style::default().fg(theme.muted),
        )));
    } else {
        // Display-only row: no Start button, an honest "not available" note.
        lines.push(Line::from(Span::styled(
            v.badge.map_or_else(
                || "Not available yet.".to_string(),
                |b| format!("Marked “{b}” — planned, not yet built."),
            ),
            Style::default().fg(theme.muted),
        )));
    }

    // Reserve a dedicated, always-visible row for HELD_LEGEND rather than
    // appending it as just another line in `lines`: this whole block renders
    // through a clipped/wrapped `Paragraph`, and "What you'll do" plus the
    // step list can easily push a trailing line off-screen in a short
    // viewport even while the held marker earlier in `lines` is visible.
    let [content, legend] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(u16::from(live_any_held)),
        ])
        .areas(inner);
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), content);
    if live_any_held {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format::HELD_LEGEND,
                Style::default().fg(theme.muted),
            ))),
            legend,
        );
    }
}

/// State-aware "current status" lines for a verb, or empty when the verb has no
/// live view.
///
/// Surfaces what's actually installed (ROCm/engines), known (model catalog), or
/// running (instances) so the detail pane reflects the system — including the
/// running-instances view inline, without opening the services popup.
/// Returns the state-aware detail lines plus whether any displayed row's
/// value carries a held marker. The caller renders `HELD_LEGEND` in its own
/// guaranteed-visible row rather than this function appending it inline —
/// see the comment at the `draw_detail` render site.
fn live_lines(action: KeyAction, state: &AppState, theme: &Theme) -> (Vec<Line<'static>>, bool) {
    let head = |t: String| {
        Line::from(Span::styled(
            t,
            Style::default()
                .fg(theme.muted)
                .add_modifier(Modifier::BOLD),
        ))
    };
    let mark = |ok: bool| {
        if ok {
            Span::styled("✓ ", Style::default().fg(theme.ok))
        } else {
            Span::styled("· ", Style::default().fg(theme.muted))
        }
    };
    let info = || {
        state
            .latest
            .as_ref()
            .and_then(|s| s.gpu_system_info.as_ref())
    };

    match action {
        KeyAction::OpenInstall => {
            let mut lines = vec![head("Installed now".into())];
            let rocm = info().and_then(|i| i.rocm_version.clone());
            lines.push(Line::from(vec![
                mark(rocm.is_some()),
                Span::styled(
                    rocm.map_or_else(|| "ROCm — not detected".into(), |v| format!("ROCm {v}")),
                    Style::default().fg(theme.fg),
                ),
            ]));
            let driver = info().and_then(|i| i.driver_version.clone());
            lines.push(Line::from(vec![
                mark(driver.is_some()),
                Span::styled(
                    driver.map_or_else(|| "Driver —".into(), |v| format!("Driver {v}")),
                    Style::default().fg(theme.fg),
                ),
            ]));
            if !state.runtimes.is_empty() {
                lines.push(head(format!("{} runtime(s)", state.runtimes.len())));
                for rt in state.runtimes.iter().take(4) {
                    lines.push(Line::from(vec![
                        Span::styled(
                            if rt.active { "● " } else { "  " },
                            Style::default().fg(theme.ok),
                        ),
                        Span::styled(
                            format!("{} ({} {})", rt.key, rt.channel, rt.version),
                            Style::default().fg(theme.fg),
                        ),
                    ]));
                }
            }
            (lines, false)
        }
        KeyAction::OpenEngineManager => {
            let mut lines = vec![head("Engines".into())];
            // External endpoint evidence is live; managed installation evidence
            // is a separate launch-time Detect snapshot. GPU metadata carries
            // neither verdict and must never imply an absent engine.
            let external_running = state
                .instances
                .values()
                .any(|i| i.is_external_lemonade() && i.status.is_serving());
            let lemon = if external_running {
                "Lemonade — externally running"
            } else {
                match state.lemonade_managed_installed {
                    Some(true) => "Lemonade — managed runtime installed",
                    Some(false) => "Lemonade — managed runtime not installed",
                    None => "Lemonade — unknown, run `rocm engines list`",
                }
            };
            lines.push(Line::from(vec![
                mark(external_running || state.lemonade_managed_installed == Some(true)),
                Span::styled(lemon, Style::default().fg(theme.fg)),
            ]));
            if external_running {
                let managed = match state.lemonade_managed_installed {
                    Some(true) => "Managed runtime — installed",
                    Some(false) => "Managed runtime — not installed",
                    None => "Managed runtime — unknown, run `rocm engines list`",
                };
                lines.push(Line::from(Span::styled(
                    managed,
                    Style::default().fg(theme.muted),
                )));
            }
            // The daemon doesn't surface vLLM detection yet. `rocm engines list`
            // prints each engine's Detect result, so point there rather than
            // claiming a status we don't have.
            lines.push(Line::from(vec![
                Span::styled("· ", Style::default().fg(theme.muted)),
                Span::styled(
                    "vLLM — unknown, run `rocm engines list`",
                    Style::default().fg(theme.muted),
                ),
            ]));
            // The managed-runtime verdict is a launch-time snapshot: installing
            // from the Engines overlay does not update it. The test
            // `managed_install_verdict_is_not_changed_by_live_events` proves it.
            if state.lemonade_managed_installed.is_some() {
                lines.push(Line::from(Span::styled(
                    "Install status is read at startup. Restart the dashboard to refresh.",
                    Style::default().fg(theme.muted),
                )));
            }
            (lines, false)
        }
        KeyAction::OpenServeWizard => {
            let n = state.model_recipes.len();
            if n == 0 {
                return (Vec::new(), false);
            }
            let mut lines = vec![head(format!("Model catalog — {n} known"))];
            for r in state.model_recipes.iter().take(4) {
                lines.push(Line::from(vec![
                    Span::styled("• ", Style::default().fg(theme.accent_2)),
                    Span::styled(r.id.clone(), Style::default().fg(theme.fg)),
                ]));
            }
            if let Some(tail) = format::overflow_line(n, 4) {
                lines.push(Line::from(Span::styled(
                    tail,
                    Style::default().fg(theme.muted),
                )));
            }
            (lines, false)
        }
        KeyAction::OpenServices => {
            let running: Vec<_> = state
                .instances
                .values()
                .filter(|i| {
                    crate::ui::services_manager::is_listed_service(i) && i.status.is_serving()
                })
                .collect();
            let mut lines = vec![head(format!("Managed running — {}", running.len()))];
            let mut any_held = false;
            if running.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  no managed services running",
                    Style::default().fg(theme.muted),
                )));
            } else {
                for i in running.iter().take(5) {
                    // `gen_tps_compact` only ever prints HELD_MARKER for a
                    // finite `Some` gen_tps — mirror that condition here so
                    // held metadata on a `None`/non-finite value (rendered
                    // as "gen -" with no marker) can't add an unexplained
                    // legend.
                    any_held |= i.gen_tps.is_some_and(f64::is_finite)
                        && i.gen_tps_observation
                            .as_ref()
                            .is_some_and(|m| m.freshness == ObservationFreshness::Held);
                    let port = i.port.map_or_else(String::new, |p| format!(" :{p}"));
                    lines.push(Line::from(vec![
                        Span::styled("● ", Style::default().fg(theme.ok)),
                        Span::styled(i.model_name.clone(), Style::default().fg(theme.fg)),
                        Span::styled(port, Style::default().fg(theme.muted)),
                        Span::styled(
                            format!(
                                "  gen {}",
                                format::gen_tps_compact(i.gen_tps, i.gen_tps_observation.as_ref())
                            ),
                            Style::default().fg(theme.muted),
                        ),
                    ]));
                }
                if let Some(tail) = format::overflow_line(running.len(), 5) {
                    lines.push(Line::from(Span::styled(
                        tail,
                        Style::default().fg(theme.muted),
                    )));
                }
            }
            (lines, any_held)
        }
        _ => (Vec::new(), false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_text(state: &AppState) -> String {
        live_lines(KeyAction::OpenEngineManager, state, &Theme::default())
            .0
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn glyph_of(state: &AppState, needle: &str) -> String {
        let (lines, _) = live_lines(KeyAction::OpenEngineManager, state, &Theme::default());
        let line = lines
            .iter()
            .find(|l| l.to_string().contains(needle))
            .unwrap_or_else(|| panic!("no line containing {needle:?}"));
        line.spans[0].content.to_string()
    }

    #[test]
    fn lemonade_line_glyph_is_check_only_when_installed() {
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        state.lemonade_managed_installed = Some(true);
        assert_eq!(
            glyph_of(&state, "Lemonade — managed runtime installed"),
            "✓ "
        );
        state.lemonade_managed_installed = Some(false);
        assert_eq!(
            glyph_of(&state, "Lemonade — managed runtime not installed"),
            "· "
        );
        state.lemonade_managed_installed = None;
        assert_eq!(glyph_of(&state, "Lemonade — unknown"), "· ");
    }

    #[test]
    fn external_running_lemonade_line_has_check_glyph() {
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        state.instances.insert(
            "x".into(),
            Instance {
                managed: false,
                container_id: "x".into(),
                container_name: "lemonade".into(),
                status: InstanceStatus::Ready,
                ..Instance::default()
            },
        );
        assert_eq!(glyph_of(&state, "Lemonade — externally running"), "✓ ");
    }

    #[test]
    fn startup_snapshot_note_shown_only_with_a_verdict() {
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        assert!(!engine_text(&state).contains("read at startup"));
        state.lemonade_managed_installed = Some(false);
        let text = engine_text(&state);
        // This pins the wording only. That the status really is captured once
        // at startup is proved by
        // `managed_install_verdict_is_not_changed_by_live_events`.
        assert!(
            text.contains("Install status is read at startup. Restart the dashboard to refresh."),
            "{text}"
        );
        assert!(
            text.contains("vLLM — unknown, run `rocm engines list`"),
            "{text}"
        );
    }

    #[test]
    fn managed_install_verdict_is_not_changed_by_live_events() {
        // Proves the claim the pane's "read at startup" note makes: the verdict
        // is set once from the startup args (`event_loop.rs`) and no later
        // daemon event rewrites it, so the note is true.
        use rocm_dash_core::metrics::{Instance, InstanceStatus, Snapshot};
        use rocm_dash_core::protocol::Event;

        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        state.lemonade_managed_installed = Some(false);
        let managed = Instance {
            managed: true,
            container_id: "svc".into(),
            container_name: "lemonade".into(),
            status: InstanceStatus::Ready,
            ..Instance::default()
        };
        state.apply_event(Event::InstanceDiscovered(managed));
        state.apply_event(Event::Snapshot(Snapshot::default()));
        state.apply_event(Event::InstanceGone {
            container_id: "svc".into(),
        });
        assert_eq!(state.lemonade_managed_installed, Some(false));
        assert!(engine_text(&state).contains("Lemonade — managed runtime not installed"));
    }

    #[test]
    fn installed_stopped_managed_lemonade_is_installed_without_gpu_metadata() {
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        state.lemonade_managed_installed = Some(true);
        assert!(state.instances.is_empty());
        assert!(engine_text(&state).contains("Lemonade — managed runtime installed"));
        assert!(!engine_text(&state).contains("running"));
    }

    #[test]
    fn absent_managed_lemonade_is_reported_only_from_inventory() {
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        state.lemonade_managed_installed = Some(false);
        assert!(engine_text(&state).contains("Lemonade — managed runtime not installed"));
    }

    #[test]
    fn external_running_and_stopped_instances_do_not_invent_managed_installation() {
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        let id = "lemonade-localhost-13305-model-61".to_owned();
        state.instances.insert(
            id.clone(),
            Instance {
                container_id: id.clone(),
                container_name: "lemonade".into(),
                status: InstanceStatus::Ready,
                ..Instance::default()
            },
        );
        let text = engine_text(&state);
        assert!(text.contains("Lemonade — externally running"));
        assert!(text.contains("Managed runtime — unknown, run `rocm engines list`"));
        state.lemonade_managed_installed = Some(true);
        assert!(engine_text(&state).contains("Managed runtime — installed"));
        state.instances.get_mut(&id).unwrap().status = InstanceStatus::Stopped;
        assert!(engine_text(&state).contains("Lemonade — managed runtime installed"));
        assert!(!engine_text(&state).contains("externally running"));
    }

    #[test]
    fn external_lemonade_verdict_uses_ownership_not_id_prefix() {
        use rocm_dash_core::metrics::{Instance, InstanceStatus};
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        let id = "svc-ordinary-id".to_owned();
        state.instances.insert(
            id.clone(),
            Instance {
                managed: false,
                container_id: id.clone(),
                container_name: "lemonade".into(),
                status: InstanceStatus::Ready,
                ..Instance::default()
            },
        );
        assert!(engine_text(&state).contains("Lemonade — externally running"));
        let external = state.instances.remove(&id).unwrap();
        assert!(!engine_text(&state).contains("externally running"));
        state.instances.insert(
            "lemonade-looks-external".into(),
            Instance {
                managed: true,
                container_id: "lemonade-looks-external".into(),
                ..external
            },
        );
        assert!(!engine_text(&state).contains("externally running"));
    }

    fn services_text(state: &AppState) -> String {
        live_lines(KeyAction::OpenServices, state, &Theme::default())
            .0
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn serving(id: &str, managed: bool) -> rocm_dash_core::metrics::Instance {
        rocm_dash_core::metrics::Instance {
            managed,
            container_id: id.into(),
            model_name: format!("model-{id}"),
            status: rocm_dash_core::metrics::InstanceStatus::Ready,
            ..rocm_dash_core::metrics::Instance::default()
        }
    }

    #[test]
    fn services_preview_ignores_external_only_instances() {
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        for id in ["ext-a", "ext-b"] {
            state.instances.insert(id.into(), serving(id, false));
        }
        let text = services_text(&state);
        assert!(text.contains("Managed running — 0"), "{text}");
        assert!(text.contains("no managed services running"), "{text}");
        assert!(!text.contains("model-ext"), "{text}");
        // Same set the overlay would list.
        assert!(crate::ui::services_manager::service_rows(&state.instances).is_empty());
    }

    #[test]
    fn services_preview_counts_only_managed_instances() {
        let mut state = AppState::new("unix:/unused".into(), "dark".into());
        state.instances.insert("ext".into(), serving("ext", false));
        state.instances.insert("own".into(), serving("own", true));
        let text = services_text(&state);
        assert!(text.contains("Managed running — 1"), "{text}");
        assert!(text.contains("model-own"), "{text}");
        assert!(!text.contains("model-ext"), "{text}");
    }

    #[test]
    fn unknown_lemonade_installation_is_not_claimed_absent() {
        let state = AppState::new("unix:/unused".into(), "dark".into());
        let (lines, _) = live_lines(KeyAction::OpenEngineManager, &state, &Theme::default());
        let text = lines
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("Lemonade — unknown, run `rocm engines list`"),
            "{text}"
        );
        assert!(
            !text.contains("not installed"),
            "missing GPU metadata is not installation evidence"
        );
    }
}
