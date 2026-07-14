// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat tab — a TUI-local conversation surface.
//!
//! Phase 1 is render-only with a local echo backend; later phases wire a Rig-built agent behind the
//! `AgentClient` trait. The transcript and input buffer are plain TUI state on
//! `AppState` (`rocm-dash-core` carries no chat types).

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::app::{AppState, ChatConsent, ChatRole};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// Block glyph used to mark the text-entry caret while focused.
const CURSOR_GLYPH: &str = "▋";

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    // Until the user consents to the detected endpoint, the tab shows a gate /
    // empty-state instead of the transcript+input surface.
    if state.chat_consent != ChatConsent::Accepted {
        draw_consent(f, area, state, theme);
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    draw_transcript(f, rows[0], state, theme);
    draw_input(f, rows[1], state, theme);
}

/// Render the consent prompt / empty-state, depending on detection + decision.
fn draw_consent(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let endpoint = state
        .chat_llm
        .as_ref()
        .map(|c| format!("{}  (model: {})", c.base_url, c.model));

    // Detect-flow states take over the gate: a probe in flight, then an offer.
    let detect_hint = Line::from(vec![
        Span::styled(
            "[d] ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("detect a local engine", Style::default().fg(theme.fg)),
    ]);

    let (title, lines): (&str, Vec<Line>) = if state.chat_detecting {
        (
            " Chat — detecting… ",
            vec![Line::from(Span::styled(
                "Probing for a local engine (Lemonade :13305 / vLLM :8000 / rocm serve :11435)…",
                Style::default().fg(theme.fg),
            ))],
        )
    } else if let Some(offer) = state.chat_detect_offer.as_ref() {
        (
            " Chat — use detected engine? ",
            vec![
                Line::from(Span::styled(
                    "Detected a local engine:",
                    Style::default().fg(theme.fg),
                )),
                Line::raw(""),
                Line::from(Span::styled(
                    format!("{}  (model: {})", offer.base_url, offer.model),
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::raw(""),
                Line::from(vec![
                    Span::styled(
                        "[y] ",
                        Style::default().fg(theme.ok).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("use now    ", Style::default().fg(theme.fg)),
                    Span::styled(
                        "[s] ",
                        Style::default()
                            .fg(theme.accent)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("use & save    ", Style::default().fg(theme.fg)),
                    Span::styled(
                        "[n] ",
                        Style::default().fg(theme.err).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("dismiss", Style::default().fg(theme.fg)),
                ]),
                Line::raw(""),
                Line::from(Span::styled(
                    "'use now' lasts this session; 'use & save' writes tui.chat_url to your config.",
                    Style::default().fg(theme.muted),
                )),
            ],
        )
    } else {
        // Normal consent gate, with a detect affordance + last-attempt message.
        let (title, mut lines) = consent_gate_lines(state, theme, endpoint);
        lines.push(Line::raw(""));
        lines.push(detect_hint);
        if let Some(m) = state.chat_detect_msg.as_ref() {
            lines.push(Line::from(Span::styled(
                m.clone(),
                Style::default().fg(theme.warn),
            )));
        }
        (title, lines)
    };

    let inner = panel::bento(f, area, Some(title), BoxRole::Neutral, false, theme);
    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

/// The base consent-gate content for the current consent state (without the
/// detect affordance, which the caller appends).
fn consent_gate_lines<'a>(
    state: &AppState,
    theme: &Theme,
    endpoint: Option<String>,
) -> (&'a str, Vec<Line<'a>>) {
    match state.chat_consent {
        ChatConsent::Pending => (
            " Chat — use this endpoint? ",
            vec![
                Line::from(Span::styled(
                    "An LLM endpoint was detected for chat:",
                    Style::default().fg(theme.fg),
                )),
                Line::raw(""),
                Line::from(Span::styled(
                    endpoint.unwrap_or_else(|| "(unknown)".into()),
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::raw(""),
                Line::from(vec![
                    Span::styled(
                        "[y] ",
                        Style::default().fg(theme.ok).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("use it    ", Style::default().fg(theme.fg)),
                    Span::styled(
                        "[n] ",
                        Style::default().fg(theme.err).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("not now", Style::default().fg(theme.fg)),
                ]),
                Line::raw(""),
                Line::from(Span::styled(
                    "Your request only leaves this machine after you accept.",
                    Style::default().fg(theme.muted),
                )),
            ],
        ),
        ChatConsent::Declined => (
            " Chat — disabled ",
            vec![
                Line::from(Span::styled(
                    "Chat is off. No requests will be sent.",
                    Style::default().fg(theme.fg),
                )),
                Line::raw(""),
                Line::from(vec![
                    Span::styled(
                        "[y] ",
                        Style::default().fg(theme.ok).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        endpoint.map_or_else(|| "enable".into(), |e| format!("enable {e}")),
                        Style::default().fg(theme.fg),
                    ),
                ]),
            ],
        ),
        // Unavailable (or Accepted, which never reaches here).
        _ => (
            " Chat — no endpoint detected ",
            vec![
                Line::from(Span::styled(
                    "No LLM endpoint was detected.",
                    Style::default().fg(theme.fg),
                )),
                Line::raw(""),
                Line::from(Span::styled(
                    "To enable chat, point it at an OpenAI-compatible endpoint:",
                    Style::default().fg(theme.muted),
                )),
                Line::from(Span::styled(
                    "  • --chat-url http://host:port  (or tui.chat_url in config)",
                    Style::default().fg(theme.muted),
                )),
                Line::from(Span::styled(
                    "  • OPENAI_BASE_URL in the environment",
                    Style::default().fg(theme.muted),
                )),
                Line::from(Span::styled(
                    "  • or run a local endpoint (vLLM :8000, rocm serve :11435)",
                    Style::default().fg(theme.muted),
                )),
            ],
        ),
    }
}

/// Render the scrolling transcript. Empty state shows an actionable hint.
fn draw_transcript(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let inner = panel::bento(f, area, Some("Chat"), BoxRole::Secondary, false, theme);

    let lines = if state.chat.is_empty() {
        vec![
            Line::from(Span::styled(
                "No messages yet.",
                Style::default().fg(theme.muted),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                "Press i (or Enter) to start typing, Enter to send, Esc to leave insert mode.",
                Style::default().fg(theme.muted),
            )),
        ]
    } else {
        transcript_lines(state, theme)
    };

    let p = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((state.chat_scroll, 0));
    f.render_widget(p, inner);
}

/// Pure transcript → styled lines mapping. Each turn becomes a role-prefixed,
/// role-colored line. Kept free of `Frame` so it can be unit-tested.
pub fn transcript_lines<'a>(state: &'a AppState, theme: &Theme) -> Vec<Line<'a>> {
    let mut lines: Vec<Line> = Vec::with_capacity(state.chat.len());
    for turn in &state.chat {
        let (prefix, color) = match turn.role {
            ChatRole::User => ("you  ", theme.accent),
            ChatRole::Agent => ("rocm ", theme.fg),
            ChatRole::Error => ("err  ", theme.err),
            ChatRole::System => ("··   ", theme.muted),
        };
        // Multi-line content (e.g. an answer plus a "⚙ via: …" Skill annotation)
        // renders one terminal line per segment; continuation lines are indented
        // to align under the first line's content.
        for (i, seg) in turn.content.split('\n').enumerate() {
            let lead = if i == 0 { prefix } else { "     " };
            lines.push(Line::from(vec![
                Span::styled(
                    lead,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(seg, Style::default().fg(color)),
            ]));
        }
    }
    lines
}

/// Render the single-row input line. While focused, a caret glyph trails the
/// buffer; otherwise a muted hint invites focus.
fn draw_input(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    // While a request is in flight, show a spinner and suppress the caret —
    // input is disabled until the reply or error turn lands.
    if state.chat_sending {
        let inner = panel::bento(
            f,
            area,
            Some("Message (sending…)"),
            BoxRole::Warning,
            false,
            theme,
        );
        let line = Line::from(Span::styled(
            "⠿ waiting for the agent…",
            Style::default().fg(theme.muted),
        ));
        f.render_widget(Paragraph::new(line), inner);
        return;
    }

    // Focused input is the primary actionable surface; idle reads muted.
    let (title, role) = if state.chat_focused {
        ("Message (insert)", BoxRole::Primary)
    } else {
        ("Message", BoxRole::Muted)
    };
    let inner = panel::bento(f, area, Some(title), role, false, theme);

    let line = if state.chat_focused {
        Line::from(vec![
            Span::styled(state.chat_input.as_str(), Style::default().fg(theme.fg)),
            Span::styled(CURSOR_GLYPH, Style::default().fg(theme.accent)),
        ])
    } else if state.chat_input.is_empty() {
        Line::from(Span::styled(
            "press i to type…",
            Style::default().fg(theme.muted),
        ))
    } else {
        Line::from(Span::styled(
            state.chat_input.as_str(),
            Style::default().fg(theme.muted),
        ))
    };

    f.render_widget(Paragraph::new(line), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{AppState, ChatTurn};

    #[test]
    fn transcript_lines_one_per_single_line_turn() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.chat.push(ChatTurn::user("hello"));
        s.chat.push(ChatTurn::agent("echo: hello"));
        s.chat.push(ChatTurn::error("boom"));
        let theme = s.theme;
        let lines = transcript_lines(&s, &theme);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn transcript_renders_skill_annotation_on_its_own_line() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        // An agent reply carrying a tool-call surfacing annotation.
        s.chat.push(ChatTurn::agent(
            "GPU-2 is at 87% util, 71°C.\n⚙ via: gpu_status",
        ));
        let theme = s.theme;
        let lines = transcript_lines(&s, &theme);
        // The annotation splits into its own line and is visible in the render.
        assert_eq!(lines.len(), 2, "answer + annotation render as two lines");
        assert!(
            format!("{lines:?}").contains("gpu_status"),
            "the fired Skill is surfaced in the transcript"
        );
    }

    #[test]
    fn draw_does_not_panic_across_consent_states() {
        use crate::app::ChatConsent;
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let render = |s: &AppState| {
            let backend = TestBackend::new(80, 24);
            let mut term = Terminal::new(backend).unwrap();
            term.draw(|f| draw(f, f.area(), s, &s.theme)).unwrap();
        };
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = crate::app::ActiveTab::Chat;

        // Unavailable empty-state.
        render(&s);

        let llm = crate::llm::LlmConfig {
            base_url: "http://127.0.0.1:8000".into(),
            model: "m".into(),
            api_key: None,
            auth_header: None,
        };
        // Pending consent prompt.
        s.set_chat_config(Some(llm.clone()), false);
        render(&s);
        // Declined.
        s.chat_consent = ChatConsent::Declined;
        render(&s);

        // Accepted: populated transcript + focused input.
        s.set_chat_config(Some(llm), true);
        s.chat_focused = true;
        s.chat_input = "what's GPU-2 doing?".into();
        s.chat.push(ChatTurn::user("hi"));
        s.chat.push(ChatTurn::agent("echo: hi"));
        render(&s);
    }

    fn render_str(s: &AppState) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, f.area(), s, &s.theme)).unwrap();
        let buf = term.backend().buffer().clone();
        buf.content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    #[test]
    fn gate_shows_detect_affordance_and_message() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = crate::app::ActiveTab::Chat;
        // Unavailable empty-state offers detection.
        assert!(render_str(&s).contains("detect a local engine"));
        // After a fruitless detect, the message shows.
        s.set_detect_result(None);
        assert!(render_str(&s).contains("no local engine found"));
    }

    #[test]
    fn detecting_state_renders_progress() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = crate::app::ActiveTab::Chat;
        s.request_detect();
        assert!(render_str(&s).contains("Probing for a local engine"));
    }

    #[test]
    fn offer_state_shows_endpoint_and_choices() {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = crate::app::ActiveTab::Chat;
        s.set_detect_result(Some(crate::llm::detected_llm_config(
            "http://localhost:13305/v1",
            "Llama-3.2-3B",
        )));
        let out = render_str(&s);
        assert!(out.contains("Detected a local engine"));
        assert!(out.contains("localhost:13305"));
        assert!(out.contains("Llama-3.2-3B"));
        assert!(out.contains("use now"));
        assert!(out.contains("use & save"));
    }
}
