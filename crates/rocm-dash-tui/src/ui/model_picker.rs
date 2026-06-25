// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Model picker (Phase 3 Wave 1).
//!
//! A reusable, filterable list of model recipes — the sub-step the serve wizard
//! opens from its Model field so the user can pick a built-in recipe instead of
//! typing a model name. Selecting a recipe fills the model id and (when known)
//! pre-selects the recipe's preferred engine.
//!
//! Recipes are passed in as plain TUI-local [`ModelRecipeSummary`] values — the
//! bin (`apps/rocm`, which has `rocm-core`) adapts the full `ModelRecipeRecord`
//! registry into these summaries, so this layer needs no `rocm-core` dep
//! Navigation/selection is pure and testable.

use crossterm::event::KeyCode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// A flattened model-recipe entry for the picker. Mirrors the fields of
/// `rocm-core::ModelRecipeRecord` the picker needs, with no core dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRecipeSummary {
    /// Canonical model id (what `rocm serve` is invoked with).
    pub id: String,
    /// Aliases the recipe also matches (searchable).
    pub aliases: Vec<String>,
    /// Human task label (e.g. "chat", "embedding").
    pub task: String,
    /// Preferred serving engine, if the recipe declares one.
    pub preferred_engine: Option<String>,
}

/// Picker state: a filter query + a cursor into the filtered list.
#[derive(Debug, Clone, Default)]
pub struct ModelPicker {
    pub query: String,
    pub selected: usize,
}

/// The result of handling a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickerOutcome {
    /// Key consumed, still picking.
    None,
    /// The user chose a recipe. The caller owns it.
    Chosen(ModelRecipeSummary),
    /// The user dismissed the picker.
    Cancelled,
}

impl ModelPicker {
    /// The recipes matching the current query (case-insensitive substring over
    /// id + aliases). An empty query matches everything.
    pub fn filtered<'a>(&self, recipes: &'a [ModelRecipeSummary]) -> Vec<&'a ModelRecipeSummary> {
        if self.query.trim().is_empty() {
            return recipes.iter().collect();
        }
        let q = self.query.to_lowercase();
        recipes
            .iter()
            .filter(|r| {
                r.id.to_lowercase().contains(&q)
                    || r.aliases.iter().any(|a| a.to_lowercase().contains(&q))
            })
            .collect()
    }

    /// Pure key handler. Filesystem-free; only mutates the cursor/query.
    pub fn on_key(&mut self, key: KeyCode, recipes: &[ModelRecipeSummary]) -> PickerOutcome {
        let len = self.filtered(recipes).len();
        match key {
            KeyCode::Esc => PickerOutcome::Cancelled,
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                PickerOutcome::None
            }
            KeyCode::Down => {
                if len > 0 {
                    self.selected = (self.selected + 1).min(len - 1);
                }
                PickerOutcome::None
            }
            KeyCode::Enter => self
                .filtered(recipes)
                .get(self.selected)
                .map_or(PickerOutcome::None, |r| PickerOutcome::Chosen((*r).clone())),
            KeyCode::Backspace => {
                self.query.pop();
                self.selected = 0;
                PickerOutcome::None
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                self.selected = 0;
                PickerOutcome::None
            }
            _ => PickerOutcome::None,
        }
    }
}

/// Render the picker over `area`.
pub fn draw_model_picker(
    f: &mut Frame,
    area: Rect,
    picker: &ModelPicker,
    recipes: &[ModelRecipeSummary],
    theme: &Theme,
) {
    let popup = centered_rect(76, 78, 110, 28, area);
    let inner = draw_popup_frame(f, popup, "Pick a model recipe", theme);
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

    let query_display = if picker.query.is_empty() {
        "(type to filter)".to_string()
    } else {
        picker.query.clone()
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("filter: ", Style::default().fg(theme.muted)),
            Span::styled(
                query_display,
                Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
            ),
        ])),
        rows[0],
    );

    let filtered = picker.filtered(recipes);
    if filtered.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No recipes match. Backspace to widen, or Esc to type a name directly.",
                Style::default().fg(theme.muted),
            ))),
            rows[1],
        );
    } else {
        let items: Vec<ListItem> = filtered
            .iter()
            .map(|r| {
                let eng = r.preferred_engine.as_deref().unwrap_or("—");
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<30}", trunc(&r.id, 30)),
                        Style::default().fg(theme.fg),
                    ),
                    Span::styled(
                        format!("{:<12}", trunc(&r.task, 12)),
                        Style::default().fg(theme.muted),
                    ),
                    Span::styled(format!("engine: {eng}"), Style::default().fg(theme.accent)),
                ]))
            })
            .collect();
        let mut ls = ListState::default();
        ls.select(Some(picker.selected.min(filtered.len().saturating_sub(1))));
        let list = List::new(items).highlight_style(
            Style::default()
                .bg(theme.surface_2)
                .add_modifier(Modifier::BOLD),
        );
        f.render_stateful_widget(list, rows[1], &mut ls);
    }

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "type filter · ↑↓ select · Enter choose · Esc cancel",
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

    fn recipes() -> Vec<ModelRecipeSummary> {
        vec![
            ModelRecipeSummary {
                id: "Qwen3-4B-Instruct".into(),
                aliases: vec!["qwen".into()],
                task: "chat".into(),
                preferred_engine: Some("lemonade".into()),
            },
            ModelRecipeSummary {
                id: "GLM-4".into(),
                aliases: vec!["glm".into()],
                task: "chat".into(),
                preferred_engine: Some("vllm".into()),
            },
            ModelRecipeSummary {
                id: "Llama-3.2-3B".into(),
                aliases: vec!["llama".into()],
                task: "chat".into(),
                preferred_engine: None,
            },
        ]
    }

    #[test]
    fn empty_query_matches_all() {
        let p = ModelPicker::default();
        assert_eq!(p.filtered(&recipes()).len(), 3);
    }

    #[test]
    fn filter_matches_id_and_alias_case_insensitive() {
        let rs = recipes();
        let mut p = ModelPicker {
            query: "QWEN".into(),
            ..Default::default()
        };
        let f = p.filtered(&rs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].id, "Qwen3-4B-Instruct");
        // Alias match.
        p.query = "glm".into();
        assert_eq!(p.filtered(&rs)[0].id, "GLM-4");
    }

    #[test]
    fn typing_filters_and_resets_cursor() {
        let mut p = ModelPicker {
            selected: 2,
            ..Default::default()
        };
        let out = p.on_key(KeyCode::Char('g'), &recipes());
        assert_eq!(out, PickerOutcome::None);
        assert_eq!(p.query, "g");
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn enter_chooses_the_selected_filtered_recipe() {
        let mut p = ModelPicker {
            query: "glm".into(),
            ..Default::default()
        };
        let out = p.on_key(KeyCode::Enter, &recipes());
        match out {
            PickerOutcome::Chosen(r) => {
                assert_eq!(r.id, "GLM-4");
                assert_eq!(r.preferred_engine.as_deref(), Some("vllm"));
            }
            other => panic!("expected Chosen, got {other:?}"),
        }
    }

    #[test]
    fn down_clamps_to_filtered_len() {
        let mut p = ModelPicker::default();
        for _ in 0..10 {
            p.on_key(KeyCode::Down, &recipes());
        }
        assert_eq!(p.selected, 2);
    }

    #[test]
    fn enter_with_no_match_is_none() {
        let mut p = ModelPicker {
            query: "zzz-nothing".into(),
            ..Default::default()
        };
        assert_eq!(p.on_key(KeyCode::Enter, &recipes()), PickerOutcome::None);
    }

    #[test]
    fn esc_cancels() {
        let mut p = ModelPicker::default();
        assert_eq!(p.on_key(KeyCode::Esc, &recipes()), PickerOutcome::Cancelled);
    }

    #[test]
    fn snapshot_lists_recipes() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(116, 24);
        let mut term = Terminal::new(backend).unwrap();
        let p = ModelPicker::default();
        let rs = recipes();
        term.draw(|f| draw_model_picker(f, f.area(), &p, &rs, &theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        let out: String = buf
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("Pick a model recipe"));
        assert!(out.contains("Qwen3-4B-Instruct"));
        assert!(out.contains("engine"));
    }
}
