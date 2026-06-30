// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reusable folder browser (Phase 3 Wave 0).
//!
//! A shared input primitive: pick (or create-under) a directory. Dependency of
//! serve / install / runtime / onboarding. Models the frozen rocm-cli
//! `folder_browser` entry kinds (`UseCurrent`, `NewChild`, `Parent`,
//! `Directory`) without the per-screen re-implementation.
//!
//! Navigation/selection is pure and unit-testable; only [`build_entries`]
//! touches the filesystem (tolerantly — an unreadable dir yields no children).

use std::path::{Path, PathBuf};

use crossterm::event::KeyCode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};

use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;

/// What a row in the browser represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FolderEntryKind {
    /// Choose the directory currently being browsed.
    UseCurrent,
    /// Create-and-choose a new child directory (CLI side performs the mkdir).
    NewChild,
    /// Ascend to the parent directory.
    Parent,
    /// Descend into a child directory.
    Directory,
}

/// One selectable row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderEntry {
    pub label: String,
    pub path: PathBuf,
    pub kind: FolderEntryKind,
}

/// The result of handling a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FolderOutcome {
    /// Key consumed, still browsing.
    None,
    /// The browser navigated to a new directory (entries refreshed).
    Navigated,
    /// The user chose `path` (UseCurrent or NewChild). The caller owns it.
    Chosen(PathBuf),
    /// The user dismissed the browser.
    Cancelled,
}

/// Folder browser state. Construct with [`FolderBrowser::new`]; drive with
/// [`FolderBrowser::on_key`].
#[derive(Debug, Clone)]
pub struct FolderBrowser {
    pub title: String,
    pub current_dir: PathBuf,
    pub entries: Vec<FolderEntry>,
    pub selected: usize,
}

impl FolderBrowser {
    /// Open the browser rooted at `start`, listing its contents.
    pub fn new(title: impl Into<String>, start: PathBuf) -> Self {
        let mut b = Self {
            title: title.into(),
            current_dir: start,
            entries: Vec::new(),
            selected: 0,
        };
        b.refresh();
        b
    }

    /// Re-list `current_dir` and clamp the cursor.
    pub fn refresh(&mut self) {
        self.entries = build_entries(&self.current_dir);
        if self.selected >= self.entries.len() {
            self.selected = self.entries.len().saturating_sub(1);
        }
    }

    /// Move the cursor by `delta`, clamped to the entry list.
    pub fn move_sel(&mut self, delta: isize) {
        if self.entries.is_empty() {
            return;
        }
        let max = self.entries.len().cast_signed() - 1;
        self.selected = (self.selected.cast_signed() + delta).clamp(0, max) as usize;
    }

    fn navigate_to(&mut self, dir: PathBuf) {
        self.current_dir = dir;
        self.selected = 0;
        self.refresh();
    }

    /// Pure-ish key handler. Filesystem is only touched on navigation refresh.
    pub fn on_key(&mut self, key: KeyCode) -> FolderOutcome {
        match key {
            KeyCode::Up | KeyCode::BackTab => {
                self.move_sel(-1);
                FolderOutcome::None
            }
            KeyCode::Down | KeyCode::Tab => {
                self.move_sel(1);
                FolderOutcome::None
            }
            KeyCode::Home => {
                self.selected = 0;
                FolderOutcome::None
            }
            KeyCode::End => {
                self.selected = self.entries.len().saturating_sub(1);
                FolderOutcome::None
            }
            KeyCode::Left | KeyCode::Backspace => {
                if let Some(parent) = self.current_dir.parent() {
                    self.navigate_to(parent.to_path_buf());
                    FolderOutcome::Navigated
                } else {
                    FolderOutcome::None
                }
            }
            KeyCode::Esc => FolderOutcome::Cancelled,
            KeyCode::Enter => self.activate(),
            _ => FolderOutcome::None,
        }
    }

    /// Act on the selected entry.
    fn activate(&mut self) -> FolderOutcome {
        let Some(entry) = self.entries.get(self.selected).cloned() else {
            return FolderOutcome::None;
        };
        match entry.kind {
            FolderEntryKind::UseCurrent | FolderEntryKind::NewChild => {
                FolderOutcome::Chosen(entry.path)
            }
            FolderEntryKind::Parent | FolderEntryKind::Directory => {
                self.navigate_to(entry.path);
                FolderOutcome::Navigated
            }
        }
    }
}

/// Build the row list for `dir`: `[Use this folder]`, `[..]` (if a parent
/// exists), the child directories (sorted), then `[+ new folder]`.
pub fn build_entries(dir: &Path) -> Vec<FolderEntry> {
    let mut entries = vec![FolderEntry {
        label: "[ use this folder ]".to_string(),
        path: dir.to_path_buf(),
        kind: FolderEntryKind::UseCurrent,
    }];

    if let Some(parent) = dir.parent() {
        entries.push(FolderEntry {
            label: "..".to_string(),
            path: parent.to_path_buf(),
            kind: FolderEntryKind::Parent,
        });
    }

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(read) = std::fs::read_dir(dir) {
        for ent in read.flatten() {
            let path = ent.path();
            if path.is_dir() {
                dirs.push(path);
            }
        }
    }
    dirs.sort();
    for path in dirs {
        let label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        entries.push(FolderEntry {
            label: format!("{label}/"),
            path,
            kind: FolderEntryKind::Directory,
        });
    }

    entries.push(FolderEntry {
        label: "[ + new folder ]".to_string(),
        path: dir.join("new-folder"),
        kind: FolderEntryKind::NewChild,
    });

    entries
}

/// Render the browser over `area`.
pub fn draw_folder_browser(f: &mut Frame, area: Rect, fb: &FolderBrowser, theme: &Theme) {
    let popup = centered_rect(80, 80, 110, 30, area);
    let inner = draw_popup_frame(f, popup, &fb.title, theme);
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

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            fb.current_dir.display().to_string(),
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))),
        rows[0],
    );

    let items: Vec<ListItem> = fb
        .entries
        .iter()
        .map(|e| {
            let color = match e.kind {
                FolderEntryKind::UseCurrent => theme.ok,
                FolderEntryKind::NewChild => theme.warn,
                FolderEntryKind::Parent => theme.muted,
                FolderEntryKind::Directory => theme.fg,
            };
            ListItem::new(Line::from(Span::styled(
                e.label.clone(),
                Style::default().fg(color),
            )))
        })
        .collect();

    let mut list_state = ListState::default();
    if !fb.entries.is_empty() {
        list_state.select(Some(fb.selected));
    }
    let list = List::new(items).highlight_style(
        Style::default()
            .bg(theme.surface_2)
            .add_modifier(Modifier::BOLD),
    );
    f.render_stateful_widget(list, rows[1], &mut list_state);

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "↑↓ move · → / Enter open · ← parent · Enter on “use”/“new” chooses · Esc cancel",
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> FolderBrowser {
        FolderBrowser {
            title: "Pick".into(),
            current_dir: PathBuf::from("/home/user/models"),
            entries: vec![
                FolderEntry {
                    label: "[ use this folder ]".into(),
                    path: PathBuf::from("/home/user/models"),
                    kind: FolderEntryKind::UseCurrent,
                },
                FolderEntry {
                    label: "..".into(),
                    path: PathBuf::from("/home/user"),
                    kind: FolderEntryKind::Parent,
                },
                FolderEntry {
                    label: "llama/".into(),
                    path: PathBuf::from("/home/user/models/llama"),
                    kind: FolderEntryKind::Directory,
                },
                FolderEntry {
                    label: "[ + new folder ]".into(),
                    path: PathBuf::from("/home/user/models/new-folder"),
                    kind: FolderEntryKind::NewChild,
                },
            ],
            selected: 0,
        }
    }

    #[test]
    fn use_current_chooses_browsed_dir() {
        let mut fb = synthetic();
        // selected = 0 = UseCurrent
        let out = fb.on_key(KeyCode::Enter);
        assert_eq!(
            out,
            FolderOutcome::Chosen(PathBuf::from("/home/user/models"))
        );
    }

    #[test]
    fn new_child_chooses_child_path() {
        let mut fb = synthetic();
        fb.selected = 3; // NewChild
        let out = fb.on_key(KeyCode::Enter);
        assert_eq!(
            out,
            FolderOutcome::Chosen(PathBuf::from("/home/user/models/new-folder"))
        );
    }

    #[test]
    fn enter_on_directory_navigates_not_chooses() {
        let mut fb = synthetic();
        fb.selected = 2; // llama/
        let out = fb.on_key(KeyCode::Enter);
        assert_eq!(out, FolderOutcome::Navigated);
        assert_eq!(fb.current_dir, PathBuf::from("/home/user/models/llama"));
        assert_eq!(fb.selected, 0);
    }

    #[test]
    fn movement_clamps() {
        let mut fb = synthetic();
        fb.on_key(KeyCode::Up); // already at top
        assert_eq!(fb.selected, 0);
        fb.on_key(KeyCode::End);
        assert_eq!(fb.selected, 3);
        fb.on_key(KeyCode::Down); // clamp at bottom
        assert_eq!(fb.selected, 3);
    }

    #[test]
    fn esc_cancels() {
        let mut fb = synthetic();
        assert_eq!(fb.on_key(KeyCode::Esc), FolderOutcome::Cancelled);
    }

    #[test]
    fn build_entries_lists_children_on_real_fs() {
        // Create a unique temp dir with two child dirs + one file.
        let base = std::env::temp_dir().join(format!("rocmdash-fb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("alpha")).unwrap();
        std::fs::create_dir_all(base.join("beta")).unwrap();
        std::fs::write(base.join("note.txt"), b"x").unwrap();

        let entries = build_entries(&base);
        // First is UseCurrent, last is NewChild.
        assert_eq!(entries.first().unwrap().kind, FolderEntryKind::UseCurrent);
        assert_eq!(entries.last().unwrap().kind, FolderEntryKind::NewChild);
        let dir_labels: Vec<&str> = entries
            .iter()
            .filter(|e| e.kind == FolderEntryKind::Directory)
            .map(|e| e.label.as_str())
            .collect();
        assert_eq!(dir_labels, vec!["alpha/", "beta/"]);
        // The plain file is not listed.
        assert!(!entries.iter().any(|e| e.label.contains("note")));

        let _ = std::fs::remove_dir_all(&base);
    }
}
