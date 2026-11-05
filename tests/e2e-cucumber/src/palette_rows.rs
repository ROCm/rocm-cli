// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which destination rows of the dashboard's command palette a screen still lacks.

/// The palette's filter line, drawn at the popup's first text column.
const FILTER_HINT: &str = ": type to filter";

/// The popup's title, drawn on its top border.
const TITLE: &str = "Go to";

/// What `screen` still lacks of the command palette, one entry per absent piece
/// (empty once every row is drawn). `labels` are the destinations in the order
/// the palette lists them.
///
/// A row is looked for at the popup's own text column, which the filter line
/// fixes, and each row must sit on a later line than the one before it. The
/// popup clears its interior, so the dimmed backdrop cannot be at that column on
/// a finished frame, and a backdrop word beside the popup (a two-space prefix and
/// a destination's name are enough) does not count as a row. The last label is
/// the last row drawn, so a screen with none missing is drawn through the
/// palette's bottom.
///
/// The column is counted in characters, which is exact while the lines left of
/// the popup hold no double-width glyph.
#[must_use]
pub fn palette_rows_missing(screen: &str, labels: &[&str]) -> Vec<String> {
    let lines: Vec<&str> = screen.lines().collect();
    let mut missing = Vec::new();
    if !screen.contains(TITLE) {
        missing.push(format!("the {TITLE:?} title"));
    }
    let Some((hint_line, column)) = lines.iter().enumerate().find_map(|(index, line)| {
        line.find(FILTER_HINT)
            .map(|byte| (index, line[..byte].chars().count()))
    }) else {
        missing.push(format!("the palette's {FILTER_HINT:?} line"));
        return missing;
    };
    let mut next_line = hint_line + 1;
    for label in labels {
        let row = lines
            .iter()
            .enumerate()
            .skip(next_line)
            .find(|(_, line)| is_row(line, column, label));
        match row {
            Some((index, _)) => next_line = index + 1,
            None => missing.push(format!("the palette row {label:?}")),
        }
    }
    missing
}

/// Whether `line`, from `column` on, is a palette row for `label`: the cursor
/// column (`▸ ` on the selected row, two spaces on the others), then the label.
fn is_row(line: &str, column: usize, label: &str) -> bool {
    let from_column: String = line.chars().skip(column).collect();
    from_column.starts_with(&format!("▸ {label}")) || from_column.starts_with(&format!("  {label}"))
}

#[cfg(test)]
mod tests {
    use super::palette_rows_missing;

    const LABELS: [&str; 5] = ["Home", "ROCm", "Serving", "Observe", "Chat"];

    /// A finished frame: a tab bar above (it names the destinations too), the
    /// popup in the middle with a backdrop word on each side of it.
    fn frame(rows: &[&str]) -> String {
        let mut lines = vec![
            " Home   ROCm   Serving   ● Observe   Chat".to_string(),
            "  backdrop ╭── Go to… ────────╮ backdrop".to_string(),
            "  backdrop │ : type to filter │ backdrop".to_string(),
            "  backdrop │                  │ backdrop".to_string(),
        ];
        lines.extend(
            rows.iter()
                .map(|row| format!("  backdrop │ {row:<16} │ backdrop")),
        );
        lines.join("\n")
    }

    #[test]
    fn a_finished_palette_lacks_nothing() {
        let screen = frame(&["▸ Home", "  ROCm", "  Serving", "  Observe", "  Chat"]);
        assert_eq!(palette_rows_missing(&screen, &LABELS), Vec::<String>::new());
    }

    #[test]
    fn the_selected_row_may_be_any_of_them() {
        let screen = frame(&["  Home", "  ROCm", "  Serving", "  Observe", "▸ Chat"]);
        assert_eq!(palette_rows_missing(&screen, &LABELS), Vec::<String>::new());
    }

    #[test]
    fn a_palette_drawn_only_down_to_observe_lacks_chat() {
        let screen = frame(&["▸ Home", "  ROCm", "  Serving", "  Observe"]);
        assert_eq!(
            palette_rows_missing(&screen, &LABELS),
            vec!["the palette row \"Chat\"".to_string()]
        );
    }

    #[test]
    fn a_backdrop_word_beside_the_popup_is_not_a_row() {
        // Two spaces and a destination's name, right of the popup and on a line
        // below its drawn rows: the earlier whole-screen match took this for the
        // Chat row and ended the wait before Chat was drawn.
        let mut screen = frame(&["▸ Home", "  ROCm", "  Serving", "  Observe"]);
        screen.push_str("\n  backdrop │ (blank)          │   Chat history");
        assert_eq!(
            palette_rows_missing(&screen, &LABELS),
            vec!["the palette row \"Chat\"".to_string()]
        );
    }

    #[test]
    fn a_backdrop_word_on_the_left_is_not_a_row() {
        let mut screen = frame(&["▸ Home", "  ROCm", "  Serving", "  Observe"]);
        screen.push_str("\n  Chat     │ (blank)          │ backdrop");
        assert_eq!(
            palette_rows_missing(&screen, &LABELS),
            vec!["the palette row \"Chat\"".to_string()]
        );
    }

    #[test]
    fn rows_must_come_in_ascending_lines() {
        // Chat above Home: every label is on screen as a row, in the wrong order,
        // so the rows after the first that is out of order are not all found.
        let screen = frame(&["  Chat", "▸ Home", "  ROCm", "  Serving", "  Observe"]);
        let missing = palette_rows_missing(&screen, &LABELS);
        assert_eq!(missing, vec!["the palette row \"Chat\"".to_string()]);
    }

    #[test]
    fn the_tab_bar_alone_is_not_a_palette() {
        let screen = " Home   ROCm   Serving   ● Observe   Chat";
        assert_eq!(
            palette_rows_missing(screen, &LABELS),
            vec![
                "the \"Go to\" title".to_string(),
                "the palette's \": type to filter\" line".to_string(),
            ]
        );
    }

    #[test]
    fn a_half_drawn_popup_lacks_every_row() {
        let screen = frame(&[]);
        assert_eq!(palette_rows_missing(&screen, &LABELS).len(), LABELS.len());
    }
}
