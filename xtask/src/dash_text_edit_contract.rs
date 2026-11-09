// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: a dash-TUI text field edits its `String` through
//! `apply_char_key` / `apply_char_key_filtered` in `ui/widgets.rs`, not through
//! its own "`Char` appends, `Backspace` pops one char" code.
//!
//! The helper replaced the hand-rolled copies, and the convention lives in
//! `AGENTS.md` and `docs/architecture.md`. Prose does not stop the next copy, so
//! this scan does.
//!
//! The rule is per file, not per `match` arm, because the copies were not all
//! arms: `install_manager.rs` had `KeyCode::Char(c) => i.type_char(c)` with the
//! `push` in a separate `fn type_char(&mut self, c: char)`. In the production
//! code of every file under `crates/rocm-dash-tui/src/` other than `ui/widgets.rs`:
//!
//! * a no-argument `.pop()` call in a file that names `KeyCode::Backspace` is
//!   reported, and
//! * `.push(x)` (or `.push(*x)`, `.push(x.method())`, `.push(x as char)`, and each
//!   of them with a trailing comma such as `.push(x,)`) is reported where `x` is
//!   a name the same file binds as a key character (`KeyCode::Char(x)`,
//!   `KeyCode::Char(mut x)`) or declares as a `char` parameter (`x: char`), in a
//!   file that names `KeyCode::Char(`.
//!
//! Production code means what is left after [`crate::rust_source`]'s
//! `strip_literals_and_comments` blanks comments and literals and every
//! `#[cfg(test)] mod name { .. }` block is blanked too, so a unit test may drive
//! a buffer by hand. A `#[cfg(test)]` on anything else exempts nothing. An
//! out-of-line test module (`#[cfg(test)] mod x;`) is a separate file, which is
//! scanned in full. [`crate::chat_import_contract`] blanks test modules over a
//! token stream instead; this one works over the stripped text because it needs
//! only the module's extent, and the two are not shared.
//!
//! Known limits, stated rather than hidden:
//!
//! * It looks for a `String` edit by shape, not by type. A `.pop()` that takes a
//!   value (`let top = stack.pop()`) in a file that handles `Backspace` is a
//!   false alarm; rename it, or add the file to `ALLOWED` with the reason.
//!   Bindings are per file, not per function, so an unrelated `out.push(c)` is
//!   also reported when the same file binds `KeyCode::Char(c)` somewhere else.
//! * Only `.pop()` and `.push(x)` are matched, and a push only when its argument
//!   is the bound name alone, as listed above: `.push((x))`, `.push(char::from(x))`,
//!   `.push(if shift { x.to_ascii_uppercase() } else { x })` and a call written
//!   with a gap before the paren (`.push (x)`, which rustfmt never produces) are
//!   not seen. `push_str`, `truncate`, `clear`, `String::pop(&mut b)`,
//!   `.insert(..)`, `.remove(..)` and `replace_range(..)` are not scanned. A cursor editor such as the serve wizard's Host field
//!   (`InlineEditor` in `ui/serve_wizard.rs`) cannot use the append-only helper,
//!   so it is a different thing, and a second copy of one goes unseen.
//! * A file that never names `KeyCode::Char(` is not checked for a `char`
//!   push. A typed key can reach such a file through a call from another one (the
//!   `install_manager` `type_char` shape, with the arm in a different file), and
//!   that split is not seen. `x: char` counts as a binding wherever it appears,
//!   as a parameter, a field or a `let`; `x: &char` does not.
//! * Only `#[cfg(test)]` exactly is read as a test gate. `#[cfg(all(test, ..))]`
//!   and a file-level `#![cfg(test)]` read as production code, which fails toward
//!   a false alarm. `use KeyCode::*` or an alias of `KeyCode` defeats every
//!   trigger, since the scan looks for the path `KeyCode::`.
//! * A `Backspace` handler that never pops (`ui/folder_browser.rs` navigates to the
//!   parent directory) is not a text edit and is not reported, so it needs no
//!   `ALLOWED` entry.
//! * A buffer edited through a wrapper that is itself not in these files is not
//!   followed, and a binding renamed on the way to the `push` (`let ch = c;`) is
//!   not seen.
//! * An `ALLOWED` entry pins the number of offenses in its file, so a second
//!   hand-rolled edit added there fails, but it does not say which one is the
//!   documented one: swapping an edit for another keeps the count.

#[cfg(test)]
mod tests {
    use crate::rust_source::strip_literals_and_comments;
    use std::path::{Path, PathBuf};

    /// The one file allowed to own the append/pop edit.
    const HELPER_FILE: &str = "crates/rocm-dash-tui/src/ui/widgets.rs";

    /// The directory the scan covers, relative to the repository root.
    const SCANNED_DIR: &str = "crates/rocm-dash-tui/src";

    /// Files that legitimately keep their own `push` / `pop`: the file, how many
    /// offenses it is expected to carry (the documented ones), and the reason.
    /// The scan fails when a file carries a different number, so a file that
    /// stops needing its entry, and a further hand-rolled edit added next to the
    /// documented one, are both caught.
    const ALLOWED: &[(&str, usize, &str)] = &[(
        "crates/rocm-dash-tui/src/app/actions.rs",
        2,
        "the Chat input: its keys are first mapped to a `KeyAction` by a pure, \
         separately tested table, and `apply_action` then pushes and pops the \
         chat buffer from that action, so there is no `KeyCode` match to route \
         through the helper. The two offenses are that push and that pop. The \
         push is found because the file also binds `KeyCode::Char(c)` elsewhere: \
         bindings are file-wide, not followed through `KeyAction::ChatInput(c)`",
    )];

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Offense {
        line: usize,
        what: String,
    }

    fn is_ident_char(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    /// Blank `text[from..to]` to spaces, keeping newlines so line numbers hold.
    fn blank(chars: &mut [char], from: usize, to: usize) {
        for c in &mut chars[from..to] {
            if *c != '\n' {
                *c = ' ';
            }
        }
    }

    fn skip_ws(chars: &[char], mut i: usize) -> usize {
        while chars.get(i).is_some_and(|c| c.is_whitespace()) {
            i += 1;
        }
        i
    }

    fn starts_with_at(chars: &[char], i: usize, word: &str) -> bool {
        word.chars()
            .enumerate()
            .all(|(k, w)| chars.get(i + k) == Some(&w))
    }

    /// The index past the bracket group opening at `open`, or the end of input.
    fn past_balanced(chars: &[char], open: usize) -> usize {
        let mut depth = 0usize;
        for (i, c) in chars.iter().enumerate().skip(open) {
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return i + 1;
                    }
                }
                _ => {}
            }
        }
        chars.len()
    }

    /// `source` with comments, literals and every `#[cfg(test)] mod` blanked,
    /// line structure preserved.
    fn production_text(source: &str) -> Vec<char> {
        const GATE: &str = "#[cfg(test)]";
        let mut chars: Vec<char> = strip_literals_and_comments(source).chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if !starts_with_at(&chars, i, GATE) {
                i += 1;
                continue;
            }
            let mut j = skip_ws(&chars, i + GATE.len());
            // Attributes stacked under the gate, then a visibility.
            loop {
                if starts_with_at(&chars, j, "#[") {
                    j = skip_ws(&chars, past_balanced(&chars, j + 1));
                } else if starts_with_at(&chars, j, "pub")
                    && !chars.get(j + 3).is_some_and(|c| is_ident_char(*c))
                {
                    j = skip_ws(&chars, j + 3);
                    if chars.get(j) == Some(&'(') {
                        j = skip_ws(&chars, past_balanced(&chars, j));
                    }
                } else {
                    break;
                }
            }
            let is_mod = starts_with_at(&chars, j, "mod")
                && chars.get(j + 3).is_some_and(|c| c.is_whitespace());
            if !is_mod {
                i += GATE.len();
                continue;
            }
            j = skip_ws(&chars, j + 3);
            while chars.get(j).is_some_and(|c| is_ident_char(*c)) {
                j += 1;
            }
            j = skip_ws(&chars, j);
            let end = match chars.get(j) {
                Some('{') => past_balanced(&chars, j),
                Some(';') => j + 1,
                _ => {
                    i += GATE.len();
                    continue;
                }
            };
            blank(&mut chars, i, end);
            i = end;
        }
        chars
    }

    fn line_of(chars: &[char], index: usize) -> usize {
        1 + chars[..index].iter().filter(|c| **c == '\n').count()
    }

    fn find_all(chars: &[char], word: &str) -> Vec<usize> {
        (0..chars.len())
            .filter(|i| starts_with_at(chars, *i, word))
            .collect()
    }

    /// The identifier starting at `i`, if one does.
    fn ident_at(chars: &[char], i: usize) -> Option<(String, usize)> {
        let first = chars.get(i)?;
        if !(first.is_alphabetic() || *first == '_') {
            return None;
        }
        let mut end = i;
        while chars.get(end).is_some_and(|c| is_ident_char(*c)) {
            end += 1;
        }
        Some((chars[i..end].iter().collect(), end))
    }

    /// Names the file binds as a typed-in character: `KeyCode::Char(x)` and
    /// `x: char`.
    fn char_bindings(chars: &[char]) -> Vec<String> {
        let mut names = Vec::new();
        for at in find_all(chars, "KeyCode::Char(") {
            let mut i = skip_ws(chars, at + "KeyCode::Char(".len());
            // `mut c` and `ref c` bind the name `c`; the modifier is not it.
            while let Some((name, end)) = ident_at(chars, i) {
                if name == "mut" || name == "ref" {
                    i = skip_ws(chars, end);
                    continue;
                }
                names.push(name);
                break;
            }
        }
        for at in find_all(chars, "char") {
            let after = at + "char".len();
            if chars.get(after).is_some_and(|c| is_ident_char(*c)) {
                continue;
            }
            // Walk back over `: ` to the name before it. A `char` that ends a
            // longer name (`mychar`) has no `:` before it and stops there.
            let mut k = at;
            while k > 0 && chars[k - 1].is_whitespace() {
                k -= 1;
            }
            if k == 0 || chars[k - 1] != ':' || (k >= 2 && chars[k - 2] == ':') {
                continue;
            }
            k -= 1;
            while k > 0 && chars[k - 1].is_whitespace() {
                k -= 1;
            }
            let end = k;
            while k > 0 && is_ident_char(chars[k - 1]) {
                k -= 1;
            }
            if k < end {
                names.push(chars[k..end].iter().collect());
            }
        }
        names
    }

    /// Whether the text after a `.push(` argument's leading name (ending at `end`)
    /// closes the call with that name as the whole argument: the bare name
    /// (`push(c)`), a method on it (`push(c.to_ascii_lowercase())`), a no-op cast
    /// to `char` (`push(c as char)`), and any of the first and last with a
    /// trailing comma (`push(c,)`, which rustfmt writes when a call spans lines).
    /// A second argument (`push(c, 1)`) or a cast to another type
    /// (`push(c as u8)`, a `Vec<u8>` push) is not a `String` append.
    fn is_whole_push_argument(chars: &[char], end: usize) -> bool {
        let mut i = skip_ws(chars, end);
        if starts_with_at(chars, i, "as") && chars.get(i + 2).is_some_and(|c| c.is_whitespace()) {
            let ty = skip_ws(chars, i + 2);
            // Any other cast (`as u8`) leaves `i` on the `as`, which closes nothing.
            if starts_with_at(chars, ty, "char")
                && !chars
                    .get(ty + "char".len())
                    .is_some_and(|c| is_ident_char(*c))
            {
                i = skip_ws(chars, ty + "char".len());
            }
        }
        match chars.get(i) {
            Some(')' | '.') => true,
            Some(',') => chars.get(skip_ws(chars, i + 1)) == Some(&')'),
            _ => false,
        }
    }

    /// Every hand-rolled append or pop in `source`'s production code.
    ///
    /// Takes the raw file text, so a caller cannot forget to strip comments,
    /// literals and test modules first.
    fn hand_rolled_edits(source: &str) -> Vec<Offense> {
        let chars = production_text(source);
        let mut offenses = Vec::new();
        let names_backspace = !find_all(&chars, "KeyCode::Backspace").is_empty();
        if names_backspace {
            for at in find_all(&chars, ".pop(") {
                let close = skip_ws(&chars, at + ".pop(".len());
                if chars.get(close) == Some(&')') {
                    offenses.push(Offense {
                        line: line_of(&chars, at),
                        what: "a `.pop()` in a file that handles `KeyCode::Backspace`".to_owned(),
                    });
                }
            }
        }
        if !find_all(&chars, "KeyCode::Char(").is_empty() {
            let bound = char_bindings(&chars);
            for at in find_all(&chars, ".push(") {
                let mut arg = skip_ws(&chars, at + ".push(".len());
                // `push(*c)` when the key is matched by reference.
                while matches!(chars.get(arg), Some('*' | '&')) {
                    arg = skip_ws(&chars, arg + 1);
                }
                let Some((name, end)) = ident_at(&chars, arg) else {
                    continue;
                };
                if is_whole_push_argument(&chars, end) && bound.contains(&name) {
                    offenses.push(Offense {
                        line: line_of(&chars, at),
                        what: format!("`.push({name})` of a typed-in character"),
                    });
                }
            }
        }
        offenses.sort_by_key(|o| o.line);
        offenses
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every `.rs` file under the dash-TUI `src/`, as `(repo-relative `/` path,
    /// text)`.
    fn dash_tui_sources(root: &Path) -> Vec<(String, String)> {
        fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(root, &path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let relative = path
                        .strip_prefix(root)
                        .expect("under the root")
                        .to_string_lossy()
                        .replace('\\', "/");
                    let text = std::fs::read_to_string(&path)
                        .unwrap_or_else(|err| panic!("cannot read {relative}: {err}"));
                    out.push((relative, text));
                }
            }
        }
        let mut found = Vec::new();
        walk(root, &root.join(SCANNED_DIR), &mut found);
        found.sort();
        found
    }

    /// The failure lines for `files`: each offense outside the helper file and
    /// the allowlist, and each allowlist entry that no longer needs its place.
    fn check(files: &[(String, String)]) -> Vec<String> {
        let mut failures = Vec::new();
        for (path, source) in files {
            if path == HELPER_FILE {
                continue;
            }
            let offenses = hand_rolled_edits(source);
            let Some((_, expected, _)) = ALLOWED.iter().find(|(file, _, _)| file == path) else {
                for o in offenses {
                    failures.push(format!(
                        "{path}:{}: {}. A text field edits its buffer with `apply_char_key` or \
                         `apply_char_key_filtered` from ui/widgets.rs, not its own push/pop",
                        o.line, o.what
                    ));
                }
                continue;
            };
            let lines = offenses
                .iter()
                .map(|o| o.line.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            if offenses.is_empty() {
                failures.push(format!(
                    "ALLOWED entry `{path}` is stale: it no longer hand-rolls a text edit. \
                     Remove the entry, or the exception outlives its reason"
                ));
            } else if offenses.len() > *expected {
                failures.push(format!(
                    "ALLOWED entry `{path}` documents {expected} hand-rolled edit(s) and the \
                     file now has {} (lines {lines}). A further text field belongs on \
                     `apply_char_key`; raise the count only for a reviewed exception",
                    offenses.len()
                ));
            } else if offenses.len() < *expected {
                failures.push(format!(
                    "ALLOWED entry `{path}` documents {expected} hand-rolled edit(s) but the \
                     file now has {} (lines {lines}): part of the exception is gone. Lower \
                     the count to match, or remove the entry if none is left",
                    offenses.len()
                ));
            }
        }
        for (file, _, _) in ALLOWED {
            if !files.iter().any(|(path, _)| path == file) {
                failures.push(format!(
                    "ALLOWED entry `{file}` names a file that does not exist; update it in \
                     xtask/src/dash_text_edit_contract.rs"
                ));
            }
        }
        failures
    }

    /// A file with one append and one pop: the shape of the Chat input's pair.
    const TWO_EDITS: &str = "\
fn push(c: KeyCode, q: &mut String) { if let KeyCode::Char(x) = c { q.push(x); } }
fn pop(c: KeyCode, q: &mut String) { if c == KeyCode::Backspace { q.pop(); } }
";

    /// The failure lines that are about `name`, so a test about one file does
    /// not depend on every other file in the tree being clean.
    fn in_file<'a>(failures: &'a [String], name: &str) -> Vec<&'a String> {
        failures.iter().filter(|f| f.contains(name)).collect()
    }

    fn file(path: &str, source: &str) -> (String, String) {
        (path.to_owned(), source.to_owned())
    }

    fn lines(offenses: &[Offense]) -> Vec<usize> {
        offenses.iter().map(|o| o.line).collect()
    }

    #[test]
    fn the_dash_tui_has_no_hand_rolled_text_edit_outside_the_helper() {
        let files = dash_tui_sources(&repo_root());
        let failures = check(&files);
        assert!(
            failures.is_empty(),
            "dash-TUI text fields must edit through the shared helper:\n  {}",
            failures.join("\n  ")
        );
    }

    #[test]
    fn the_scan_reaches_the_helper_and_the_converted_modules() {
        let files = dash_tui_sources(&repo_root());
        for needed in [
            HELPER_FILE,
            "crates/rocm-dash-tui/src/ui/logs_view.rs",
            "crates/rocm-dash-tui/src/ui/install_manager.rs",
            "crates/rocm-dash-tui/src/app/actions.rs",
            "crates/rocm-dash-tui/src/ui/folder_browser.rs",
        ] {
            assert!(
                files.iter().any(|(path, _)| path == needed),
                "the walk did not reach {needed}"
            );
        }
        // The helper itself is the shape being banned elsewhere, so it must
        // read as an offender here: that is what the exemption is for.
        let helper = &files.iter().find(|(p, _)| p == HELPER_FILE).unwrap().1;
        assert_eq!(hand_rolled_edits(helper).len(), 2, "push and pop");
    }

    #[test]
    fn every_allowed_file_carries_a_reason() {
        for (file, expected, reason) in ALLOWED {
            assert!(*expected > 0, "ALLOWED entry `{file}` expects no offenses");
            assert!(
                reason.len() > 40,
                "ALLOWED entry `{file}` has no real justification: {reason:?}"
            );
        }
    }

    #[test]
    fn a_hand_rolled_arm_is_reported_with_its_line() {
        let source = "\
fn on_key(l: &mut Logs, key: KeyEvent) {
    match key.code {
        KeyCode::Char(c) => l.query.push(c),
        KeyCode::Backspace => {
            l.query.pop();
        }
        _ => {}
    }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![3, 5]);
    }

    #[test]
    fn the_helper_split_shape_that_the_arm_scan_would_miss_is_reported() {
        // install_manager.rs before the shared helper: the arms call methods,
        // and the push/pop live in those methods.
        let source = "\
fn on_key(i: &mut Install, key: KeyEvent) {
    match key.code {
        KeyCode::Backspace => i.backspace(),
        KeyCode::Char(ch) => i.type_char(ch),
        _ => {}
    }
}
impl Install {
    fn type_char(&mut self, c: char) {
        match self.field {
            Field::Channel => self.channel.push(c),
            _ => {}
        }
    }
    fn backspace(&mut self) {
        match self.field {
            Field::Channel => {
                self.channel.pop();
            }
            _ => {}
        }
    }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![11, 18]);
    }

    #[test]
    fn a_char_parameter_is_a_binding_even_when_no_arm_names_it() {
        // Only `x: char` binds `c` here; `KeyCode::Char('x')` just makes this a
        // file that handles typed keys.
        let source = "\
fn on_key(code: KeyCode, i: &mut Install) {
    if code == KeyCode::Char('x') { i.clear_channel(); }
}
fn type_char(q: &mut String, c: char) {
    q.push(c);
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![5]);
    }

    #[test]
    fn a_binding_by_mut_or_ref_or_a_pushed_deref_or_method_is_reported() {
        // One file each: a name bound by any form is bound for the whole file,
        // so a shared `c` would let one form's binding cover for another's.
        for source in [
            "fn a(code: KeyCode, q: &mut String) {\n    if let KeyCode::Char(mut c) = code { q.push(c); }\n}\n",
            "fn b(code: &KeyCode, q: &mut String) {\n    if let KeyCode::Char(c) = code { q.push(*c); }\n}\n",
            "fn d(code: KeyCode, q: &mut String) {\n    if let KeyCode::Char(ref c) = code { q.push(c.to_ascii_lowercase()); }\n}\n",
        ] {
            assert_eq!(lines(&hand_rolled_edits(source)), vec![2], "{source}");
        }
    }

    #[test]
    fn binding_names_are_whole_identifiers_not_single_letters() {
        // `key_char` carries an underscore and `mychar` ends in the word `char`:
        // only the first is a char parameter, and only it is pushed.
        let source = "\
fn on_key(code: KeyCode, i: &mut Install) {
    if code == KeyCode::Char('x') { i.clear_channel(); }
}
fn typed(q: &mut String, key_char: char) {
    q.push(key_char);
}
fn bytes(out: &mut Vec<u8>, mychar: u8) {
    out.push(mychar);
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![5]);
    }

    #[test]
    fn a_push_through_two_dereferences_is_reported() {
        let source = "\
fn a(code: &&KeyCode, q: &mut String) {
    if let KeyCode::Char(c) = code { q.push(**c); }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![2]);
    }

    /// A key arm that appends `shape` to a `String`, wrapped in a function so the
    /// whole file is a valid fixture for [`hand_rolled_edits`].
    fn key_arm_pushing(shape: &str) -> String {
        format!(
            "fn f(code: KeyCode, q: &mut String) {{\n    if let KeyCode::Char(c) = code {{\n        \
             {shape};\n    }}\n}}\n"
        )
    }

    #[test]
    fn a_push_with_a_trailing_comma_is_reported() {
        // `q.push(c,)` is a valid call, and the multi-line form below is what
        // rustfmt writes when a call is too long for one line. Both append a
        // typed-in character, so both are offenses.
        let one_line = key_arm_pushing("q.push(c,)");
        assert_eq!(lines(&hand_rolled_edits(&one_line)), vec![3], "{one_line}");
        let multi_line = key_arm_pushing("q.push(\n            c,\n        )");
        assert_eq!(
            lines(&hand_rolled_edits(&multi_line)),
            vec![3],
            "{multi_line}"
        );
    }

    #[test]
    fn the_other_spellings_of_a_pushed_typed_in_char_are_reported() {
        let mut missed = Vec::new();
        for (shape, call) in [
            ("spaces around the comma", "q.push( c , )"),
            ("a dereference and a comma", "q.push(*c,)"),
            ("a method and a comma", "q.push(c.to_ascii_lowercase(),)"),
            ("a comment before the paren", "q.push(c /* typed */)"),
            ("a cast to char", "q.push(c as char)"),
            ("a cast to char and a comma", "q.push(c as char,)"),
            (
                "the argument on its own line",
                "q.push(\n            c\n        )",
            ),
        ] {
            let source = key_arm_pushing(call);
            if hand_rolled_edits(&source).len() != 1 {
                missed.push(shape);
            }
        }
        assert!(missed.is_empty(), "not reported: {missed:?}");
    }

    #[test]
    fn a_push_with_more_than_the_typed_in_char_is_not_reported() {
        // Not a `String` append of the typed key: two arguments, or a cast to a
        // type a `String` cannot take.
        for (shape, call) in [
            ("two arguments", "q.push(c, 1)"),
            ("two arguments over lines", "q.push(c,\n            1)"),
            ("a cast to a byte", "q.push(c as u8,)"),
            ("a cast to a byte", "q.push(c as u8)"),
        ] {
            let source = key_arm_pushing(call);
            assert!(hand_rolled_edits(&source).is_empty(), "{shape}: {source}");
        }
    }

    #[test]
    fn a_pop_with_an_argument_and_a_push_of_a_computed_value_are_ignored() {
        let source = "\
fn f(code: KeyCode, history: &mut History, bytes: &mut Vec<u8>) {
    if code == KeyCode::Backspace { history.pop(2); }
    if let KeyCode::Char(c) = code { bytes.push(c as u8); }
}
";
        assert!(hand_rolled_edits(source).is_empty());
    }

    #[test]
    fn a_push_of_something_that_is_not_a_typed_in_char_is_ignored() {
        let source = "\
fn on_key(key: KeyEvent, out: &mut Vec<String>, label: &str) {
    match key.code {
        KeyCode::Char('x') => out.push(label.to_owned()),
        KeyCode::Char(c) if c.is_ascii_digit() => out.push(format!(\"{c}\")),
        _ => {}
    }
    out.push(label);
}
";
        assert!(hand_rolled_edits(source).is_empty());
    }

    #[test]
    fn a_pop_or_push_in_a_file_that_handles_no_text_key_is_ignored() {
        // `exec.rs` and similar pop a Vec and render chars; neither is a field.
        let source = "\
fn pad(out: &mut String, c: char) {
    out.push(c);
}
fn last(words: &mut Vec<String>) -> Option<String> {
    words.pop()
}
";
        assert!(hand_rolled_edits(source).is_empty());
    }

    #[test]
    fn a_pop_that_takes_a_value_is_still_reported_in_a_backspace_file() {
        // The scan cannot tell a `Vec` pop from a `String` pop; the allowlist
        // is the way out, documented in the module comment.
        let source = "\
fn f(stack: &mut Vec<u8>, code: KeyCode) {
    if code == KeyCode::Backspace {
        let _ = stack.pop();
    }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![3]);
    }

    #[test]
    fn code_inside_a_test_module_is_allowed_and_code_around_it_is_not() {
        let source = "\
fn on_key(code: KeyCode, q: &mut String) {
    if let KeyCode::Char(c) = code { q.push(c); }
}
#[cfg(test)]
mod tests {
    fn drive(code: KeyCode, q: &mut String) {
        match code {
            KeyCode::Char(c) => q.push(c),
            KeyCode::Backspace => { q.pop(); }
            _ => {}
        }
    }
}
fn late(code: KeyCode, q: &mut String) {
    if code == KeyCode::Backspace { q.pop(); }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![2, 15]);
    }

    #[test]
    fn a_test_module_with_stacked_attributes_and_a_visibility_is_still_a_test_module() {
        let source = "\
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod helpers {
    fn drive(code: KeyCode, q: &mut String) {
        if let KeyCode::Char(c) = code { q.push(c); }
    }
}
#[cfg(test)]
pub mod more {
    fn drive(code: KeyCode, q: &mut String) {
        if code == KeyCode::Backspace { q.pop(); }
    }
}
fn late(code: KeyCode, q: &mut String) {
    if code == KeyCode::Backspace { q.pop(); }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![15]);
    }

    #[test]
    fn an_out_of_line_test_module_declaration_leaves_the_rest_of_the_file_scanned() {
        let source = "\
#[cfg(test)]
mod tests;
fn late(code: KeyCode, q: &mut String) {
    if code == KeyCode::Backspace { q.pop(); }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![4]);
    }

    #[test]
    fn a_test_gate_on_a_function_exempts_nothing() {
        let source = "\
#[cfg(test)]
fn on_key(code: KeyCode, q: &mut String) {
    if let KeyCode::Char(c) = code { q.push(c); }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![3]);
    }

    #[test]
    fn a_test_module_ends_at_its_own_brace_not_at_one_in_a_literal() {
        let source = "\
#[cfg(test)]
mod tests {
    fn s() -> &'static str { \"}}\" }
}
fn on_key(code: KeyCode, q: &mut String) {
    if let KeyCode::Char(c) = code { q.push(c); }
}
";
        assert_eq!(lines(&hand_rolled_edits(source)), vec![6]);
    }

    #[test]
    fn mentions_in_comments_and_strings_are_not_edits() {
        let source = "\
// KeyCode::Backspace pops with q.pop();
/// KeyCode::Char(c) pushes with q.push(c).
fn f(code: KeyCode) -> &'static str {
    let _ = KeyCode::Backspace;
    \"q.pop(); KeyCode::Char(c) q.push(c)\"
}
";
        assert!(hand_rolled_edits(source).is_empty());
    }

    #[test]
    fn the_failure_names_the_file_the_line_and_the_helper() {
        let files = [
            file(
                "crates/rocm-dash-tui/src/ui/new_form.rs",
                "fn k(c: KeyCode, q: &mut String) {\n    if let KeyCode::Char(x) = c { q.push(x); }\n}\n",
            ),
            file(ALLOWED[0].0, TWO_EDITS),
        ];
        let failures = check(&files);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].starts_with("crates/rocm-dash-tui/src/ui/new_form.rs:2:"));
        assert!(failures[0].contains("apply_char_key"));
    }

    #[test]
    fn the_helper_file_and_an_allowed_file_are_not_reported() {
        let files = [file(HELPER_FILE, TWO_EDITS), file(ALLOWED[0].0, TWO_EDITS)];
        assert_eq!(check(&files), Vec::<String>::new());
    }

    #[test]
    fn an_allowed_file_that_loses_part_of_its_exception_fails_with_the_right_advice() {
        // One of the two documented edits is gone: the count is stale, and the
        // advice must be to lower it, not to move a further field onto the helper.
        let one = TWO_EDITS.lines().next().unwrap().to_owned() + "\n";
        let failures = check(&[file(ALLOWED[0].0, &one)]);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("documents 2"), "{failures:?}");
        assert!(failures[0].contains("now has 1"), "{failures:?}");
        assert!(
            failures[0].contains("part of the exception is gone"),
            "{failures:?}"
        );
        assert!(!failures[0].contains("further text field"), "{failures:?}");
    }

    #[test]
    fn a_further_edit_in_an_allowed_file_fails() {
        let three = format!(
            "{TWO_EDITS}fn more(c: KeyCode, q: &mut String) {{ if let KeyCode::Char(x) = c {{ q.push(x); }} }}\n"
        );
        let failures = check(&[file(ALLOWED[0].0, &three)]);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("documents 2"), "{failures:?}");
        assert!(failures[0].contains("now has 3"), "{failures:?}");
        assert!(failures[0].contains("further text field"), "{failures:?}");
        assert!(failures[0].contains("apply_char_key"), "{failures:?}");
    }

    #[test]
    fn an_allowed_file_that_stops_hand_rolling_fails() {
        let files = [file(ALLOWED[0].0, "fn nothing() {}\n")];
        let failures = check(&files);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("is stale"));
    }

    #[test]
    fn an_allowed_file_that_no_longer_exists_fails() {
        let failures = check(&[]);
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("does not exist"));
    }

    #[test]
    fn the_guard_finds_a_hand_rolled_arm_added_to_a_converted_module() {
        // Liveness on real files: the fixtures prove the scan on small strings;
        // this proves it on a file it has to get through, in both of the
        // spellings the helper replaced.
        let files = dash_tui_sources(&repo_root());
        let logs = "crates/rocm-dash-tui/src/ui/logs_view.rs";
        let original = &files.iter().find(|(p, _)| p == logs).unwrap().1;
        let helper_call = "apply_char_key(&mut l.query, code);";
        assert!(original.contains(helper_call), "the mutation target moved");
        assert!(hand_rolled_edits(original).is_empty(), "baseline is clean");

        let inline = original.replace(
            helper_call,
            "match code { KeyCode::Char(c) => l.query.push(c), \
             KeyCode::Backspace => { l.query.pop(); } _ => {} }",
        );
        let mut mutated = files.clone();
        mutated.iter_mut().find(|(p, _)| p == logs).unwrap().1 = inline;
        let all = check(&mutated);
        let failures = in_file(&all, "logs_view.rs");
        assert_eq!(failures.len(), 2, "{failures:?}");

        let appended = format!(
            "{original}\nfn late(code: KeyCode, q: &mut String) {{\n    \
             if code == KeyCode::Backspace {{ q.pop(); }}\n}}\n"
        );
        let mut mutated = files.clone();
        mutated.iter_mut().find(|(p, _)| p == logs).unwrap().1 = appended;
        let all = check(&mutated);
        let failures = in_file(&all, "logs_view.rs");
        assert_eq!(
            failures.len(),
            1,
            "after the file's test module: {failures:?}"
        );

        // The same edits inside the file's real `mod tests` are test code.
        let in_tests = original.replacen(
            "mod tests {",
            "mod tests {\n    fn drive(code: KeyCode, q: &mut String) {\n        \
             match code { KeyCode::Char(c) => q.push(c), \
             KeyCode::Backspace => { q.pop(); } _ => {} }\n    }",
            1,
        );
        assert_ne!(&in_tests, original, "the test module moved");
        let mut mutated = files;
        mutated.iter_mut().find(|(p, _)| p == logs).unwrap().1 = in_tests;
        let all = check(&mutated);
        let failures = in_file(&all, "logs_view.rs");
        assert!(failures.is_empty(), "{failures:?}");
    }
}
