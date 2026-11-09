// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: the freeform planner and the leaf modules it shares
//! with `chat.rs` must not import `chat` in production code.
//!
//! `chat.rs` calls the freeform planner (`plan_freeform_request_with_context`),
//! so a planner that also imported `chat` would be a module cycle. The cycle was
//! broken by moving what the planner used from `chat.rs` into leaf modules
//! (`prompt_text.rs`, `rocm_argv.rs`, `tool_policy.rs`) that import nothing from
//! either side. `cargo xtask check-crate-edges` cannot see that rule, because it
//! compares crates, not modules inside `apps/rocm`, so without this test a later
//! change can put the import back and nothing fails.
//!
//! The scan is on `use` paths, not on the word `chat`: modules in this crate are
//! full of identifiers such as `chat_cli_arg_value` and of prose about the chat
//! assistant, and neither is an import. What it flags in production code is
//! `chat` reached as a module: `crate::chat`, `super::chat` or `self::chat` as a
//! path (a `use`, or an inline call such as `crate::chat::f()`), `chat` as a
//! member of a `use crate::{..}` group, and `mod chat`. A raw identifier
//! (`r#chat`) is the same name and is flagged the same way.
//!
//! Test code is exempt, one way only: a `#[cfg(test)] mod name { .. }` block
//! (or `mod name;`, with any attributes and a visibility between the gate and
//! `mod`) is removed before the scan, by matching brackets over text from which
//! comments and literals have already been blanked by [`crate::rust_source`]'s
//! `strip_literals_and_comments`, so a brace inside a string cannot move where
//! the block appears to end. A `#[cfg(test)]` on anything else (a `use`, a
//! function, a field, a variant, an expression) exempts nothing: what it covers
//! is scanned like production code, which fails toward a false alarm rather
//! than a missed import. A test that needs `chat` goes in the test module.
//!
//! Known limits, stated rather than hidden:
//!
//! * Only `#[cfg(test)]` exactly is recognised as a test gate. `#[cfg(all(test,
//!   ..))]` reads as production code, which fails toward a false alarm.
//! * The scan covers the files in `GUARDED`, not the modules they import. A
//!   module the planner depends on that imports `chat` would re-form the cycle
//!   one step away; add it to `GUARDED` if that ever needs pinning.
//! * `chat` is found only as a path segment after `crate::`, `super::` or
//!   `self::`, as a member of a group opened by one of those, or as `mod chat`.
//!   A route that never writes that path is not seen: an alias (`use crate as
//!   root; root::chat::f()`), a glob (`use crate::*;` then `chat::f()`) or a
//!   `#[path = ".."] mod` that points at `chat.rs`. None is used in these files.

#[cfg(test)]
mod tests {
    use crate::rust_source::strip_literals_and_comments;
    use std::path::Path;

    /// The files that must not import `chat` in production code, relative to
    /// the repository root.
    ///
    /// The planner itself, then the leaf modules it reaches the shared tool
    /// policy, argv helpers and prompt text through. The test below fails if
    /// one of these no longer exists, so a rename has to update this list
    /// rather than quietly dropping the file out of the guard.
    const GUARDED: &[&str] = &[
        "apps/rocm/src/freeform_planner.rs",
        "apps/rocm/src/prompt_text.rs",
        "apps/rocm/src/rocm_argv.rs",
        "apps/rocm/src/tool_policy.rs",
    ];

    /// The reserved words, strict and future. Only these keep their `r#` in
    /// [`tokenize`]; any other raw identifier is the plain name.
    const KEYWORDS: &[&str] = &[
        "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
        "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl",
        "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub",
        "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "try",
        "type", "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
    ];

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Tok {
        Ident(String),
        /// `::`
        PathSep,
        Punct(char),
    }

    #[derive(Clone, Debug)]
    struct Token {
        tok: Tok,
        line: usize,
    }

    /// A production-code import of `chat`.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Offense {
        line: usize,
        /// What was found, for the failure message.
        what: &'static str,
    }

    /// Split already-stripped source into identifiers, `::` and single
    /// punctuation characters, keeping the 1-based line of each.
    fn tokenize(stripped: &str) -> Vec<Token> {
        let chars: Vec<char> = stripped.chars().collect();
        let mut tokens = Vec::new();
        let mut line = 1;
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '\n' {
                line += 1;
                i += 1;
            } else if c.is_whitespace() {
                i += 1;
            } else if c == 'r'
                && chars.get(i + 1) == Some(&'#')
                && chars
                    .get(i + 2)
                    .is_some_and(|next| next.is_alphabetic() || *next == '_')
            {
                // A raw identifier, `r#chat`: one token that names `chat`.
                let start = i + 2;
                i = start;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                // `r#type` is a name that happens to spell a keyword, so it
                // keeps its prefix and cannot be mistaken for the keyword.
                let name = if KEYWORDS.contains(&name.as_str()) {
                    format!("r#{name}")
                } else {
                    name
                };
                tokens.push(Token {
                    tok: Tok::Ident(name),
                    line,
                });
            } else if c.is_alphanumeric() || c == '_' {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                tokens.push(Token {
                    tok: Tok::Ident(chars[start..i].iter().collect()),
                    line,
                });
            } else if c == ':' && chars.get(i + 1) == Some(&':') {
                tokens.push(Token {
                    tok: Tok::PathSep,
                    line,
                });
                i += 2;
            } else {
                tokens.push(Token {
                    tok: Tok::Punct(c),
                    line,
                });
                i += 1;
            }
        }
        tokens
    }

    fn is_punct(tokens: &[Token], index: usize, wanted: char) -> bool {
        tokens
            .get(index)
            .is_some_and(|t| t.tok == Tok::Punct(wanted))
    }

    fn is_ident(tokens: &[Token], index: usize, wanted: &str) -> bool {
        tokens
            .get(index)
            .is_some_and(|t| matches!(&t.tok, Tok::Ident(name) if name == wanted))
    }

    /// The length of a `#[cfg(test)]` attribute starting at `index`, if one does.
    fn cfg_test_attribute_len(tokens: &[Token], index: usize) -> Option<usize> {
        let matches = is_punct(tokens, index, '#')
            && is_punct(tokens, index + 1, '[')
            && is_ident(tokens, index + 2, "cfg")
            && is_punct(tokens, index + 3, '(')
            && is_ident(tokens, index + 4, "test")
            && is_punct(tokens, index + 5, ')')
            && is_punct(tokens, index + 6, ']');
        matches.then_some(7)
    }

    /// The index just past the `(`, `[` or `{` at `open`, once its matching
    /// closer has been consumed (the end of the input when it has none).
    fn past_balanced(tokens: &[Token], open: usize) -> usize {
        let mut depth = 0usize;
        let mut i = open;
        while i < tokens.len() {
            match tokens[i].tok {
                Tok::Punct('(' | '[' | '{') => depth += 1,
                Tok::Punct(')' | ']' | '}') => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        return i + 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        tokens.len()
    }

    /// The first token of what a gate covers: past any attributes stacked under
    /// the gate and a `pub` or `pub(..)` visibility.
    fn first_token_gated(tokens: &[Token], mut index: usize) -> usize {
        loop {
            if is_punct(tokens, index, '#') && is_punct(tokens, index + 1, '[') {
                index = past_balanced(tokens, index + 1);
            } else if is_ident(tokens, index, "pub") {
                index += 1;
                if is_punct(tokens, index, '(') {
                    index = past_balanced(tokens, index);
                }
            } else {
                return index;
            }
        }
    }

    /// The index just past the `#[cfg(test)] mod name { .. }` (or `mod name;`)
    /// whose gate ends just before `start`, or `None` when the gate is on
    /// anything else.
    fn past_test_module(tokens: &[Token], start: usize) -> Option<usize> {
        let module = first_token_gated(tokens, start);
        if !is_ident(tokens, module, "mod") {
            return None;
        }
        let after_name = module + 2;
        if is_punct(tokens, after_name, '{') {
            Some(past_balanced(tokens, after_name))
        } else if is_punct(tokens, after_name, ';') {
            Some(after_name + 1)
        } else {
            None
        }
    }

    /// The tokens left once every `#[cfg(test)] mod` block is removed. A gate on
    /// anything else removes nothing, so what it covers is scanned as the
    /// production code it may be.
    fn without_test_modules(tokens: &[Token]) -> Vec<Token> {
        let mut kept = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if let Some(len) = cfg_test_attribute_len(tokens, i)
                && let Some(end) = past_test_module(tokens, i + len)
            {
                i = end;
                continue;
            }
            kept.push(tokens[i].clone());
            i += 1;
        }
        kept
    }

    fn is_path_root(tokens: &[Token], index: usize) -> bool {
        ["crate", "super", "self"]
            .iter()
            .any(|root| is_ident(tokens, index, root))
    }

    /// Every place `source` reaches the `chat` module outside test code.
    ///
    /// Takes the raw file text, so a caller (the real-tree test, the fixtures)
    /// cannot forget to strip comments and literals first.
    fn chat_imports_in_production_code(source: &str) -> Vec<Offense> {
        let tokens = without_test_modules(&tokenize(&strip_literals_and_comments(source)));
        let mut offenses = Vec::new();
        // One entry per open `{`: whether it directly follows `crate::`,
        // `super::` or `self::`, so its members are children of the root.
        let mut groups: Vec<bool> = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            match &token.tok {
                Tok::Punct('{') => {
                    let after_root =
                        i >= 2 && tokens[i - 1].tok == Tok::PathSep && is_path_root(&tokens, i - 2);
                    groups.push(after_root);
                }
                Tok::Punct('}') => {
                    groups.pop();
                }
                Tok::Ident(name) if name == "chat" => {
                    let after_root_path =
                        i >= 2 && tokens[i - 1].tok == Tok::PathSep && is_path_root(&tokens, i - 2);
                    let group_member = groups.last() == Some(&true)
                        && (is_punct(&tokens, i - 1, '{') || is_punct(&tokens, i - 1, ','));
                    let module_decl = i >= 1 && is_ident(&tokens, i - 1, "mod");
                    let what = if after_root_path {
                        Some("a path through `crate::chat` (or `super::`/`self::`)")
                    } else if group_member {
                        Some("`chat` as a member of a `use crate::{..}` group")
                    } else if module_decl {
                        Some("a `mod chat` declaration")
                    } else {
                        None
                    };
                    if let Some(what) = what {
                        offenses.push(Offense {
                            line: token.line,
                            what,
                        });
                    }
                }
                _ => {}
            }
        }
        offenses
    }

    /// One failure line per offense, locating it as `file:line`.
    fn report(file: &str, offenses: &[Offense]) -> Vec<String> {
        offenses
            .iter()
            .map(|o| format!("{file}:{}: {}", o.line, o.what))
            .collect()
    }

    #[test]
    fn the_freeform_planner_and_its_leaf_modules_never_import_chat() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory");
        let mut failures = Vec::new();
        for relative in GUARDED {
            let source = std::fs::read_to_string(root.join(relative)).unwrap_or_else(|err| {
                panic!(
                    "cannot read {relative}: {err}. If the module moved or was renamed, \
                     update GUARDED in xtask/src/chat_import_contract.rs so it stays guarded."
                )
            });
            failures.extend(report(relative, &chat_imports_in_production_code(&source)));
        }
        assert!(
            failures.is_empty(),
            "these modules import `chat` in non-test code, which closes the cycle \
             `chat` -> freeform planner -> `chat` again:\n  {}\n\
             Move whatever the module needs from `chat.rs` into a leaf module \
             (`prompt_text.rs`, `rocm_argv.rs` or `tool_policy.rs`, or a new one that \
             does not import `chat`) and import it from there. `chat` may be \
             referenced from a `#[cfg(test)]` item.",
            failures.join("\n  ")
        );
    }

    #[test]
    fn the_guard_finds_an_import_in_the_real_planner_source() {
        // Liveness: the fixtures below prove the scan on small strings; this
        // proves it on the real file, whose test items and braces it has to
        // get past before it reaches the production code.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory");
        let source = std::fs::read_to_string(root.join(GUARDED[0])).expect("planner source");
        // Counted against the file's own result, so this stays a statement
        // about the scanner when the real-tree test above is red.
        let baseline = chat_imports_in_production_code(&source).len();
        let with_import = format!("use crate::chat;\n{source}");
        let hits = chat_imports_in_production_code(&with_import);
        assert_eq!(hits.len(), baseline + 1, "{hits:?}");
        assert_eq!(hits[0].line, 1, "{hits:?}");
        let at_the_end = format!("{source}\nfn late() {{ crate::chat::f(); }}\n");
        let hits = chat_imports_in_production_code(&at_the_end);
        assert_eq!(
            hits.len(),
            baseline + 1,
            "a call appended after the file's `mod tests` is production code: {hits:?}"
        );
    }

    fn lines_of(source: &str) -> Vec<usize> {
        chat_imports_in_production_code(source)
            .iter()
            .map(|o| o.line)
            .collect()
    }

    #[test]
    fn a_plain_use_of_chat_is_reported_with_its_line() {
        let source = "use anyhow::Result;\n\nuse crate::chat;\n";
        let offenses = chat_imports_in_production_code(source);
        assert_eq!(offenses.len(), 1, "{offenses:?}");
        assert_eq!(
            report("apps/rocm/src/freeform_planner.rs", &offenses),
            vec![
                "apps/rocm/src/freeform_planner.rs:3: \
                 a path through `crate::chat` (or `super::`/`self::`)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn every_spelling_of_the_import_is_reported() {
        for (source, line) in [
            ("use crate::chat::any_substring;\n", 1),
            ("use crate::chat as c;\n", 1),
            ("use super::chat::ensure_flag;\n", 1),
            ("use self::chat;\n", 1),
            ("use crate::{Cli, chat, providers};\n", 1),
            ("use crate::{Cli, providers, chat};\n", 1),
            ("use crate::{\n    Cli,\n    chat,\n    providers,\n};\n", 3),
            ("use crate::{chat::any_substring, Cli};\n", 1),
            ("fn f() {\n    let _ = crate::chat::f();\n}\n", 2),
            ("mod chat;\n", 1),
            ("#[cfg(not(test))]\nuse crate::chat;\n", 2),
            // A re-export, and the shape the cycle had before the leaf modules:
            // `chat` in the import group, then bare `chat::f()` calls.
            ("pub use crate::chat::any_substring;\n", 1),
            ("pub(crate) use crate::chat::{A, B};\n", 1),
            (
                "use crate::{Cli, chat, providers};\nfn f() { chat::any_substring(); }\n",
                1,
            ),
        ] {
            assert_eq!(lines_of(source), vec![line], "{source:?}");
        }
    }

    #[test]
    fn a_chat_import_inside_a_test_module_is_allowed() {
        for source in [
            "fn prod() {}\n\n#[cfg(test)]\nmod tests {\n    use crate::chat;\n    use crate::{Cli, chat as c};\n    #[test]\n    fn t() {\n        crate::chat::f();\n    }\n}\n",
            "#[cfg(test)]\n#[allow(dead_code)]\npub(crate) mod helpers {\n    pub fn f() -> bool {\n        crate::chat::f()\n    }\n}\n",
            "#[cfg(test)]\n#[path = \"planner_tests.rs\"]\nmod tests;\n",
        ] {
            assert!(
                chat_imports_in_production_code(source).is_empty(),
                "{source:?}"
            );
        }
    }

    #[test]
    fn production_code_around_a_test_module_is_still_scanned() {
        let source = "use crate::chat::A;\n\
                      #[cfg(test)]\nmod tests {\n    use crate::chat::B;\n}\n\
                      use crate::chat::C;\n\
                      #[cfg(test)]\nmod more {\n    fn f() { crate::chat::D; }\n}\n\
                      fn prod() {\n    crate::chat::E;\n}\n";
        assert_eq!(lines_of(source), vec![1, 6, 12]);
    }

    #[test]
    fn a_test_module_ends_at_its_own_closing_brace_not_at_a_brace_in_a_literal() {
        // The `use` between the two literals is test code and stays exempt: a
        // `}` read out of the first literal would close the item before it.
        let source = "#[cfg(test)]\nmod tests {\n    const S: &str = \"}\";\n    use crate::chat;\n    const T: &str = \"{\";\n}\n\nuse crate::chat;\n";
        assert_eq!(lines_of(source), vec![8]);
    }

    #[test]
    fn mentions_of_chat_that_are_not_imports_are_ignored() {
        for source in [
            "// use crate::chat;\n",
            "/// Moved out of `crate::chat` so the planner never imports it.\n",
            "/* use crate::chat; */\n",
            "const NOTE: &str = \"crate::chat::f\";\n",
            "use crate::rocm_argv::chat_cli_arg_value;\n",
            "use crate::{providers, tool_policy};\nfn f() { chat_tool_call_is_read_only(); }\n",
            "fn f() {\n    let chat = 1;\n    let _ = chat;\n}\n",
            "use crate::providers::chat;\n",
            "use crate::{providers::{chat}, Cli};\n",
            "use crate::{a::{b, chat}};\n",
            "fn f(s: &str) -> bool {\n    s.contains(\"chat\")\n}\n",
        ] {
            assert!(
                chat_imports_in_production_code(source).is_empty(),
                "{source:?}"
            );
        }
    }

    #[test]
    fn a_raw_identifier_spelling_of_chat_is_reported() {
        // `r#chat` names the same module as `chat`, so every form above has a
        // raw twin that must fail the guard too.
        for (source, line) in [
            ("use crate::r#chat;\n", 1),
            ("use crate::{r#chat};\n", 1),
            ("use crate::{Cli, r#chat, providers};\n", 1),
            ("use crate::{Cli, r#chat as c};\n", 1),
            ("use crate::{\n    Cli,\n    r#chat,\n};\n", 3),
            ("use crate::{r#chat::any_substring, Cli};\n", 1),
            ("use super::r#chat::ensure_flag;\n", 1),
            ("use self::r#chat;\n", 1),
            ("mod r#chat;\n", 1),
            ("pub mod r#chat;\n", 1),
            ("fn f() {\n    let _ = crate::r#chat::f();\n}\n", 2),
            ("#[cfg(not(test))]\nuse crate::r#chat;\n", 2),
        ] {
            assert_eq!(lines_of(source), vec![line], "{source:?}");
        }
    }

    #[test]
    fn raw_identifiers_that_are_not_the_chat_module_are_ignored() {
        for source in [
            "use crate::providers::r#chat;\n",
            "use crate::r#chat_cli_arg_value;\n",
            "fn f() {\n    let r#chat = 1;\n    let _ = r#chat;\n}\n",
            "use crate::{providers::{r#chat}, Cli};\n",
            "struct S {\n    r#type: u8,\n}\n",
        ] {
            assert!(
                chat_imports_in_production_code(source).is_empty(),
                "{source:?}"
            );
        }
    }

    #[test]
    fn a_raw_identifier_is_one_token_and_a_raw_keyword_stays_distinct() {
        let names = |source: &str| -> Vec<String> {
            tokenize(source)
                .into_iter()
                .filter_map(|t| match t.tok {
                    Tok::Ident(name) => Some(name),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(names("r#chat"), vec!["chat"]);
        assert_eq!(names("crate::r#chat::f"), vec!["crate", "chat", "f"]);
        // `r#type` is a field or variable called `type`, not the keyword that
        // starts a type alias, so it must not read as one.
        assert_eq!(names("r#type"), vec!["r#type"]);
        assert_eq!(names("r#match r#fn"), vec!["r#match", "r#fn"]);
        // An `r` that is only the start of an ordinary identifier is untouched.
        assert_eq!(names("r rest #[x]"), vec!["r", "rest", "x"]);
    }

    #[test]
    fn a_test_gate_on_anything_but_a_module_exempts_nothing() {
        // Only a `#[cfg(test)] mod` block is exempt. A gate on a `use`, a
        // function, a field, a variant, a parameter or an expression covers
        // nothing the scan skips, so it cannot hide what follows it either:
        // this is the shape that let a comparison `<` inside a gated element
        // swallow the production elements after it when the scan tried to find
        // where each such gate ends. Every line below must still be reported.
        for (source, lines) in [
            ("#[cfg(test)]\nuse crate::chat::any_substring;\n", &[2][..]),
            (
                "#[cfg(test)]\nfn only_in_tests() {\n    crate::chat::f();\n}\n",
                &[3],
            ),
            (
                "struct S {\n    #[cfg(test)]\n    m: crate::chat::T,\n    n: crate::chat::U,\n}\n",
                &[3, 4],
            ),
            (
                "fn f() -> S {\n    S {\n        #[cfg(test)]\n        a: x < 1,\n        b: crate::chat::B,\n    }\n}\n",
                &[5],
            ),
            (
                "fn f(x: u8) {\n    match x {\n        #[cfg(test)]\n        0 => crate::chat::f(),\n        _ => crate::chat::g(),\n    }\n}\n",
                &[4, 5],
            ),
            ("#[cfg(all(test, unix))]\nuse crate::chat;\n", &[2]),
        ] {
            assert_eq!(lines_of(source), lines, "{source:?}");
        }
    }
}
