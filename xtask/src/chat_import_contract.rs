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
//! member of a `use crate::{..}` group, and `mod chat`.
//!
//! Test code is exempt. A `#[cfg(test)]` item (a `mod tests` block, or a single
//! `#[cfg(test)] use`/`fn`) may reach into `chat`: a test is not part of the
//! module graph the shipped binary has. The scan removes those items by matching
//! braces over text from which comments and literals have already been blanked,
//! by [`crate::env_mutation_contract`]'s `strip_literals_and_comments`, so a
//! brace inside a string cannot move where a test item appears to end.
//!
//! Known limits, stated rather than hidden:
//!
//! * Only `#[cfg(test)]` exactly is recognised as a test gate. `#[cfg(all(test,
//!   ..))]` reads as production code, which fails toward a false alarm rather
//!   than a missed import.
//! * The scan covers the files in `GUARDED`, not the modules they import. A
//!   module the planner depends on that imports `chat` would re-form the cycle
//!   one step away; add it to `GUARDED` if that ever needs pinning.

#[cfg(test)]
mod tests {
    use crate::env_mutation_contract::tests::strip_literals_and_comments;
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

    /// The index just past the item that begins at `index`: up to the first
    /// `;` outside parentheses and brackets, or, when a `{` comes first, past
    /// its matching `}`.
    ///
    /// Any attributes stacked under the gate need no handling of their own:
    /// their brackets are tracked like any others, so the item still ends where
    /// it should. Parentheses and brackets are tracked so the `;` of `[u8; 2]`
    /// or of `fn f(x: [u8; 2])` is not taken for the end of the item. A `;`
    /// right after the closing `}` (`use a::{b, c};`, `const X: Foo = Foo { .. };`)
    /// is left behind as a stray token, which nothing here looks at. A const
    /// whose value has a second block, such as `if c { 1 } else { 2 }`, ends at
    /// the first `}` and leaves the rest as production code, which can only
    /// make the scan stricter.
    fn item_end(tokens: &[Token], index: usize) -> usize {
        let mut nesting = 0usize;
        let mut i = index;
        while i < tokens.len() {
            match tokens[i].tok {
                Tok::Punct('(' | '[') => nesting += 1,
                Tok::Punct(')' | ']') => nesting = nesting.saturating_sub(1),
                Tok::Punct(';') if nesting == 0 => return i + 1,
                Tok::Punct('{') if nesting == 0 => {
                    let mut depth = 0usize;
                    while i < tokens.len() {
                        match tokens[i].tok {
                            Tok::Punct('{') => depth += 1,
                            Tok::Punct('}') => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    return i + 1;
                }
                _ => {}
            }
            i += 1;
        }
        tokens.len()
    }

    /// The tokens left once every `#[cfg(test)]` item is removed.
    fn without_test_items(tokens: &[Token]) -> Vec<Token> {
        let mut kept = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if let Some(len) = cfg_test_attribute_len(tokens, i) {
                i = item_end(tokens, i + len);
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
        let tokens = without_test_items(&tokenize(&strip_literals_and_comments(source)));
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
        ] {
            assert_eq!(lines_of(source), vec![line], "{source:?}");
        }
    }

    #[test]
    fn a_chat_import_inside_a_test_item_is_allowed() {
        for source in [
            "fn prod() {}\n\n#[cfg(test)]\nmod tests {\n    use crate::chat;\n    use crate::{Cli, chat as c};\n    #[test]\n    fn t() {\n        crate::chat::f();\n    }\n}\n",
            "#[cfg(test)]\nuse crate::chat::any_substring;\n",
            "#[cfg(test)]\nuse crate::{Cli, chat};\n",
            "#[cfg(test)]\n#[allow(dead_code)]\npub(crate) fn only_in_tests() -> bool {\n    crate::chat::f()\n}\n",
            "#[cfg(test)]\nconst X: Foo = Foo { a: crate::chat::A };\n",
            "#[cfg(test)]\nstatic T: [&str; 2] = [crate::chat::A, crate::chat::B];\n",
        ] {
            assert!(
                chat_imports_in_production_code(source).is_empty(),
                "{source:?}"
            );
        }
    }

    #[test]
    fn production_code_after_a_test_item_is_still_scanned() {
        let source = "#[cfg(test)]\nuse crate::chat::A;\n\
                      use crate::chat::B;\n\
                      #[cfg(test)]\nfn only_in_tests() {\n    crate::chat::f();\n}\n\
                      fn prod() {\n    crate::chat::g();\n}\n";
        assert_eq!(lines_of(source), vec![3, 9]);
    }

    #[test]
    fn a_test_item_ends_at_its_own_closing_brace_not_at_a_brace_in_a_literal() {
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
}
