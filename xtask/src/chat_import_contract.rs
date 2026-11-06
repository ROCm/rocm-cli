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
//! Test code is exempt. A `#[cfg(test)]` item (a `mod tests` block, or a single
//! `#[cfg(test)] use`/`fn`) may reach into `chat`: a test is not part of the
//! module graph the shipped binary has. The scan removes what a gate covers by
//! matching brackets over text from which comments and literals have already
//! been blanked, by [`crate::rust_source`]'s
//! `strip_literals_and_comments`, so a brace inside a string cannot move where a
//! test item appears to end. The gate covers one thing, which is not always an
//! item: it can sit on a struct field, an enum variant, a parameter, a
//! struct-literal field, a match arm or a statement, and then it ends with that
//! element (see `gated_end`), not with the next function in the file. Whether
//! the gate sits on a list element or on an item is decided by the bracket
//! around it: anything inside `(..)` or `[..]` is an element (so the
//! function-pointer fields of `struct S(#[cfg(test)] unsafe fn(), T);` end at
//! the comma and `T` stays scanned), and inside `{..}` by the first word of
//! what is gated (`fn`, `use`, `let` begin an item or statement, a field's
//! name does not).
//!
//! Known limits, stated rather than hidden:
//!
//! * Only `#[cfg(test)]` exactly is recognised as a test gate. `#[cfg(all(test,
//!   ..))]` reads as production code, which fails toward a false alarm rather
//!   than a missed import.
//! * The scan covers the files in `GUARDED`, not the modules they import. A
//!   module the planner depends on that imports `chat` would re-form the cycle
//!   one step away; add it to `GUARDED` if that ever needs pinning.
//! * `chat` is found only as a path segment after `crate::`, `super::` or
//!   `self::`, as a member of a group opened by one of those, or as `mod chat`.
//!   A route that never writes that path is not seen: an alias (`use crate as
//!   root; root::chat::f()`), a glob (`use crate::*;` then `chat::f()`) or a
//!   `#[path = ".."] mod` that points at `chat.rs`. None is used in these files.
//! * Where a gate ends is decided from the tokens, not from a parse. A `<` that
//!   follows an identifier counts as the start of generic arguments when the
//!   gate sits on a list element, so a comparison at the top level of a gated
//!   element can extend the exemption to the closing bracket of its list; that
//!   hides at most the elements after it in the same braces.
//! * An item gated inside a macro invocation written with `(..)` or `[..]`
//!   (`lazy_static!( #[cfg(test)] fn f<A, B>() where A: X, B: Y {..} )`) is
//!   read as a list element, so a comma in its generics or `where` clause ends
//!   the exemption early; what follows is then scanned as production, which
//!   can only make the scan stricter. A match arm in `{..}` whose pattern
//!   starts with an item keyword (an inline `const { .. }` pattern) is read as
//!   an item and ends at its block or `;` instead of at its comma.

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

    /// Whether the token at `index` begins an item or a statement, as opposed
    /// to one element of a comma-separated list (a field, a variant, a
    /// parameter, a struct-literal field, a match arm).
    ///
    /// The difference is where the gated thing ends. An item or statement ends
    /// at its `;` or its block, and the commas in its generics and `where`
    /// clause are not its end (`fn f<A, B>() where A: X, B: Y {..}`). An
    /// element ends at the comma that separates it from the next one.
    fn begins_item_or_statement(tokens: &[Token], index: usize) -> bool {
        const STARTERS: &[&str] = &[
            "fn", "const", "static", "struct", "enum", "impl", "trait", "type", "mod", "use",
            "extern", "unsafe", "async", "let", "if", "match", "for", "while", "loop", "return",
            "break", "continue",
        ];
        let followed_by_a_name =
            matches!(tokens.get(index + 1).map(|t| &t.tok), Some(Tok::Ident(_)));
        match tokens.get(index).map(|t| &t.tok) {
            Some(Tok::Ident(name)) => {
                STARTERS.contains(&name.as_str())
                    // `union` is only a keyword where it declares one; a field
                    // may be called `union`.
                    || (name == "union" && followed_by_a_name)
            }
            Some(Tok::Punct('{')) => true,
            _ => false,
        }
    }

    /// The bracket that most closely encloses the token at `index`: the `(`,
    /// `[` or `{` whose closer has not yet been read there, or `None` at the
    /// top level of the file.
    fn enclosing_opener(tokens: &[Token], index: usize) -> Option<char> {
        let mut closers = 0usize;
        for token in tokens[..index.min(tokens.len())].iter().rev() {
            match token.tok {
                Tok::Punct(')' | ']' | '}') => closers += 1,
                Tok::Punct(open @ ('(' | '[' | '{')) => {
                    if closers == 0 {
                        return Some(open);
                    }
                    closers -= 1;
                }
                _ => {}
            }
        }
        None
    }

    /// Whether the gate at `gate`, covering what starts at `start`, sits on one
    /// element of a comma-separated list rather than on an item or statement.
    ///
    /// Inside `(..)` or `[..]` that is always so: a tuple struct's or variant's
    /// fields, a parameter list, call arguments and array elements are lists,
    /// and none holds an item or a statement. That matters because such an
    /// element can start with the same word as an item, and only the
    /// enclosing bracket tells them apart: the field type `fn()`, `unsafe fn()`
    /// or `extern "C" fn()` is not a function item. In a `{..}` the first word
    /// decides (see [`begins_item_or_statement`]), since a braced struct's
    /// field starts with its name and no list element there starts with a
    /// keyword that begins an item.
    fn gate_is_on_a_list_element(tokens: &[Token], gate: usize, start: usize) -> bool {
        matches!(enclosing_opener(tokens, gate), Some('(' | '['))
            || !begins_item_or_statement(tokens, first_token_gated(tokens, start))
    }

    /// The index just past what the `#[cfg(test)]` gate covers, where `gate` is
    /// the index of the gate's `#` and `start` is the first token after it.
    ///
    /// The gate covers one item, statement or list element, so the exemption
    /// ends at the first of:
    ///
    /// * a `;` outside brackets (consumed);
    /// * for a list element only, a `,` outside brackets and outside generic
    ///   arguments (consumed);
    /// * the end of the first `{..}` block that opens outside brackets
    ///   (consumed), which is where an item with a body, a match arm with a
    ///   block and a struct-like variant end;
    /// * a closer that has no opener inside the gate: the `}` of the struct,
    ///   enum, block or match the gated field, variant, parameter or tail
    ///   expression sits in, or the `)` of its parameter list (not consumed,
    ///   so what follows it is scanned as the production code it is).
    ///
    /// Without the last rule a gate on the last field of a struct, with no
    /// trailing comma, carried on past the struct's `}` and swallowed the next
    /// function. Any attributes stacked under the gate need no handling of
    /// their own: their brackets are tracked like any others. Brackets are
    /// tracked so the `;` of `[u8; 2]` or of `fn f(x: [u8; 2])` is not taken
    /// for the end. A `;` right after a closing `}` (`use a::{b, c};`,
    /// `const X: Foo = Foo { .. };`) is left behind as a stray token, which
    /// nothing here looks at. A value with a second block, such as
    /// `if c { 1 } else { 2 }`, ends at the first `}` and leaves the rest as
    /// production code, which can only make the scan stricter.
    ///
    /// Generic arguments are told apart from a comparison by one test: a `<`
    /// that follows an identifier or `::`. A list element has no comparison at
    /// its top level in practice, and a miscount in this direction only ever
    /// extends the exemption to the enclosing closer, so a gated element with
    /// an odd `<` hides at most the elements after it in the same braces.
    fn gated_end(tokens: &[Token], gate: usize, start: usize) -> usize {
        let element = gate_is_on_a_list_element(tokens, gate, start);
        let mut nesting = 0usize;
        let mut generics = 0usize;
        let mut i = start;
        while i < tokens.len() {
            match tokens[i].tok {
                Tok::Punct('(' | '[') => nesting += 1,
                Tok::Punct('{') if nesting == 0 => return past_balanced(tokens, i),
                Tok::Punct('{') => nesting += 1,
                Tok::Punct(')' | ']' | '}') => {
                    if nesting == 0 {
                        return i;
                    }
                    nesting -= 1;
                }
                Tok::Punct(';') if nesting == 0 => return i + 1,
                Tok::Punct(',') if element && nesting == 0 && generics == 0 => return i + 1,
                Tok::Punct('<')
                    if element
                        && nesting == 0
                        && i > start
                        && matches!(tokens[i - 1].tok, Tok::Ident(_) | Tok::PathSep) =>
                {
                    generics += 1;
                }
                Tok::Punct('>')
                    if element
                        && nesting == 0
                        && generics > 0
                        // `->` and `=>` are not the end of generic arguments.
                        && !is_punct(tokens, i.wrapping_sub(1), '-')
                        && !is_punct(tokens, i.wrapping_sub(1), '=') =>
                {
                    generics -= 1;
                }
                _ => {}
            }
            i += 1;
        }
        tokens.len()
    }

    /// The tokens left once everything a `#[cfg(test)]` gate covers is removed.
    fn without_test_items(tokens: &[Token]) -> Vec<Token> {
        let mut kept = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if let Some(len) = cfg_test_attribute_len(tokens, i) {
                i = gated_end(tokens, i, i + len);
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
    fn a_test_gate_on_a_field_or_variant_covers_only_that_element() {
        // Each case is (source, lines expected). The gated element is exempt;
        // everything after it, in the same braces or after them, is not.
        let cases: [(&str, &[usize]); 15] = [
            // `->` inside the generic arguments is not their closing `>`.
            (
                "struct S { #[cfg(test)] m: HashMap<fn() -> V, crate::chat::K> }\n",
                &[],
            ),
            // The shape Copilot reported: a gated last field with no comma.
            (
                "struct State { #[cfg(test)] marker: () }\nfn production() { crate::chat::f(); }\n",
                &[2],
            ),
            (
                "struct State {\n    #[cfg(test)]\n    marker: (),\n}\nfn production() {\n    crate::chat::f();\n}\n",
                &[6],
            ),
            (
                "struct State { a: u8, #[cfg(test)] pub(crate) marker: () }\nfn production() { crate::chat::f(); }\n",
                &[2],
            ),
            // Stacked attributes, and a field after the gated one.
            (
                "struct State {\n    #[cfg(test)]\n    #[allow(dead_code)]\n    marker: (),\n    real: crate::chat::T,\n}\n",
                &[5],
            ),
            // A comma inside the gated field's generic arguments is not its end.
            (
                "struct S {\n    #[cfg(test)]\n    m: HashMap<A, crate::chat::T>,\n    n: crate::chat::U,\n}\n",
                &[4],
            ),
            ("struct S { #[cfg(test)] m: crate::chat::T }\n", &[]),
            // Enum variants: bare, tuple, struct-like, and the last one.
            (
                "enum E { A, #[cfg(test)] B }\nfn production() { crate::chat::f(); }\n",
                &[2],
            ),
            (
                "enum E {\n    A,\n    #[cfg(test)]\n    B(crate::chat::T),\n    C(crate::chat::U),\n}\n",
                &[5],
            ),
            (
                "enum E {\n    #[cfg(test)]\n    B { x: crate::chat::T },\n    C { y: crate::chat::U },\n}\n",
                &[4],
            ),
            // A tuple struct's field closes with `)`, not `}`.
            (
                "struct S(u8, #[cfg(test)] u8);\nfn production() { crate::chat::f(); }\n",
                &[2],
            ),
            // A parameter: the body after the parameter list is production.
            (
                "fn f(a: u8, #[cfg(test)] b: u8) {\n    crate::chat::f();\n}\n",
                &[2],
            ),
            // A tail expression with no `;` before the closing brace.
            (
                "fn f() {\n    #[cfg(test)]\n    g()\n}\nfn production() { crate::chat::f(); }\n",
                &[5],
            ),
            // A field of a struct literal, and an arm of a match.
            (
                "fn f() -> S {\n    S {\n        #[cfg(test)]\n        a: crate::chat::A,\n        b: crate::chat::B,\n    }\n}\n",
                &[5],
            ),
            (
                "fn f(x: u8) {\n    match x {\n        #[cfg(test)]\n        0 => crate::chat::f(),\n        _ => crate::chat::g(),\n    }\n}\n",
                &[5],
            ),
        ];
        // Every case is checked before failing, so one run shows them all.
        let wrong: Vec<String> = cases
            .iter()
            .filter(|(source, expected)| lines_of(source) != *expected)
            .map(|(source, expected)| {
                format!(
                    "{source:?}: expected {expected:?}, got {:?}",
                    lines_of(source)
                )
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    fn a_test_gate_inside_parentheses_or_brackets_covers_one_list_element() {
        // Inside `(..)` or `[..]` there are no items or statements, only list
        // elements, so a function-pointer type that starts with `fn`, `unsafe`
        // or `extern` is a field to end at its comma, not an item to end at a
        // `;` or a block. Each case keeps the production field after it
        // visible. Cases that already held (a `{` block as the element, a
        // field that starts with a name) pin the boundary of the rule.
        let cases: [(&str, &[usize]); 12] = [
            // The shape Copilot reported, and its `unsafe` twin.
            ("struct State(#[cfg(test)] fn(), crate::chat::T);\n", &[1]),
            (
                "struct State(#[cfg(test)] unsafe fn(), crate::chat::T);\n",
                &[1],
            ),
            (
                "struct State(#[cfg(test)] extern \"C\" fn(u8), crate::chat::T);\n",
                &[1],
            ),
            (
                "struct State(#[cfg(test)] unsafe extern \"C\" fn(u8, u8) -> u8, crate::chat::T);\n",
                &[1],
            ),
            (
                "struct State(#[cfg(test)] pub unsafe fn(), crate::chat::T);\n",
                &[1],
            ),
            (
                "struct State(#[cfg(test)] for<'a> fn(&'a u8), crate::chat::T);\n",
                &[1],
            ),
            // The same fields in an enum variant, over several lines.
            (
                "enum E {\n    V(\n        #[cfg(test)]\n        unsafe fn(),\n        crate::chat::T,\n    ),\n}\n",
                &[5],
            ),
            // A gated function-pointer field is exempt, and so is the chat
            // path inside it: only what follows the comma is production.
            ("struct State(#[cfg(test)] fn(crate::chat::T), u8);\n", &[]),
            // A gated last field with no comma ends at the closing `)`.
            (
                "struct State(u8, #[cfg(test)] unsafe fn());\nfn production() { crate::chat::f(); }\n",
                &[2],
            ),
            // Braced fields start with a name, and an array element or a call
            // argument that is a block ends with the block: both already held.
            (
                "struct S {\n    #[cfg(test)]\n    f: unsafe fn(),\n    x: crate::chat::T,\n}\n",
                &[4],
            ),
            (
                "fn f() { g([#[cfg(test)] unsafe { 1 }, crate::chat::A]); }\n",
                &[1],
            ),
            (
                "fn f() { g(#[cfg(test)] unsafe { h() }, crate::chat::A); }\n",
                &[1],
            ),
        ];
        // Every case is checked before failing, so one run shows them all.
        let wrong: Vec<String> = cases
            .iter()
            .filter(|(source, expected)| lines_of(source) != *expected)
            .map(|(source, expected)| {
                format!(
                    "{source:?}: expected {expected:?}, got {:?}",
                    lines_of(source)
                )
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    #[test]
    fn a_gated_item_or_statement_is_not_ended_by_a_comma_in_its_generics() {
        // The comma rule is for elements of a list. An item or a statement
        // has commas in its generics and where clause that are not its end.
        let cases: [(&str, &[usize]); 7] = [
            // A visibility, and attributes stacked under the gate, come before
            // the keyword that shows this is an item.
            (
                "#[cfg(test)]\npub(crate) fn only_in_tests<A, B>() {\n    crate::chat::f();\n}\n",
                &[],
            ),
            (
                "#[cfg(test)]\n#[allow(dead_code)]\nfn only_in_tests<A, B>() {\n    crate::chat::f();\n}\n",
                &[],
            ),
            // `union` declares a type only when a name follows; a field can
            // be called `union`.
            (
                "#[cfg(test)]\nunion U<A, B> {\n    a: crate::chat::X,\n}\n",
                &[],
            ),
            (
                "struct S {\n    #[cfg(test)]\n    union: u8,\n    x: crate::chat::T,\n}\n",
                &[4],
            ),
            (
                "#[cfg(test)]\nfn only_in_tests<A, B>(a: A, b: B) where A: X, B: Y {\n    crate::chat::f();\n}\nfn production() { crate::chat::g(); }\n",
                &[5],
            ),
            (
                "#[cfg(test)]\nimpl<A, B> Foo for Bar<A, B> {\n    fn f() { crate::chat::f(); }\n}\n",
                &[],
            ),
            (
                "fn p() {\n    #[cfg(test)]\n    let x: HashMap<A, B> = crate::chat::f();\n    crate::chat::g();\n}\n",
                &[4],
            ),
        ];
        // Every case is checked before failing, so one run shows them all.
        let wrong: Vec<String> = cases
            .iter()
            .filter(|(source, expected)| lines_of(source) != *expected)
            .map(|(source, expected)| {
                format!(
                    "{source:?}: expected {expected:?}, got {:?}",
                    lines_of(source)
                )
            })
            .collect();
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
