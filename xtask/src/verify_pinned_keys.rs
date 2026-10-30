// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Assert the release/metadata public keys pinned in the installers and
//! `apps/rocm/src/therock/fetch.rs` match the canonical published keys under
//! `docs/keys/`.
//!
//! This closes the gap where CI verifies release artifacts against the public key
//! configured in its environment (see `scripts/release_readiness.py`) while
//! installers and the binary trust a *separately embedded* copy. If those diverge,
//! CI can bless a release that installers then reject once default-on verification
//! lands. This check enforces a single source of truth: the committed
//! `docs/keys/*.pem` files.
//!
//! Design notes:
//!
//! - **What it guarantees.** This catches a pinned key that drifts from
//!   `docs/keys/` by mistake or slips past review: a stale or commented-out copy,
//!   a second assignment, a key pasted into the wrong slot. It is a textual check
//!   of plain installer code, not an interpreter, and does not defend against
//!   deliberately obfuscated code (a name assembled at runtime through `eval` or
//!   `Invoke-Expression`, an assignment hidden inside a multi-line string or a
//!   block that never runs). Anyone who can commit that can also change
//!   `docs/keys/` in the same commit, so review is the control there.
//! - **Dormant slots stay empty.** Until a canonical `docs/keys/` file exists and
//!   is non-empty, its slot is not compared, but it must be exactly `""`: the
//!   installers trust any non-empty slot, so a key pasted into one with nothing
//!   published would be trusted with nothing to compare it against.
//! - **Per-constant equality.** Each pinned key is embedded in a specific named
//!   constant (shell `NAME="…"`, PowerShell string/here-string, Rust `const`). We
//!   isolate *that constant's own value span*, reduce it to its base64 body, and
//!   compare it for equality to the canonical key. Comparing the whole file (rather
//!   than the specific constant) would let a wrong/tampered constant pass whenever
//!   the canonical body appears anywhere else in the file — e.g. in the `NEXT` slot
//!   mid-rotation or a stale comment. The value must be one well-formed PEM block
//!   (the BEGIN line, non-empty lines of base64, the END line, nothing else),
//!   free of `$`, backticks and backslashes (Rust may write line breaks as `\n`),
//!   and the statement must end with the string (only `;` or a comment may
//!   follow), so the text compared is the text the installer ends up holding.
//! - **The assignment the runtime uses.** The shell and PowerShell keep the *last*
//!   value they assign, so reading the first textual occurrence of the name could
//!   compare a commented-out line, a longer identifier sharing the prefix, or a
//!   value later overwritten, while the installer trusts another key. So each
//!   source must assign the name exactly once, and that assignment must be the
//!   plain declaration as its own statement (leading whitespace allowed): shell
//!   `NAME="…"` (optionally after `export`/`readonly`), PowerShell `$Name = "…"`
//!   or `$Name = @"…"@` (any case, optionally `$script:`, `$global:`, `$local:`
//!   or `$private:`; `$env:Name` is an environment variable, so it never counts
//!   as the declaration, only as a write), Rust `const NAME: &str = "…";`
//!   (optionally `pub` or `pub(crate)`). Names match as whole identifiers, case-insensitively for
//!   PowerShell, and whole-line comments are skipped. Every other write of the
//!   name counts towards "exactly once": shell `NAME=` anywhere on a line,
//!   `NAME+=`, `${NAME:=…}`, and any other mention of the name, quoted or not,
//!   except a `readonly`/`export` command that only names variables (this is how
//!   `unset` and `read` name it); PowerShell plain and compound assignment to any
//!   reference spelling (`$Name`, `$scope:Name`, `${Name}`), and any mention of
//!   the name outside a `$` reference, quoted or not (how `Set-Variable`,
//!   `Remove-Variable` and `Set-Item variable:` name it); Rust
//!   `const`/`static`/`let` bindings in any scope. Over-counting only fails the
//!   check, so prose naming the variable outside a comment line is rejected
//!   rather than guessed at.
//! - **Read-only at runtime.** `install.sh` must mark each slot `readonly` after
//!   assigning it, so a later write fails instead of silently replacing the trust
//!   root. `install.ps1` does not: `irm … | iex` runs it in the caller's session,
//!   where a read-only variable would outlive the run and make the next install in
//!   that session fail (`-Force` would avoid that, but also defeats the point).
//! - **CI cross-check.** When a CI signing public key is configured — the file
//!   named by `ROCM_CLI_SIGNING_PUBLIC_KEY_PATH`, else the inline
//!   `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM` release/nightly CI wires from the secret —
//!   it must equal the canonical *current* release key, so the key CI verifies
//!   against is exactly the one users pin. Both environment sources are resolved
//!   in the same order `scripts/release_readiness.py` uses (path before PEM). An
//!   explicit `--public-key` passed to the gate is not visible here, and this
//!   runs as its own workflow step, so a *relative* path would resolve against
//!   each process's own working directory. No workflow does either today.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

const PEM_BEGIN: &str = "-----BEGIN PUBLIC KEY-----";
const PEM_END: &str = "-----END PUBLIC KEY-----";

const CURRENT_RELEASE_KEY: &str = "release-current";

/// The key sources, in the order `scripts/release_readiness.py` resolves them.
const CI_PUBLIC_KEY_PATH_ENV: &str = "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH";
const CI_PUBLIC_KEY_PEM_ENV: &str = "ROCM_CLI_SIGNING_PUBLIC_KEY_PEM";

/// How a pinned constant embeds its PEM string, so its value span can be isolated.
#[derive(Clone, Copy)]
enum Embedding {
    /// Shell double-quoted assignment: `NAME="…"` (value runs to the next `"`).
    Shell,
    /// Rust string const: `const NAME: &str = "…";` (value runs to the next `"`).
    Rust,
    /// PowerShell string or here-string: `$Name = "…"` or `$Name = @"…"@`.
    PowerShell,
}

/// A source file plus the specific constant within it that must embed the key.
struct Source {
    path: &'static str,
    /// Identifier that holds the pinned PEM (includes the leading `$` for PowerShell).
    token: &'static str,
    embedding: Embedding,
}

/// A canonical published key and every source that must embed the identical bytes.
struct PinnedKey {
    name: &'static str,
    canonical: &'static str,
    sources: &'static [Source],
}

const PINNED_KEYS: &[PinnedKey] = &[
    PinnedKey {
        name: CURRENT_RELEASE_KEY,
        canonical: "docs/keys/rocm-cli-release-current-public.pem",
        sources: &[
            Source {
                path: "install.sh",
                token: "PINNED_RELEASE_PUBLIC_KEY_CURRENT",
                embedding: Embedding::Shell,
            },
            Source {
                path: "install.ps1",
                token: "$PinnedReleasePublicKeyCurrent",
                embedding: Embedding::PowerShell,
            },
        ],
    },
    PinnedKey {
        name: "release-next",
        canonical: "docs/keys/rocm-cli-release-next-public.pem",
        sources: &[
            Source {
                path: "install.sh",
                token: "PINNED_RELEASE_PUBLIC_KEY_NEXT",
                embedding: Embedding::Shell,
            },
            Source {
                path: "install.ps1",
                token: "$PinnedReleasePublicKeyNext",
                embedding: Embedding::PowerShell,
            },
        ],
    },
    PinnedKey {
        name: "metadata",
        canonical: "docs/keys/rocm-cli-metadata-public.pem",
        sources: &[Source {
            path: "apps/rocm/src/therock/fetch.rs",
            token: "PINNED_METADATA_PUBLIC_KEY_PEM",
            embedding: Embedding::Rust,
        }],
    },
];

/// Reduce arbitrary text to its base64 alphabet, dropping everything else.
///
/// Newline escape sequences are removed first: a literal `\n` (shell/Rust) or
/// `` `n `` (PowerShell) would otherwise leave a stray `n`/`r`/`t` — all
/// base64-alphabet letters — and corrupt the payload. Real newlines are plain
/// whitespace and drop out with everything else non-base64.
fn base64_only(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if (c == '\\' || c == '`') && i + 1 < bytes.len() {
            let next = bytes[i + 1] as char;
            if next == 'n' || next == 'r' || next == 't' {
                i += 2;
                continue;
            }
        }
        if c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// Return the base64 body of the first PUBLIC KEY PEM in `text`, or `None`.
fn pem_body(text: &str) -> Option<String> {
    let begin = text.find(PEM_BEGIN)?;
    let start = begin + PEM_BEGIN.len();
    let end = text[start..].find(PEM_END)? + start;
    let body = base64_only(&text[start..end]);
    if body.is_empty() { None } else { Some(body) }
}

impl Embedding {
    /// What a whole-line comment starts with in this language.
    const fn comment_prefix(self) -> &'static str {
        match self {
            Self::Shell | Self::PowerShell => "#",
            Self::Rust => "//",
        }
    }

    /// Why a second write defeats the comparison, for error messages.
    const fn why_one(self) -> &'static str {
        match self {
            Self::Shell => "the shell keeps the last value it assigns",
            Self::PowerShell => "PowerShell keeps the last value it assigns",
            Self::Rust => "another binding can shadow the constant that is compared",
        }
    }

    /// The one assignment shape the value can be read from, for error messages.
    fn supported_form(self, name: &str) -> String {
        match self {
            Self::Shell => format!("`{name}=\"…\"` as its own statement"),
            Self::PowerShell => {
                format!("`${name} = \"…\"` or `${name} = @\"…\"@` as its own statement")
            }
            Self::Rust => format!("`const {name}: &str = \"…\";` as its own statement"),
        }
    }
}

/// One occurrence of a pinned name that can set its value.
struct Assignment {
    /// 1-based line number, for error messages.
    line: usize,
    /// Byte offset in the whole source just past the name.
    after_name: usize,
    /// Whether it is the supported declaration shape whose value can be read.
    declaration: bool,
}

const fn is_ident_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Whether `line` is `readonly`/`export` followed only by names and flags, which
/// sets attributes without changing any value. `export NAME; unset NAME` is not.
fn is_attribute_command(line: &str) -> bool {
    let code = line.split(" #").next().unwrap_or(line);
    let mut words = code.split_whitespace();
    matches!(words.next(), Some("readonly" | "export"))
        && words.all(|word| {
            word.starts_with('-')
                || word.bytes().all(is_ident_byte) && !word.as_bytes()[0].is_ascii_digit()
        })
}

/// Start offsets of `name` in `line` as a whole identifier: a longer identifier
/// that merely starts or ends with `name` is a different variable.
fn identifier_offsets(line: &str, name: &str, ignore_case: bool) -> Vec<usize> {
    let (haystack, needle) = if ignore_case {
        (line.to_ascii_lowercase(), name.to_ascii_lowercase())
    } else {
        (line.to_owned(), name.to_owned())
    };
    let bytes = haystack.as_bytes();
    haystack
        .match_indices(&needle)
        .map(|(start, _)| start)
        .filter(|&start| {
            let end = start + needle.len();
            (start == 0 || !is_ident_byte(bytes[start - 1]))
                && (end == bytes.len() || !is_ident_byte(bytes[end]))
        })
        .collect()
}

/// Classify one whole-identifier occurrence of `name` at `start` in `line`:
/// `None` if it cannot change the value, otherwise whether it is the supported
/// declaration shape.
fn classify(line: &str, start: usize, name: &str, embedding: Embedding) -> Option<bool> {
    let head = &line[..start];
    let tail = &line[start + name.len()..];
    match embedding {
        Embedding::Shell => {
            // `NAME=`, `NAME+=` (bash) and `${NAME=…}`/`${NAME:=…}` all assign.
            if tail.starts_with('=') || tail.starts_with("+=") || tail.starts_with(":=") {
                let prefix = head.trim();
                let declaration = tail.starts_with('=')
                    && (prefix.is_empty() || prefix == "export" || prefix == "readonly");
                return Some(declaration);
            }
            // `$NAME`, `${NAME}`, `${#NAME}` and `${!NAME}` only read it.
            if ["$", "${", "${#", "${!"].iter().any(|r| head.ends_with(r)) {
                return None;
            }
            // `readonly NAME …` / `export NAME …` naming variables only set
            // attributes.
            if is_attribute_command(line) {
                return None;
            }
            // Any other bare mention is how `unset`, `read`, `getopts` and
            // `printf -v` name the variable; treat it as a write and fail closed.
            Some(false)
        }
        Embedding::PowerShell => {
            // A variable reference is `$Name`, `$scope:Name`, `${Name}` or
            // `${scope:Name}`. Anything else naming the variable — `Set-Variable
            // -Name Name`, `New-Variable Name`, `Set-Item variable:Name` — is
            // treated as a write: the cmdlet forms are too varied to tell apart,
            // and over-counting fails closed.
            let mut reference = head;
            let mut scope = "";
            if let Some(scoped) = reference.strip_suffix(':') {
                reference =
                    scoped.trim_end_matches(|c: char| c.is_ascii_alphanumeric() || c == '_');
                scope = &scoped[reference.len()..];
            }
            let braced = reference.ends_with("${");
            let Some(before) = reference
                .strip_suffix("${")
                .or_else(|| reference.strip_suffix('$'))
            else {
                return Some(false);
            };
            let mut after = tail;
            if braced {
                after = after.strip_prefix('}')?;
            }
            let after = after.trim_start();
            let plain = after.starts_with('=') && !after.starts_with("==");
            let compound = ["+=", "-=", "*=", "/=", "%=", "??="]
                .iter()
                .any(|op| after.starts_with(op));
            if !(plain || compound) {
                return None;
            }
            // Only a variable scope names this variable; `$env:Name` and other
            // drive-qualified names are something else and leave it unset.
            let variable_scope = ["", "script", "global", "local", "private"]
                .iter()
                .any(|s| scope.eq_ignore_ascii_case(s));
            Some(plain && !braced && variable_scope && before.trim().is_empty())
        }
        Embedding::Rust => {
            // `const NAME:`, `static NAME:`, `let NAME =` / `let NAME:` (a shadowing
            // binding) all introduce a value under the name; a path `NAME::`,
            // comparison `NAME ==` or match arm `NAME =>` does not.
            let after = tail.trim_start();
            let binds = (after.starts_with(':') && !after.starts_with("::"))
                || (after.starts_with('=') && !after.starts_with("==") && !after.starts_with("=>"));
            if !binds {
                return None;
            }
            let words: Vec<&str> = head.split_whitespace().collect();
            let declaration = after.starts_with(':')
                && matches!(
                    words.as_slice(),
                    ["const"] | ["pub" | "pub(crate)", "const"]
                );
            Some(declaration)
        }
    }
}

/// Every occurrence of `name` in `source` that can set its value, skipping
/// whole-line comments, plus the lines of `readonly` commands naming it.
/// PowerShell names are case-insensitive.
fn find_assignments(
    source: &str,
    name: &str,
    embedding: Embedding,
) -> (Vec<Assignment>, Vec<usize>) {
    let ignore_case = matches!(embedding, Embedding::PowerShell);
    let mut found = Vec::new();
    let mut readonly_lines = Vec::new();
    let mut offset = 0;
    for (index, line) in source.split_inclusive('\n').enumerate() {
        if !line.trim_start().starts_with(embedding.comment_prefix()) {
            for start in identifier_offsets(line, name, ignore_case) {
                match classify(line, start, name, embedding) {
                    Some(declaration) => found.push(Assignment {
                        line: index + 1,
                        after_name: offset + start + name.len(),
                        declaration,
                    }),
                    None if matches!(embedding, Embedding::Shell)
                        && is_attribute_command(line)
                        && line.split_whitespace().next() == Some("readonly") =>
                    {
                        readonly_lines.push(index + 1);
                    }
                    None => {}
                }
            }
        }
        offset += line.len();
    }
    (found, readonly_lines)
}

/// The single declaration of a pinned constant and the raw text it assigns.
struct Constant<'a> {
    value: &'a str,
    /// 1-based line of the declaration.
    line: usize,
    /// Whether the declaration is `readonly NAME=…` or a later `readonly` command
    /// names it. Only meaningful for the shell.
    readonly: bool,
}

/// Isolate the raw text span assigned to `token` under `embedding`.
///
/// The value is read from the *single* assignment of the name, because that is
/// the one the runtime uses. A first-occurrence search read whatever came first —
/// a commented-out line, a longer identifier sharing the prefix, or an assignment
/// later overwritten — while the shell and PowerShell keep the last value they
/// execute. So zero or several assignments is an error, as is a single one in a
/// shape the value cannot be read from. PEM payloads contain no `"`, so the first
/// quote after the opening one reliably closes the value.
fn extract_constant<'a>(
    source: &'a str,
    token: &str,
    embedding: Embedding,
) -> std::result::Result<Constant<'a>, String> {
    let name = match embedding {
        Embedding::PowerShell => token.strip_prefix('$').unwrap_or(token),
        Embedding::Shell | Embedding::Rust => token,
    };
    let (assignments, readonly_lines) = find_assignments(source, name, embedding);
    let assignment = match assignments.as_slice() {
        [] => return Err(format!("has no assignment of {token}")),
        [one] => one,
        many => {
            let lines = many
                .iter()
                .map(|a| a.line.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "assigns {token} {} times (lines {lines}); {}, so exactly one \
                 assignment is allowed",
                many.len(),
                embedding.why_one()
            ));
        }
    };
    if !assignment.declaration {
        return Err(format!(
            "assigns {token} on line {} in an unsupported form; expected {}",
            assignment.line,
            embedding.supported_form(name)
        ));
    }

    let after = &source[assignment.after_name..];
    let until_quote = |value: &'a str| value.find('"').map(|close| &value[..close]);
    let unreadable = || {
        format!(
            "assigns {token} on line {} to a value that is not a double-quoted string",
            assignment.line
        )
    };
    // The value and the length of the delimiter that closes it: `"@` only for a
    // here-string that opened with `@"`, a plain `"` everywhere else.
    let plain = |value: &'a str| (value, 1);
    let value = match embedding {
        // `NAME="…"` — the quote follows the `=` directly.
        Embedding::Shell => after.strip_prefix("=\"").and_then(until_quote).map(plain),
        // `const NAME: &str = "…";` — skip the type to the `=`, then the quote.
        Embedding::Rust => after
            .find('=')
            .and_then(|eq| after[eq + 1..].trim_start().strip_prefix('"'))
            .and_then(until_quote)
            .map(plain),
        // `$Name = @"…"@` (here-string) or `$Name = "…"`.
        Embedding::PowerShell => after
            .trim_start()
            .strip_prefix('=')
            .map(str::trim_start)
            .and_then(|rhs| match rhs.strip_prefix("@\"") {
                Some(body) => body.find("\"@").map(|close| (&body[..close], 2)),
                None => rhs.strip_prefix('"').and_then(until_quote).map(plain),
            }),
    };
    let (value, closer) = value.ok_or_else(unreadable)?;

    // The statement must end with the string: anything after it (`-replace …`,
    // `+ "…"`, a second quoted word, a command) would change the value the
    // runtime ends up with, so the text compared here would not be it.
    let value_end = value.as_ptr() as usize - source.as_ptr() as usize + value.len();
    let line_rest = source[value_end + closer..].lines().next().unwrap_or("");
    let semicolon = line_rest.trim_start().strip_prefix(';');
    let after_semicolon = semicolon.unwrap_or(line_rest);
    let rest = after_semicolon.trim();
    // A shell or PowerShell comment starts a word: `""#x` is the string `#x`,
    // not an empty string and a comment, while `"" #x` and `"";#x` are.
    let comment = rest.starts_with(embedding.comment_prefix())
        && (matches!(embedding, Embedding::Rust)
            || semicolon.is_some()
            || after_semicolon.starts_with(char::is_whitespace));
    if !(rest.is_empty() || comment) {
        return Err(format!(
            "assigns {token} on line {} to a string followed by `{rest}`; the statement \
             must end with the string, so the value compared is the value used",
            assignment.line
        ));
    }

    let line_start = source[..assignment.after_name]
        .rfind('\n')
        .map_or(0, |n| n + 1);
    Ok(Constant {
        value,
        line: assignment.line,
        readonly: source[line_start..].trim_start().starts_with("readonly")
            || readonly_lines.iter().any(|&line| line > assignment.line),
    })
}

/// Reduce a pinned constant's raw value to the base64 body of its key, or `None`
/// for an empty slot (`""` exactly: the shell tests a slot with `[ -n … ]`, so a
/// lone space would count as a key there).
///
/// The value must be one well-formed PUBLIC KEY PEM block and nothing else: the
/// BEGIN line, non-empty lines of base64, the END line. Comparing a looser
/// reading would pass a key that `openssl` cannot load (stray punctuation, a
/// blank line in the body) or, with a second key prepended through `${…}`, one
/// it loads instead. Outside PowerShell, whose installer trims each line, a line
/// may not start with whitespace either. `$`, backticks and backslashes are
/// refused in the installers,
/// which would expand them or keep them verbatim; Rust may write line breaks as
/// `\n`, since a Rust string literal turns them into real ones.
fn pinned_pem_body(
    value: &str,
    embedding: Embedding,
) -> std::result::Result<Option<String>, String> {
    let text = match embedding {
        Embedding::Rust => value.replace("\\n", "\n"),
        Embedding::Shell | Embedding::PowerShell => value.to_owned(),
    };
    if let Some(found) = text.chars().find(|c| matches!(c, '$' | '`' | '\\')) {
        return Err(format!(
            "holds `{found}`, which the installer would expand or keep verbatim; a \
             pinned key is plain PEM text"
        ));
    }
    if value.is_empty() {
        return Ok(None);
    }
    let malformed = || {
        "is not one well-formed PUBLIC KEY PEM block (a BEGIN line, lines of base64, an \
         END line, and nothing else)"
            .to_owned()
    };
    // install.ps1 trims each line before decoding, so indentation is harmless
    // there. The shell writes the value out verbatim and `openssl` cannot read an
    // indented PEM, so outside PowerShell a line may not start with whitespace.
    if !matches!(embedding, Embedding::PowerShell)
        && text
            .lines()
            .any(|line| line.starts_with(char::is_whitespace))
    {
        return Err(
            "has a key line that starts with whitespace, which `openssl` cannot read".to_owned(),
        );
    }
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    // A here-string opens and closes on its own lines; nothing else may be blank.
    let start = lines
        .iter()
        .position(|line| !line.is_empty())
        .ok_or_else(malformed)?;
    let end = lines
        .iter()
        .rposition(|line| !line.is_empty())
        .ok_or_else(malformed)?
        + 1;
    let [first, body @ .., last] = &lines[start..end] else {
        return Err(malformed());
    };
    let base64_line = |line: &&str| {
        !line.is_empty()
            && line
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '='))
    };
    if *first != PEM_BEGIN || *last != PEM_END || body.is_empty() || !body.iter().all(base64_line) {
        return Err(malformed());
    }
    Ok(Some(body.concat()))
}

/// Check every populated canonical key against its pinned sources. `ci_public_key`
/// is the PEM text CI would verify against, already resolved by
/// [`resolve_ci_public_key`] from either key source, or `None`; it is passed in
/// rather than read here so tests need not mutate the process environment.
fn check_pinned_keys(root: &Path, ci_public_key: Option<&str>) -> Result<Vec<String>> {
    let mut messages = Vec::new();
    let mut current_release_body: Option<String> = None;

    for entry in PINNED_KEYS {
        let canonical_path = root.join(entry.canonical);
        let canonical_body = match std::fs::read_to_string(&canonical_path) {
            Ok(text) if !text.trim().is_empty() => Some(pem_body(&text).with_context(|| {
                format!(
                    "{}: {} is not a valid PUBLIC KEY PEM",
                    entry.name, entry.canonical
                )
            })?),
            _ => None,
        };
        if entry.name == CURRENT_RELEASE_KEY {
            current_release_body.clone_from(&canonical_body);
        }

        for source in entry.sources {
            let source_path = root.join(source.path);
            let source_text = std::fs::read_to_string(&source_path).with_context(|| {
                format!("{}: source file is missing: {}", entry.name, source.path)
            })?;
            let constant = extract_constant(&source_text, source.token, source.embedding)
                .map_err(|reason| anyhow!("{}: {} {reason}", entry.name, source.path))?;
            if matches!(source.embedding, Embedding::Shell) && !constant.readonly {
                bail!(
                    "{}: {} does not mark {} readonly after assigning it on line {}; add \
                     `readonly {}` so no later line can replace the trust root",
                    entry.name,
                    source.path,
                    source.token,
                    constant.line,
                    source.token
                );
            }
            let pinned = pinned_pem_body(constant.value, source.embedding).map_err(|reason| {
                anyhow!(
                    "{}: {} {} on line {} {reason}",
                    entry.name,
                    source.path,
                    source.token,
                    constant.line
                )
            })?;
            match (&canonical_body, pinned) {
                (Some(canonical), Some(pinned)) if pinned == *canonical => {}
                (Some(_), _) => bail!(
                    "{}: {} does not embed the canonical key {} in {} — the pinned \
                     constant and the published key have diverged",
                    entry.name,
                    source.path,
                    entry.canonical,
                    source.token
                ),
                (None, None) => {}
                (None, Some(_)) => bail!(
                    "{}: {} pins a key in {} on line {}, but {} holds no canonical key; \
                     publish the key there or leave the slot empty",
                    entry.name,
                    source.path,
                    source.token,
                    constant.line,
                    entry.canonical
                ),
            }
        }
        let sources = entry
            .sources
            .iter()
            .map(|s| s.path)
            .collect::<Vec<_>>()
            .join(", ");
        messages.push(if canonical_body.is_some() {
            format!("{}: pinned in {sources}", entry.name)
        } else {
            format!(
                "{}: no canonical key yet — slots empty in {sources} (dormant)",
                entry.name
            )
        });
    }

    messages.extend(check_ci_public_key(
        ci_public_key,
        current_release_body.as_deref(),
    )?);
    Ok(messages)
}

/// When a CI signing public key is configured, it must equal the canonical current
/// release key — the key CI verifies release artifacts against.
///
/// An absent key is reported rather than passed over in silence. This check cannot
/// *demand* one: `ci.yml` never exposes the signing secret, so requiring it would
/// fail every PR. But "no key configured" and "key matches" must not look alike in
/// the log, or a secret that was rotated away reads as a clean run.
fn check_ci_public_key(
    ci_public_key: Option<&str>,
    current_release_body: Option<&str>,
) -> Result<Vec<String>> {
    let env_pem = ci_public_key.unwrap_or_default();
    if env_pem.trim().is_empty() {
        return Ok(vec![
            "ci public key: not configured — cross-check skipped".to_owned(),
        ]);
    }
    let env_body = pem_body(env_pem)
        .context("the configured CI signing public key is not a valid PUBLIC KEY PEM")?;
    let Some(current) = current_release_body else {
        return Ok(vec![
            "ci public key: configured, but no canonical current release key to compare \
             against yet — skipped"
                .to_owned(),
        ]);
    };
    if env_body != current {
        bail!(
            "the CI signing public key does not match the canonical current release key \
             ({}); CI would verify against a different key than installers pin",
            PINNED_KEYS[0].canonical
        );
    }
    Ok(vec![
        "ci public key: matches the canonical current release key".to_owned(),
    ])
}

/// Resolve the CI signing public key the way `scripts/release_readiness.py`'s
/// `resolve_signing_key` does: the file named by `ROCM_CLI_SIGNING_PUBLIC_KEY_PATH`
/// first, then the inline `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM`.
///
/// Reading only the PEM would make the cross-check skip silently whenever the key
/// was supplied by path, while the readiness gate verified release artifacts
/// against that very key — a trust root nobody compared with `docs/keys/`, and
/// reported in the log as "not configured".
///
/// An empty variable counts as absent; a variable naming a missing or empty file
/// is an error. The blank case matches the Python gate exactly: `env_path` and
/// `env_text` both `strip()` the value and treat whatever is left of nothing as
/// unset, which is the shape an unset GitHub secret takes when it expands to the
/// empty string. Erroring on a whitespace-only path here while the gate fell
/// through to the PEM would make the two disagree about which key is in use.
///
/// Once a path *is* named, a file that cannot be read or that holds nothing is an
/// error rather than a fall-through: the operator named a key, so reporting "not
/// configured" and exiting 0 would be the exact confusion the doc on
/// [`check_ci_public_key`] forbids.
///
/// `lookup` is injected so the variable names themselves are covered: reading
/// them directly here left callers free to query a misspelled name with every
/// test still green.
fn resolve_ci_public_key(lookup: impl Fn(&str) -> Option<OsString>) -> Result<Option<String>> {
    if let Some(raw) = lookup(CI_PUBLIC_KEY_PATH_ENV) {
        // `OsString`, not `String`: a non-UTF-8 path must still name the file the
        // operator meant, not silently fall through to the inline PEM. Lossy
        // replacement characters are not whitespace, so such a path still counts
        // as named.
        if !raw.to_string_lossy().trim().is_empty() {
            let path = PathBuf::from(&raw);
            let pem = std::fs::read_to_string(&path).with_context(|| {
                format!(
                    "failed to read the CI signing public key named by \
                     {CI_PUBLIC_KEY_PATH_ENV}: {}",
                    path.display()
                )
            })?;
            if pem.trim().is_empty() {
                bail!(
                    "the CI signing public key named by {CI_PUBLIC_KEY_PATH_ENV} is \
                     empty: {}; a configured key that holds nothing must not be \
                     reported as 'not configured'",
                    path.display()
                );
            }
            return Ok(Some(pem));
        }
    }
    Ok(lookup(CI_PUBLIC_KEY_PEM_ENV)
        .map(|raw| raw.to_string_lossy().into_owned())
        .filter(|value| !value.trim().is_empty()))
}

fn repo_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is the `xtask/` crate dir; its parent is the repo root,
    // so this is correct regardless of the working directory CI invokes us from.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate has a parent directory")
        .to_path_buf()
}

/// A named shim rather than a closure in [`run`]: the generic `std::env::var_os`
/// fn item cannot satisfy the higher-ranked `Fn(&str)` bound, and a closure
/// spelled out at the call site is a place a wrong variable name could live
/// where no test could reach it. Here it is a pass-through with nothing to get
/// wrong, and `env_lookup_passes_the_name_through` pins that it stays one.
fn env_lookup(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// Resolve the CI key through `lookup`, then run every pinned-key check against
/// `root`, returning the lines the caller should print.
///
/// Split out of [`run`] so the resolution-to-comparison handoff is testable: with
/// it inlined there, a `run` that queried the wrong variable, resolved nothing,
/// or simply handed `check_pinned_keys` a `None` left every test green while the
/// cross-check silently degraded to "not configured".
fn run_with(root: &Path, lookup: impl Fn(&str) -> Option<OsString>) -> Result<Vec<String>> {
    let ci_public_key = resolve_ci_public_key(lookup)?;
    check_pinned_keys(root, ci_public_key.as_deref())
}

pub fn run() -> Result<()> {
    for message in run_with(&repo_root(), env_lookup)? {
        println!("pinned key check: {message}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "-----BEGIN PUBLIC KEY-----\n\
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtesttesttesttest+/\n\
        abcDEF0123456789testtesttesttesttesttesttesttesttesttestABCD==\n\
        -----END PUBLIC KEY-----\n";

    #[test]
    fn base64_only_strips_escapes_and_non_alphabet() {
        assert_eq!(base64_only("AB\\nCD `n EF-=+/"), "ABCDEF=+/");
    }

    #[test]
    fn pem_body_extracts_inner_base64() {
        let body = pem_body(SAMPLE).expect("valid pem");
        assert!(body.starts_with("MIIBIjAN"));
        assert!(!body.contains("BEGIN"));
        assert_eq!(pem_body("not a pem"), None);
    }

    #[test]
    fn extract_constant_isolates_each_syntax() {
        let shell = "PINNED_RELEASE_PUBLIC_KEY_CURRENT=\"the-current\"\n\
                     PINNED_RELEASE_PUBLIC_KEY_NEXT=\"the-next\"\n";
        assert_eq!(
            extract_constant(shell, "PINNED_RELEASE_PUBLIC_KEY_CURRENT", Embedding::Shell)
                .map(|c| c.value),
            Ok("the-current")
        );
        assert_eq!(
            extract_constant(shell, "PINNED_RELEASE_PUBLIC_KEY_NEXT", Embedding::Shell)
                .map(|c| c.value),
            Ok("the-next")
        );

        let rust = "const PINNED_METADATA_PUBLIC_KEY_PEM: &str = \"the-meta\";\n";
        assert_eq!(
            extract_constant(rust, "PINNED_METADATA_PUBLIC_KEY_PEM", Embedding::Rust)
                .map(|c| c.value),
            Ok("the-meta")
        );

        let ps = "$PinnedReleasePublicKeyCurrent = @\"\nthe-here\n\"@\n\
                  $PinnedReleasePublicKeyNext = \"the-plain\"\n";
        assert_eq!(
            extract_constant(ps, "$PinnedReleasePublicKeyCurrent", Embedding::PowerShell)
                .map(|c| c.value),
            Ok("\nthe-here\n")
        );
        assert_eq!(
            extract_constant(ps, "$PinnedReleasePublicKeyNext", Embedding::PowerShell)
                .map(|c| c.value),
            Ok("the-plain")
        );
    }

    #[test]
    fn extract_targets_the_named_constant_not_the_whole_file() {
        // The bug: a wrong CURRENT passes if the real key is anywhere in the file.
        // Here CURRENT holds a wrong key while NEXT holds the real one; extracting
        // CURRENT must return the wrong value, not the file's real key.
        let shell = format!(
            "PINNED_RELEASE_PUBLIC_KEY_CURRENT=\"{}\"\nPINNED_RELEASE_PUBLIC_KEY_NEXT=\"{}\"\n",
            SAMPLE.replace("test", "wrong"),
            SAMPLE
        );
        let current = extract_constant(
            &shell,
            "PINNED_RELEASE_PUBLIC_KEY_CURRENT",
            Embedding::Shell,
        )
        .ok()
        .and_then(|c| pem_body(c.value))
        .unwrap();
        assert_ne!(current, pem_body(SAMPLE).unwrap());
    }

    /// A second and third distinct key, used as the canonical next-release and
    /// metadata keys.
    const NEXT_KEY: &str = "-----BEGIN PUBLIC KEY-----\n\
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAnextnextnextnext+/\n\
        abcDEF0123456789nextnextnextnextnextnextnextnextnextnextABCD==\n\
        -----END PUBLIC KEY-----\n";
    const META: &str = "-----BEGIN PUBLIC KEY-----\n\
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmetametametameta+/\n\
        abcDEF0123456789metametametametametametametametametametaABCD==\n\
        -----END PUBLIC KEY-----\n";

    const SH_CURRENT: &str = "PINNED_RELEASE_PUBLIC_KEY_CURRENT";
    const PS_CURRENT: &str = "$PinnedReleasePublicKeyCurrent";
    const RS_META: &str = "PINNED_METADATA_PUBLIC_KEY_PEM";

    /// A scratch repository with all three canonical keys published, so every
    /// slot is live rather than dormant.
    fn scratch_repo(install_sh: &str, install_ps1: &str, therock_rs: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("docs/keys")).expect("mkdir docs/keys");
        std::fs::create_dir_all(root.join("apps/rocm/src")).expect("mkdir apps/rocm/src");
        for (file, key) in [
            ("rocm-cli-release-current-public.pem", SAMPLE),
            ("rocm-cli-release-next-public.pem", NEXT_KEY),
            ("rocm-cli-metadata-public.pem", META),
        ] {
            std::fs::write(root.join("docs/keys").join(file), key).expect("write canonical key");
        }
        std::fs::write(root.join("install.sh"), install_sh).expect("write install.sh");
        std::fs::write(root.join("install.ps1"), install_ps1).expect("write install.ps1");
        std::fs::write(root.join("apps/rocm/src/therock.rs"), therock_rs)
            .expect("write therock.rs");
        dir
    }

    /// A key that differs from every canonical key in its base64 body.
    fn wrong() -> String {
        SAMPLE.replace("test", "evil")
    }

    /// An install.sh whose CURRENT slot is `current` (the caller's lines), with a
    /// correct NEXT slot and both marked readonly after them.
    fn sh(current: &str) -> String {
        format!(
            "{current}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n\
             readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n"
        )
    }

    fn sh_current() -> String {
        format!("PINNED_RELEASE_PUBLIC_KEY_CURRENT=\"{SAMPLE}\"\n")
    }

    fn good_sh() -> String {
        sh(&sh_current())
    }

    /// An install.ps1 whose CURRENT slot is `current`, with a correct NEXT slot.
    fn ps1(current: &str) -> String {
        format!("{current}$PinnedReleasePublicKeyNext = @\"\n{NEXT_KEY}\"@\n")
    }

    fn ps1_current() -> String {
        format!("$PinnedReleasePublicKeyCurrent = @\"\n{SAMPLE}\"@\n")
    }

    fn good_ps1() -> String {
        ps1(&ps1_current())
    }

    fn good_rs() -> String {
        format!("const PINNED_METADATA_PUBLIC_KEY_PEM: &str = \"{META}\";\n")
    }

    /// Run the check over a scratch repo, require it to fail, and require the
    /// error to contain every `expected` fragment, so a test cannot pass on an
    /// unrelated failure in another slot.
    #[track_caller]
    fn rejected(install_sh: &str, install_ps1: &str, therock_rs: &str, expected: &[&str]) {
        let repo = scratch_repo(install_sh, install_ps1, therock_rs);
        let error = match check_pinned_keys(repo.path(), None) {
            Ok(messages) => panic!(
                "the check passed, but the source trusts a different key than \
                 docs/keys; reported {messages:?}"
            ),
            Err(error) => format!("{error:#}"),
        };
        for fragment in expected {
            assert!(
                error.contains(fragment),
                "expected {fragment:?} in: {error}"
            );
        }
    }

    #[track_caller]
    fn accepted(install_sh: &str, install_ps1: &str, therock_rs: &str) -> Vec<String> {
        let repo = scratch_repo(install_sh, install_ps1, therock_rs);
        check_pinned_keys(repo.path(), None).unwrap_or_else(|error| panic!("{error:#}"))
    }

    #[test]
    fn scratch_repo_with_matching_keys_passes() {
        let messages = accepted(&good_sh(), &good_ps1(), &good_rs());
        for expected in [
            "release-current: pinned in install.sh, install.ps1",
            "release-next: pinned in install.sh, install.ps1",
            "metadata: pinned in apps/rocm/src/therock.rs",
        ] {
            assert!(messages.iter().any(|m| m == expected), "{messages:?}");
        }
    }

    /// A slot without a canonical key must be empty: the installers trust any
    /// non-empty slot, so a key pasted there would be trusted with nothing to
    /// compare it against.
    #[test]
    fn a_dormant_slot_must_be_empty() {
        let empty_next = |sh_next: &str, ps_next: &str, rs: &str| {
            let sh = format!(
                "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{sh_next}\"\n\
                 readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
                sh_current()
            );
            let ps = format!(
                "{}$PinnedReleasePublicKeyNext = \"{ps_next}\"\n",
                ps1_current()
            );
            let repo = scratch_repo(&sh, &ps, rs);
            for file in [
                "rocm-cli-release-next-public.pem",
                "rocm-cli-metadata-public.pem",
            ] {
                std::fs::remove_file(repo.path().join("docs/keys").join(file)).expect("rm");
            }
            check_pinned_keys(repo.path(), None).map_err(|error| format!("{error:#}"))
        };
        let messages = empty_next(
            "",
            "",
            "const PINNED_METADATA_PUBLIC_KEY_PEM: &str = \"\";\n",
        )
        .expect("empty dormant slots pass");
        assert!(
            messages.iter().any(|m| m
                == "release-next: no canonical key yet — slots empty in install.sh, \
                    install.ps1 (dormant)"),
            "{messages:?}"
        );

        let error = empty_next(
            NEXT_KEY,
            "",
            "const PINNED_METADATA_PUBLIC_KEY_PEM: &str = \"\";\n",
        )
        .expect_err("a key pasted into a dormant shell slot must fail");
        assert!(
            error.contains("install.sh pins a key in PINNED_RELEASE_PUBLIC_KEY_NEXT on line 6")
                && error.contains("holds no canonical key"),
            "{error}"
        );
        let error = empty_next(
            "",
            NEXT_KEY,
            "const PINNED_METADATA_PUBLIC_KEY_PEM: &str = \"\";\n",
        )
        .expect_err("a key pasted into a dormant PowerShell slot must fail");
        assert!(
            error.contains("install.ps1 pins a key in $PinnedReleasePublicKeyNext"),
            "{error}"
        );
        let error = empty_next("", "", &good_rs())
            .expect_err("a key pasted into a dormant Rust slot must fail");
        assert!(
            error.contains("apps/rocm/src/therock.rs pins a key in PINNED_METADATA_PUBLIC_KEY_PEM"),
            "{error}"
        );
    }

    /// A commented-out line holding the canonical key, above the real assignment
    /// holding another key. The runtime never executes the comment, so the value
    /// compared must be the real assignment's.
    #[test]
    fn shell_commented_canonical_above_a_different_assignment_is_rejected() {
        let current = format!(
            "# {SH_CURRENT}=\"{SAMPLE}\"\n{SH_CURRENT}=\"{}\"\n",
            wrong()
        );
        rejected(
            &sh(&current),
            &good_ps1(),
            &good_rs(),
            &["install.sh does not embed the canonical key", SH_CURRENT],
        );
    }

    #[test]
    fn powershell_commented_canonical_above_a_different_assignment_is_rejected() {
        let current = format!(
            "# {PS_CURRENT} = @\"\n{SAMPLE}\"@\n{PS_CURRENT} = @\"\n{}\"@\n",
            wrong()
        );
        rejected(
            &good_sh(),
            &ps1(&current),
            &good_rs(),
            &["install.ps1 does not embed the canonical key", PS_CURRENT],
        );
    }

    #[test]
    fn rust_commented_canonical_above_a_different_const_is_rejected() {
        let rs = format!(
            "// const {RS_META}: &str = \"{META}\";\nconst {RS_META}: &str = \"{}\";\n",
            wrong()
        );
        rejected(
            &good_sh(),
            &good_ps1(),
            &rs,
            &[
                "apps/rocm/src/therock.rs does not embed the canonical key",
                RS_META,
            ],
        );
    }

    /// The canonical key assigned first and a different one assigned later: the
    /// shell keeps the last value it executed.
    #[test]
    fn shell_reassignment_after_the_canonical_key_is_rejected() {
        let current = format!("{}{SH_CURRENT}=\"{}\"\n", sh_current(), wrong());
        rejected(
            &sh(&current),
            &good_ps1(),
            &good_rs(),
            &["install.sh assigns PINNED_RELEASE_PUBLIC_KEY_CURRENT 2 times (lines 1, 6)"],
        );
    }

    #[test]
    fn powershell_reassignment_after_the_canonical_key_is_rejected() {
        let current = format!("{}{PS_CURRENT} = \"{}\"\n", ps1_current(), wrong());
        rejected(
            &good_sh(),
            &ps1(&current),
            &good_rs(),
            &["install.ps1 assigns $PinnedReleasePublicKeyCurrent 2 times (lines 1, 7)"],
        );
    }

    /// A longer identifier that merely starts with the token is a different
    /// variable, and must not stand in for the real assignment after it.
    #[test]
    fn shell_identifier_sharing_the_token_as_a_prefix_is_not_the_assignment() {
        let current = format!(
            "{SH_CURRENT}_FPR=\"{SAMPLE}\"\n{SH_CURRENT}=\"{}\"\n",
            wrong()
        );
        rejected(
            &sh(&current),
            &good_ps1(),
            &good_rs(),
            &["install.sh does not embed the canonical key", SH_CURRENT],
        );
    }

    #[test]
    fn powershell_identifier_sharing_the_token_as_a_prefix_is_not_the_assignment() {
        let current = format!(
            "{PS_CURRENT}Fpr = \"{SAMPLE}\"\n{PS_CURRENT} = \"{}\"\n",
            wrong()
        );
        rejected(
            &good_sh(),
            &ps1(&current),
            &good_rs(),
            &["install.ps1 does not embed the canonical key", PS_CURRENT],
        );
    }

    /// PowerShell variable names are case-insensitive, so a reassignment spelled
    /// in another case is the same variable.
    #[test]
    fn powershell_reassignment_in_a_different_case_is_rejected() {
        let current = format!(
            "{}$pinnedreleasepublickeycurrent = \"{}\"\n",
            ps1_current(),
            wrong()
        );
        rejected(
            &good_sh(),
            &ps1(&current),
            &good_rs(),
            &["install.ps1", PS_CURRENT, "2 times"],
        );
    }

    /// Comments, expansions and other reads of the name around the single real
    /// assignment are not assignments and must not disturb the check; nor are
    /// longer identifiers that contain the name.
    #[test]
    fn mentions_and_uses_around_the_single_assignment_pass() {
        let sh = format!(
            "# {SH_CURRENT} is the \"current\" trust root.\n\
             # {SH_CURRENT}=\"a retired key\"\n\
             {}\
             [ -n \"${{{SH_CURRENT}}}\" ] || exit 1\n\
             [ ${{#{SH_CURRENT}}} -gt 0 ] || exit 1\n\
             : ${{!{SH_CURRENT}}} \"${{!{SH_CURRENT}}}\"\n\
             for pem in \"${SH_CURRENT}\" \"${{PINNED_RELEASE_PUBLIC_KEY_NEXT}}\"; do :; done\n\
             export PINNED_RELEASE_PUBLIC_KEY_NEXT\n\
             {SH_CURRENT}_FPR=\"other\"\n\
             OLD_{SH_CURRENT}=\"other\"\n",
            good_sh()
        );
        let ps1 = format!(
            "# {PS_CURRENT} holds the \"current\" trust root.\n\
             # {PS_CURRENT} = \"a retired key\"\n\
             {}\
             if ([string]::IsNullOrWhiteSpace({PS_CURRENT})) {{ exit 1 }}\n\
             foreach ($p in @({PS_CURRENT}, $PinnedReleasePublicKeyNext)) {{ }}\n\
             $copy = {PS_CURRENT}\n\
             $trimmed = \"$({PS_CURRENT}.Trim())\"\n\
             if ({PS_CURRENT} -eq \"\") {{ }}\n\
             {PS_CURRENT}Fpr = \"other\"\n\
             Set-Variable -Name OldPinnedReleasePublicKeyCurrent -Value \"other\"\n\
             Set-Variable -Name PinnedReleasePublicKeyCurrentFpr -Value \"other\"\n",
            good_ps1()
        );
        let rs = format!(
            "// {RS_META} stays \"empty\" until published.\n\
             // const {RS_META}: &str = \"a retired key\";\n\
             {}\
             fn key() -> &'static str {{ {RS_META}.trim() }}\n\
             fn empty() -> bool {{ {RS_META} == \"\" }}\n\
             fn arm(k: &str) {{ match k {{ {RS_META} => {{}} _ => {{}} }} }}\n\
             const QUOTE: char = '\"';\n\
             const OLD_{RS_META}: &str = \"other\";\n",
            good_rs()
        );
        accepted(&sh, &ps1, &rs);
    }

    /// Every shape that writes the variable counts towards the single-assignment
    /// rule, before or after the canonical declaration.
    #[test]
    fn every_write_form_counts_as_an_assignment() {
        let w = wrong();
        let shell_writes = [
            format!("export {SH_CURRENT}=\"{w}\"\n"),
            format!("readonly {SH_CURRENT}=\"{w}\"\n"),
            format!("  {SH_CURRENT}=\"{w}\"\n"),
            format!("true && {SH_CURRENT}=\"{w}\"\n"),
            format!(": \"${{{SH_CURRENT}:={w}}}\"\n"),
            format!("{SH_CURRENT}+=\"x\"\n"),
            format!("unset {SH_CURRENT}\n"),
            format!("read -r {SH_CURRENT} < /dev/null\n"),
            format!("export {SH_CURRENT}; unset {SH_CURRENT}\n"),
            format!("echo \"$(unset {SH_CURRENT})\"\n"),
            format!("[ ${{#{SH_CURRENT}}} -gt 0 ] && unset {SH_CURRENT}\n"),
            // Quoting the name does not stop `unset` or `read` from writing it.
            format!("unset '{SH_CURRENT}'\n"),
            format!("read \"{SH_CURRENT}\" < /dev/null\n"),
        ];
        for write in &shell_writes {
            for current in [
                format!("{}{write}", sh_current()),
                format!("{write}{}", sh_current()),
            ] {
                rejected(
                    &sh(&current),
                    &good_ps1(),
                    &good_rs(),
                    &["install.sh", SH_CURRENT, " times (lines "],
                );
            }
        }

        let mut powershell_writes = vec![
            format!("$script:PinnedReleasePublicKeyCurrent = \"{w}\"\n"),
            format!("$Global:PINNEDRELEASEPUBLICKEYCURRENT=\"{w}\"\n"),
            format!("${{PinnedReleasePublicKeyCurrent}} = \"{w}\"\n"),
            format!("if ($true) {{ {PS_CURRENT} = \"{w}\" }}\n"),
            format!("Set-Variable -Name PinnedReleasePublicKeyCurrent -Value \"{w}\"\n"),
            format!("New-Variable PinnedReleasePublicKeyCurrent \"{w}\" -Force\n"),
            format!("Set-Item variable:PinnedReleasePublicKeyCurrent \"{w}\"\n"),
            "Remove-Variable PinnedReleasePublicKeyCurrent\n".to_owned(),
            // Quoting the name does not stop a cmdlet from writing it.
            format!("Set-Variable -Name 'PinnedReleasePublicKeyCurrent' -Value '{w}'\n"),
            format!("Set-Variable -Name \"PinnedReleasePublicKeyCurrent\" -Value \"{w}\"\n"),
            format!("Set-Item 'variable:PinnedReleasePublicKeyCurrent' '{w}'\n"),
            // A drive-qualified name is not this variable, but counting it fails closed.
            format!("$env:PinnedReleasePublicKeyCurrent = \"{w}\"\n"),
        ];
        powershell_writes.extend(
            ["+=", "-=", "*=", "/=", "%=", "??="]
                .iter()
                .map(|op| format!("{PS_CURRENT} {op} \"x\"\n")),
        );
        for write in &powershell_writes {
            for current in [
                format!("{}{write}", ps1_current()),
                format!("{write}{}", ps1_current()),
            ] {
                rejected(
                    &good_sh(),
                    &ps1(&current),
                    &good_rs(),
                    &["install.ps1", PS_CURRENT, "2 times"],
                );
            }
        }

        for write in [
            format!("fn f() {{ let {RS_META} = \"{w}\"; }}\n"),
            format!("static {RS_META}: &str = \"{w}\";\n"),
            format!("mod m {{ const {RS_META}: &str = \"{w}\"; }}\n"),
            format!("/*\nconst {RS_META}: &str = \"{w}\";\n*/\n"),
        ] {
            for rs in [
                format!("{}{write}", good_rs()),
                format!("{write}{}", good_rs()),
            ] {
                rejected(
                    &good_sh(),
                    &good_ps1(),
                    &rs,
                    &["apps/rocm/src/therock.rs", RS_META, "2 times"],
                );
            }
        }
    }

    /// The declaration shapes the value is read from, beyond the plain ones.
    #[test]
    fn supported_declaration_forms_are_read() {
        // `readonly NAME=` marks itself; `export NAME=` still needs the readonly line.
        let sh = format!(
            "readonly {SH_CURRENT}=\"{SAMPLE}\"\n\
             export PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n\
             readonly PINNED_RELEASE_PUBLIC_KEY_NEXT\n"
        );
        let ps1 = format!(
            "$script:pinnedReleasePublicKeyCurrent = \"{SAMPLE}\"\n\
             $Global:PinnedReleasePublicKeyNext = \"{NEXT_KEY}\" # rotation slot\n"
        );
        for rs in [
            format!("pub const {RS_META}: &str = \"{META}\"; // metadata trust root\n"),
            format!("pub(crate) const {RS_META}: &str = \"{META}\";\n"),
        ] {
            accepted(&sh, &ps1, &rs);
        }
        for scope in ["local", "private", "SCRIPT", "global"] {
            accepted(
                &good_sh(),
                &good_ps1().replacen(
                    "$PinnedReleasePublicKeyCurrent = ",
                    &format!("${scope}:PinnedReleasePublicKeyCurrent = "),
                    1,
                ),
                &good_rs(),
            );
        }
        // CRLF line endings, as a Windows checkout of install.ps1 may have.
        accepted(&good_sh(), &good_ps1().replace('\n', "\r\n"), &good_rs());
    }

    /// No assignment at all, or a single one the value cannot be read from, is an
    /// error naming the file and the variable rather than a silent mismatch.
    #[test]
    fn missing_or_unreadable_assignment_is_reported() {
        rejected(
            &sh(""),
            &good_ps1(),
            &good_rs(),
            &["install.sh has no assignment of PINNED_RELEASE_PUBLIC_KEY_CURRENT"],
        );

        // A lone write in a shape the value cannot be read from: conditional,
        // compound, braced, a cmdlet, or a drive-qualified name such as `$env:`
        // that is not this variable at all.
        for lone in [
            format!("Set-Variable -Name PinnedReleasePublicKeyCurrent -Value \"{SAMPLE}\"\n"),
            format!("if ($true) {{ {PS_CURRENT} = \"{SAMPLE}\" }}\n"),
            format!("{PS_CURRENT} += \"{SAMPLE}\"\n"),
            format!("${{PinnedReleasePublicKeyCurrent}} = \"{SAMPLE}\"\n"),
            format!("$env:PinnedReleasePublicKeyCurrent = \"{SAMPLE}\"\n"),
            format!("$function:PinnedReleasePublicKeyCurrent = \"{SAMPLE}\"\n"),
        ] {
            rejected(
                &good_sh(),
                &ps1(&lone),
                &good_rs(),
                &[
                    "install.ps1 assigns $PinnedReleasePublicKeyCurrent on line",
                    "unsupported form",
                ],
            );
        }
        for lone in [
            format!("[ -n \"$CI\" ] && {SH_CURRENT}=\"{SAMPLE}\"\n"),
            format!(": \"${{{SH_CURRENT}:={SAMPLE}}}\"\n"),
            format!("{SH_CURRENT}+=\"{SAMPLE}\"\n"),
        ] {
            rejected(
                &sh(&lone),
                &good_ps1(),
                &good_rs(),
                &[
                    "install.sh assigns PINNED_RELEASE_PUBLIC_KEY_CURRENT on line",
                    "unsupported form",
                ],
            );
        }
        for rs in [
            format!("fn f() {{ let {RS_META}: &str = \"{META}\"; }}\n"),
            format!("static {RS_META}: &str = \"{META}\";\n"),
        ] {
            rejected(
                &good_sh(),
                &good_ps1(),
                &rs,
                &["therock.rs assigns", "unsupported form"],
            );
        }

        rejected(
            &good_sh(),
            &ps1(&format!("{PS_CURRENT} = '{SAMPLE}'\n")),
            &good_rs(),
            &["not a double-quoted string"],
        );
        rejected(
            &sh(&format!("{SH_CURRENT}=$(printf '%s' \"{SAMPLE}\")\n")),
            &good_ps1(),
            &good_rs(),
            &["not a double-quoted string"],
        );
    }

    /// The value must be the key the runtime sees: one well-formed PEM block,
    /// and nothing the installer would expand or keep verbatim.
    #[test]
    fn the_value_must_be_exactly_the_pem_the_runtime_sees() {
        const MALFORMED: &str = "is not one well-formed PUBLIC KEY PEM block";
        // A key prepended through an expansion: openssl uses the first PEM.
        rejected(
            &sh(&format!("{SH_CURRENT}=\"${{EXTRA_PEM}}{SAMPLE}\"\n")),
            &good_ps1(),
            &good_rs(),
            &["install.sh PINNED_RELEASE_PUBLIC_KEY_CURRENT on line 1 holds `$`"],
        );
        rejected(
            &sh(&format!("{SH_CURRENT}=\"`cat extra.pem`{SAMPLE}\"\n")),
            &good_ps1(),
            &good_rs(),
            &["install.sh PINNED_RELEASE_PUBLIC_KEY_CURRENT on line 1 holds ```, which"],
        );
        rejected(
            &good_sh(),
            &ps1(&format!("{PS_CURRENT} = @\"\n$($extra){SAMPLE}\"@\n")),
            &good_rs(),
            &["install.ps1 $PinnedReleasePublicKeyCurrent on line 1 holds `$`"],
        );
        // In a shell double-quoted string `\n` stays a backslash and an `n`.
        rejected(
            &sh(&format!(
                "{SH_CURRENT}=\"{}\"\n",
                SAMPLE.replace('\n', "\\n")
            )),
            &good_ps1(),
            &good_rs(),
            &["holds `\\`"],
        );
        // Accidental edits that leave the base64 letters intact but a key openssl
        // cannot load, or a second block after the first.
        let body_line = SAMPLE.lines().nth(1).expect("a body line");
        for broken in [
            SAMPLE.replacen("MIIBIjAN", "MIIBIjAN.-,;", 1),
            SAMPLE.replacen(body_line, &format!("{body_line}\n"), 1),
            SAMPLE.replacen(body_line, &format!("{body_line}\n----"), 1),
            format!("{SAMPLE}{}", wrong()),
            format!("junk\n{SAMPLE}"),
            format!("{SAMPLE}junk"),
            format!("{PEM_BEGIN}\n{PEM_END}\n"),
            // A different key type's armour around the right body.
            SAMPLE.replace(PEM_BEGIN, "-----BEGIN RSA PUBLIC KEY-----"),
            SAMPLE.replace(PEM_END, "-----END RSA PUBLIC KEY-----"),
        ] {
            rejected(
                &sh(&format!("{SH_CURRENT}=\"{broken}\"\n")),
                &good_ps1(),
                &good_rs(),
                &[
                    "install.sh PINNED_RELEASE_PUBLIC_KEY_CURRENT on line 1",
                    MALFORMED,
                ],
            );
        }
        rejected(
            &good_sh(),
            &ps1(&format!(
                "{PS_CURRENT} = @\"\n{}\"@\n",
                SAMPLE.replacen("MIIB", "MI;IB", 1)
            )),
            &good_rs(),
            &[
                "install.ps1 $PinnedReleasePublicKeyCurrent on line 1",
                MALFORMED,
            ],
        );
        // A key cut short is well formed but not the canonical key.
        let truncated = SAMPLE.replacen(&format!("{body_line}\n"), "", 1);
        rejected(
            &sh(&format!("{SH_CURRENT}=\"{truncated}\"\n")),
            &good_ps1(),
            &good_rs(),
            &["install.sh does not embed the canonical key"],
        );
        // An indented key: openssl cannot read it from the file install.sh writes,
        // while install.ps1 trims each line before decoding.
        for indent in ["  ", "\t"] {
            let indented = SAMPLE.replace('\n', &format!("\n{indent}"));
            let indented = indented.trim_end();
            rejected(
                &sh(&format!("{SH_CURRENT}=\"{indent}{indented}\n\"\n")),
                &good_ps1(),
                &good_rs(),
                &[
                    "install.sh PINNED_RELEASE_PUBLIC_KEY_CURRENT on line 1 has a key line that starts with whitespace",
                ],
            );
            rejected(
                &good_sh(),
                &good_ps1(),
                &format!(
                    "const {RS_META}: &str = \"{}\";\n",
                    META.replace('\n', &format!("\n{indent}")).trim_end()
                ),
                &[
                    "therock.rs PINNED_METADATA_PUBLIC_KEY_PEM on line 1 has a key line that starts with whitespace",
                ],
            );
            accepted(
                &good_sh(),
                &ps1(&format!("{PS_CURRENT} = @\"\n{indent}{indented}\n\"@\n")),
                &good_rs(),
            );
        }
        // Rust turns `\n` into a line break, so that form is the same key; any
        // other escape is not.
        accepted(
            &good_sh(),
            &good_ps1(),
            &format!(
                "const {RS_META}: &str = \"{}\";\n",
                META.replace('\n', "\\n")
            ),
        );
        rejected(
            &good_sh(),
            &good_ps1(),
            &format!(
                "const {RS_META}: &str = \"{}\";\n",
                META.replace('\n', "\\t")
            ),
            &["therock.rs PINNED_METADATA_PUBLIC_KEY_PEM on line 1 holds `\\`"],
        );
    }

    /// Publishing a canonical key while a slot is still empty must fail: an empty
    /// slot turns signature verification off in the installers.
    #[test]
    fn a_published_key_with_an_empty_slot_is_rejected() {
        rejected(
            &sh(&format!("{SH_CURRENT}=\"\"\n")),
            &good_ps1(),
            &good_rs(),
            &["install.sh does not embed the canonical key", SH_CURRENT],
        );
        rejected(
            &good_sh(),
            &ps1(&format!("{PS_CURRENT} = \"\"\n")),
            &good_rs(),
            &["install.ps1 does not embed the canonical key", PS_CURRENT],
        );
        rejected(
            &good_sh(),
            &good_ps1(),
            &format!("const {RS_META}: &str = \"\";\n"),
            &["therock.rs does not embed the canonical key", RS_META],
        );
    }

    /// A canonical file that is empty means "not published yet"; one that holds
    /// something other than a key is an error rather than a silent skip.
    #[test]
    fn canonical_files_are_read_strictly() {
        let repo = scratch_repo(&good_sh(), &good_ps1(), &good_rs());
        let next = repo
            .path()
            .join("docs/keys/rocm-cli-release-next-public.pem");
        std::fs::write(&next, "not a key\n").expect("write garbage");
        let error = format!(
            "{:#}",
            check_pinned_keys(repo.path(), None).expect_err("garbage")
        );
        assert!(
            error.contains(
                "docs/keys/rocm-cli-release-next-public.pem is not a valid PUBLIC KEY PEM"
            ),
            "{error}"
        );
        std::fs::write(&next, "  \n").expect("write blank");
        let error = format!(
            "{:#}",
            check_pinned_keys(repo.path(), None).expect_err("blank")
        );
        assert!(
            error.contains("pins a key in PINNED_RELEASE_PUBLIC_KEY_NEXT")
                && error.contains("holds no canonical key"),
            "a blank canonical file is unpublished, so the filled slot must fail: {error}"
        );
    }

    /// The statement must end with the string. Anything after the closing quote
    /// changes the value the runtime ends up with: a PowerShell operator or
    /// method rewrites it, a second quoted word appends to it, and a trailing
    /// shell command turns the assignment into a prefix that leaves it unset.
    #[test]
    fn nothing_may_follow_the_closing_quote() {
        for tail in [
            " -replace \"MIIB\", \"MIIC\"",
            ".Replace(\"MIIB\", \"MIIC\")",
            ".Substring(5)",
            " + \"x\"",
        ] {
            rejected(
                &good_sh(),
                &ps1(&format!("{PS_CURRENT} = \"{SAMPLE}\"@\n")),
                &good_rs(),
                &[
                    "install.ps1 assigns $PinnedReleasePublicKeyCurrent on line 1 to a string followed by `@",
                ],
            );
            rejected(
                &good_sh(),
                &ps1(&format!("{PS_CURRENT} = @\"\n{SAMPLE}\"@{tail}\n")),
                &good_rs(),
                &[
                    "install.ps1 assigns $PinnedReleasePublicKeyCurrent on line 1 to a string followed by",
                ],
            );
        }
        for tail in [" true", " | cat", "'extra'", "x", "; unset X", "@"] {
            rejected(
                &sh(&format!("{SH_CURRENT}=\"{SAMPLE}\"{tail}\n")),
                &good_ps1(),
                &good_rs(),
                &[
                    "install.sh assigns PINNED_RELEASE_PUBLIC_KEY_CURRENT on line 1 to a string followed by",
                ],
            );
        }
        rejected(
            &good_sh(),
            &good_ps1(),
            &format!("const {RS_META}: &str = \"{META}\"; const X: u8 = 1;\n"),
            &[
                "therock.rs assigns PINNED_METADATA_PUBLIC_KEY_PEM on line 1 to a string followed by",
            ],
        );

        // An empty-looking slot that the runtime fills: the dormant NEXT slot has
        // no canonical key to compare against, so this is the case that matters.
        let dormant = |sh: &str, ps: &str| {
            let repo = scratch_repo(sh, ps, &good_rs());
            std::fs::remove_file(
                repo.path()
                    .join("docs/keys/rocm-cli-release-next-public.pem"),
            )
            .expect("rm");
            check_pinned_keys(repo.path(), None).map_err(|error| format!("{error:#}"))
        };
        let error = dormant(
            &format!(
                "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"\"'{NEXT_KEY}'\n\
                 readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
                sh_current()
            ),
            &good_ps1(),
        )
        .expect_err("a quoted key glued to an empty shell slot must fail");
        assert!(
            error.contains("PINNED_RELEASE_PUBLIC_KEY_NEXT on line 6 to a string followed by"),
            "{error}"
        );
        let error = dormant(
            &format!(
                "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"\"\n\
                 readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
                sh_current()
            ),
            &format!(
                "{}$PinnedReleasePublicKeyNext = \"\" + @\"\n{NEXT_KEY}\"@\n",
                ps1_current()
            ),
        )
        .expect_err("a here-string added to an empty PowerShell slot must fail");
        assert!(
            error.contains("$PinnedReleasePublicKeyNext on line 7 to a string followed by"),
            "{error}"
        );

        // `""#junk` is the string `#junk`, and `" "` is not empty to `[ -n … ]`.
        for next in ["\"\"#junk", "\" \""] {
            let error = dormant(
                &format!(
                    "{}PINNED_RELEASE_PUBLIC_KEY_NEXT={next}\n\
                     readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
                    sh_current()
                ),
                &good_ps1(),
            )
            .expect_err("a slot that only looks empty must fail");
            assert!(
                error.contains("PINNED_RELEASE_PUBLIC_KEY_NEXT on line 6"),
                "{error}"
            );
        }

        // A trailing comment, or `;` before one, is fine.
        accepted(
            &sh(&format!("{SH_CURRENT}=\"{SAMPLE}\" # current trust root\n")),
            &ps1(&format!("{PS_CURRENT} = @\"\n{SAMPLE}\"@ # current\n")),
            &format!("const {RS_META}: &str = \"{META}\"; // metadata trust root\n"),
        );
        accepted(
            &sh(&format!("{SH_CURRENT}=\"{SAMPLE}\";# current\n")),
            &ps1(&format!("{PS_CURRENT} = \"{SAMPLE}\"; # current\n")),
            &good_rs(),
        );
    }

    /// install.sh must mark each slot readonly after assigning it, so a later
    /// write fails at runtime rather than silently swapping the trust root.
    #[test]
    fn install_sh_must_mark_each_slot_readonly() {
        let missing = format!(
            "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n",
            sh_current()
        );
        rejected(
            &missing,
            &good_ps1(),
            &good_rs(),
            &[
                "install.sh does not mark PINNED_RELEASE_PUBLIC_KEY_CURRENT readonly after assigning it on line 1",
            ],
        );
        let only_next = format!(
            "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\nreadonly PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
            sh_current()
        );
        rejected(
            &only_next,
            &good_ps1(),
            &good_rs(),
            &["does not mark PINNED_RELEASE_PUBLIC_KEY_CURRENT"],
        );
        let before = format!(
            "readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n{}\
             PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n",
            sh_current()
        );
        rejected(
            &before,
            &good_ps1(),
            &good_rs(),
            &["does not mark PINNED_RELEASE_PUBLIC_KEY_CURRENT"],
        );
        let commented = format!(
            "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n\
             # readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
            sh_current()
        );
        rejected(
            &commented,
            &good_ps1(),
            &good_rs(),
            &["does not mark PINNED_RELEASE_PUBLIC_KEY_CURRENT"],
        );
        // `export` sets an attribute too, but not this one.
        let exported = format!(
            "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n\
             export PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
            sh_current()
        );
        rejected(
            &exported,
            &good_ps1(),
            &good_rs(),
            &["does not mark PINNED_RELEASE_PUBLIC_KEY_CURRENT"],
        );

        // A `readonly` command must name the slot itself, not merely mention it.
        for line in [
            "readonly COPY=\"${PINNED_RELEASE_PUBLIC_KEY_CURRENT}\"",
            "readonly OTHER=1 # PINNED_RELEASE_PUBLIC_KEY_CURRENT",
        ] {
            let mentioned = format!(
                "{}PINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n{line}\n\
                 readonly PINNED_RELEASE_PUBLIC_KEY_NEXT\n",
                sh_current()
            );
            rejected(
                &mentioned,
                &good_ps1(),
                &good_rs(),
                &["install.sh", SH_CURRENT],
            );
        }

        // Accepted: `readonly NAME=` marks itself, and a trailing comment is fine.
        accepted(
            &format!(
                "readonly {SH_CURRENT}=\"{SAMPLE}\"\nPINNED_RELEASE_PUBLIC_KEY_NEXT=\"{NEXT_KEY}\"\n\
                 readonly PINNED_RELEASE_PUBLIC_KEY_NEXT # rotation slot\n"
            ),
            &good_ps1(),
            &good_rs(),
        );
    }

    /// The real installers and therock.rs pass as committed, and the check still
    /// bites on them. The copy takes every canonical key the repository
    /// publishes, so this keeps holding once the NEXT or metadata key is added.
    #[test]
    fn the_real_sources_pass_and_their_hardening_is_enforced() {
        let real = repo_root();
        let copy = |install_sh: &str| {
            let dir = tempfile::tempdir().expect("tempdir");
            let published = PINNED_KEYS
                .iter()
                .map(|key| key.canonical)
                .filter(|path| real.join(path).exists());
            for path in published.chain(["install.ps1", "apps/rocm/src/therock.rs"]) {
                let to = dir.path().join(path);
                std::fs::create_dir_all(to.parent().unwrap()).expect("mkdir");
                std::fs::copy(real.join(path), to).expect("copy");
            }
            std::fs::write(dir.path().join("install.sh"), install_sh).expect("write install.sh");
            let result = check_pinned_keys(dir.path(), None).map_err(|e| format!("{e:#}"));
            (dir, result)
        };
        let installer = std::fs::read_to_string(real.join("install.sh")).expect("read install.sh");
        copy(&installer).1.expect("the committed sources pass");

        // Without its readonly line(s), install.sh fails.
        let without_readonly: String = installer
            .lines()
            .filter(|line| !(is_attribute_command(line) && line.starts_with("readonly ")))
            .map(|line| format!("{line}\n"))
            .collect::<Vec<_>>()
            .concat();
        assert_ne!(
            without_readonly, installer,
            "install.sh has no readonly line"
        );
        let error = copy(&without_readonly).1.expect_err("readonly enforced");
        assert!(error.contains("readonly after assigning it"), "{error}");

        // A commented copy of the real key above a different one fails.
        let declaration = "PINNED_RELEASE_PUBLIC_KEY_CURRENT=\"-----BEGIN PUBLIC KEY-----\n";
        let start = installer
            .find(declaration)
            .expect("the CURRENT declaration");
        let end = start + installer[start..].find("-----\"\n").expect("its end") + 7;
        let commented = installer[start..end]
            .lines()
            .map(|line| format!("# {line}\n"))
            .collect::<Vec<_>>()
            .concat();
        let tampered = format!(
            "{}{commented}{}",
            &installer[..start],
            installer[start..].replacen("MIIB", "MIIC", 1)
        );
        let error = copy(&tampered)
            .1
            .expect_err("commented canonical above a wrong key");
        assert!(
            error.contains("does not embed the canonical key"),
            "{error}"
        );
    }

    #[test]
    fn ci_public_key_cross_check() {
        let current = pem_body(SAMPLE).unwrap();

        // No CI key configured -> nothing to assert, but the skip is reported so an
        // unset/rotated-away secret does not read as a passing cross-check.
        for absent in [None, Some("   "), Some("")] {
            let reported = check_ci_public_key(absent, Some(&current)).unwrap();
            assert!(
                reported.iter().any(|m| m.contains("not configured")),
                "absent CI key should report a skip, got {reported:?}"
            );
        }

        // Configured but no canonical current key yet -> skipped, not an error.
        let skipped = check_ci_public_key(Some(SAMPLE), None).unwrap();
        assert!(skipped.iter().any(|m| m.contains("skipped")));

        // Matching CI key -> accepted.
        assert!(check_ci_public_key(Some(SAMPLE), Some(&current)).is_ok());

        // Mismatched CI key -> rejected (CI would verify against a different key).
        let other = SAMPLE.replace("test", "diff");
        assert!(check_ci_public_key(Some(&other), Some(&current)).is_err());

        // Malformed CI key -> rejected.
        assert!(check_ci_public_key(Some("not a pem"), Some(&current)).is_err());
    }

    /// Each variable name must be the one the release gate reads *for that role*.
    ///
    /// The lookup tests below match on these same constants, so they cannot see a
    /// wrong *value*: misspelling one leaves every case green while `run` queries
    /// a variable nothing ever sets, and the cross-check silently skips.
    ///
    /// Matching the gate's assignment lines, not merely the name somewhere in the
    /// file, is what makes this bite. The literals also appear in the gate's
    /// self-test environments, so a name-anywhere search passed three real
    /// regressions: swapping the two constants (release CI would read the PEM text
    /// as a file path), and pointing either at `ROCM_CLI_METADATA_*` (the
    /// cross-check would quietly skip). Pinning `BINDING = "NAME"` ties each name
    /// to the role the gate gives it.
    ///
    /// This pins the names only. [`key_resolution_order_matches_the_release_gate`]
    /// pins the order they are consulted in, which is a separate rule.
    #[test]
    fn ci_public_key_env_names_match_the_release_gate() {
        let gate = std::fs::read_to_string(repo_root().join("scripts/release_readiness.py"))
            .expect("read scripts/release_readiness.py");
        for (binding, name) in [
            ("SIGNING_PUBLIC_KEY_PATH_ENV", CI_PUBLIC_KEY_PATH_ENV),
            ("SIGNING_PUBLIC_KEY_ENV", CI_PUBLIC_KEY_PEM_ENV),
        ] {
            let assignment = format!("\n{binding} = \"{name}\"\n");
            assert!(
                gate.contains(&assignment),
                "release_readiness.py does not bind {binding} to {name}; this \
                 cross-check would resolve a different key than the gate verifies \
                 with. Expected the line {assignment:?}"
            );
        }
    }

    /// The source of one top-level `def`, up to the next top-level `def`.
    ///
    /// Slicing to the function matters: both constants appear a dozen other
    /// places in the gate (`PRODUCTION_TRUST_ENV_NAMES`, `validate_production_trust`,
    /// the self-test cases), so a file-wide search would order unrelated lines and
    /// pass whatever the resolver actually did.
    fn python_function_body<'a>(source: &'a str, name: &str) -> &'a str {
        let header = format!("\ndef {name}(");
        let start = source
            .find(&header)
            .unwrap_or_else(|| panic!("scripts/release_readiness.py has no `def {name}(`"))
            + 1;
        let rest = &source[start..];
        let end = rest[1..]
            .find("\ndef ")
            .map_or(rest.len(), |index| index + 2);
        &rest[..end]
    }

    /// The path-before-PEM precedence must stay in lockstep across both
    /// implementations, not merely the variable names.
    ///
    /// [`ci_public_key_env_names_match_the_release_gate`] pins *which* variable
    /// each side reads for each role, but nothing pinned the *order* they are
    /// consulted in. Reversing either side alone left both suites green: the
    /// lookup tests below only observe the Rust resolver, and the gate's
    /// `--self-test` only observes the Python one. A release would then be
    /// verified against the inline PEM while this cross-check compared the key
    /// named by path (or the reverse) — two different trust roots, with the run
    /// still reporting success and naming only one of them.
    ///
    /// So the order is established twice, independently: observed by *running*
    /// the Rust resolver with both sources set, and read out of the gate's own
    /// `resolve_signing_key` body. Both must say path-before-PEM.
    #[test]
    fn key_resolution_order_matches_the_release_gate() {
        // The Rust order, observed rather than read off the source: with both
        // sources set to different keys, whichever comes back is the winner.
        let dir = tempfile::tempdir().expect("tempdir");
        let key_file = dir.path().join("by-path.pem");
        std::fs::write(&key_file, SAMPLE).expect("write key");
        let inline = SAMPLE.replace("test", "diff");

        let both_set = |name: &str| match name {
            CI_PUBLIC_KEY_PATH_ENV => Some(OsString::from(key_file.as_os_str())),
            CI_PUBLIC_KEY_PEM_ENV => Some(OsString::from(inline.as_str())),
            other => panic!("unexpected variable queried: {other}"),
        };
        let resolved = resolve_ci_public_key(both_set)
            .expect("both sources set resolves a key")
            .expect("a key");
        assert_eq!(
            pem_body(&resolved).unwrap(),
            pem_body(SAMPLE).unwrap(),
            "resolve_ci_public_key stopped preferring {CI_PUBLIC_KEY_PATH_ENV} over \
             {CI_PUBLIC_KEY_PEM_ENV}. The release gate still resolves the path first, so \
             this cross-check would compare a different key than releases are verified \
             against, and report a match for a key nobody verified with"
        );

        // The Python order, read out of the gate's own resolver.
        let gate = std::fs::read_to_string(repo_root().join("scripts/release_readiness.py"))
            .expect("read scripts/release_readiness.py");
        let body = python_function_body(&gate, "resolve_signing_key");
        let locate = |needle: &str| {
            body.find(needle).unwrap_or_else(|| {
                panic!(
                    "resolve_signing_key no longer resolves its key via `{needle}`, so the \
                     two implementations' precedence can no longer be compared. Re-establish \
                     the comparison rather than deleting it: the order is the security rule. \
                     The body read was:\n{body}"
                )
            })
        };
        let path_branch = locate("env_path(SIGNING_PUBLIC_KEY_PATH_ENV)");
        let pem_branch = locate("env_text(SIGNING_PUBLIC_KEY_ENV)");
        assert!(
            path_branch < pem_branch,
            "release_readiness.py's resolve_signing_key now consults \
             {CI_PUBLIC_KEY_PEM_ENV} before {CI_PUBLIC_KEY_PATH_ENV}, while \
             resolve_ci_public_key above still resolves the path first. The gate would \
             verify a release against one key and this cross-check would compare another. \
             Change both or neither"
        );
    }

    /// The cross-check must see a key supplied by path, not just the inline PEM.
    ///
    /// `release_readiness.py` prefers `ROCM_CLI_SIGNING_PUBLIC_KEY_PATH` and
    /// verifies artifacts against it. Resolving only the PEM here meant that key
    /// was never compared with `docs/keys/`, and the run still logged
    /// "not configured — cross-check skipped" while a key was in use.
    #[test]
    fn ci_public_key_resolves_the_path_variable_before_the_inline_pem() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key_file = dir.path().join("ci-public-key.pem");
        std::fs::write(&key_file, SAMPLE).expect("write key");
        let other = SAMPLE.replace("test", "diff");

        // A lookup over a fixed map, so the *names* queried are covered too: with
        // the variables read directly, `run()` could query a misspelled name and
        // every case here would still pass.
        let env = |path: Option<&str>, pem: Option<&str>| {
            let path = path.map(OsString::from);
            let pem = pem.map(OsString::from);
            move |name: &str| match name {
                CI_PUBLIC_KEY_PATH_ENV => path.clone(),
                CI_PUBLIC_KEY_PEM_ENV => pem.clone(),
                other => panic!("unexpected variable queried: {other}"),
            }
        };
        let named = key_file.to_str().expect("utf-8 path");

        // Path wins over the PEM, so the key cross-checked is the key verified with.
        let resolved = resolve_ci_public_key(env(Some(named), Some(&other)))
            .expect("readable key")
            .expect("a key");
        assert_eq!(pem_body(&resolved).unwrap(), pem_body(SAMPLE).unwrap());

        // ...and that resolved key is what reaches the comparison: a path key that
        // disagrees with the canonical one must fail rather than skip.
        let canonical = pem_body(&other).unwrap();
        assert!(check_ci_public_key(Some(&resolved), Some(&canonical)).is_err());

        // The PEM is still used when no path is named.
        assert_eq!(
            resolve_ci_public_key(env(None, Some(SAMPLE)))
                .expect("inline pem")
                .as_deref(),
            Some(SAMPLE)
        );

        // Unset counts as absent -- the shape of an unset GitHub secret, which
        // expands to the empty string.
        for blank in [None, Some(""), Some("   ")] {
            assert!(
                resolve_ci_public_key(env(None, blank))
                    .expect("no key")
                    .is_none(),
                "blank PEM {blank:?} should resolve to no key"
            );
        }
        // Same for the path variable, and whitespace counts as blank: the Python
        // gate's `env_path` strips the value and falls through to the PEM, so
        // erroring here instead would make the two sides disagree about which key
        // a release is verified with.
        for blank in [None, Some(""), Some("   "), Some("\t\n")] {
            assert_eq!(
                resolve_ci_public_key(env(blank, Some(SAMPLE)))
                    .unwrap_or_else(|e| panic!("blank path {blank:?} should not error: {e}"))
                    .as_deref(),
                Some(SAMPLE),
                "a blank path variable {blank:?} should fall through to the PEM"
            );
        }

        // A named key that cannot be read is an error, never a quiet fall-through
        // to a different key than the operator named. The *message* is pinned too:
        // a failed read that fell through as an empty string would still be an
        // error, but the operator would be told the key "is empty" and go looking
        // at a file that is fine.
        let missing = dir.path().join("absent.pem");
        let read_error = resolve_ci_public_key(env(missing.to_str(), Some(SAMPLE)))
            .expect_err("an unreadable named key must fail rather than fall back to the PEM");
        assert!(
            format!("{read_error:#}").contains("failed to read"),
            "an unreadable named key should report the read failure, got: {read_error:#}"
        );

        // A named key that exists but holds nothing must be an error too. Passing
        // it on as `Some("")` made `check_ci_public_key` trim it away and report
        // "not configured" while a key was in fact configured -- exactly the
        // confusion that check's own doc forbids.
        for empty in ["", "   \n\t "] {
            let blank_file = dir.path().join("blank.pem");
            std::fs::write(&blank_file, empty).expect("write blank key");
            let error = resolve_ci_public_key(env(blank_file.to_str(), Some(SAMPLE)))
                .expect_err("an empty named key must be an error");
            assert!(
                error.to_string().contains("is empty"),
                "unexpected error for an empty named key: {error}"
            );
        }
    }

    /// `run_with` is the whole path `run` takes, so the resolved key must reach
    /// the comparison rather than being dropped on the way.
    ///
    /// Nothing previously ran through this seam. A `run` whose lookup always
    /// queried the PEM name, always returned `None`, or simply handed
    /// `check_pinned_keys` a `None` turned the path resolution into a silent
    /// no-op and reported "not configured — cross-check skipped" on a run that
    /// did have a key. Each case below fails under exactly those mutations.
    #[test]
    fn run_with_carries_the_resolved_key_into_the_comparison() {
        let root = repo_root();
        let canonical =
            std::fs::read_to_string(root.join("docs/keys/rocm-cli-release-current-public.pem"))
                .expect("the canonical current release key is committed");
        let dir = tempfile::tempdir().expect("tempdir");

        let env = |path: Option<&str>, pem: Option<&str>| {
            let path = path.map(OsString::from);
            let pem = pem.map(OsString::from);
            move |name: &str| match name {
                CI_PUBLIC_KEY_PATH_ENV => path.clone(),
                CI_PUBLIC_KEY_PEM_ENV => pem.clone(),
                other => panic!("unexpected variable queried: {other}"),
            }
        };

        // No key anywhere: the run passes, and says the cross-check was skipped.
        // This is the only place that message is exercised through the code path
        // that actually prints it.
        let skipped = run_with(&root, env(None, None)).expect("no key configured is not an error");
        assert!(
            skipped
                .iter()
                .any(|m| m.contains("not configured — cross-check skipped")),
            "a run with no key must say the cross-check was skipped, got {skipped:?}"
        );

        // A key named by path must be compared, not skipped: a wrong one fails.
        let wrong = dir.path().join("wrong.pem");
        std::fs::write(&wrong, SAMPLE).expect("write wrong key");
        let error = run_with(&root, env(wrong.to_str(), None))
            .expect_err("a path key that disagrees with docs/keys must fail the run");
        assert!(
            format!("{error:#}").contains("does not match the canonical current release key"),
            "unexpected failure for a mismatched path key: {error:#}"
        );

        // ...and the right one passes, reported as a match rather than a skip.
        let right = dir.path().join("right.pem");
        std::fs::write(&right, &canonical).expect("write canonical key");
        let matched = run_with(&root, env(right.to_str(), None)).expect("canonical key matches");
        assert!(
            matched
                .iter()
                .any(|m| m.contains("matches the canonical current release key")),
            "a path key equal to docs/keys must be reported as a match, got {matched:?}"
        );

        // The PEM source still reaches the comparison too, and loses to the path.
        let via_pem = run_with(&root, env(None, Some(&canonical))).expect("canonical pem matches");
        assert!(
            via_pem
                .iter()
                .any(|m| m.contains("matches the canonical current release key")),
            "an inline canonical PEM must be reported as a match, got {via_pem:?}"
        );
        assert!(
            run_with(&root, env(wrong.to_str(), Some(&canonical))).is_err(),
            "the path key must win, so a wrong path key is not rescued by a good PEM"
        );
    }

    /// `run`'s only remaining untestable surface is this shim, so pin it.
    ///
    /// It must pass the name it is given straight to the environment. A shim that
    /// ignored its argument, or always answered `None`, would make `run` resolve
    /// the wrong variable or no variable at all while every fake-lookup test above
    /// stayed green. `PATH` is used because it is set in every environment this
    /// runs in, so a shim that ignores the name cannot accidentally agree.
    #[test]
    fn env_lookup_passes_the_name_through() {
        assert_eq!(env_lookup("PATH"), std::env::var_os("PATH"));
        assert!(
            env_lookup("PATH").is_some(),
            "PATH is unset, so this test cannot tell a pass-through from a stub"
        );
        for name in [CI_PUBLIC_KEY_PATH_ENV, CI_PUBLIC_KEY_PEM_ENV] {
            assert_eq!(
                env_lookup(name),
                std::env::var_os(name),
                "{name} must be read from the environment under its own name"
            );
        }
        assert_eq!(
            env_lookup("ROCM_CLI_VERIFY_PINNED_KEYS_NOT_A_REAL_VARIABLE"),
            None,
            "an unset variable must resolve to None, not to some other variable"
        );
    }
}
