// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reading `/etc/os-release`.
//!
//! The one parser every reader in the workspace uses — the driver plan, the
//! package installs it approves (`ensure_openmpi_for_vllm`,
//! `ensure_torch_runtime_dep`), `rocm examine`, the OpenMPI hint, and the host
//! distro name — so they cannot read the same file as different distros.
//!
//! The reference is a POSIX shell sourcing the file: `os-release(5)` defines the
//! format as shell-compatible assignments, and the WSL probe in
//! [`crate::examine`] literally sources it. The rule this module keeps is
//!
//! > **either the value `sh` would assign, or `None` — never anything else.**
//!
//! It returns `sh`'s value for every file made only of blank lines, `#`
//! comments, and assignments the spec allows:
//!
//! * a value bare, in double quotes, or in single quotes;
//! * inside double quotes, the backslash sequences `\"`, `\\`, `` \` `` and
//!   `\$` (any other backslash is literal, as in a shell);
//! * inside single quotes, nothing decoded;
//! * in a bare value, `\x` meaning `x`;
//! * spaces and tabs before the name, and after the value;
//! * a `#` comment after the value, separated from it by a space or tab;
//! * an empty value (`ID=`), which assigns the empty string;
//! * a later assignment replacing an earlier one.
//!
//! **Any other line makes the whole file unreadable** — every field is `None`,
//! not just the one the line names, and [`parse`] reports that line. To `sh`
//! such a line is a command, or the start of one: `export ID=debian` and
//! `X=1; ID=debian` assign `ID`; `unset ID` and `X= unset ID` remove it;
//! `NAME="foo` opens a string that swallows the lines after it, so an `ID=`
//! below it is not an assignment at all; `ID="deb"ian` concatenates; `"a$b"`
//! and `` `cmd` `` expand. A command can change any variable, including ones
//! assigned before it, so no key's value can be vouched for once one appears.
//! The parser reproduces none of this; it fails closed.
//!
//! A line other than a blank or a comment is one of those unless it is a valid
//! shell name, `=`, and a value with none of: an unterminated quote; anything
//! after a closing quote but spaces, tabs and a `#` comment; a space or tab
//! between `=` and the value; an unquoted `$` or `` ` ``, anywhere outside
//! single quotes; in a bare value, any unquoted quote, `;`, `|`, `&`, `<`, `>`,
//! `(`, `)`, or trailing lone `\`, or a `~` at the start or after a `:` (which
//! `sh` expands to a home directory); or any control character.
//!
//! The control-character rule is deliberate rather than an omission, and it is
//! what a CRLF file trips: `sh` reads `ID=ubuntu\r` as `ubuntu\r`, a value that
//! matches no distro and would reach a terminal. So a CRLF file is unreadable.
//! Only spaces and tabs are trimmed or separate words; any other whitespace is
//! an ordinary character, as it is to `sh`.
//!
//! What `None` means downstream: never a plan for a distro the file does not
//! declare. When the whole file is unreadable, the driver plan and `rocm
//! examine` say which line, rather than reporting an unsupported distro.

use std::collections::HashMap;
use std::fmt;

/// The first line of an os-release file that is not a blank, a comment or a
/// plain assignment, which makes the whole file unreadable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadableLine {
    /// 1-based, as an editor shows it.
    pub number: usize,
    /// The line as written, cut to [`UnreadableLine::SHOWN_CHARS`] characters.
    pub text: String,
}

impl UnreadableLine {
    /// How much of the offending line is kept: enough to recognise it, not so
    /// much that one bad line floods a report.
    pub const SHOWN_CHARS: usize = 120;
}

/// `line N is not a plain assignment: "<text>"` — the text `Debug`-quoted, so
/// a control character in it is shown as a `\u{…}` sequence rather than acted
/// on, and the message stays on one line.
impl fmt::Display for UnreadableLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "line {} is not a plain assignment: {:?}",
            self.number, self.text
        )
    }
}

/// The value of `key` in os-release `text`, as `sh` would assign it, or `None`.
///
/// `None` when the key is absent, or when the file is unreadable — see
/// [`parse`] for which line made it so.
#[must_use]
pub fn field(text: &str, key: &str) -> Option<String> {
    parse(text).ok()?.remove(key)
}

/// Every assignment in `text`, the last of each name winning — or the first
/// line that is not a blank, a comment or a plain assignment.
///
/// One pass over the whole file, because the verdict on any key depends on
/// every line, not only the ones that name it.
///
/// # Errors
///
/// [`UnreadableLine`] for the first line `sh` would run as a command or that
/// this parser cannot read exactly as `sh` would; see the module docs.
pub fn parse(text: &str) -> Result<HashMap<String, String>, UnreadableLine> {
    let mut fields = HashMap::new();
    // Split on `\n` only: `str::lines` would also drop a `\r` before it, which
    // `sh` keeps as part of the value.
    for (index, raw_line) in text.split('\n').enumerate() {
        let unreadable = || UnreadableLine {
            number: index + 1,
            text: raw_line.chars().take(UnreadableLine::SHOWN_CHARS).collect(),
        };
        let line = raw_line.trim_start_matches([' ', '\t']);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, raw) = line.split_once('=').ok_or_else(unreadable)?;
        if !is_shell_name(name) {
            return Err(unreadable());
        }
        let value = decode(raw.trim_end_matches([' ', '\t'])).ok_or_else(unreadable)?;
        fields.insert(name.to_owned(), value);
    }
    Ok(fields)
}

/// Whether `name` is something `sh` accepts on the left of an assignment.
fn is_shell_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

/// Whether what follows a value is nothing, or a `#` comment after blanks —
/// the only things `sh` lets end an assignment without starting a command.
fn ends_cleanly(rest: &str) -> bool {
    rest.is_empty()
        || (rest.starts_with([' ', '\t']) && rest.trim_start_matches([' ', '\t']).starts_with('#'))
}

/// Decode one assignment's right-hand side, trailing spaces and tabs already
/// gone. `None` if it is malformed.
fn decode(raw: &str) -> Option<String> {
    // A tab is a blank to `sh`, like a space, and harmless on a terminal; every
    // other control character makes the line unreadable.
    if raw.chars().any(|ch| ch.is_control() && ch != '\t') {
        return None;
    }
    let mut chars = raw.chars();
    match chars.next() {
        None => Some(String::new()),
        Some('"') => {
            let mut value = String::new();
            loop {
                match chars.next()? {
                    '"' => break,
                    // Unquoted to the shell even inside `"…"`: it would expand.
                    '$' | '`' => return None,
                    '\\' => match chars.next()? {
                        literal @ ('"' | '\\' | '`' | '$') => value.push(literal),
                        other => {
                            value.push('\\');
                            value.push(other);
                        }
                    },
                    ch => value.push(ch),
                }
            }
            ends_cleanly(chars.as_str()).then_some(value)
        }
        Some('\'') => {
            let rest = chars.as_str();
            let end = rest.find('\'')?;
            ends_cleanly(&rest[end + 1..]).then(|| rest[..end].to_owned())
        }
        Some(_) => {
            let mut value = String::new();
            let mut chars = raw.chars();
            // `sh` expands an unquoted `~` at the start of the value and after
            // each unquoted `:`.
            let mut tilde_expands = true;
            while let Some(ch) = chars.next() {
                match ch {
                    '\\' => {
                        value.push(chars.next()?);
                        tilde_expands = false;
                        continue;
                    }
                    // A blank ends the value. Past it, only a comment may follow.
                    ' ' | '\t' => {
                        let rest = format!("{ch}{}", chars.as_str());
                        return ends_cleanly(&rest).then_some(value);
                    }
                    '~' if tilde_expands => return None,
                    '"' | '\'' | '$' | '`' | ';' | '|' | '&' | '<' | '>' | '(' | ')' => {
                        return None;
                    }
                    ch => value.push(ch),
                }
                tilde_expands = ch == ':';
            }
            Some(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_both_quote_styles_and_bare_values_alike() {
        for text in [
            "VERSION_CODENAME=noble\n",
            "VERSION_CODENAME=\"noble\"\n",
            "VERSION_CODENAME='noble'\n",
            "VERSION_CODENAME=noble \t\n",
            " \tVERSION_CODENAME=noble\n",
            "VERSION_CODENAME=noble",
        ] {
            assert_eq!(
                field(text, "VERSION_CODENAME").as_deref(),
                Some("noble"),
                "{text:?}"
            );
        }
    }

    #[test]
    fn decodes_the_backslash_sequences_a_double_quoted_value_may_carry() {
        assert_eq!(field(r#"V="24\"04""#, "V").as_deref(), Some("24\"04"));
        assert_eq!(field(r#"V="a\\b""#, "V").as_deref(), Some("a\\b"));
        assert_eq!(field(r#"V="a\$b""#, "V").as_deref(), Some("a$b"));
        assert_eq!(field(r#"V="a\`b""#, "V").as_deref(), Some("a`b"));
        // Any other backslash is literal inside double quotes, as in a shell.
        assert_eq!(field(r#"V="a\nb""#, "V").as_deref(), Some("a\\nb"));
        // And no backslash sequence means anything inside single quotes.
        assert_eq!(field(r"V='a\b'", "V").as_deref(), Some("a\\b"));
    }

    #[test]
    fn a_malformed_assignment_reads_as_absent() {
        for text in [
            "VERSION_ID=\"24.04\n",
            "VERSION_ID='24.04\n",
            "VERSION_ID=\"24\"04\n",
            "VERSION_ID='24'.04\n",
            "VERSION_ID=24'04\n",
            "VERSION_ID=\"24.04\"\"\n",
            "VERSION_ID= 24.04\n",
            "VERSION_ID=24 04\n",
            "VERSION_ID=\"24$X\"\n",
            "VERSION_ID=24;x\n",
            "VERSION_ID=24>x\n",
            "VERSION_ID=24|x\n",
            "VERSION_ID=24&x\n",
            "VERSION_ID=(24)\n",
            "VERSION_ID=~\n",
            "VERSION_ID=a:~\n",
            "VERSION_ID=24\\\n",
        ] {
            assert_eq!(field(text, "VERSION_ID"), None, "{text:?}");
        }
    }

    /// A line that is not a blank, a comment or a well-formed `NAME=value` is a
    /// command to `sh`, and a command can set or unset *any* variable, including
    /// ones assigned before it — so every field reads as `None`. Each of these
    /// was read wrongly before: the parser returned a value `sh` does not
    /// assign. What `sh` does with each, run through `dash`:
    #[test]
    fn a_file_that_runs_a_command_reads_as_absent_throughout() {
        for (text, what_sh_does) in [
            (
                "ID=ubuntu\nNAME=\"foo\nID=debian\n\"\n",
                "ID=debian is inside NAME's string; sh keeps ubuntu",
            ),
            (
                "ID=ubuntu\nNAME='foo\nID=debian\n'\n",
                "the same, single-quoted",
            ),
            (
                "ID=ubuntu\nNAME=foo\\\nID=debian\n",
                "a trailing backslash joins the lines; sh keeps ubuntu",
            ),
            ("ID=ubuntu\nX=1; ID=debian\n", "sh assigns debian"),
            ("ID=ubuntu\nexport ID=debian\n", "sh assigns debian"),
            ("ID=ubuntu\nunset ID\n", "sh leaves ID unset"),
            (
                "ID=ubuntu\nX= unset ID\n",
                "unset runs with a temporary X; sh leaves ID unset",
            ),
            ("ID=ubuntu\nID =debian\n", "runs a command named ID"),
        ] {
            assert_eq!(field(text, "ID"), None, "{text:?}: {what_sh_does}");
        }
    }

    /// `\r` is not whitespace to `sh`: `ID=ubuntu\r` assigns `ubuntu\r`, which
    /// matches no distro and would reach a terminal. A control character in a
    /// value therefore makes the file malformed — so a CRLF file reads as
    /// `None` throughout, deliberately, rather than being quietly tolerated.
    #[test]
    fn a_control_character_or_crlf_file_reads_as_absent() {
        for text in [
            "ID=ubuntu\r\nVERSION_ID=24.04\r\n",
            "ID=\"ub\u{1b}[2Kuntu\"\n",
            "ID='ub\u{7}untu'\n",
        ] {
            assert_eq!(field(text, "ID"), None, "{text:?}");
        }
        // A tab is the exception: a blank to `sh`, kept inside quotes as `sh`
        // keeps it.
        assert_eq!(field("ID='ub\tuntu'\n", "ID").as_deref(), Some("ub\tuntu"));
        // Other Unicode whitespace is an ordinary character to `sh`, so it is
        // kept, not trimmed.
        assert_eq!(
            field("ID=ubuntu\u{a0}\n", "ID").as_deref(),
            Some("ubuntu\u{a0}")
        );
    }

    #[test]
    fn the_last_assignment_wins_as_in_a_shell() {
        assert_eq!(
            field("ID=debian\nID=ubuntu\n", "ID").as_deref(),
            Some("ubuntu")
        );
        // Including an empty one: `sh` assigns "" here, it does not keep ubuntu.
        assert_eq!(field("ID=ubuntu\nID=\n", "ID").as_deref(), Some(""));
    }

    #[test]
    fn comments_blank_lines_and_near_miss_names() {
        let text = "# ID=commented \"\n\n  \t\nXID=prefixed\nID_LIKE=suffixed\n";
        assert_eq!(field(text, "ID"), None);
        assert_eq!(field(text, "ID_LIKE").as_deref(), Some("suffixed"));
    }

    /// `sh` ends an unquoted word at a blank, and a `#` that starts the next
    /// word starts a comment — so a comment may follow any value form.
    #[test]
    fn a_comment_after_the_value_is_not_part_of_it() {
        for text in [
            "ID=ubuntu # the distro\n",
            "ID=ubuntu\t#\n",
            "ID=\"ubuntu\" # quoted\n",
            "ID='ubuntu'   # single-quoted\n",
        ] {
            assert_eq!(field(text, "ID").as_deref(), Some("ubuntu"), "{text:?}");
        }
        // Not after a blank, `#` is an ordinary character…
        assert_eq!(field("ID=ubu#ntu\n", "ID").as_deref(), Some("ubu#ntu"));
        // …and after a closing quote with no blank, it is concatenation.
        assert_eq!(field("ID=\"ubuntu\"#x\n", "ID"), None);
        // A blank followed by anything but a comment is a command.
        assert_eq!(field("ID=ubuntu debian\n", "ID"), None);
    }

    /// An unreadable file names its first offending line, 1-based, so the
    /// reason a caller gives points at something a person can open and fix —
    /// and quotes it so a control character in it is shown, not acted on.
    #[test]
    fn an_unreadable_file_reports_its_first_offending_line() {
        let text = "# header\nID=ubuntu\n\nexport ID=debian\nunset ID\n";
        let unreadable = parse(text).expect_err("a command line makes the file unreadable");
        assert_eq!(
            unreadable,
            UnreadableLine {
                number: 4,
                text: "export ID=debian".to_owned()
            }
        );
        assert_eq!(
            unreadable.to_string(),
            "line 4 is not a plain assignment: \"export ID=debian\""
        );

        let crlf = parse("ID=ubuntu\r\nVERSION_ID=24.04\r\n").unwrap_err();
        assert_eq!(crlf.number, 1);
        assert_eq!(
            crlf.to_string(),
            "line 1 is not a plain assignment: \"ID=ubuntu\\r\""
        );

        let long = format!("X={}", "a b".repeat(100));
        let cut = parse(&long).unwrap_err();
        assert_eq!(cut.text.chars().count(), UnreadableLine::SHOWN_CHARS);
    }

    /// Characters drawn from to build values: mostly the ones the encodings
    /// treat specially, so every backslash and quoting path is exercised.
    const ALPHABET: &[char] = &[
        'a', '1', '.', ' ', '"', '\'', '\\', '$', '`', '#', '=', ':', '~',
    ];

    /// Every string over `alphabet` up to `max_len` characters. Exhaustive
    /// rather than sampled: the shapes that break a parser are short, and an
    /// enumeration cannot miss one by chance.
    fn every_string(alphabet: &[char], max_len: usize) -> Vec<String> {
        let mut out = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..max_len {
            frontier = frontier
                .iter()
                .flat_map(|prefix| {
                    alphabet.iter().map(move |ch| {
                        let mut next = prefix.clone();
                        next.push(*ch);
                        next
                    })
                })
                .collect();
            out.extend(frontier.iter().cloned());
        }
        out
    }

    /// Every spec-valid way to write `value`: double-quoted with backslashes
    /// always; single-quoted when it holds no `'`; bare, with a backslash before
    /// every non-alphanumeric character, when it is non-empty.
    fn encodings(value: &str) -> Vec<String> {
        let mut out = vec![format!(
            "\"{}\"",
            value
                .chars()
                .map(|ch| match ch {
                    '"' | '\\' | '`' | '$' => format!("\\{ch}"),
                    ch => ch.to_string(),
                })
                .collect::<String>()
        )];
        if !value.contains('\'') {
            out.push(format!("'{value}'"));
        }
        // Bare, escaping everything that is not a letter or digit. A trailing
        // backslash-space is left out: trailing blanks are trimmed before
        // decoding, so `a\ ` reads as `a\` — a dangling backslash, which is
        // malformed. Failing closed on it is allowed; it is just not "valid".
        if !value.is_empty() && !value.ends_with(' ') {
            out.push(
                value
                    .chars()
                    .map(|ch| {
                        if ch.is_ascii_alphanumeric() {
                            ch.to_string()
                        } else {
                            format!("\\{ch}")
                        }
                    })
                    .collect(),
            );
        }
        out
    }

    /// Every spec-valid encoding of every short value decodes back to it.
    #[test]
    fn every_valid_encoding_decodes_to_its_value() {
        let values = every_string(ALPHABET, 3);
        assert!(values.len() > 1_000, "enumeration too small to mean much");
        for value in &values {
            for encoded in encodings(value) {
                let text = format!("K={encoded}\n");
                assert_eq!(
                    field(&text, "K").as_deref(),
                    Some(value.as_str()),
                    "{encoded} should decode to {value:?}"
                );
            }
        }
    }

    /// The parser checked against `sh` itself. Unix-only: there is no POSIX
    /// shell on a Windows runner, and these would be dead code there.
    #[cfg(unix)]
    mod sh_oracle {
        use super::*;
        use std::fmt::Write as _;
        use std::path::{Path, PathBuf};

        /// `HOME` for the shell, so a `~` that expands is visible as this rather
        /// than as nothing. Unset, `dash` leaves `~` literal and an expansion the
        /// parser missed would go unnoticed.
        const HOME_SENTINEL: &str = "/rocm-os-release-home";

        /// A scratch directory removed on drop, so a failing assertion does not
        /// leave it behind.
        struct ScratchDir(PathBuf);

        impl ScratchDir {
            fn new(tag: &str) -> Self {
                let dir = std::env::temp_dir().join(format!(
                    "rocm-os-release-{tag}-{}-{}",
                    std::process::id(),
                    crate::unix_time_millis()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                Self(dir)
            }
        }

        impl Drop for ScratchDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        /// `sh` sourcing `path` from inside `dir`, with nothing it could run:
        /// an empty environment, `PATH` naming a directory that does not exist
        /// (an *empty* `PATH` makes `dash` search the current directory), and
        /// `/bin/sh` by absolute path. A value like `K= x` or `` `x` `` then
        /// names a command that is not found.
        ///
        /// The script goes in a file rather than `-c`: the all-valid-encodings
        /// case reads thousands of keys, past the kernel's argument-size limit.
        fn sh_source(dir: &Path, path: &Path, script_tail: &str) -> std::process::Output {
            let script = dir.join("probe.sh");
            std::fs::write(
                &script,
                format!(". '{}' || exit 3\n{script_tail}", path.display()),
            )
            .unwrap();
            std::process::Command::new("/bin/sh")
                .arg(&script)
                .current_dir(dir)
                .env_clear()
                .env("PATH", "/nonexistent")
                .env("HOME", HOME_SENTINEL)
                .output()
                .expect("sh should run")
        }

        /// What `sh` leaves in each of `keys` after sourcing `text`: `None` for
        /// unset, and `None` for every key if sourcing failed. Each value is
        /// printed NUL-terminated behind a set/unset marker.
        fn sh_reads(dir: &ScratchDir, text: &str, keys: &[&str]) -> Vec<Option<String>> {
            let path = dir.0.join("os-release");
            std::fs::write(&path, text).unwrap();
            let mut tail = String::new();
            for key in keys {
                writeln!(
                    tail,
                    "case ${{{key}+x}} in x) printf 'S%s\\0' \"${key}\" ;; *) printf 'U\\0' ;; esac"
                )
                .expect("writing to a String cannot fail");
            }
            let output = sh_source(&dir.0, &path, &tail);
            if !output.status.success() {
                return vec![None; keys.len()];
            }
            output
                .stdout
                .split(|byte| *byte == 0)
                .take(keys.len())
                .map(|chunk| {
                    let chunk = String::from_utf8_lossy(chunk);
                    chunk.strip_prefix('S').map(str::to_owned)
                })
                .collect()
        }

        /// The module's rule for one file: for every key, the parser reads
        /// either nothing or exactly what `sh` left in it. Returns how many keys
        /// read as a value.
        fn assert_sh_or_none(dir: &ScratchDir, text: &str, keys: &[&str]) -> usize {
            let from_shell = sh_reads(dir, text, keys);
            let mut read = 0;
            for (key, shell) in keys.iter().zip(&from_shell) {
                if let Some(value) = field(text, key) {
                    read += 1;
                    assert_eq!(
                        Some(&value),
                        shell.as_ref(),
                        "{key} in {text:?}: parser read {value:?}, sh left {shell:?}"
                    );
                }
            }
            read
        }

        /// Every spec-valid encoding, in one file sourced once, reads exactly as
        /// `sh` reads it — never `None`. The encoder and the parser could share a
        /// misunderstanding of the format; `sh` cannot.
        #[test]
        fn a_shell_sourcing_the_file_reads_what_this_parser_reads() {
            let mut text = String::new();
            let mut expected = Vec::new();
            for value in every_string(ALPHABET, 3) {
                for encoded in encodings(&value) {
                    let key = format!("K{}", expected.len());
                    writeln!(text, "{key}={encoded}").expect("writing to a String cannot fail");
                    expected.push((key, value.clone()));
                }
            }
            let dir = ScratchDir::new("valid");
            let keys: Vec<&str> = expected.iter().map(|(key, _)| key.as_str()).collect();
            let from_shell = sh_reads(&dir, &text, &keys);
            // `parse` once rather than `field` per key: each `field` call reads
            // the whole file, and this one has thousands of keys.
            let parsed = parse(&text).expect("every line here is a valid assignment");
            for ((key, value), shell) in expected.iter().zip(&from_shell) {
                assert_eq!(
                    shell.as_deref(),
                    Some(value.as_str()),
                    "sh disagrees with the encoder for {key}"
                );
                assert_eq!(
                    parsed.get(key.as_str()).map(String::as_str),
                    Some(value.as_str()),
                    "this parser disagrees with sh for {key}"
                );
            }
        }

        /// The value assigned before each case in the next test, spelled with
        /// letters outside [`RHS_ALPHABET`] so no right-hand side can produce it.
        const PRIOR: &str = "PRIOR";

        /// Right-hand sides for the next test, valid and malformed alike — with
        /// a newline, so a value can open a string or a continuation that
        /// swallows the line after it, and `:` with `~`, which `sh` expands.
        /// `>`, `<`, `|`, `&` and `(` are left out because `sh` would act on them
        /// (create a file, open a pipe); their refusal is pinned by
        /// `a_malformed_assignment_reads_as_absent` instead.
        const RHS_ALPHABET: &[char] = &[
            'a', ' ', '\\', '"', '\'', '$', '`', ';', '#', '~', ':', '\n',
        ];

        /// Right-hand sides that are well-formed and must read as exactly what
        /// `sh` assigns — never `None`. Without this list, a parser that
        /// returned `None` for everything would satisfy "`sh`'s value or `None`".
        const MUST_READ: &[&str] = &[
            "",
            "a",
            r"a\a",
            r"\a",
            r"a\ a",
            r"a\$",
            "a ",
            "'a'",
            "\"a\"",
            "\"\"",
            "a:a",
            r"\~",
            "a~",
            "'~'",
            // A `#` comment after a blank ends the value, for every value form;
            // a `#` that does not follow a blank is part of the value.
            "a #",
            "a #a",
            "a\t# the distro",
            "'a' #a",
            "\"a\" #",
            " #a",
            "a#",
            "#a",
        ];

        /// The rule the module states, against `sh`, for every right-hand side
        /// up to three characters over a hostile alphabet. The file is
        /// `L=PRIOR`, `K=PRIOR`, `K=<rhs>`, then `L=b` — so a right-hand side
        /// that opens a string or a continuation can swallow the next line, and
        /// both keys are checked: for each, the parser reads `sh`'s value or
        /// nothing. In particular it never reads back `PRIOR` where `sh` moved
        /// on, nor `b` where `sh` never reached it. Every [`MUST_READ`] case
        /// must read as a value.
        ///
        /// One `sh` per case: a malformed line can abort the source, so a
        /// shared file would stop at the first one.
        #[test]
        fn every_right_hand_side_reads_as_sh_assigns_it_or_not_at_all() {
            let dir = ScratchDir::new("any");
            let mut cases = every_string(RHS_ALPHABET, 3);
            cases.extend(MUST_READ.iter().map(|case| (*case).to_owned()));
            let mut read = 0usize;
            for rhs in &cases {
                let text = format!("L={PRIOR}\nK={PRIOR}\nK={rhs}\nL=b\n");
                read += assert_sh_or_none(&dir, &text, &["K", "L"]);
                if MUST_READ.contains(&rhs.as_str()) {
                    let shell = sh_reads(&dir, &text, &["K"]).remove(0);
                    assert!(
                        shell.is_some(),
                        "MUST_READ case K={rhs:?} is not valid to sh"
                    );
                    assert_eq!(
                        field(&text, "K"),
                        shell,
                        "K={rhs:?} is well-formed and must read as sh assigns it"
                    );
                }
            }
            // Printed so `--nocapture` reports the split, not only that it
            // cleared the bar.
            println!(
                "reach: {read} of {} key reads returned a value; the rest fail closed",
                cases.len() * 2
            );
            assert!(
                read > cases.len() / 10,
                "only {read} key reads returned a value; a parser that refuses \
                 nearly everything satisfies the rule vacuously"
            );
        }

        /// The files that run a command, checked against `sh` too, for every key
        /// they mention.
        #[test]
        fn a_file_that_runs_a_command_never_reads_as_something_sh_does_not_assign() {
            let dir = ScratchDir::new("commands");
            for text in [
                "ID=ubuntu\nNAME=\"foo\nID=debian\n\"\n",
                "ID=ubuntu\nNAME='foo\nID=debian\n'\n",
                "ID=ubuntu\nNAME=foo\\\nID=debian\n",
                "ID=ubuntu\nX=1; ID=debian\n",
                "ID=ubuntu\nexport ID=debian\n",
                "ID=ubuntu\nVERSION_ID=24.04\nunset VERSION_ID\n",
                "ID=ubuntu\nVERSION_ID=24.04\nX= unset VERSION_ID\n",
                "ID=ubuntu\nVERSION_ID=a:~\n",
            ] {
                assert_sh_or_none(&dir, text, &["ID", "NAME", "VERSION_ID", "X"]);
            }
        }
    }
}
