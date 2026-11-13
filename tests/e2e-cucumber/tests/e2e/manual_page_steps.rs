// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `manual_page.feature`: the rocm(1) page the binary under test
//! prints for the installer (`rocm __man-page`).
//!
//! The steps read the roff source directly, not `man`'s rendering of it, so the
//! scenarios need nothing on the runner beyond the binary: an entry's term is
//! the line after its `.TP` request, with roff's font and character escapes
//! decoded. An escape the decoder does not know fails the step rather than
//! being passed through, so a change in how the page is written is noticed
//! instead of silently mangling a term. That the installed page is found and
//! shown by `man rocm` is lifecycle-36, on the lane that installs it.

use cucumber::{then, when};

use crate::{E2eWorld, run_rocm, run_rocm_ok};

/// `text` with the roff escapes the page uses decoded to the characters a
/// reader sees: font changes dropped, `\-` a hyphen, `\(aq` and `\*(Aq` an
/// apostrophe, `\&` (a zero-width guard) dropped, `\e` a backslash.
fn decode_roff(text: &str) -> String {
    const ESCAPES: &[(&str, &str)] = &[
        ("\\fB", ""),
        ("\\fI", ""),
        ("\\fR", ""),
        ("\\fP", ""),
        ("\\-", "-"),
        ("\\(aq", "'"),
        ("\\*(Aq", "'"),
        ("\\&", ""),
        ("\\e", "\\"),
    ];
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('\\') {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let (escape, replacement) = ESCAPES
            .iter()
            .find(|(escape, _)| tail.starts_with(escape))
            .unwrap_or_else(|| panic!("unknown roff escape in {text:?}; teach decode_roff it"));
        out.push_str(replacement);
        rest = &tail[escape.len()..];
    }
    out.push_str(rest);
    out
}

/// The decoded term of every `.TP` entry in the section headed `name`.
fn section_terms(roff: &str, name: &str) -> Vec<String> {
    let heading = format!(".SH {name}");
    let heading_quoted = format!(".SH \"{name}\"");
    let mut lines = roff
        .lines()
        .skip_while(|line| *line != heading && *line != heading_quoted);
    assert!(
        lines.next().is_some(),
        "no section {name} in the page:\n{roff}"
    );
    let body: Vec<&str> = lines.take_while(|line| !line.starts_with(".SH ")).collect();
    body.windows(2)
        .filter(|pair| pair[0] == ".TP")
        .map(|pair| decode_roff(pair[1]).trim().to_owned())
        .collect()
}

/// The commands `rocm --help` lists, without clap's own `help`.
fn help_commands(help: &str) -> Vec<String> {
    let mut names: Vec<String> = help
        .lines()
        .skip_while(|line| !line.trim_start().starts_with("Commands:"))
        .skip(1)
        .take_while(|line| !line.trim().is_empty())
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| *name != "help")
        .map(str::to_owned)
        .collect();
    names.sort();
    names
}

#[when("the user reads the rocm manual page")]
async fn when_user_reads_manual(world: &mut E2eWorld) {
    let roff = run_rocm_ok(world, &["__man-page"]);
    assert!(
        roff.starts_with(".TH ROCM 1 "),
        "rocm __man-page did not print a rocm(1) page:\n{roff}"
    );
    world.cli_output = Some(roff);
}

#[then("the manual describes every command the help lists, and none it hides")]
async fn then_manual_lists_help_commands(world: &mut E2eWorld) {
    let roff = world.cli_output.clone().expect("the manual was not read");
    let mut listed = section_terms(&roff, "COMMANDS");
    listed.sort();
    let help = run_rocm_ok(world, &["--help"]);
    let visible = help_commands(&help);
    assert!(
        visible.len() > 1,
        "could not read the commands from rocm --help:\n{help}"
    );
    assert_eq!(
        listed, visible,
        "the manual's COMMANDS differ from what rocm --help lists\nmanual:\n{roff}"
    );
}

#[then("rocm accepts every example the manual gives")]
async fn then_examples_are_accepted(world: &mut E2eWorld) {
    let roff = world.cli_output.clone().expect("the manual was not read");
    let examples = section_terms(&roff, "EXAMPLES");
    assert!(
        !examples.is_empty(),
        "the manual gives no examples:\n{roff}"
    );
    for example in &examples {
        let words: Vec<&str> = example.split_whitespace().collect();
        assert_eq!(
            words.first(),
            Some(&"rocm"),
            "example {example:?} is not a rocm command"
        );
        // `--help` after the example's own words: rocm parses every word, so an
        // unknown command or flag is still refused, but nothing is run.
        let mut args: Vec<&str> = words[1..].to_vec();
        args.push("--help");
        let (stdout, stderr, rc) = run_rocm(world, &args);
        assert_eq!(
            rc, 0,
            "rocm refused the manual's example `{example}`:\n{stdout}{stderr}"
        );
    }
}
