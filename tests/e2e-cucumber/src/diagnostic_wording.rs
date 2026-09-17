// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Keeps the TUI driver's wait diagnostics and the doc comments describing
//! them from disagreeing.
//!
//! `tests/e2e/tui_driver.rs` documents two deliberately different conventions
//! for its wait messages: `wait_for_screen_where` interpolates a caller's
//! clause verbatim into four templates, and `terminal_state_after_wait` quotes
//! its own bare marker in two. Both are prose, and prose is the part no test
//! runs — this file's own history is the argument for these checks, with three
//! consecutive review rounds finding those comments describing code they did
//! not match, each caught only because a person read both.
//!
//! So this asserts BOTH directions, because the two failures that actually
//! recurred point opposite ways:
//!
//! - a template changes and the doc still describes the old one (drift), and
//! - the doc describes something the code never had (wrong when written) —
//!   which is what the last two rounds found, and what a one-sided check that
//!   only looks for templates in the code would miss entirely.
//!
//! Each documented rendering must appear in its doc block AND its template
//! must appear in the code, and the number of `{describe}` diagnostics must be
//! the number the doc claims — so a fifth one cannot be added while the doc
//! goes on saying four.
//!
//! Source text rather than rendered output on purpose: rendering a message
//! needs a live pty, a child process and a reader thread, which is what the
//! scenarios themselves are for. What rots here is wording, and the wording is
//! in the file. The weakness that remains is that this reads source rather
//! than behaviour: a template could be spelled correctly and never reached.

/// The driver whose diagnostics and doc comments this file holds together.
const DRIVER: &str = "tests/e2e/tui_driver.rs";

/// The step file whose marker the driver's doc quotes as its example.
const MARKER_CALLER: &str = "tests/e2e/dash_steps.rs";

/// The marker `send_until`'s only caller passes, quoted in the driver's doc as
/// the example rendering. Pinned here so the doc cannot go stale against it.
const EXAMPLE_MARKER: &str = "● Observe";

fn read(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The file's doc comments as flowing text: `///` markers stripped and line
/// breaks collapsed, so a documented rendering can be searched for without
/// caring where the author happened to wrap it.
fn doc_text(source: &str) -> String {
    source
        .lines()
        .map(str::trim_start)
        .filter_map(|line| {
            line.strip_prefix("///")
                .or_else(|| line.strip_prefix("//!"))
        })
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
}

/// A format placeholder as it appears in source: `arg("describe")` is
/// `{describe}`. Built rather than written so no literal in this file carries
/// an uninterpolated `{…}`, which clippy reads as a missed interpolation.
fn arg(name: &str) -> String {
    format!("{{{name}}}")
}

/// One documented diagnostic: the `format!` template in the code, and the
/// rendering the doc comment shows for it. Both must be present.
struct Documented {
    template: String,
    documented_as: String,
}

fn clause_diagnostics() -> Vec<Documented> {
    let describe = arg("describe");
    let timeout = arg("timeout:?");
    vec![
        Documented {
            template: format!("panicked while waiting until {describe}"),
            documented_as: format!("waiting until {describe}"),
        },
        Documented {
            template: format!("timed out after {timeout} waiting until {describe}"),
            documented_as: format!("timed out … waiting until {describe}"),
        },
        Documented {
            template: format!("before {describe}."),
            documented_as: format!("before {describe}"),
        },
        Documented {
            // The fourth is assembled from two fragments: the drain message
            // interpolates a context that the label is built into.
            template: format!("draining the final frame{}", arg("context")),
            documented_as: format!("draining the final frame, waiting until {describe}"),
        },
    ]
}

fn marker_diagnostics() -> Vec<Documented> {
    let marker = arg("marker:?");
    vec![
        Documented {
            template: format!("panicked while waiting for {marker}"),
            documented_as: format!("panicked while waiting for {EXAMPLE_MARKER:?}"),
        },
        Documented {
            template: format!("before {marker} appeared."),
            documented_as: format!("before {EXAMPLE_MARKER:?} appeared."),
        },
    ]
}

fn assert_both_sides(source: &str, documented: &[Documented], owner: &str) {
    let docs = doc_text(source);
    for Documented {
        template,
        documented_as,
    } in documented
    {
        assert!(
            source.contains(template.as_str()),
            "{DRIVER} no longer contains the template {template:?}, but `{owner}`'s doc \
             comment still describes it — update the doc with this change"
        );
        assert!(
            docs.contains(documented_as.as_str()),
            "`{owner}`'s doc comment no longer shows {documented_as:?}, but {DRIVER} still \
             emits that diagnostic — the doc must keep naming every message it claims to \
             enumerate, or this guard is checking nothing"
        );
    }
}

/// The four clause-shaped diagnostics, in both directions.
#[test]
fn every_clause_diagnostic_is_both_emitted_and_documented() {
    assert_both_sides(
        &read(DRIVER),
        &clause_diagnostics(),
        "wait_for_screen_where",
    );
}

/// The two bare-marker diagnostics, in both directions.
#[test]
fn every_marker_diagnostic_is_both_emitted_and_documented() {
    assert_both_sides(
        &read(DRIVER),
        &marker_diagnostics(),
        "terminal_state_after_wait",
    );
}

/// The doc says "four diagnostics". A fifth one added without touching the doc
/// would satisfy every check above — each documented message would still be
/// present — while the count silently became a lie. This is the direction a
/// list of substrings cannot see.
#[test]
fn the_documented_count_of_clause_diagnostics_is_the_real_one() {
    let source = read(DRIVER);
    let describe = arg("describe");
    // The drain message reaches the label through `{context}` rather than
    // naming it, so it is counted by its own placeholder. Both are code lines:
    // the doc's own occurrences sit behind `///` and are filtered out.
    let context = arg("context");
    // Occurrences, not lines: two placeholders on one line are two diagnostics
    // as far as a caller's clause is concerned, and counting lines would miss
    // the second.
    let emitted: usize = source
        .lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .map(|line| {
            line.matches(describe.as_str()).count() + line.matches(context.as_str()).count()
        })
        .sum();
    let documented = clause_diagnostics().len();
    assert_eq!(
        emitted, documented,
        "{DRIVER} emits {emitted} diagnostics carrying the caller's clause, but \
         `wait_for_screen_where`'s doc enumerates {documented}. A clause has to read \
         correctly after every one of them, so a new diagnostic belongs in that list — \
         and in this file."
    );
}

/// The driver's doc quotes a real caller's marker as its example. Nothing else
/// connects the two files, so changing the marker would leave the doc quietly
/// illustrating a call that no longer happens.
#[test]
fn the_documented_example_marker_is_the_one_its_caller_passes() {
    let caller = read(MARKER_CALLER);
    assert!(
        caller.contains(&format!("{EXAMPLE_MARKER:?}")),
        "{MARKER_CALLER} no longer passes {EXAMPLE_MARKER:?}, which \
         `terminal_state_after_wait`'s doc quotes as its worked example — update the \
         doc, and this constant, to whatever it passes now"
    );
}
