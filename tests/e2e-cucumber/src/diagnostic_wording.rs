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
//! runs — this file's own history is the argument for these checks, with
//! several consecutive review rounds finding those comments describing code
//! they did not match, each caught only because a person read both.
//!
//! What is asserted, for each convention:
//!
//! - every documented rendering appears in the doc block of the function that
//!   claims to enumerate it — not merely somewhere in the file, so the
//!   enumeration cannot be moved off the function it describes;
//! - every `format!` template it names is still in the code, INCLUDING the
//!   fragments of any message assembled in more than one place;
//! - the count the doc states in words matches the number of diagnostics that
//!   actually carry the caller's text, so a message cannot be added or removed
//!   while the prose goes on claiming a different number;
//! - the caller's clause is never Debug-formatted, which would quote it a
//!   second time on top of whatever quoting the caller supplied;
//! - the marker the doc quotes as its worked example is still the one its
//!   caller passes.
//!
//! Both directions are checked because both have failed here: a template has
//! been reworded while the doc kept describing the old one, and a doc has been
//! written describing something the code never had. A guard that only looks
//! for templates in the code sees neither.
//!
//! Source text rather than rendered output on purpose: rendering a message
//! needs a live pty, a child process and a reader thread, which is what the
//! scenarios themselves are for. What rots here is wording, and the wording is
//! in the file. The weakness that remains is exactly that: a template can be
//! spelled correctly here and never reached at runtime.

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

/// The doc comment attached to `fn <name>`, as flowing text: the contiguous
/// `///` block immediately above it, markers stripped and wrapping collapsed.
///
/// Scoped to the function on purpose. Flattening the whole file would let an
/// enumeration be moved onto an unrelated method and still satisfy every check
/// — the doc would be somewhere, just not on the thing it claims to describe.
fn doc_block_of(source: &str, name: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let signature = lines
        .iter()
        .position(|line| line.contains(&format!("fn {name}(")))
        .unwrap_or_else(|| panic!("{DRIVER} no longer declares `fn {name}`"));

    let mut first = signature;
    while first > 0 {
        let candidate = lines[first - 1].trim_start();
        // Attributes and ordinary comments sit between a doc block and its
        // item, so step over them; anything else ends the block.
        if candidate.starts_with("///")
            || candidate.starts_with("#[")
            || candidate.starts_with("//")
        {
            first -= 1;
        } else {
            break;
        }
    }

    lines[first..signature]
        .iter()
        .map(|line| line.trim_start())
        .filter_map(|line| line.strip_prefix("///"))
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

/// One documented diagnostic: the `format!` fragments that build it in the
/// code, and the rendering the doc comment shows for it.
///
/// `templates` is a list because a message assembled in more than one place is
/// only as pinned as its least-pinned fragment. The drain message is the
/// standing example: its label arrives through a `{context}` built by a
/// separate function, and pinning only the outer template left that function
/// free to reword the label into a shape the doc denies.
///
/// `templates` must be looked for in CODE ONLY, never the raw file. Both sides
/// of this check live in the same source, and a doc rendering is often the
/// template spelled out — so a file-wide `contains` lets the prose vouch for
/// itself and the code side stops testing anything. That is not hypothetical:
/// adding a period to the third rendering, to match its `documented_as`, did
/// exactly this and went unnoticed for a commit.
struct Documented {
    templates: Vec<String>,
    documented_as: String,
}

fn clause_diagnostics() -> Vec<Documented> {
    let describe = arg("describe");
    let timeout = arg("timeout:?");
    let wanted = arg("wanted");
    vec![
        Documented {
            templates: vec![format!("panicked while waiting until {describe}: ")],
            // Distinguishing prefix on purpose: a bare `waiting until
            // {describe}` is a substring of the timeout rendering below, so
            // deleting this line from the doc would leave the check green.
            documented_as: format!("panicked while waiting until {describe}"),
        },
        Documented {
            templates: vec![format!(
                "timed out after {timeout} waiting until {describe}"
            )],
            documented_as: format!("timed out … waiting until {describe}"),
        },
        Documented {
            templates: vec![format!("before {describe}.")],
            documented_as: format!("before {describe}."),
        },
        Documented {
            // Two fragments in two functions: the drain message interpolates a
            // context, and the context is where the label is actually phrased.
            templates: vec![
                format!("draining the final frame{}", arg("context")),
                format!(", waiting until {wanted}"),
            ],
            documented_as: format!("draining the final frame, waiting until {describe}"),
        },
    ]
}

fn marker_diagnostics() -> Vec<Documented> {
    let marker = arg("marker:?");
    vec![
        Documented {
            templates: vec![format!("panicked while waiting for {marker}: ")],
            documented_as: format!("panicked while waiting for {EXAMPLE_MARKER:?}"),
        },
        Documented {
            templates: vec![format!("before {marker} appeared.")],
            documented_as: format!("before {EXAMPLE_MARKER:?} appeared."),
        },
    ]
}

/// The driver with every comment line removed — the only text a code-side
/// assertion may be made against. See [`Documented`] for why.
fn code_text(source: &str) -> String {
    source
        .lines()
        .map(str::trim_start)
        // `/*` and a leading `*` too: an own-line block comment is prose just
        // as a `//` line is, and could vouch for a template the same way.
        .filter(|line| !line.starts_with("//") && !line.starts_with("/*") && !line.starts_with('*'))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_both_sides(source: &str, documented: &[Documented], owner: &str) {
    let docs = doc_block_of(source, owner);
    let code = code_text(source);
    for Documented {
        templates,
        documented_as,
    } in documented
    {
        for template in templates {
            assert!(
                code.contains(template.as_str()),
                "{DRIVER} no longer contains the fragment {template:?}, but `{owner}`'s doc \
                 comment still describes the message it builds as {documented_as:?} — update \
                 the doc with this change"
            );
        }
        assert!(
            docs.contains(documented_as.as_str()),
            "`{owner}`'s own doc comment no longer shows {documented_as:?}, but {DRIVER} still \
             emits that diagnostic — the doc must keep naming every message it claims to \
             enumerate, or this guard is checking nothing"
        );
    }
}

/// The doc states its count in words. Assert the word, not this file's list
/// length: comparing the code against our own `Vec` would pass whatever the
/// prose said, which is the defect this test is named for.
fn assert_documented_count(source: &str, owner: &str, count: usize, noun: &str) {
    let spelled = match count {
        2 => "two",
        3 => "three",
        4 => "four",
        5 => "five",
        6 => "six",
        other => panic!("no spelling for {other}; add one when the list grows"),
    };
    let docs = doc_block_of(source, owner);
    assert!(
        docs.contains(&format!("{spelled} {noun}")),
        "`{owner}`'s doc comment does not say {spelled:?} {noun}, but that is how many there \
         are — the stated count and the real one must agree, or the prose is telling the next \
         reader something the code denies"
    );
}

/// The lines that build a wait diagnostic: every one of them ends its message
/// with the framed screen, which is what makes it a report about a stuck
/// terminal rather than an ordinary error string. Signatures, forwarding calls
/// and the clause `wait_for_screen` assembles for its own caller all mention
/// the same placeholders without being diagnostics, so counting placeholders
/// file-wide would count those too.
fn diagnostic_lines(source: &str) -> impl Iterator<Item = &str> {
    source
        .lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .filter(|line| line.contains("\\n{}"))
}

/// The four clause-shaped diagnostics, in both directions, fragments included.
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

/// A diagnostic added or removed without touching the doc satisfies every
/// check above — each documented message is still present — while the count
/// silently becomes a lie. Both the code's count and the doc's stated one are
/// read here, so neither side can drift alone.
#[test]
fn the_documented_count_of_clause_diagnostics_is_the_real_one() {
    let source = read(DRIVER);
    let expected = clause_diagnostics().len();
    assert_documented_count(&source, "wait_for_screen_where", expected, "diagnostics");

    // The drain message reaches the label through `{context}` rather than
    // naming it, so it is counted by its own placeholder. Occurrences, not
    // lines: two placeholders on one line are two diagnostics as far as a
    // caller's clause is concerned. Doc lines are excluded — they quote these
    // same placeholders.
    let describe = arg("describe");
    let context = arg("context");
    let emitted: usize = diagnostic_lines(&source)
        .map(|line| {
            line.matches(describe.as_str()).count() + line.matches(context.as_str()).count()
        })
        .sum();
    assert_eq!(
        emitted, expected,
        "{DRIVER} emits {emitted} diagnostics carrying the caller's clause, but \
         `wait_for_screen_where` documents {expected}. A clause has to read correctly after \
         every one of them, so a new diagnostic belongs in that list — and in this file."
    );
}

/// `terminal_state_after_wait`'s doc says a third path names no marker. A
/// third marker-bearing message would contradict it silently.
#[test]
fn the_documented_count_of_marker_diagnostics_is_the_real_one() {
    let source = read(DRIVER);
    let expected = marker_diagnostics().len();
    assert_documented_count(&source, "terminal_state_after_wait", expected, "messages");

    let marker = arg("marker:?");
    let emitted: usize = diagnostic_lines(&source)
        .map(|line| line.matches(marker.as_str()).count())
        .sum();
    assert_eq!(
        emitted, expected,
        "{DRIVER} emits {emitted} messages quoting a bare marker, but \
         `terminal_state_after_wait` documents {expected}"
    );
}

/// The drain message is the one diagnostic whose clause is optional at the
/// call: `drain_final_frame_where` takes an `Option`, and passing `None` would
/// leave every template above intact while the message silently stopped
/// carrying the caller's text. The doc counts it among the four, so the call
/// that makes that true is pinned too.
#[test]
fn the_drain_is_still_reached_with_the_callers_clause() {
    let code = code_text(&read(DRIVER));
    let describe = arg("describe");
    assert!(
        code.contains("drain_final_frame_where(Some(describe),"),
        "`wait_for_screen_where` no longer hands its clause to the drain, but its doc still \
         counts `draining the final frame, waiting until {describe}` among the four \
         diagnostics that carry it"
    );
}

/// The contract says the clause is interpolated verbatim. A `{describe:?}`
/// anywhere would quote it a second time on top of whatever the caller already
/// put in — and would also slip past the count above, which looks for the
/// undecorated placeholder.
#[test]
fn the_clause_label_is_never_debug_formatted() {
    let source = read(DRIVER);
    assert!(
        !source.contains(&arg("describe:?")),
        "a diagnostic Debug-formats the caller's clause, but `wait_for_screen_where` \
         documents it as interpolated verbatim — a caller that quotes its own text would \
         now be double-quoted"
    );
}

/// The driver's doc quotes a real caller's marker as its example. Nothing else
/// connects the two files, so changing the marker would leave the doc quietly
/// illustrating a call that no longer happens.
///
/// Matched at the call itself rather than anywhere in the file: a stale comment
/// still mentioning the old marker would otherwise satisfy this.
#[test]
fn the_documented_example_marker_is_the_one_its_caller_passes() {
    let caller = read(MARKER_CALLER);
    let call = format!("send_until(\"4\", {EXAMPLE_MARKER:?}");
    assert!(
        caller.contains(&call),
        "{MARKER_CALLER} no longer calls `send_until` with {EXAMPLE_MARKER:?}, which \
         `terminal_state_after_wait`'s doc quotes as its worked example — update the doc, \
         and this constant, to whatever it passes now"
    );
}
