// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Drift guard for what `skills/rocm-doctor` quotes from the CLI: the shape of
//! `reference.md`'s catalog, the counts `SKILL.md` states about it, and the
//! exit-code and `fix` field names it lists.
//!
//! The `rocm_doctor_skill.feature` scenarios compare that table against the
//! real `rocm fix` listing, cell for cell, so a cell the table spells its own
//! way is a contract failure. Those scenarios need a built `rocm` binary and
//! only run under `cargo xtask e2e`; this file runs in the ordinary `cargo
//! test` set, so the malformed cell is caught without one.
//!
//! Narrow on purpose: the CLI is authoritative for *which* fixes exist and what
//! they are scoped to, and the feature file already checks that. For the
//! catalog this checks only that the document uses the vocabulary the CLI
//! prints, so the comparison there can stay a literal one. For `SKILL.md` it
//! checks the print-only count against the table, and the exit-code, `fix`
//! field and `fix.class` lists below against what the CLI publishes.
//!
//! Fixture sentences in `SKILL.md`. Tests read these phrases from it, so reword
//! them only together with the test that reads them. Each test panics naming
//! the phrase it cannot find; whitespace and line wrapping do not matter. The
//! phrases are the constants in `e2e_cucumber::skill_anchors`, and
//! `every_skill_anchor_is_named_here_and_is_in_the_skill` fails when one listed
//! there is missing from this list or from `SKILL.md`:
//!
//! - "requires `contract_version` N", read by `skill-07` in `skill_steps.rs`;
//! - "The other N are **print-only**", read by
//!   `free_standing_catalog_counts_match_the_table` below;
//! - "by its `exit_codes` name", which starts the exit-code list read by
//!   `exit_code_names_listed_in_the_skill_are_the_manifest_failure_exits`;
//! - "a `fix` whose ... are the fields", the `fix` field list read by
//!   `fix_fields_quoted_in_the_skill_exist_in_the_diagnosis_json`;
//! - "is the entry's `fix.class`", which starts step 3's list of classes read by
//!   `fix_classes_listed_in_the_skill_are_the_ones_the_cli_serializes`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration-test crate: its helpers sit outside `#[test]` fns, and a failure must fail the test"
)]

use e2e_cucumber::flatten_whitespace;
use e2e_cucumber::skill_anchors::{
    EXIT_CODES as EXIT_CODES_MARKER, FIX_CLASSES as FIX_CLASSES_MARKER, FIX_FIELDS_END,
    FIX_FIELDS_START, PRINT_ONLY_CLAUSE, PRINT_ONLY_COUNT,
};
use rocm_core::fix::FixClass;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The platform atoms `rocm fix` prints, which it builds by joining a recipe's
/// `applies_on` with `/` (see `FixRecipe` in `crates/rocm-core/src/fix/catalog.rs`).
/// A catalog cell is any non-empty `/`-joined subset of these.
const PLATFORM_ATOMS: &[&str] = &["linux", "windows", "wsl"];

/// The marker words `rocm fix`'s listing prints: `FixClass::marker` of every
/// class, lower-cased to match the table's style.
fn marker_words() -> BTreeSet<String> {
    FixClass::ALL
        .iter()
        .map(|class| class.marker().to_lowercase())
        .collect()
}

/// `yes`/`no`, recognised only so this guard can name them explicitly as the
/// vocabulary the class rewrite retired, rather than reporting them as merely
/// unrecognised.
const RETIRED_MARKER_WORDS: &[&str] = &["yes", "no"];

fn reference_md_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("skills")
        .join("rocm-doctor")
        .join("reference.md")
}

/// `(fix-id, os-scope cell, marker cell)` for every catalog row, taken the same
/// way `skill_steps::parse_reference_catalog` takes them: a leading `|`, at
/// least five cells, and a backticked `fix-*` or `info-*` id first.
fn catalog_rows(md: &str) -> Vec<(String, String, String)> {
    let mut rows = Vec::new();
    for line in md.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        if cells.len() < 5 {
            continue;
        }
        let id = cells[0].trim_matches('`');
        // `info-` as well as `fix-`: the catalog's DIAGNOSE-ONLY entries carry
        // an `info-` id precisely because they have no fix to offer, and they
        // are rows in this table like any other. Matching `fix-` alone dropped
        // them silently, so the table could disagree with the CLI about them
        // and nothing would say so.
        if !e2e_cucumber::is_catalog_id(id) {
            continue;
        }
        rows.push((
            id.to_owned(),
            cells[1].to_owned(),
            (*cells.last().expect("row has cells")).to_owned(),
        ));
    }
    rows
}

fn skill_md_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("skills")
        .join("rocm-doctor")
        .join("SKILL.md")
}

/// How many catalog rows have a marker other than a plain `print-only`, which is
/// the complement of the set `SKILL.md`'s "the other N are print-only" counts.
/// The rows and their marker cells are [`catalog_rows`]'s, the ones the
/// vocabulary checks read, so the count cannot read a different column.
///
/// Deliberately not a count of "auto-applicable": that phrase is per host, and
/// `reference.md` lists which ids it covers on each one (three ids across all
/// hosts, `fix-9-igpu-dgpu` not among them). `SKILL.md` claims a count of
/// print-only entries, so counting the complement checks that claim.
fn catalog_not_print_only_count(md: &str) -> usize {
    catalog_rows(md)
        .iter()
        .filter(|(_, _, marker)| marker != "print-only")
        .count()
}

/// The count reads the marker the vocabulary checks read (the last cell), and
/// counts only catalog rows: a header, a separator, a row with too few cells and
/// a row whose first cell is not a catalog id are not entries.
#[test]
fn the_not_print_only_count_reads_the_rows_and_markers_the_other_checks_read() {
    let md = "\
| id | OS | cause | symptom | auto |
|---|---|---|---|---|
| `fix-1-a` | linux | c | s | print-only |
| `fix-2-b` | linux | c | s | auto |
| `info-3-c` | linux/windows | c | s | diagnose-only |
| `fix-4-d` | linux | c | s | extra | print-only |
| `fix-5-e` | linux | c | auto |
| `note` | linux | c | s | auto |
";
    let markers: Vec<String> = catalog_rows(md).into_iter().map(|row| row.2).collect();
    assert_eq!(
        markers,
        ["print-only", "auto", "diagnose-only", "print-only"]
    );
    // `fix-4-d` has `extra` in its fifth cell and `print-only` last: a count
    // reading the fifth cell would say 3.
    assert_eq!(catalog_not_print_only_count(md), 2);
}

/// Two free-standing counts restate the table in prose rather than in cells a
/// parser already checks: `reference.md`'s "(N failure modes)" heading, and
/// `SKILL.md`'s "the other M are print-only" line. `rocm_doctor_skill.feature`
/// does not parse them, so they can go stale even while the table itself
/// stays correct, and says that this file checks both counts instead. That is
/// exactly what happened here (the
/// catalog grew by one row and both numbers were left behind). This does not
/// reopen that scope decision; it only adds a floor cheap enough that the next
/// drift fails a `cargo test` instead of waiting for someone to count rows by
/// hand.
#[test]
fn free_standing_catalog_counts_match_the_table() {
    let reference_md = std::fs::read_to_string(reference_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", reference_md_path().display()));
    let rows = catalog_rows(&reference_md);
    assert!(
        !rows.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );
    let total = rows.len();
    let not_print_only = catalog_not_print_only_count(&reference_md);

    let heading = reference_md
        .lines()
        .find(|line| line.trim_start().starts_with("## Closed catalog ("))
        .unwrap_or_else(|| {
            panic!(
                "no '## Closed catalog (N failure modes)' heading found in {}",
                reference_md_path().display()
            )
        });
    let heading_count: usize = heading
        .split('(')
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("could not parse the failure-mode count out of {heading:?}"));
    assert_eq!(
        heading_count, total,
        "skills/rocm-doctor/reference.md's '(N failure modes)' heading says {heading_count}, \
         but the table has {total} rows"
    );

    let skill_md = std::fs::read_to_string(skill_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", skill_md_path().display()));
    // "The other N" and "are **print-only**" sit on adjacent source lines that
    // wrap a single sentence; `number_after_phrase` joins the doc on whitespace
    // first, so a line-scoped search cannot miss the number. The clause has to
    // follow the number directly: a "print-only" somewhere later in the sentence
    // would not say what the number counts.
    let (print_only_count, rest) = e2e_cucumber::number_after_phrase(&skill_md, PRINT_ONLY_COUNT)
        .unwrap_or_else(|| {
            panic!(
                "no 'The other N ... print-only' phrase with a number found in {}",
                skill_md_path().display()
            )
        });
    assert!(
        rest.starts_with(PRINT_ONLY_CLAUSE),
        "'The other {print_only_count}' in {} is not followed by 'are **print-only**' as expected: {:?}",
        skill_md_path().display(),
        rest.chars().take(60).collect::<String>()
    );
    let print_only_count = usize::try_from(print_only_count).expect("print-only count fits usize");
    assert_eq!(
        print_only_count,
        total - not_print_only,
        "skills/rocm-doctor/SKILL.md says {print_only_count} fixes are print-only, but the \
         table has {total} rows of which {not_print_only} carry some other marker \
         ({} expected)",
        total - not_print_only
    );
}

#[test]
fn catalog_os_scopes_use_the_cli_spellings() {
    // A shorthand such as `both` reads as "every platform" but cannot say
    // which, so any mapping of it back onto the CLI's atoms is a guess --
    // `linux/windows` for a row that meant linux/windows/wsl drops `wsl`
    // silently, which is precisely the mismatch the skill contract exists to
    // catch. The document does not get to invent one: it spells the platforms
    // out, the comparison stays literal, and nothing has to be normalised.
    let md = std::fs::read_to_string(reference_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", reference_md_path().display()));
    let rows = catalog_rows(&md);
    assert!(
        !rows.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );

    let allowed: BTreeSet<&str> = PLATFORM_ATOMS.iter().copied().collect();
    let bad: Vec<&(String, String, String)> = rows
        .iter()
        .filter(|(_, scope, _)| {
            scope.is_empty() || scope.split('/').any(|atom| !allowed.contains(atom))
        })
        .collect();
    assert!(
        bad.is_empty(),
        "skills/rocm-doctor/reference.md scopes these fixes with words `rocm fix` never prints: \
         {bad:?}.\n\
         An OS-scope cell must be a `/`-joined list of {PLATFORM_ATOMS:?}, matching the recipe's \
         `applies_on` in crates/rocm-core/src/fix/catalog.rs. Spell the platforms out instead of \
         introducing a shorthand: the skill contract compares these cells literally, and a \
         shorthand would have to be translated -- which is how a dropped platform hides."
    );
}

#[test]
fn catalog_markers_use_the_cli_vocabulary() {
    // `skill_steps::parse_fix_listing` and `parse_reference_catalog` drop any
    // row whose marker is unrecognised rather than panic, so a reclassified
    // entry whose table cell still says `yes`/`no` -- the vocabulary the
    // `FixClass` rewrite retired -- disappears from the comparison silently
    // instead of failing on a marker mismatch. That is exactly the trap that
    // let `fix-9-igpu-dgpu`'s `NEEDS-ARG` reclassification vanish from the
    // skill-01 scenario's set comparison instead of showing up as a mismatch
    // on skill-02. This guard runs in the ordinary `cargo test` set, with no
    // built `rocm` binary needed, so the retired word is caught here first.
    let md = std::fs::read_to_string(reference_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", reference_md_path().display()));
    let rows = catalog_rows(&md);
    assert!(
        !rows.is_empty(),
        "no catalog rows found in {}",
        reference_md_path().display()
    );

    // A marker cell is a bare word, or a word plus a per-platform exception --
    // `print-only (auto on windows)` -- for the one entry whose behaviour
    // depends on more than the host. Only the base word is checked against the
    // vocabulary; `parse_marker_cell` in skill_steps.rs owns the exception
    // syntax itself.
    let base_word = |cell: &str| -> String {
        match cell.find('(') {
            Some(i) => cell[..i].trim().to_owned(),
            None => cell.trim().to_owned(),
        }
    };

    let allowed = marker_words();
    let retired: BTreeSet<&str> = RETIRED_MARKER_WORDS.iter().copied().collect();
    let bad: Vec<(&String, String)> = rows
        .iter()
        .map(|(id, _, marker)| (id, base_word(marker)))
        .filter(|(_, word)| !allowed.contains(word))
        .collect();
    assert!(
        bad.is_empty(),
        "skills/rocm-doctor/reference.md marks these fixes with words `rocm fix` never prints: \
         {bad:?}.\n\
         A marker cell must be one of {allowed:?}, matching `FixClass::marker` in \
         crates/rocm-core/src/fix/options.rs (lower-cased). {} of those are `yes`/`no`, the vocabulary \
         the class rewrite retired -- replace with the CLI's own marker word.",
        bad.iter()
            .filter(|(_, word)| retired.contains(word.as_str()))
            .count()
    );
}

/// Every backtick-delimited span of `text`, in order, with the text that
/// follows its closing backtick.
fn backticked_with_tail(text: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        out.push((&after[..close], &after[close + 1..]));
        rest = &after[close + 1..];
    }
    out
}

fn is_identifier(span: &str) -> bool {
    !span.is_empty()
        && span.starts_with(|c: char| c.is_ascii_lowercase())
        && span
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn skill_md_text() -> String {
    std::fs::read_to_string(skill_md_path())
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", skill_md_path().display()))
}

/// The part of `flat` after `marker`, or a panic naming the missing marker: a
/// parser that silently found nothing would turn its check into a no-op.
fn after_marker<'a>(flat: &'a str, marker: &str) -> &'a str {
    match flat.find(marker) {
        Some(at) => &flat[at + marker.len()..],
        None => panic!(
            "marker missing: {marker:?} is no longer in SKILL.md. It is a fixture phrase this \
             test reads; reword it back, or update the test, its constant in \
             `e2e_cucumber::skill_anchors` and the list in this file's module doc"
        ),
    }
}

/// The bullet heads of the list that starts after `marker` in `skill_md`: the
/// `name` of each " - `name` — " bullet, in order. Reads the text with each
/// paragraph's whitespace collapsed, so re-wrapping SKILL.md does not matter, and
/// with a [`PARAGRAPH_BREAK`] where a blank line was.
///
/// The list ends at the first " - " that does not open such a bullet: the next
/// list (step 3's "- (a)"), a list under a heading (" - **Lemonade**"), or a
/// bullet of another shape. Once it has a bullet it also ends at a blank line
/// (the first bullet may follow the paragraph that introduces the list), so a
/// paragraph after the list that holds no " - " is not read as part of the last
/// bullet, and a matching bullet further down does not join the list. A list cut
/// short thus does not borrow a bullet from further down the file, which skipping
/// ahead to the next " - `" did. A bullet whose own text holds a blank line ends
/// the list early, which shows as a missing name rather than passing.
fn bullet_heads_after(skill_md: &str, marker: &str) -> Vec<String> {
    let flat = flatten_paragraphs(skill_md);
    let mut heads = Vec::new();
    let mut rest = after_marker(&flat, marker);
    while let Some(at) = rest.find(" - ") {
        if !heads.is_empty() && rest[..at].contains(PARAGRAPH_BREAK.trim()) {
            break;
        }
        let Some(head) = rest[at + 3..].strip_prefix('`') else {
            break;
        };
        let Some((name, tail)) = head.split_once('`') else {
            break;
        };
        let Some(tail) = tail.strip_prefix(" — ") else {
            break;
        };
        heads.push(name.to_owned());
        rest = tail;
    }
    heads
}

/// What [`flatten_paragraphs`] puts where a blank line was: a pilcrow, which
/// SKILL.md does not use.
const PARAGRAPH_BREAK: &str = " ¶ ";

/// `text` with the whitespace inside each paragraph collapsed (as
/// [`flatten_whitespace`] does for the whole text) and the paragraphs joined by
/// [`PARAGRAPH_BREAK`], so a reader of the flat text can still see where one ended.
fn flatten_paragraphs(text: &str) -> String {
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                paragraphs.push(flatten_whitespace(&current));
                current.clear();
            }
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }
    if !current.is_empty() {
        paragraphs.push(flatten_whitespace(&current));
    }
    paragraphs.join(PARAGRAPH_BREAK)
}

/// The exit-code names SKILL.md quotes for `rocm fix`: the bullet heads of the
/// list that follows "by its `exit_codes` name".
fn exit_code_names_in(skill_md: &str) -> BTreeSet<String> {
    bullet_heads_after(skill_md, EXIT_CODES_MARKER)
        .into_iter()
        .collect()
}

/// The manifest's non-zero exit-code names. SKILL.md lists the failures an agent
/// can meet; `ok` is the success code and is not one of them.
fn published_failure_exits() -> BTreeSet<String> {
    rocm_core::catalog_manifest()
        .exit_codes
        .keys()
        .filter(|name| name.as_str() != "ok")
        .cloned()
        .collect()
}

/// `Err` when the exit-code list in `skill_md` is not the manifest's failure
/// exits. A list the parser reads as empty, or that names an exit the manifest
/// does not publish (`not-applicable`), differs from them too, so it cannot pass.
fn check_exit_code_list(skill_md: &str) -> Result<(), String> {
    let listed = exit_code_names_in(skill_md);
    let published = published_failure_exits();
    if listed == published {
        Ok(())
    } else {
        Err(format!(
            "the exit-code list in SKILL.md is {listed:?}; the manifest's non-zero \
             `exit_codes` are {published:?}"
        ))
    }
}

/// SKILL.md reads the manifest's `exit_codes` by name, so a name it quotes that
/// the manifest does not publish sends an agent looking for a code that never
/// comes, and a published one it leaves out is a failure the skill has no
/// instruction for. `the_published_exit_codes_are_the_ones_apply_returns` in
/// rocm-core pins the manifest's keys; this pins the skill's list to them.
#[test]
fn exit_code_names_listed_in_the_skill_are_the_manifest_failure_exits() {
    if let Err(problem) = check_exit_code_list(&skill_md_text()) {
        panic!("{problem}");
    }
}

/// The same check, run on lists that are wrong, must fail: otherwise it would
/// pass for whatever the parser happened to read.
#[test]
fn a_wrong_exit_code_list_is_caught() {
    let flat = flatten_whitespace(&skill_md_text());
    for (what, wrong) in [
        ("a misspelt name", " - `fail` — "),
        ("a name it cannot read", " - `not-applicable` — "),
    ] {
        let text = replace_once(&flat, " - `failed` — ", wrong);
        assert!(
            check_exit_code_list(&text).is_err(),
            "{what} in the exit-code list went unnoticed"
        );
    }
    // A published exit left out: its bullet's text runs on in the bullet before.
    let left_out = replace_once(&flat, " - `internal` — ", " ");
    assert!(
        check_exit_code_list(&left_out).is_err(),
        "an exit left out of the list went unnoticed"
    );
    // A name the manifest does not publish, added as a bullet of its own.
    let extra = replace_once(
        &flat,
        " - `declined` — ",
        " - `retried` — try again. - `declined` — ",
    );
    assert!(
        check_exit_code_list(&extra).is_err(),
        "an extra name in the list went unnoticed"
    );
    // The list cut short, and the missing name a bullet further down the file:
    // the list has ended by then, so the later bullet is not part of it.
    let cut_short = format!(
        "{} - `declined` — a bullet in a later section.",
        replace_once(&flat, " - `declined` — ", " ")
    );
    assert!(
        check_exit_code_list(&cut_short).is_err(),
        "a list cut short was completed by a bullet further down the file"
    );
    // A paragraph with no " - " after the list, then a bullet the list lacks: the
    // blank line ends the list, so the bullet below it is not part of it.
    let raw = skill_md_text().replace("\r\n", "\n");
    let after_paragraph = replace_once(
        &replace_once(&raw, " - `declined` — ", " "),
        "(step (c) above).\n",
        "(step (c) above).\n\nA paragraph that holds no dash.\n\n- `declined` — a later bullet.\n",
    );
    assert!(
        check_exit_code_list(&after_paragraph).is_err(),
        "a list ended by a paragraph was completed by a bullet below it"
    );
}

/// `text` with the one occurrence of `from` replaced by `to`; a panic when there
/// is not exactly one, so a case cannot pass by changing nothing.
fn replace_once(text: &str, from: &str, to: &str) -> String {
    assert_eq!(
        text.matches(from).count(),
        1,
        "{from:?} must occur exactly once in SKILL.md for this case to mean anything"
    );
    text.replacen(from, to, 1)
}

/// The `fix.class` names SKILL.md's step 3 explains: the bullet heads of the list
/// after "is the entry's `fix.class`".
fn fix_class_names_in(skill_md: &str) -> BTreeSet<String> {
    bullet_heads_after(skill_md, FIX_CLASSES_MARKER)
        .into_iter()
        .collect()
}

/// The names `fix.class` takes in the JSON the CLI prints: every [`FixClass`]
/// run through its serializer.
fn serialized_fix_classes() -> BTreeSet<String> {
    FixClass::ALL
        .iter()
        .map(|class| {
            serde_json::to_value(class)
                .expect("a class serializes")
                .as_str()
                .expect("a class serializes to a string")
                .to_owned()
        })
        .collect()
}

/// `Err` when step 3's class list is not the set of names `fix.class` takes.
fn check_fix_class_list(skill_md: &str) -> Result<(), String> {
    let listed = fix_class_names_in(skill_md);
    let serialized = serialized_fix_classes();
    if listed == serialized {
        Ok(())
    } else {
        Err(format!(
            "step 3 of SKILL.md explains the classes {listed:?}; `fix.class` takes \
             {serialized:?}"
        ))
    }
}

/// Step 3 tells an agent what each `fix.class` lets `rocm fix` do. A class the
/// CLI prints that the list leaves out is one the agent has no instruction for,
/// and a spelling the CLI never prints is an instruction that never applies.
/// `every_class_serializes_to_its_kebab_case_name_and_back` in rocm-core pins
/// the serializer; this pins the skill's list to it.
#[test]
fn fix_classes_listed_in_the_skill_are_the_ones_the_cli_serializes() {
    if let Err(problem) = check_fix_class_list(&skill_md_text()) {
        panic!("{problem}");
    }
}

#[test]
fn a_wrong_fix_class_list_is_caught() {
    let flat = flatten_whitespace(&skill_md_text());
    for (what, text) in [
        (
            "the listing's marker instead of the JSON name",
            replace_once(&flat, " - `needs-argument` — ", " - `needs-arg` — "),
        ),
        (
            "a class left out",
            replace_once(&flat, " - `diagnose-only` — ", " "),
        ),
    ] {
        assert!(
            check_fix_class_list(&text).is_err(),
            "{what} in the class list went unnoticed"
        );
    }
    // The last class cut from the list and put back as a bullet below it, with
    // the file ending there so no other later bullet muddies the set: the list has
    // ended at step (a)'s "- (a)" by then, so the bullet is not part of it.
    let without = replace_once(&flat, " - `diagnose-only` — ", " ");
    let below = without
        .find(" - (b) ")
        .expect("step 3's sub-list has a (b) after its (a)");
    let cut_short = format!(
        "{} - `diagnose-only` — a bullet below the list.",
        &without[..below]
    );
    assert!(
        check_fix_class_list(&cut_short).is_err(),
        "a class list cut short was completed by a bullet further down the file"
    );
}

#[test]
#[should_panic(expected = "marker missing")]
fn a_reworded_fix_class_marker_is_loud() {
    fix_class_names_in(
        &flatten_whitespace(&skill_md_text()).replace(FIX_CLASSES_MARKER, "is the entry's class"),
    );
}

/// The lines of this file's module doc, with the `//!` taken off.
fn module_doc() -> String {
    include_str!("skill_reference.rs")
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("//!"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The module doc is where a reader learns which SKILL.md sentences are fixtures
/// and which test reads each. A phrase a test reads that the doc does not quote
/// is a fixture nobody is warned about; one SKILL.md no longer holds fails its
/// own test too, but here it fails by name. Only the phrases listed in
/// `e2e_cucumber::skill_anchors::ALL` are checked.
#[test]
fn every_skill_anchor_is_named_here_and_is_in_the_skill() {
    let doc = flatten_whitespace(&module_doc());
    let skill = flatten_whitespace(&skill_md_text());
    for anchor in e2e_cucumber::skill_anchors::ALL {
        assert!(
            doc.contains(anchor),
            "{anchor:?} is read from SKILL.md but this file's module doc does not quote it"
        );
        assert!(
            skill.contains(anchor),
            "{anchor:?} is no longer in SKILL.md"
        );
    }
}

/// The `fix` fields SKILL.md names: the list between "a `fix` whose" and "are the
/// fields". Reads the whitespace-collapsed text. Panics when either end of the
/// list is missing (rather than reading every identifier to the end of the file
/// as a field), when the list names no field, and on a backticked span that is
/// not a field name (`needs-sudo`, `needsSudo`), which would otherwise be dropped
/// instead of compared.
fn fix_fields_in(skill_md: &str) -> BTreeSet<String> {
    let flat = flatten_whitespace(skill_md);
    let mut fields = BTreeSet::new();
    let after_start = after_marker(&flat, FIX_FIELDS_START);
    let Some(end_at) = after_start.find(FIX_FIELDS_END) else {
        panic!(
            "marker missing: {FIX_FIELDS_END:?} no longer follows {FIX_FIELDS_START:?} in \
             SKILL.md. It is a fixture phrase this test reads; reword it back, or update the \
             test, its constant in `e2e_cucumber::skill_anchors` and the list in this file's \
             module doc"
        );
    };
    for (span, _) in backticked_with_tail(&after_start[..end_at]) {
        assert!(
            is_identifier(span),
            "{span:?} in SKILL.md's `fix` field list is not a field name (lower-case letters, \
             digits and `_` only)"
        );
        fields.insert(span.to_owned());
    }
    assert!(
        !fields.is_empty(),
        "SKILL.md's `fix` field list, between {FIX_FIELDS_START:?} and {FIX_FIELDS_END:?}, names \
         no field"
    );
    fields
}

/// The same check for the `fix` sub-fields, one-way: a field SKILL.md tells the
/// agent to act on has to exist in the diagnosis JSON, but the skill need not
/// name every field the JSON carries. `needs` is set so the one field the
/// serializer omits when absent is present in the sample.
#[test]
fn fix_fields_quoted_in_the_skill_exist_in_the_diagnosis_json() {
    let quoted = fix_fields_in(&skill_md_text());
    let sample = rocm_core::Fix {
        needs: Some("--device-index".to_owned()),
        ..rocm_core::Fix::default()
    };
    let json = serde_json::to_value(&sample).expect("a Fix serializes");
    let emitted: BTreeSet<String> = json
        .as_object()
        .expect("a Fix is a JSON object")
        .keys()
        .cloned()
        .collect();
    let unknown: Vec<&String> = quoted.difference(&emitted).collect();
    assert!(
        unknown.is_empty(),
        "SKILL.md names `fix` fields the diagnosis JSON does not carry: {unknown:?}. \
         Carried: {emitted:?}"
    );
}

/// `text` re-wrapped to lines of at most `width` characters, so every phrase in it
/// may now break across lines in a different place.
fn rewrap(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut line_len = 0;
    for word in text.split_whitespace() {
        if line_len > 0 && line_len + 1 + word.len() > width {
            out.push('\n');
            line_len = 0;
        } else if line_len > 0 {
            out.push(' ');
            line_len += 1;
        }
        out.push_str(word);
        line_len += word.len();
    }
    out
}

/// Re-wrapping SKILL.md must not change what the list parsers read: they match
/// phrases on the whitespace-collapsed text, not one physical line at a time.
#[test]
fn the_skill_parsers_do_not_depend_on_line_wrapping() {
    let skill_md = skill_md_text();
    for width in [37, 61, 200] {
        let wrapped = rewrap(&skill_md, width);
        assert_eq!(
            exit_code_names_in(&wrapped),
            exit_code_names_in(&skill_md),
            "exit-code names changed when SKILL.md was re-wrapped to {width} columns"
        );
        assert_eq!(
            fix_class_names_in(&wrapped),
            fix_class_names_in(&skill_md),
            "fix classes changed when SKILL.md was re-wrapped to {width} columns"
        );
        assert_eq!(
            fix_fields_in(&wrapped),
            fix_fields_in(&skill_md),
            "fix fields changed when SKILL.md was re-wrapped to {width} columns"
        );
    }
}

/// A fixture phrase that is gone must fail loudly, not leave a parser finding
/// nothing (or, for the field list, everything to the end of the file).
#[test]
#[should_panic(expected = "marker missing")]
fn a_reworded_exit_code_marker_is_loud() {
    exit_code_names_in(
        &flatten_whitespace(&skill_md_text()).replace(EXIT_CODES_MARKER, "by its code name"),
    );
}

#[test]
#[should_panic(expected = "marker missing")]
fn a_reworded_fix_field_start_is_loud() {
    fix_fields_in(&flatten_whitespace(&skill_md_text()).replace(FIX_FIELDS_START, "a fix whose "));
}

#[test]
#[should_panic(expected = "marker missing")]
fn a_reworded_fix_field_end_is_loud() {
    fix_fields_in(
        &flatten_whitespace(&skill_md_text()).replace(FIX_FIELDS_END, " are the columns"),
    );
}

#[test]
#[should_panic(expected = "names no field")]
fn an_emptied_fix_field_list_is_loud() {
    fix_fields_in(&format!("{FIX_FIELDS_START}x{FIX_FIELDS_END}"));
}

#[test]
#[should_panic(expected = "is not a field name")]
fn a_hyphenated_fix_field_is_loud() {
    fix_fields_in(&flatten_whitespace(&skill_md_text()).replace("`needs_sudo`", "`needs-sudo`"));
}

#[test]
#[should_panic(expected = "is not a field name")]
fn a_camel_case_fix_field_is_loud() {
    fix_fields_in(&flatten_whitespace(&skill_md_text()).replace("`needs_sudo`", "`needsSudo`"));
}
