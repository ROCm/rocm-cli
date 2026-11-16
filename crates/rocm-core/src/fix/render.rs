// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Rendering of the catalog for people: the listing `rocm fix` prints with no
//! fix-id, a recipe's plan, its verify lines, the flag summary `rocm diagnose`
//! shares, and the refusal printed when a diagnosis position (`#2`) is passed
//! as a fix-id.

use super::catalog::{FixRecipe, RECIPES};
use super::consent::current_os;
use super::options::FixClass;

/// Whether `value` is a `rocm diagnose` ranking position (`#2`, or a bare `2`)
/// rather than a fix-id.
///
/// Used only to turn an unknown-id refusal into a corrective one; it never
/// selects a fix, because a position is meaningful only within the report that
/// produced it and `fix` has no memory of that report.
pub(super) fn looks_like_a_diagnosis_position(value: &str) -> bool {
    let trimmed = value.trim();
    let digits = trimmed.strip_prefix('#').unwrap_or(trimmed);
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

/// List every fix-id (id, class on this machine, OS scope, title).
///
/// The marker is what the entry does **here**, not everywhere: `fix-2` reports
/// on Linux and applies itself on Windows, and a listing that averaged the two
/// would be wrong on both. The question a reader is asking is what happens if
/// they run it on the machine in front of them.
#[must_use]
pub fn list_recipes() -> String {
    use std::fmt::Write as _;
    let mut out = String::from("Available fix-ids (mirror the diagnosis catalog):\n");
    // The markers were printed with nothing saying what they mean.
    out.push_str("  On this machine: AUTO = `rocm fix <id>` carries it out; NEEDS-ARG = it does once you supply an argument;\n");
    out.push_str("  PRINT-ONLY = it prints the steps for you to run; DIAGNOSE-ONLY = it explains the problem and applies nothing.\n");
    for r in RECIPES {
        let kind = r.class_here().marker();
        let scope = r.os_scope().join("/");
        let _ = writeln!(
            out,
            "  [{kind:>13}] [{scope:>14}] {}  -- {}",
            r.fix_id, r.title
        );
    }
    out
}

/// Canonical wording for a fix's remediation flags, shared by `rocm fix <id>`
/// and `rocm diagnose` so the same `(sudo, reboot, relogin, class)` values
/// render as the same text from either command. This only standardizes wording,
/// not the underlying values: `FixRecipe` (fix/catalog.rs) and diagnose's `Fix` still
/// supply those independently, so a fix-id's rendered flags can still differ if
/// the two disagree on a value; `assert_needs_reboot_matches_the_catalog` and
/// `assert_plan_matches_the_catalog_copy` are targeted regression tests that
/// pin specific fix-ids against that drift, not a blanket guarantee for every
/// fix-id. Also out of scope: the bare `rocm fix` catalog listing
/// (`list_recipes`), which describes the same property with the shorter
/// AUTO/NEEDS-ARG/PRINT-ONLY/DIAGNOSE-ONLY markers.
///
/// `class` replaced a bool. The bool could say "the CLI will run this" or "it
/// will not", and three of the four answers here are "not, but for different
/// reasons" -- which is the distinction this wording exists to carry. `needs`
/// names the argument a [`FixClass::NeedsArgument`] entry is waiting for, and is
/// ignored for every other class.
///
/// Returns owned strings because that argument name is not a literal.
// These mirror the `FixRecipe`/`Fix` struct fields, where
// `clippy::struct_excessive_bools` is already allowed workspace-wide; that
// allow doesn't reach this free function's parameters, so
// `clippy::fn_params_excessive_bools` is separately allowed below.
#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn format_flags(
    needs_sudo: bool,
    needs_reboot: bool,
    needs_relogin: bool,
    class: FixClass,
    needs: Option<&str>,
) -> Vec<String> {
    let mut flags = Vec::new();
    if needs_sudo {
        flags.push("requires sudo".to_owned());
    }
    if needs_reboot {
        flags.push("requires reboot".to_owned());
    }
    if needs_relogin {
        flags.push("requires re-login".to_owned());
    }
    flags.push(match class {
        FixClass::Auto => "rocm fix can run it".to_owned(),
        // Named rather than described as "manual only": the CLI *will* act, just
        // not until it is told what to act on. Calling that manual sends the
        // user off to run the steps by hand for no reason.
        FixClass::NeedsArgument => format!(
            "needs {} before `rocm fix` will run it",
            needs.unwrap_or("an argument")
        ),
        FixClass::PrintOnly => "manual only (`rocm fix` will NOT run it automatically)".to_owned(),
        FixClass::DiagnoseOnly => {
            "no reliable fix (`rocm fix` will NOT change anything)".to_owned()
        }
    });
    flags
}

pub(super) fn print_recipe(r: &FixRecipe) {
    println!("Fix:        {}  -- {}", r.fix_id, r.title);
    println!("OS scope:   {}", r.os_scope().join(", "));
    println!("Rationale:  {}", r.rationale);
    if !r.commands.is_empty() {
        println!("Commands:");
        for c in r.commands {
            println!("  $ {c}");
        }
    }
    let flags = format_flags(
        r.needs_sudo,
        r.needs_reboot,
        r.needs_relogin,
        r.class_here(),
        r.platform(current_os()).and_then(|p| p.needs),
    );
    println!("Flags:      {}", flags.join(", "));
    for n in r.notes {
        println!("Note:       {n}");
    }
    for line in verify_lines(r) {
        println!("{line}");
    }
}

/// The `Verify` lines of the plan `print_recipe` prints, in print order: the
/// line every OS can run, then the Linux and WSL line when the recipe has one.
/// Split out so the labels are testable, as `format_flags` is. A multi-line
/// line keeps its continuation under the label.
pub(super) fn verify_lines(r: &FixRecipe) -> Vec<String> {
    let mut lines = Vec::new();
    if !r.verify.is_empty() {
        lines.push(format!(
            "Verify:     {}",
            indent_continuation(r.verify, "            ")
        ));
    }
    if let Some(linux) = r.verify_linux {
        lines.push(format!(
            "Verify (Linux, WSL): {}",
            indent_continuation(linux, "            ")
        ));
    }
    lines
}

/// The commands a print-only plan tells the user to run to verify, as printed.
pub(super) fn verify_steps(r: &FixRecipe) -> Vec<String> {
    let mut steps = Vec::new();
    if !r.verify.is_empty() {
        steps.push(format!("  $ {}", indent_continuation(r.verify, "    ")));
    }
    if let Some(linux) = r.verify_linux {
        steps.push(format!(
            "  $ {}   # Linux, WSL",
            indent_continuation(linux, "    ")
        ));
    }
    steps
}

/// Indent every line after the first so a multi-line `verify` (fix-7 lists one
/// command per package manager) stays under its `Verify:` / `$` label.
fn indent_continuation(text: &str, indent: &str) -> String {
    text.replace('\n', &format!("\n{indent}"))
}

/// What `rocm fix` says when it is handed a `rocm diagnose` ranking position
/// (`#1`) rather than a fix-id.
///
/// Lifted out of [`apply`] so the wording is reachable from a test: `fail!`
/// writes straight to the process's own stderr, which an in-process test cannot
/// capture, so the only other way to cover these two lines would be to spawn
/// the binary.
pub(super) fn diagnosis_position_refusal(fix_id: &str) -> [String; 2] {
    [
        format!("`{fix_id}` looks like a position in a `rocm diagnose` report, not a fix-id."),
        "Use the `id:` shown against that cause - `rocm diagnose` prints an `apply with:` line you can copy.".to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fix::catalog::{find_recipe, needs_on};

    /// fix-6's plan carries one verify line that runs on Windows and one for
    /// Linux and WSL that shows the GPU. The plain line is what Windows runs, so
    /// it cannot need `rocminfo`; the Linux line must not need `hipcc`, which a
    /// runtime-only install lacks. The note says exactly that, so the two
    /// claims in it are asserted against the lines it names.
    #[test]
    fn the_path_fix_verifies_the_gpu_on_linux_without_needing_hipcc() {
        let recipe = find_recipe("fix-6-path").expect("fix-6 is in the catalog");
        let linux = recipe.verify_linux.expect("fix-6 has a Linux verify line");
        assert!(
            linux.starts_with("rocminfo") && !linux.contains("hipcc"),
            "the Linux line must show the GPU and not need hipcc: {linux}"
        );
        assert!(
            !recipe.verify.contains("rocminfo"),
            "the plain line is shown to Windows users, where rocminfo does not exist: {}",
            recipe.verify
        );
        assert_eq!(
            verify_lines(recipe),
            [
                "Verify:     hipcc --version",
                "Verify (Linux, WSL): rocminfo | head -n 5"
            ]
        );
        let note = recipe
            .notes
            .iter()
            .find(|n| n.contains("runtime-only"))
            .expect("a note names the runtime-only install");
        assert!(
            note.contains("hipcc") && note.contains("Verify (Linux, WSL)"),
            "{note}"
        );
    }

    /// fix-2's plan carries the line every OS can run (`python`, as cmd and
    /// PowerShell run it) and a Linux and WSL line that does not need Python,
    /// because a stock Ubuntu has `python3` only. The note names both lines, so
    /// the claims in it are asserted against what is printed: the Linux line
    /// really needs no `python` and no external command, and it reports the
    /// same answer as the plain line (the override's value, or OK_UNSET).
    // The `${...}` in the expected line is shell text, not a format argument.
    #[allow(clippy::literal_string_with_formatting_args)]
    #[test]
    fn the_override_fix_has_a_linux_verify_line_that_needs_no_python() {
        let recipe = find_recipe("fix-2-unset-override").expect("fix-2 is in the catalog");
        let linux = recipe.verify_linux.expect("fix-2 has a Linux verify line");
        assert_eq!(
            verify_lines(recipe),
            [
                format!("Verify:     {}", recipe.verify),
                "Verify (Linux, WSL): echo \"${HSA_OVERRIDE_GFX_VERSION:-OK_UNSET}\"".to_owned()
            ]
        );
        assert!(
            recipe.verify.starts_with("python -c "),
            "the plain line is the one that runs in cmd and PowerShell: {}",
            recipe.verify
        );
        assert!(!linux.contains("python"), "{linux}");
        let note = recipe
            .notes
            .iter()
            .find(|n| n.contains("python3"))
            .expect("a note names the stock-Ubuntu python3 case");
        assert!(
            note.contains("Verify (Linux, WSL)") && note.contains("`python`"),
            "{note}"
        );
    }

    #[test]
    fn list_includes_all_ids() {
        let listing = list_recipes();
        for r in RECIPES {
            assert!(listing.contains(r.fix_id), "listing missing {}", r.fix_id);
        }
    }

    #[test]
    fn listing_explains_its_applicability_markers() {
        // The markers were emitted with nothing saying what they mean, leaving
        // the reader to guess whether PRINT-ONLY was a debug flag.
        let listing = list_recipes();
        assert!(
            listing.contains("AUTO =") && listing.contains("PRINT-ONLY ="),
            "the listing must explain its markers:\n{listing}"
        );
    }

    #[test]
    fn diagnosis_positions_are_recognised_as_positions() {
        // What `rocm diagnose` shows as `#1`/`#2`, plus the bare number a user
        // might type instead.
        for value in ["#1", "#2", "1", "12", " #3 "] {
            assert!(
                looks_like_a_diagnosis_position(value),
                "{value} should read as a ranking position"
            );
        }
        // Real ids and other typos must keep the generic refusal — claiming a
        // fix-id is a "position" would be worse than saying nothing.
        for value in [
            "fix-4-render-group",
            "fix-1-arch",
            "bogus",
            "#",
            "",
            "fix-#1",
        ] {
            assert!(
                !looks_like_a_diagnosis_position(value),
                "{value} should NOT read as a ranking position"
            );
        }
    }

    #[test]
    fn format_flags_covers_every_flag_combination_and_every_class() {
        // Exhaustive over all 2^3 combinations of the optional flags
        // (sudo/reboot/relogin) x all four classes, so a wording regression on
        // any one flag, or on the always-present class marker, fails here
        // rather than only being visible by eyeballing `rocm fix`/`rocm
        // diagnose` output. The class replaced a bool: three of the four
        // answers are "the CLI will not run it", for different reasons, and a
        // bool could only say one of them.
        for bits in 0..8u8 {
            let needs_sudo = bits & 1 != 0;
            let needs_reboot = bits & 2 != 0;
            let needs_relogin = bits & 4 != 0;
            for class in [
                FixClass::Auto,
                FixClass::NeedsArgument,
                FixClass::PrintOnly,
                FixClass::DiagnoseOnly,
            ] {
                let mut expected: Vec<String> = Vec::new();
                if needs_sudo {
                    expected.push("requires sudo".to_owned());
                }
                if needs_reboot {
                    expected.push("requires reboot".to_owned());
                }
                if needs_relogin {
                    expected.push("requires re-login".to_owned());
                }
                expected.push(
                    match class {
                        FixClass::Auto => "rocm fix can run it",
                        FixClass::NeedsArgument => {
                            "needs --device-index before `rocm fix` will run it"
                        }
                        FixClass::PrintOnly => {
                            "manual only (`rocm fix` will NOT run it automatically)"
                        }
                        FixClass::DiagnoseOnly => {
                            "no reliable fix (`rocm fix` will NOT change anything)"
                        }
                    }
                    .to_owned(),
                );

                let flags = format_flags(
                    needs_sudo,
                    needs_reboot,
                    needs_relogin,
                    class,
                    Some("--device-index"),
                );
                assert_eq!(
                    flags, expected,
                    "sudo={needs_sudo} reboot={needs_reboot} relogin={needs_relogin} class={class:?}"
                );
            }
        }
    }

    /// The four classes never render the same sentence.
    ///
    /// The bool this replaced could say two things, and the reason for keeping
    /// four is that a user acts differently on each: wait for the CLI, supply an
    /// argument, run the steps by hand, or stop looking for a fix. Two classes
    /// that read alike would put us back where we started with a different type.
    #[test]
    fn every_class_says_something_different_about_what_will_happen() {
        let rendered: Vec<String> = [
            FixClass::Auto,
            FixClass::NeedsArgument,
            FixClass::PrintOnly,
            FixClass::DiagnoseOnly,
        ]
        .into_iter()
        .map(|c| {
            format_flags(false, false, false, c, Some("--device-index"))
                .pop()
                .expect("format_flags always ends with the class marker")
        })
        .collect();
        let mut unique = rendered.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            rendered.len(),
            "two classes render the same flag text, so a reader cannot tell them apart: {rendered:?}"
        );
    }

    /// `rocm fix` and `rocm diagnose` name the same argument for the same entry.
    ///
    /// They render the flags line from two call sites, which is why
    /// `format_flags` is shared at all. Passing `None` from one of them would
    /// degrade its text to "an argument" while the other named the flag, and
    /// nothing else in the suite compares the two.
    #[test]
    fn a_needs_argument_entry_names_the_argument_rather_than_describing_it() {
        let named = format_flags(
            false,
            false,
            false,
            FixClass::NeedsArgument,
            needs_on("fix-9-igpu-dgpu", "linux"),
        );
        assert!(
            named.iter().any(|f| f.contains("--device-index")),
            "the catalog knows which argument fix-9-igpu-dgpu waits for, so the flags line \
             has to say it: {named:?}"
        );
    }

    #[test]
    fn indent_continuation_keeps_multi_line_verify_under_its_label() {
        assert_eq!(indent_continuation("a\nb\nc", "  "), "a\n  b\n  c");
        assert_eq!(indent_continuation("single", "  "), "single");
    }
}
