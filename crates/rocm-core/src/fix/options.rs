// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The knobs and vocabulary of `rocm fix`.
//!
//! [`exit`] names the codes `apply` returns, [`FixOptions`] carries what the
//! caller asked for (`--yes`, `--dry-run`, `--device-index`), and [`FixClass`]
//! says what running a catalog entry will do to the machine. The catalog and
//! its runners live in the sibling modules of `fix.rs`, which re-exports the
//! public names from here.

use serde::{Deserialize, Serialize};

/// The exit codes `apply` returns, named so the published manifest and the code
/// that produces them cannot come to disagree.
///
/// These were written down only in the module comment above. A consumer reading
/// the manifest is being told what a number means, so the number and its meaning
/// have to come from one place -- a manifest that restated them by hand would be
/// the same defect this whole contract exists to remove, one level up.
pub mod exit {
    // What each code means is published in [`MEANINGS`] and said only there;
    // a second wording here would be a copy that can drift from the one a
    // caller reads.

    /// `ok` in [`MEANINGS`].
    pub const OK: i32 = 0;
    /// `internal` in [`MEANINGS`].
    pub const INTERNAL: i32 = 1;
    /// `unknown_id` in [`MEANINGS`].
    pub const UNKNOWN_ID: i32 = 2;
    /// `not_applicable` in [`MEANINGS`].
    pub const NOT_APPLICABLE: i32 = 3;
    /// `failed` in [`MEANINGS`].
    pub const FAILED: i32 = 4;
    /// `declined` in [`MEANINGS`].
    pub const DECLINED: i32 = 5;

    /// How the stderr line refusing an unknown fix-id starts.
    ///
    /// Published inside the [`UNKNOWN_ID`] description, because it is the only
    /// way a caller can tell that refusal from a usage error, which exits with
    /// the same code (see [`MEANINGS`]).
    pub const UNKNOWN_ID_PREFIX: &str = "Unknown fix-id:";

    /// One exit code as the manifest publishes it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Meaning {
        /// The key a reader looks the code up by.
        pub name: &'static str,
        pub code: i32,
        /// What a caller should conclude on receiving the code.
        pub description: &'static str,
    }

    /// Every code `rocm fix` returns, with its name and what it means.
    ///
    /// The single table both published maps (`exit_codes`,
    /// `exit_code_descriptions`) are built from, so a code cannot gain a name
    /// without a meaning or move without its description moving with it.
    /// `every_exit_constant_is_published_and_nothing_else_is` (in
    /// `manifest.rs`) holds this table to the constants above, so a new
    /// constant fails until it is described here.
    ///
    /// Each description is a claim about what the CLI did, so the comment above
    /// each entry names the tests that prove it (AGENTS.md section 3), and the
    /// branches no test drives.
    ///
    /// Exit 2 is shared on purpose and documented rather than split: clap
    /// returns 2 for a malformed command line, and giving a usage error a code
    /// of its own would change what existing readers of exit 2 receive.
    pub const MEANINGS: &[Meaning] = &[
        // Proved in `fix.rs` by
        // `a_problem_with_no_reliable_fix_reports_success_and_does_not_look_portable`
        // (diagnose-only) and `print_only_fix_returns_zero` (printed steps); in
        // `fix/runners.rs` by `granting_consent_appends_the_export_and_returns_0`
        // (applied), `an_rc_file_that_already_pins_a_device_is_left_alone`
        // (nothing to change), `dry_run_returns_0_and_writes_nothing_even_with_consent`
        // (preview) and `windows_path_read_failure_aborts_before_plan_or_write`
        // (a preview that exits 4 instead); listing the catalog by diagnose-34
        // (`@id:diagnose-fix-catalog-is-machine-readable`).
        Meaning {
            name: "ok",
            code: OK,
            description: "Nothing failed: the fix was applied, found nothing to change \
                          (already in place, or a setting it leaves for you to edit), was \
                          previewed with `--dry-run`, or only printed what to do (steps to \
                          run yourself, or the explanation of a diagnose-only entry); or \
                          the catalog was listed. A `--dry-run` preview is not always 0: \
                          it exits 3 or 4 when the fix cannot be planned.",
        },
        // Proved in `fix.rs` by
        // `a_recipe_the_catalog_says_is_applied_with_no_runner_exits_internal`, and
        // for `--json` with a fix-id by diagnose-35
        // (`@id:diagnose-fix-catalog-json-rejects-a-fix-id`).
        Meaning {
            name: "internal",
            code: INTERNAL,
            description: "The CLI is inconsistent with itself (a recipe it says it can carry \
                          out, with nothing to carry it out with); never a fault of the \
                          machine. Exit 1 is also any other error the command reports on \
                          stderr, such as `--json` combined with a fix-id.",
        },
        // Proved by `the_unknown_id_description_names_the_usage_error_it_shares_a_code_with`
        // (`manifest.rs`), `a_usage_error_exits_with_the_code_published_for_an_unknown_fix_id`
        // (`apps/rocm/src/main.rs`) and diagnose-34, which runs the binary.
        Meaning {
            name: "unknown_id",
            code: UNKNOWN_ID,
            description: "No catalog entry has that fix-id; no fix step was run, and a \
                          stderr line starts with `Unknown fix-id:`. Exit 2 is also what \
                          the argument parser returns for a malformed command line (an \
                          unknown option, a missing value, or a value it cannot read, such \
                          as a non-number for `--device-index`, or a negative one written \
                          after a space, `--device-index -1`, which it reads as an \
                          unknown option), before any fix-id is looked up; that refusal has a stderr line starting with `error:` \
                          and never says `Unknown fix-id:`. Another line, such as a \
                          one-time `rocm:` notice, can come first. The code alone cannot \
                          tell the two apart; the message can.",
        },
        // Proved in `fix.rs` by `the_class_gate_is_reached_only_after_the_platform_gate`
        // and `the_published_exit_codes_are_the_ones_apply_returns` (another
        // platform), and in `fix/runners.rs` by
        // `a_negative_device_index_is_not_applicable_and_nothing_runs` (an argument
        // it cannot use); that only `--device-index=-1` reaches the runner, and
        // `--device-index -1` is a usage error, is pinned through the real parser
        // by `a_negative_device_index_reaches_the_fix_only_when_written_with_equals`
        // in `apps/rocm/src/main.rs`. The missing tool, install and home-directory
        // branches are not driven by a test. A missing home directory returns 3
        // only from the fixes that write a file there (on Linux, fix-6, and fix-9
        // when given `--device-index`; without one fix-9 only prints the query and
        // exits 0 before it looks); fix-2 on Linux only reports, and exits 0
        // without one.
        Meaning {
            name: "not_applicable",
            code: NOT_APPLICABLE,
            description: "The entry exists but cannot be used here: it is for another \
                          platform, the host lacks something the fix cannot do without (a \
                          tool it runs, or `sudo` when not root; an install it points at; \
                          the current user's name; or a home directory to write a file \
                          in), or an argument it was given is one it cannot \
                          use, such as a negative device index written `--device-index=-1`. \
                          Nothing was tried. Written with a space instead, `--device-index -1` \
                          is a usage error and exits 2 (see `unknown_id`).",
        },
        // Proved in `fix/runners.rs` by
        // `a_failed_write_returns_4_rather_than_reporting_success` (a file write)
        // and `windows_path_read_failure_aborts_before_plan_or_write` (a failed
        // read, under `--dry-run`). A command that exits non-zero (`usermod`,
        // `setx`, PowerShell) is not driven by a test: those runners start the
        // real program.
        Meaning {
            name: "failed",
            code: FAILED,
            description: "A step of the fix failed: a command it ran, a file it wrote, or a \
                          read of state it needed before planning (the current User PATH \
                          on Windows, read even under `--dry-run`). The change that step \
                          was to make was not made.",
        },
        // Proved in `fix/runners.rs` by `declining_consent_returns_5_and_writes_nothing`,
        // `windows_decline_on_user_scope_returns_5` and
        // `windows_decline_with_machine_scope_also_set_still_returns_5`, and for a
        // stdin that is not a terminal by
        // `non_interactive_shell_refuses_instead_of_proceeding` in `fix/consent.rs`.
        Meaning {
            name: "declined",
            code: DECLINED,
            description: "The fix needed confirmation and did not get it: the user answered \
                          no, or stdin is not a terminal and `--yes` was not given. Nothing \
                          the confirmation guarded was changed.",
        },
    ];
}

/// Options controlling how a fix is applied.
#[derive(Debug, Clone, Default)]
pub struct FixOptions {
    /// Skip the interactive confirmation (the user already approved the plan).
    pub yes: bool,
    /// Show the plan without changing anything.
    pub dry_run: bool,
    /// For `fix-9-igpu-dgpu`: the discrete GPU index to pin.
    pub device_index: Option<i64>,
}

/// What `rocm fix <id>` will do with an entry — the answer to "will running
/// this plainly change my machine?".
///
/// This was a `bool` (`auto_applicable`), and a bool could not say two things
/// the catalog needed to say. `fix-2-unset-override` applies itself on Windows
/// but only reports on Linux, where its runner takes no [`FixOptions`] and
/// never mutates. `fix-9-igpu-dgpu` applies itself only once it is told which
/// device to pin, and merely prints the identifying query otherwise. Both were
/// marked auto-applicable, so both told a user — and an agent — that the CLI
/// was about to act when it was not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FixClass {
    /// `rocm fix <id>` carries it out (asking first, unless `--yes`).
    Auto,
    /// `rocm fix <id>` carries it out only once the argument named in
    /// [`Platform::needs`] is supplied; without it, it reports what to pass and
    /// changes nothing.
    NeedsArgument,
    /// `rocm fix <id>` leaves the machine alone. The fix is real; carrying it
    /// out needs sudo, a reboot, a reinstall or a download.
    ///
    /// This says nothing about whether a runner runs. `fix-2-unset-override`
    /// is print-only on Linux and still reads the live override and scans the
    /// shell rc files to report where it is set — it just never writes.
    #[default]
    PrintOnly,
    /// The problem can be detected and explained, but no reliable fix exists.
    /// `rocm fix <id>` names it and applies nothing.
    ///
    /// Defaulting to [`FixClass::PrintOnly`] rather than this one is
    /// deliberate: a missing class should understate what the CLI will do, and
    /// "there is no fix" is a stronger claim than "here are the steps".
    DiagnoseOnly,
}

impl FixClass {
    /// Whether `rocm fix <id>`, invoked with no extra arguments, will change
    /// the machine. This is the claim the old `auto_applicable` bool was making.
    #[must_use]
    pub const fn applies_itself(self) -> bool {
        matches!(self, Self::Auto)
    }

    /// The marker shown in the catalog listing and in `diagnose` output.
    #[must_use]
    pub const fn marker(self) -> &'static str {
        match self {
            Self::Auto => "AUTO",
            Self::NeedsArgument => "NEEDS-ARG",
            Self::PrintOnly => "PRINT-ONLY",
            Self::DiagnoseOnly => "DIAGNOSE-ONLY",
        }
    }
}
