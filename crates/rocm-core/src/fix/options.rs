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
    /// Applied, previewed with `--dry-run`, printed, or listed. Nothing failed.
    pub const OK: i32 = 0;
    /// The CLI is inconsistent with itself -- a recipe it says it can carry out,
    /// with nothing to carry it out with. Never a fault of the machine.
    pub const INTERNAL: i32 = 1;
    /// No such fix-id.
    pub const UNKNOWN_ID: i32 = 2;
    /// Real, but not for this machine: wrong platform, or a precondition the
    /// host does not meet. Distinct from [`FAILED`], which means it was tried.
    pub const NOT_APPLICABLE: i32 = 3;
    /// Attempted, and a command it ran failed.
    pub const FAILED: i32 = 4;
    /// The user was asked and said no.
    pub const DECLINED: i32 = 5;
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
