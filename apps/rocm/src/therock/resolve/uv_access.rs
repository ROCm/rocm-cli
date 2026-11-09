// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Who may fetch `uv` while a wheel stack is resolved.
//!
//! On the ROCm 10 layout `uv pip compile` is the one step of resolution that can
//! download the managed `uv` and create the data folder's `tools` folder. A real
//! install and an update ([`UvAccess::Acquire`]) take it as they always did; a
//! preview ([`UvAccess::PresentOnly`]) runs a `uv` that is already there and
//! otherwise skips the step. [`plan_version_resolution`] decides which, as a pure
//! function of the layout, the access and whether a `uv` is present, so that every
//! combination is tested without a network. `wheel.rs` acts on the answer.

use super::super::source::SourceLayout;
use anyhow::{Context, Result};
use std::path::PathBuf;

/// Whether a resolution may fetch `uv` to choose the ROCm 10 package versions.
///
/// Only the ROCm 10 (`next`) layout needs `uv`: it has `uv pip compile` pick the
/// `torch` and `torchvision` builds. That is the one step of resolution that can
/// download the managed `uv` and create the data folder's `tools` folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UvAccess {
    /// A real install or an update: take the managed `uv`, downloading it if no
    /// copy is installed yet. This is what resolution has always done.
    Acquire,
    /// A preview: use a `uv` that is already there and never fetch one.
    PresentOnly,
}

/// What a resolution may do about `uv`: how far it may go ([`UvAccess`]) and the two
/// things it can ask for. The real ones, `find_uv_binary` and `ensure_uv_binary`,
/// are built where a resolution starts; they are fields so that a test can count
/// the calls instead of reaching the network.
pub(super) struct UvPolicy<'a> {
    pub(super) access: UvAccess,
    /// A `uv` that is already there, without acquiring one.
    pub(super) find_present: &'a dyn Fn() -> Option<PathBuf>,
    /// The managed `uv`, downloading it when none is installed.
    pub(super) acquire: &'a dyn Fn() -> Result<PathBuf>,
}

/// How a resolution chooses the package versions, given [`UvAccess`] and whether a
/// `uv` is present. A pure function of those inputs: see [`plan_version_resolution`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum VersionResolution {
    /// The canonical layout: read each package's simple index. No `uv`.
    ScrapeIndexes,
    /// The ROCm 10 layout: run `uv pip compile` with this `uv`.
    Compile(UvChoice),
    /// The ROCm 10 layout with no `uv` to run and none to fetch: stop with what is
    /// known without one, and leave the versions `uv` would choose unresolved.
    Skip,
}

/// Which `uv` the compile step runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum UvChoice {
    /// Ask `ensure_uv_binary`, which downloads the managed `uv` when none is installed.
    Acquire,
    /// A `uv` that was already there.
    Existing(PathBuf),
}

/// Decide how the versions are chosen, without doing it.
///
/// `find_present` looks for a `uv` that is already there. It runs only when the
/// answer depends on it: never for the canonical layout, which needs no `uv`, and
/// never for [`UvAccess::Acquire`], whose `ensure_uv_binary` does its own looking.
/// So a real install and an update decide, and call, exactly what they did before a
/// preview could skip the step, and no answer for them is [`VersionResolution::Skip`].
pub(super) fn plan_version_resolution(
    layout: SourceLayout,
    access: UvAccess,
    find_present: impl FnOnce() -> Option<PathBuf>,
) -> VersionResolution {
    match (layout, access) {
        (SourceLayout::Canonical, _) => VersionResolution::ScrapeIndexes,
        (SourceLayout::Next, UvAccess::Acquire) => VersionResolution::Compile(UvChoice::Acquire),
        (SourceLayout::Next, UvAccess::PresentOnly) => find_present()
            .map_or(VersionResolution::Skip, |uv| {
                VersionResolution::Compile(UvChoice::Existing(uv))
            }),
    }
}

/// The `uv` the compile step runs: the one already there, or the managed `uv` that
/// `acquire` (`ensure_uv_binary`) fetches when none is installed.
pub(super) fn uv_for_compile(
    choice: UvChoice,
    acquire: impl FnOnce() -> Result<PathBuf>,
) -> Result<PathBuf> {
    match choice {
        UvChoice::Existing(uv) => Ok(uv),
        UvChoice::Acquire => {
            acquire().context("failed to acquire uv for ROCm X metadata resolution")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::bail;

    /// Asks [`plan_version_resolution`] for one cell of the matrix and reports
    /// the answer together with how many times a present `uv` was looked for.
    fn decide(
        layout: SourceLayout,
        access: UvAccess,
        uv_present: bool,
    ) -> (VersionResolution, usize) {
        let looked = std::cell::Cell::new(0);
        let decision = plan_version_resolution(layout, access, || {
            looked.set(looked.get() + 1);
            uv_present.then(|| PathBuf::from("/opt/uv"))
        });
        (decision, looked.get())
    }

    /// The state matrix of the decision: both layouts, both accesses, a `uv`
    /// present or not. Only the ROCm 10 layout needs `uv`; a real install and an
    /// update (`Acquire`) always go on to acquire it and never ask whether one is
    /// present, because `ensure_uv_binary` does its own looking; a preview
    /// (`PresentOnly`) asks, and with none present stops. `Skip` therefore exists
    /// for exactly one row.
    #[test]
    fn the_version_step_is_decided_by_layout_access_and_a_present_uv_alone() {
        use SourceLayout::{Canonical, Next};
        use UvAccess::{Acquire, PresentOnly};
        let existing = || VersionResolution::Compile(UvChoice::Existing(PathBuf::from("/opt/uv")));
        let rows = [
            // (layout, access, uv present, decision, times a present uv was looked for)
            (
                Canonical,
                Acquire,
                false,
                VersionResolution::ScrapeIndexes,
                0,
            ),
            (
                Canonical,
                Acquire,
                true,
                VersionResolution::ScrapeIndexes,
                0,
            ),
            (
                Canonical,
                PresentOnly,
                false,
                VersionResolution::ScrapeIndexes,
                0,
            ),
            (
                Canonical,
                PresentOnly,
                true,
                VersionResolution::ScrapeIndexes,
                0,
            ),
            (
                Next,
                Acquire,
                false,
                VersionResolution::Compile(UvChoice::Acquire),
                0,
            ),
            (
                Next,
                Acquire,
                true,
                VersionResolution::Compile(UvChoice::Acquire),
                0,
            ),
            (Next, PresentOnly, false, VersionResolution::Skip, 1),
            (Next, PresentOnly, true, existing(), 1),
        ];
        for (layout, access, uv_present, expected, looked) in rows {
            let (decision, calls) = decide(layout, access, uv_present);
            let row = format!("{layout:?}, {access:?}, uv present: {uv_present}");
            assert_eq!(decision, expected, "decision for {row}");
            assert_eq!(calls, looked, "times a present uv was looked for in {row}");
        }
    }

    /// The compile step runs the `uv` it was given, and asks for one only on
    /// `Acquire`, with the context a failed fetch has always carried.
    #[test]
    fn the_compile_step_acquires_uv_only_when_told_to() {
        let acquired = std::cell::Cell::new(0);
        let acquire = || {
            acquired.set(acquired.get() + 1);
            Ok(PathBuf::from("/managed/uv"))
        };
        assert_eq!(
            uv_for_compile(UvChoice::Acquire, acquire).unwrap(),
            PathBuf::from("/managed/uv")
        );
        assert_eq!(acquired.get(), 1);
        assert_eq!(
            uv_for_compile(UvChoice::Existing(PathBuf::from("/opt/uv")), acquire).unwrap(),
            PathBuf::from("/opt/uv")
        );
        assert_eq!(acquired.get(), 1, "a present uv is not acquired again");

        let error = uv_for_compile(UvChoice::Acquire, || bail!("no network")).unwrap_err();
        assert_eq!(
            format!("{error:#}"),
            "failed to acquire uv for ROCm X metadata resolution: no network"
        );
    }
}
