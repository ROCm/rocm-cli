// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static seed table resolving the self-hosted E2E `sdk_version` matrix labels
//! (`current` / `n-1` / `n-2`) to concrete TheRock package versions.
//!
//! Source of truth: the three most recent rows of the official release list,
//! <https://rocm.docs.amd.com/en/latest/release/versions.html>, as fetched
//! 2026-10-02 (current=10.0.0, n-1=7.14.1, n-2=7.14.0). Checked in rather than
//! scraped at CI runtime (ROCMAI-430): update by hand when a new version ships.

/// One of the three legs of the self-hosted E2E `sdk_version` matrix.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ReleaseLabel {
    Current,
    #[value(name = "n-1")]
    NMinus1,
    #[value(name = "n-2")]
    NMinus2,
}

impl ReleaseLabel {
    /// The TheRock package version to pin `e2e-prewarm --version` to.
    ///
    /// `Current` resolves to `None`: it means "track the channel's latest",
    /// the pre-warm's existing default when no pin is given, so the common
    /// (non-matrix) path is unaffected by this ever having existed.
    pub const fn pin(self) -> Option<&'static str> {
        match self {
            Self::Current => None,
            Self::NMinus1 => Some("7.14.1"),
            Self::NMinus2 => Some("7.14.0"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReleaseLabel;

    #[test]
    fn current_tracks_latest_others_pin_official_versions() {
        assert_eq!(ReleaseLabel::Current.pin(), None);
        assert_eq!(ReleaseLabel::NMinus1.pin(), Some("7.14.1"));
        assert_eq!(ReleaseLabel::NMinus2.pin(), Some("7.14.0"));
    }
}
