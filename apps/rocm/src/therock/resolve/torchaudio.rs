// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which torchaudio build a ROCm 10 or newer install pins, and why. The wheel
//! indexes publish torchaudio rebuilds under one `+rocm<version>` label that
//! nothing in their metadata tells apart, so `wheel` asks `torchaudio_pin` for
//! the build instead of leaving it to `uv`: the hardware-verified release when
//! the index publishes it, the newest build for that ROCm build otherwise. The
//! `TorchaudioPin` it returns carries the reason, and the plan line and the
//! install-log note are both derived from it. Depends on `wheel` for the PEP 440
//! ordering of package versions.

use super::wheel::{package_release_key, package_rocm_suffix, package_versions_matching_rocm};
use anyhow::{Context, Result, bail};

/// The torchaudio build pinned for a ROCm 10 install, and why it was chosen.
/// The plan line and install-log note are both [`TorchaudioPin::describe`], so
/// what is printed is derived from the decision rather than restated beside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::therock) struct TorchaudioPin {
    /// The published version pinned, `+rocm…` label included.
    pub(super) version: String,
    /// The ROCm build `version` was chosen for.
    rocm_version: String,
    basis: TorchaudioPinBasis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TorchaudioPinBasis {
    /// `version` is the hardware-verified release for this ROCm line.
    Verified,
    /// A verified release is recorded for this ROCm line, but the index
    /// publishes no build of it labelled for this exact ROCm build (`+rocm`
    /// label) for this Python and platform, so `version` is the newest build
    /// instead.
    VerifiedNotPublished {
        release: &'static str,
        /// A build of `release` this Python and platform can install, labelled
        /// for another ROCm build, when one is published. The verified release
        /// is matched by ROCm line, so a `10.1.1` request can find it published
        /// only as `+rocm10.1.0`; the reason must then name the ROCm build, not
        /// the Python or platform.
        other_build: Option<String>,
    },
    /// No verified release is recorded for this ROCm line, so `version` is the
    /// newest build.
    NoVerifiedRelease,
}

impl TorchaudioPin {
    const fn is_verified(&self) -> bool {
        matches!(self.basis, TorchaudioPinBasis::Verified)
    }

    pub(in crate::therock) fn describe(&self) -> String {
        let Self {
            version,
            rocm_version,
            basis,
        } = self;
        match basis {
            TorchaudioPinBasis::Verified => format!(
                "torchaudio {version}, the build verified on hardware for ROCm {rocm_version}"
            ),
            TorchaudioPinBasis::VerifiedNotPublished {
                release,
                other_build: None,
            } => format!(
                "torchaudio {version}, the newest build published for ROCm {rocm_version}; it is not hardware-verified, because the verified release, {release}, is not published for this Python and platform"
            ),
            TorchaudioPinBasis::VerifiedNotPublished {
                release,
                other_build: Some(other_build),
            } => format!(
                "torchaudio {version}, the newest build published for ROCm {rocm_version}; it is not hardware-verified, because the verified release, {release}, is published for this Python and platform only for another ROCm build ({other_build}), not for ROCm {rocm_version}"
            ),
            TorchaudioPinBasis::NoVerifiedRelease => format!(
                "torchaudio {version}, the newest build published for ROCm {rocm_version}; it is not hardware-verified, because no torchaudio release is recorded as verified for this ROCm line"
            ),
        }
    }
}

/// The install-log line for a torchaudio pin that is not hardware-verified, or
/// `None` for one that is.
///
/// An unverified torchaudio is the one stack choice ROCm CLI makes on its own
/// judgement, so a real install says so in the install log, not only in the
/// plan's `torchaudio_selection:` line. A verified pin needs no such warning.
/// The text is [`TorchaudioPin::describe`], so it cannot disagree with the pin.
pub(in crate::therock) fn unverified_torchaudio_note(pin: &TorchaudioPin) -> Option<String> {
    (!pin.is_verified()).then(|| format!("Note: {}", pin.describe()))
}

/// The torchaudio build to pin for `rocm_version`, from `published` (this
/// Python and platform's builds on the index).
///
/// `verified_release` comes first: it is the release confirmed on hardware for
/// this ROCm line ([`rocm_engine_vllm::verified_torchaudio_release`], read from
/// the vLLM ROCm 10.x discover-build table so the SDK and its vLLM install pin
/// the same build). Recency alone is not a safe choice among builds sharing a
/// `+rocm<version>` label: `whl-next` publishes `2.11.0`, `2.11.0.2` and
/// `2.11.0.3` all as `+rocm10.1.0`, and `2.11.0` is a CPU-only rebuild with no
/// HIP kernels, which nothing in the version or label distinguishes. Only when
/// no verified release is recorded, or it is not published here, does this fall
/// back to [`latest_build_for_rocm`], and the returned basis says so.
///
/// Refuses when no build for `rocm_version` is published at all, or when none of
/// the published builds' versions can be read — the second is not the first,
/// and reporting it as "none published" would send the user to the wrong place.
pub(super) fn torchaudio_pin(
    published: &[String],
    rocm_version: &str,
    verified_release: Option<&'static str>,
    index_url: &str,
) -> Result<TorchaudioPin> {
    let built_for_rocm = package_versions_matching_rocm(published, rocm_version);
    if built_for_rocm.is_empty() {
        bail!(
            "no torchaudio built for ROCm {rocm_version} is published for this Python and platform in {index_url}"
        );
    }
    let pin = |version, basis| TorchaudioPin {
        version,
        rocm_version: rocm_version.to_owned(),
        basis,
    };
    let basis = match verified_release {
        Some(release) => match published_build_of(&built_for_rocm, release) {
            Some(version) => return Ok(pin(version, TorchaudioPinBasis::Verified)),
            // Not among `built_for_rocm`, so a labelled build of it in the
            // whole listing is one for another ROCm build.
            None => TorchaudioPinBasis::VerifiedNotPublished {
                release,
                other_build: published_build_of(
                    &published
                        .iter()
                        .filter(|version| package_rocm_suffix(version).is_some())
                        .cloned()
                        .collect::<Vec<_>>(),
                    release,
                ),
            },
        },
        None => TorchaudioPinBasis::NoVerifiedRelease,
    };
    let newest = latest_build_for_rocm(published, rocm_version).with_context(|| {
        format!(
            "torchaudio builds for ROCm {rocm_version} are published in {index_url}, but none has a PEP 440 version ROCm CLI can order: {}",
            built_for_rocm.join(", ")
        )
    })?;
    Ok(pin(newest, basis))
}

/// The build in `versions` whose public version is `release` under PEP 440
/// (local label ignored, trailing zeros insignificant), as `uv` would match
/// `torchaudio=={release}`. Not a prefix match: `2.11.0` must not find
/// `2.11.0.2`.
fn published_build_of(versions: &[String], release: &str) -> Option<String> {
    let wanted = package_release_key(release)?;
    versions
        .iter()
        .find(|version| package_release_key(version).as_ref() == Some(&wanted))
        .cloned()
}

/// The newest of `versions` built for exactly `rocm_version`, chosen the way
/// `uv` would choose among them: in PEP 440 order, with final releases ahead of
/// any pre-release.
///
/// The fallback for [`torchaudio_pin`] only, never the first choice: newest is
/// not a safe discriminator among builds sharing a `+rocm<version>` label (see
/// there for the CPU-only `2.11.0+rocm10.1.0` precedent, and the vLLM engine's
/// ROCm 10.x discover-build table, which pins `2.11.0.2` exactly for the same
/// reason). Nothing on the index pairs a torchaudio build with a torch build,
/// so this assumes the newest torchaudio of a ROCm build is the one built
/// against that build's torch — the same choice `uv` made unaided when the
/// pinned build was also the newest one on the index.
///
/// Not `select_latest_stack_package` (in `wheel.rs`): `parse_version` reads only
/// `major.minor.patch` and rejects the rest, and torchaudio publishes rebuilds
/// as extra release segments (`2.11.0.2`, `2.11.0.3`), so that would drop every
/// rebuild and settle on the original `2.11.0` build.
fn latest_build_for_rocm(versions: &[String], rocm_version: &str) -> Option<String> {
    package_versions_matching_rocm(versions, rocm_version)
        .into_iter()
        .filter_map(|version| package_release_key(&version).map(|key| (key, version)))
        .max_by(|(left, _), (right, _)| (left.is_final(), left).cmp(&(right.is_final(), right)))
        .map(|(_, version)| version)
}

#[cfg(test)]
mod tests {
    use super::super::wheel::published_pip_requirements;
    use super::*;

    /// The live ROCm 10 index publishes torchaudio rebuilds as a fourth release
    /// segment beside the original build, for more than one ROCm build at once.
    #[test]
    fn torchaudio_pin_is_the_newest_rebuild_of_the_requested_rocm_build() {
        let published = [
            "2.11.0+rocm10.0.0",
            "2.11.0.2+rocm10.0.0",
            "2.11.0+rocm10.1.0",
            "2.11.0.2+rocm10.1.0",
            "2.11.0.3+rocm10.1.0",
        ]
        .map(str::to_owned);

        assert_eq!(
            latest_build_for_rocm(&published, "10.0.0").as_deref(),
            Some("2.11.0.2+rocm10.0.0")
        );
        assert_eq!(
            latest_build_for_rocm(&published, "10.1.0").as_deref(),
            Some("2.11.0.3+rocm10.1.0")
        );
        assert_eq!(latest_build_for_rocm(&published, "10.2.0"), None);
    }

    /// Mirrors uv's default: a pre-release is only chosen when no final
    /// release exists for that build, and a post-release counts as final.
    #[test]
    fn torchaudio_pin_prefers_final_releases_like_uv() {
        let published = [
            "2.12.0a0+rocm10.0.0",
            "2.11.0.2+rocm10.0.0",
            "2.11.0+rocm10.0.0",
        ]
        .map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&published, "10.0.0").as_deref(),
            Some("2.11.0.2+rocm10.0.0")
        );

        let only_pre = ["2.12.0a0+rocm10.0.0", "2.12.0a1+rocm10.0.0"].map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&only_pre, "10.0.0").as_deref(),
            Some("2.12.0a1+rocm10.0.0")
        );

        let post = ["2.11.0+rocm10.0.0", "2.11.0.post1+rocm10.0.0"].map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&post, "10.0.0").as_deref(),
            Some("2.11.0.post1+rocm10.0.0")
        );
    }

    /// The live `whl-next` shape for `+rocm10.1.0`: a CPU-only `2.11.0`, the
    /// hardware-verified `2.11.0.2`, and a newer `2.11.0.3` sharing the label.
    fn rocm10_1_torchaudio_builds() -> Vec<String> {
        [
            "2.11.0+rocm10.1.0",
            "2.11.0.2+rocm10.1.0",
            "2.11.0.3+rocm10.1.0",
            "2.11.0.2+rocm10.0.0",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    /// A verified release that is published wins over a newer build with the
    /// same `+rocm` label, and the printed line says it is the verified one.
    #[test]
    fn torchaudio_pin_takes_the_published_verified_release_over_a_newer_build() -> Result<()> {
        let pin = torchaudio_pin(
            &rocm10_1_torchaudio_builds(),
            "10.1.0",
            Some("2.11.0.2"),
            "https://index.test",
        )?;
        assert_eq!(pin.version, "2.11.0.2+rocm10.1.0");
        assert_eq!(pin.basis, TorchaudioPinBasis::Verified);
        assert!(pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains("torchaudio 2.11.0.2+rocm10.1.0, the build verified on hardware")
                && !line.contains("not hardware-verified"),
            "{line}"
        );
        // The requirement handed to `uv` names the same build the line does.
        assert_eq!(
            published_pip_requirements("10.1.0", &pin.version, "gfx1200", false)[3],
            "torchaudio==2.11.0.2+rocm10.1.0"
        );
        Ok(())
    }

    /// The verified release is matched as a whole PEP 440 version, never a
    /// prefix: a verified `2.11.0` must not be satisfied by `2.11.0.2`.
    #[test]
    fn torchaudio_pin_matches_the_verified_release_exactly() -> Result<()> {
        let published = ["2.11.0.2+rocm10.1.0", "2.11.0.3+rocm10.1.0"].map(str::to_owned);
        let pin = torchaudio_pin(&published, "10.1.0", Some("2.11.0"), "https://index.test")?;
        assert_eq!(
            pin.basis,
            TorchaudioPinBasis::VerifiedNotPublished {
                release: "2.11.0",
                other_build: None,
            }
        );
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.0");
        Ok(())
    }

    /// A verified release published only for another ROCm build falls back to
    /// the newest build for the requested one, and the printed reason names the
    /// ROCm build as the cause, not the Python or platform: the verified release
    /// is matched by ROCm line, so it can be published here under another label
    /// of the same line.
    #[test]
    fn torchaudio_pin_falls_back_to_newest_when_the_verified_release_is_published_for_another_rocm_build()
    -> Result<()> {
        let published = [
            "2.11.0+rocm10.1.1",
            "2.11.0.3+rocm10.1.1",
            "2.11.0.2+rocm10.1.0",
        ]
        .map(str::to_owned);
        let pin = torchaudio_pin(&published, "10.1.1", Some("2.11.0.2"), "https://index.test")?;
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.1");
        assert_eq!(
            pin.basis,
            TorchaudioPinBasis::VerifiedNotPublished {
                release: "2.11.0.2",
                other_build: Some("2.11.0.2+rocm10.1.0".to_owned()),
            }
        );
        assert!(!pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains(
                "torchaudio 2.11.0.3+rocm10.1.1, the newest build published for ROCm 10.1.1"
            ) && line.contains("not hardware-verified")
                && line.contains(
                    "the verified release, 2.11.0.2, is published for this Python and platform only for another ROCm build (2.11.0.2+rocm10.1.0), not for ROCm 10.1.1"
                )
                && !line.contains("is not published for this Python and platform"),
            "{line}"
        );
        Ok(())
    }

    /// A verified release with no build at all for this Python and platform
    /// falls back to the newest build, and the printed line says the release is
    /// not published here. An unlabelled build is not a ROCm build, so it does
    /// not count as one "for another ROCm build".
    #[test]
    fn torchaudio_pin_falls_back_to_newest_when_the_verified_release_is_unpublished() -> Result<()>
    {
        let published = ["2.11.0+rocm10.1.0", "2.11.0.3+rocm10.1.0", "2.11.0.2"].map(str::to_owned);
        let pin = torchaudio_pin(&published, "10.1.0", Some("2.11.0.2"), "https://index.test")?;
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.0");
        assert_eq!(
            pin.basis,
            TorchaudioPinBasis::VerifiedNotPublished {
                release: "2.11.0.2",
                other_build: None,
            }
        );
        assert!(!pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains(
                "torchaudio 2.11.0.3+rocm10.1.0, the newest build published for ROCm 10.1.0"
            ) && line.contains("not hardware-verified")
                && line.contains(
                    "the verified release, 2.11.0.2, is not published for this Python and platform"
                ),
            "{line}"
        );
        Ok(())
    }

    /// A real install notes an unverified torchaudio in its install log, for
    /// both reasons a pin can be unverified, and stays quiet for a verified
    /// one. The note is the pin's own `describe()` line, so it states the
    /// decision that was actually taken.
    #[test]
    fn unverified_torchaudio_note_is_logged_only_for_an_unverified_pin() -> Result<()> {
        let builds = rocm10_1_torchaudio_builds();
        let verified = torchaudio_pin(&builds, "10.1.0", Some("2.11.0.2"), "https://index.test")?;
        let not_published =
            torchaudio_pin(&builds, "10.1.0", Some("2.11.0.9"), "https://index.test")?;
        let no_verified_release = torchaudio_pin(&builds, "10.1.0", None, "https://index.test")?;
        assert_eq!(verified.basis, TorchaudioPinBasis::Verified);
        assert!(matches!(
            not_published.basis,
            TorchaudioPinBasis::VerifiedNotPublished { .. }
        ));
        assert_eq!(
            no_verified_release.basis,
            TorchaudioPinBasis::NoVerifiedRelease
        );

        assert_eq!(unverified_torchaudio_note(&verified), None);
        for pin in [&not_published, &no_verified_release] {
            let note = unverified_torchaudio_note(pin);
            assert_eq!(note, Some(format!("Note: {}", pin.describe())));
            assert!(
                note.as_deref()
                    .is_some_and(|note| note.contains("torchaudio 2.11.0.3+rocm10.1.0")
                        && note.contains("not hardware-verified")),
                "{note:?}"
            );
        }
        Ok(())
    }

    /// With no verified release recorded for the line, the newest build is
    /// pinned, and the printed line says it is not hardware-verified.
    #[test]
    fn torchaudio_pin_falls_back_to_newest_when_no_release_is_verified() -> Result<()> {
        let pin = torchaudio_pin(
            &rocm10_1_torchaudio_builds(),
            "10.1.0",
            None,
            "https://index.test",
        )?;
        assert_eq!(pin.version, "2.11.0.3+rocm10.1.0");
        assert_eq!(pin.basis, TorchaudioPinBasis::NoVerifiedRelease);
        assert!(!pin.is_verified());
        let line = pin.describe();
        assert!(
            line.contains(
                "torchaudio 2.11.0.3+rocm10.1.0, the newest build published for ROCm 10.1.0"
            ) && line.contains("not hardware-verified")
                && line.contains("no torchaudio release is recorded as verified"),
            "{line}"
        );
        Ok(())
    }

    /// The verified release the SDK reads is the vLLM engine's: 10.1 has one,
    /// 10.0 has none yet. Pins the wiring, not the table (which the engine's
    /// own tests own), so a change to it is seen here as a deliberate choice.
    #[test]
    fn torchaudio_verified_release_comes_from_the_vllm_discover_table() {
        assert_eq!(
            rocm_engine_vllm::verified_torchaudio_release("10.1.0"),
            Some("2.11.0.2")
        );
        assert_eq!(
            rocm_engine_vllm::verified_torchaudio_release("10.0.0"),
            None
        );
    }

    /// A version the parser cannot read is left out, never guessed at: a newer
    /// unreadable build must not win the pin.
    #[test]
    fn torchaudio_pin_skips_versions_it_cannot_read() -> Result<()> {
        let published = [
            "2.11.0+rocm10.0.0",
            "2.12.0-local-hack+rocm10.0.0",
            "2.11.0.2+rocm10.0.0",
        ]
        .map(str::to_owned);
        assert_eq!(
            torchaudio_pin(&published, "10.0.0", None, "https://index.test")?.version,
            "2.11.0.2+rocm10.0.0"
        );
        Ok(())
    }

    /// The two refusals name different problems, so each says which one it is.
    #[test]
    fn torchaudio_pin_refusal_says_whether_the_build_is_missing_or_unreadable() {
        let other_build = ["2.11.0.3+rocm10.1.0"].map(str::to_owned);
        let missing = torchaudio_pin(
            &other_build,
            "10.0.0",
            Some("2.11.0.2"),
            "https://index.test",
        )
        .expect_err("no torchaudio for the pinned build must refuse")
        .to_string();
        assert!(
            missing.contains("no torchaudio built for ROCm 10.0.0 is published"),
            "{missing}"
        );

        let unreadable = ["2.12.0-local-hack+rocm10.0.0"].map(str::to_owned);
        let unreadable = torchaudio_pin(&unreadable, "10.0.0", None, "https://index.test")
            .expect_err("an unreadable-only build must refuse")
            .to_string();
        assert!(
            unreadable.contains("none has a PEP 440 version")
                && unreadable.contains("2.12.0-local-hack+rocm10.0.0")
                && !unreadable.contains("no torchaudio built"),
            "{unreadable}"
        );
    }

    /// A post-release that is itself a dev build is still a pre-release to
    /// `uv`, so a final release wins over it.
    #[test]
    fn torchaudio_pin_treats_a_dev_post_release_as_a_pre_release() {
        let published = ["2.11.0+rocm10.0.0", "2.11.0.post1.dev0+rocm10.0.0"].map(str::to_owned);
        assert_eq!(
            latest_build_for_rocm(&published, "10.0.0").as_deref(),
            Some("2.11.0+rocm10.0.0")
        );
    }
}
