// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The content of a Doctor report, and the rule that decides whether one may
//! exist at all.
//!
//! A report is destined for a public, indexed issue tracker. Nothing here sends
//! one: this module builds content and refuses to build it, and that is all. No
//! network, no filesystem, no paths.
//!
//! The report is assembled field by field. A larger structure is never copied
//! wholesale, because that is how host names, file paths and error text leak
//! into something published.

use serde::{Deserialize, Serialize};

use crate::examine::Examination;

/// The agreement a reader and a report share.
///
/// Follows `ENGINE_RECIPE_CONTRACT_VERSION` and the Doctor catalog's
/// `contract_version`. An added field keeps this number; a removed field or a
/// changed type raises it.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

/// Where [`APPROVED_ARCHITECTURES`] was transcribed from.
///
/// The list is not maintained here. AMD's ROCm compatibility matrix is the
/// authoritative statement of which hardware ROCm supports, and this is a
/// snapshot of it, compiled in so that it ships signed and is never fetched.
///
/// Stamped rather than remembered: a snapshot with no provenance cannot be told
/// apart from a current one, and this list going stale is the failure mode that
/// matters. Reviewing it belongs to the per-release catalog review.
pub const APPROVED_ARCHITECTURES_SOURCE: &str = "ROCm compatibility matrix, ROCm 7.1";

/// Hardware the ROCm compatibility matrix lists as supported, by LLVM gfx
/// target.
///
/// The same vocabulary [`crate::examine::Gpu::gfx_target`] reports, so no
/// marketing-name mapping sits between the machine and this decision.
///
/// This is an allowlist and it is the only control preventing an unannounced
/// product from being named in a public issue. Anything absent is refused,
/// including anything unreadable. Absence from this list is not a claim about
/// retail availability -- plenty of hardware sold today is simply not on the
/// matrix yet, or never will be -- it is a claim about ROCm support, which is
/// the only question this gate is positioned to answer.
// `rustfmt::skip` because the line breaks here are meaning, not formatting:
// each comment labels the group beneath it, and reflowing packs the targets
// onto shared lines so every label ends up trailing the group *above* it. That
// is how this list came to say CDNA parts were RDNA 2 -- a comment that changed
// meaning because of what it ended up next to, in the one file whose job is to
// be exact about which hardware may be named in public.
#[rustfmt::skip]
pub const APPROVED_ARCHITECTURES: &[&str] = &[
    // CDNA 1 through 4.
    "gfx908", "gfx90a", "gfx942", "gfx950",
    // RDNA 2.
    "gfx1030",
    // RDNA 3 and 3.5.
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    // RDNA 4.
    "gfx1200", "gfx1201",
];

/// Why no report was produced.
///
/// Named rather than a bare `None`: Doctor has to explain the refusal, and
/// "hardware we cannot identify" and "hardware that is not released" call for
/// different sentences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A GPU on this machine is not on the ROCm compatibility matrix.
    ///
    /// Carries no identifier on purpose. Naming the target here would put it in
    /// whatever the caller prints, which is the leak this refusal exists to
    /// prevent.
    UnreleasedHardware,
    /// No GPU architecture could be read, so nothing confirms the hardware is
    /// on the compatibility matrix. Refused rather than assumed.
    ArchitectureUnreadable,
}

/// A report, as it would be published.
///
/// Every field is here because it was agreed, not because it was available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub schema: u32,
    /// The gfx target, which is also the only thing said about the hardware.
    pub architecture: String,
    /// Which snapshot of the ROCm compatibility matrix this build checked
    /// `architecture` against, verbatim from [`APPROVED_ARCHITECTURES_SOURCE`].
    ///
    /// The matrix changes release to release, and a `Report` carries no other
    /// trace of which revision decided its verdict. Without this, two reports
    /// naming the same architecture could disagree about whether it was
    /// supported, and nothing would say why.
    pub architecture_matrix: String,
    /// The catalog entry that matched, or [`UNRECOGNISED`] when none did.
    pub entry: String,
    pub os_family: String,
    /// Major only, e.g. `"22"` on Linux or `"10"` on Windows. The exact build
    /// identifies a machine far more narrowly than it helps group a problem,
    /// and this crate never reads that source: see `os_major` in
    /// `report.rs` for the field each platform's value actually comes from.
    pub os_major: String,
    pub cli_version: String,
    pub fix_offered: bool,
}

/// What a report says when the catalog recognised nothing.
pub const UNRECOGNISED: &str = "unrecognised";

/// What a reader made of a report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Understood(Box<Report>),
    /// The report is written to an agreement this reader does not know.
    ///
    /// Distinct from an empty report on purpose. A counter that read this as
    /// "nothing here" would silently undercount every report from a newer CLI,
    /// and the counts would look healthy while being wrong.
    Unread {
        schema_seen: u32,
    },
}

/// Whether a gfx target is on the ROCm compatibility matrix.
#[must_use]
pub fn is_rocm_supported(gfx_target: &str) -> bool {
    APPROVED_ARCHITECTURES.contains(&gfx_target)
}

/// Build the report for this machine, or refuse and say which rule refused.
///
/// # Errors
/// When the machine holds hardware that is not on the ROCm compatibility
/// matrix, or hardware whose architecture could not be read.
pub fn prepare_report(
    examination: &Examination,
    entry: Option<&str>,
    fix_offered: bool,
) -> Result<Report, Refusal> {
    // Every AMD GPU is checked, not just the one a finding concerns. A released
    // GPU sitting beside an unreleased one does not make the machine
    // reportable: publishing the released half would leak the other's existence
    // by the shape of what was withheld.
    let amd: Vec<&str> = examination
        .gpus
        .iter()
        .filter(|g| g.is_amd)
        .map(|g| g.gfx_target.trim())
        .collect();

    // Default-deny, and this is the branch that enforces it. A machine with no
    // readable AMD architecture has nothing confirming its hardware is on the
    // compatibility matrix, and "we could not tell" is not permission.
    if amd.is_empty() || amd.iter().any(|gfx| gfx.is_empty()) {
        return Err(Refusal::ArchitectureUnreadable);
    }
    if !amd.iter().all(|gfx| is_rocm_supported(gfx)) {
        return Err(Refusal::UnreleasedHardware);
    }

    // `entry` is the only value here that comes from a caller rather than from
    // the machine. Checked against the catalog so that a forged or mistaken id
    // cannot carry caller-supplied text onto a public tracker.
    let entry_recognised = entry.is_some_and(crate::fix::is_catalog_id);
    let entry = if entry_recognised {
        entry.expect("checked Some above").to_owned()
    } else {
        UNRECOGNISED.to_owned()
    };
    // A fix cannot be offered for a cause the catalog did not establish: that
    // is a self-contradictory fact once it reaches a public tracker. Derived
    // here rather than trusted from the caller, because `prepare_report` is
    // `pub` and re-exported, and nothing else enforces the two fields agree.
    let fix_offered = fix_offered && entry_recognised;

    Ok(Report {
        schema: REPORT_SCHEMA_VERSION,
        // The first AMD architecture. All of them are on the compatibility
        // matrix by the check above, so this narrows what is said rather than
        // choosing what to hide; a machine holding two approved architectures
        // is rare enough that a second field would buy grouping accuracy
        // nobody needs.
        architecture: (*amd.first().expect("checked non-empty above")).to_owned(),
        architecture_matrix: APPROVED_ARCHITECTURES_SOURCE.to_owned(),
        entry,
        os_family: examination.os_family.clone(),
        os_major: os_major(examination),
        cli_version: env!("CARGO_PKG_VERSION").to_owned(),
        fix_offered,
    })
}

/// The population-level OS major version to publish.
///
/// Sourced per platform, because no single `Examination` field holds "the OS
/// release" on both: see the two branches below for what each one actually
/// reads. Whatever the source, the result is passed through [`leading_digits`],
/// which discards anything that is not purely numeric -- so a future change to
/// either probe cannot reopen the leak this closes by feeding free text back
/// in through here.
fn os_major(examination: &Examination) -> String {
    match examination.os_family.as_str() {
        "linux" => {
            // `distro_version` is `VERSION_ID` from `/etc/os-release`
            // (`examine.rs::probe_os`), e.g. "22.04" -- the actual distro
            // release. `os_version` on Linux is `uname -v`'s kernel *build*
            // banner (e.g. "#1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC
            // 2026"): a timestamp, not a release, and reading it here is the
            // leak this function exists to close. A field named `os_version`
            // sitting beside `os_family` reads as "the OS release" to any
            // competent reader; it is not, on this platform.
            leading_digits(&examination.distro_version)
        }
        "windows" => {
            // Windows has no `/etc/os-release` analogue: `distro_id` /
            // `distro_version` are populated only under `runtime_is_linux()`
            // in `examine.rs::probe_os`, and stay empty here. The only
            // version-bearing field on this platform is `os_version` itself,
            // `cmd /C ver`'s banner, e.g. "Microsoft Windows [Version
            // 10.0.22631.4460]" -- extract the NT major component that
            // follows "Version " rather than reading the banner whole.
            windows_os_major(&examination.os_version)
        }
        // No other platform is supported by `examine.rs`; nothing here is
        // known to hold a release, so nothing is published.
        _ => String::new(),
    }
}

/// The leading dot/dash-delimited component of `version`, kept only when it
/// is entirely ASCII digits.
///
/// A full build string narrows a machine much further than it helps group a
/// problem: "22.04.3 with kernel 6.5.0-41" is close to an identifier, while
/// "22" is a population. Anything that survives the split but is not a bare
/// number is discarded rather than passed through -- that non-numeric
/// remainder is exactly the free text a report bound for a public tracker
/// must not carry, whatever field it came from.
fn leading_digits(version: &str) -> String {
    let candidate = version.split(['.', '-']).next().unwrap_or_default();
    if !candidate.is_empty() && candidate.bytes().all(|b| b.is_ascii_digit()) {
        candidate.to_owned()
    } else {
        String::new()
    }
}

/// The NT major version out of a `cmd /C ver` banner such as
/// "Microsoft Windows [Version 10.0.22631.4460]", or empty when the banner
/// does not have the expected "Version " marker (a localized banner, for
/// instance) -- an unrecognised shape is refused rather than guessed at.
fn windows_os_major(ver_banner: &str) -> String {
    ver_banner
        .split_once("Version ")
        .map_or(String::new(), |(_, rest)| {
            leading_digits(rest.trim_end_matches(']'))
        })
}

/// Read a report written by some version of this CLI.
#[must_use]
pub fn read_report(json: &str) -> ReadOutcome {
    // The schema is read before the body. Deserializing first and checking
    // after would make a newer report look malformed rather than merely
    // unfamiliar, and those need different answers from a counter.
    let Ok(envelope) = serde_json::from_str::<serde_json::Value>(json) else {
        return ReadOutcome::Unread { schema_seen: 0 };
    };
    let schema_seen = envelope
        .get("schema")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0);
    if schema_seen != REPORT_SCHEMA_VERSION {
        return ReadOutcome::Unread { schema_seen };
    }
    serde_json::from_value::<Report>(envelope)
        .map_or(ReadOutcome::Unread { schema_seen }, |report| {
            ReadOutcome::Understood(Box::new(report))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examine::Gpu;

    /// A machine whose every free-text field is a distinctive marker.
    ///
    /// The markers are what [`no_report_carries_a_value_the_machine_did_not_agree_to_publish`]
    /// sweeps for. Real values are avoided on purpose: a plausible-looking path
    /// could be missed by eye in a rendered report, whereas one of these could
    /// not.
    fn machine_of_sentinels(gfx: &str) -> Examination {
        Examination {
            os_family: "linux".to_owned(),
            // A real `uname -v` shape (this is what it prints on the host
            // this fix was written on), not a release: see
            // [`the_reported_os_major_comes_from_the_distro_release_not_the_kernel_banner`]
            // for why that distinction is the whole point.
            os_version: "#1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026".to_owned(),
            distro_version: "22.04".to_owned(),
            user_name: "SENTINEL-USER".to_owned(),
            rocm_path: "/SENTINEL-PATH/rocm".to_owned(),
            kernel_cmdline: "SENTINEL-CMDLINE".to_owned(),
            hip_sdk_path: "C:/SENTINEL-PATH".to_owned(),
            cpu_model: "SENTINEL-CPU".to_owned(),
            distro_id: "SENTINEL-DISTRO".to_owned(),
            rocminfo_status: "SENTINEL-ERROR-TEXT".to_owned(),
            has_amd_gpu: true,
            gpus: vec![Gpu {
                name: "SENTINEL-MARKETING-NAME".to_owned(),
                gfx_target: gfx.to_owned(),
                pci_id: "SENTINEL-PCI".to_owned(),
                is_apu: Some(false),
                is_amd: true,
            }],
            ..Examination::default()
        }
    }

    /// Every marker planted above, so the sweep cannot silently check fewer
    /// than it was given.
    const SENTINELS: &[&str] = &[
        "SENTINEL-USER",
        "SENTINEL-PATH",
        "SENTINEL-CMDLINE",
        "SENTINEL-CPU",
        "SENTINEL-DISTRO",
        "SENTINEL-ERROR-TEXT",
        "SENTINEL-MARKETING-NAME",
        "SENTINEL-PCI",
    ];

    /// An architecture no product will ever have.
    const NOT_RELEASED: &str = "gfx9999";

    /// I1 — a machine holding hardware that is not on the ROCm compatibility
    /// matrix never produces a report.
    ///
    /// The paired assertion is the one that matters. "Unapproved machine is
    /// refused" alone is satisfied by an implementation that refuses
    /// everything, so the same machine with the hardware swapped for something
    /// released has to come back with a report.
    #[test]
    fn hardware_off_the_rocm_compatibility_matrix_never_produces_a_report() {
        let released = prepare_report(&machine_of_sentinels("gfx1100"), None, false);
        assert!(
            released.is_ok(),
            "premise failed: a machine holding only released hardware must produce a report, \
             otherwise the refusal below is satisfied by refusing everything. Got {released:?}"
        );

        assert_eq!(
            prepare_report(&machine_of_sentinels(NOT_RELEASED), None, false),
            Err(Refusal::UnreleasedHardware),
            "{NOT_RELEASED} is on no compatibility matrix, so it must not be describable"
        );
    }

    /// I1, continued — one unreleased GPU withholds the whole report.
    ///
    /// Reporting the released half would leak the other's existence by the
    /// shape of what was withheld.
    #[test]
    fn one_unreleased_gpu_withholds_the_whole_report_not_just_its_own_entry() {
        let mut mixed = machine_of_sentinels("gfx1100");
        mixed.gpus.push(Gpu {
            name: "SENTINEL-MARKETING-NAME".to_owned(),
            gfx_target: NOT_RELEASED.to_owned(),
            pci_id: "SENTINEL-PCI".to_owned(),
            is_apu: Some(false),
            is_amd: true,
        });
        assert_eq!(
            prepare_report(&mixed, None, false),
            Err(Refusal::UnreleasedHardware),
            "a released GPU beside an unreleased one does not make the machine reportable"
        );
    }

    /// I3 — hardware that could not be identified is refused, not assumed.
    #[test]
    fn hardware_that_could_not_be_identified_is_refused_rather_than_assumed() {
        assert_eq!(
            prepare_report(&machine_of_sentinels(""), None, false),
            Err(Refusal::ArchitectureUnreadable),
            "nothing confirmed this hardware is on the compatibility matrix, and default-deny \
             is the whole point"
        );
    }

    /// I2 — no report carries a value the machine did not agree to publish.
    ///
    /// Sweeps the serialized bytes rather than enumerating field names, because
    /// the mistake being guarded against is a larger structure copied wholesale:
    /// a field list check passes while a nested examination rides along inside
    /// an approved field.
    #[test]
    fn no_report_carries_a_value_the_machine_did_not_agree_to_publish() {
        let report = prepare_report(&machine_of_sentinels("gfx1100"), Some("fix-6-path"), true)
            .expect("a released machine must produce a report");
        let serialized = serde_json::to_string(&report).expect("a report must serialize");

        // Non-vacuity. An empty report carries no markers either, so without
        // this the sweep below would pass against a report that says nothing.
        assert!(
            serialized.contains("gfx1100"),
            "the report has to actually describe the machine before 'it leaks nothing' means \
             anything: {serialized}"
        );

        for sentinel in SENTINELS {
            assert!(
                !serialized.contains(sentinel),
                "{sentinel} reached a report bound for a public issue tracker: {serialized}"
            );
        }
    }

    /// I6 — the report's OS major comes from the distro release, never from
    /// the kernel build banner that actually lives in `os_version` on Linux.
    ///
    /// Found by mutation, not by design: publishing `examination.os_version`
    /// whole passed every other test here. The sentinel sweep could not catch
    /// it because a kernel banner is not a planted marker, it is a real and
    /// plausible-looking value — and that is exactly what makes it easy to
    /// ship. A prior version of this test set `os_version = "22.04.3"`, a
    /// shape `probe_os` cannot produce on Linux (`uname -v` prints a build
    /// banner, not a release), so the assertion was satisfied by an invented
    /// fixture rather than by the production path. These three banners are
    /// real: the first is what `uname -v` prints on the host this fix was
    /// written on; the other two are the Ubuntu and Debian shapes.
    #[test]
    fn the_reported_os_major_comes_from_the_distro_release_not_the_kernel_banner() {
        let kernel_banners = [
            "#1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026",
            "#139-Ubuntu SMP Fri Sep 27 14:22:11 UTC 2024",
            "#1 SMP PREEMPT_DYNAMIC Debian 6.1.129-1",
        ];
        for banner in kernel_banners {
            let mut machine = machine_of_sentinels("gfx1100");
            machine.os_version = banner.to_owned();
            machine.distro_version = "22.04".to_owned();

            let report = prepare_report(&machine, None, false)
                .expect("a released machine must produce a report");
            assert_eq!(
                report.os_major, "22",
                "os_major must come from distro_version, not the kernel banner in os_version \
                 ({banner:?})"
            );

            let serialized = serde_json::to_string(&report).expect("a report must serialize");
            assert!(
                !serialized.contains(banner),
                "the kernel build banner reached a report bound for a public tracker: {serialized}"
            );
        }
    }

    /// I6, continued — the Windows equivalent. `distro_version` is never
    /// populated there (`examine.rs::probe_os` only sets it under
    /// `runtime_is_linux()`), so the NT major version has to come out of
    /// `os_version` itself, `cmd /C ver`'s banner — but only the major
    /// component, never the banner whole.
    #[test]
    fn the_reported_os_major_on_windows_is_the_nt_major_version_not_the_ver_banner() {
        let mut machine = machine_of_sentinels("gfx1100");
        machine.os_family = "windows".to_owned();
        machine.os_version = "Microsoft Windows [Version 10.0.22631.4460]".to_owned();

        let report = prepare_report(&machine, None, false)
            .expect("a released machine must produce a report");
        assert_eq!(
            report.os_major, "10",
            "the report groups by NT major version, so that is what it carries"
        );

        let serialized = serde_json::to_string(&report).expect("a report must serialize");
        assert!(
            !serialized.contains("22631"),
            "the exact Windows build reached a report bound for a public tracker: {serialized}"
        );
        assert!(
            !serialized.contains("Microsoft Windows"),
            "the ver banner reached a report bound for a public tracker: {serialized}"
        );
    }

    /// I6, continued — a non-numeric source is refused rather than
    /// published, regardless of platform. This is the guard that keeps the
    /// leak closed even if a future edit changes the source again: whatever
    /// feeds `os_major` next, free text still cannot pass through it.
    #[test]
    fn a_non_numeric_os_release_source_never_reaches_the_report_as_free_text() {
        let mut machine = machine_of_sentinels("gfx1100");
        machine.distro_version = "SENTINEL-UNPARSEABLE-RELEASE".to_owned();

        let report = prepare_report(&machine, None, false)
            .expect("a released machine must produce a report");
        assert_eq!(
            report.os_major, "",
            "a release string that does not reduce to a bare number must not pass through as \
             free text"
        );
    }

    /// I5 — an entry id the catalog does not know is never published verbatim.
    ///
    /// `entry` is the one field whose value comes from a caller rather than
    /// from the machine, which makes it the one free-text hole in a structure
    /// that is otherwise assembled field by field. A forged or mistaken id must
    /// not ride through to a public tracker.
    #[test]
    fn an_entry_id_the_catalog_does_not_know_is_never_published_verbatim() {
        // Non-vacuity: a real id has to reach the report, or "the forged one
        // does not" is satisfied by discarding every id.
        let known = prepare_report(&machine_of_sentinels("gfx1100"), Some("fix-6-path"), false)
            .expect("a released machine must produce a report");
        assert_eq!(
            known.entry, "fix-6-path",
            "premise failed: a real catalog id must reach the report, otherwise the assertion \
             below passes against an implementation that publishes no id at all"
        );

        let forged = prepare_report(
            &machine_of_sentinels("gfx1100"),
            Some("SENTINEL-FORGED-ENTRY"),
            false,
        )
        .expect("a released machine must produce a report");
        assert_eq!(
            forged.entry, UNRECOGNISED,
            "an id the catalog does not know is not a finding, and publishing it verbatim would \
             put caller-supplied text on a public tracker"
        );
    }

    /// I4 — a reader that does not know the agreement says so, rather than
    /// reading the report as carrying nothing.
    #[test]
    fn a_reader_that_does_not_understand_a_report_says_so_rather_than_counting_it_as_empty() {
        let newer = format!(
            r#"{{"schema":{},"architecture":"gfx1100","entry":"fix-6-path","os_family":"linux","os_major":"22","cli_version":"9.9.9","fix_offered":true,"field_added_later":"x"}}"#,
            REPORT_SCHEMA_VERSION + 1
        );
        assert_eq!(
            read_report(&newer),
            ReadOutcome::Unread {
                schema_seen: REPORT_SCHEMA_VERSION + 1
            },
            "a newer agreement is unread, never counted as zero"
        );

        let current = serde_json::to_string(
            &prepare_report(&machine_of_sentinels("gfx1100"), Some("fix-6-path"), true)
                .expect("a released machine must produce a report"),
        )
        .expect("a report must serialize");
        assert!(
            matches!(read_report(&current), ReadOutcome::Understood(_)),
            "premise failed: a report this reader does write must be one it can read, or the \
             assertion above is satisfied by a reader that understands nothing"
        );
    }
}
