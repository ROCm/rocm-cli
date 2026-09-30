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

/// Distribution identifiers that may be published, as `/etc/os-release` spells
/// them in `ID=`.
///
/// This list exists for a different reason than [`APPROVED_ARCHITECTURES`].
/// No distribution is a secret, and none is withheld here. The hazard is that
/// `ID=` is free text read from a file on the user's machine: a vendor image,
/// a derivative, or a private build writes whatever it likes there, and a
/// report bound for an issue tracker must not carry it. Anything absent
/// becomes [`DISTRO_OTHER`], which still groups and says nothing.
///
/// Generous on purpose. A name that is missing costs grouping accuracy for
/// real users, while a name that is present costs nothing, so the bar for
/// adding one is only that it is a distribution rather than a description of
/// somebody's fleet.
// `rustfmt::skip` for the same reason as `APPROVED_ARCHITECTURES`: the comments
// label the group beneath them, and reflowing moves each label onto the group
// above it.
#[rustfmt::skip]
pub const APPROVED_DISTROS: &[&str] = &[
    // Named by the ROCm compatibility matrix.
    "ubuntu", "rhel", "sles", "ol", "debian", "rocky", "azurelinux",
    // Common elsewhere, and grouped rather than flattened into `other`.
    "almalinux", "centos", "fedora", "opensuse-leap", "opensuse-tumbleweed",
    "arch", "linuxmint", "pop",
];

/// What the distribution field says when the identifier is not one this build
/// recognises. A real value, so that such machines still group together.
pub const DISTRO_OTHER: &str = "other";

/// What a field says when this build looked and could not tell.
///
/// Kept apart from [`NONE`]: "no ROCm is installed" and "ROCm is installed and
/// its version could not be read" are different facts, and a counter that
/// merged them would report an install problem as an absence.
pub const UNKNOWN: &str = "unknown";

/// What a field says when the thing is absent rather than unreadable.
pub const NONE: &str = "none";

/// Engine names that may be published.
///
/// Every value is written by this crate rather than parsed from a machine, so
/// this guards against a future probe rather than against today's. The
/// accompanying version is parsed, and is truncated instead.
pub const APPROVED_ENGINES: &[&str] = &["pytorch", "llama-cpp"];

/// Why no report was produced.
///
/// Named rather than a bare `None`: Doctor has to explain the refusal, and
/// "hardware we cannot identify" and "hardware that is not released" call for
/// different sentences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Refusal {
    /// The marker a written refusal carries, and the value
    /// [`ReadOutcome::Refused`] hands back.
    ///
    /// Here rather than at the point of printing, so that the writer, the
    /// reader and the tests all name the same constant. Spelled out arm by arm
    /// rather than derived, because this vocabulary is part of the schema: a
    /// renamed variant must not silently rename a marker that reports already
    /// in the field were written with.
    #[must_use]
    pub const fn marker(self) -> &'static str {
        match self {
            Self::UnreleasedHardware => "unreleased-hardware",
            Self::ArchitectureUnreadable => "architecture-unreadable",
        }
    }

    /// Every refusal, so a reader or a test cannot cover fewer than exist.
    pub const ALL: &'static [Self] = &[Self::UnreleasedHardware, Self::ArchitectureUnreadable];
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
    /// The distribution, as an `/etc/os-release` `ID=` value on the approved
    /// list, or [`DISTRO_OTHER`]. `"windows"` on Windows, which has no such
    /// file.
    ///
    /// Carried beside `os_family` rather than replacing it. Grouping needs to
    /// tell Ubuntu 22 from any other distribution numbered 22, which the
    /// family and the major version cannot do between them.
    pub distro: String,
    /// The installed ROCm release as major and minor, e.g. `"7.1"`, or
    /// [`NONE`] / [`UNKNOWN`].
    pub rocm: String,
    /// The inference engine found on this machine, or [`NONE`] / [`UNKNOWN`].
    pub engine: String,
    /// That engine's release as major and minor, truncated the same way as
    /// every other version here because it is parsed from an installed
    /// package rather than written by this crate.
    pub engine_version: String,
    pub cli_version: String,
    pub fix_offered: bool,
}

/// What a report says when the catalog recognised nothing.
pub const UNRECOGNISED: &str = "unrecognised";

/// What a reader made of a report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    Understood(Box<Report>),
    /// The machine declined to describe itself, and named the rule that
    /// declined.
    ///
    /// A refusal is evidence, not a gap. It is the only trace the disclosure
    /// guard leaves, so counting refusals is how anybody learns whether the
    /// guard fires on the machines it was meant to fire on. Merged into
    /// [`ReadOutcome::Unread`] it would instead read as a reader fault, and
    /// the guard working would be indistinguishable from the reader broken.
    Refused {
        /// The marker the writer used, e.g. `"unreleased-hardware"`. The
        /// schema version gates this vocabulary: a reader that accepted the
        /// schema has accepted the set of markers that go with it.
        reason: String,
    },
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
    // Together, so the pair cannot disagree. A version beside `none` would
    // describe an engine the report also says is not installed.
    let (engine, engine_version) = engine_and_version(examination);

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
        distro: distro(examination),
        rocm: rocm_release(examination),
        engine,
        engine_version,
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

/// The distribution to publish.
///
/// Linux reads `ID=` from `/etc/os-release`, which is free text written by
/// whoever built the image. It is checked against [`APPROVED_DISTROS`] rather
/// than published, because an unrecognised value is as likely to name a
/// company as a distribution.
///
/// Windows has no such file: `examine.rs::probe_os` populates `distro_id` only
/// under `runtime_is_linux()`, so the value there is a constant rather than
/// anything read from the machine.
fn distro(examination: &Examination) -> String {
    match examination.os_family.as_str() {
        "linux" => {
            let id = examination.distro_id.trim().to_ascii_lowercase();
            if id.is_empty() {
                // `/etc/os-release` was missing or unreadable. Distinct from an
                // unrecognised name: this build did not get to decide.
                UNKNOWN.to_owned()
            } else if APPROVED_DISTROS.contains(&id.as_str()) {
                id
            } else {
                DISTRO_OTHER.to_owned()
            }
        }
        "windows" => "windows".to_owned(),
        _ => String::new(),
    }
}

/// The installed ROCm release, as major and minor.
///
/// `rocm_path` decides absence and `rocm_version` decides readability, because
/// `examine.rs` collapses both into an empty string: `rocm_version` is
/// `install.version.unwrap_or_default()`, so "no install was found" and "an
/// install was found whose version could not be read" arrive identical. Those
/// are different facts to anybody counting, in the same way [`ReadOutcome`]
/// keeps an unreadable report apart from an absent one.
///
/// The path itself is only ever read here. It is never published.
fn rocm_release(examination: &Examination) -> String {
    if examination.rocm_path.trim().is_empty() {
        return NONE.to_owned();
    }
    let release = major_minor(&examination.rocm_version);
    if release.is_empty() {
        UNKNOWN.to_owned()
    } else {
        release
    }
}

/// The engine and its release, decided together.
///
/// `framework` is written by this crate from a closed set, so it is checked
/// against [`APPROVED_ENGINES`] to catch a future probe rather than today's.
/// `framework_version` is parsed out of an installed package, so it is
/// truncated like every other version here.
///
/// `"unknown"` is `Examination`'s struct default for this field
/// (`examine.rs`), and a completed probe leaves it in place when neither
/// engine was found, so it maps to [`NONE`] rather than [`UNKNOWN`]: the
/// ordinary "no engine installed" case is an absence, not a case where this
/// build looked and could not tell (the vocabulary [`UNKNOWN`] and [`NONE`]
/// document, which `docs/testing.md`'s report-fields section also states).
/// This mapping presumes the caller always ran a probe before building a
/// report -- an `Examination` built some other way, whose `framework` was
/// simply never touched, would be indistinguishable from that ordinary case
/// and would also read as absent here.
///
/// `"skipped"` is what `examine.rs` records when
/// [`crate::examine::FrameworkProbe::Skip`] was requested and the probe never
/// ran at all. Unlike the default, that is not
/// "looked and found nothing", so it is left to fall through the allowlist
/// check below rather than special-cased, and comes back [`UNKNOWN`].
fn engine_and_version(examination: &Examination) -> (String, String) {
    let name = examination.framework.trim().to_ascii_lowercase();
    if name == UNKNOWN {
        return (NONE.to_owned(), NONE.to_owned());
    }
    if !APPROVED_ENGINES.contains(&name.as_str()) {
        return (UNKNOWN.to_owned(), UNKNOWN.to_owned());
    }
    let version = major_minor(&examination.framework_version);
    let version = if version.is_empty() {
        UNKNOWN.to_owned()
    } else {
        version
    };
    (name, version)
}

/// The leading `major.minor` of `version`, keeping only components that are
/// entirely ASCII digits.
///
/// Wider than [`leading_digits`], which keeps the major alone. A ROCm or
/// engine release without its minor does not group: 7.0 and 7.1 are different
/// problems, while 22.04 and 22.10 are the same population. Anything past the
/// minor is a build, which narrows toward one machine, so it is dropped.
fn major_minor(version: &str) -> String {
    let mut parts = version.trim().split(['.', '-', '+', '_']);
    let major = parts.next().unwrap_or_default();
    if major.is_empty() || !major.bytes().all(|b| b.is_ascii_digit()) {
        return String::new();
    }
    match parts.next() {
        Some(minor) if !minor.is_empty() && minor.bytes().all(|b| b.is_ascii_digit()) => {
            format!("{major}.{minor}")
        }
        // A bare major is still a population. A non-numeric minor is the free
        // text this function exists to drop, and dropping it must not take the
        // major with it.
        _ => major.to_owned(),
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

/// The JSON envelope `rocm diagnose --report --json` prints for a refusal.
///
/// Built here rather than at the call site, so the writer and this module's
/// own reader tests construct the identical shape from the identical
/// function. A hand-rolled duplicate literal at either end can drift from
/// what the other actually produces and stay green; calling this cannot.
#[must_use]
pub fn refusal_envelope(refusal: Refusal, explanation: &str) -> serde_json::Value {
    serde_json::json!({
        "schema": REPORT_SCHEMA_VERSION,
        "refused": refusal.marker(),
        "explanation": explanation,
        "architecture_matrix": APPROVED_ARCHITECTURES_SOURCE,
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
    // Before the body, because a refusal envelope carries the schema and none
    // of the report's fields, so deserializing first would fail and report a
    // deliberate refusal as a reader fault.
    //
    // Checked against `Refusal::ALL` rather than accepted verbatim: the schema
    // version gates this vocabulary, so a marker outside it cannot have been
    // written by any version of this CLI that speaks this schema. Accepting it
    // anyway would carry whatever a forged or corrupted envelope put there
    // through to a reason field a counter treats as trusted vocabulary, which
    // is the same free-text leak this whole module exists to refuse elsewhere.
    if let Some(reason) = envelope.get("refused").and_then(serde_json::Value::as_str) {
        return if Refusal::ALL
            .iter()
            .any(|refusal| refusal.marker() == reason)
        {
            ReadOutcome::Refused {
                reason: reason.to_owned(),
            }
        } else {
            ReadOutcome::Unread { schema_seen }
        };
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
            // Real shapes with a marker in the tail, not pure markers. A pure
            // marker is rejected outright and proves only that garbage is
            // dropped; these prove the published head survives while the build
            // tail -- the part that narrows toward one machine -- does not.
            // `framework` is a real approved value for the same reason: an
            // unapproved one short-circuits before the version is ever read,
            // so the version sweep would pass without that path running.
            rocm_version: "7.1.0-SENTINEL-ROCM-BUILD".to_owned(),
            framework: "pytorch".to_owned(),
            framework_version: "2.5.1+SENTINEL-ENGINE-BUILD".to_owned(),
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
        "SENTINEL-ROCM-BUILD",
        "SENTINEL-ENGINE-BUILD",
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

    /// A machine described by the things a report may carry, so a population
    /// can be built out of them.
    ///
    /// Every argument is something the report is supposed to distinguish. The
    /// fields varied *inside* this helper are the ones it is supposed to
    /// ignore, and they differ on every call, so a report that leaked any of
    /// them would split a group that must stay whole.
    fn machine(gfx: &str, distro: (&str, &str), rocm: &str, engine: (&str, &str)) -> Examination {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NTH: AtomicU32 = AtomicU32::new(0);
        let nth = NTH.fetch_add(1, Ordering::Relaxed);

        Examination {
            os_family: "linux".to_owned(),
            distro_id: distro.0.to_owned(),
            // A patch component that differs per machine. Two hosts on 22.04
            // and 22.04.3 are one population, and a report that said otherwise
            // would make every host its own group.
            distro_version: format!("{}.{nth}", distro.1),
            os_version: format!("#{nth} SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026"),
            rocm_path: "/opt/rocm".to_owned(),
            rocm_version: format!("{rocm}.{nth}"),
            framework: engine.0.to_owned(),
            framework_version: format!("{}.{nth}", engine.1),
            cpu_model: format!("cpu-model-{nth}"),
            user_name: format!("user-{nth}"),
            has_amd_gpu: true,
            gpus: vec![Gpu {
                name: format!("marketing-name-{nth}"),
                gfx_target: gfx.to_owned(),
                pci_id: format!("pci-{nth}"),
                is_apu: Some(false),
                is_amd: true,
            }],
            ..Examination::default()
        }
    }

    /// The description a grouper matches on: every published field except the
    /// ones that describe this build rather than this machine.
    fn description(report: &Report) -> String {
        format!(
            "{}|{}|{}-{}|{}|{}-{}",
            report.entry,
            report.architecture,
            report.distro,
            report.os_major,
            report.rocm,
            report.engine,
            report.engine_version,
        )
    }

    fn describe(machine: &Examination, entry: Option<&str>) -> String {
        description(&prepare_report(machine, entry, false).expect("a released machine reports"))
    }

    /// Reports are grouped by exact match on their fields, so the fields have
    /// to put the same problem in one group and different problems in
    /// different ones. Nothing else checks this, and neither half is visible
    /// by reading the field list.
    ///
    /// The failure this exists to catch is not a wrong value. It is a field
    /// set that is too fine, making every machine its own group, or too
    /// coarse, collapsing distinct problems into one. Both look fine field by
    /// field and make the whole reporting path worthless.
    #[test]
    fn the_published_fields_group_one_problem_together_and_two_problems_apart() {
        // Same problem, different machines. Every difference here is something
        // a report must not carry: a kernel build, a patch release, a CPU, a
        // GPU's marketing name, a PCI address, a user.
        let same: std::collections::HashSet<String> = (0..8)
            .map(|_| {
                describe(
                    &machine("gfx942", ("ubuntu", "22"), "7.1", ("pytorch", "2.5")),
                    Some("fix-6-path"),
                )
            })
            .collect();
        assert_eq!(
            same.len(),
            1,
            "eight machines with one problem produced {} groups. A field set this fine gives \
             every host its own group, and no group ever describes a problem: {same:?}",
            same.len()
        );

        // Different problems. Each differs from the first in exactly one thing
        // the report is supposed to separate on.
        let distinct = [
            // The case that motivated the distribution field. Same family,
            // same major: without the distribution these two are one group.
            describe(
                &machine("gfx942", ("rhel", "22"), "7.1", ("pytorch", "2.5")),
                Some("fix-6-path"),
            ),
            describe(
                &machine("gfx1100", ("ubuntu", "22"), "7.1", ("pytorch", "2.5")),
                Some("fix-6-path"),
            ),
            describe(
                &machine("gfx942", ("ubuntu", "24"), "7.1", ("pytorch", "2.5")),
                Some("fix-6-path"),
            ),
            // A minor release apart. 7.0 and 7.1 are different problems.
            describe(
                &machine("gfx942", ("ubuntu", "22"), "7.0", ("pytorch", "2.5")),
                Some("fix-6-path"),
            ),
            describe(
                &machine("gfx942", ("ubuntu", "22"), "7.1", ("llama-cpp", "0.9")),
                Some("fix-6-path"),
            ),
            describe(
                &machine("gfx942", ("ubuntu", "22"), "7.1", ("pytorch", "2.5")),
                None,
            ),
        ];
        let base = same.into_iter().next().expect("one group above");
        for (nth, other) in distinct.iter().enumerate() {
            assert_ne!(
                *other, base,
                "difference {nth} did not change the description, so two different problems \
                 land in one group and a team reading it cannot tell them apart"
            );
        }
        let unique: std::collections::HashSet<&String> = distinct.iter().collect();
        assert_eq!(
            unique.len(),
            distinct.len(),
            "two different problems share a description: {distinct:?}"
        );
    }

    /// The grouping fields carry a population, not a machine.
    ///
    /// Asserts the published values rather than the absence of the markers.
    /// The sweep alone would pass if every one of these fields were empty, and
    /// an empty field is exactly what a grouper cannot use: this states that
    /// the numeric head survived while the build tail did not.
    #[test]
    fn the_grouping_fields_keep_the_release_and_drop_the_build() {
        let report = prepare_report(&machine_of_sentinels("gfx1100"), None, false)
            .expect("a released machine must produce a report");

        assert_eq!(
            report.rocm, "7.1",
            "the ROCm release has to survive truncation: 7.0 and 7.1 are different problems"
        );
        assert_eq!(report.engine, "pytorch");
        assert_eq!(
            report.engine_version, "2.5",
            "the engine release has to survive truncation the same way"
        );
        assert_eq!(
            report.distro, DISTRO_OTHER,
            "an ID this build does not recognise has to group, not be republished"
        );
    }

    /// A distribution name is checked, not trusted.
    ///
    /// Paired, because "always answers `other`" satisfies the unrecognised
    /// half on its own and would throw away every real distribution.
    #[test]
    fn a_recognised_distribution_is_named_and_an_unrecognised_one_is_not() {
        let mut known = machine_of_sentinels("gfx1100");
        known.distro_id = "Ubuntu".to_owned();
        let report = prepare_report(&known, None, false).expect("a released machine reports");
        assert_eq!(
            report.distro, "ubuntu",
            "premise failed: a distribution on the list must be named, otherwise the case below \
             is satisfied by discarding every name"
        );

        // The shape that matters: a private image whose `ID=` names its owner
        // rather than a distribution.
        let mut vendor = machine_of_sentinels("gfx1100");
        vendor.distro_id = "SENTINEL-CORP-INTERNAL-IMAGE".to_owned();
        let report = prepare_report(&vendor, None, false).expect("a released machine reports");
        assert_eq!(report.distro, DISTRO_OTHER);
        let serialized = serde_json::to_string(&report).expect("a report must serialize");
        assert!(
            !serialized.contains("SENTINEL-CORP"),
            "an ID written by whoever built the image reached a public tracker: {serialized}"
        );
    }

    /// A missing `/etc/os-release` reads as unreadable, not as an unrecognised
    /// distribution.
    ///
    /// Distinct from [`DISTRO_OTHER`] on purpose: "the file was missing" and
    /// "the file named something this build does not recognise" are different
    /// facts, and only the first is this build never getting to decide.
    #[test]
    fn a_missing_os_release_reads_as_unknown_not_as_an_unrecognised_distribution() {
        let mut machine = machine_of_sentinels("gfx1100");
        machine.distro_id = String::new();
        let report = prepare_report(&machine, None, false).expect("a released machine reports");
        assert_eq!(
            report.distro, UNKNOWN,
            "an empty ID means the file could not be read, which is different from a file that \
             named something off the list"
        );
    }

    /// A release with no minor still groups, and a release with an unusual
    /// build separator still yields its release.
    ///
    /// Both are `major_minor`'s fallback paths: a bare major survives rather
    /// than being discarded with a non-numeric minor, and every delimiter the
    /// function accepts, not only `.`, is exercised at least once.
    #[test]
    fn a_release_with_no_minor_or_an_unusual_build_separator_still_groups() {
        let mut bare_major = machine_of_sentinels("gfx1100");
        bare_major.rocm_version = "7".to_owned();
        let report = prepare_report(&bare_major, None, false).expect("a released machine reports");
        assert_eq!(
            report.rocm, "7",
            "a release with no minor is still a population and must not be discarded"
        );

        let mut dash_delimited = machine_of_sentinels("gfx1100");
        dash_delimited.rocm_version = "7-1-0".to_owned();
        let report =
            prepare_report(&dash_delimited, None, false).expect("a released machine reports");
        assert_eq!(
            report.rocm, "7.1",
            "a dash-delimited build string must yield the same release as a dot-delimited one"
        );
    }

    /// A refusal reads as a refusal, not as a report nobody could parse.
    ///
    /// Both refusals are checked, because a reader that recognised only one
    /// would leave the other counted as a reader fault, and the two rules are
    /// the two halves of the disclosure guard.
    ///
    /// The fixture is built by calling [`refusal_envelope`], the same function
    /// `diagnose --report --json` calls to print one, rather than a second,
    /// independent `json!` literal. The two cannot drift apart: either both
    /// change together, through the one function, or neither does.
    #[test]
    fn a_refusal_is_read_as_a_refusal_rather_than_as_an_unreadable_report() {
        for refusal in Refusal::ALL {
            let marker = refusal.marker();
            let envelope = refusal_envelope(*refusal, "why no report was prepared").to_string();

            match read_report(&envelope) {
                ReadOutcome::Refused { reason } => assert_eq!(
                    reason, marker,
                    "the refusal was recognised but its rule was lost, so nothing can tell the \
                     two halves of the guard apart"
                ),
                other => panic!(
                    "a refusal read as {other:?}. Counted that way, the guard firing is \
                     indistinguishable from the reader failing, and the trial that exists to \
                     watch the guard cannot see it."
                ),
            }
        }
    }

    /// A `"refused"` value outside the known markers is not trusted vocabulary.
    ///
    /// The schema version gates the marker set, so a marker this reader does
    /// not recognise cannot have been written by any CLI that speaks this
    /// schema -- it is forged or corrupted, and reading it as a genuine refusal
    /// would carry that text through to a reason field a counter treats as
    /// trusted. Found by mutation: accepting `reason` verbatim, with no check
    /// against [`Refusal::ALL`], passed every other test in this file.
    #[test]
    fn an_unrecognised_refusal_marker_is_not_read_as_a_refusal() {
        let envelope = serde_json::json!({
            "schema": REPORT_SCHEMA_VERSION,
            "refused": "SENTINEL-FORGED-REFUSAL",
            "explanation": "why no report was prepared",
        })
        .to_string();

        assert_eq!(
            read_report(&envelope),
            ReadOutcome::Unread {
                schema_seen: REPORT_SCHEMA_VERSION
            },
            "a marker outside the schema-gated vocabulary must not be trusted as a genuine refusal"
        );
    }

    /// The refusal markers are wire vocabulary, and are pinned as literals.
    ///
    /// The round-trip test above cannot catch a change here: it derives the
    /// value it expects from the same function it is checking, so swapping the
    /// two arms keeps it green. Found by mutation. A swap would attribute
    /// every "hardware not released" refusal to "architecture unreadable" and
    /// the reverse, which is the precise question the trial exists to answer,
    /// so these strings are written out rather than computed.
    #[test]
    fn the_refusal_markers_are_the_strings_already_written_into_the_field() {
        assert_eq!(Refusal::UnreleasedHardware.marker(), "unreleased-hardware");
        assert_eq!(
            Refusal::ArchitectureUnreadable.marker(),
            "architecture-unreadable"
        );
        assert_eq!(
            Refusal::ALL.len(),
            2,
            "a refusal was added without deciding what it is called on the wire"
        );
    }

    /// A report still reads as a report.
    ///
    /// The paired half: a reader that answered `Refused` to everything would
    /// satisfy the test above on its own.
    #[test]
    fn recognising_refusals_did_not_stop_reports_being_read() {
        let report = prepare_report(&machine_of_sentinels("gfx1100"), None, false)
            .expect("a released machine must produce a report");
        let serialized = serde_json::to_string(&report).expect("a report must serialize");

        match read_report(&serialized) {
            ReadOutcome::Understood(read_back) => assert_eq!(*read_back, report),
            other => panic!("a genuine report read as {other:?}"),
        }
    }

    /// An engine name this build does not recognise is not published.
    ///
    /// Today every value of `framework` is a literal written by `examine.rs`,
    /// so this guards a future probe rather than the current one: the moment
    /// one sets that field from parsed output, an allowlist is the difference
    /// between a name and whatever the parse produced. Found by mutation --
    /// removing the check changed nothing, because the shared fixture uses an
    /// approved engine and so never reached the branch.
    #[test]
    fn an_engine_this_build_does_not_recognise_is_not_named_in_the_report() {
        let mut machine = machine_of_sentinels("gfx1100");
        machine.framework = "SENTINEL-UNAPPROVED-ENGINE".to_owned();
        let report = prepare_report(&machine, None, false).expect("a released machine reports");

        assert_eq!(
            report.engine, UNKNOWN,
            "an engine name off the list has to be withheld, not republished"
        );
        assert_eq!(
            report.engine_version, UNKNOWN,
            "a version cannot describe an engine the report declines to name"
        );
        let serialized = serde_json::to_string(&report).expect("a report must serialize");
        assert!(
            !serialized.contains("SENTINEL-UNAPPROVED"),
            "an unrecognised engine name reached a public tracker: {serialized}"
        );
    }

    /// A machine with no engine installed reads as absent, not unreadable.
    ///
    /// `"unknown"` is `Examination::framework`'s struct default, and a
    /// completed probe leaves it there when neither engine was found -- the
    /// ordinary case for most machines. Found by mutation: the prior code
    /// special-cased an empty string here, a value `examine.rs` never
    /// actually writes, so every one of these tests passed against a branch
    /// that could never run, while the case that does run every day fell
    /// through to [`UNKNOWN`] and mislabelled an absence as unreadable.
    #[test]
    fn no_engine_found_by_a_completed_probe_reads_as_none_not_unknown() {
        let mut machine = machine_of_sentinels("gfx1100");
        machine.framework = "unknown".to_owned();
        let report = prepare_report(&machine, None, false).expect("a released machine reports");

        assert_eq!(
            report.engine, NONE,
            "the default a completed probe leaves in place means no engine was found, which is \
             an absence, not a case where this build looked and could not tell"
        );
        assert_eq!(
            report.engine_version, NONE,
            "a version cannot describe an engine the report says is absent"
        );
    }

    /// A skipped probe reads as unreadable, not absent.
    ///
    /// `"skipped"` means [`crate::examine::FrameworkProbe::Skip`] was
    /// requested and the probe never ran at all, which is a different fact
    /// from the probe running and finding nothing: this build did not look,
    /// so it cannot say the engine is absent.
    #[test]
    fn a_skipped_probe_reads_as_unreadable_not_as_no_engine_installed() {
        let mut machine = machine_of_sentinels("gfx1100");
        machine.framework = "skipped".to_owned();
        let report = prepare_report(&machine, None, false).expect("a released machine reports");

        assert_eq!(
            report.engine, UNKNOWN,
            "a probe that never ran cannot report an absence; that would say a machine has no \
             engine when this build simply never looked"
        );
        assert_eq!(
            report.engine_version, UNKNOWN,
            "a version cannot describe an engine this build never checked for"
        );
    }

    /// An absent ROCm and an unreadable one are different facts.
    ///
    /// `examine.rs` collapses both into an empty `rocm_version`, so without
    /// this the report would say "not installed" about a machine whose install
    /// merely could not be read -- turning an install problem into an absence
    /// for anybody counting.
    #[test]
    fn no_rocm_installed_reads_differently_from_a_rocm_that_could_not_be_read() {
        let mut absent = machine_of_sentinels("gfx1100");
        absent.rocm_path = String::new();
        absent.rocm_version = String::new();
        let absent = prepare_report(&absent, None, false).expect("a released machine reports");

        let mut unreadable = machine_of_sentinels("gfx1100");
        unreadable.rocm_version = String::new();
        let unreadable =
            prepare_report(&unreadable, None, false).expect("a released machine reports");

        assert_eq!(absent.rocm, NONE);
        assert_eq!(unreadable.rocm, UNKNOWN);
        assert_ne!(
            absent.rocm, unreadable.rocm,
            "a counter cannot tell an absent ROCm from an unreadable one"
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
    ///
    /// Also pins `fix_offered`, the field derived from `entry_recognised`: a
    /// fix cannot be offered for a cause the catalog did not establish, so a
    /// caller asking for `fix_offered: true` alongside a forged id must be
    /// overruled. Both calls below pass `true`, so the paired assertion cannot
    /// be satisfied by an implementation that forces the flag `false`
    /// unconditionally -- the real-id case has to show the flag surviving.
    #[test]
    fn an_entry_id_the_catalog_does_not_know_is_never_published_verbatim() {
        // Non-vacuity: a real id has to reach the report, or "the forged one
        // does not" is satisfied by discarding every id.
        let known = prepare_report(&machine_of_sentinels("gfx1100"), Some("fix-6-path"), true)
            .expect("a released machine must produce a report");
        assert_eq!(
            known.entry, "fix-6-path",
            "premise failed: a real catalog id must reach the report, otherwise the assertion \
             below passes against an implementation that publishes no id at all"
        );
        assert!(
            known.fix_offered,
            "premise failed: fix_offered must survive for a recognised entry, otherwise the \
             assertion below passes against an implementation that forces the flag false \
             unconditionally"
        );

        let forged = prepare_report(
            &machine_of_sentinels("gfx1100"),
            Some("SENTINEL-FORGED-ENTRY"),
            true,
        )
        .expect("a released machine must produce a report");
        assert_eq!(
            forged.entry, UNRECOGNISED,
            "an id the catalog does not know is not a finding, and publishing it verbatim would \
             put caller-supplied text on a public tracker"
        );
        assert!(
            !forged.fix_offered,
            "a fix cannot be offered for a cause the catalog did not establish; this would \
             publish a fix pointer next to an entry the report itself calls unrecognised"
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
