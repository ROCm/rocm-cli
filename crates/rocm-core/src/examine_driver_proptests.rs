// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the small text readers `examine` uses to judge the
//! driver side of a host — the WSL distro floor, device-node permissions, the
//! modprobe blacklist and the `iommu=` kernel parameter — and for the install
//! family a reported gfx target normalizes to.
//!
//! Each oracle is stated independently of the code under test: a monotone
//! floor, POSIX class precedence, the modprobe.d grammar, the kernel's
//! whitespace-separated cmdline tokens, and the set of families the CLI
//! recognizes. Inputs are drawn from small real-shaped alphabets rather than
//! random text, so every branch is reached; `inputs_reach_every_region` checks
//! that.

use super::{
    WSL_MIN_UBUNTU, distro_clears_wsl_floor, line_blacklists_amdgpu, mode_access, parse_iommu_param,
};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Case count: `default`, unless `PROPTEST_CASES` (proptest's own knob, which
/// an explicit count would otherwise override) asks for a longer run.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// gfx targets a real host reports, as `examine` and the install probes see
/// them.
const GFX_TARGETS: &[&str] = &[
    "gfx900", "gfx906", "gfx908", "gfx90a", "gfx90c", "gfx942", "gfx950", "gfx1010", "gfx1030",
    "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036", "gfx1100", "gfx1101",
    "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1200", "gfx1201",
];

fn ubuntu_release() -> impl Strategy<Value = (u32, u32)> {
    (14u32..40, prop_oneof![Just(4u32), Just(10u32)])
}

/// A `stat -c %A` mode string (file type, nine permission bits, optional ACL
/// marker) and whether the user is the owner and in the owning group.
fn mode_case() -> impl Strategy<Value = (String, Vec<char>, bool, bool)> {
    (
        prop::sample::select(&['c', '-', 'b'][..]),
        prop::collection::vec(prop::sample::select(&['r', 'w', 'x', '-'][..]), 9..=9),
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(|(kind, bits, owner_is_user, in_owner_group, acl)| {
            let mut mode: String = std::iter::once(kind).chain(bits.iter().copied()).collect();
            if acl {
                mode.push('+');
            }
            (mode, bits, owner_is_user, in_owner_group)
        })
}

/// A modprobe.d line built around the `blacklist` directive. The separator is
/// never empty here: `blacklistamdgpu` is #506, pinned separately below.
fn blacklist_line() -> impl Strategy<Value = (String, bool)> {
    (
        prop::sample::select(&["", " ", "\t", "   ", "#", "# "][..]),
        prop::sample::select(&[" ", "\t", "  "][..]),
        prop::sample::select(
            &[
                "amdgpu",
                "amdgpu_foo",
                "amdgpufoo",
                "amdgpu2",
                "radeon",
                "amdkfd",
            ][..],
        ),
        prop::sample::select(&["", " ", " # why", "\t"][..]),
    )
        .prop_map(|(lead, sep, module, trail)| {
            // A real amdgpu blacklist: no comment marker before the directive,
            // and the module is exactly `amdgpu`.
            let expected = !lead.contains('#') && module == "amdgpu";
            (format!("{lead}blacklist{sep}{module}{trail}"), expected)
        })
}

/// A kernel cmdline with at most one `iommu=` token, among parameters whose
/// names merely end in `iommu`. A cmdline with two `iommu=` tokens is left out
/// on purpose: which one takes effect is the kernel's call, not a property of
/// tokenization.
fn iommu_cmdline() -> impl Strategy<Value = (String, Option<String>)> {
    let distractors = || {
        prop::collection::vec(
            prop::sample::select(
                &[
                    "BOOT_IMAGE=/vmlinuz",
                    "quiet",
                    "splash",
                    "nomodeset",
                    "amd_iommu=on",
                    "intel_iommu=off",
                    "amd_iommu=pt",
                ][..],
            ),
            0..4,
        )
    };
    (
        distractors(),
        prop::option::of(prop::sample::select(&["pt", "soft", "force", "off"][..])),
        distractors(),
    )
        .prop_map(|(before, value, after)| {
            let mut tokens: Vec<String> = before.iter().map(|t| (*t).to_owned()).collect();
            if let Some(value) = value {
                tokens.push(format!("iommu={value}"));
            }
            tokens.extend(after.iter().map(|t| (*t).to_owned()));
            (tokens.join(" "), value.map(str::to_owned))
        })
}

/// A gfx target as a probe might print it: any case, with an `:xnack-` feature
/// suffix or trailing whitespace.
fn spelled_target() -> impl Strategy<Value = String> {
    (
        prop::sample::select(GFX_TARGETS),
        prop::sample::select(&["", ":xnack-", " ", "\t"][..]),
        any::<bool>(),
    )
        .prop_map(|(target, suffix, upper)| {
            let target = if upper {
                target.to_uppercase()
            } else {
                target.to_owned()
            };
            format!("{target}{suffix}")
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(256)))]

    /// The WSL distro floor is monotone: a newer Ubuntu release is never less
    /// supported than an older one.
    #[test]
    fn the_wsl_distro_floor_is_monotone(a in ubuntu_release(), b in ubuntu_release()) {
        let clears = |(major, minor): (u32, u32)| {
            distro_clears_wsl_floor("ubuntu", &format!("{major}.{minor:02}"))
        };
        let (Some(older), Some(newer)) = (clears(a.min(b)), clears(a.max(b))) else {
            return Err(TestCaseError::fail(format!("{a:?} or {b:?} did not parse")));
        };
        prop_assert!(
            !older || newer,
            "{:?} clears the floor but the newer {:?} does not",
            a.min(b),
            a.max(b),
        );
    }

    /// `mode_access` follows POSIX class precedence — owner, then group, then
    /// other — read independently from the same `stat -c %A` string.
    #[test]
    fn mode_access_follows_posix_precedence(
        (mode, bits, owner_is_user, in_owner_group) in mode_case(),
    ) {
        let groups = if in_owner_group {
            vec!["render".to_owned()]
        } else {
            vec!["users".to_owned()]
        };
        let user = if owner_is_user { "root" } else { "alice" };
        let (read, write) = mode_access(&mode, "root", "render", user, &groups);
        let base = if owner_is_user {
            0
        } else if in_owner_group {
            3
        } else {
            6
        };
        prop_assert_eq!(read, Some(bits[base] == 'r'), "read class for {}", mode);
        prop_assert_eq!(write, Some(bits[base + 1] == 'w'), "write class for {}", mode);
    }

    /// `blacklist amdgpu` is recognized per the modprobe.d grammar: not when
    /// commented out, and not for a different module whose name starts with
    /// `amdgpu`.
    #[test]
    fn amdgpu_blacklist_recognition_follows_the_modprobe_grammar(
        (line, expected) in blacklist_line(),
    ) {
        prop_assert_eq!(
            line_blacklists_amdgpu(&line),
            expected,
            "line_blacklists_amdgpu({:?})",
            line,
        );
    }

    /// `iommu=` is read from its own whitespace-separated cmdline token and
    /// never from a different parameter whose name ends in `iommu`.
    #[test]
    fn iommu_param_is_read_from_its_own_token((cmdline, expected) in iommu_cmdline()) {
        prop_assert_eq!(parse_iommu_param(&cmdline), expected, "cmdline {:?}", cmdline);
    }

    /// Every spelling of a target a host reports normalizes to a family the
    /// CLI recognizes, and normalizing that family again changes nothing.
    #[test]
    fn a_reported_target_normalizes_to_a_known_family(spelled in spelled_target()) {
        let family = crate::normalize_therock_family(&spelled);
        prop_assert!(family.is_some(), "{:?} normalizes to no family", spelled);
        let family = family.unwrap_or_default();
        prop_assert!(
            crate::known_therock_families().contains(&family.as_str()),
            "{} (from {:?}) is not in known_therock_families()",
            family,
            spelled,
        );
        let again = crate::normalize_therock_family(&family);
        prop_assert_eq!(
            again.as_deref(),
            Some(family.as_str()),
            "normalizing {} again changed it",
            family,
        );
    }
}

/// #506: `blacklistamdgpu`, with no whitespace after the directive, is not a
/// modprobe directive, yet it is read as one. The fix for #506 should
/// un-ignore this, and can then let `blacklist_line` draw an empty separator.
#[test]
#[ignore = "known defect: a directive with no whitespace before the module is read as a blacklist (#506)"]
fn a_blacklist_directive_needs_whitespace_before_the_module() {
    for line in ["blacklistamdgpu", "  blacklistamdgpu # why"] {
        assert!(!line_blacklists_amdgpu(line), "{line:?}");
    }
}

/// The properties above pass vacuously if their inputs never reach the
/// interesting side of each oracle, so draw from the same strategies and
/// require every region to be sampled.
#[test]
fn inputs_reach_every_region() {
    const DRAWS: usize = 2048;
    let mut runner = TestRunner::deterministic();
    let mut draw = |strategy: &dyn Fn(&mut TestRunner) -> [bool; 2]| {
        let mut hits = [0usize; 2];
        for _ in 0..DRAWS {
            let [first, second] = strategy(&mut runner);
            hits[0] += usize::from(first);
            hits[1] += usize::from(second);
        }
        hits
    };

    // A pair on opposite sides of the floor, and a pair both above it.
    let floor = draw(&|runner| {
        let (a, b) = (ubuntu_release(), ubuntu_release())
            .new_tree(runner)
            .expect("ubuntu strategy")
            .current();
        let above = |release: (u32, u32)| release >= WSL_MIN_UBUNTU;
        [above(a) != above(b), above(a) && above(b)]
    });
    // Owner and group classes both decide some draws (other is the rest).
    let mode = draw(&|runner| {
        let (_, _, owner, group) = mode_case().new_tree(runner).expect("mode").current();
        [owner, !owner && group]
    });
    let blacklist = draw(&|runner| {
        let (_, expected) = blacklist_line()
            .new_tree(runner)
            .expect("blacklist")
            .current();
        [expected, !expected]
    });
    // A real `iommu=` value, and a cmdline with only look-alike parameters.
    let iommu = draw(&|runner| {
        let (cmdline, expected) = iommu_cmdline().new_tree(runner).expect("iommu").current();
        [
            expected.is_some(),
            expected.is_none() && cmdline.contains("iommu="),
        ]
    });
    let family = draw(&|runner| {
        let spelled = spelled_target().new_tree(runner).expect("target").current();
        [
            spelled.contains(':'),
            spelled.chars().any(char::is_uppercase),
        ]
    });

    eprintln!(
        "reach over {DRAWS} draws: wsl-floor straddled={} both-above={}; \
         mode owner={} group={}; blacklist positive={} negative={}; \
         iommu present={} look-alike-only={}; target suffixed={} uppercase={}",
        floor[0],
        floor[1],
        mode[0],
        mode[1],
        blacklist[0],
        blacklist[1],
        iommu[0],
        iommu[1],
        family[0],
        family[1],
    );
    for (region, hits) in [
        ("wsl floor straddled", floor[0]),
        ("wsl floor both above", floor[1]),
        ("mode owner class", mode[0]),
        ("mode group class", mode[1]),
        ("blacklist positive", blacklist[0]),
        ("blacklist negative", blacklist[1]),
        ("iommu present", iommu[0]),
        ("iommu look-alike only", iommu[1]),
        ("target with feature suffix", family[0]),
        ("target in upper case", family[1]),
    ] {
        assert!(
            hits * 20 >= DRAWS,
            "{region} reached in only {hits} of {DRAWS} draws"
        );
    }
}
