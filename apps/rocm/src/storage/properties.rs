// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property-based tests for the retention policy in [`super`].
//!
//! The policy decides which multi-gigabyte install folders get deleted, so the
//! interesting failures are the ones a hand-written example misses: pointers
//! spelled in a different case, two installs whose keys differ only in case,
//! a whole family sharing one runtime id, tied install times, a pointer naming
//! something that is not installed, and `--keep 0`.
//!
//! The generators therefore draw from deliberately tiny alphabets. A uniform
//! generator over fresh random strings never collides, so it would sample none
//! of that and pass against a defect that is plainly there.
//! `generator_reach_report` measures how often those cases come up and fails
//! if the generator stops reaching them.
//!
//! The expected groups and holds are worked out here rather than by calling the
//! policy's own helpers, so a defect in one of those helpers cannot hide
//! itself. They follow the `--keep` help and the `HoldReason` descriptions, plus
//! the rules the policy implements in its own code: pointers and group fields
//! match case-insensitively, and the default holds an install only when it
//! names exactly one. A shared design mistake in those rules would not be
//! caught here; the hand-written examples in `storage::tests` pin them.
//!
//! `therock::version_proptests` checks further properties of the same policy:
//! that it ignores the order the registry lists installs in, and that
//! "recent" means most recently installed rather than highest version.

use proptest::prelude::*;

use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

use super::{HoldReason, RetentionInputs, select_runtimes_to_remove};
use crate::therock::{InstalledRuntimeManifest, WheelRuntimeComposition};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Ten keys over eight letters. Two letters come in both cases, so case
/// twins occur within a handful of generated sets; the eight letters mean a
/// set usually has more installs than its few pointers name, so recency has
/// something left to decide about. (Exact duplicates are drawn too, but
/// `manifests()` drops them: the registry holds one file per key.)
const KEYS: &[&str] = &["a", "A", "b", "B", "c", "d", "e", "f", "g", "h"];
/// Repeated entries weight the draw. Most installs land in one channel, format
/// and family, so a retention group usually has several candidates and
/// recency actually has something to decide; the other spellings and values
/// keep case-folding and separate groups in play.
const CHANNELS: &[&str] = &["release", "release", "release", "Release", "nightly"];
const FORMATS: &[&str] = &["wheel", "wheel", "wheel", "Wheel", "tarball"];
const FAMILIES: &[&str] = &["gfx110X-all", "gfx110X-all", "gfx110x-all", "gfx120X-all"];

fn pick(options: &'static [&'static str]) -> impl Strategy<Value = String> {
    proptest::sample::select(options).prop_map(str::to_owned)
}

/// A key that may name an installed runtime, may differ only in case, and may
/// name nothing at all (`"z"` is never installed).
fn pointer_key() -> impl Strategy<Value = String> {
    prop_oneof![
        5 => pick(KEYS),
        3 => pick(KEYS).prop_map(|key| key.to_ascii_uppercase()),
        3 => pick(KEYS).prop_map(|key| key.to_ascii_lowercase()),
        1 => Just("z".to_owned()),
    ]
}

/// How a runtime records its toolchain choice.
///
/// Both spellings are real: `rocm install sdk` writes the package specs it
/// handed `uv`, while older records only carry the `devel` flag. When a record
/// has both, the specs say what was actually installed, so a record whose flag
/// disagrees with them is generated too: it tells apart a policy that reads the
/// specs from one that only reads the flag. Extras are always generated in
/// lower case and every `rocm` requirement is spelled `rocm[...]`, so how the
/// policy treats `Devel` or a requirement on another `rocm-*` package is not
/// covered here.
#[derive(Debug, Clone, Copy)]
enum Toolchain {
    /// A `rocm[<extras>]` requirement plus the flag, which may disagree.
    Specs {
        extras: &'static str,
        flag: bool,
    },
    /// A `torch[cuda]` requirement first, then optionally a `rocm[<extras>]`
    /// one: only a requirement on `rocm` itself says anything about the
    /// toolchain, and without one only the flag does.
    OtherSpecFirst {
        rocm_extras: Option<&'static str>,
        flag: bool,
    },
    FlagOnly(bool),
}

const DEVEL_EXTRAS: &str = "libraries,devel,device-gfx1100";
const PLAIN_EXTRAS: &str = "libraries,device-gfx1100";
/// Still the toolchain: extras are trimmed before they are compared.
const SPACED_DEVEL_EXTRAS: &str = "libraries, devel ,device-gfx1100";
/// Not the toolchain: an extra that merely starts with `devel`.
const NEAR_MISS_EXTRAS: &str = "libraries,develop,device-gfx1100";

/// Weighted towards `--devel` for the same reason the channel, format and
/// family alphabets repeat their first entry: so that groups fill up.
fn toolchain() -> impl Strategy<Value = Toolchain> {
    prop_oneof![
        3 => Just(Toolchain::Specs { extras: DEVEL_EXTRAS, flag: true }),
        1 => Just(Toolchain::Specs { extras: PLAIN_EXTRAS, flag: false }),
        1 => Just(Toolchain::Specs { extras: DEVEL_EXTRAS, flag: false }),
        1 => Just(Toolchain::Specs { extras: PLAIN_EXTRAS, flag: true }),
        1 => Just(Toolchain::Specs { extras: SPACED_DEVEL_EXTRAS, flag: false }),
        1 => Just(Toolchain::Specs { extras: NEAR_MISS_EXTRAS, flag: true }),
        1 => Just(Toolchain::OtherSpecFirst { rocm_extras: None, flag: true }),
        1 => Just(Toolchain::OtherSpecFirst { rocm_extras: Some(PLAIN_EXTRAS), flag: true }),
        2 => Just(Toolchain::FlagOnly(true)),
        1 => Just(Toolchain::FlagOnly(false)),
    ]
}

#[derive(Debug, Clone)]
struct GeneratedManifest {
    runtime_key: String,
    channel: String,
    format: String,
    family: String,
    installed_at_unix_ms: u128,
    read_only: bool,
    imported: bool,
    toolchain: Toolchain,
}

fn generated_manifest() -> impl Strategy<Value = GeneratedManifest> {
    (
        pick(KEYS),
        pick(CHANNELS),
        pick(FORMATS),
        pick(FAMILIES),
        0_u128..4,
        // Adopted and imported installs are the minority on a real machine.
        prop_oneof![9 => Just(false), 1 => Just(true)],
        prop_oneof![9 => Just(false), 1 => Just(true)],
        toolchain(),
    )
        .prop_map(
            |(
                runtime_key,
                channel,
                format,
                family,
                installed_at_unix_ms,
                read_only,
                imported,
                toolchain,
            )| GeneratedManifest {
                runtime_key,
                channel,
                format,
                family,
                installed_at_unix_ms,
                read_only,
                imported,
                toolchain,
            },
        )
}

impl GeneratedManifest {
    fn build(&self) -> InstalledRuntimeManifest {
        let version = format!("7.{}.0", self.installed_at_unix_ms);
        let (devel, wheel_composition) = match self.toolchain {
            // `rocm install sdk --devel` puts the `devel` extra in the specs;
            // plain `rocm install sdk` leaves it out: same family, no compiler.
            Toolchain::Specs { extras, flag } => (
                flag,
                Some(WheelRuntimeComposition {
                    source_layout_generation: "canonical".to_owned(),
                    package_specs: vec![format!("rocm[{extras}]=={version}")],
                    rocm_sdk_target: Some("gfx1100".to_owned()),
                }),
            ),
            Toolchain::OtherSpecFirst { rocm_extras, flag } => {
                let mut package_specs = vec!["torch[cuda]==2.9.0".to_owned()];
                package_specs
                    .extend(rocm_extras.map(|extras| format!("rocm[{extras}]=={version}")));
                (
                    flag,
                    Some(WheelRuntimeComposition {
                        source_layout_generation: "canonical".to_owned(),
                        package_specs,
                        rocm_sdk_target: Some("gfx1100".to_owned()),
                    }),
                )
            }
            Toolchain::FlagOnly(devel) => (devel, None),
        };
        InstalledRuntimeManifest {
            runtime_key: self.runtime_key.clone(),
            // The shape `rocm install sdk` writes: channel and family, never a
            // version, so a whole family shares one id.
            runtime_id: format!("therock-{}:{}", self.channel, self.family),
            channel: self.channel.clone(),
            format: self.format.clone(),
            family: self.family.clone(),
            family_source: "test".to_owned(),
            version,
            install_root: PathBuf::from("rocm-storage-proptest").join(&self.runtime_key),
            selected_artifact_url: "https://example.invalid/therock".to_owned(),
            source_layout_generation: None,
            index_url: None,
            tarball_file_name: None,
            python_launcher: None,
            python_executable: None,
            pip_cache_dir: None,
            rocm_sdk: None,
            sdk_torch: None,
            wheel_composition,
            read_only: self.read_only,
            imported_from: self
                .imported
                .then(|| PathBuf::from("elsewhere").join("manifest.json")),
            devel,
            installed_at_unix_ms: self.installed_at_unix_ms,
        }
    }
}

/// The registry stores one file per runtime (`<runtime_key>.json`), so two
/// records never carry the byte-identical key. Keys that differ only in case
/// *are* two files on a case-sensitive filesystem, and `rocm runtimes adopt
/// --runtime-key` passes a user-typed key straight through without
/// slugifying it, so those are kept in the sample.
fn manifests() -> impl Strategy<Value = Vec<InstalledRuntimeManifest>> {
    proptest::collection::vec(generated_manifest(), 0..12).prop_map(|generated| {
        let mut seen: Vec<String> = Vec::new();
        generated
            .iter()
            .filter(|candidate| {
                let fresh = !seen.contains(&candidate.runtime_key);
                if fresh {
                    seen.push(candidate.runtime_key.clone());
                }
                fresh
            })
            .map(GeneratedManifest::build)
            .collect()
    })
}

fn retention_inputs() -> impl Strategy<Value = RetentionInputs> {
    (
        proptest::option::of(pointer_key()),
        proptest::option::of(pointer_key()),
        proptest::option::of(
            (pick(CHANNELS), pick(FAMILIES))
                .prop_map(|(channel, family)| format!("therock-{channel}:{family}")),
        ),
        // The marker is a second source of truth and may disagree with config.
        proptest::collection::vec(pointer_key(), 0..3),
    )
        .prop_map(
            |(
                active_runtime_key,
                previous_runtime_key,
                default_runtime_id,
                marker_runtime_keys,
            )| {
                RetentionInputs {
                    active_runtime_key,
                    previous_runtime_key,
                    default_runtime_id,
                    marker_runtime_keys,
                }
            },
        )
}

/// Keys that `eq_ignore_ascii_case`-match `pointer`, as the rest of the runtime
/// code compares them.
fn installed_matching(
    manifests: &[InstalledRuntimeManifest],
    pointer: Option<&str>,
) -> Vec<String> {
    let Some(pointer) = pointer else {
        return Vec::new();
    };
    manifests
        .iter()
        .filter(|manifest| manifest.runtime_key.eq_ignore_ascii_case(pointer))
        .map(|manifest| manifest.runtime_key.clone())
        .collect()
}

/// The install the configured default names: the default id carries no
/// version, so it only names an install when exactly one install has that id
/// (the rule `resolved_default_runtime_key` documents, re-derived here).
fn expected_default_key(
    manifests: &[InstalledRuntimeManifest],
    inputs: &RetentionInputs,
) -> Option<String> {
    let id = inputs.default_runtime_id.as_deref()?;
    let matching: Vec<&InstalledRuntimeManifest> = manifests
        .iter()
        .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(id))
        .collect();
    match matching.as_slice() {
        [only] => Some(only.runtime_key.clone()),
        _ => None,
    }
}

/// Whether anything other than recency keeps `manifest`: it is in use, the
/// rollback target, the default, named by the marker, or not ROCm CLI's.
fn held_regardless_of_recency(
    manifest: &InstalledRuntimeManifest,
    manifests: &[InstalledRuntimeManifest],
    inputs: &RetentionInputs,
) -> bool {
    let key = &manifest.runtime_key;
    installed_matching(manifests, inputs.active_runtime_key.as_deref()).contains(key)
        || installed_matching(manifests, inputs.previous_runtime_key.as_deref()).contains(key)
        || installed_matching(
            manifests,
            expected_default_key(manifests, inputs).as_deref(),
        )
        .contains(key)
        || inputs
            .marker_runtime_keys
            .iter()
            .any(|marker| marker.eq_ignore_ascii_case(key))
        || manifest.read_only
        || manifest.imported_from.is_some()
}

/// Whether the install carries the compiler toolchain: the `devel` extra in
/// the recorded `rocm[...]` package spec when there is one, otherwise the
/// `devel` flag (older records only have the flag).
fn installed_the_toolchain(manifest: &InstalledRuntimeManifest) -> bool {
    let recorded_extras = manifest.wheel_composition.as_ref().and_then(|composition| {
        composition.package_specs.iter().find_map(|spec| {
            let rest = spec.strip_prefix("rocm[")?;
            Some(rest.split_once(']')?.0.to_owned())
        })
    });
    match recorded_extras {
        Some(extras) => extras.split(',').any(|extra| extra.trim() == "devel"),
        None => manifest.devel,
    }
}

/// The bucket `--keep` counts in. The `--keep` help: "How many recent installs
/// to keep for each channel, format, GPU family, and toolchain choice."
/// Channel, format and family are compared case-insensitively, as the policy
/// compares them.
type Group = (String, String, String, bool);

fn expected_group(manifest: &InstalledRuntimeManifest) -> Group {
    (
        manifest.channel.to_ascii_lowercase(),
        manifest.format.to_ascii_lowercase(),
        manifest.family.to_ascii_lowercase(),
        installed_the_toolchain(manifest),
    )
}

/// Installs that only recency decides about, bucketed by [`expected_group`].
fn recency_groups<'a>(
    manifests: &'a [InstalledRuntimeManifest],
    inputs: &RetentionInputs,
) -> BTreeMap<Group, Vec<&'a InstalledRuntimeManifest>> {
    let mut groups: BTreeMap<Group, Vec<&InstalledRuntimeManifest>> = BTreeMap::new();
    for manifest in manifests {
        if !held_regardless_of_recency(manifest, manifests, inputs) {
            groups
                .entry(expected_group(manifest))
                .or_default()
                .push(manifest);
        }
    }
    groups
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// The documented hard safety invariant, from the `--keep` help: "The one
    /// in use and the rollback target are always kept on top of this count,
    /// whatever it is set to."
    #[test]
    fn the_install_in_use_is_never_selected_for_removal(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        for key in installed_matching(&manifests, inputs.active_runtime_key.as_deref()) {
            prop_assert!(
                !removable.contains(&key),
                "active runtime {key:?} selected for removal (active_runtime_key={:?}, \
                 keep={keep}, removable={removable:?})",
                inputs.active_runtime_key,
            );
        }
    }

    #[test]
    fn the_rollback_target_is_never_selected_for_removal(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        for key in installed_matching(&manifests, inputs.previous_runtime_key.as_deref()) {
            prop_assert!(
                !removable.contains(&key),
                "rollback target {key:?} selected for removal (previous_runtime_key={:?}, \
                 keep={keep}, removable={removable:?})",
                inputs.previous_runtime_key,
            );
        }
    }

    /// The marker is the second source of truth and may disagree with config.
    #[test]
    fn an_install_named_by_the_marker_is_never_selected_for_removal(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        for marker_key in &inputs.marker_runtime_keys {
            for key in installed_matching(&manifests, Some(marker_key)) {
                prop_assert!(
                    !removable.contains(&key),
                    "marked runtime {key:?} selected for removal \
                     (marker={marker_key:?}, keep={keep}, removable={removable:?})",
                );
            }
        }
    }

    /// Adopted and imported folders are not ROCm CLI's to delete.
    #[test]
    fn an_install_rocm_cli_does_not_own_is_never_selected_for_removal(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        for manifest in &manifests {
            if manifest.read_only || manifest.imported_from.is_some() {
                prop_assert!(
                    !removable.contains(&manifest.runtime_key),
                    "unowned runtime {:?} selected for removal",
                    manifest.runtime_key,
                );
            }
        }
    }

    /// Every install is accounted for exactly once, so nothing is silently
    /// dropped from the review the user confirms against, and no install is
    /// listed as both kept and removed.
    #[test]
    fn every_install_is_either_kept_or_removed_exactly_once(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, held) = select_runtimes_to_remove(&manifests, &inputs, keep);
        let mut accounted: Vec<String> = removable;
        accounted.extend(held.into_iter().map(|(key, _)| key));
        accounted.sort();
        let mut expected: Vec<String> = manifests
            .iter()
            .map(|manifest| manifest.runtime_key.clone())
            .collect();
        expected.sort();
        prop_assert_eq!(accounted, expected);
    }

    /// Raising `--keep` must never delete more. A user who is unsure and picks
    /// a bigger number must not be punished for it.
    #[test]
    fn removal_is_monotone_in_keep(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (fewer, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        let (more, _) = select_runtimes_to_remove(&manifests, &inputs, keep + 1);
        for key in &more {
            prop_assert!(
                fewer.contains(key),
                "--keep {} removes {key:?} but --keep {keep} does not",
                keep + 1,
            );
        }
    }

    /// Within each group, the `--keep` most recent installs that nothing else
    /// holds are kept and every older one is removed: `--keep` is the exact
    /// count, so a policy that keeps too much fails here as surely as one that
    /// deletes too much.
    #[test]
    fn each_group_keeps_its_most_recent_installs_and_removes_the_rest(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        for (group, members) in recency_groups(&manifests, &inputs) {
            let (removed, kept): (Vec<&InstalledRuntimeManifest>, Vec<_>) = members
                .iter()
                .copied()
                .partition(|manifest| removable.contains(&manifest.runtime_key));
            prop_assert_eq!(
                removed.len(),
                members.len().saturating_sub(keep),
                "group {:?} removed {} of {} with --keep {}",
                group,
                removed.len(),
                members.len(),
                keep,
            );
            for gone in &removed {
                for stays in &kept {
                    // Newest first; between two installed at the same moment
                    // the policy documents that the key decides, so the
                    // later key in sort order is the one that goes.
                    prop_assert!(
                        (gone.installed_at_unix_ms, &stays.runtime_key)
                            < (stays.installed_at_unix_ms, &gone.runtime_key),
                        "removed {:?} (installed at {}) but kept {:?} ({})",
                        gone.runtime_key,
                        gone.installed_at_unix_ms,
                        stays.runtime_key,
                        stays.installed_at_unix_ms,
                    );
                }
            }
        }
    }

    /// The policy hands the deleter a *string*, not the manifest it decided
    /// about: `build_prune_plan` resolves that string with `==`, and the
    /// removal it delegates to resolves it again through
    /// `select_runtime_manifest`. If those two land on different records, the
    /// removal acts on an install the policy never cleared, so each selected
    /// key must resolve to itself.
    ///
    /// The real resolver is called here rather than a copy of its matching
    /// rule, so this keeps holding whatever that rule becomes. When the two
    /// disagreed, the failing cases were two installs whose keys differ only
    /// in letter case; the checked-in seed replays one with `--keep 0`.
    #[test]
    fn the_deleter_resolves_each_selected_key_to_the_install_the_policy_cleared(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (removable, _) = select_runtimes_to_remove(&manifests, &inputs, keep);
        // `load_runtime_manifests` hands every caller this order: newest
        // install first, ties broken by key.
        let mut newest_first = manifests;
        newest_first.sort_by(|left, right| {
            right
                .installed_at_unix_ms
                .cmp(&left.installed_at_unix_ms)
                .then_with(|| left.runtime_key.cmp(&right.runtime_key))
        });

        for key in &removable {
            let resolved = crate::runtimes_cmd::select_runtime_manifest(&newest_first, key)
                .expect("a selected key always names exactly one installed runtime");
            prop_assert_eq!(
                &resolved.runtime_key,
                key,
                "the plan says remove {:?}, but the deleter resolves that key to {:?}",
                key,
                resolved.runtime_key,
            );
        }
    }

    /// Each kept install is reported with the right reason: the first one
    /// that applies, in the order `HoldReason` declares them (the policy keeps
    /// them "most-specific first so the reported reason is the most useful
    /// one"). Pointers match case-insensitively, so a case twin of the install
    /// in use is held as Active too; the default only counts when it names
    /// exactly one install. A wrong reason sends a user who is about to free
    /// disk space looking at the wrong install.
    #[test]
    fn every_hold_reason_is_the_first_that_applies(
        manifests in manifests(),
        inputs in retention_inputs(),
        keep in 0_usize..4,
    ) {
        let (_, held) = select_runtimes_to_remove(&manifests, &inputs, keep);
        let groups = recency_groups(&manifests, &inputs);
        let default_key = expected_default_key(&manifests, &inputs);
        for (key, reason) in &held {
            let manifest = manifests
                .iter()
                .find(|manifest| &manifest.runtime_key == key)
                .expect("a held key always names an installed runtime");
            let expected = if installed_matching(&manifests, inputs.active_runtime_key.as_deref())
                .contains(key)
            {
                Some(HoldReason::Active)
            } else if installed_matching(&manifests, inputs.previous_runtime_key.as_deref())
                .contains(key)
            {
                Some(HoldReason::Previous)
            } else if installed_matching(&manifests, default_key.as_deref()).contains(key) {
                Some(HoldReason::Default)
            } else if inputs
                .marker_runtime_keys
                .iter()
                .any(|marker| marker.eq_ignore_ascii_case(key))
            {
                Some(HoldReason::Marker)
            } else if manifest.read_only || manifest.imported_from.is_some() {
                Some(HoldReason::NotOwned)
            } else if groups[&expected_group(manifest)]
                .iter()
                .filter(|other| other.installed_at_unix_ms > manifest.installed_at_unix_ms)
                .count()
                < keep
            {
                Some(HoldReason::WithinKeepLimit)
            } else {
                None
            };
            prop_assert_eq!(
                Some(reason),
                expected.as_ref(),
                "{:?} held as {:?}",
                key,
                reason,
            );
        }
    }
}

/// Printed with `cargo test -- --nocapture`. This measures the generators, not
/// the policy: it fails only if they stop reaching the cases the properties
/// above depend on. Several of those properties assert inside a loop over
/// pointers, groups or selected keys, and would pass vacuously on a generator
/// that never fills the loop, so every condition about how installs are grouped
/// or ordered is measured among the installs only recency decides about, inside
/// one retention group, not just anywhere in the set.
///
/// The RNG is the fixed deterministic one, so the counts are the same on every
/// run. The floors in [`FLOOR`] sit at about four fifths of them; any change to
/// the generators shifts every count, so re-measure them all after one.
#[test]
fn generator_reach_report() {
    const SAMPLES: u32 = 20_000;
    let mut runner = TestRunner::deterministic();
    let draw = (manifests(), retention_inputs());

    let labels: [&str; ROWS] = [
        "two keys differing only in case",
        "two installs sharing a runtime id",
        "an adopted or imported install",
        "a pointer spelled in another case",
        "a pointer naming nothing installed",
        "a default naming exactly one install",
        "a default naming several installs",
        "in a group: tied install times",
        "in a group: case-twin keys",
        "in a group: channel spelled two ways",
        "in a group: format spelled two ways",
        "in a group: family spelled two ways",
        "in a group: devel flag disagreeing with the specs",
        "in a group: a spaced devel extra",
        "in a group: a near-miss devel extra",
        "in a group: another package's spec first",
        "two groups differing only in the toolchain",
        "a group with more candidates than --keep 1",
        "a group with more candidates than --keep 2",
        "a group with more candidates than --keep 3",
    ];
    let mut hits = [0_u32; ROWS];
    for _ in 0..SAMPLES {
        let (manifests, inputs) = draw.new_tree(&mut runner).expect("generator").current();
        let mut saw = [false; ROWS];
        for (index, left) in manifests.iter().enumerate() {
            for right in manifests.iter().skip(index + 1) {
                saw[0] |= left.runtime_key.eq_ignore_ascii_case(&right.runtime_key);
                saw[1] |= left.runtime_id == right.runtime_id;
            }
        }
        saw[2] = manifests
            .iter()
            .any(|manifest| manifest.read_only || manifest.imported_from.is_some());
        let pointers = inputs
            .active_runtime_key
            .iter()
            .chain(inputs.previous_runtime_key.iter())
            .chain(inputs.marker_runtime_keys.iter());
        for pointer in pointers {
            let matching = installed_matching(&manifests, Some(pointer));
            saw[3] |= matching.iter().any(|key| key != pointer);
            saw[4] |= matching.is_empty();
        }
        if let Some(id) = inputs.default_runtime_id.as_deref() {
            let named = manifests
                .iter()
                .filter(|manifest| manifest.runtime_id.eq_ignore_ascii_case(id))
                .count();
            saw[5] = named == 1;
            saw[6] = named > 1;
        }
        let groups = recency_groups(&manifests, &inputs);
        for members in groups.values() {
            for (index, left) in members.iter().enumerate() {
                for right in members.iter().skip(index + 1) {
                    saw[7] |= left.installed_at_unix_ms == right.installed_at_unix_ms;
                    saw[8] |= left.runtime_key.eq_ignore_ascii_case(&right.runtime_key);
                    saw[9] |= left.channel != right.channel;
                    saw[10] |= left.format != right.format;
                    saw[11] |= left.family != right.family;
                }
            }
            if members.len() > 1 {
                let first_specs = |manifest: &&InstalledRuntimeManifest| {
                    manifest
                        .wheel_composition
                        .as_ref()
                        .and_then(|composition| composition.package_specs.first().cloned())
                        .unwrap_or_default()
                };
                saw[12] |= members
                    .iter()
                    .any(|manifest| manifest.devel != installed_the_toolchain(manifest));
                saw[13] |= members
                    .iter()
                    .any(|manifest| first_specs(manifest).contains(SPACED_DEVEL_EXTRAS));
                saw[14] |= members
                    .iter()
                    .any(|manifest| first_specs(manifest).contains(NEAR_MISS_EXTRAS));
                saw[15] |= members
                    .iter()
                    .any(|manifest| first_specs(manifest).starts_with("torch["));
            }
            saw[17] |= members.len() > 1;
            saw[18] |= members.len() > 2;
            saw[19] |= members.len() > 3;
        }
        // Two groups that differ only in the toolchain choice: what keeps a
        // plain install from evicting a `--devel` one.
        saw[16] = groups.keys().any(|(channel, format, family, devel)| {
            *devel && groups.contains_key(&(channel.clone(), format.clone(), family.clone(), false))
        });
        for (count, seen) in hits.iter_mut().zip(saw) {
            *count += u32::from(seen);
        }
    }

    println!("\n=== storage retention generator reach ({SAMPLES} samples) ===");
    let mut starved = Vec::new();
    for ((label, count), floor) in labels.iter().zip(hits).zip(FLOOR) {
        println!("  {label:<52} {count:>6} (floor {floor})");
        if count < floor {
            starved.push(*label);
        }
    }
    assert!(
        starved.is_empty(),
        "the generators stopped reaching: {starved:?}"
    );
}

const ROWS: usize = 20;

/// Floors for `generator_reach_report`, in the same order as its rows.
const FLOOR: [u32; ROWS] = [
    5_380, 9_450, 8_270, 5_310, 10_760, 1_800, 3_650, 1_740, 430, 2_030, 2_040, 2_310, 3_070, 970,
    650, 1_570, 4_270, 4_990, 960, 130,
];
