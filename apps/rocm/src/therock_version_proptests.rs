// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property-based tests for version comparison, "newest" ordering, and
//! update-source selection.
//!
//! These exercise the chain that decides *which* ROCm runtime a user ends up
//! with: [`super::compare_version_strings`] and everything that sorts with it,
//! [`super::runtime_freshness`] (the update/no-update verdict), and the
//! manifest pickers that choose an update source.
//!
//! Generator design: the alphabets below are deliberately tiny and
//! hand-picked. A uniform random `String`/`u64` generator never visits the
//! interesting region — ties, duplicate timestamps, shared `runtime_id`s and
//! near-miss versions have to occur on nearly every draw for these properties
//! to be worth running. `generator_reach_report` measures that and prints the
//! rates.

use std::cmp::Ordering;

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config as ProptestConfig, TestRunner};

use super::{
    InstalledRuntimeManifest, ParsedVersion, RuntimeFreshness, compare_version_strings,
    parse_host_version, parse_version, runtime_freshness, select_rocm_version,
    select_startup_update_manifest,
};
use crate::storage::{RetentionInputs, select_runtimes_to_remove};

// ---------------------------------------------------------------------------
// Alphabets
// ---------------------------------------------------------------------------

/// Version shapes that actually show up in the streams this code reads: a
/// PyPI-style simple index (`rocm`, `torch`, ...) and a TheRock tarball
/// catalogue whose "version" is just the middle of a file name.
///
/// Every entry is a shape one of those sources really publishes:
/// * `7.9.0` / `7.10.0` — the numeric-vs-lexicographic near miss.
/// * `7.9` — a two-component name (tarball catalogue, host-reported ROCm).
/// * `7.0.0rc1` / `7.0.0a1` / `7.0.0b1` — PEP 440 pre-releases. `rc` and `a`
///   are understood by `parse_version`; `b` (beta) is not, and betas are
///   completely ordinary on PyPI.
/// * `7.0.0.post1` / `7.0.0.dev1` — PEP 440 post/dev releases, also unparsed.
/// * `7.9.0+local` — local version metadata.
/// * `07.9.0` — leading zero.
/// * `7.0.0rc20250929` — TheRock's real nightly date-stamped rc.
const VERSIONS: &[&str] = &[
    "7.0.0",
    "7.0.0a1",
    "7.0.0b1",
    "7.0.0rc1",
    "7.0.0rc2",
    "7.0.0rc20250929",
    "7.0.0rc20251001",
    "7.0.0.post1",
    "7.0.0.dev1",
    "7.9",
    "7.9.0",
    "7.9.0+local",
    "07.9.0",
    "7.10.0",
];

const CHANNELS: &[&str] = &["release", "nightly"];
const FORMATS: &[&str] = &["wheel", "tarball"];
const FAMILIES: &[&str] = &["gfx110X-all", "gfx120X-all"];
/// Three values, so equal `installed_at_unix_ms` is the common case rather
/// than a rarity. Real installs land milliseconds apart; what matters here is
/// that a tie is reachable at all, and the tie is where order-dependence hides.
const TIMESTAMPS: &[u128] = &[1_000, 2_000, 3_000];

fn version() -> impl Strategy<Value = String> {
    proptest::sample::select(VERSIONS).prop_map(str::to_owned)
}

fn version_list() -> impl Strategy<Value = Vec<String>> {
    proptest::collection::vec(version(), 1..6)
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// The intended numeric order of two versions, when the repository's own
/// lenient parser can read both.
///
/// [`super::parse_host_version`] lives in the same file as
/// [`super::parse_version`] and handles the shapes the strict one rejects
/// (missing patch, `-build` suffix). Using it as the oracle keeps this a
/// self-consistency check between two parsers that ship together rather than
/// an opinion imported from outside: whatever the project means by "newer", it
/// cannot mean two opposite things in one module.
///
/// Returns `None` when either side is unreadable, so unparseable junk never
/// asserts an intended order.
fn oracle(left: &str, right: &str) -> Option<Ordering> {
    let left: ParsedVersion = parse_host_version(left)?;
    let right: ParsedVersion = parse_host_version(right)?;
    Some(left.cmp(&right))
}

// ---------------------------------------------------------------------------
// Manifest generation
// ---------------------------------------------------------------------------

fn manifest(
    runtime_key: String,
    channel: String,
    format: String,
    family: String,
    version: String,
    devel: bool,
    installed_at_unix_ms: u128,
) -> InstalledRuntimeManifest {
    InstalledRuntimeManifest {
        runtime_id: format!("therock-{channel}:{family}"),
        runtime_key,
        channel,
        format,
        family,
        family_source: "test".to_owned(),
        version,
        install_root: std::path::PathBuf::from("runtime-root"),
        selected_artifact_url: "https://example.invalid/rocm".to_owned(),
        source_layout_generation: None,
        index_url: None,
        tarball_file_name: None,
        python_launcher: None,
        python_executable: None,
        pip_cache_dir: None,
        rocm_sdk: None,
        sdk_torch: None,
        wheel_composition: None,
        read_only: false,
        imported_from: None,
        devel,
        installed_at_unix_ms,
    }
}

/// A small set of manifests over the tiny alphabets above.
///
/// `runtime_key` is derived from the other axes (the way a real
/// `wheel_runtime_key` is), then de-duplicated: the registry stores one JSON
/// file per `runtime_key` (`runtime_manifest_path`), so two installed
/// manifests sharing a key is not a state the CLI can be in, and generating it
/// would only manufacture failures the code is not responsible for.
fn manifest_set() -> impl Strategy<Value = Vec<InstalledRuntimeManifest>> {
    proptest::collection::vec(
        (
            proptest::sample::select(CHANNELS),
            proptest::sample::select(FORMATS),
            proptest::sample::select(FAMILIES),
            version(),
            any::<bool>(),
            proptest::sample::select(TIMESTAMPS),
        ),
        1..5,
    )
    .prop_map(|rows| {
        let mut seen = std::collections::BTreeSet::new();
        rows.into_iter()
            .filter_map(|(channel, format, family, version, devel, installed_at)| {
                let devel_token = if devel { "devel" } else { "runtime" };
                let key = format!("therock-{channel}-{format}-{family}-{devel_token}-{version}");
                seen.insert(key.clone()).then(|| {
                    manifest(
                        key,
                        (*channel).to_owned(),
                        (*format).to_owned(),
                        (*family).to_owned(),
                        version,
                        devel,
                        installed_at,
                    )
                })
            })
            .collect()
    })
}

/// A permutation of `items`, as an index list, so a property can feed the same
/// multiset in a different order.
fn permutation(len: usize) -> impl Strategy<Value = Vec<usize>> {
    Just((0..len).collect::<Vec<_>>()).prop_shuffle()
}

fn permute<T: Clone>(items: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|index| items[*index].clone()).collect()
}

// ---------------------------------------------------------------------------
// A. Comparator algebra
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 4096, ..ProptestConfig::default() })]

    /// `compare_version_strings` must be antisymmetric.
    #[test]
    fn compare_version_strings_is_antisymmetric(left in version(), right in version()) {
        prop_assert_eq!(
            compare_version_strings(&left, &right),
            compare_version_strings(&right, &left).reverse(),
            "cmp({}, {}) is not the reverse of cmp({}, {})",
            left, right, right, left
        );
    }

    /// `compare_version_strings` must be transitive — `sort_by` requires a
    /// total order, and every "newest" decision in this module is a sort.
    #[test]
    fn compare_version_strings_is_transitive(
        a in version(),
        b in version(),
        c in version(),
    ) {
        if compare_version_strings(&a, &b) == Ordering::Less
            && compare_version_strings(&b, &c) == Ordering::Less
        {
            prop_assert_eq!(
                compare_version_strings(&a, &c),
                Ordering::Less,
                "{} < {} < {}, but cmp({}, {}) = {:?}",
                a, b, c, a, c,
                compare_version_strings(&a, &c)
            );
        }
    }

    /// Sorting the same multiset in two different input orders must produce
    /// the same sequence.
    #[test]
    fn sorting_versions_is_order_independent(
        versions in version_list(),
        order in permutation(5),
    ) {
        let order: Vec<usize> = order
            .into_iter()
            .filter(|index| *index < versions.len())
            .collect();
        prop_assume!(order.len() == versions.len());
        let mut baseline = versions.clone();
        baseline.sort_by(|left, right| compare_version_strings(left, right));
        let mut shuffled = permute(&versions, &order);
        shuffled.sort_by(|left, right| compare_version_strings(left, right));
        prop_assert_eq!(
            &baseline,
            &shuffled,
            "sorting the same versions in a different input order gave a different result"
        );
    }
}

// ---------------------------------------------------------------------------
// B. Comparator versus the module's own lenient parser
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 4096, ..ProptestConfig::default() })]

    /// When both versions are readable, `compare_version_strings` must not
    /// report the opposite of the numeric order.
    #[test]
    fn compare_version_strings_never_inverts_the_numeric_order(
        left in version(),
        right in version(),
    ) {
        let Some(expected) = oracle(&left, &right) else {
            return Ok(());
        };
        let actual = compare_version_strings(&left, &right);
        let inverted = matches!(
            (expected, actual),
            (Ordering::Less, Ordering::Greater) | (Ordering::Greater, Ordering::Less)
        );
        prop_assert!(
            !inverted,
            "cmp({}, {}) = {:?} but numerically it is {:?}",
            left, right, actual, expected
        );
    }
}

// ---------------------------------------------------------------------------
// C. Version selection
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 4096, ..ProptestConfig::default() })]

    /// Picking the newest ROCm version must be a function of the candidate
    /// set, not of the order the index happened to list it in.
    #[test]
    fn select_rocm_version_is_order_independent(
        versions in version_list(),
        order in permutation(5),
    ) {
        let order: Vec<usize> = order
            .into_iter()
            .filter(|index| *index < versions.len())
            .collect();
        prop_assume!(order.len() == versions.len());
        let reordered = permute(&versions, &order);

        for channel in [super::TheRockChannel::Release, super::TheRockChannel::Nightly] {
            prop_assert_eq!(
                select_rocm_version(channel, &versions, None),
                select_rocm_version(channel, &reordered, None),
                "{:?}: reordering the index changed the selected version ({:?} vs {:?})",
                channel, versions, reordered
            );
        }
    }

    /// The selected version must be the greatest candidate: no other
    /// candidate may be numerically newer than the one that was picked.
    #[test]
    fn select_rocm_version_picks_the_newest_candidate(versions in version_list()) {
        for channel in [super::TheRockChannel::Release, super::TheRockChannel::Nightly] {
            let Some(selected) = select_rocm_version(channel, &versions, None) else {
                continue;
            };
            for candidate in &versions {
                // Only candidates this channel would actually accept count.
                if matches!(channel, super::TheRockChannel::Release)
                    && !super::is_stable_runtime_version(candidate)
                {
                    continue;
                }
                if oracle(&selected, candidate) == Some(Ordering::Less) {
                    prop_assert!(
                        false,
                        "{:?}: selected {} from {:?}, but {} is newer",
                        channel, selected, versions, candidate
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D. The update verdict
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 4096, ..ProptestConfig::default() })]

    /// The update/no-update verdict must point the same way as the numeric
    /// order: a runtime older than the index is never "ahead of the index",
    /// and a runtime newer than the index never offers an "update" that is
    /// really a downgrade.
    #[test]
    fn runtime_freshness_never_points_the_wrong_way(
        installed in version(),
        latest in version(),
    ) {
        let Some(expected) = oracle(&installed, &latest) else {
            return Ok(());
        };
        let manifest = manifest(
            "therock-release-wheel-gfx120X-all-runtime".to_owned(),
            "release".to_owned(),
            "wheel".to_owned(),
            "gfx120X-all".to_owned(),
            installed.clone(),
            false,
            1_000,
        );
        let verdict = runtime_freshness(&manifest, &latest, None, &manifest.runtime_key);
        match expected {
            Ordering::Less => prop_assert_ne!(
                verdict,
                RuntimeFreshness::AheadOfIndex,
                "installed {} is older than index {}, but the CLI reports it as \
                 ahead of the index (no update offered)",
                installed, latest
            ),
            Ordering::Greater => prop_assert_ne!(
                verdict,
                RuntimeFreshness::UpdateAvailable,
                "installed {} is newer than index {}, but the CLI offers an \
                 update (a downgrade)",
                installed, latest
            ),
            Ordering::Equal => {}
        }
    }
}

// ---------------------------------------------------------------------------
// E. Manifest selection and retention
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// The startup update check must pick the same runtime regardless of the
    /// order the registry directory happened to be read in.
    #[test]
    fn select_startup_update_manifest_is_order_independent(
        manifests in manifest_set(),
        order in permutation(4),
    ) {
        let order: Vec<usize> = order
            .into_iter()
            .filter(|index| *index < manifests.len())
            .collect();
        prop_assume!(order.len() == manifests.len());

        // Both views go through the same "newest install first" sort the
        // registry loader applies.
        let mut baseline = manifests.clone();
        let mut reordered = permute(&manifests, &order);
        baseline.sort_by_key(|item| std::cmp::Reverse(item.installed_at_unix_ms));
        reordered.sort_by_key(|item| std::cmp::Reverse(item.installed_at_unix_ms));

        let left = select_startup_update_manifest(&baseline, None);
        let right = select_startup_update_manifest(&reordered, None);
        prop_assert_eq!(
            left.map(|item| item.runtime_key.as_str()),
            right.map(|item| item.runtime_key.as_str()),
            "registry read order changed which runtime the startup check looks at"
        );
    }

    /// Retention must be a function of the manifest set, not of its order.
    #[test]
    fn select_runtimes_to_remove_is_order_independent(
        manifests in manifest_set(),
        order in permutation(4),
        keep in 0usize..3,
    ) {
        let order: Vec<usize> = order
            .into_iter()
            .filter(|index| *index < manifests.len())
            .collect();
        prop_assume!(order.len() == manifests.len());
        let reordered = permute(&manifests, &order);
        let inputs = RetentionInputs::default();
        let (removable_a, held_a) = select_runtimes_to_remove(&manifests, &inputs, keep);
        let (removable_b, held_b) = select_runtimes_to_remove(&reordered, &inputs, keep);
        prop_assert_eq!(&removable_a, &removable_b, "removal set depends on input order");
        prop_assert_eq!(&held_a, &held_b, "hold set depends on input order");
    }

    /// A `--devel` install and its plain sibling are separate runtimes, so a
    /// newer runtime-only install must never evict the toolchain one.
    #[test]
    fn retention_never_lets_a_plain_install_evict_its_devel_sibling(
        manifests in manifest_set(),
        keep in 1usize..3,
    ) {
        let inputs = RetentionInputs::default();
        let (removable, _held) = select_runtimes_to_remove(&manifests, &inputs, keep);
        for manifest in &manifests {
            if !removable.contains(&manifest.runtime_key) {
                continue;
            }
            // Something in the same channel/format/family with the *same*
            // toolchain choice must have survived it; being evicted purely by
            // a sibling with a different toolchain choice is the bug.
            let same_toolchain_survivors = manifests
                .iter()
                .filter(|other| {
                    other.runtime_key != manifest.runtime_key
                        && other.channel == manifest.channel
                        && other.format == manifest.format
                        && other.family == manifest.family
                        && other.includes_devel() == manifest.includes_devel()
                        && !removable.contains(&other.runtime_key)
                })
                .count();
            prop_assert!(
                same_toolchain_survivors >= keep.min(1),
                "{} was removed with no same-toolchain sibling kept",
                manifest.runtime_key
            );
        }
    }

    /// Retention keeps the most *recently installed*, not the highest
    /// version: after a deliberate downgrade the older version is the newer
    /// install and must be the one kept.
    #[test]
    fn retention_keeps_the_most_recent_install_not_the_highest_version(
        manifests in manifest_set(),
    ) {
        let inputs = RetentionInputs::default();
        let (removable, _held) = select_runtimes_to_remove(&manifests, &inputs, 1);
        for removed in &removable {
            let removed_manifest = manifests
                .iter()
                .find(|item| &item.runtime_key == removed)
                .expect("removed key must exist");
            let newer_sibling_exists = manifests.iter().any(|other| {
                other.runtime_key != removed_manifest.runtime_key
                    && other.channel == removed_manifest.channel
                    && other.format == removed_manifest.format
                    && other.family == removed_manifest.family
                    && other.includes_devel() == removed_manifest.includes_devel()
                    && other.installed_at_unix_ms >= removed_manifest.installed_at_unix_ms
            });
            prop_assert!(
                newer_sibling_exists,
                "{removed} was removed although nothing in its group was installed later"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// Which runtime an update reads its channel/format/family/version from
    /// must depend on the registry contents and the config, not on the order
    /// the registry directory was read in.
    #[test]
    fn select_runtime_update_source_is_order_independent(
        manifests in manifest_set(),
        order in permutation(4),
        active_index in 0usize..4,
        use_active_key in any::<bool>(),
    ) {
        let order: Vec<usize> = order
            .into_iter()
            .filter(|index| *index < manifests.len())
            .collect();
        prop_assume!(order.len() == manifests.len());
        let reordered = permute(&manifests, &order);

        let anchor = &manifests[active_index % manifests.len()];
        let mut config = rocm_core::RocmCliConfig::default();
        if use_active_key {
            config.active_runtime_key = Some(anchor.runtime_key.clone());
        }
        config.default_runtime_id = Some(anchor.runtime_id.clone());

        let left = crate::select_runtime_update_source(&manifests, &config, None);
        let right = crate::select_runtime_update_source(&reordered, &config, None);
        match (left, right) {
            (Ok(left), Ok(right)) => prop_assert_eq!(
                &left.runtime_key,
                &right.runtime_key,
                "registry read order changed the runtime an update is based on"
            ),
            (Err(_), Err(_)) => {}
            (left, right) => prop_assert!(
                false,
                "registry read order changed whether an update source resolves at all: \
                 {:?} vs {:?}",
                left.map(|item| item.runtime_key.clone()),
                right.map(|item| item.runtime_key.clone())
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal reproducers
// ---------------------------------------------------------------------------
//
// The properties above are the detectors; these pin the exact inputs proptest
// shrank to, so the defects stay reproducible regardless of which equivalent
// counterexample a future shrink lands on. Each asserts the *current* (wrong)
// behaviour and says what it must become once the comparator is fixed.

/// `compare_version_strings` has a three-element cycle over version strings a
/// PEP 440 index publishes verbatim.
///
/// `7.0.0 < 7.0.0b1` and `7.0.0b1 < 7.0.0rc1`, but `7.0.0 > 7.0.0rc1`. The
/// cycle exists because the comparator falls back to a plain string compare
/// whenever either side misses `parse_version`'s `X.Y.Z[rcN|aN]` grammar —
/// `b1` (beta), `.post1`, `.dev1` and a missing patch component all miss it —
/// while parseable pairs are compared numerically. Mixing the two relations in
/// one comparator cannot be transitive, so `sort_by` has no defined result.
#[test]
fn repro_compare_version_strings_has_a_cycle() {
    assert_eq!(compare_version_strings("7.0.0", "7.0.0b1"), Ordering::Less);
    assert_eq!(
        compare_version_strings("7.0.0b1", "7.0.0rc1"),
        Ordering::Less
    );
    // Transitivity demands `Less`; the comparator says the opposite.
    assert_eq!(
        compare_version_strings("7.0.0", "7.0.0rc1"),
        Ordering::Greater
    );

    // The same cycle with `.post1`, which is what a rebuilt wheel carries.
    assert_eq!(
        compare_version_strings("7.0.0", "7.0.0.post1"),
        Ordering::Less
    );
    assert_eq!(
        compare_version_strings("7.0.0.post1", "7.0.0rc1"),
        Ordering::Less
    );
}

/// A two-component version is compared as text, so `7.9` reads as newer than
/// `7.10.0`.
///
/// `parse_host_version`, in the same module, reads `7.9` as `7.9.0` and gets
/// the opposite answer — the module disagrees with itself about which build is
/// newer.
#[test]
fn repro_compare_version_strings_inverts_7_9_against_7_10_0() {
    assert_eq!(
        compare_version_strings("7.9", "7.10.0"),
        Ordering::Greater,
        "should be Less: 7.9 is 7.9.0, which precedes 7.10.0"
    );
    assert_eq!(oracle("7.9", "7.10.0"), Some(Ordering::Less));
}

/// Which version an install picks depends on the order the index listed them.
///
/// `parse_simple_index_versions` returns versions in index-document order, so
/// this is the publisher's HTML ordering deciding which build a user gets.
#[test]
fn repro_select_rocm_version_depends_on_index_order() {
    let listed_one = [
        "7.0.0.post1".to_owned(),
        "7.0.0".to_owned(),
        "7.0.0a1".to_owned(),
    ];
    let listed_two = [
        "7.0.0a1".to_owned(),
        "7.0.0".to_owned(),
        "7.0.0.post1".to_owned(),
    ];
    let first = select_rocm_version(super::TheRockChannel::Nightly, &listed_one, None);
    let second = select_rocm_version(super::TheRockChannel::Nightly, &listed_two, None);
    assert_ne!(
        first, second,
        "the same three versions in two orders give two different answers"
    );
    // Neither is right: `7.0.0.post1` is the newest of the three.
    assert_eq!(first.as_deref(), Some("7.0.0a1"));
    assert_eq!(second.as_deref(), Some("7.0.0.post1"));
}

/// `select_rocm_version` does not return the newest candidate.
#[test]
fn repro_select_rocm_version_picks_the_older_of_two() {
    let versions = ["7.10.0".to_owned(), "7.9".to_owned()];
    assert_eq!(
        select_rocm_version(super::TheRockChannel::Nightly, &versions, None).as_deref(),
        Some("7.9"),
        "should be 7.10.0"
    );
}

/// The update verdict points the wrong way in both directions.
///
/// `AheadOfIndex` makes `update_available()` false, so `rocm update --apply`
/// reports "no newer runtime found" and installs nothing; `UpdateAvailable`
/// makes it proceed and install `latest_version`.
#[test]
fn repro_runtime_freshness_offers_a_downgrade_and_hides_an_upgrade() {
    let at = |version: &str| {
        manifest(
            "therock-release-wheel-gfx120X-all-runtime".to_owned(),
            "release".to_owned(),
            "wheel".to_owned(),
            "gfx120X-all".to_owned(),
            version.to_owned(),
            false,
            1_000,
        )
    };

    // Installed 7.10.0, index offers 7.9 -> the CLI calls 7.9 an update.
    let newer_installed = at("7.10.0");
    assert_eq!(
        runtime_freshness(&newer_installed, "7.9", None, &newer_installed.runtime_key),
        RuntimeFreshness::UpdateAvailable,
        "should be AheadOfIndex"
    );

    // Installed 7.9, index offers 7.10.0 -> the CLI refuses to update.
    let older_installed = at("7.9");
    assert_eq!(
        runtime_freshness(
            &older_installed,
            "7.10.0",
            None,
            &older_installed.runtime_key
        ),
        RuntimeFreshness::AheadOfIndex,
        "should be UpdateAvailable"
    );
}

// ---------------------------------------------------------------------------
// Generator reach
// ---------------------------------------------------------------------------

/// Measure, rather than assume, that the generators above visit the region
/// these properties are about.
///
/// Printed with `cargo test -- --nocapture`. This is a measurement, not an
/// assertion about the code under test; it asserts only that the generator
/// itself is not degenerate.
#[test]
fn generator_reach_report() {
    const SAMPLES: u32 = 20_000;
    let mut runner = TestRunner::new(ProptestConfig {
        cases: SAMPLES,
        ..ProptestConfig::default()
    });

    let mut pair_total = 0u32;
    let mut pair_both_parse_strict = 0u32;
    let mut pair_mixed_parse = 0u32;
    let mut pair_neither_parse = 0u32;
    let mut pair_oracle_comparable = 0u32;
    let mut pair_equal_strings = 0u32;
    let mut pair_oracle_equal_strings_differ = 0u32;

    let pair = (version(), version());
    for _ in 0..SAMPLES {
        let tree = pair.new_tree(&mut runner).expect("generator");
        let (left, right) = tree.current();
        pair_total += 1;
        match (parse_version(&left), parse_version(&right)) {
            (Some(_), Some(_)) => pair_both_parse_strict += 1,
            (None, None) => pair_neither_parse += 1,
            _ => pair_mixed_parse += 1,
        }
        if let Some(ordering) = oracle(&left, &right) {
            pair_oracle_comparable += 1;
            if ordering == Ordering::Equal && left != right {
                pair_oracle_equal_strings_differ += 1;
            }
        }
        if left == right {
            pair_equal_strings += 1;
        }
    }

    let mut set_total = 0u32;
    let mut set_with_timestamp_tie = 0u32;
    let mut set_with_duplicate_key = 0u32;
    let mut set_with_shared_runtime_id = 0u32;
    let mut set_with_devel_and_plain_sibling = 0u32;
    let mut set_with_downgrade = 0u32;
    let mut set_with_full_retention_group_collision = 0u32;

    let sets = manifest_set();
    for _ in 0..SAMPLES {
        let tree = sets.new_tree(&mut runner).expect("generator");
        let manifests = tree.current();
        set_total += 1;

        let mut saw_tie = false;
        let mut saw_duplicate_key = false;
        let mut saw_shared_id = false;
        let mut saw_devel_pair = false;
        let mut saw_downgrade = false;
        let mut saw_group_collision = false;
        for (index, left) in manifests.iter().enumerate() {
            for right in manifests.iter().skip(index + 1) {
                if left.installed_at_unix_ms == right.installed_at_unix_ms {
                    saw_tie = true;
                }
                if left.runtime_key == right.runtime_key {
                    saw_duplicate_key = true;
                }
                if left.runtime_id == right.runtime_id {
                    saw_shared_id = true;
                }
                let same_group = left.channel == right.channel
                    && left.format == right.format
                    && left.family == right.family;
                if same_group && left.includes_devel() != right.includes_devel() {
                    saw_devel_pair = true;
                }
                if same_group && left.includes_devel() == right.includes_devel() {
                    saw_group_collision = true;
                    // A downgrade: the *later* install carries the *older*
                    // version.
                    let (earlier, later) =
                        if left.installed_at_unix_ms <= right.installed_at_unix_ms {
                            (left, right)
                        } else {
                            (right, left)
                        };
                    if earlier.installed_at_unix_ms != later.installed_at_unix_ms
                        && oracle(&later.version, &earlier.version) == Some(Ordering::Less)
                    {
                        saw_downgrade = true;
                    }
                }
            }
        }
        set_with_timestamp_tie += u32::from(saw_tie);
        set_with_duplicate_key += u32::from(saw_duplicate_key);
        set_with_shared_runtime_id += u32::from(saw_shared_id);
        set_with_devel_and_plain_sibling += u32::from(saw_devel_pair);
        set_with_downgrade += u32::from(saw_downgrade);
        set_with_full_retention_group_collision += u32::from(saw_group_collision);
    }

    let pct = |count: u32, total: u32| f64::from(count) * 100.0 / f64::from(total);
    println!("\n=== generator reach ({SAMPLES} samples each) ===");
    println!("version pairs (n={pair_total}):");
    println!(
        "  both parse under parse_version      {pair_both_parse_strict:>6} ({:.1}%)",
        pct(pair_both_parse_strict, pair_total)
    );
    println!(
        "  exactly one parses (mixed)          {pair_mixed_parse:>6} ({:.1}%)",
        pct(pair_mixed_parse, pair_total)
    );
    println!(
        "  neither parses                      {pair_neither_parse:>6} ({:.1}%)",
        pct(pair_neither_parse, pair_total)
    );
    println!(
        "  oracle can order them               {pair_oracle_comparable:>6} ({:.1}%)",
        pct(pair_oracle_comparable, pair_total)
    );
    println!(
        "  identical strings                   {pair_equal_strings:>6} ({:.1}%)",
        pct(pair_equal_strings, pair_total)
    );
    println!(
        "  numerically equal, textually differ {pair_oracle_equal_strings_differ:>6} ({:.1}%)",
        pct(pair_oracle_equal_strings_differ, pair_total)
    );
    println!("manifest sets (n={set_total}):");
    println!(
        "  equal installed_at_unix_ms          {set_with_timestamp_tie:>6} ({:.1}%)",
        pct(set_with_timestamp_tie, set_total)
    );
    println!(
        "  duplicate runtime_key               {set_with_duplicate_key:>6} ({:.1}%)",
        pct(set_with_duplicate_key, set_total)
    );
    println!(
        "  shared runtime_id                   {set_with_shared_runtime_id:>6} ({:.1}%)",
        pct(set_with_shared_runtime_id, set_total)
    );
    println!(
        "  devel + plain sibling in one group  {set_with_devel_and_plain_sibling:>6} ({:.1}%)",
        pct(set_with_devel_and_plain_sibling, set_total)
    );
    println!(
        "  two installs in one retention group {set_with_full_retention_group_collision:>6} ({:.1}%)",
        pct(set_with_full_retention_group_collision, set_total)
    );
    println!(
        "  deliberate downgrade (newer install, older version) \
         {set_with_downgrade:>6} ({:.1}%)",
        pct(set_with_downgrade, set_total)
    );
    println!();

    // The generator itself must not be degenerate.
    assert!(
        pair_mixed_parse > SAMPLES / 20,
        "generator rarely mixes parseable and unparseable versions"
    );
    assert!(
        set_with_timestamp_tie > SAMPLES / 20,
        "generator rarely produces equal install timestamps"
    );
    assert!(
        set_with_downgrade > SAMPLES / 200,
        "generator rarely produces a downgrade"
    );
}
