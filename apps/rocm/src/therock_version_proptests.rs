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
    InstalledRuntimeManifest, RuntimeFreshness, VersionOrderKey, VersionStage,
    compare_version_strings, parse_host_version, parse_version, parse_version_for_ordering,
    runtime_freshness, select_rocm_version, select_startup_update_manifest,
    sort_manifests_newest_install_first,
};
use crate::storage::{RetentionInputs, select_runtimes_to_remove};

// ---------------------------------------------------------------------------
// Alphabets
// ---------------------------------------------------------------------------

/// Version shapes that actually show up in the streams this code reads: a
/// PyPI-style simple index (`rocm`, `torch`, ...) and a TheRock tarball
/// catalogue whose "version" is just the middle of a file name.
///
/// Every entry is a shape one of those sources really publishes, and most were
/// chosen because `parse_version` — the strict grammar gate, which the
/// comparator used to order with — rejects them:
/// * `7.9.0` / `7.10.0` — the numeric-vs-lexicographic near miss.
/// * `7.9` — a two-component name (tarball catalogue, host-reported ROCm).
/// * `7.0.0rc1` / `7.0.0a1` / `7.0.0b1` — PEP 440 pre-releases. `rc` and `a`
///   are in `parse_version`'s grammar; `b` (beta) is not, and betas are
///   completely ordinary on PyPI.
/// * `7.0.0.post1` / `7.0.0.dev1` — PEP 440 post/dev releases, also outside it.
/// * `7.9.0+local` — local version metadata.
/// * `07.9.0` — leading zero.
/// * `7.0.0rc9` / `7.0.0rc10` — a stage number where the numeric and the
///   lexicographic answer differ, so a `stage_number` compared as text is
///   caught. (`7.0.0rc20250929` / `7.0.0rc20251001` have equal digit counts and
///   cannot distinguish the two.)
/// * `7.0.0rc20250929` — TheRock's real nightly date-stamped rc.
/// * `7.9rc1` — a stage riding on a two-component version, which is the one
///   `parse_version_for_ordering` branch nothing else here reaches.
/// * `7.2.4.70204` — a four-component release, the shape ROCm's own packages
///   are named with.
/// * `7.0.0-rc1` / `v7.0.0` — PEP 440 spellings of versions already in this
///   list. Every pair of spellings is one version but two strings, which is
///   the distinction the comparator and the "is there a newer build?" callers
///   have to draw differently: the comparator must keep them apart so `sort` is
///   deterministic, while an update verdict must call them the same build.
/// * `custom-build` / `latest` — not versions at all. A manifest adopted from
///   an existing environment carries whatever its `rocm_sdk` probe reported, so
///   the comparator's unreadable arms are reachable in production and have to
///   be reachable here too. *Two* of them, so a triple can mix two distinct
///   unreadable strings with a readable one — the `(None, None)` arm is
///   otherwise only ever reached reflexively.
const VERSIONS: &[&str] = &[
    "7.0.0",
    "7.0.0a1",
    "7.0.0b1",
    "7.0.0rc1",
    "7.0.0rc2",
    "7.0.0rc9",
    "7.0.0rc10",
    "7.0.0rc20250929",
    "7.0.0rc20251001",
    "7.0.0.post1",
    "7.0.0.dev1",
    "7.9",
    "7.9rc1",
    "7.9.0",
    "7.9.0+local",
    "07.9.0",
    "7.10.0",
    "7.2.4.70204",
    "7.0.0-rc1",
    "v7.0.0",
    "custom-build",
    "latest",
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
///
/// Note what this oracle is and is not. It caught the original defect because
/// the two parsers really did disagree. Now that `parse_host_version` defers to
/// the same parser the comparator orders with, it can no longer independently
/// confirm *what* the order should be — it confirms that the comparator does
/// not contradict the parsed key, which still catches a reversed comparison or
/// a tiebreak applied ahead of the numeric key. The semantics themselves are
/// pinned by the hand-written ladder in
/// `compare_version_strings_orders_pep440_stages`, whose expected values are
/// written out rather than derived from the implementation.
fn oracle(left: &str, right: &str) -> Option<Ordering> {
    let left: VersionOrderKey = parse_host_version(left)?;
    let right: VersionOrderKey = parse_host_version(right)?;
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
            // Two spellings of one version are one version. With no required
            // composition there is nothing to repair, so the only honest
            // verdict is "up to date" — offering an update here would have the
            // CLI re-download a multi-gigabyte runtime it already has.
            Ordering::Equal => prop_assert_eq!(
                verdict,
                RuntimeFreshness::UpToDate,
                "installed {} and index {} are the same version, but the CLI \
                 reports {:?}",
                installed, latest, verdict
            ),
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

        // Both views go through the registry loader's own "newest install
        // first" sort, called rather than re-implemented so the property
        // cannot drift away from what the loader actually does.
        let mut baseline = manifests.clone();
        let mut reordered = permute(&manifests, &order);
        sort_manifests_newest_install_first(&mut baseline);
        sort_manifests_newest_install_first(&mut reordered);

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
// Regression tests
// ---------------------------------------------------------------------------
//
// The properties above are the detectors; these pin the exact inputs proptest
// shrank to when the comparator mixed a numeric and a string relation, so the
// specific defects stay covered regardless of which equivalent counterexample a
// future shrink lands on. Each names the behaviour it used to have.

/// `compare_version_strings` orders one release's stages the way PEP 440 does.
///
/// It used to have a three-element cycle here — `7.0.0 < 7.0.0b1`, `7.0.0b1 <
/// 7.0.0rc1`, yet `7.0.0 > 7.0.0rc1` — because it fell back to a plain string
/// compare whenever either side missed `parse_version`'s `X.Y.Z[rcN|aN]`
/// grammar. `b1` (beta), `.post1`, `.dev1` and a missing patch component all
/// miss it, while parseable pairs were compared numerically, and mixing two
/// relations in one comparator cannot be transitive.
#[test]
fn compare_version_strings_orders_pep440_stages() {
    // dev < alpha < beta < rc < final < post, all within 7.0.0.
    let ascending = [
        "7.0.0.dev1",
        "7.0.0a1",
        "7.0.0b1",
        "7.0.0rc1",
        "7.0.0rc2",
        "7.0.0",
        "7.0.0.post1",
    ];
    for (index, earlier) in ascending.iter().enumerate() {
        for later in &ascending[index + 1..] {
            assert_eq!(
                compare_version_strings(earlier, later),
                Ordering::Less,
                "{earlier} must precede {later}"
            );
            assert_eq!(
                compare_version_strings(later, earlier),
                Ordering::Greater,
                "{later} must follow {earlier}"
            );
        }
    }

    // The pairs that used to form the cycle, stated directly.
    assert_eq!(
        compare_version_strings("7.0.0", "7.0.0b1"),
        Ordering::Greater
    );
    assert_eq!(
        compare_version_strings("7.0.0b1", "7.0.0rc1"),
        Ordering::Less
    );
    assert_eq!(
        compare_version_strings("7.0.0", "7.0.0rc1"),
        Ordering::Greater
    );

    // TheRock's real date-stamped nightly rcs. Equal digit counts, so numeric
    // and lexicographic agree and the old comparator got this right too — kept
    // as a shape check on the real input, not as evidence of the fix.
    assert_eq!(
        compare_version_strings("7.0.0rc20250929", "7.0.0rc20251001"),
        Ordering::Less
    );
    // Unequal digit counts, which is where a stage number compared as text
    // diverges. This one the old comparator also got right (both sides match
    // its strict grammar and `stage_number` was always numeric); it is here
    // because `rc9`/`rc10` is the pair a reader expects to see pinned.
    assert_eq!(
        compare_version_strings("7.0.0rc9", "7.0.0rc10"),
        Ordering::Less
    );
}

/// Exhaustive total-order check over the whole alphabet.
///
/// The `proptest` properties above draw triples at random; over an alphabet
/// this small every triple can simply be enumerated, which is both cheaper and
/// complete — no sampling gap, no dependence on a seed. It is kept alongside
/// the properties rather than replacing them because the properties also run
/// over generated *manifest sets*, which are not enumerable.
///
/// A comparator that is antisymmetric, transitive and never reports `Equal` for
/// two distinct strings is a total order, which is exactly what `sort_by`
/// requires and what the old comparator failed to be.
#[test]
fn compare_version_strings_is_a_total_order_over_the_alphabet() {
    for left in VERSIONS {
        assert_eq!(
            compare_version_strings(left, left),
            Ordering::Equal,
            "{left:?} must equal itself"
        );
        for right in VERSIONS {
            assert_eq!(
                compare_version_strings(left, right),
                compare_version_strings(right, left).reverse(),
                "antisymmetry broken for {left:?} / {right:?}"
            );
            assert!(
                left == right || compare_version_strings(left, right) != Ordering::Equal,
                "distinct strings {left:?} and {right:?} compare Equal, so a sort \
                 could order them either way"
            );
        }
    }
    for a in VERSIONS {
        for b in VERSIONS {
            if compare_version_strings(a, b) != Ordering::Less {
                continue;
            }
            for c in VERSIONS {
                if compare_version_strings(b, c) != Ordering::Less {
                    continue;
                }
                assert_eq!(
                    compare_version_strings(a, c),
                    Ordering::Less,
                    "transitivity broken: {a:?} < {b:?} < {c:?}"
                );
            }
        }
    }
    // Every rotation of the alphabet sorts to the same sequence, which is the
    // property `select_rocm_version` depends on: the chosen version must not
    // depend on the order the index listed its candidates in.
    let mut baseline: Vec<&str> = VERSIONS.to_vec();
    baseline.sort_by(|left, right| compare_version_strings(left, right));
    for shift in 0..VERSIONS.len() {
        let mut rotated: Vec<&str> = VERSIONS[shift..]
            .iter()
            .chain(&VERSIONS[..shift])
            .copied()
            .collect();
        rotated.sort_by(|left, right| compare_version_strings(left, right));
        assert_eq!(rotated, baseline, "rotation by {shift} sorted differently");
    }
}

/// An update verdict asks "is there a newer build?", which is a question about
/// versions, not about strings.
///
/// The comparator's last tiebreak is the raw string, so two spellings of one
/// version sort apart — deliberately, because `sort` needs every distinct
/// string to have a distinct place. Inheriting that tiebreak for the update
/// decision turns a spelling difference into an update offer, and applying it
/// re-downloads a multi-gigabyte runtime the machine already has.
///
/// This is reachable precisely because the parser was widened to accept these
/// spellings: PyPI normalises versions, so an index can serve `7.0.0rc1` while
/// an older manifest recorded the `7.0.0-rc1` it was installed from.
#[test]
fn runtime_freshness_treats_equal_spellings_as_one_version() {
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

    for (installed, latest) in [
        ("7.0.0-rc1", "7.0.0rc1"),
        ("7.0.0_rc1", "7.0.0rc1"),
        ("7.0.0RC1", "7.0.0rc1"),
        ("v7.0.0", "7.0.0"),
        ("7.0.0rc", "7.0.0rc0"),
        ("7.9", "7.9.0"),
        ("7.2.4.0", "7.2.4"),
        ("7.0.0alpha1", "7.0.0a1"),
    ] {
        let runtime = at(installed);
        assert_eq!(
            runtime_freshness(&runtime, latest, None, &runtime.runtime_key),
            RuntimeFreshness::UpToDate,
            "installed {installed} and index {latest} are one version"
        );
        // ...and the same both ways round, so no spelling is privileged.
        let reversed = at(latest);
        assert_eq!(
            runtime_freshness(&reversed, installed, None, &reversed.runtime_key),
            RuntimeFreshness::UpToDate,
            "installed {latest} and index {installed} are one version"
        );

        // The comparator still separates them: that is what `sort` needs, and
        // keeping both behaviours pinned together is the point of this test.
        assert_ne!(
            compare_version_strings(installed, latest),
            Ordering::Equal,
            "the comparator must still order {installed} and {latest} \
             deterministically"
        );
    }
}

/// The other two callers that ask "is this a different build?" answer it the
/// same way as the update verdict.
///
/// Both print a line to the user: the install approval names the displaced
/// default as an "upgrade"/"downgrade"/"reinstall", and the no-wheels warning
/// says a newer version exists but "installing X instead". A spelling
/// difference must read as a reinstall and must not produce a warning naming
/// one version on both sides; a version that cannot be identified must not
/// produce a warning either.
#[test]
fn install_messages_treat_equal_spellings_as_one_version() {
    let active = manifest(
        "nightly-wheel-gfx120X-all-7-0-0-rc1".to_owned(),
        "nightly".to_owned(),
        "wheel".to_owned(),
        "gfx120X-all".to_owned(),
        "7.0.0-rc1".to_owned(),
        false,
        1_000,
    );
    let text = super::active_default_relation_text(
        &active,
        super::TheRockChannel::Nightly,
        "gfx120X-all",
        "7.0.0rc1",
    );
    assert!(
        text.starts_with("reinstall from installed"),
        "a respelling of the installed version is not an upgrade: {text}"
    );
    // A genuinely newer version still reads as one.
    let text = super::active_default_relation_text(
        &active,
        super::TheRockChannel::Nightly,
        "gfx120X-all",
        "7.0.0",
    );
    assert!(text.starts_with("upgrade from installed"), "{text}");

    assert_eq!(
        super::repo_version_without_wheels(Some("7.0.0rc1"), "7.0.0-rc1"),
        None
    );
    assert_eq!(
        super::repo_version_without_wheels(Some("7.14.0"), "custom-build"),
        None
    );
    assert_eq!(
        super::repo_version_without_wheels(Some("7.14.0"), "7.13.0").as_deref(),
        Some("7.14.0")
    );
}

/// A four-component release is a real shape — ROCm's own packages are named
/// `7.2.4.70204` — and it must order by its numbers, not sink below everything.
///
/// Rejecting it would be worse than the bug this suite exists for: an installed
/// `7.2.4.70204` would sort below an index `6.4.3` and `rocm update --apply`
/// would offer to install ROCm 6 over ROCm 7.
#[test]
fn compare_version_strings_orders_four_component_releases() {
    assert_eq!(
        compare_version_strings("7.2.4.70204", "6.4.3"),
        Ordering::Greater
    );
    // Above its own three-component release, below the next patch.
    assert_eq!(
        compare_version_strings("7.2.4.70204", "7.2.4"),
        Ordering::Greater
    );
    assert_eq!(
        compare_version_strings("7.2.4.70204", "7.2.5"),
        Ordering::Less
    );
    // And as versions, not only as strings: the comparator's string tiebreak
    // would still order these two if the fourth component were dropped, but
    // the update verdict would then call `7.2.4` up to date against an index
    // offering `7.2.4.70204`.
    assert_eq!(
        super::version_relation("7.2.4.70204", "7.2.4"),
        Some(Ordering::Greater)
    );
    // Trailing zeros do not make a new release: PEP 440 pads the shorter side.
    assert_eq!(oracle("7.2.4.0", "7.2.4"), Some(Ordering::Equal));
    assert_eq!(oracle("7.9.0.0", "7.9"), Some(Ordering::Equal));
}

/// PEP 440 spells one version several ways, and the module says its order is
/// PEP 440's, so the spellings must land on one key.
///
/// `7.0.0-rc1` is the case that matters most: it is a legal spelling of
/// `7.0.0rc1`, and treating it as unreadable would sort a release candidate
/// below `0.0.0`.
#[test]
fn compare_version_strings_accepts_pep440_spellings() {
    for spelling in [
        "7.0.0rc1",
        "7.0.0.rc1",
        "7.0.0-rc1",
        "7.0.0_rc1",
        "7.0.0RC1",
        "7.0.0c1",
        "7.0.0pre1",
        "v7.0.0rc1",
    ] {
        assert_eq!(
            oracle(spelling, "7.0.0rc1"),
            Some(Ordering::Equal),
            "{spelling} must be the same version as 7.0.0rc1"
        );
        assert_eq!(
            compare_version_strings(spelling, "7.0.0"),
            Ordering::Less,
            "{spelling} must precede the final release"
        );
    }
    assert_eq!(oracle("7.0.0alpha1", "7.0.0a1"), Some(Ordering::Equal));
    assert_eq!(oracle("7.0.0beta1", "7.0.0b1"), Some(Ordering::Equal));
    assert_eq!(oracle("7.0.0rev1", "7.0.0.post1"), Some(Ordering::Equal));
    // An omitted stage numeral is zero, per PEP 440.
    assert_eq!(oracle("7.0.0rc", "7.0.0rc0"), Some(Ordering::Equal));

    // A host build number is still not a stage: `-98` is packaging metadata,
    // and `parse_host_version` drops it without disturbing `-rc1`.
    assert_eq!(oracle("7.2.4-98", "7.2.4"), Some(Ordering::Equal));
    assert_eq!(oracle("7.2.4-98", "7.13.0"), Some(Ordering::Less));
}

/// A two-component version is a release, not a string: `7.9` is `7.9.0`, which
/// precedes `7.10.0`.
///
/// This used to compare as text and report `Greater`, disagreeing with
/// `parse_host_version` in the same module, which already read `7.9` as
/// `7.9.0`. Both now share one parser, so the module cannot contradict itself.
#[test]
fn compare_version_strings_orders_7_9_below_7_10_0() {
    assert_eq!(compare_version_strings("7.9", "7.10.0"), Ordering::Less);
    assert_eq!(compare_version_strings("7.10.0", "7.9"), Ordering::Greater);
    assert_eq!(oracle("7.9", "7.10.0"), Some(Ordering::Less));

    // A stage may ride on the minor component when no patch was given, so
    // `7.9rc1` is `7.9.0rc1` and precedes `7.9`. Dropping the stage on a
    // two-component version would silently make these two equal.
    assert_eq!(compare_version_strings("7.9rc1", "7.9"), Ordering::Less);
    assert_eq!(compare_version_strings("7.9rc1", "7.9.0"), Ordering::Less);
    assert_eq!(
        compare_version_strings("7.9rc1", "7.8.0"),
        Ordering::Greater
    );

    // Numerically equal but textually different stays a *total* order: the two
    // spellings are separated by the string, never reported as equal.
    assert_eq!(compare_version_strings("7.9", "7.9.0"), Ordering::Less);
    assert_eq!(
        compare_version_strings("7.9.0+local", "7.9.0"),
        Ordering::Greater
    );
    assert_eq!(oracle("7.9", "7.9.0"), Some(Ordering::Equal));
}

/// A string the ordering parser cannot read sorts below every version it can,
/// and unreadable strings are ordered against each other by the string.
///
/// This is the position that replaced the old silent switch to a string
/// compare. Sorting low is the safe end *here*: `select_rocm_version` takes the
/// maximum, so junk can only be selected when every candidate is junk. It is
/// the wrong end for `runtime_freshness`, which is why that caller asks
/// [`super::version_relation`] instead — see
/// `runtime_freshness_follows_the_version_order`.
#[test]
fn compare_version_strings_places_unreadable_versions_below_readable_ones() {
    // What is genuinely unreadable: an epoch, a combined stage suffix, prose,
    // and the empty string. Note what is NOT in this list — `7.1.2.3` is a
    // four-component release and `7.0.0rc` is `rc0`; both are real versions and
    // are ordered as such.
    //
    // Each unreadable shape is chosen so that a parser which *approximated* it
    // instead of declining would land ABOVE `7.0.0.dev1` and fail here: the
    // epoch's release is `8.0` (dropping the epoch reads as 8.0), and the
    // combined suffix folded onto its first stage reads as `rc1`.
    for unreadable in ["1!8.0", "7.0.0rc1.dev1", "latest", ""] {
        assert_eq!(
            compare_version_strings(unreadable, "7.0.0.dev1"),
            Ordering::Less,
            "{unreadable:?} must sort below the lowest readable version"
        );
        assert_eq!(
            compare_version_strings("7.0.0.dev1", unreadable),
            Ordering::Greater
        );
    }
    assert_eq!(compare_version_strings("alpha", "beta"), Ordering::Less);

    // A readable candidate wins over an unreadable one however they are listed.
    for listed in [["latest", "7.0.0"], ["7.0.0", "latest"]] {
        let versions = listed.map(str::to_owned);
        assert_eq!(
            select_rocm_version(super::TheRockChannel::Nightly, &versions, None).as_deref(),
            Some("7.0.0")
        );
    }
}

/// Which version an install picks is a function of the candidate set, not of
/// the order the index listed them in.
///
/// `parse_simple_index_versions` returns versions in index-document order, so
/// this used to let the publisher's HTML ordering decide which build a user
/// got: these same three versions gave `7.0.0a1` in one order and
/// `7.0.0.post1` in the other.
#[test]
fn select_rocm_version_ignores_index_order() {
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
    assert_eq!(first, second, "index order must not change the answer");
    // `7.0.0.post1` is the newest of the three, and an alpha never wins.
    assert_eq!(first.as_deref(), Some("7.0.0.post1"));
}

/// `select_rocm_version` returns the newest candidate.
///
/// It used to return `7.9` here, because `7.9` beat `7.10.0` as text.
#[test]
fn select_rocm_version_picks_the_newer_of_two() {
    let versions = ["7.10.0".to_owned(), "7.9".to_owned()];
    assert_eq!(
        select_rocm_version(super::TheRockChannel::Nightly, &versions, None).as_deref(),
        Some("7.10.0")
    );
}

/// The registry sort puts the newest install first and breaks a tie on the
/// runtime key, in that order.
///
/// The order-independence property cannot pin this down: it sorts both views
/// with this same function, so any deterministic rule satisfies it. Reversing
/// the tiebreak, or sorting oldest-install-first, would both pass it. The
/// direction is user-observable, because `select_startup_update_manifest`
/// reports on the first entry when no active runtime key is configured.
#[test]
fn registry_sort_is_newest_install_first_then_by_key() {
    let at = |key: &str, installed_at_unix_ms: u128| {
        manifest(
            key.to_owned(),
            "release".to_owned(),
            "wheel".to_owned(),
            "gfx120X-all".to_owned(),
            "7.0.0".to_owned(),
            false,
            installed_at_unix_ms,
        )
    };
    let keys = |manifests: &[InstalledRuntimeManifest]| {
        manifests
            .iter()
            .map(|item| item.runtime_key.clone())
            .collect::<Vec<_>>()
    };

    // Newest install first, regardless of how they were read.
    let mut by_time = vec![at("older", 1_000), at("newer", 2_000)];
    sort_manifests_newest_install_first(&mut by_time);
    assert_eq!(keys(&by_time), ["newer", "older"]);
    let mut by_time_reversed = vec![at("newer", 2_000), at("older", 1_000)];
    sort_manifests_newest_install_first(&mut by_time_reversed);
    assert_eq!(keys(&by_time_reversed), ["newer", "older"]);

    // Same millisecond: the lower key wins, and the input order does not.
    let mut tied = vec![at("bbb", 2_000), at("aaa", 2_000)];
    sort_manifests_newest_install_first(&mut tied);
    assert_eq!(keys(&tied), ["aaa", "bbb"]);
    let mut tied_reversed = vec![at("aaa", 2_000), at("bbb", 2_000)];
    sort_manifests_newest_install_first(&mut tied_reversed);
    assert_eq!(keys(&tied_reversed), ["aaa", "bbb"]);

    // The timestamp outranks the key: a later install with a higher key still
    // comes first.
    let mut mixed = vec![at("aaa", 1_000), at("zzz", 2_000)];
    sort_manifests_newest_install_first(&mut mixed);
    assert_eq!(keys(&mixed), ["zzz", "aaa"]);

    // And that is what the startup update check reports on.
    assert_eq!(
        select_startup_update_manifest(&tied, None).map(|item| item.runtime_key.as_str()),
        Some("aaa")
    );
}

/// The update verdict points the same way as the version order.
///
/// Both directions used to be inverted for this pair. `AheadOfIndex` makes
/// `update_available()` false, so `rocm update --apply` reported "no newer
/// runtime found" for a runtime that really was a release behind;
/// `UpdateAvailable` made it proceed and install `latest_version`, which for an
/// installed `7.10.0` against an index offering `7.9` was a silent downgrade.
#[test]
fn runtime_freshness_follows_the_version_order() {
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

    // Installed 7.10.0, index offers 7.9 -> nothing to install.
    let newer_installed = at("7.10.0");
    assert_eq!(
        runtime_freshness(&newer_installed, "7.9", None, &newer_installed.runtime_key),
        RuntimeFreshness::AheadOfIndex
    );

    // Installed 7.9, index offers 7.10.0 -> a real upgrade is offered.
    let older_installed = at("7.9");
    assert_eq!(
        runtime_freshness(
            &older_installed,
            "7.10.0",
            None,
            &older_installed.runtime_key
        ),
        RuntimeFreshness::UpdateAvailable
    );

    // A four-component installed version against an older index must not be
    // offered an "update". Sorting an unreadable string low is safe for the
    // "newest candidate" pickers and would be a downgrade offer here, which is
    // why this caller requires both sides to be identifiable rather than
    // reusing the sort order.
    let packaged = at("7.2.4.70204");
    assert_eq!(
        runtime_freshness(&packaged, "6.4.3", None, &packaged.runtime_key),
        RuntimeFreshness::AheadOfIndex
    );

    // A runtime adopted from an existing environment carries whatever its probe
    // reported. That cannot be ordered against an index version at all, so no
    // update is offered rather than one that might be a downgrade.
    let unidentifiable = at("custom-build");
    assert_eq!(
        runtime_freshness(&unidentifiable, "7.10.0", None, &unidentifiable.runtime_key),
        RuntimeFreshness::AheadOfIndex
    );
    // ...but an exact string match is still "the same build", so an
    // unparseable version that equals the index's stays up to date rather than
    // being declared ahead of it.
    assert_eq!(
        runtime_freshness(
            &unidentifiable,
            "custom-build",
            None,
            &unidentifiable.runtime_key
        ),
        RuntimeFreshness::UpToDate
    );

    // A pre-release installed against its own final release is an upgrade, and
    // the final release against the pre-release is not.
    let prerelease = at("7.0.0rc1");
    assert_eq!(
        runtime_freshness(&prerelease, "7.0.0", None, &prerelease.runtime_key),
        RuntimeFreshness::UpdateAvailable
    );
    let final_release = at("7.0.0");
    assert_eq!(
        runtime_freshness(&final_release, "7.0.0rc1", None, &final_release.runtime_key),
        RuntimeFreshness::AheadOfIndex
    );
}

// ---------------------------------------------------------------------------
// Generator reach
// ---------------------------------------------------------------------------

/// Measure, rather than assume, that the generators above visit the region
/// these properties are about.
///
/// Printed with `cargo test -- --nocapture`. This is a measurement, not an
/// assertion about the code under test; it fails only if the generator stops
/// reaching the cases the other properties in this file depend on.
///
/// The RNG is a fixed, deterministic one (not seeded from the environment),
/// so these floors cannot flake from run to run.
#[test]
fn generator_reach_report() {
    const SAMPLES: u32 = 20_000;
    let mut runner = TestRunner::deterministic();

    let mut pair_total = 0u32;
    let mut pair_both_parse_strict = 0u32;
    let mut pair_mixed_parse = 0u32;
    let mut pair_neither_parse = 0u32;
    let mut pair_oracle_comparable = 0u32;
    let mut pair_equal_strings = 0u32;
    let mut pair_oracle_equal_strings_differ = 0u32;
    let mut pair_unreadable_for_ordering = 0u32;
    let mut pair_four_component_release = 0u32;
    let mut pair_stage_dev = 0u32;
    let mut pair_stage_alpha = 0u32;
    let mut pair_stage_beta = 0u32;
    let mut pair_stage_rc = 0u32;
    let mut pair_stage_post = 0u32;
    let mut pair_update_downgrade_direction = 0u32;

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
            if ordering == Ordering::Greater {
                pair_update_downgrade_direction += 1;
            }
        }
        if left == right {
            pair_equal_strings += 1;
        }

        let left_key = parse_version_for_ordering(&left);
        let right_key = parse_version_for_ordering(&right);
        if left_key.is_none() || right_key.is_none() {
            pair_unreadable_for_ordering += 1;
        }
        let keys = [&left_key, &right_key];
        if keys.into_iter().flatten().any(|key| key.release.len() >= 4) {
            pair_four_component_release += 1;
        }
        let mut saw_dev = false;
        let mut saw_alpha = false;
        let mut saw_beta = false;
        let mut saw_rc = false;
        let mut saw_post = false;
        for key in keys.into_iter().flatten() {
            match key.stage {
                VersionStage::Dev => saw_dev = true,
                VersionStage::Alpha => saw_alpha = true,
                VersionStage::Beta => saw_beta = true,
                VersionStage::Rc => saw_rc = true,
                VersionStage::Post => saw_post = true,
                VersionStage::Stable => {}
            }
        }
        pair_stage_dev += u32::from(saw_dev);
        pair_stage_alpha += u32::from(saw_alpha);
        pair_stage_beta += u32::from(saw_beta);
        pair_stage_rc += u32::from(saw_rc);
        pair_stage_post += u32::from(saw_post);
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
    println!(
        "  unreadable by the ordering parser   {pair_unreadable_for_ordering:>6} ({:.1}%)",
        pct(pair_unreadable_for_ordering, pair_total)
    );
    println!(
        "  installed newer than catalog        {pair_update_downgrade_direction:>6} ({:.1}%)",
        pct(pair_update_downgrade_direction, pair_total)
    );
    println!(
        "  four-component (or longer) release  {pair_four_component_release:>6} ({:.1}%)",
        pct(pair_four_component_release, pair_total)
    );
    println!(
        "  dev-stage version                   {pair_stage_dev:>6} ({:.1}%)",
        pct(pair_stage_dev, pair_total)
    );
    println!(
        "  alpha-stage version                 {pair_stage_alpha:>6} ({:.1}%)",
        pct(pair_stage_alpha, pair_total)
    );
    println!(
        "  beta-stage version                  {pair_stage_beta:>6} ({:.1}%)",
        pct(pair_stage_beta, pair_total)
    );
    println!(
        "  rc-stage version                    {pair_stage_rc:>6} ({:.1}%)",
        pct(pair_stage_rc, pair_total)
    );
    println!(
        "  post-stage version                  {pair_stage_post:>6} ({:.1}%)",
        pct(pair_stage_post, pair_total)
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
    // The comparator has an arm for a version the *ordering* parser cannot
    // read, and it is only exercised if the generator can produce one. Guarding
    // on `pair_mixed_parse` alone does not cover this: that counter is about
    // `parse_version`'s strict grammar, and every entry in `VERSIONS` could be
    // readable for ordering while still mixing under the strict one.
    assert!(
        pair_unreadable_for_ordering > SAMPLES / 50,
        "generator rarely produces a version the ordering parser cannot read, so \
         the comparator's unreadable arms go unexercised"
    );
    // Two spellings of one version are where the sort order and the update
    // verdict must answer differently, and the verdict property's `Equal` arm
    // asserts nothing unless the generator produces such a pair.
    assert!(
        pair_oracle_equal_strings_differ > SAMPLES / 200,
        "generator rarely produces two spellings of one version, so the update \
         verdict's equal-version arm goes unexercised"
    );
    assert!(
        set_with_timestamp_tie > SAMPLES / 20,
        "generator rarely produces equal install timestamps"
    );
    assert!(
        set_with_downgrade > SAMPLES / 200,
        "generator rarely produces a downgrade"
    );
    assert!(
        pair_four_component_release > SAMPLES / 50,
        "generator rarely produces a four-component (or longer) release, so \
         that release-length handling goes unexercised"
    );
    assert!(
        pair_stage_dev > SAMPLES / 50,
        "generator rarely produces a dev-stage version, so VersionStage::Dev goes unexercised"
    );
    assert!(
        pair_stage_alpha > SAMPLES / 50,
        "generator rarely produces an alpha-stage version, so VersionStage::Alpha goes unexercised"
    );
    assert!(
        pair_stage_beta > SAMPLES / 50,
        "generator rarely produces a beta-stage version, so VersionStage::Beta goes unexercised"
    );
    assert!(
        pair_stage_rc > SAMPLES / 50,
        "generator rarely produces an rc-stage version, so VersionStage::Rc goes unexercised"
    );
    assert!(
        pair_stage_post > SAMPLES / 50,
        "generator rarely produces a post-stage version, so VersionStage::Post goes unexercised"
    );
    assert!(
        pair_update_downgrade_direction > SAMPLES / 50,
        "generator rarely produces an installed-newer-than-catalog pair, so that \
         update/downgrade direction goes unexercised"
    );
}
